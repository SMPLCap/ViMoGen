import argparse
import json
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mbench.motion_quality import compute_jitter_degree
from google import genai
from google.genai import types

from motion_gating.vlm_utils import interpret_vlm_output

PROMPT_TEMPLATE = """
You will be given a short natural language description that specifies a human motion.
You will also receive a rendered video showing a motion (single person, clean background), with no interacting objects.
Please follow the instructions strictly:
1. Analyze the human motion in the video in detail by describing the person's body posture, limb movements, and any repetitive or distinctive motion patterns you observe. 
2. Compare the observed motion with the provided description. When evaluating, prioritize the semantic consistency of the action (e.g., whether the core intent and type of movement match) over the visual quality, smoothness, or physical realism of the rendering.
3. Output a JSON object with two keys:
   - "analysis": your concise reasoning (one sentence).
   - "matches": either true or false depending on whether the motion matches the description.
Only output valid JSON. Do not include any additional commentary.
Description: "{description}"
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute MBench motion metrics + Gemini alignment to set use_ref_motion."
    )
    parser.add_argument(
        "--meta-json",
        type=Path,
        default=Path("data_samples/example_archive_wi_ref_eval.json"),
        help="MBench metadata with video_path/mbench_eval_path and without use_ref_motion.",
    )
    parser.add_argument(
        "--quality-report",
        type=Path,
        default=Path("data_samples/example_archive_wi_ref_quality.json"),
        help="JSON with per-id metrics and decisions.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for metric computation.",
    )
    parser.add_argument(
        "--jitter-threshold",
        type=float,
        default=0.04,
        help="Jitter threshold.",
    )
    parser.add_argument(
        "--gemini-api-key",
        type=str,
        required=True,
        help="Gemini API key.",
    )
    parser.add_argument(
        "--video-field",
        type=str,
        default="video_path",
        help="Field name in meta JSON that points to the rendered video path.",
    )
    parser.add_argument(
        "--eval-field",
        type=str,
        default="mbench_eval_path",
        help="Field name in meta JSON that points to the .pt cache with joints.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=2,
        help="Parallel Gemini requests.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit for debugging.",
    )
    parser.add_argument(
        "--sampling-fps",
        type=float,
        default=5.0,
        help="Sampling frames per second for Gemini video understanding.",
    )
    return parser.parse_args()


def resolve_repo_path(meta_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        repo_root = find_repo_root(meta_path)
        path = (repo_root / path).resolve()
    return path


def find_repo_root(meta_path: Path) -> Path:
    for parent in meta_path.parents:
        if (parent / "README.md").exists():
            return parent
    return meta_path.parent


def list_motion_samples(
    entries: List[Dict[str, Any]], meta_path: Path, eval_field: str
) -> List[Tuple[int, Path]]:
    samples: List[Tuple[int, Path]] = []
    for entry in entries:
        eval_path = entry.get(eval_field)
        if eval_path is None:
            raise KeyError(f"Missing '{eval_field}' in meta entry id={entry.get('id')}")
        path = resolve_repo_path(meta_path, str(eval_path))
        if not path.exists():
            raise FileNotFoundError(f"Missing eval cache: {path}")
        # Prefer unique global_id; fall back to id if absent
        motion_id_raw = entry.get("global_id", entry.get("id"))
        if motion_id_raw is None:
            raise KeyError("Entry missing both 'global_id' and 'id'")
        try:
            motion_id = int(motion_id_raw)
        except Exception:
            motion_id = motion_id_raw
        samples.append((motion_id, path))
    return samples


def compute_metrics_for_dimension(
    samples: List[Tuple[int, Path]],
    dimension: str,
    compute_fn,
    device: str,
) -> Dict[int, float]:
    if not samples:
        return {}
    payload = [
        {
            "dimension": dimension,
            "id": motion_id,
            "prompt": "",
            "evaluation_file": str(pt_path),
        }
        for motion_id, pt_path in samples
    ]
    with tempfile.NamedTemporaryFile("w", suffix=f"_{dimension}.json", delete=False) as tmp:
        json.dump(payload, tmp)
        temp_path = Path(tmp.name)
    try:
        result = compute_fn(str(temp_path), device)
    finally:
        temp_path.unlink(missing_ok=True)

    per_motion: Dict[int, float] = {}
    for entry in result.get("per_motion", []):
        try:
            per_motion[int(entry["id"])] = float(entry["value"])
        except (TypeError, ValueError, KeyError):
            continue
    return per_motion


def compute_motion_quality_metrics(
    samples: List[Tuple[int, Path]],
    device: str,
) -> Dict[int, Dict[str, float]]:
    metrics = {motion_id: {} for motion_id, _ in samples}
    per_dimension = compute_metrics_for_dimension(samples, "Jitter_Degree", compute_jitter_degree, device)
    for motion_id, value in per_dimension.items():
        if motion_id in metrics:
            metrics[motion_id]["Jitter_Degree"] = value
    return metrics


def call_gemini(client, video_path: Path, prompt: str, fps: int = 5) -> Dict[str, Any]:
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=types.Content(
            role="user",
            parts=[
                types.Part(
                    inline_data=types.Blob(data=video_bytes, mime_type="video/mp4"),
                    video_metadata=types.VideoMetadata(fps=fps)
                ),
                types.Part(text=prompt),
            ],
        ),
    )
    raw_text = response.text.strip() if response.text else ""
    analysis, matches = interpret_vlm_output(raw_text)
    return {"analysis": analysis, "matches": matches, "raw": raw_text}

def run_alignment_checks(
    entries: List[Dict[str, Any]],
    meta_path: Path,
    video_field: str,
    api_key: str,
    num_threads: int,
    sampling_fps: int,
) -> Dict[int, Dict[str, Any]]:
    client = genai.Client(api_key=api_key)
    tasks: List[Tuple[int, Path, str]] = []
    for entry in entries:
        # Use global_id as the unique identifier
        motion_id = entry.get("global_id", entry.get("id"))
        if motion_id is None:
            continue
        raw_video_path = entry.get(video_field)
        if raw_video_path is None:
            continue
        video_path = resolve_repo_path(meta_path, str(raw_video_path))
        if not video_path.exists():
            continue
        prompt = PROMPT_TEMPLATE.format(description=entry.get("prompt", ""))
        tasks.append((int(motion_id), video_path, prompt))

    results: Dict[int, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_id = {
            executor.submit(call_gemini, client, video_path, prompt, sampling_fps): motion_id
            for motion_id, video_path, prompt in tasks
        }
        for future in tqdm(as_completed(future_to_id), total=len(future_to_id), desc="Gemini"):
            motion_id = future_to_id[future]
            try:
                results[motion_id] = future.result()
            except Exception as exc:
                results[motion_id] = {"matches": False, "analysis": str(exc), "raw": ""}
    return results


def decide_quality(
    motion_metrics: Dict[int, Dict[str, float]],
    vlm_results: Dict[int, Dict[str, Any]],
    jitter_threshold: float,
) -> Dict[int, bool]:
    final = {}
    for motion_id, metrics in motion_metrics.items():
        jitter = metrics.get("Jitter_Degree", float("inf"))
        vlm = vlm_results.get(motion_id, {"matches": False})
        is_good = jitter < jitter_threshold and vlm.get("matches", False)
        final[motion_id] = is_good
    return final


def apply_quality_gate(
    meta_json: Path,
    quality_report: Path,
    *,
    device: str,
    jitter_threshold: float,
    max_samples: Optional[int],
    gemini_api_key: Optional[str],
    video_field: str,
    eval_field: str,
    num_threads: int,
    sampling_fps: float,
) -> None:
    entries = json.loads(meta_json.read_text())
    if not isinstance(entries, list):
        raise ValueError(f"Expected list in {meta_json}")

    if max_samples:
        entries = entries[:max_samples]

    samples = list_motion_samples(entries, meta_json, eval_field)
    motion_metrics = compute_motion_quality_metrics(samples, device)
    sample_ids = [motion_id for motion_id, _ in samples]

    if not gemini_api_key:
        raise RuntimeError("Gemini API key is required for VLM gating.")

    sampling_fps_int = max(1, int(round(sampling_fps)))
    vlm_results = run_alignment_checks(entries, meta_json, video_field, gemini_api_key, num_threads, sampling_fps_int)

    final_quality = decide_quality(motion_metrics, vlm_results, jitter_threshold)

    updated_entries = []
    for entry in entries:
        motion_id = entry.get("global_id", entry.get("id"))
        use_ref = bool(final_quality.get(motion_id, False))
        new_entry = dict(entry)
        new_entry["use_ref_motion"] = use_ref
        updated_entries.append(new_entry)

    meta_json.write_text(json.dumps(updated_entries, indent=2))

    quality_records = []
    for motion_id in sample_ids:
        vlm_entry = vlm_results.get(motion_id, {})
        quality_records.append(
            {
                "global_id": motion_id,
                "motion_metrics": motion_metrics.get(motion_id, {}),
                "vlm_analysis": vlm_entry.get("analysis"),
                "vlm_raw": vlm_entry.get("raw"),
                "vlm_matches": vlm_entry.get("matches", False),
                "final_quality": final_quality.get(motion_id, False),
            }
        )
    quality_report.parent.mkdir(parents=True, exist_ok=True)
    quality_report.write_text(
        json.dumps({"source_meta": str(meta_json), "records": quality_records}, indent=2)
    )
    print(f"Wrote updated metadata with use_ref_motion -> {meta_json}")
    print(f"Wrote quality report -> {quality_report}")


def main() -> None:
    args = parse_args()
    apply_quality_gate(
        args.meta_json.expanduser().resolve(),
        args.quality_report.expanduser().resolve(),
        device=args.device,
        jitter_threshold=args.jitter_threshold,
        max_samples=args.max_samples,
        gemini_api_key=args.gemini_api_key,
        video_field=args.video_field,
        eval_field=args.eval_field,
        num_threads=args.num_threads,
        sampling_fps=args.sampling_fps,
    )


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
