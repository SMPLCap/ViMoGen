import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motion_gating.mbench_render import convert_and_render


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render MBench-style videos and update meta JSON.")
    parser.add_argument(
        "--meta-json",
        type=Path,
        default=Path("data_samples/example_archive_wi_ref.json"),
        help="metadata JSON containing motion_path entries.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data_samples/example_archive_wi_ref_eval.json"),
        help="Where to write updated metadata with video_path and mbench_eval_path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_samples/mb_render"),
        help="Directory to store rendered mp4/pt outputs.",
    )
    parser.add_argument(
        "--smplx-model-dir",
        type=Path,
        default=Path("data/body_models/smplx"),
        help="Directory containing SMPLX model files.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Frames per second for rendered videos.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=960,
        help="Video width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=960,
        help="Video height.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device used by MBench renderer (CUDA recommended).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit to process first N entries (debug).",
    )
    return parser.parse_args()


def find_repo_root(meta_path: Path) -> Path:
    for parent in meta_path.parents:
        if (parent / "README.md").exists():
            return parent
    return meta_path.parent


def resolve_repo_path(meta_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        repo_root = find_repo_root(meta_path)
        path = (repo_root / path).resolve()
    return path


def render_single_motion(
    motion_file: Path,
    motion_id: Any,
    output_dir: Path,
    smplx_model_dir: Path,
    device: str,
    fps: int,
    width: int,
    height: int,
) -> Dict[str, str]:
    pt_path, mp4_path = convert_and_render(
        motion_file,
        str(motion_id),
        output_dir,
        smplx_model_dir,
        device=device,
        fps=fps,
        width=width,
        height=height,
    )
    return {"pt": str(pt_path), "mp4": str(mp4_path)}


def update_entry_paths(entry: Dict[str, Any], repo_root: Path, pt_path: Path, mp4_path: Path) -> Dict[str, Any]:
    new_entry = dict(entry)
    try:
        new_entry["video_path"] = str(mp4_path.relative_to(repo_root))
    except ValueError:
        new_entry["video_path"] = str(mp4_path)
    try:
        new_entry["mbench_eval_path"] = str(pt_path.relative_to(repo_root))
    except ValueError:
        new_entry["mbench_eval_path"] = str(pt_path)
    return new_entry


def main() -> None:
    args = parse_args()
    meta_json = args.meta_json.expanduser().resolve()
    output_json = args.output_json.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    smplx_model_dir = args.smplx_model_dir.expanduser().resolve()
    repo_root = find_repo_root(meta_json)

    entries = json.loads(meta_json.read_text())
    if not isinstance(entries, list):
        raise ValueError(f"Expected a list in {meta_json}, got {type(entries).__name__}")

    updated_entries: List[Dict[str, Any]] = []
    to_process = entries if args.limit is None else entries[: args.limit]

    for entry in tqdm(to_process, desc="Render MBench videos"):
        motion_path = entry.get("motion_path")
        if not motion_path:
            updated_entries.append(entry)
            continue
        motion_file = resolve_repo_path(meta_json, str(motion_path))
        # Use global_id to ensure unique file names across dimensions
        motion_id = entry.get("global_id", entry.get("id", Path(motion_file).stem))
        outputs = render_single_motion(
            motion_file,
            motion_id,
            output_dir,
            smplx_model_dir,
            args.device,
            fps=args.fps,
            width=args.width,
            height=args.height,
        )
        updated_entry = update_entry_paths(entry, repo_root, Path(outputs["pt"]), Path(outputs["mp4"]))
        updated_entries.append(updated_entry)

    # Preserve any remaining entries if limit was used
    if args.limit is not None and len(entries) > args.limit:
        updated_entries.extend(entries[args.limit :])

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(updated_entries, indent=2))
    print(f"Rendered videos/PT caches to {output_dir}")
    print(f"Wrote updated meta JSON with video_path/mbench_eval_path to {output_json}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
