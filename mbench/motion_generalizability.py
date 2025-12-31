import os
import glob
import time
import re
import random
import json
from collections import defaultdict
from tqdm import tqdm
from typing import List, Dict, Any, Optional
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from mbench.utils import load_dimension_info
from mbench.vlm_utils import call_gemini_api, GEMINI_AVAILABLE


def summarize_binary(values: List[float]) -> Dict[str, float]:
    """Return mean/std/count summary for binary values."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "num_samples": 0}
    arr = np.asarray(values, dtype=np.float32)
    std = float(arr.std()) if arr.size > 1 else 0.0
    return {"mean": float(arr.mean()), "std": std, "num_samples": int(arr.size)}


def compute_motion_generalizability(full_info_path: str, device: str,
                          api_key: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """
    Compute action accuracy for motion videos using Gemini API
    
    Args:
        full_info_path: Path to JSON file with video information
        device: Device to use (not used in this implementation)
        api_key: Gemini API key (can also be set via GEMINI_API_KEY env var)
        vlm_fps: Target effective fps for VLM sampling (default: 5)
        **kwargs: Additional arguments
        
    Returns:
        Dictionary with evaluation results
    """
    # Get API key from parameter or environment
    if api_key is None:
        api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        raise ValueError("Gemini API key must be provided either as parameter or GEMINI_API_KEY environment variable")
    
    # Load dimension information using the same pattern as motion_quality.py
    prompt_dict_ls = load_dimension_info(full_info_path, dimension='Motion_Generalizability')
    
    # Base prompt for motion analysis
    prompt = """
    You are given a rendered video showing captured human motion, with no background or interacting objects. First, analyze the human motion in the video in detail by describing the person's body posture, limb movements, and any repetitive or distinctive motion patterns you observe. 
    Then, compare the observed motion with the provided textual description of an expected motion. When evaluating, prioritize the semantic consistency of the action (e.g., whether the core intent and type of movement match) over the visual quality, smoothness, or physical realism of the rendering. Finally, determine whether the observed motion matches the described motion. Respond with a short justification followed by a final answer in "Answer: yes" or "Answer: no".
    Example output is: "The person in the video is standing and repeatedly clapping their hands above chest level. The motion in the video matches the given motion description 'clapping'. Answer: yes."
    The provided motion description to verify: """
    
    gemini_workers = kwargs.get('gemini_workers', 8)
    vlm_fps = kwargs.get('vlm_fps', 5)  # Default 1fps for VLM sampling
    
    cnt = 0
    cor_num = 0
    accuracy_list = []
    per_motion_metrics = []

    def evaluate_entry(prompt_dict):
        video_label = prompt_dict.get('prompt', 'unknown')
        video_path = prompt_dict.get('evaluation_file', '')
        motion_id = prompt_dict.get('id')

        if not os.path.exists(video_path):
            return {"skip": True, "id": motion_id, "error": f"Missing video {video_path}"}

        try:
            # Read video bytes directly - fps is now handled by Gemini API
            with open(video_path, 'rb') as f:
                video_bytes = f.read()
        except Exception as exc:
            return {"skip": True, "id": motion_id, "error": f"Video read error: {exc}"}


        try:
            response = call_gemini_api(video_bytes, prompt + f'"{video_label}".', api_key, fps=vlm_fps)

        except Exception as exc:
            return {"skip": True, "id": motion_id, "error": f"Gemini error: {exc}"}

        response = response.encode('utf-8').decode('utf-8').strip()
        decision_text = response.split("Answer:", 1)[-1].strip()
        is_correct = "yes" in decision_text.lower()

        return {
            "skip": False,
            "id": motion_id,
            "is_correct": is_correct,
            "video_label": video_label,
            "video_path": video_path,
            "decision_text": decision_text,
            "raw_response": response,
        }

    with ThreadPoolExecutor(max_workers=gemini_workers) as executor:
        futures = {executor.submit(evaluate_entry, prompt_dict): prompt_dict for prompt_dict in prompt_dict_ls}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Gemini MG"):
            try:
                result = future.result()
            except Exception as exc:
                print(f"Gemini worker failed: {exc}")
                continue

            if result.get("skip"):
                error = result.get("error")
                if error:
                    print(error)
                continue

            cnt += 1
            if result["is_correct"]:
                cor_num += 1
                accuracy_list.append(1.0)
            else:
                accuracy_list.append(0.0)

            per_motion_metrics.append(
                {
                    "id": result["id"],
                    "prompt": result["video_label"],
                    "value": 1.0 if result["is_correct"] else 0.0,
                    "video_path": result["video_path"],
                    "decision": result["decision_text"],
                    "raw_response": result["raw_response"],
                }
            )

    accuracy = cor_num / cnt if cnt > 0 else 0.0
    aggregate = summarize_binary(accuracy_list)
    aggregate.update({"accuracy": accuracy, "correct": cor_num, "total": cnt})
    
    return {
        "aggregate": aggregate,
        "per_motion": per_motion_metrics,
    }
