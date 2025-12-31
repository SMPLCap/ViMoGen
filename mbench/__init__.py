import os
import json
import importlib
import shutil
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import torch

from .utils import save_json, load_json


class MBench(object):
    """
    MBench: Motion generation benchmark for evaluating human motion videos
    """
    
    def __init__(self, device: str, output_path: str, full_info_dir: Optional[str] = None):
        """
        Initialize MBench evaluator
        
        Args:
            device: Device to use for evaluation ('cuda' or 'cpu')
            output_path: Directory to save evaluation results
            full_info_dir: Path to full info JSON file (optional)
        """
        self.device = device
        self.output_path = output_path
        self.full_info_dir = full_info_dir
        os.makedirs(self.output_path, exist_ok=True)
        self.category_dir_map = {
            "motion_quality": "Motion_Quality",
            "pose_quality": "Pose_Quality",
            "motion_condition_consistency": "Motion_Condition_Consistency",
            "motion_generalizability": "Motion_Generalizability"
        }

    def build_full_dimension_list(self) -> List[str]:
        """
        Build the complete list of available evaluation dimensions
        
        Returns:
            List of available dimension names organized by categories
        """
        return [
            # Motion Quality category dimensions
            "Jitter_Degree", 
            "Ground_Penetration",
            "Foot_Floating", 
            "Foot_Sliding",
            "Dynamic_Degree",
            "Body_Penetration",
            "Pose_Quality",
            # Motion-Condition Consistency category
            "Motion_Condition_Consistency",
            # Motion Generalizability category
            "Motion_Generalizability"
        ]

    def get_category(self, dimension: str) -> str:
        """
        Get mapping of dimensions to their categories
        
        Returns:
            mapping category name
        """
        dimension_categories = {
            # Motion_Quality category
            "Jitter_Degree": "motion_quality",
            "Ground_Penetration": "motion_quality",
            "Foot_Floating": "motion_quality",
            "Foot_Sliding": "motion_quality",
            "Dynamic_Degree": "motion_quality",
            "Body_Penetration": "pose_quality",
            "Pose_Quality": "pose_quality",
            
            # Motion_Condition_Consistency category
            "Motion_Condition_Consistency": "motion_condition_consistency",
            
            # Motion_Generalizability category
            "Motion_Generalizability": "motion_generalizability"
        }
        return dimension_categories[dimension]

    def _get_mbench_dir(self, evaluation_path: str) -> Path:
        """Return the directory that stores converted assets (same as evaluation path)."""
        mbench_dir = Path(evaluation_path)
        mbench_dir.mkdir(parents=True, exist_ok=True)
        return mbench_dir

    def _motion_candidates(self, evaluation_path: Path, motion_id: int) -> List[Path]:
        """Return ordered candidate motion file paths to probe."""
        candidates = []
        for suffix in (".pt", ".npy"):
            candidates.append(evaluation_path / f"{motion_id}{suffix}")
        return candidates

    def _video_candidates(self, evaluation_path: Path, motion_id: int) -> List[Path]:
        """Return ordered candidate video file paths to probe."""
        candidates = []
        candidates.append(evaluation_path / f"{motion_id}.mp4")
        return candidates

    def _find_source_motion_path(self, evaluation_path: str, category: str, motion_id: int) -> Optional[Path]:
        """Locate the raw motion file for a given category/id."""
        evaluation_path = Path(evaluation_path)
        category_dir = self.category_dir_map.get(category)
        if category_dir:
            base_dir = evaluation_path / category_dir / str(motion_id)
            for filename in ["motion.pt", "motion.npy"]:
                candidate = base_dir / filename
                if candidate.exists():
                    return candidate

        for candidate in self._motion_candidates(evaluation_path, motion_id):
            if candidate.exists():
                return candidate
        return None

    def _find_source_video_path(self, evaluation_path: str, category: str, motion_id: int) -> Optional[Path]:
        """Locate an already rendered video for a given category/id."""
        evaluation_path = Path(evaluation_path)
        category_dir = self.category_dir_map.get(category)
        if category_dir:
            video_candidate = evaluation_path / category_dir / str(motion_id) / "motion.mp4"
            if video_candidate.exists():
                return video_candidate

        for candidate in self._video_candidates(evaluation_path, motion_id):
            if candidate.exists():
                return candidate
        return None

    def _load_motion_array(self, motion_file: Path):
        """Load motion data from a supported file into a numpy array."""
        if motion_file.suffix == ".pt":
            motions = torch.load(motion_file, map_location="cpu")
            if isinstance(motions, dict):
                motions = motions.get("joints", motions.get("motion", motions))
            motions = motions.numpy() if hasattr(motions, "numpy") else motions
        elif motion_file.suffix == ".npy":
            motions = np.load(motion_file)
        else:
            raise ValueError(f"Unsupported motion file format: {motion_file}")
        return motions

    def render_motions_to_videos(self, evaluation_path: str, dimension_list: List[str], 
                            name: str, device_id: int = 0) -> str:
        """
        Render motion files to intermediate files for evaluation
        Only renders each category once
        
        Args:
            evaluation_path: Path containing dimension folders with motion files
            dimension_list: List of dimensions to process
            name: Name for the evaluation run
            device_id: CUDA device ID for rendering
        """
        from .render import render
        
        # Load full info list
        full_info_list = load_json(self.full_info_dir)
        if not full_info_list:
            print("WARNING: No entries found in full info list")
            return
        
        processed_categories = set()
        rendered_count = 0

        # Only video-based metrics need SMPLify + video rendering
        # pose_quality needs SMPLify for pose/vertices but no video
        # motion_quality only needs raw 3D joints (already in .npy format)
        video_categories = {"motion_condition_consistency", "motion_generalizability"}
        smplify_categories = {"pose_quality"}  # Need SMPLify but no video
        joints_only_categories = {"motion_quality"}  # Only need raw joints
        mbench_dir = self._get_mbench_dir(evaluation_path)
        
        for dimension in dimension_list:
            category = self.get_category(dimension)
            
            if category in processed_categories:
                print(f'Skipping dimension {dimension} already processed')
                continue
            processed_categories.add(category)

            # For motion_quality, skip SMPLify entirely - use raw 3D joints from .npy files
            if category in joints_only_categories:
                print(f'Dimension {dimension} uses raw 3D joints; skipping SMPLify rendering')
                continue

            # Filter entries for current dimension
            dimension_entries = [
                entry for entry in full_info_list if entry["dimension"] == dimension
            ]
            
            for entry in dimension_entries:
                motion_id = entry["id"]
                try:
                    target_pt = mbench_dir / f"{motion_id}.pt"
                    target_video = mbench_dir / f"{motion_id}.mp4"

                    # For video categories, check if video already exists
                    if category in video_categories and target_video.exists():
                        print(f'Using existing rendered video for ID {motion_id} at {target_video}')
                        continue

                    # For smplify_categories (pose_quality), check if .pt file already exists with required keys
                    if category in smplify_categories and target_pt.exists():
                        try:
                            existing_data = torch.load(target_pt, map_location='cpu', weights_only=False)
                            if isinstance(existing_data, dict) and 'pose' in existing_data and 'vertices' in existing_data:
                                print(f'Using existing SMPLify cache for ID {motion_id} at {target_pt}')
                                continue
                        except Exception:
                            pass  # Will re-render

                    # Try to reuse already rendered videos before rerendering
                    if category in video_categories and not target_video.exists():
                        source_video = self._find_source_video_path(evaluation_path, category, motion_id)
                        if source_video and source_video.resolve() != target_video.resolve():
                            shutil.copy2(source_video, target_video)
                            print(f'Copied existing video for ID {motion_id} from {source_video} to {target_video}')
                            continue

                    motion_file = self._find_source_motion_path(evaluation_path, category, motion_id)
                    if motion_file is None:
                        print(f'WARNING: No motion file found for ID {motion_id} in any supported format (.pt, .npy)')
                        continue

                    motions = self._load_motion_array(motion_file)

                    # Handle different motion data shapes
                    if len(motions.shape) == 4:  # (batch, frames, joints, features)
                        motions = motions[0]  # Take first sample
                    elif len(motions.shape) == 3:  # (frames, joints, features)
                        pass  # Already correct shape
                    else:
                        print(f'WARNING: Unexpected motion data shape {motions.shape} in {motion_file}')
                        continue

                    # Set render_video based on category and only render when needed
                    render_video = category in video_categories and not target_video.exists()
                    
                    # Render motion into the shared MBench cache directory
                    render(motions, outdir=str(mbench_dir), device_id=device_id,
                            name=str(motion_id), render_video=render_video)
                    
                    rendered_count += 1
                    
                except Exception as e:
                    print(f'ERROR rendering motion {motion_id}: {str(e)}')
                    continue
        
        print(f'Successfully rendered {rendered_count} unique motions across all categories')

    def build_full_info_json(self, evaluation_path: str, name: str, 
                           dimension_list: List[str], **kwargs) -> str:
        """
        Build the full information JSON file for evaluation
        
        Args:
            evaluation_path: Path containing dimension folders with motion files
            name: Name for the evaluation run
            dimension_list: List of dimensions to evaluate
            **kwargs: Additional arguments
            
        Returns:
            Path to the created full info JSON file
        """
        cur_full_info_list = []
        mbench_dir = self._get_mbench_dir(evaluation_path)
        
        # Separate video-based, pose-quality (needs SMPLify .pt), and motion-quality (needs .npy) dimensions  
        video_dimension_list = ["Motion_Condition_Consistency", "Motion_Generalizability"]
        pose_quality_dimension_list = ["Pose_Quality", "Body_Penetration"]  # Need .pt with pose/vertices
        
        video_dimensions = [dim for dim in dimension_list if dim in video_dimension_list]
        pose_dimensions = [dim for dim in dimension_list if dim in pose_quality_dimension_list]
        motion_dimensions = [dim for dim in dimension_list if dim not in video_dimensions and dim not in pose_dimensions]
        
        full_info_list = load_json(self.full_info_dir)
        for prompt_dict in full_info_list:
            if prompt_dict["dimension"] in video_dimensions: 
                candidates = [
                    mbench_dir / f'{prompt_dict["id"]}.mp4',
                    self._find_source_video_path(evaluation_path, self.get_category(prompt_dict["dimension"]), prompt_dict["id"])
                ]
            elif prompt_dict["dimension"] in pose_dimensions:
                # For pose_quality dimensions, need .pt files with pose/vertices (from SMPLify)
                candidates = [
                    mbench_dir / f'{prompt_dict["id"]}.pt',   # SMPLify output with pose/vertices
                ]
            elif prompt_dict["dimension"] in motion_dimensions:
                # For motion_quality dimensions, check .npy first (raw joints), then .pt
                candidates = [
                    mbench_dir / f'{prompt_dict["id"]}.npy',  # Raw joints from organize script
                    mbench_dir / f'{prompt_dict["id"]}.pt',   # Processed from SMPLify
                ]
            else:
                candidates = []

            evaluation_file = None
            for candidate in candidates:
                if candidate and Path(candidate).exists():
                    evaluation_file = str(candidate)
                    break

            if evaluation_file is None:
                print(f'WARNING: No evaluation file found for ID {prompt_dict["id"]} and dimension {prompt_dict["dimension"]}')
                continue

            prompt_copy = prompt_dict.copy()
            prompt_copy["evaluation_file"] = evaluation_file
            cur_full_info_list.append(prompt_copy)
        
        cur_full_info_path = os.path.join(self.output_path, f'{name}_full_info.json')
        save_json(cur_full_info_list, cur_full_info_path)
        print(f'Evaluation meta data saved to {cur_full_info_path}')
        return cur_full_info_path

    def evaluate(self, evaluation_path: str, name: str, 
                dimension_list: List[str] = None, 
                device_id: int = 0, **kwargs) -> Dict[str, Any]:
        """
        Run evaluation on motion data by first rendering to videos and then evaluating
        
        Args:
            evaluation_path: Path containing dimension folders with motion files
            name: Name for this evaluation run
            dimension_list: List of dimensions to evaluate (if None, evaluates all)
            device_id: CUDA device ID for rendering
            **kwargs: Additional arguments passed to evaluation functions
            
        Returns:
            Dictionary containing evaluation results for each dimension
        """
        results_dict = {}
        
        if dimension_list is None:
            dimension_list = self.build_full_dimension_list()
        
        # First, render motions to intermediate videos for video-based dimensions
        # if "Motion_Condition_Consistency" in dimension_list or "Motion_Generalizability" in dimension_list:
        #     print("Rendering motions to intermediate videos...")
        self.render_motions_to_videos(evaluation_path, dimension_list, name, device_id)
        
        # Build the full info JSON file using both videos and pt files
        print("Building evaluation metadata...")
        cur_full_info_path = self.build_full_info_json(
            evaluation_path, name, dimension_list, **kwargs
        )
        
        # Evaluate each dimension
        print("Running evaluations...")
        per_motion_registry: Dict[str, Dict[str, Any]] = {}

        for dimension in dimension_list:
            try:
                # Import the dimension module
                category = self.get_category(dimension)
                dimension_lower = dimension.lower()
                dimension_module = importlib.import_module(f'mbench.{category}')
                evaluate_func = getattr(dimension_module, f'compute_{dimension_lower}')
                
                print(f'Evaluating dimension: {dimension}')
                results = evaluate_func(cur_full_info_path, self.device, **kwargs)
                results_dict[dimension] = results

                per_motion_entries = []
                if isinstance(results, dict):
                    per_motion_entries = results.get("per_motion") or []
                for entry in per_motion_entries:
                    motion_id = entry.get("id")
                    if motion_id is None:
                        continue
                    motion_key = str(motion_id)
                    base_record = per_motion_registry.setdefault(
                        motion_key,
                        {
                            "id": motion_id,
                            "prompt": entry.get("prompt"),
                            "motion_duration": entry.get("motion_duration"),
                            "dimensions": {},
                        },
                    )
                    if base_record.get("prompt") is None and entry.get("prompt") is not None:
                        base_record["prompt"] = entry.get("prompt")
                    if base_record.get("motion_duration") is None and entry.get("motion_duration") is not None:
                        base_record["motion_duration"] = entry.get("motion_duration")

                    dimension_payload = {
                        key: value
                        for key, value in entry.items()
                        if key not in {"id", "prompt", "dimension", "motion_duration"}
                    }
                    base_record["dimensions"][dimension] = dimension_payload
                
            except (ImportError, AttributeError) as e:
                print(f'Error loading dimension {dimension}: {e}')
                results_dict[dimension] = {'error': str(e)}
            except Exception as e:
                print(f'Error evaluating dimension {dimension}: {e}')
                results_dict[dimension] = {'error': str(e)}
        
        # Save results
        output_name = os.path.join(self.output_path, f'{name}_eval_results.json')
        save_json(results_dict, output_name)
        print(f'Evaluation results saved to {output_name}')

        if per_motion_registry:
            per_motion_output = os.path.join(self.output_path, f'{name}_per_motion_results.json')
            def _sort_key(rec: Dict[str, Any]):
                rec_id = rec.get("id")
                if isinstance(rec_id, (int, float)):
                    return (False, float(rec_id))
                if rec_id is None:
                    return (True, float("inf"))
                return (False, str(rec_id))

            sorted_records = sorted(per_motion_registry.values(), key=_sort_key)
            per_motion_payload = {
                "evaluation_name": name,
                "dimensions": dimension_list,
                "motions": sorted_records,
            }
            save_json(per_motion_payload, per_motion_output)
            print(f'Per-motion detailed results saved to {per_motion_output}')
        
        return results_dict 
