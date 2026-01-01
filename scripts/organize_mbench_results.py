import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

# Add project root to path for motion_rep import
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np

# Coordinate conversion matrices (same as in motion_gating/mbench_render.py)
_BASE_CONVERSION = torch.tensor(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=torch.float32,
)
_FRONT_ROTATION = torch.tensor(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=torch.float32,
)
COORD_CONVERSION = torch.matmul(_FRONT_ROTATION, _BASE_CONVERSION)


def parse_args():
    parser = argparse.ArgumentParser(description='Organize MBench evaluation results')
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Input directory containing inference results'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='exp/mbench_eval_input',
        help='Output directory for organized results'
    )
    return parser.parse_args()


def load_motion_tensor(path: Path) -> torch.Tensor:
    """Load and validate motion tensor from a .pt file."""
    data = torch.load(path, map_location="cpu")
    if isinstance(data, dict):
        if "motion" in data:
            data = data["motion"]
        else:
            raise ValueError(f"{path} contains a dict but no 'motion' field.")
    if not torch.is_tensor(data):
        data = torch.as_tensor(data)
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D motion tensor in {path}, got shape {tuple(data.shape)}")
    return data.float()


def convert_motion_to_joints(motion_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert 276-dim motion representation to 3D joints for MBench evaluation.
    
    Args:
        motion_tensor: Shape (frames, 276) raw motion representation
        
    Returns:
        joints: Shape (frames, 22, 3) 3D joint positions
    """
    from motion_rep.retarget_motion import motion_rep_to_SMPL
    
    # Extract SMPL parameters and recovered joints
    smpl_data, recovered_joints = motion_rep_to_SMPL(
        motion_tensor,
        recover_from_velocity=True,
        equal_length=False,
    )
    
    # Apply coordinate conversion (same as mbench_render.py)
    # recovered_joints shape: (frames, 22, 3)
    joints = torch.einsum("ij,tvj->tvi", COORD_CONVERSION, recovered_joints)
    
    return joints.numpy().astype("float32")


def find_motion_file(folder_path: Path) -> Path:
    """Find the motion .pt file in a folder, prioritizing motion-conditioned results."""
    motion_files = [
        'motion_gen_condition_on_motion.pt',
        'motion_gen_condition_on_text.pt',
    ]
    
    for motion_file in motion_files:
        file_path = folder_path / motion_file
        if file_path.exists():
            return file_path
    
    # Fallback: find any .pt file
    pt_files = list(folder_path.glob('*.pt'))
    if pt_files:
        return pt_files[0]
    
    return None


def main():
    args = parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all subdirectories (numbered folders)
    subfolders = [f for f in input_dir.iterdir() if f.is_dir()]
    
    # Sort by numeric value
    try:
        subfolders.sort(key=lambda x: int(x.name))
    except ValueError:
        subfolders.sort()
    
    print(f"Found {len(subfolders)} result folders in {input_dir}")
    
    collected_count = 0
    error_count = 0
    
    for folder in tqdm(subfolders, desc="Processing motions"):
        motion_file = find_motion_file(folder)
        
        if motion_file is None:
            print(f"Warning: No motion file found in {folder}")
            error_count += 1
            continue
        
        try:
            # Load motion tensor
            motion_tensor = load_motion_tensor(motion_file)
            
            # Convert to 3D joints with coordinate transformation
            joints = convert_motion_to_joints(motion_tensor)
            
            # Output filename: {folder_id}.pt or .npy
            output_filename = f"{folder.name}.npy"
            output_path = output_dir / output_filename
            
            # Save as numpy array (MBench render.py expects (frames, joints, 3))
            np.save(output_path, joints)
            
            collected_count += 1
            
        except Exception as e:
            print(f"Error processing {folder.name}: {e}")
            error_count += 1
            continue
    
    print(f"\n=== Summary ===")
    print(f"Total folders processed: {len(subfolders)}")
    print(f"Successfully converted: {collected_count}")
    print(f"Errors: {error_count}")
    print(f"Output directory: {output_dir}")
    print(f"\nTo run evaluation:")
    print(f"  python evaluate_mbench.py --evaluation_path {output_dir}")


if __name__ == "__main__":
    main()
