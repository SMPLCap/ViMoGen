import argparse
import json
import os
import numpy as np
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def load_motion_tensor(motion_path):
    """Load motion tensor from .pt or .npy file."""
    if motion_path.endswith(".pt"):
        motion = torch.load(motion_path, weights_only=True, map_location="cpu")
    elif motion_path.endswith(".npy"):
        motion = torch.from_numpy(np.load(motion_path)).float()
    else:
        raise ValueError(f"Unsupported motion file format: {motion_path}")
    
    if isinstance(motion, dict):
        motion = motion["motion"]
    return motion


def compute_statistics(data_list, num_workers=8):
    """Compute mean and std from motion data using the full dataset."""
    # Collect all motion tensors
    all_motions = []
    failed_count = 0
    
    def load_single_motion(entry):
        motion_path = entry['motion_path']
        try:
            motion = load_motion_tensor(motion_path)
            return motion
        except Exception as e:
            return None
    
    # Use thread pool for parallel loading
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(load_single_motion, entry): entry for entry in data_list}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading motions for stats"):
            result = future.result()
            if result is not None:
                all_motions.append(result)
            else:
                failed_count += 1
    
    print(f"Loaded {len(all_motions)} motions, {failed_count} failed")
    
    # Concatenate all motions along frame dimension
    all_frames = torch.cat(all_motions, dim=0)
    print(f"Total frames: {all_frames.shape[0]}, Motion dim: {all_frames.shape[1]}")
    
    # Compute mean and std
    mean = all_frames.mean(dim=0).numpy()
    std = all_frames.std(dim=0).numpy()
    
    # Avoid division by zero
    std = np.clip(std, a_min=1e-8, a_max=None)
    
    return mean, std


def main():
    parser = argparse.ArgumentParser(description="Prepare ViMoGen-228K training data")
    parser.add_argument("--input_json", type=str, required=True,
                        help="Path to original ViMoGen-228K.json")
    parser.add_argument("--motion_root", type=str, default="./data/ViMoGen-228K",
                        help="Root directory for motion files")
    parser.add_argument("--output_dir", type=str, default="./data/meta_info",
                        help="Output directory for processed files")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of workers for parallel loading")
    parser.add_argument("--skip_stats", action="store_true",
                        help="Skip statistics computation (use pre-computed mean/std in ./data/meta_info/)")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load original JSON
    print(f"Loading {args.input_json}...")
    with open(args.input_json, 'r') as f:
        data_list = json.load(f)
    print(f"Loaded {len(data_list)} entries")
    
    # Process each entry
    print("Processing entries...")
    for entry in tqdm(data_list, desc="Adding sample_id and motion_root"):
        # Add sample_id from id field
        entry['sample_id'] = str(entry['id'])
        
        # Prefix motion_path with motion_root
        original_path = entry['motion_path']
        entry['motion_path'] = os.path.join(args.motion_root, original_path)
    
    # Compute statistics unless skipped
    if not args.skip_stats:
        print("Computing motion statistics from full dataset...")
        mean, std = compute_statistics(data_list, args.num_workers)
        
        # Save statistics
        mean_path = os.path.join(args.output_dir, "mean.npy")
        std_path = os.path.join(args.output_dir, "std.npy")
        np.save(mean_path, mean)
        np.save(std_path, std)
        print(f"Saved mean to {mean_path}")
        print(f"Saved std to {std_path}")
        print(f"Mean shape: {mean.shape}, Std shape: {std.shape}")
    else:
        print("Skipping statistics computation (--skip_stats)")
    
    # Save processed JSON
    output_json = os.path.join(args.output_dir, "ViMoGen-228K_train.json")
    print(f"Saving processed JSON to {output_json}...")
    with open(output_json, 'w') as f:
        json.dump(data_list, f, indent=2)
    print(f"Done! Processed {len(data_list)} entries.")


if __name__ == "__main__":
    main()
