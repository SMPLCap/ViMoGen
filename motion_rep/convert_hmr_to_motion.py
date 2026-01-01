import argparse
import torch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smplx import SMPLX
from motion_rep.retarget_motion import collect_motion_rep_DART, process_hmr_motion, JOINT_NUM


def load_smplx_model(model_path: str, batch_size: int, device: str) -> SMPLX:
    """Load SMPLX model with specified batch size."""
    model = SMPLX(
        model_path=model_path,
        gender='neutral',
        use_pca=False,
        num_betas=10,
        batch_size=batch_size,
    ).to(device)
    model.eval()
    return model


def convert_hmr_to_motion(
    hmr_data: dict,
    smplx_model: SMPLX,
    device: str = 'cuda:0',
) -> dict:
    """
    Convert HMR SMPL output to our 276-dim motion representation.
    
    Args:
        hmr_data: dict containing SMPL params and optionally camera info from HMR
        smplx_model: loaded SMPLX model
        device: compute device
        
    Returns:
        dict with 'motion', 'intrinsic', 'extrinsic'
    """
    # Extract SMPL params (assume axis-angle format)
    global_orient = hmr_data['global_orient'].to(device)  # (T, 3)
    body_pose = hmr_data['body_pose'].to(device)          # (T, 63)
    transl = hmr_data['transl'].to(device)                # (T, 3)
    
    seq_len = global_orient.shape[0]
    
    # Prepare params for SMPLX forward
    # SMPLX expects axis-angle inputs: global_orient (T, 3), body_pose (T, 63)
    smpl_params = {
        'global_orient': global_orient,  # (T, 3)
        'body_pose': body_pose,          # (T, 63)
        'transl': transl,                # (T, 3)
    }
    
    # Run SMPLX forward to get joints
    with torch.no_grad():
        smpl_results = smplx_model(**smpl_params)
    joints = smpl_results.joints[:, :JOINT_NUM]  # (T, 22, 3)
    
    # Prepare axis-angle params for motion rep collection
    smpl_params_aa = {
        'global_orient': global_orient,  # (T, 3)
        'body_pose': body_pose,          # (T, 63)
        'transl': transl,                # (T, 3)
    }
    
    # Collect motion representation
    hmr_motion = collect_motion_rep_DART(smpl_params_aa, joints)  # (T-1, 276)
    
    # Build intrinsic matrix from camera params (optional)
    if 'focal_length' in hmr_data and 'width' in hmr_data and 'height' in hmr_data:
        focal = hmr_data['focal_length']
        if isinstance(focal, torch.Tensor):
            focal = focal.flatten()[0].item()
        
        width = hmr_data['width']
        if isinstance(width, torch.Tensor):
            width = width.flatten()[0].item()
            
        height = hmr_data['height']
        if isinstance(height, torch.Tensor):
            height = height.flatten()[0].item()
        
        intrinsic = torch.tensor([
            [focal, 0.0, width / 2],
            [0.0, focal, height / 2],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32, device=device)
    else:
        # Use identity intrinsic if camera params not provided
        intrinsic = torch.eye(3, dtype=torch.float32, device=device)
    
    # Process motion (align to floor, canonicalize)
    # process_hmr_motion returns (data_dict, joints_canonical)
    # data_dict contains: motion, extrinsic, intrinsic
    output_data, joints_canonical = process_hmr_motion(
        hmr_motion, intrinsic, set_floor=True, to_cpu=True
    )
    
    return output_data


def main():
    parser = argparse.ArgumentParser(description='Convert HMR output to motion representation')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to input HMR .pt file')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Path to output .pt file')
    parser.add_argument('--smplx_model_path', type=str, 
                        default='./data/body_models/smplx',
                        help='Path to SMPLX model directory')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Compute device')
    args = parser.parse_args()
    
    # Check input exists
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading HMR data from: {input_path}")
    hmr_data = torch.load(input_path, map_location='cpu', weights_only=False)
    
    # Validate required keys
    required_keys = ['global_orient', 'body_pose', 'transl']
    missing_keys = [k for k in required_keys if k not in hmr_data]
    if missing_keys:
        raise KeyError(f"Missing required keys in HMR data: {missing_keys}")
    
    print(f"Input sequence length: {hmr_data['global_orient'].shape[0]} frames")
    
    # Check optional camera params
    if 'focal_length' in hmr_data and 'width' in hmr_data and 'height' in hmr_data:
        print(f"Camera: focal={hmr_data['focal_length']}, size={hmr_data['width']}x{hmr_data['height']}")
    else:
        print("Camera params not provided, using identity intrinsic (reprojection not available)")
    
    # Load SMPLX model
    seq_len = hmr_data['global_orient'].shape[0]
    print(f"Loading SMPLX model from: {args.smplx_model_path}")
    smplx_model = load_smplx_model(args.smplx_model_path, seq_len, args.device)
    
    # Convert
    print("Converting to motion representation...")
    output = convert_hmr_to_motion(
        hmr_data, smplx_model, 
        device=args.device
    )
    
    print(f"Output motion shape: {output['motion'].shape}")
    print(f"Saving to: {output_path}")
    torch.save(output, output_path)
    
    print("Done!")


if __name__ == '__main__':
    main()
