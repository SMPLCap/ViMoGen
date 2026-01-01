import torch
import numpy as np
from smplx import SMPL
from mesh_intersection.bvh_search_tree import BVH
from mbench.third_party.NRDF import load_model, axis_angle_to_quaternion
from tqdm import tqdm
from mbench.utils import load_dimension_info
from typing import List


def summarize_scores(values: List[float]):
    """Return mean/std/count summary for scalar values."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "num_samples": 0}
    arr = torch.tensor(values, dtype=torch.float32)
    std = arr.std(unbiased=False).item() if arr.numel() > 1 else 0.0
    return {
        "mean": arr.mean().item(),
        "std": std,
        "num_samples": arr.numel(),
    }

def get_smpl_model(gender='neutral', model_path='data/body_models/smpl', batch_size=1, device='cuda'):
    """
    Load the SMPL model.
    :param gender: 'neutral', 'male', or 'female'
    :param model_path: Path to the SMPL model files
    :return: SMPL model
    """
    kwargs = {'model_path': model_path, 
              'gender': 'neutral', 
              'num_betas': 16,
              'batch_size': batch_size,
              'use_pca': False}
    smpl_model = SMPL(**kwargs).to(device)
    return smpl_model

def smpl_to_joints_and_verts(smpl_model, smpl_params):
    """
    Convert SMPL pose parameters to joint positions and vertices.
    smpl_model: SMPL model object
    smpl_params: Dictionary of SMPL parameters (e.g. body_pose, global_orient, transl)
    """
    
    # Get joints and vertices from SMPL model
    with torch.no_grad():
        output = smpl_model(**smpl_params)
        joints = output.joints  # (T, 24, 3)
        vertices = output.vertices  # (T, 6890, 3)
    
    return joints, vertices


def load_pose_data(evaluation_file: str, device: str, require_pose: bool = False, require_vertices: bool = False):
    """
    Load pose data from evaluation file.
    
    Args:
        evaluation_file: Path to the evaluation file (.npy or .pt)
        device: Target device
        require_pose: If True, raises an error if 'pose' key is missing
        require_vertices: If True, raises an error if 'vertices' key is missing
        
    Returns:
        Dictionary with available data ('joints', 'pose', 'vertices')
        
    Raises:
        ValueError: If required keys are missing
    """
    import numpy as np
    
    if evaluation_file.endswith('.npy'):
        # .npy files only contain raw joints
        if require_pose or require_vertices:
            raise ValueError(
                f"File {evaluation_file} is .npy format (raw joints only). "
                f"Pose_Quality and Body_Penetration require SMPLify-processed .pt files with 'pose' and 'vertices'. "
                f"Please run SMPLify rendering first."
            )
        joints = np.load(evaluation_file)
        return {'joints': torch.from_numpy(joints).float().to(device)}
    else:
        # .pt files from SMPLify have pose, joints, vertices
        data = torch.load(evaluation_file, map_location=device, weights_only=False)
        
        if require_pose and 'pose' not in data:
            raise ValueError(f"File {evaluation_file} missing required 'pose' key")
        if require_vertices and 'vertices' not in data:
            raise ValueError(f"File {evaluation_file} missing required 'vertices' key")
            
        return data

def compute_pose_quality(full_info_path: str, device: str, **kwargs):
    prompt_dict_ls = load_dimension_info(full_info_path, dimension='Pose_Quality')
    nrdf_model_dir = 'checkpoints/nrdf/amass_softplus_l1_0.0001_10000_dist0.5_eik0.0_man0.1'
    nrdf_model = load_model(nrdf_model_dir)

    pose_quality_list = []
    per_motion_metrics = []
    skipped = 0
    
    for prompt_dict in tqdm(prompt_dict_ls):
        evaluation_file = prompt_dict["evaluation_file"]
        try:
            data = load_pose_data(evaluation_file, device, require_pose=True)
        except ValueError as e:
            skipped += 1
            continue
        
        pred_pose = torch.as_tensor(data['pose'], device=device, dtype=torch.float32)

        # Overall pose quality evaluation based on NRDF
        # Convert predicted pose to quaternion
        pose_quat = axis_angle_to_quaternion(pred_pose)
        dist_pred = nrdf_model(pose_quat, train=False)['dist_pred']
        pose_quality = dist_pred.mean().item() * 10

        pose_quality_value = float(pose_quality)
        pose_quality_list.append(pose_quality_value)
        per_motion_metrics.append(
            {
                "id": prompt_dict.get("id"),
                "prompt": prompt_dict.get("prompt"),
                "value": pose_quality_value,
                "evaluation_file": evaluation_file,
                "motion_duration": prompt_dict.get("motion_duration"),
            }
        )
    
    if skipped > 0:
        print(f"Pose_Quality: Skipped {skipped} samples (missing SMPLify data). Run SMPLify rendering first.")
    
    return {
        "aggregate": summarize_scores(pose_quality_list),
        "per_motion": per_motion_metrics,
    }


def compute_body_penetration(full_info_path: str, device: str, **kwargs):
    """
    Compute body penetration/collision percentage using BVH collision detection
    """
    prompt_dict_ls = load_dimension_info(full_info_path, dimension='Body_Penetration')
    gender = 'neutral'
    model_path = 'data/body_models/smpl/SMPL_NEUTRAL.pkl'

    collision_percentage_list = []
    per_motion_metrics = []
    skipped = 0
    
    for prompt_dict in tqdm(prompt_dict_ls):
        evaluation_file = prompt_dict["evaluation_file"]
        try:
            data = load_pose_data(evaluation_file, device, require_vertices=True)
        except ValueError as e:
            skipped += 1
            continue
        
        pred_joints = torch.as_tensor(data['joints'], device=device, dtype=torch.float32)
        pred_vertices = torch.as_tensor(data['vertices'], device=device, dtype=torch.float32)
        motion_length = pred_joints.shape[0]

        # Initialize SMPL model
        smpl_model = get_smpl_model(gender=gender, model_path=model_path, batch_size=motion_length, device=device)
        faces_np = np.asarray(smpl_model.faces, dtype=np.int64)
        faces = torch.as_tensor(faces_np, device=device, dtype=torch.long)
        
        # Self-collision detection
        bvh_model = BVH(max_collisions=8)
        collision_percentage = []
        for frame_idx in range(pred_vertices.shape[0]):
            vertices = pred_vertices[frame_idx].unsqueeze(dim=0)
            triangles = vertices[:, faces]
            outputs = bvh_model(triangles)
            outputs = outputs.detach().cpu().numpy().squeeze()
            collisions = outputs[outputs[:, 0] >= 0, :]
            collision_percentage.append(collisions.shape[0] / float(triangles.shape[1]) * 100)
        collision_percentage = np.mean(collision_percentage)

        collision_value = float(collision_percentage)
        collision_percentage_list.append(collision_value)
        per_motion_metrics.append(
            {
                "id": prompt_dict.get("id"),
                "prompt": prompt_dict.get("prompt"),
                "value": collision_value,
                "evaluation_file": evaluation_file,
                "motion_duration": prompt_dict.get("motion_duration"),
            }
        )
    
    if skipped > 0:
        print(f"Body_Penetration: Skipped {skipped} samples (missing SMPLify data). Run SMPLify rendering first.")
    
    return {
        "aggregate": summarize_scores(collision_percentage_list),
        "per_motion": per_motion_metrics,
    }

