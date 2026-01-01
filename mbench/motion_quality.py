import torch
import numpy as np
from tqdm import tqdm
from mbench.utils import load_dimension_info
from mbench.third_party.visualize.rotation_conversions import axis_angle_to_rotation_6d
from scipy.signal import find_peaks

FOOT_IDX = [10, 11]


def load_joints(evaluation_file: str, device: str = 'cuda') -> torch.Tensor:
    """
    Load joints from either .npy or .pt file format.
    
    Args:
        evaluation_file: Path to the evaluation file (.npy or .pt)
        device: Target device for the tensor
        
    Returns:
        Tensor of shape (T, num_joints, 3)
    """
    if evaluation_file.endswith('.npy'):
        # Direct numpy array of joints (T, 22, 3)
        joints = np.load(evaluation_file)
        return torch.from_numpy(joints).float().to(device)
    else:
        # PyTorch dict with 'joints' key
        data = torch.load(evaluation_file, map_location=device)
        if isinstance(data, dict) and 'joints' in data:
            joints = data['joints']
            if isinstance(joints, np.ndarray):
                return torch.from_numpy(joints).float().to(device)
            return joints.float().to(device)
        else:
            raise ValueError(f"Unexpected data format in {evaluation_file}")

def find_common_intervals(intervals1, intervals2):
    """Find common intervals between two sets of intervals."""
    intervals1 = torch.tensor(intervals1, dtype=torch.float32)
    intervals2 = torch.tensor(intervals2, dtype=torch.float32)
    start_max = torch.max(intervals1[:, 0].unsqueeze(1), intervals2[:, 0])
    end_min = torch.min(intervals1[:, 1].unsqueeze(1), intervals2[:, 1])
    overlap_mask = start_max < end_min
    common_intervals = torch.stack((start_max, end_min), dim=2)[overlap_mask]
    return common_intervals

def calculate_angle(vector1, vector2):
    """Calculate angle between two vectors."""
    vector1 = vector1 / (torch.norm(vector1, dim=-1).unsqueeze(-1) + 1e-6)
    vector2 = vector2 / (torch.norm(vector2, dim=-1).unsqueeze(-1) + 1e-6)
    dot_product = torch.sum(vector1 * vector2, dim=-1)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    angle = torch.acos(dot_product)
    return angle

def get_contact(joints: torch.Tensor, foot_idx: list = FOOT_IDX, vel_ts: float = 0.01, height_ts: float = 0.02, device: str = 'cuda') -> torch.Tensor:
    """Detect contact from foot velocities and height."""
    foot_pos = joints[:, foot_idx]  # (frames, 2, 3)
    # Calculate foot velocity
    foot_vel = foot_pos[1:] - foot_pos[:-1]
    foot_vel = torch.cat([foot_vel, foot_vel[-1:]], dim=0)

    vel_ts = torch.tensor(vel_ts, dtype=torch.float32).to(device)
    height_ts = torch.tensor(height_ts, dtype=torch.float32).to(device)

    # Calculate the velocity magnitudes (norm)
    delta = torch.norm(foot_vel, dim=-1)
    # Detect foot contact based on velocity and height threshold
    contact = (delta < vel_ts) | (foot_pos[:, :, 2] < height_ts).to(device)
    return contact.int()

def get_range(contact: torch.Tensor, contact_state: bool = True) -> list:
    """Get contact range from contact info.

    Args:
        contact (torch.Tensor): contact information (frames, 2).
        contact_state (bool, optional): contact state. Defaults to True.

    Returns:
        list: contact ranges.
    """
    contact_state = int(contact_state)
    frames = contact.shape[0]
    # Get contact range
    contact_range = []
    for i in range(contact.shape[1]):
        rge = []
        start = -1
        end = -1
        for idx in range(frames):
            if contact[idx, i] != contact_state:
                continue
            if start == -1:
                start = idx
                end = idx
            else:
                if idx - end == 1:
                    end += 1
                else:
                    rge.append([start, end])
                    start = idx
                    end = idx
        if end != -1:
            rge.append([start, end])
        contact_range.append(rge)
    return contact_range

def remove_global_translation(joints):
    """Remove global translation by subtracting the root joint's position."""
    root_joint = joints[:, 0:1, :]  # Root joint is joint 0 (shape: T, 1, 3)
    local_joints = joints - root_joint  # Subtract root joint position
    return local_joints


def summarize_scores(values):
    """Return mean/std/count summary for a list of scalar values."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "num_samples": 0}
    tensor = torch.tensor(values, dtype=torch.float32)
    std = tensor.std(unbiased=False).item() if tensor.numel() > 1 else 0.0
    return {
        "mean": tensor.mean().item(),
        "std": std,
        "num_samples": tensor.numel(),
    }

def compute_jitter_degree(full_info_path: str, device: str, **kwargs):
    """
    Compute the jitter degree of the motion based on the acceleration of the joints.
    """
    prompt_dict_ls = load_dimension_info(full_info_path, dimension='Jitter_Degree')
    
    jitter_degree_list = []
    per_motion_metrics = []
    
    for prompt_dict in tqdm(prompt_dict_ls):
        evaluation_file = prompt_dict["evaluation_file"]
        pred_joints = load_joints(evaluation_file, device)

        # Global jitter degree
        velocity = pred_joints[1:] - pred_joints[:-1]  # Shape: (T-1, 24, 3)
        acceleration = velocity[1:] - velocity[:-1]  # Shape: (T-2, 24, 3)
        acceleration_magnitude = torch.norm(acceleration, dim=2)  # Shape: (T-2, 24)
        global_jitter = acceleration_magnitude.mean()
        
        # Local jitter degree (remove global translation)
        local_joints = remove_global_translation(pred_joints)
        local_velocity = local_joints[1:] - local_joints[:-1]  # Shape: (T-1, 24, 3)
        local_acceleration = local_velocity[1:] - local_velocity[:-1]  # Shape: (T-2, 24, 3)
        local_acceleration_magnitude = torch.norm(local_acceleration, dim=2)  # Shape: (T-2, 24)
        local_jitter = local_acceleration_magnitude.mean()
        
        # Combined jitter degree
        combined_jitter = global_jitter + local_jitter
        combined_value = combined_jitter.item()
        jitter_degree_list.append(combined_value)
        per_motion_metrics.append(
            {
                "id": prompt_dict.get("id"),
                "prompt": prompt_dict.get("prompt"),
                "value": combined_value,
                "evaluation_file": evaluation_file,
                "motion_duration": prompt_dict.get("motion_duration"),
            }
        )
    
    return {
        "aggregate": summarize_scores(jitter_degree_list),
        "per_motion": per_motion_metrics,
    }

def compute_ground_penetration(full_info_path: str, device: str, **kwargs):
    """
    Compute foot-floor penetration based on the foot joints.
    """
    prompt_dict_ls = load_dimension_info(full_info_path, dimension='Ground_Penetration')
    
    penetration_list = []
    per_motion_metrics = []
    
    for prompt_dict in tqdm(prompt_dict_ls):
        evaluation_file = prompt_dict["evaluation_file"]
        pred_joints = load_joints(evaluation_file, device)
        
        delta_ts = 0.005  # 5mm tolerance
        floor_height = 0.0
        
        foot_pos = pred_joints[:, FOOT_IDX]  # (frames, 2, 3)
        foot_ground_height = foot_pos[:, :, 2] - floor_height
        
        # Compute penetration distance (below the ground)
        penetration_dist = torch.abs(foot_ground_height[foot_ground_height < -delta_ts])
        penetration_score = penetration_dist.mean() if penetration_dist.numel() > 0 else torch.tensor(0.0)
        penetration_value = penetration_score.item()
        
        penetration_list.append(penetration_value)
        per_motion_metrics.append(
            {
                "id": prompt_dict.get("id"),
                "prompt": prompt_dict.get("prompt"),
                "value": penetration_value,
                "evaluation_file": evaluation_file,
                "motion_duration": prompt_dict.get("motion_duration"),
            }
        )
    
    return {
        "aggregate": summarize_scores(penetration_list),
        "per_motion": per_motion_metrics,
    }

def compute_foot_floating(full_info_path: str, device: str, **kwargs):
    """
    Check for foot floating in the motion data.
    """
    prompt_dict_ls = load_dimension_info(full_info_path, dimension='Foot_Floating')
    
    floating_list = []
    per_motion_metrics = []
    
    for prompt_dict in tqdm(prompt_dict_ls):
        evaluation_file = prompt_dict["evaluation_file"]
        pred_joints = load_joints(evaluation_file, device)

        
        contact = get_contact(pred_joints, device=device)
        frames = pred_joints.shape[0]
        
        delta_ts = 0.001
        rate_ts = 0.6
        rate_high_ts = 1.75
        
        # Root position and velocity
        root_pos = pred_joints[:, 0]
        root_vel = root_pos[1:] - root_pos[:-1]
        root_vel = torch.cat([root_vel, root_vel[-1:]], dim=0)

        # Foot positions and velocities
        foot_pos = pred_joints[:, FOOT_IDX]  # (frames, 2, 3)
        foot_vel = foot_pos[1:] - foot_pos[:-1]
        foot_vel = torch.cat([foot_vel, foot_vel[-1:]], dim=0)

        # Relative foot positions and velocities
        rel_foot_pos = foot_pos - root_pos.unsqueeze(1)
        rel_foot_vel = rel_foot_pos[1:] - rel_foot_pos[:-1]
        rel_foot_vel = torch.cat([rel_foot_vel, rel_foot_vel[-1:]], dim=0)

        # Check frame floating
        left_foot_fl_rate = torch.zeros((frames, 1)).to(device)
        right_foot_fl_rate = torch.zeros((frames, 1)).to(device)
        invalid_flag = torch.ones((frames, 2)).to(device)
        
        for f in range(frames):
            root_dis = torch.norm(root_vel[f], p=2, dim=-1)
            left_parent_dis = torch.norm(rel_foot_vel[f, 0], p=2, dim=-1)
            right_parent_dis = torch.norm(rel_foot_vel[f, 1], p=2, dim=-1)
            rate_left = left_parent_dis / (root_dis + 1e-6)
            rate_right = right_parent_dis / (root_dis + 1e-6)

            left_foot_fl_rate[f] = rate_left
            right_foot_fl_rate[f] = rate_right

            left_foot_dis = torch.norm(foot_vel[f, 0], p=2, dim=-1)
            right_foot_dis = torch.norm(foot_vel[f, 1], p=2, dim=-1)

            if root_dis < delta_ts:
                continue

            lf_l_invalid = rate_left < rate_ts and left_foot_dis > 1.2e-4
            lf_h_invalid = rate_left > rate_high_ts and left_foot_dis > 1.2e-4
            lf_invalid = lf_l_invalid or (lf_h_invalid and root_dis > 1.2e-4)

            rf_l_invalid = rate_right < rate_ts and right_foot_dis > 1.2e-4
            rf_h_invalid = rate_right > rate_high_ts and right_foot_dis > 1.2e-4
            rf_invalid = rf_l_invalid or (rf_h_invalid and root_dis > 1.2e-4)

            if torch.sum(contact[f]) == 2 and lf_invalid and rf_invalid:
                invalid_flag[f, 0] = 0
                invalid_flag[f, 1] = 0
            elif contact[f, 0] == 1 and contact[f, 1] == 0 and lf_invalid:
                invalid_flag[f, 0] = 0
            elif contact[f, 1] == 1 and contact[f, 0] == 0 and rf_invalid:
                invalid_flag[f, 1] = 0

        # Get not contact range
        all_rates = torch.cat([left_foot_fl_rate, right_foot_fl_rate], dim=-1)
        no_contact_range = get_range(contact, False)

        # Check sequence floating
        floating_range_lens = [0]
        for i in range(len(no_contact_range)):
            for j, rge in enumerate(no_contact_range[i]):
                rates = all_rates[rge[0]:rge[1]+1, i]
                if len(rates) < 4:
                    continue

                skip_n = 0
                for f in range(rge[0], rge[1]+1):
                    if torch.norm(root_vel[f], p=2, dim=-1) < delta_ts:
                        skip_n += 1

                if skip_n / (rge[1] - rge[0] + 1) > 0.5:
                    continue

                cur_invalid_flag = rates < (rate_ts - 0.2)
                diff = torch.diff(torch.cat(
                    [torch.tensor([0]).to(device),
                     cur_invalid_flag.float(),
                     torch.tensor([0]).to(device)]))
                start_indices = torch.where(diff == 1)[0]
                end_indices = torch.where(diff == -1)[0]

                if len(start_indices) != 0:
                    lengths = end_indices - start_indices
                    floating_range_lens.extend(lengths.tolist())

        # Check mass floating
        mass_floating_len_list = []
        if len(no_contact_range[0]) != 0 and len(no_contact_range[1]) != 0:
            merged_no_contact_range = find_common_intervals(
                no_contact_range[0], no_contact_range[1])
            for i, rge in enumerate(merged_no_contact_range):
                start, end = rge
                start, end = int(start), int(end)
                if (end - start + 1) < 4:
                    continue
                l_start_end_vec = foot_pos[end, 0] - foot_pos[start, 0]
                agl_list = []
                for f in range(start + 1, end + 1):
                    l_start_cur_vec = foot_pos[f, 0] - foot_pos[start, 0]
                    vec_angle = torch.rad2deg(torch.abs(
                        calculate_angle(l_start_cur_vec, l_start_end_vec)))
                    agl_list.append(vec_angle.detach().cpu().numpy())
                peaks, _ = find_peaks(agl_list)
                if len(peaks) > 2:
                    mass_floating_len_list.append(end - start + 1)

        # Check valid
        merge_invalid_flag = invalid_flag[:, 0] + invalid_flag[:, 1]
        merge_invalid_flag = merge_invalid_flag <= 1
        invalid_n = len(merge_invalid_flag[merge_invalid_flag])

        invalid_n += sum(floating_range_lens) / 2
        invalid_n += sum(mass_floating_len_list)
        floating_score = invalid_n / frames
        floating_value = float(floating_score)
        floating_list.append(floating_value)
        per_motion_metrics.append(
            {
                "id": prompt_dict.get("id"),
                "prompt": prompt_dict.get("prompt"),
                "value": floating_value,
                "evaluation_file": evaluation_file,
                "motion_duration": prompt_dict.get("motion_duration"),
            }
        )
    return {
        "aggregate": summarize_scores(floating_list),
        "per_motion": per_motion_metrics,
    }

def compute_foot_sliding(full_info_path: str, device: str, **kwargs):
    """
    Check for foot sliding in the motion data.
    """
    prompt_dict_ls = load_dimension_info(full_info_path, dimension='Foot_Sliding')
    
    sliding_list = []
    per_motion_metrics = []
    
    for prompt_dict in tqdm(prompt_dict_ls):
        evaluation_file = prompt_dict["evaluation_file"]
        pred_joints = load_joints(evaluation_file, device)

        
        contact = get_contact(pred_joints, device=device)
        
        foot_pos = pred_joints[:, FOOT_IDX]   # (frames, 2, 3)
        # Compute foot velocity and delta joints
        foot_vel = foot_pos[1:] - foot_pos[:-1]
        foot_vel = torch.cat([foot_vel, foot_vel[-1:]], dim=0)
        foot_delta = torch.norm(foot_vel[:, :, :2], dim=-1)
        
        # Calculate sliding distance for left and right foot
        left_sliding_dis = (foot_delta[:, 0] * contact[:, 0]).sum(dim=0) / ((contact[:, 0]).sum(dim=0) + 1e-6)
        right_sliding_dis = (foot_delta[:, 1] * contact[:, 1]).sum(dim=0) / ((contact[:, 1]).sum(dim=0) + 1e-6)

        sliding_score = (left_sliding_dis + right_sliding_dis) / 2
        sliding_value = sliding_score.item()
        sliding_list.append(sliding_value)
        per_motion_metrics.append(
            {
                "id": prompt_dict.get("id"),
                "prompt": prompt_dict.get("prompt"),
                "value": sliding_value,
                "evaluation_file": evaluation_file,
                "motion_duration": prompt_dict.get("motion_duration"),
            }
        )
    
    return {
        "aggregate": summarize_scores(sliding_list),
        "per_motion": per_motion_metrics,
    }

def compute_dynamic_degree(full_info_path: str, device: str, **kwargs):
    """
    Compute the dynamic degree of the motion based on the average velocity of the joints.
    """
    prompt_dict_ls = load_dimension_info(full_info_path, dimension='Dynamic_Degree')
    
    dynamic_degree_list = []
    per_motion_metrics = []
    
    for prompt_dict in tqdm(prompt_dict_ls):
        evaluation_file = prompt_dict["evaluation_file"]
        pred_joints = load_joints(evaluation_file, device)

        
        # Global dynamic degree
        velocity = torch.norm(pred_joints[1:] - pred_joints[:-1], dim=2)  # Shape: (T-1, 24)
        global_dynamic = velocity.mean()
        
        # Local dynamic degree (remove global translation)
        local_joints = remove_global_translation(pred_joints)
        local_velocity = torch.norm(local_joints[1:] - local_joints[:-1], dim=2)  # Shape: (T-1, 24)
        local_dynamic = local_velocity.mean()
        
        # Combined dynamic degree
        combined_dynamic = global_dynamic + local_dynamic
        dynamic_value = combined_dynamic.item()
        dynamic_degree_list.append(dynamic_value)
        per_motion_metrics.append(
            {
                "id": prompt_dict.get("id"),
                "prompt": prompt_dict.get("prompt"),
                "value": dynamic_value,
                "evaluation_file": evaluation_file,
                "motion_duration": prompt_dict.get("motion_duration"),
            }
        )
    
    return {
        "aggregate": summarize_scores(dynamic_degree_list),
        "per_motion": per_motion_metrics,
    }
