import os

import torch

from .rotation_transform import rot6d_to_axis_angle, rot6d_to_mat3x3, axis_angle_to_mat3x3, mat3x3_to_axis_angle, mat3x3_to_rot6d, axis_angle_to_rot6d


JOINT_NUM = 22
# The coordinates of the root joint (i.e., the joints[0]) of the static body models
smplx_root = torch.load("./data/body_models/smplx_root.pt", weights_only=True)


def collect_motion_rep_DART(smpl_params, joints):
    # follow DART: (https://arxiv.org/pdf/2410.05260), final motion: (seq_len, 276)
    seq_len = smpl_params['transl'].shape[0]
    # global_orient & angle velocity of global_orient
    global_orient_aa = smpl_params['global_orient'].view(seq_len, 3)  # (seq_len, 3)
    rot_R = axis_angle_to_mat3x3(global_orient_aa)  # (seq_len, 3, 3)
    rot_vel = torch.matmul(rot_R[1:], rot_R[:-1].transpose(-1, -2))  # (seq_len-1, 3, 3)
    # Use Rot6D as representation
    rot_6D = mat3x3_to_rot6d(rot_R)  # Shape: (seq_len, 6)
    rot_vel_6D = mat3x3_to_rot6d(rot_vel)  # Shape: (seq_len-1, 6)
    
    # translation & velocity of translation
    trans = smpl_params['transl']  # Shape: (seq_len, 3)
    trans_vel = smpl_params['transl'][1:] - smpl_params['transl'][:-1]  # Shape: (seq_len-1, 3)

    # joints & velocity of joints 
    joints = joints.reshape(seq_len, -1)  # Shape: (seq_len, joints_num*3)
    joints_vel = (joints[1:] - joints[:-1])  # Shape: (seq_len-1, joints_num*3)

    # body poses axis-angle -> Rot6D
    body_poses = smpl_params['body_pose'].reshape(-1, 3)  # Shape: (seq_len*(joints_num-1), 3)
    body_poses = axis_angle_to_rot6d(body_poses).reshape(seq_len, -1)  # Shape: (seq_len, (joints_num-1)*6)

    motion = torch.cat([body_poses[:-1], joints[:-1], joints_vel, rot_6D[:-1], rot_vel_6D, trans[:-1], trans_vel], dim=-1)
    return motion

def motion_rep_to_SMPL(motion, recover_from_velocity=False, equal_length=False):
    """
    Convert 276-dim global (aligned) motion representation back to SMPL params + joints.
    The representation layout follows collect_motion_rep_DART:
    [body_pose(126), joints(66), joints_vel(66), root_rot6d(6), root_rot_vel6d(6), transl(3), transl_vel(3)].
    """
    expected_dim = JOINT_NUM * 12 + 12  # 276 for JOINT_NUM=22
    if motion.shape[1] != expected_dim:
        raise ValueError(f"get unexpected motion shape: {motion.shape}, expect {expected_dim} (global DART rep)")

    seq_len = motion.shape[0]
    body_poses = motion[:, :(JOINT_NUM-1)*6]    # (seq_len-1, (joints_num-1)*6)
    body_poses = rot6d_to_axis_angle(body_poses.reshape(-1, 6)).reshape(seq_len, -1)    # (seq_len-1, (joints_num-1)*3)
    joints = motion[:, (JOINT_NUM*6-6):(JOINT_NUM*9-6)].reshape(seq_len, -1, 3)    # (seq_len-1, joints_num, 3) 
    joints_vel = motion[:, (JOINT_NUM*9-6):(JOINT_NUM*12-6)].reshape(seq_len, -1, 3)    # (seq_len-1, joints_num, 3)  
    global_orient = motion[:, (JOINT_NUM*12-6):(JOINT_NUM*12)]    # (seq_len-1, 6)
    global_orient = rot6d_to_axis_angle(global_orient)    # (seq_len-1, 3)
    trans = motion[:, (JOINT_NUM*12+6):(JOINT_NUM*12+9)]    # (seq_len-1, 3)
    trans_vel = motion[:, (JOINT_NUM*12+9):(JOINT_NUM*12+12)]

    smpl_data = {
        'global_orient': global_orient, 
        'body_pose': body_poses,
        'transl': trans
    }

    if recover_from_velocity or equal_length:
        seq_end = seq_len + 1 if equal_length else seq_len
        # recover the global_orient seq from velocity
        R_first = rot6d_to_mat3x3(motion[0:1, (JOINT_NUM*12-6):(JOINT_NUM*12)])  # (1, 6)
        R_vel = rot6d_to_mat3x3(motion[:, (JOINT_NUM*12):(JOINT_NUM*12+6)])  # (seq_len-1, 6)
        # Recover global orientation by cumulative multiplication of velocities
        R_rec = [R_first]
        for i in range(1, seq_end):
            R_curr = torch.matmul(R_vel[i-1:i], R_rec[i-1])  # (1,3,3) x (1,3,3)
            R_rec.append(R_curr)
        R_rec = torch.cat(R_rec, dim=0) # (seq_len, 3, 3)

        # Similarly, recover translations:
        trans_first_frame = trans[0:1]  # (1,3)
        trans_recovered = [trans_first_frame]
        for i in range(1, seq_end):
            trans_recovered.append(trans_recovered[i-1] + trans_vel[i-1:i])
        trans_recovered = torch.cat(trans_recovered, dim=0)  # (seq_len, 3)

        # recover the joints sequence
        joints_recovered = [joints[0:1]]
        for i in range(1, seq_end):
            joints_recovered.append(joints_recovered[i-1] + joints_vel[i-1:i])
        joints_recovered = torch.cat(joints_recovered, dim=0)  # (seq_len, joints_num, 3)
        
        smpl_data['global_orient'] = mat3x3_to_axis_angle(R_rec)  # (seq_len, 3)
        smpl_data['transl'] = trans_recovered
        if equal_length:
            last_frame_body_pose = smpl_data['body_pose'][-1:]
            smpl_data['body_pose'] = torch.cat([smpl_data['body_pose'], last_frame_body_pose], dim=0)
        joints = joints_recovered

    return smpl_data, joints

def get_transform_DART(joints):
    """
    Compute the translation and rotation to align the SMPL-X model output to the canonical coordinate frame.

    Args:
        joints (torch.Tensor): SMPL-X joints of shape (seq_len, joints_num, 3).

    Returns:
        delta_transl (torch.Tensor): Translation vector of shape (3,).
        root_quat_init (torch.Tensor): Rotation quaternion of shape (1, 4).
    """
    # Indices of the relevant joints (ensure these are correct for SMPL-X)
    pelvis_index = 0      # Pelvis (root joint)
    r_hip, l_hip, sdr_r, sdr_l = [2, 1, 17, 16]    # Right hip, Left hip, Right Shoulder, Left Shoulder

    device = joints.device

    # First frame joints positions
    joints_0 = joints[0]  # Shape: (joints_num, 3)

    # Step 1: Compute Rotation Quaternion

    # Compute x_axis (from left hip to right hip, projected onto xy-plane)
    x_axis = (joints_0[r_hip] - joints_0[l_hip])    # Shape: (3,)
    x_axis[2] = 0   # Project to the xy-plane (set z-component to zero)
    x_axis = x_axis / torch.norm(x_axis)  # Normalize

    # z_axis is pointing upwards (inverse gravity direction)
    z_axis = torch.tensor([0, 0, 1], dtype=torch.float32, device=device)

    # Compute y_axis as the cross product of z_axis and x_axis
    y_axis = torch.cross(z_axis, x_axis, dim=-1)
    y_axis = y_axis / torch.norm(y_axis)  # Normalize

    # Build rotation matrix R (from world frame to canonical frame)
    R_inv = torch.stack([x_axis, y_axis, z_axis], dim=1).T  # Shape: (3, 3)

    return R_inv

def apply_rotation(smpl_params, R):
    """
    Update SMPL parameters based on the rotation matrix R.

    Args:
        smpl_params (dict): Dictionary containing SMPL parameters ('global_orient', 'transl', etc.).
        R (torch.Tensor): Rotation matrix of shape (3, 3).

    Returns:
        smpl_params (dict): Updated SMPL parameters.
    """
    N = smpl_params['global_orient'].shape[0]  # Number of frames
    device = smpl_params['global_orient'].device

    # Convert global_orient from axis-angle to 3x3 matrix
    global_orient_mat = axis_angle_to_mat3x3(smpl_params['global_orient'].view(-1, 3))  # Shape: (N, 3, 3)

    # Adjust the global orientation by the computed rotation
    adjusted_global_orient_mat = torch.matmul(R[None,], global_orient_mat)  # Shape: (N, 3, 3)

    # Convert adjusted global_orient back to axis-angle
    smpl_params['global_orient'] = mat3x3_to_axis_angle(adjusted_global_orient_mat)  # Shape: (N, 3)

    # Adjust the translation by rotating
    smpl_params['transl'] += smplx_root.to(device)
    smpl_params['transl'] = torch.matmul(R[None,], smpl_params['transl'][..., None]).squeeze(-1)
    smpl_params['transl'] -= smplx_root.to(device)

    return smpl_params

def canonicalize_motion(smpl_params, joints, set_floor=False):
    # Get transformation and update smpl_params
    R_inv = get_transform_DART(joints)
    aligned_smpl_params = apply_rotation(smpl_params, R_inv)
    joints_base = torch.matmul(R_inv[None, None, :, :], joints.unsqueeze(-1)).squeeze(-1)
    
    delta_transl = -joints_base[0, 0:1]  # fetch the pelvis joint from first frame, Shape: (1,3)
    if set_floor:   
        # For gravity axis (Z), set the minimum z-coordinate to 0 as the floor
        delta_transl[0, 2] = - torch.min(joints_base[..., 2])
    joints = joints_base + delta_transl[None,]
    aligned_smpl_params['transl'] += delta_transl

    # Convert to motion representation
    motion = collect_motion_rep_DART(aligned_smpl_params, joints)   # 276-dim representation

    return motion, joints[:-1], R_inv, delta_transl

def process_hmr_motion(hmr_motion, intrinsic, to_cpu=True, set_floor=False):
    new_data = {}
    device = hmr_motion.device
    seq_len = hmr_motion.shape[0]
    # Step1: hmr -> amass
    R_motionx_to_amass = torch.tensor([[-1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=torch.float32, device=device)
    smpl_params, original_joints = motion_rep_to_SMPL(hmr_motion, equal_length=True)
    joints_amass = torch.matmul(R_motionx_to_amass[None, None, :, :], original_joints.unsqueeze(-1)).squeeze(-1)
    smpl_params_amass = apply_rotation(smpl_params, R_motionx_to_amass)

    # Step2: amass -> dart
    aligned_motion, joints_canonical, R_inv, delta_transl = canonicalize_motion(smpl_params_amass, joints_amass, set_floor=set_floor)
    new_data['motion'] = aligned_motion.detach()
    # rotation
    extrinsic_R = torch.tensor([[1.0, 0.0, 0.0],
                              [0.0, 0.0, -1.0],
                              [0.0, 1.0, 0.0]], dtype=torch.float32, device=device)
    extrinsic_R = mat3x3_to_rot6d(torch.matmul(R_inv, extrinsic_R)[None,]).repeat(seq_len, 1)
    # translation
    extrinsic_T = - torch.matmul(delta_transl, R_inv)[:, [0,2,1]]
    extrinsic_T[0, 2] *= -1
    extrinsic_T = extrinsic_T.repeat(seq_len, 1)
    
    extrinsic = torch.cat([extrinsic_R, extrinsic_T], dim=-1)
    new_data['extrinsic'] = extrinsic.detach()
    new_data['intrinsic'] = intrinsic.detach()
    if to_cpu:
        new_data = {k: v.cpu() for k, v in new_data.items()}
    
    return new_data, joints_canonical