import torch
import torchgeometry as tgm
from torch.nn import functional as F

def mat3x3_to_rot6d(R):
    # rot6d takes the first two columns of R
    return R[..., :, :2].reshape(R.shape[0], 6)

def rot6d_to_mat3x3(rot6d):
    """
    Convert 6d rotation representation to 3x3 rotation matrix.
    Shape:
        - Input: :Torch:`(N, 6)`
        - Output: :Torch:`(N, 3, 3)`
    """
    rot6d = rot6d.view(-1, 3, 2)
    a1 = rot6d[:, :, 0]
    a2 = rot6d[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2, dim=-1)
    rot_mat = torch.stack((b1, b2, b3), dim=-1)  # 3x3 rotation matrix
    return rot_mat
    
def axis_angle_to_rot6d(angle_axis):
    """Convert 3d vector of axis-angle rotation to 6d rotation representation.
    Shape:
        - Input: :Torch:`(N, 3)`
        - Output: :Torch:`(N, 6)`
    """
    rot_mat = tgm.angle_axis_to_rotation_matrix(angle_axis)  # 4x4 rotation matrix
    rot6d = rot_mat[:, :3, :2]
    rot6d = rot6d.reshape(-1, 6)

    return rot6d

def rot6d_to_axis_angle(rot6d):
    """Convert 6d rotation representation to 3d vector of axis-angle rotation.
    Shape:
        - Input: :Torch:`(N, 6)`
        - Output: :Torch:`(N, 3)`
    """
    batch_size = rot6d.shape[0]

    rot6d = rot6d.view(batch_size, 3, 2)
    a1 = rot6d[:, :, 0]
    a2 = rot6d[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2, dim=-1)
    rot_mat = torch.stack((b1, b2, b3), dim=-1)  # 3x3 rotation matrix

    rot_mat = torch.cat([rot_mat, torch.zeros((batch_size, 3, 1), device=rot_mat.device).float()],
                        2)  # 3x4 rotation matrix
    axis_angle = tgm.rotation_matrix_to_angle_axis(rot_mat).reshape(-1, 3)  # axis-angle
    axis_angle[torch.isnan(axis_angle)] = 0.0
    return axis_angle

def axis_angle_to_mat3x3(angle_axis):
    """
    Convert 3d vector of axis-angle rotation to 3x3 rotation matrix.
    Shape:
        - Input: :Torch:`(N, 3)`
        - Output: :Torch:`(N, 3, 3)`
    """
    rot_mat = tgm.angle_axis_to_rotation_matrix(angle_axis)  # 4x4 rotation matrix

    return rot_mat[:, :3, :3]

def mat3x3_to_axis_angle(rot_mat):
    """
    Convert 3x3 rotation matrix to 3d vector of axis-angle rotation.
    Shape:
        - Input: :Torch:`(N, 3, 3)`
        - Output: :Torch:`(N, 3)`
    """
    rot_mat = torch.cat([rot_mat, torch.zeros((rot_mat.shape[0], 3, 1), device=rot_mat.device).float()], 2)
    axis_angle = tgm.rotation_matrix_to_angle_axis(rot_mat).reshape(-1, 3)  # axis-angle
    axis_angle[torch.isnan(axis_angle)] = 0.0
    return axis_angle

def quaternion_to_axis_angle(quaternion):
    """
    Convert 4d quaternion to 3d vector of axis-angle rotation.
    Shape:
        - Input: :Torch:`(..., 4)`
        - Output: :Torch:`(..., 3)`
    """
    angle_axis = tgm.quaternion_to_angle_axis(quaternion)
    return angle_axis

def axis_angle_to_quaternion(angle_axis):
    """
    Convert 3d vector of axis-angle rotation to 4d quaternion.
    Shape:
        - Input: :Torch:`(..., 3)`
        - Output: :Torch:`(..., 4)`
    """
    quaternion = tgm.angle_axis_to_quaternion(angle_axis)
    return quaternion

def quaternion_to_rot6d(quaternion):
    """
    Convert 4d quaternion to 6d rotation representation.
    Shape:
        - Input: :Torch:`(N, 4)`
        - Output: :Torch:`(N, 6)`
    """

    return axis_angle_to_rot6d(quaternion_to_axis_angle(quaternion))

def rot6d_to_quaternion(rot6d):
    """
    Convert 6d rotation representation to 4d quaternion.
    Shape:
        - Input: :Torch:`(N, 4)`
        - Output: :Torch:`(N, 6)`
    """

    return axis_angle_to_quaternion(rot6d_to_axis_angle(rot6d))