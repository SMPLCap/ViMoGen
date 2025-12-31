import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple

import os

# Use osmesa by default for headless rendering
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import imageio
import pyrender
import trimesh
from smplx import SMPLX
from shapely import geometry
from pyrender.constants import RenderFlags

from motion_rep.retarget_motion import motion_rep_to_SMPL

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


def load_motion_tensor(path: Path) -> torch.Tensor:
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


def run_smplx(
    motion_tensor: torch.Tensor,
    smplx_model,
    device: torch.device,
    noise_std: float = 0.0,
) -> dict:
    smpl_data, recovered_joints = motion_rep_to_SMPL(
        motion_tensor,
        recover_from_velocity=True,
        equal_length=False,
    )
    if noise_std > 0.0:
        for key, value in smpl_data.items():
            if torch.is_tensor(value):
                smpl_data[key] = value + torch.randn_like(value) * noise_std
        recovered_joints = recovered_joints + torch.randn_like(recovered_joints) * noise_std
    smpl_params = {k: v.to(device) for k, v in smpl_data.items()}
    recovered_joints = recovered_joints.to(device)
    betas = torch.zeros(
        (motion_tensor.shape[0], smplx_model.num_betas),
        dtype=smpl_params["body_pose"].dtype,
        device=device,
    )
    with torch.no_grad():
        smpl_output = smplx_model(**smpl_params, betas=betas)

    convert_matrix = COORD_CONVERSION.to(device)
    vertices = torch.einsum("ij,tvj->tvi", convert_matrix, smpl_output.vertices)
    joints = torch.einsum("ij,tvj->tvi", convert_matrix, smpl_output.joints[:, :22])
    recovered_joints = torch.einsum("ij,tvj->tvi", convert_matrix, recovered_joints)
    smpl_params["transl"] = torch.einsum("ij,tj->ti", convert_matrix, smpl_params["transl"])

    vertices = vertices.detach().cpu().numpy().astype("float32")
    joints = joints.detach().cpu().numpy().astype("float32")
    recovered_np = recovered_joints.detach().cpu().numpy().astype("float32")
    pose = torch.cat(
        [smpl_params["global_orient"].cpu(), smpl_params["body_pose"].cpu()], dim=1
    )
    pose = pose.view(pose.shape[0], -1, 3).numpy().astype("float32")
    smpl_param_np = {k: v.detach().cpu().numpy().astype("float32") for k, v in smpl_params.items()}

    return {
        "pose": pose,
        "joints": joints,
        "vertices": vertices,
        "smpl_params": smpl_param_np,
        "recovered_joints": recovered_np,
    }


def render_video_from_vertices(
    vertices: np.ndarray,
    joints: Optional[np.ndarray],
    faces: np.ndarray,
    output_path: Path,
    *,
    width: int,
    height: int,
    fps: int,
) -> None:
    verts = vertices.copy()
    if joints is not None:
        offset_x = joints[0, 0, 0]
        offset_z = joints[0, 0, 2]
        verts[:, :, 0] -= offset_x
        verts[:, :, 2] -= offset_z
    mins = verts.reshape(-1, 3).min(axis=0)
    maxs = verts.reshape(-1, 3).max(axis=0)
    verts[:, :, 1] -= (mins[1] + maxs[1]) / 2.0
    mins = verts.reshape(-1, 3).min(axis=0)
    maxs = verts.reshape(-1, 3).max(axis=0)

    minx, miny, minz = mins
    maxx, maxy, maxz = maxs

    polygon = geometry.Polygon(
        [[minx - 0.5, minz - 0.5], [minx - 0.5, maxz + 0.5], [maxx + 0.5, maxz + 0.5], [maxx + 0.5, minz - 0.5]]
    )
    polygon_mesh = trimesh.creation.extrude_polygon(polygon, 1e-5)
    polygon_mesh.visual.face_colors = [0, 0, 0, 0.21]

    base_color = (0.11, 0.53, 0.8, 0.5)
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.7, alphaMode="OPAQUE", baseColorFactor=base_color)

    bg_color = [1, 1, 1, 0.8]
    camera = pyrender.PerspectiveCamera(yfov=(np.pi / 3.0))
    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=300)

    c_ground = np.pi / 2
    ground_pose = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(c_ground), -np.sin(c_ground), mins[1]],
            [0, np.sin(c_ground), np.cos(c_ground), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )

    light_poses = [
        np.array([[1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=np.float32),
        np.array([[1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=np.float32),
        np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 2], [0, 0, 0, 1]], dtype=np.float32),
    ]

    center_x = (minx + maxx) / 2.0
    c_cam = -np.pi / 6
    camera_pose = np.array(
        [
            [1, 0, 0, center_x],
            [0, np.cos(c_cam), -np.sin(c_cam), 1.5],
            [0, np.sin(c_cam), np.cos(c_cam), max(4, minz + (1.5 - mins[1]) * 2, (maxx - minx))],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )

    renderer = pyrender.OffscreenRenderer(width, height)
    scene = pyrender.Scene(bg_color=bg_color, ambient_light=(0.4, 0.4, 0.4))

    polygon_render = pyrender.Mesh.from_trimesh(polygon_mesh, smooth=False)
    scene.add(polygon_render, pose=ground_pose)
    for lp in light_poses:
        scene.add(light, pose=lp)
    scene.add(camera, pose=camera_pose)

    mesh_node = None
    writer = imageio.get_writer(str(output_path), fps=fps, codec="libx264", quality=8)
    try:
        for idx in range(verts.shape[0]):
            mesh = trimesh.Trimesh(vertices=verts[idx], faces=faces, process=False)
            mesh_render = pyrender.Mesh.from_trimesh(mesh, material=material)
            if mesh_node is None:
                mesh_node = scene.add(mesh_render)
            else:
                mesh_node.mesh = mesh_render

            color, _ = renderer.render(scene, flags=RenderFlags.RGBA)
            writer.append_data(color)
    finally:
        writer.close()
        renderer.delete()


def convert_and_render(
    motion_path: Path,
    motion_id: str,
    output_dir: Path,
    smplx_model_dir: Path,
    *,
    device: str,
    fps: int,
    width: int,
    height: int,
    noise_std: float = 0.0,
) -> Tuple[Path, Path]:
    pt_path = output_dir / f"{motion_id}.pt"
    mp4_path = output_dir / f"{motion_id}.mp4"
    if pt_path.exists() and mp4_path.exists():
        return pt_path, mp4_path

    motion_tensor = load_motion_tensor(motion_path)

    smplx_model = SMPLX(
        model_path=str(smplx_model_dir),
        gender="neutral",
        use_pca=False,
        batch_size=motion_tensor.shape[0],
    ).to(device)
    smplx_model.eval()
    payload = run_smplx(motion_tensor, smplx_model, torch.device(device), noise_std=noise_std)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, pt_path)

    render_video_from_vertices(
        payload["vertices"],
        payload["joints"],
        getattr(smplx_model, "faces").detach().cpu().numpy()
        if torch.is_tensor(getattr(smplx_model, "faces"))
        else np.asarray(getattr(smplx_model, "faces")),
        mp4_path,
        width=width,
        height=height,
        fps=fps,
    )
    return pt_path, mp4_path
