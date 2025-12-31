from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from smplx import SMPLH, SMPLX
from pytorch3d.renderer import FoVPerspectiveCameras
from pytorch3d.renderer.mesh import rasterize_meshes
from pytorch3d.structures import Meshes

try:
    from .retarget_motion import motion_rep_to_SMPL
    from .rotation_transform import rot6d_to_mat3x3
except ImportError:
    script_dir = Path(__file__).resolve().parent
    parent_dir = script_dir.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from motion_rep.retarget_motion import motion_rep_to_SMPL
    from motion_rep.rotation_transform import rot6d_to_mat3x3


def _default_smpl_model_path(smpl_type: str) -> str:
    """Resolve SMPL/SMPLX family model path with sensible defaults.

    Env overrides:
      - SMPLX_MODEL_PATH for smplx
      - SMPLH_MODEL_PATH for smplh
    Fallback to project-local body_models/{smpl_type}
    """
    if smpl_type == "smplx":
        env_path = os.environ.get("SMPLX_MODEL_PATH")
        fallback = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "body_models", "smplx"))
    elif smpl_type == "smplh":
        env_path = os.environ.get("SMPLH_MODEL_PATH")
        fallback = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "body_models", "smplh"))
    else:
        raise ValueError(f"Unsupported smpl_type: {smpl_type}")
    return env_path or fallback


def get_video_properties(video_path: str) -> Tuple[int, int, int]:
    """Extract width, height, and fps from video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return width, height, fps


def get_R_T(
    joints: torch.Tensor, set_center: bool = True, zero_trans: bool = False, identity_R: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute simple camera rotation/translation from joints."""
    seq_len = joints.shape[0]
    roots = joints[:, 0]
    xyz_move = roots.amax(dim=0) - roots.amin(dim=0)
    y_max = roots.amax(dim=0)[1]
    z_max = roots.amax(dim=0)[2]
    x_move = xyz_move[0]
    y_move = xyz_move[1]
    z_move = xyz_move[2]
    if identity_R:
        R = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
            dtype=torch.float32,
            device=joints.device,
        )
        depth_offset = 2.5 + 1.0 * x_move + 2.0 * y_move + z_max
        T = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=joints.device) * depth_offset
        if set_center:
            T[1] = -roots[0, 1]
    else:
        R = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
            device=joints.device,
        )
        depth_offset = 2.5 + 1.0 * x_move + 2.0 * z_move + y_max
        T = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=joints.device) * depth_offset
        if set_center:
            T[1] = -roots[0, 2]

    if zero_trans:
        T = torch.zeros_like(T)
    R, T = R[None, :].repeat(seq_len, 1, 1), T[None].repeat(seq_len, 1)
    return R, T


def estimate_focal_length(img_w: int, img_h: int, fov: float = 55) -> float:
    """Compute focal length based on FOV."""
    fov_rad = fov * np.pi / 180
    larger_side = max(img_w, img_h)
    focal = larger_side / (2 * np.tan(fov_rad / 2))
    return float(focal)


def visualize_depth_map(depth_map: torch.Tensor) -> np.ndarray:
    """Normalize depth map to uint8 images (white is closer)."""
    B, C, H, W = depth_map.shape
    assert C == 1, "Depth map should have a single channel."
    max_depth = torch.amax(depth_map, dim=[0, 2, 3], keepdim=True)
    mask = depth_map != -1
    depth_map = depth_map.clone()
    depth_map[~mask] = float("inf")
    min_positive_depth = torch.amin(depth_map, dim=[0, 2, 3], keepdim=True)
    normalized_depth_map = (max_depth - depth_map) / (max_depth - min_positive_depth)
    normalized_depth_map *= mask.float()
    normalized_depth_map = (normalized_depth_map * 255).clamp(0, 255).byte()
    return normalized_depth_map.squeeze(1).cpu().numpy()


def pytorch3d_rasterize(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    image_size: Tuple[int, int],
    blur_radius: float = 0.0,
    sigma: float = 1e-8,
    faces_per_pixel: int = 1,
    perspective_correct: bool = False,
    clip_barycentric_coords: bool = True,
    reverse_axis: bool = False,
) -> torch.Tensor:
    """Rasterize to obtain depth buffer."""
    if reverse_axis:
        fixed_vertices = vertices.clone()
        fixed_vertices[..., :2] = -fixed_vertices[..., :2]
    else:
        fixed_vertices = vertices.clone()

    meshes_screen = Meshes(verts=fixed_vertices.float(), faces=faces.long())
    pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
        meshes_screen,
        image_size=image_size,
        blur_radius=blur_radius,
        faces_per_pixel=faces_per_pixel,
        perspective_correct=perspective_correct,
        clip_barycentric_coords=clip_barycentric_coords,
        bin_size=0,
    )
    depth = zbuf.squeeze(-1)
    return depth


def render_depth_maps(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    image_size: Tuple[int, int],
    focal_length: Optional[float] = None,
    R: Optional[torch.Tensor] = None,
    T: Optional[torch.Tensor] = None,
    reverse_axis: bool = False,
) -> torch.Tensor:
    """
    Render depth maps for batched meshes.
    vertices: [batch_size, num_vertices, 3]
    faces: [batch_size, num_faces, 3]
    """
    device = vertices.device
    fov = 2 * np.arctan(min(image_size) / (2 * focal_length))
    camera_kwargs = {"fov": fov, "znear": 0.005, "zfar": 1000, "device": device, "degrees": False}
    if R is not None:
        vertices = torch.matmul(vertices, R)
    if T is not None:
        vertices = vertices + T[:, None]
    cameras = FoVPerspectiveCameras(**camera_kwargs)
    projected_vertices = cameras.transform_points(vertices)
    projected_vertices[..., -1] = vertices[..., -1]
    depth = pytorch3d_rasterize(projected_vertices, faces, image_size=image_size, reverse_axis=reverse_axis)
    depth_maps = depth.unsqueeze(1).detach()
    return depth_maps


def rendering_batches(
    all_verts: torch.Tensor,
    faces: torch.Tensor,
    width: int,
    height: int,
    focal_length: float,
    R: Optional[torch.Tensor] = None,
    T: Optional[torch.Tensor] = None,
    batch_size: int = 8,
    render_multiple: bool = True,
    reverse_axis: bool = False,
) -> torch.Tensor:
    """Render depth maps for all frames in batches."""
    person_num = all_verts.shape[0]
    frame_num = all_verts.shape[1]
    verts_num = all_verts.shape[2]

    if render_multiple:
        multiple_person_faces = []
        vert_offset = 0
        for _ in range(person_num):
            faces_offset = faces + vert_offset
            multiple_person_faces.append(faces_offset)
            vert_offset += verts_num
        multiple_person_faces = torch.cat(multiple_person_faces, dim=0)

    all_depth_maps = []
    for i in range(0, frame_num, batch_size):
        batch_start = i
        batch_end = min(i + batch_size, frame_num)
        batch_verts = all_verts[:, batch_start:batch_end].transpose(1, 0)
        actual_batch_size = batch_end - batch_start
        batch_R = R[batch_start:batch_end] if R is not None else None
        batch_T = T[batch_start:batch_end] if T is not None else None

        if render_multiple:
            verts = batch_verts.reshape(actual_batch_size, person_num * verts_num, 3)
            faces_tensor = multiple_person_faces.unsqueeze(0).repeat(actual_batch_size, 1, 1)
        else:
            verts = batch_verts
            faces_tensor = faces.unsqueeze(0).repeat(actual_batch_size, 1, 1)

        depth_maps = render_depth_maps(
            vertices=verts,
            faces=faces_tensor,
            image_size=(height, width),
            focal_length=focal_length,
            reverse_axis=reverse_axis,
            R=batch_R,
            T=batch_T,
        )
        all_depth_maps.append(depth_maps)

    depth_maps = torch.cat(all_depth_maps, dim=0)
    return depth_maps


def render_and_save_overlay(
    verts: torch.Tensor,
    faces: torch.Tensor,
    R: torch.Tensor,
    T: torch.Tensor,
    rgb_video_path: str,
    width: int,
    height: int,
    focal: float,
    batch_size: int = 24,
    fps: int = 60,
    output_path: Optional[str] = None,
    verbose: bool = False,
) -> str:
    """Render depth and overlay on RGB video."""
    # Render depth maps
    start_time = time.time()
    depth_maps = rendering_batches(
        verts, faces, width, height, focal, R, T, batch_size=batch_size, render_multiple=True, reverse_axis=False
    )
    render_time = time.time() - start_time
    if verbose:
        print(f"Rendering time: {render_time:.2f}s")

    # Visualize depth
    start_time = time.time()
    depth_images = visualize_depth_map(depth_maps)
    vis_time = time.time() - start_time
    if verbose:
        print(f"Visualization time: {vis_time:.2f}s")

    # Read RGB video and overlay
    start_time = time.time()
    cap = cv2.VideoCapture(rgb_video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open RGB video: {rgb_video_path}")

    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Use ffmpeg for better codec compatibility (H.264)
    ffmpeg_command = [
        "ffmpeg",
        "-loglevel", "quiet",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        output_path,
    ]
    ffmpeg_proc = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)

    frame_idx = 0
    total_frames = len(depth_images)
    while cap.isOpened() and frame_idx < total_frames:
        ret, rgb_frame = cap.read()
        if not ret:
            break

        # Resize RGB frame if needed
        if rgb_frame.shape[:2] != (height, width):
            rgb_frame = cv2.resize(rgb_frame, (width, height))

        # Convert depth to 3-channel for overlay
        depth_vis = cv2.cvtColor(depth_images[frame_idx], cv2.COLOR_GRAY2BGR)

        # Overlay: blend depth with RGB (50% alpha)
        overlay = cv2.addWeighted(rgb_frame, 0.5, depth_vis, 0.5, 0)
        ffmpeg_proc.stdin.write(overlay.tobytes())
        frame_idx += 1

    cap.release()
    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()
    save_time = time.time() - start_time
    if verbose:
        print(f"Overlay saving time: {save_time:.2f}s")
        print(f"Total processing time: {render_time + vis_time + save_time:.2f}s")

    return output_path


def render_and_save(
    verts: torch.Tensor,
    faces: torch.Tensor,
    R: torch.Tensor,
    T: torch.Tensor,
    width: int = 1024,
    height: int = 1024,
    focal: float = 2000,
    batch_size: int = 24,
    fps: int = 60,
    output_path: Optional[str] = None,
    motion_name: Optional[str] = None,
    verbose: bool = False,
) -> str:
    """Render depth and save as video."""

    def ffmpeg_command(path: str, pix_fmt: str):
        return [
            "ffmpeg",
            "-loglevel",
            "quiet",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            pix_fmt,
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            path,
        ]

    start_time = time.time()
    depth_maps = rendering_batches(
        verts, faces, width, height, focal, R, T, batch_size=batch_size, render_multiple=True, reverse_axis=False
    )
    render_time = time.time() - start_time
    if verbose:
        print(f"Rendering time: {render_time:.2f}s")

    start_time = time.time()
    depth_images = visualize_depth_map(depth_maps)
    vis_time = time.time() - start_time
    if verbose:
        print(f"Visualization time: {vis_time:.2f}s")

    start_time = time.time()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    depth_ffmpeg = subprocess.Popen(ffmpeg_command(output_path, "gray"), stdin=subprocess.PIPE)
    for depth_image in depth_images:
        depth_ffmpeg.stdin.write(depth_image.tobytes())
    depth_ffmpeg.stdin.close()
    depth_ffmpeg.wait()
    save_time = time.time() - start_time
    if verbose:
        print(f"Video saving time: {save_time:.2f}s")
        print(f"Total processing time: {render_time + vis_time + save_time:.2f}s")

    return output_path


def motion_vis(
    motion_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    batch_size: int = 24,
    H: int = 1024,
    W: int = 1024,
    fps: int = 20,
    motion_name: Optional[str] = None,
    recover_from_velocity: bool = False,
    zero_trans: bool = False,
    device: str = "cuda:0",
    verbose: bool = False,
    do_visulize: bool = True,
    motion_data: Optional[torch.Tensor] = None,
    video_file: Optional[str] = None,
    smpl_model_path: Optional[str] = None,
    smpl_type: str = "smplx",
):
    """
    Visualize motion by rendering a depth video.
    This mirrors depth_render.motion_checker.motion_vis with a narrowed dependency surface.
    """
    # Override H, W, fps from video if provided
    if video_file is not None and os.path.exists(video_file):
        W, H, fps = get_video_properties(video_file)
        if verbose:
            print(f"Using video properties: W={W}, H={H}, fps={fps}")

    if motion_data is None:
        motion_data = torch.load(motion_file, map_location=device, weights_only=True)
    if motion_name is None and motion_file is not None:
        motion_name = os.path.basename(motion_file).split(".")[0]
    elif motion_name is None:
        motion_name = "motion"

    if isinstance(motion_data, dict):
        if verbose:
            print("Reading training data")
        motion = motion_data["motion"]
        smpl_params, joints = motion_rep_to_SMPL(motion, recover_from_velocity)
        extrinsic = motion_data["extrinsic"]
        R, T = extrinsic.split([6, 3], dim=-1)
        R = rot6d_to_mat3x3(R)
        intrinsic = motion_data.get("intrinsic", None)
        if intrinsic is None:
            intrinsic = torch.tensor(
                [
                    [estimate_focal_length(W, H), 0, W / 2],
                    [0, estimate_focal_length(W, H), H / 2],
                    [0, 0, 1],
                ]
            ).float()
    else:
        if verbose:
            print("Reading testing data")
        if motion_data.shape[1] == 276:
            motion = motion_data
            smpl_params, joints = motion_rep_to_SMPL(motion, recover_from_velocity)
            R, T = get_R_T(joints, zero_trans=zero_trans)
            intrinsic = torch.tensor(
                [
                    [estimate_focal_length(W, H), 0, W / 2],
                    [0, estimate_focal_length(W, H), H / 2],
                    [0, 0, 1],
                ]
            ).float()
        else:
            raise ValueError(f"Invalid motion data shape: {motion_data.shape}")

    if not do_visulize:
        return smpl_params, joints

    # Create SMPL model
    smpl_type = smpl_type.lower()
    if smpl_type not in {"smplx", "smplh"}:
        raise ValueError(f"Unsupported smpl_type: {smpl_type}")
    if verbose:
        print(f"Using SMPL type: {smpl_type}")

    model_path = smpl_model_path or _default_smpl_model_path(smpl_type)
    smpl_kwargs = {
        "model_path": model_path,
        "gender": "neutral",
        "num_betas": 10,
        "batch_size": joints.shape[0],
        "use_pca": False,
    }
    smpl_model = SMPLX(**smpl_kwargs).to(device) if smpl_type == "smplx" else SMPLH(**smpl_kwargs).to(device)
    model_output = smpl_model(**smpl_params)
    verts = model_output.vertices
    faces = torch.from_numpy(smpl_model.faces).long().to(verts.device)

    focal = intrinsic[0, 0].item()
    width = int(2 * intrinsic[0, 2].item())
    height = int(2 * intrinsic[1, 2].item())
    if verbose:
        print(f"Focal length: {focal}, Width: {width}, Height: {height}")
        print(f"R[0]: {R[0]}, T[0]: {T[0]}")

    os.makedirs(output_dir, exist_ok=True)
    if recover_from_velocity:
        output_video_path = os.path.join(output_dir, f"{motion_name}_depth_recover_velocity.mp4")
    else:
        output_video_path = os.path.join(output_dir, f"{motion_name}_depth.mp4")

    # Generate overlay video if RGB video provided
    if video_file is not None and os.path.exists(video_file):
        overlay_path = os.path.join(output_dir, f"{motion_name}_overlay.mp4")
        render_and_save_overlay(
            verts=verts[None],
            faces=faces,
            R=R,
            T=T,
            rgb_video_path=video_file,
            width=width,
            height=height,
            focal=focal,
            batch_size=batch_size,
            fps=fps,
            output_path=overlay_path,
            verbose=verbose,
        )
        if verbose:
            print(f"Overlay visualization saved at: {overlay_path}")
    else:
        render_and_save(
            verts=verts[None],
            faces=faces,
            R=R,
            T=T,
            width=width,
            height=height,
            focal=focal,
            batch_size=batch_size,
            fps=fps,
            output_path=output_video_path,
            motion_name=motion_name,
            verbose=verbose,
        )
        if verbose:
            print(f"Visualization saved at: {output_video_path}")

    return smpl_params, joints


def main():
    parser = argparse.ArgumentParser(description="Visualize Depth Maps from Motion File")
    parser.add_argument("--motion_file", type=str, required=True, help="Path to .pt motion file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for visualizations")
    parser.add_argument("--video_file", type=str, default=None, help="Optional RGB video to overlay depth (overrides H/W/fps)")
    parser.add_argument("--smpl_type", type=str, default="smplx", choices=["smplx", "smplh"], help="Body model type")
    parser.add_argument("--smpl_model_path", type=str, default=None, help="Override body model directory")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size for rendering frames")
    parser.add_argument("--H", type=int, default=1080, help="Image height (ignored if video_file provided)")
    parser.add_argument("--W", type=int, default=1920, help="Image width (ignored if video_file provided)")
    parser.add_argument("--fps", type=int, default=20, help="Frames per second (ignored if video_file provided)")
    parser.add_argument("--recover_from_velocity", "-rfv", action="store_true", help="Recover params from velocity")
    parser.add_argument("--zero_trans", "-zt", action="store_true", help="Set translation to zero")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run model on")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose mode")
    args = parser.parse_args()

    device = args.device
    motion_vis(
        args.motion_file,
        args.output_dir,
        args.batch_size,
        args.H,
        args.W,
        args.fps,
        None,
        args.recover_from_velocity,
        device=device,
        verbose=args.verbose,
        zero_trans=args.zero_trans,
        video_file=args.video_file,
        smpl_model_path=args.smpl_model_path,
        smpl_type=args.smpl_type,
    )


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
