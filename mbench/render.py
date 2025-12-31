from mbench.third_party.visualize.rotation2xyz import Rotation2xyz
import numpy as np
from trimesh import Trimesh
import os

# os.environ['PYOPENGL_PLATFORM'] = "egl"
os.environ['PYOPENGL_PLATFORM'] = "osmesa"

import torch
from mbench.third_party.visualize.simplify_loc2rot import joints2smpl
import pyrender
import matplotlib.pyplot as plt

import io
import imageio
from shapely import geometry
import trimesh
from pyrender.constants import RenderFlags
import math
import time
# import ffmpeg
from PIL import Image

class WeakPerspectiveCamera(pyrender.Camera):
    def __init__(self,
                 scale,
                 translation,
                 znear=pyrender.camera.DEFAULT_Z_NEAR,
                 zfar=None,
                 name=None):
        super(WeakPerspectiveCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )
        self.scale = scale
        self.translation = translation

    def get_projection_matrix(self, width=None, height=None):
        P = np.eye(4)
        P[0, 0] = self.scale[0]
        P[1, 1] = self.scale[1]
        P[0, 3] = self.translation[0] * self.scale[0]
        P[1, 3] = -self.translation[1] * self.scale[1]
        P[2, 2] = -1
        return P

def render(motions, outdir='test_vis', device_id=0, name=None, render_video=False):
    frames, njoints, nfeats = motions.shape
    MINS = motions.min(axis=0).min(axis=0)
    MAXS = motions.max(axis=0).max(axis=0)

    height_offset = MINS[1]
    motions[:, :, 1] -= height_offset
    trajec = motions[:, 0, [0, 2]]

    j2s = joints2smpl(num_frames=frames, device_id=device_id, cuda=True)
    rot2xyz = Rotation2xyz(device=torch.device(f"cuda:{device_id}"))
    faces = rot2xyz.smpl_model.faces

    pt_output_path = os.path.join(outdir, f'{name}.pt')
    video_output_path = os.path.join(outdir, f'{name}.mp4')
    
    smplify_time = 0
    if not os.path.exists(pt_output_path): 
        print(f'Running SMPLify, it may take a few minutes.')
        smplify_start = time.time()
        
        pred_pose, motion_tensor, opt_dict = j2s.joint2smpl(motions)  # [nframes, njoints, 3]

        vertices = rot2xyz(motion_tensor.clone().detach(), mask=None,
                                        pose_rep='rot6d', translation=True, glob=True,
                                        jointstype='vertices',
                                        vertstrans=True)

        pred_pose_save = pred_pose.reshape(-1, 24, 3).cpu()
        vertices_save = vertices.squeeze(0).permute(2, 0, 1).cpu()
        data_to_save = {
            'pose': pred_pose_save,
            'joints': motions,
            'vertices': vertices_save
        }
        torch.save(data_to_save, pt_output_path)
        
        smplify_time = time.time() - smplify_start
        print(f'SMPLify completed in {smplify_time:.2f} seconds')
    else:
        print(f'Loading existing SMPLify results from {pt_output_path}')
        if render_video:
            vertices = torch.load(pt_output_path)['vertices']
            vertices = vertices.permute(1, 2, 0).unsqueeze(0)
    
    render_time = 0
    if render_video:
        print(f'Starting video rendering for {frames} frames...')
        render_start = time.time()
        
        frames = vertices.shape[3]
        MINS = torch.min(torch.min(vertices[0], axis=0)[0], axis=1)[0]
        MAXS = torch.max(torch.max(vertices[0], axis=0)[0], axis=1)[0]

        # Pre-compute static elements
        minx = MINS[0] - 0.5
        maxx = MAXS[0] + 0.5
        minz = MINS[2] - 0.5 
        maxz = MAXS[2] + 0.5
        polygon = geometry.Polygon([[minx, minz], [minx, maxz], [maxx, maxz], [maxx, minz]])
        polygon_mesh = trimesh.creation.extrude_polygon(polygon, 1e-5)
        polygon_mesh.visual.face_colors = [0, 0, 0, 0.21]

        # Pre-create static scene elements
        base_color = (0.11, 0.53, 0.8, 0.5)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.7,
            alphaMode='OPAQUE',
            baseColorFactor=base_color
        )
        
        polygon_render = pyrender.Mesh.from_trimesh(polygon_mesh, smooth=False)
        bg_color = [1, 1, 1, 0.8]
        camera = pyrender.PerspectiveCamera(yfov=(np.pi / 3.0))
        light = pyrender.DirectionalLight(color=[1,1,1], intensity=300)

        # Pre-compute static poses
        c_ground = np.pi / 2
        ground_pose = np.array([[ 1, 0, 0, 0],
                               [ 0, np.cos(c_ground), -np.sin(c_ground), MINS[1].cpu().numpy()],
                               [ 0, np.sin(c_ground), np.cos(c_ground), 0],
                               [ 0, 0, 0, 1]])

        light_poses = [
            np.array([[1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1], [0, 0, 0, 1]]),
            np.array([[1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1]]),
            np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 2], [0, 0, 0, 1]])
        ]

        c_cam = -np.pi / 6
        camera_pose = np.array([[ 1, 0, 0, (minx+maxx).cpu().numpy()/2],
                               [ 0, np.cos(c_cam), -np.sin(c_cam), 1.5],
                               [ 0, np.sin(c_cam), np.cos(c_cam), max(4, minz.cpu().numpy()+(1.5-MINS[1].cpu().numpy())*2, (maxx-minx).cpu().numpy())],
                               [ 0, 0, 0, 1]])

        # **OPTIMIZATION 1: Single renderer instance**
        r = pyrender.OffscreenRenderer(960, 960)
        
        # **OPTIMIZATION 2: Batch process vertices - keep on GPU longer**
        vertices_np = vertices[0].cpu().numpy()  # Convert once: [3, njoints, nframes]
        
        vid = []
        
        # **OPTIMIZATION 3: Pre-create base scene**
        base_scene = pyrender.Scene(bg_color=bg_color, ambient_light=(0.4, 0.4, 0.4))
        base_scene.add(polygon_render, pose=ground_pose)
        for light_pose in light_poses:
            base_scene.add(light, pose=light_pose)
        base_scene.add(camera, pose=camera_pose)

        # create a placeholder mesh node once
        mesh0 = Trimesh(vertices=vertices_np[:, :, 0], faces=faces)
        mesh_render0 = pyrender.Mesh.from_trimesh(mesh0, material=material)
        mesh_node = base_scene.add(mesh_render0)  # <- keep this node handle
        frame_render_start = time.time()
        for i in range(frames):

            mesh = Trimesh(vertices=vertices[0, :, :, i].squeeze().tolist(), faces=faces)
            mesh_render = pyrender.Mesh.from_trimesh(mesh, material=material)
            
            mesh_node.mesh = mesh_render
            
            color, _ = r.render(base_scene, flags=RenderFlags.RGBA)
            vid.append(color)

        frame_render_time = time.time() - frame_render_start
        print(f'Frame rendering completed in {frame_render_time:.2f} seconds ({frames} frames, {frame_render_time/frames:.3f}s per frame)')

        # **OPTIMIZATION 6: Single renderer cleanup**
        r.delete()

        video_encode_start = time.time()
        out = np.stack(vid, axis=0)
        video_output_path = os.path.join(outdir, f'{name}.mp4')
        
        # **OPTIMIZATION 7: Use faster codec settings**
        imageio.mimsave(video_output_path, out, fps=20, codec='libx264', 
                       ffmpeg_params=['-preset', 'fast', '-crf', '23'])
        
        video_encode_time = time.time() - video_encode_start
        print(f'Video encoding completed in {video_encode_time:.2f} seconds')
        
        render_time = time.time() - render_start
        print(f'Total video rendering time: {render_time:.2f} seconds')
    
    # Print timing summary
    total_time = smplify_time + render_time
    if total_time > 0:
        print(f'=== TIMING SUMMARY for {name} ===')
        print(f'SMPLify processing: {smplify_time:.2f}s ({smplify_time/total_time*100:.1f}%)')
        print(f'Video rendering: {render_time:.2f}s ({render_time/total_time*100:.1f}%)')
        print(f'Total time: {total_time:.2f}s')
        print(f'===========================')

    
    
