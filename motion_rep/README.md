# 276-Dim Global Motion Representation (DART Style)

This project now uses a single, global, orientation-aligned motion representation with 276 features per frame. The layout follows `collect_motion_rep_DART` in `retarget_motion.py` and is derived from the [DART paper](https://arxiv.org/abs/2410.05260).

## Construction
Given SMPL params (`global_orient`, `body_pose`, `transl`) and global joints (`joints`), canonicalized to a common facing direction, we build velocities and convert rotations to 6D:

1. Body pose (21 non-root joints) in Rot6D.
2. Global joints positions.
3. Joint velocities (forward differences).
4. Root/global orientation in Rot6D.
5. Root orientation velocity (relative rotation between frames) in Rot6D.
6. Root translation.
7. Root translation velocity (forward differences).

Because velocities use frame `t+1`, the motion sequence has one fewer frame than the original SMPL sequence (`motion.shape[0] = smpl_seq_len - 1`).

## Layout (JOINT_NUM = 22)

| Index range | Length | Description |
| ----------- | ------ | ----------- |
| 0:126 | 126 | Body pose Rot6D for 21 non-root joints |
| 126:192 | 66 | Global joints XYZ (22 * 3) |
| 192:258 | 66 | Joint velocities XYZ |
| 258:264 | 6 | Root/global orientation (Rot6D) |
| 264:270 | 6 | Root orientation velocity (Rot6D) |
| 270:273 | 3 | Root translation |
| 273:276 | 3 | Root translation velocity |

## Notes
- All stored motions are **global/canonicalized** (pelvis-centered, facing alignment handled upstream in `canonicalize_motion` + `collect_motion_rep_DART`).

---

## Convert HMR Output to Motion Representation

Use `convert_hmr_to_motion.py` to convert HMR (Human Mesh Recovery) algorithm outputs to our 276-dim motion format.

**HMR** refers to visual motion capture algorithms that recover human mesh from video, such as SMPLest-X, CameraHMR, 4DHumans, etc. Note that the parameters should be in the SMPLX format if accurate overlay with the original video is desired. Otherwise, you may need to modify the `retarget_motion.py` script and replace the smplx_root with smpl(h)_root.

### Input Requirements

The input `.pt` file should contain:

**Required:**

| Key | Shape | Description |
| --- | ----- | ----------- |
| `global_orient` | (T, 3) | Root orientation in axis-angle |
| `body_pose` | (T, 63) | Body pose (21 joints × 3) in axis-angle |
| `transl` | (T, 3) | Root translation |

**Optional** (for reprojection to original video):

| Key | Shape | Description |
| --- | ----- | ----------- |
| `focal_length` | scalar or (1,) | Camera focal length |
| `width` | scalar or (1,) | Image width |
| `height` | scalar or (1,) | Image height |

### Output Format

The output `.pt` file contains:

| Key | Shape | Description |
| --- | ----- | ----------- |
| `motion` | (T-1, 276) | Motion representation (see layout above) |
| `intrinsic` | (3, 3) | Camera intrinsic matrix (identity if not provided) |
| `extrinsic` | (4, 4) | Computed camera extrinsic matrix for reprojection (identity if not provided) |

### Usage

```bash
# Basic usage
python motion_rep/convert_hmr_to_motion.py \
    --input /path/to/hmr_output.pt \
    --output /path/to/motion.pt

# Full options
python motion_rep/convert_hmr_to_motion.py \
    --input /path/to/hmr_output.pt \
    --output /path/to/motion.pt \
    --smplx_model_path ./data/body_models/smplx \
    --device cuda:0
```

### Visualize Converted Motion

Use `motion_checker.py` to visualize the converted motion:

```bash
# Render depth video only
python motion_rep/motion_checker.py \
    --motion_file /path/to/motion.pt \
    --output_dir /path/to/output

# Render overlay on original video
python motion_rep/motion_checker.py \
    --motion_file /path/to/motion.pt \
    --output_dir /path/to/output \
    --video_file /path/to/original_video.mp4

# Full options
python motion_rep/motion_checker.py \
    --motion_file /path/to/motion.pt \
    --output_dir /path/to/output \
    --video_file /path/to/original_video.mp4 \
    --smpl_type smplx \
    --smpl_model_path ./data/body_models/smplx \
    --batch_size 24 \
    --device cuda:0 \
    --verbose
```

---

## Custom Motion Preparation

For preparing custom reference motions from videos (video generation → HMR extraction → motion conversion → quality gating), see **[CUSTOM_MOTION.md](CUSTOM_MOTION.md)**.