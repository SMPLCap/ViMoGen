# Custom Motion Preparation

This document describes the full pipeline for preparing custom reference motions for TM2M (Text/Motion-to-Motion) inference, including video generation, motion extraction, format conversion, and quality gating.

## Overview

The pipeline consists of four main steps:

1. **Generate Reference Video** - Use a text-to-video model to generate human motion videos
2. **Extract Motion from Video** - Use an HMR (Human Mesh Recovery) model to extract SMPL-X parameters
3. **Convert to Motion Representation** - Convert extracted parameters to our 276-dim motion format
4. **Quality Gating** - Determine whether each motion is suitable for reference conditioning

---

## Step 1: Generate Reference Video

Generate a reference video using any text-to-video model that produces human motion videos.

### Using Wan 2.2

[Wan 2.2](https://github.com/Wan-Video/Wan2.2) is a state-of-the-art text-to-video generation model:

```bash
# Single-GPU inference (requires 80GB+ VRAM)
python generate.py --task t2v-A14B --size 1280*720 \
    --ckpt_dir ./Wan2.2-T2V-A14B \
    --offload_model True --convert_model_dtype \
    --prompt "A person performs a jumping jack" \
    --save_file output_video.mp4
```

Other text-to-video models (e.g., CogVideoX, Open-Sora) can also be used.

---

## Step 2: Extract Motion from Video

Extract SMPL-X parameters from the video using a visual motion capture (HMR) model.

### Using SMPLest-X

[SMPLest-X](https://github.com/SMPLCap/SMPLest-X) is a state-of-the-art expressive human pose and shape estimation model:

```bash
# Run inference on video (outputs to ./demo/)
sh scripts/inference.sh smplest_x_h output_video.mp4 30
```

Other HMR models (e.g., CameraHMR, 4DHumans, WHAM) can also be used.

> ⚠️ **Important:** By default, SMPLest-X outputs rendered mesh overlays but does not save the raw SMPL-X parameters. Use the drop-in scripts in `motion_rep/smplest_x_scripts/` (see **[Modifying SMPLest-X to Export Parameters](#modifying-smplest-x-to-export-parameters)**) to export the required parameters.

### Required Output Format

Regardless of which HMR model you use, the output `.pt` file should contain:

**Required:**

| Key | Shape | Description |
| --- | ----- | ----------- |
| `global_orient` | (T, 3) | Root orientation in axis-angle |
| `body_pose` | (T, 63) | Body pose (21 joints × 3) in axis-angle |
| `transl` | (T, 3) | Root translation |

**Optional** (for reprojection to original video):

| Key | Shape | Description |
| --- | ----- | ----------- |
| `focal_length` | scalar or (T, 1) | Camera focal length |
| `width` | scalar or (T, 1) | Image width |
| `height` | scalar or (T, 1) | Image height |

---

## Step 3: Convert to Motion Representation

Convert HMR output to our 276-dim motion format:

```bash
python motion_rep/convert_hmr_to_motion.py \
    --input /path/to/hmr_output.pt \
    --output /path/to/motion.pt
```

The output `.pt` file contains:
- `motion`: Tensor of shape `[T-1, 276]` (per-frame motion features)
- `intrinsic`: Camera intrinsic matrix (3×3)
- `extrinsic`: Camera extrinsic matrix (4×4)

See [motion_rep/README.md](README.md) for the detailed 276-dim layout.

### Visualize Converted Motion

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
```

---

## Step 4: Quality Gating (Determine `use_ref_motion`)

Not all extracted motions are suitable for conditioning. We provide a **motion gating** pipeline to determine whether each motion should be used as a reference (`use_ref_motion=true`) or only as weak guidance (`use_ref_motion=false`).

### Prepare Metadata JSON

Create a JSON file with your motion entries (see `data_samples/example_archive_wi_ref.json` for format):

```json
[
  {
    "id": 0,
    "prompt": "A person performs a jumping jack",
    "motion_path": "path/to/motion.pt",
    ...
  }
]
```

### Render Motions to Video

```bash
# Set PYOPENGL_PLATFORM for offscreen rendering
export PYOPENGL_PLATFORM=osmesa  # or egl

python motion_gating/render_mbench_videos.py \
    --meta-json data_samples/your_archive.json \
    --output-json data_samples/your_archive_eval.json \
    --output-dir data_samples/your_render
```

This renders each motion as an MP4 video and updates the JSON with `video_path` and `mbench_eval_path`.

### Apply Quality Gate

```bash
python motion_gating/apply_quality_gate.py \
    --meta-json data_samples/your_archive_eval.json \
    --quality-report data_samples/your_archive_quality.json \
    --gemini-api-key "YOUR_GEMINI_API_KEY" \
    --jitter-threshold 0.04
```

This script:
1. **Computes Jitter Degree**: Measures motion smoothness (lower is better)
2. **VLM Analysis**: Uses Gemini to check if rendered motion matches the text description
3. **Sets `use_ref_motion`**: 
   - `true` if jitter < threshold AND VLM matches
   - `false` otherwise

### Quality Report Output

```json
[
  {
    "id": 0,
    "jitter_degree": 0.0039,
    "vlm_analysis": "The motion shows a clear jumping jack pattern.",
    "vlm_matches": true,
    "use_ref_motion": true
  }
]
```

---

## Modifying SMPLest-X to Export Parameters

[SMPLest-X](https://github.com/SMPLCap/SMPLest-X) by default outputs rendered mesh overlays but does not save the raw SMPL-X parameters. To avoid manual code edits, we provide ready-to-use scripts under `motion_rep/smplest_x_scripts/` that you can copy into your SMPLest-X checkout.

### Use the Off-the-Shelf Scripts

Assume your SMPLest-X repo is at `$SMPLestX_ROOT`:

```bash
# In ViMoGen repo
SMPLestX_ROOT=/path/to/SMPLest-X

# Optional: backup originals
cp "$SMPLestX_ROOT/main/inference.py" "$SMPLestX_ROOT/main/inference.py.bak"
cp "$SMPLestX_ROOT/scripts/inference.sh" "$SMPLestX_ROOT/scripts/inference.sh.bak"

# Install drop-in scripts
cp motion_rep/smplest_x_scripts/inference.py "$SMPLestX_ROOT/main/inference.py"
cp motion_rep/smplest_x_scripts/inference.sh "$SMPLestX_ROOT/scripts/inference.sh"
chmod +x "$SMPLestX_ROOT/scripts/inference.sh"
```

Then run SMPLest-X inference as usual (in the SMPLest-X repo):

```bash
cd "$SMPLestX_ROOT"
sh scripts/inference.sh smplest_x_h output_video.mp4 30
```

The exported parameters will be saved to:

- `$SMPLestX_ROOT/demo/<video_basename>_params.pt`

### Notes

- `motion_rep/smplest_x_scripts/inference.sh` enables `--retarget_cam` by default, which retargets `transl` so all frames share a fixed camera (focal from the first frame, principal point at image center). The raw per-frame camera values are kept in `*_raw` fields in the exported `.pt`.
- For multi-person videos, only the first detected person (`bbox_id == 0`) is exported to ensure a fixed sequence length.
