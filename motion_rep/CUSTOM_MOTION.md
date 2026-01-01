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

> ⚠️ **Important:** By default, SMPLest-X outputs rendered mesh overlays but does not save the raw SMPL-X parameters. You need to modify the inference script to export the required parameters. See **[Modifying SMPLest-X to Export Parameters](#modifying-smplest-x-to-export-parameters)** at the end of this document for step-by-step instructions.

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

[SMPLest-X](https://github.com/SMPLCap/SMPLest-X) by default outputs rendered mesh overlays but does not save the raw SMPL-X parameters. This section provides instructions to modify SMPLest-X to export the required parameters for our motion representation pipeline.

### Understanding the Focal Length

In `main/inference.py` (lines 141-142), the focal length is computed as:

```python
focal = [cfg.model.focal[0] / cfg.model.input_body_shape[1] * bbox[2], 
         cfg.model.focal[1] / cfg.model.input_body_shape[0] * bbox[3]]
```

**Principle**: The model uses a virtual large focal length (default 5000) during training. At inference time, this is mapped back to the original image's pixel space using the detection bounding box size. The computed `focal[0]` (x-axis focal length) is what you need for reprojection.

### Code Modifications

Modify `main/inference.py` in the SMPLest-X repository:

#### Step 1: Initialize Result Container

Before the `for frame in tqdm(range(start, end)):` loop (around line 76), add:

```python
    # ... existing code ...
    detector = YOLO(bbox_model)

    start = int(args.start)
    end = int(args.end) + 1

    # [NEW] Initialize result container
    all_results = {
        'global_orient': [],
        'body_pose': [],
        'transl': [],
        'focal_length': [],
        'width': [],
        'height': []
    }

    for frame in tqdm(range(start, end)):
        # ... existing code ...
```

#### Step 2: Collect Data in the Loop

After `with torch.no_grad():` where `out` is obtained (around line 136), add data collection:

```python
            # ... existing code ...
            # mesh recovery
            with torch.no_grad():
                out = demoer.model(inputs, targets, meta_info, 'test')

            mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]

            # render mesh (existing focal computation)
            focal = [cfg.model.focal[0] / cfg.model.input_body_shape[1] * bbox[2], 
                     cfg.model.focal[1] / cfg.model.input_body_shape[0] * bbox[3]]
            
            # [NEW] Collect current frame data
            # Only collect for bbox_id == 0 (first person) to avoid dimension mismatch
            if bbox_id == 0:
                # global_orient: (1, 3)
                all_results['global_orient'].append(out['smplx_root_pose'].detach().cpu())
                # body_pose: (1, 63)
                all_results['body_pose'].append(out['smplx_body_pose'].detach().cpu())
                # transl: (1, 3)
                all_results['transl'].append(out['cam_trans'].detach().cpu())
                # focal_length: scalar
                all_results['focal_length'].append(torch.tensor([focal[0]], dtype=torch.float32))
                # width, height
                all_results['width'].append(torch.tensor([original_img.shape[1]], dtype=torch.float32))
                all_results['height'].append(torch.tensor([original_img.shape[0]], dtype=torch.float32))

            princpt = [cfg.model.princpt[0] / cfg.model.input_body_shape[1] * bbox[2] + bbox[0], 
                       cfg.model.princpt[1] / cfg.model.input_body_shape[0] * bbox[3] + bbox[1]]
            
            # ... existing rendering code ...
```

#### Step 3: Save Parameters After Loop

After the loop ends (around line 154), save all collected data:

```python
    # ... existing code (loop ends) ...
        # save rendered image
        frame_name = os.path.basename(img_path)
        cv2.imwrite(os.path.join(output_folder, frame_name), vis_img[:, :, ::-1])

    # [NEW] Save all parameters to .pt file after loop
    print(f"Saving parameters to {output_folder}...")
    
    # Concatenate data to form (T, ...) shape
    final_dict = {}
    if len(all_results['global_orient']) > 0:
        for k, v in all_results.items():
            final_dict[k] = torch.cat(v, dim=0)
        
        # Save path
        pt_save_path = os.path.join(output_folder, f'{args.file_name}_params.pt')
        torch.save(final_dict, pt_save_path)
        print(f"Successfully saved parameters to: {pt_save_path}")
    else:
        print("No parameters collected (maybe no person detected).")

if __name__ == "__main__":
    main()
```

### Output Format

The saved `.pt` file will contain:

| Key | Shape | Description |
| --- | ----- | ----------- |
| `global_orient` | (T, 3) | Root orientation (axis-angle) |
| `body_pose` | (T, 63) | Body pose (21 joints × 3, axis-angle) |
| `transl` | (T, 3) | Translation in camera coordinates |
| `focal_length` | (T, 1) | Estimated focal length per frame (pixels) |
| `width` | (T, 1) | Image width |
| `height` | (T, 1) | Image height |

This format is fully compatible with our `convert_hmr_to_motion.py` script.
