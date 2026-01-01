
# The Quest for Generalizable Motion Generation: Data, Model, and Evaluation

[![arXiv](https://img.shields.io/badge/arXiv-PDF-red?logo=arxiv)](https://arxiv.org/abs/2510.26794)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Paper-orange)](https://huggingface.co/papers/2510.26794)
[![MBench_leaderboard](https://img.shields.io/badge/%F0%9F%A4%97%20_MBench-Leaderboard-ffc107?color=ffc107&logoColor=white)](https://huggingface.co/spaces/wruisi/MBench_leaderboard)
[![ViMoGen-228K](https://img.shields.io/badge/%F0%9F%A4%97%20_ViMoGen228K-Data-ffc107?color=ffc107&logoColor=white)](https://huggingface.co/datasets/wruisi/ViMoGen-228K)

## Overview

![Teaser](https://github.com/SMPLCap/ViMoGen/blob/main/assets/teaser.jpg)

This is the official repository for **[The Quest for Generalizable Motion Generation: Data, Model, and Evaluation](https://huggingface.co/papers/2510.26794)**.

The repo provides a unified framework for **generalizable motion generation**, including both modeling and evaluation:

- **ViMoGen Model**: A **Diffusion Transformer** for generalizable motion generation, supporting  **Text-to-Motion (T2M)** and **Text/Motion-to-Motion (TM2M)**  

- **MBench Benchmark**: A comprehensive evaluation benchmark that decomposes motion generation into **nine dimensions** across three pillars:  **Motion Generalization**, **Motion–Condition Consistency**, and **Motion Quality**.

Together, ViMoGen and MBench enable end-to-end research on scalable and reliable motion generation.

## Introduction

Despite recent advances in 3D human motion generation (MoGen) on standard benchmarks, existing models still face a fundamental bottleneck in their generalization capability. In contrast, adjacent generative fields, most notably video generation (ViGen), have demonstrated remarkable generalization in modeling human behaviors, highlighting transferable insights that MoGen can leverage. 

Motivated by this observation, we present **ViMoGen**, a comprehensive framework that systematically transfers knowledge from ViGen to MoGen across three key pillars: **data**, **modeling**, and **evaluation**. 

*   **[ViMoGen-228K Dataset](https://huggingface.co/datasets/wruisi/ViMoGen-228K)**: A large-scale dataset comprising 228,000 high-quality motion samples that integrates high-fidelity optical MoCap data with semantically annotated motions from web videos and synthesized samples.
*   **ViMoGen Model**: A flow-matching-based diffusion transformer that unifies priors from MoCap data and ViGen models through gated multimodal conditioning.
*   **MBench Benchmark**: A hierarchical benchmark designed for fine-grained evaluation across motion quality, prompt fidelity, and generalization ability.

## News
- [2025-12-19] We have released the ViMoGen-DiT pretrained weights along with the core inference pipeline.
- [2025-12-18] We have released the [ViMoGen-228K Dataset](https://huggingface.co/datasets/wruisi/ViMoGen-228K) and [MBench leaderboard](https://huggingface.co/spaces/wruisi/MBench_leaderboard).
## Release Plan

- [x] **Inference Code**: Core inference pipeline is released.
- [x] **Pretrained Weights**: ViMoGen-DiT weights are available.
- [x] **Training System**: Training code and ViMoGen-228K dataset release.
- [x] **Evaluation Suite**: Complete MBench evaluation scripts and data.
- [x] **Motion-to-Motion Pipeline**: Detailed guide and tools for custom reference motion preparation.

## Installation

### 1. Create Conda Environment

```bash
conda create -n vigen python=3.10 -y
conda activate vigen
```

### 2. Install PyTorch

Install PyTorch with CUDA support. We recommend PyTorch 2.4+ with CUDA 12.1:

```bash
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

Or via pip:
```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Install Flash Attention (Recommended)

For better performance, install Flash Attention 2:

```bash
pip install flash-attn --no-build-isolation
```

### 5. Install PyTorch3D (Optional, for visualization)

PyTorch3D is needed for motion rendering and visualization:

```bash
# Option 1: Install from conda (recommended)
conda install pytorch3d -c pytorch3d

# Option 2: Install from source
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

### 6. Download Body Models (Required for Visualization and Evaluation)

Body models are required for motion visualization and MBench evaluation:

- **SMPL-X**: Used for motion visualization and rendering
- **SMPL**: Used for MBench evaluation metrics (`Pose_Quality`, `Body_Penetration`, and VLM-based metrics)

**Download SMPL-X** from the [official website](https://smpl-x.is.tue.mpg.de/):
1. Register and download `SMPLX_python_v1.1.zip` (Python v1.1.0).
2. Extract and place the model files in:
   ```
   data/body_models/
   └── smplx/
       ├── SMPLX_FEMALE.npz
       ├── SMPLX_MALE.npz
       └── SMPLX_NEUTRAL.npz
   ```

**Download SMPL** from the [official website](https://smpl.is.tue.mpg.de/):
1. Register and download the SMPL model (version 1.1.0 for Python).
2. Extract and place `SMPL_NEUTRAL.pkl` in:
   ```
   data/body_models/
   └── smpl/
       └── SMPL_NEUTRAL.pkl
   ```

*Note: We provide `smplx_root.pt` in `data/body_models/` for coordinate alignment.*

## Pretrained Models

Download pretrained models and place them in the `./checkpoints/` directory:

| Model | Description | Download Link / Command |
|-------|-------------|-------------------------|
| ViMoGen-DiT-1.3B | Main motion generation model | [Google Drive](https://drive.google.com/file/d/10rOvlIwH_vMpHLuvqQTYl7sOMuYyJs_u/view?usp=sharing) (Save as `./checkpoints/model.pt`) |
| Wan2.1-T2V-1.3B | Base text encoder weights and training initialization | `huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./checkpoints/Wan2.1-T2V-1.3B` |

## Data & Benchmark

For evaluation on **MBench**, you need to download and extract the benchmark data:

1. Download `mbench.tar.gz` from [Google Drive](https://drive.google.com/file/d/1TtjR-Mxw_5-xGrTq4qaVNIeRkPQwpw2p/view?usp=sharing). This package includes:
   - Reference motions generated by Wan 2.2 and processed by CameraHMR.
   - T5 text embeddings for all prompts.
2. Extract to the `./data/` directory:
   ```bash
   tar -xzvf mbench.tar.gz -C ./data/
   ```

## Project Structure

```
ViMoGen/
├── checkpoints/                # Model checkpoints
├── configs/                    # Configuration files
│   ├── tm2m_train.yaml        # Training config
│   ├── tm2m_infer.yaml        # TM2M inference config
│   └── t2m_infer.yaml         # T2M inference config
├── data/                       # Data directory
│   ├── mbench/                # MBench benchmark data (Download required)
│   ├── meta_info/             # Metadata for training/testing
│   ├── body_models/           # SMPL-X/SMPL models and alignment files
│   └── ViMoGen-228K/          # Training dataset (Download required)
├── data_samples/              # Example data for quick start
├── datasets/                   # Dataset loading utilities
├── models/                     # Model definitions
│   └── transformer/           # DiT transformer models
├── mbench/                     # MBench evaluation module
├── motion_gating/             # Motion quality gating utilities
├── motion_rep/                # Motion representation conversion tools
├── scripts/                    # Shell scripts
├── trainer/                    # Training utilities
├── parallel/                   # Distributed training utilities
├── evaluate_mbench.py          # MBench evaluation entry point
├── train_eval_vimogen.py       # Main training/inference entry point
└── utils.py                    # Common utilities
```

## Inference

### Text-to-Motion (T2M)

Generate motion from text prompts:

1. **Edit prompts**: Modify `data_samples/example_archive.json` with your desired text prompts (Or use our default prompts).

2. **Extract text embeddings**:
   ```bash
   bash scripts/text_encoding_demo.sh
   ```

3. **Run inference**:
   ```bash
   bash scripts/t2m_infer.sh
   ```

### Text/Motion-to-Motion (TM2M)

Generate motion conditioned on both text and reference motion:

1. **Prepare reference motion**: 
   - **Option A: Use MBench Benchmark**. We provide pre-processed MBench data with stored reference motions in `./data/mbench/` for immediate evaluation.
   - **Option B: Custom Preparation**. See [Custom Motion Preparation](#custom-motion-preparation) below.

2. **Run inference**:
   ```bash
   bash scripts/tm2m_infer.sh
   ```

### Custom Motion Preparation

For preparing custom reference motions for TM2M inference, we provide a complete pipeline covering:

1. **Generate Reference Video** - Using text-to-video models (e.g., Wan 2.2, CogVideoX)
2. **Extract Motion from Video** - Using HMR models (e.g., SMPLest-X, CameraHMR)
3. **Convert to Motion Representation** - Transform to our 276-dim format
4. **Quality Gating** - Determine `use_ref_motion` via VLM analysis and jitter metrics

📖 **See [motion_rep/CUSTOM_MOTION.md](motion_rep/CUSTOM_MOTION.md) for the complete guide.**

## Training

### Data Preparation

1. **Download ViMoGen-228K Dataset from HuggingFace**:
   
   The dataset is hosted on [HuggingFace](https://huggingface.co/datasets/wruisi/ViMoGen-228K) and contains:
   - **ViMoGen-228K.json**: Unified annotation file with all 228K samples
   - **Split annotation files**: `optical_mocap_data.json`, `in_the_wild_video_data.json`, `synthetic_video_data.json`
   - **Motion files**: `.pt` files organized in `motions/` directory
   
   Download using `huggingface-cli`:
   ```bash
   huggingface-cli download wruisi/ViMoGen-228K --repo-type dataset --local-dir ./data/ViMoGen-228K
   ```

   **Data Format**:
   - Motion files (`.pt`) vary by source:
     - **Visual MoCap** (in-the-wild and synthetic videos): Dictionary with
       - `motion`: Tensor of shape `[#frames, 276]` (per-frame motion features)
       - `extrinsic`: Tensor of shape `[#frames, 9]` (camera extrinsics)
       - `intrinsic`: Tensor of shape `[3, 3]` (camera intrinsics)
     - **Optical MoCap**: Direct tensor of shape `[#frames, 276]` (pure motion, no camera info)
   - Each JSON entry contains: `id`, `subset`, `split`, `motion_text_annot`, `video_text_annot`, `motion_path`, and optionally `video_path`

2. **Prepare training data** (add sample IDs and update paths):
   ```bash
   python scripts/prepare_training_data.py \
       --input_json ./data/ViMoGen-228K/ViMoGen-228K.json \
       --motion_root ./data/ViMoGen-228K \
       --output_dir ./data/meta_info \
       --skip_stats
   ```
   
   This script:
   - Adds `sample_id` field to each entry
   - Prefixes `motion_path` with the data root directory
   - Outputs `./data/meta_info/ViMoGen-228K_train.json`
   
   *Note: Use `--skip_stats` to skip mean/std computation since we provide pre-computed statistics in `./data/meta_info/`. Remove this flag if you want to recompute statistics from the full dataset.*

3. **Extract text embeddings** (requires GPU, takes several hours):
   ```bash
   bash scripts/text_encoding_train.sh
   ```
   This will:
   - Extract T5 embeddings for all text prompts
   - Save embeddings to `./data/ViMoGen-228K/text_embeddings/`
   - Update the JSON with embedding paths

### Run Training

Launch distributed training with 8 GPUs:
```bash
bash scripts/tm2m_train.sh
```

## MBench Evaluation

MBench is our hierarchical benchmark for evaluating motion generation across multiple dimensions.

### Evaluation Workflow

#### Step 1: Generate Motions for Evaluation

Run inference on the MBench evaluation set (450 prompts):

```bash
bash scripts/tm2m_infer.sh
```

#### Step 2: Organize Results for Evaluation

Convert generated motions to the format expected by MBench:

```bash
python scripts/organize_mbench_results.py \
    --input_dir exp/tm2m_infer_mbench/test_visualization/mbench_full/step00000001 \
    --output_dir exp/mbench_eval_input
```

This script:
- Collects all `motion_gen_condition_on_text.pt` or `motion_gen_condition_on_motion.pt` files
- Extracts 3D joints and applies coordinate transformation
- Saves results as `.npy` files in the expected format `(frames, 22, 3)`

#### Step 3: Run Evaluation

Run the full MBench evaluation:

```bash
python evaluate_mbench.py \
    --evaluation_path exp/mbench_eval_input \
    --gemini_api_key "YOUR_GEMINI_API_KEY"
```

**Command Options:**
- `--evaluation_path`: Directory containing processed motion files
- `--output_path`: Output directory for results (default: `./evaluation_results/`)
- `--dimension`: Specific dimensions to evaluate (optional, evaluates all by default)
- `--gemini_api_key`: Required for VLM-based metrics

> **Note on Evaluation Time:**
> - **Motion Quality** metrics (`Jitter_Degree`, `Ground_Penetration`, `Foot_Floating`, `Foot_Sliding`, `Dynamic_Degree`) compute directly on 3D joints and are **fast** (seconds to minutes).
> - **Pose Quality** metrics (`Body_Penetration`, `Pose_Quality`) require running **SMPLify** (inverse kinematics from joints to SMPL parameters) and are **moderate** in time.
> - **VLM-based** metrics (`Motion_Condition_Consistency`, `Motion_Generalizability`) require both **SMPLify** and **video rendering**, making them the most **time-consuming** (several hours for 200 samples).
> 
> Although our motion representation can directly export SMPL parameters, we use SMPLify from joints for fair comparison with other skeleton-only methods. To speed up evaluation, use `--dimension` to evaluate specific metric categories separately.

### Evaluation Dimensions

MBench evaluates across three categories:

| Category | Dimension | Description |
|----------|-----------|-------------|
| **Motion Quality** | `Jitter_Degree` | Motion smoothness |
| | `Ground_Penetration` | Feet going through ground |
| | `Foot_Floating` | Feet floating above ground |
| | `Foot_Sliding` | Feet sliding during contact |
| | `Dynamic_Degree` | Motion dynamics/expressiveness |
| **Pose Quality** | `Body_Penetration` | Self-collision detection |
| | `Pose_Quality` | Pose naturalness (NRDF) |
| **VLM-based** | `Motion_Condition_Consistency` | Prompt-motion alignment |
| | `Motion_Generalizability` | Generalization to novel prompts |

### Evaluate Specific Dimensions

To evaluate only specific dimensions:

```bash
# Motion quality only (fast, no rendering needed)
python evaluate_mbench.py \
    --evaluation_path exp/mbench_eval_input \
    --dimension Jitter_Degree Ground_Penetration Foot_Sliding

# VLM-based metrics only (requires Gemini API and video rendering)
python evaluate_mbench.py \
    --evaluation_path exp/mbench_eval_input \
    --dimension Motion_Condition_Consistency Motion_Generalizability \
    --gemini_api_key "YOUR_API_KEY"
```

### Output Format

Results are saved to `evaluation_results/`:
- `{name}_eval_results.json`: Aggregate metrics for each dimension
- `{name}_per_motion_results.json`: Per-sample detailed results
- `{name}_full_info.json`: Evaluation metadata


## Explore More [SMPLCap](https://github.com/SMPLCap) Projects

- [TPAMI'25] [SMPLest-X](https://github.com/SMPLCap/SMPLest-X): An extended version of [SMPLer-X](https://github.com/SMPLCap/SMPLer-X) with stronger foundation models.
- [ICML'25] [ADHMR](https://github.com/SMPLCap/ADHMR): A framework to align diffusion-based human mesh recovery methods via direct preference optimization.
- [ECCV'24] [WHAC](https://github.com/SMPLCap/WHAC): World-grounded human pose and camera estimation from monocular videos.
- [CVPR'24] [AiOS](https://github.com/SMPLCap/AiOS): An all-in-one-stage pipeline combining detection and 3D human reconstruction. 
- [NeurIPS'23] [SMPLer-X](https://github.com/SMPLCap/SMPLer-X): Scaling up EHPS towards a family of generalist foundation models.
- [NeurIPS'23] [RoboSMPLX](https://github.com/SMPLCap/RoboSMPLX): A framework to enhance the robustness of
whole-body pose and shape estimation.
- [ICCV'23] [Zolly](https://github.com/SMPLCap/Zolly): 3D human mesh reconstruction from perspective-distorted images.
- [arXiv'23] [PointHPS](https://github.com/SMPLCap/PointHPS): 3D HPS from point clouds captured in real-world settings.
- [NeurIPS'22] [HMR-Benchmarks](https://github.com/SMPLCap/hmr-benchmarks): A comprehensive benchmark of HPS datasets, backbones, and training strategies.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{lin2025questgeneralizablemotiongeneration,
      title={The Quest for Generalizable Motion Generation: Data, Model, and Evaluation}, 
      author={Jing Lin and Ruisi Wang and Junzhe Lu and Ziqi Huang and Guorui Song and Ailing Zeng and Xian Liu and Chen Wei and Wanqi Yin and Qingping Sun and Zhongang Cai and Lei Yang and Ziwei Liu},
      year={2025},
      journal={arXiv preprint arXiv:2510.26794}, 
}
```
