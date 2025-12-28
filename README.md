# The Quest for Generalizable Motion Generation: Data, Model, and Evaluation

[![arXiv](https://img.shields.io/badge/arXiv-PDF-red?logo=arxiv)](https://arxiv.org/abs/2510.26794)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Paper-orange)](https://huggingface.co/papers/2510.26794)
[![MBench_leaderboard](https://img.shields.io/badge/%F0%9F%A4%97%20_MBench-Leaderboard-ffc107?color=ffc107&logoColor=white)](https://huggingface.co/spaces/wruisi/MBench_leaderboard)
[![ViMoGen-228K](https://img.shields.io/badge/%F0%9F%A4%97%20_ViMoGen228K-Data-ffc107?color=ffc107&logoColor=white)](https://huggingface.co/datasets/wruisi/ViMoGen-228K)

## Overview

![Teaser](https://github.com/oneScotch/ViMoGen/blob/main/assets/teaser.jpg)

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
- [ ] **Training System**: Training code and ViMoGen-228K dataset release.
- [ ] **Evaluation Suite**: Complete MBench evaluation scripts and data.
- [ ] **Motion-to-Motion Pipeline**: Detailed guide and tools for custom reference motion preparation.

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

### 6. Download Body Models (Required for Visualization)

To visualize the generated motions, you need to download the **SMPL-X** model from the [official website](https://smpl-x.is.tue.mpg.de/).

1. Register and download `SMPLX_python_v1.1.zip` (Python v1.1.0).
2. Extract the contents and place the model files (e.g., `SMPLX_NEUTRAL.npz`) in the following directory:
   ```
   data/body_models/
   └── smplx/
       ├── SMPLX_FEMALE.npz
       ├── SMPLX_MALE.npz
       └── SMPLX_NEUTRAL.npz
   ```
*Note: We provide `smplx_root.pt` in `data/body_models/` for coordinate alignment.*

## Pretrained Models

Download pretrained models and place them in the `./checkpoints/` directory:

| Model | Description | Download Link / Command |
|-------|-------------|-------------------------|
| ViMoGen-DiT-1.3B | Main motion generation model | [Google Drive](https://drive.google.com/file/d/1clEE8oM7CkB5cWobsajSz5uJGcj6aYjt/view?usp=sharing) (Save as `./checkpoints/model.pt`) |
| Wan2.1-T2V-1.3B | Base text encoder weights | `huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./checkpoints/Wan2.1-T2V-1.3B` |

## Data & Benchmark

For evaluation on **MBench**, you need to download and extract the benchmark data:

1. Download `mbench.tar.gz` from [Google Drive](https://drive.google.com/file/d/1f2sNwIguCyqSYjxsg2qs6pak-OqohAqi/view?usp=sharing). This package includes:
   - Reference motions generated by Wan 2.1 and processed by CameraHMR.
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
│   └── body_models/           # SMPL-X models and alignment files
├── data_samples/              # Example data for quick start
├── datasets/                   # Dataset loading utilities
├── models/                     # Model definitions
│   └── transformer/           # DiT transformer models
├── scripts/                    # Shell scripts
├── trainer/                    # Training utilities
├── parallel/                   # Distributed training utilities
└── train_eval_vimogen.py       # Main entry point
```

## Inference

### Text-to-Motion (T2M)

Generate motion from text prompts:

1. **Edit prompts**: Modify `data_samples/example_archive.json` with your desired text prompts

2. **Extract text embeddings**:
   ```bash
   bash scripts/text_encoding.sh
   ```

3. **Run inference**:
   ```bash
   bash scripts/t2m_infer.sh
   ```

### Text/Motion-to-Motion (TM2M)

Generate motion conditioned on both text and reference motion:

1. **Prepare reference motion**: 
   - **Option A: Use MBench Benchmark**. We provide pre-processed MBench data with stored reference motions in `./data/mbench/` for immediate evaluation.
   - **Option B: Custom Preparation**. 
     - Generate a reference video using a video generation model (e.g., [Wan 2.1](https://github.com/Wan-Video/Wan2.1))
     - Extract motion from the video using a visual motion capture model (e.g., [CameraHMR](https://github.com/pixelite1201/CameraHMR))
     - *Detailed custom preparation guide coming soon (TBD).*

2. **Run inference**:
   ```bash
   bash scripts/tm2m_infer.sh
   ```

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
