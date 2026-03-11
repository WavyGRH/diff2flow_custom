# 🌊 Diff2Flow: Training Flow Matching Models via Diffusion Model Alignment

[![Paper](https://img.shields.io/badge/arXiv-2506.02221-b31b1b.svg)](https://arxiv.org/abs/2506.02221)
[![Conference](https://img.shields.io/badge/CVPR-2025-blue.svg)](https://arxiv.org/abs/2506.02221)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**An independent, modular implementation** of the Diff2Flow framework by Schusterbauer, Gui, Fundel & Ommer (CompVis Group @ LMU Munich).

> **TL;DR** — Diff2Flow bridges Diffusion and Flow Matching (FM) paradigms by rescaling timesteps, aligning interpolants, and deriving FM-compatible velocity fields from diffusion predictions. This enables efficient FM finetuning of pre-trained diffusion priors (e.g., Stable Diffusion) with minimal overhead.

---

## 📝 Overview

| Aspect | Diffusion Models | Flow Matching | Diff2Flow |
|---|---|---|---|
| Timesteps | Discrete `t ∈ [0, T]` | Continuous `t ∈ [0, 1]` | Maps between both |
| Interpolant | `x_t = α_t·x₀ + σ_t·ε` | `x_t = (1-t)·x₀ + t·x₁` | Aligns DM → FM |
| Prediction | ε or v-prediction | Velocity `v` | Derives v from DM output |
| Training | Denoising loss | FM velocity loss | FM loss with DM alignment |

**Key Idea**: Instead of naively training a diffusion model with FM loss (which causes slow convergence and poor LoRA performance), Diff2Flow systematically establishes correspondences between the two paradigms, enabling seamless knowledge transfer.

### Supported Tasks
1. **Text-to-Image Resolution Adaptation** — Finetune SD 2.1 (768→512) with FM objectives
2. **Monocular Depth Estimation** — Image-to-depth via domain adaptation finetuning
3. **Reflow Trajectory Straightening** — Straighten sampling trajectories for few-step inference

---

## 🛠️ Setup

### Prerequisites
- Python ≥ 3.9
- CUDA-capable GPU with ≥ 16 GB VRAM (for training; CPU works for tests)
- Git

### Installation

```bash
# Clone this repository
git clone <your-repo-url> diff2flow
cd diff2flow

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Model Checkpoints

The project uses HuggingFace `diffusers` to automatically download Stable Diffusion 2.1 weights on first run. You'll need:

1. A [HuggingFace account](https://huggingface.co/join)
2. Accept the [Stable Diffusion 2.1 license](https://huggingface.co/stabilityai/stable-diffusion-2-1)
3. Log in via CLI:
   ```bash
   huggingface-cli login
   ```

### Datasets

| Task | Dataset | Source |
|---|---|---|
| Text-to-Image | LAION-Aesthetics | [HuggingFace](https://huggingface.co/datasets/laion/laion2B-en-aesthetic) |
| Depth Estimation | Hypersim + Virtual KITTI v2 | [GitHub](https://github.com/apple/ml-hypersim) / [Website](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/) |
| Evaluation | COCO 2017, NYUv2, KITTI, etc. | See respective dataset pages |

> **Note**: Datasets are only needed for training. Unit tests and small-scale demos work without them.

---

## 🚀 Usage

### Running Tests

```bash
# Run all unit tests (no GPU/data required)
python -m pytest tests/ -v

# Run a specific test module
python -m pytest tests/test_timestep_mapping.py -v
```

### Training

```bash
# Text-to-image with Diff2Flow + LoRA (requires GPU + dataset)
python scripts/train.py \
    --config configs/default.yaml \
    --task text2img \
    --model sd21_diff2flow \
    --lora lora_base

# Monocular depth estimation
python scripts/train.py \
    --config configs/default.yaml \
    --task img2depth \
    --model sd21_diff2flow \
    --data hypersim

# Reflow trajectory straightening
python scripts/train.py \
    --config configs/default.yaml \
    --task reflow \
    --model sd21_diff2flow \
    --lora lora_small
```

### Inference

```bash
# Text-to-image generation
python scripts/inference.py \
    --task text2img \
    --prompt "A beautiful sunset over mountains" \
    --num_steps 25 \
    --output output.png

# Monocular depth estimation
python scripts/inference.py \
    --task img2depth \
    --input input_image.png \
    --num_steps 2 \
    --output depth_map.png
```

### Generate Reflow Pairs

```bash
# Generate noise-image pairs for reflow training
python scripts/generate_reflow_pairs.py \
    --checkpoint stabilityai/stable-diffusion-2-1 \
    --num_samples 10000 \
    --output_dir data/reflow_pairs/
```

---

## 📁 Project Structure

```
diff2flow/
├── diff2flow/                     # Core package
│   ├── schedules.py               # VP/VE noise schedules
│   ├── timestep_mapping.py        # ft: DM timestep ↔ FM timestep
│   ├── interpolant_align.py       # fx: DM interpolant ↔ FM interpolant
│   ├── velocity.py                # Velocity derivation from DM predictions
│   ├── converter.py               # Unified Diff2Flow converter
│   ├── model.py                   # SD UNet wrapper
│   ├── lora.py                    # LoRA adapter
│   ├── sampler.py                 # Euler ODE sampler
│   ├── trainer.py                 # Training loop
│   └── data/                      # Dataset utilities
│       ├── base_dataset.py
│       ├── text_image_dataset.py
│       ├── depth_dataset.py
│       └── reflow_dataset.py
├── configs/                       # YAML configuration files
├── scripts/                       # Training & inference entry points
├── tests/                         # Unit & integration tests
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🧮 Core Algorithm

### 1. Timestep Mapping (`ft`)
Maps diffusion timesteps `t_DM ∈ [0, 1000]` to FM timesteps `t_FM ∈ [0, 1]`:

```
t_FM = α_{t_DM} / (α_{t_DM} + σ_{t_DM})
```

The inverse uses piecewise linear interpolation between discrete neighbors.

### 2. Interpolant Alignment (`fx`)
Transforms DM interpolant to FM linear interpolant:

```
x_FM = x_DM / (α_{t_DM} + σ_{t_DM})
```

### 3. Velocity Derivation
Derives FM velocity from v-prediction:

```
x̂₀ = α_t · x_DM - σ_t · v_pred     (estimated clean image)
x̂_T = σ_t · x_DM + α_t · v_pred     (estimated noise)
v_FM = x̂₀ - x̂_T                      (FM velocity)
```

### 4. Training
Standard FM loss with Diff2Flow adjustments:

```
L = ||v_θ(x_FM, t_DM_bar) - v_target||²
```

where `v_target` is the true FM velocity and inputs are mapped via `ft⁻¹` and `fx⁻¹`.

---

## 📈 Expected Results (from Paper)

| Task | Method | FID ↓ | Notes |
|---|---|---|---|
| Text-to-Image (512²) | SD 2.1 + Diff2Flow | ~12.5 | Converges in ~2.5k iterations |
| Reflow (4 steps) | SD 1.5 + Diff2Flow | ~14.8 | 62M trainable params (LoRA) |
| Depth (NYUv2) | SD 2.1 + Diff2Flow | δ₁=0.986 | 2 sampling steps, trained on synthetic data |

---

## 🎓 Citation

```bibtex
@InProceedings{schusterbauer2024diff2flow,
    title={Diff2Flow: Training Flow Matching Models via Diffusion Model Alignment},
    author={Johannes Schusterbauer and Ming Gui and Frank Fundel and Björn Ommer},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2025}
}
```

---

## 📄 License

This implementation is released under the [MIT License](LICENSE). The original paper and official code are by the CompVis Group @ LMU Munich.

## 🙏 Acknowledgements

- [CompVis/diff2flow](https://github.com/CompVis/diff2flow) — Official implementation
- [HuggingFace Diffusers](https://github.com/huggingface/diffusers) — Stable Diffusion backbone
- [PEFT](https://github.com/huggingface/peft) — Parameter-efficient finetuning
