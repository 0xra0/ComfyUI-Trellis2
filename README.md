# 🌀 ComfyUI-Trellis2 — Local ROCm Fork (0xra0)

> **Base:** [visualbruno/ComfyUI-Trellis2](https://github.com/visualbruno/ComfyUI-Trellis2)  
> **Branch:** `local-rocm` — AMD ROCm / gfx1030 (RX 6800) patches on top of upstream.

ComfyUI custom node wrapping [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2) (non-GGUF, full-precision path).  
This fork adds ROCm-specific DINO/NAF stability fixes, a `natten` shim for platforms where natten is unavailable, and sparse VAE fixes targeting 16 GB VRAM on AMD gfx1030.

---

## What's Different from Upstream

### Commit: `local-rocm-20261015` — DINO/NAF stability, natten shim, sparse VAE

| File | Change |
|---|---|
| `nodes.py` | ROCm-safe node registrations; MultiView node fixes |
| `trellis2/models/sc_vaes/sparse_unet_vae.py` | Sparse UNet VAE stability on ROCm (dtype/device guards) |
| `trellis2/trainers/flow_matching/mixins/_natten_shim.py` | *(new)* `natten` compatibility shim — provides a pure-PyTorch fallback when the `natten` package is not installed or not available for the current platform |
| `trellis2/trainers/flow_matching/mixins/image_conditioned_proj.py` | DINO/NAF projected conditioning stability; MultiView camera pose routing for Pixal3D |

**Unstaged additions** (also active in the working tree):

- `nodes.py` — additional MultiView node refinements
- `trellis2/trainers/flow_matching/mixins/image_conditioned_proj.py` — further projection conditioning tweaks

---

## Hardware Target

| Component | Spec |
|---|---|
| GPU | AMD Radeon RX 6800 (gfx1030) |
| VRAM | 16 GB |
| ROCm | 6.x / 7.x |
| Python | 3.12 |
| PyTorch | ROCm build |

---

## Installation

### Prerequisites

- ComfyUI installed and running (see [0xra0/ComfyUI](https://github.com/0xra0/ComfyUI) for the matching ROCm-tuned base)
- ROCm 6+ drivers
- Python 3.12 environment used for ComfyUI

### 1. Clone into custom_nodes

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/0xra0/ComfyUI-Trellis2.git
cd ComfyUI-Trellis2
git checkout local-rocm
```

### 2. Install requirements

```bash
pip install -r requirements.txt
```

> [!NOTE]
> `natten` (Neighborhood Attention) is optional. If not available for your platform/ROCm version, the included `_natten_shim.py` provides a pure-PyTorch fallback automatically — no manual configuration needed.

### 3. Download models

**DINOv3** (required):
```bash
git clone https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m \
  /path/to/ComfyUI/models/facebook/dinov3-vitl16-pretrain-lvd1689m
```

**Trellis2 weights** — follow the upstream [visualbruno/ComfyUI-Trellis2](https://github.com/visualbruno/ComfyUI-Trellis2) model instructions.

---

## Usage

This node provides the full-precision (non-GGUF) Trellis2 pipeline. It is compatible with the same workflows as upstream, with the added ROCm stability layer.

### DINO/NAF Stability

The upstream DINO feature extractor and NAF (Noise-Aware Fusion) path can produce NaN or crash on ROCm due to attention backend differences. This fork guards those paths with ROCm-safe alternatives while keeping the same output contract.

### natten Shim

When `natten` is not installed, `_natten_shim.py` transparently replaces Neighborhood Attention calls with an equivalent SDPA-based implementation. This allows Trellis2 to run on ROCm without building the `natten` CUDA/ROCm extension.

### MultiView / Pixal3D

The `image_conditioned_proj.py` changes align the projection conditioning with the camera-pose routing developed in the GGUF fork, so both nodes can be used in the same ComfyUI workflow.

---

## Rebasing onto New Upstream

```bash
git remote add upstream https://github.com/visualbruno/ComfyUI-Trellis2.git
git fetch upstream
git rebase upstream/main
```

---

## Relationship to Upstream

This is a personal hardware-specific fork of [visualbruno/ComfyUI-Trellis2](https://github.com/visualbruno/ComfyUI-Trellis2).  
ROCm-specific issues belong here. Everything else belongs upstream.

---

## License

Same license as upstream — see the upstream repository for the applicable license.
