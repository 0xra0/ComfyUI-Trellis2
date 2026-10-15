"""
View-Aligned (Projection) Image Conditioned Mixin for TRELLIS2

This module implements DINOv3-based feature extraction with view-aligned projection,
supporting camera-aware 3D-to-2D feature mapping.
"""

from typing import *
import os
import torch
# AMD ROCm fallback: Upsample lr_features to target size (replaces NAF/natten).
# Optional experimental high-freq image cue: set env TRELLIS2_NAF_SHARPEN=1 to enable.
class _NAFBf16Wrapper(torch.nn.Module):
    # Runs the real NAF in bf16 to halve activation memory (fixes the
    # texture-stage full-image NAF OOM on 16GB). Output is cast back to fp32.
    def __init__(self, naf):
        super().__init__()
        self.naf = naf.to(torch.bfloat16)
    def forward(self, *args, **kwargs):
        torch.cuda.empty_cache()
        args = tuple(a.to(torch.bfloat16) if torch.is_tensor(a) and a.is_floating_point() else a for a in args)
        out = self.naf(*args, **kwargs)
        if torch.is_tensor(out):
            return out.float()
        if isinstance(out, (tuple, list)):
            return type(out)(o.float() if torch.is_tensor(o) else o for o in out)
        return out


class DummyNAFModel(torch.nn.Module):
    _proj = None
    def forward(self, image_tile, lr_features_bchw, target_size):
        if isinstance(target_size, (tuple, list)):
            H_out, W_out = int(target_size[0]), int(target_size[1])
        else:
            H_out = W_out = int(target_size)
        base = torch.nn.functional.interpolate(
            lr_features_bchw, size=(H_out, W_out), mode='bilinear', align_corners=False
        )
        if os.environ.get("TRELLIS2_NAF_SHARPEN", "0") != "1":
            return base
        try:
            C = base.shape[1]
            img = torch.nn.functional.interpolate(
                image_tile, size=(H_out, W_out), mode='bilinear', align_corners=False
            )  # [B,3,H,W]
            P = DummyNAFModel._proj
            if P is None or P.shape[0] != C or P.device != base.device:
                g = torch.Generator(device='cpu').manual_seed(1234)
                P = (torch.randn(C, 3, generator=g) * 0.15).to(base.device, base.dtype)
                DummyNAFModel._proj = P
            hi = torch.nn.functional.conv2d(img, P.unsqueeze(-1).unsqueeze(-1))  # [B,C,H,W]
            return base + hi
        except Exception as _e:
            print("[Trellis2] NAF_SHARPEN skipped:", _e)
            return base

import os

# Detect AMD GPU (ROCm)
is_amd_gpu = torch.cuda.is_available() and 'AMD' in torch.cuda.get_device_name(0)
print(f'[Trellis2] GPU Type: {"AMD ROCm" if is_amd_gpu else "NVIDIA CUDA"}')

import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import DINOv3ViTModel
import numpy as np
from PIL import Image, ImageDraw

import torch.distributed as dist
from ....utils import dist_utils
from ....utils.dist_utils import read_file_dist


# =============================================================================
# Projection Utilities
# =============================================================================

def project_points_to_image_batch(
    points_3d: torch.Tensor, 
    transform_matrix: torch.Tensor, 
    camera_angle_x: torch.Tensor, 
    resolution: int = 518
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Project 3D points to 2D image coordinates (batch processing).
    
    Args:
        points_3d: torch.Tensor, shape [N, 3] or [B, N, 3], 3D point coordinates (in [-1, 1] range)
        transform_matrix: torch.Tensor, shape [B, 4, 4], camera transformation matrix
        camera_angle_x: torch.Tensor, shape [B], horizontal field of view angle (radians)
        resolution: int, image resolution, default 518
    
    Returns:
        points_2d: torch.Tensor, shape [B, N, 2], image coordinates [x, y]
        depth: torch.Tensor, shape [B, N], depth values
        valid_mask: torch.Tensor, shape [B, N], mask for points within view
    """
    device = points_3d.device
    B = transform_matrix.shape[0]
    
    # Ensure inputs are torch.Tensor on correct device
    if not isinstance(transform_matrix, torch.Tensor):
        transform_matrix = torch.tensor(transform_matrix, dtype=torch.float32, device=device)
    if not isinstance(points_3d, torch.Tensor):
        points_3d = torch.tensor(points_3d, dtype=torch.float32, device=device)
    if not isinstance(camera_angle_x, torch.Tensor):
        camera_angle_x = torch.tensor(camera_angle_x, dtype=torch.float32, device=device)
    
    # Expand points_3d to batch dimension: [N, 3] -> [B, N, 3]
    if points_3d.dim() == 2:
        points_3d_batch = points_3d.unsqueeze(0).expand(B, -1, -1)
    else:
        points_3d_batch = points_3d
    N = points_3d_batch.shape[1]
    
    # Add homogeneous coordinates: [B, N, 3] -> [B, N, 4]
    ones = torch.ones(B, N, 1, device=device, dtype=points_3d_batch.dtype)
    points_homogeneous = torch.cat([points_3d_batch, ones], dim=-1)  # [B, N, 4]
    
    # Compute world to camera transformation matrix
    world_to_camera = torch.linalg.inv(transform_matrix.float()).to(transform_matrix.dtype)  # linalg.inv requires fp32+
    
    # Batch transform to camera coordinate system: [B, N, 4] @ [B, 4, 4]^T -> [B, N, 3]
    points_camera = torch.bmm(points_homogeneous, world_to_camera.transpose(-2, -1))[..., :3]  # [B, N, 3]
    
    # Extract camera coordinates
    x_cam = points_camera[..., 0]  # [B, N]
    y_cam = points_camera[..., 1]  # [B, N]
    z_cam = points_camera[..., 2]  # [B, N]
    
    # Depth value (Z value in camera coordinate system, note Blender camera faces -Z direction)
    depth = -z_cam  # [B, N]
    
    # Compute camera intrinsics (batch processing)
    sensor_width = 32.0  # mm
    focal_length = 16.0 / torch.tan(camera_angle_x / 2.0)  # [B]
    focal_length_pixels = focal_length * resolution / sensor_width  # [B]
    
    # Expand focal_length_pixels dimension for broadcasting: [B] -> [B, 1]
    focal_length_pixels = focal_length_pixels.unsqueeze(1)  # [B, 1]
    
    # Perspective projection to NDC coordinates
    x_ndc = focal_length_pixels * x_cam / (-z_cam + 1e-8)  # [B, N]
    y_ndc = focal_length_pixels * y_cam / (-z_cam + 1e-8)  # [B, N]
    
    # Convert to image coordinates (pixel coordinates)
    x_pixel = x_ndc + resolution / 2.0  # [B, N]
    y_pixel = -y_ndc + resolution / 2.0  # [B, N], flip Y axis
    
    # Create validity mask (points within image range and in front of camera)
    valid_mask = (
        (x_pixel >= 0) & (x_pixel < resolution) &
        (y_pixel >= 0) & (y_pixel < resolution) &
        (depth > 0)  # In front of camera
    )  # [B, N]
    
    points_2d = torch.stack([x_pixel, y_pixel], dim=-1)  # [B, N, 2]
    
    return points_2d, depth, valid_mask


def sample_features(fmap: torch.Tensor, queries_ndc: torch.Tensor) -> torch.Tensor:
    """
    Sample features from feature map at specified NDC coordinates.
    
    Args:
        fmap: torch.Tensor, shape [B, C, H, W], feature map
        queries_ndc: torch.Tensor, shape [B, K, 2], normalized device coordinates
    
    Returns:
        torch.Tensor, shape [B, C, K], sampled features
    """
    B, C, H, W = fmap.shape
    Bq, K, _ = queries_ndc.shape
    assert Bq == B, "Batch size mismatch"

    # grid_sample requires (B, out_h, out_w, 2), here we want K points -> out_h=K, out_w=1
    grid = queries_ndc.view(B, K, 1, 2)  # (B, K, 1, 2)

    # Bilinear interpolation, align_corners=False (consistent with [-1,1] pixel center convention)
    feat = F.grid_sample(
        fmap, grid, mode='bilinear',
        align_corners=False, padding_mode='border'  # border avoids out-of-bound becoming 0
    )  # (B, C, K, 1)

    return feat.squeeze(-1)  # (B, C, K)



# =============================================================================
# NAF on ROCm: MIOpen off, and exact row bands instead of image tiles
# =============================================================================
#
# Two measured problems with the real NAF on this gfx1030 box (2026-09-23,
# standalone repro under /mnt/sata/ComfyUI/_debug_naf):
#
# 1. MIOpen convolutions race. With cudnn(MIOpen) enabled, 4 of 4 runs showed a
#    conv output that read non-finite and then finite on the next read of the
#    SAME tensor; with MIOpen disabled, 0 of 4. A later kernel reads the result
#    before it is written - NaN when the stale memory is NaN, silently wrong
#    features when it is not. So every NAF call runs with MIOpen off.
#
# 2. The old image tiling was not an approximation of NAF, it was a different
#    function. Each tile got the image TILE but the WHOLE view's LR features,
#    which NAF resizes to fill the tile - so tile (1,0) received the top-left of
#    the whole view's features. GroupNorm statistics and RoPE coordinates were
#    per-tile too. NAF's only non-local op is a dilated 9x9 neighbourhood
#    attention (dilation = out/lr = 16 at 1024, so +-64 output rows). Running
#    the encoder once globally and only the attention in row bands with that
#    halo is EXACT: max|diff| = 0 against the untiled call at 512.

class _no_miopen:
    """Disable cudnn (MIOpen on ROCm) for the duration, restore afterwards."""
    def __enter__(self):
        self._prev = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
    def __exit__(self, *exc):
        torch.backends.cudnn.enabled = self._prev
        return False


def _naf_disable_miopen(model):
    """Route every call of this NAF instance - from any call site - around MIOpen."""
    if getattr(torch.version, "hip", None) is None or getattr(model, "_trellis2_no_miopen", False):
        return model
    orig = model.forward
    def forward(*args, **kwargs):
        with _no_miopen():
            return orig(*args, **kwargs)
    model.forward = forward
    model._trellis2_no_miopen = True
    return model


def _real_naf_core(model):
    """The underlying valeoai NAF module, or None for the bilinear stand-in."""
    for m in (model, getattr(model, "naf", None)):
        if m is not None and all(hasattr(m, a) for a in ("image_encoder", "key_encoder", "upsampler")):
            return m
    return None


def _nearest_exact_index(n_out, n_in, device):
    # F.interpolate(mode="nearest-exact"): src = floor((dst + 0.5) * in / out), clamped.
    idx = (torch.arange(n_out, device=device, dtype=torch.float64) + 0.5) * (n_in / n_out)
    return idx.floor().long().clamp_(max=n_in - 1)


def _naf_encode(naf, image, out_size):
    """NAF's image encoder over the WHOLE image (global GroupNorm stats, global RoPE).

    Without MIOpen the 1024 encoder needs ~7 GiB of im2col scratch on the GPU,
    so it runs there only when that much is actually free; otherwise on CPU
    (~16 s per 1024 view, same result to 4e-5).
    """
    choice = os.environ.get("TRELLIS2_NAF_ENCODER_DEVICE", "auto").lower()
    dev = image.device
    if dev.type == "cuda" and choice == "auto":
        free, _ = torch.cuda.mem_get_info(dev)
        choice = "cuda" if free >= 8 * 2**30 else "cpu"
    if dev.type != "cuda" or choice != "cpu":
        with _no_miopen():
            return naf.image_encoder(image, output_size=out_size)
    import copy
    enc = copy.deepcopy(naf.image_encoder).cpu()
    enc.rope._cached_coords = None
    enc.rope._cached_hw = None
    print(f"[Trellis2 NAF bands] encoder on CPU ({torch.cuda.mem_get_info(dev)[0] / 2**30:.1f} GiB free on GPU)")
    return enc(image.cpu(), output_size=out_size).to(dev)


@torch.no_grad()
def _naf_exact_bands(naf, image, lr, out_size, n_bands, keep_margin=2):
    """Yield (keep_lo, keep_hi, core_lo, core_hi, rows) where
    rows == NAF(image, lr, out_size)[:, :, keep_lo:keep_hi] exactly.

    keep_margin rows either side of the core let bilinear sampling of any point
    in the core read both of its neighbouring rows from this band.
    """
    from einops import rearrange
    # Take it from the class's own module globals: the hub package ("src.*") is
    # only importable while the loader's hide_src_module() context is active.
    legacy_attention = type(naf.upsampler).forward.__globals__["legacy_attention"]
    H, W = int(out_size[0]), int(out_size[1])
    x = _naf_encode(naf, image, (H, W))
    keys_lr = naf.key_encoder(x, lr)
    up = naf.upsampler
    hk, wk = keys_lr.shape[-2:]
    dil = (H // hk, W // wk)
    halo = (up.kernel_size[0] // 2) * dil[0]
    rows_all = _nearest_exact_index(H, hk, lr.device)
    cols = _nearest_exact_index(W, wk, lr.device)
    n_bands = max(1, min(int(n_bands), H))
    for b in range(n_bands):
        c0, c1 = H * b // n_bands, H * (b + 1) // n_bands
        k0, k1 = max(0, c0 - keep_margin), min(H, c1 + keep_margin)
        a0, a1 = max(0, k0 - halo), min(H, k1 + halo)
        rows = rows_all[a0:a1]
        q = rearrange(x[:, :, a0:a1], "b (n d) h w -> b h w n d", n=up.num_heads)
        k = rearrange(keys_lr[:, :, rows][:, :, :, cols], "b (n d) h w -> b h w n d", n=up.num_heads)
        v = rearrange(lr[:, :, rows][:, :, :, cols], "b (n d) h w -> b h w n d", n=up.num_heads)
        out = legacy_attention(q, k, v, up.kernel_size, dil, scale=up.scale)
        del q, k, v
        out = rearrange(out, "b h w n d -> b (n d) h w")[:, :, k0 - a0:k1 - a0]
        yield k0, k1, c0, c1, out
        del out


# =============================================================================
# Projection Grid Module
# =============================================================================

class ProjGrid(nn.Module):
    """
    3D Grid Projection Module.
    
    Projects a 3D grid of points to 2D image coordinates and samples features 
    from the image feature map at those locations.
    
    This is the core module for view-aligned feature extraction.
    """
    def __init__(self, grid_resolution: int = 16, image_resolution: int = 518):
        super().__init__()
        self.grid_resolution = grid_resolution
        self.image_resolution = image_resolution
        
        # Create 3D grid points
        one_dim = torch.linspace(-1, 1, grid_resolution)
        x, y, z = torch.meshgrid(one_dim, one_dim, one_dim, indexing='ij')
        grid_points = torch.stack((x, y, z), dim=-1)
        
        # Rotation matrix to align with Blender coordinate system
        rotation_matrix = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0]
        ])
        grid_points = torch.matmul(grid_points, rotation_matrix.T)
        grid_points = grid_points.reshape(-1, 3)
        self.register_buffer('grid_points', grid_points)  # [R³, 3]
        
        # Default front view transformation matrix
        front_view_transform_matrix = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, -2.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        self.register_buffer("front_view_transform_matrix", front_view_transform_matrix)
        
    def forward(
        self, 
        features_map: torch.Tensor, 
        camera_angle_x: torch.Tensor,
        distance: torch.Tensor,
        mesh_scale: torch.Tensor,
        transform_matrix: Optional[torch.Tensor] = None,
        BHWC: bool = True
    ) -> torch.Tensor:
        """
        Project 3D grid points to image and sample features.
        
        Args:
            features_map: Feature map, shape [B, H, W, C] if BHWC else [B, C, H, W]
            camera_angle_x: Camera FOV angle, shape [B]
            distance: Camera distance, shape [B]
            mesh_scale: Mesh scale factor, shape [B]
            transform_matrix: Optional camera transform matrix, shape [B, 4, 4]
            BHWC: Whether features_map is in BHWC format
            
        Returns:
            Projected features, shape [B, grid_resolution³, C]
        """
        if BHWC:
            B, H, W, C = features_map.shape
        else:
            B, C, H, W = features_map.shape
            
        grid_points = self.grid_points
        grid_points = grid_points.expand(B, -1, -1)
        grid_points = grid_points / mesh_scale.unsqueeze(-1).unsqueeze(-1) / 2  # Scale alignment
        assert transform_matrix is None, "transform_matrix is not None"
        if transform_matrix is None:
            transform_matrix = self.front_view_transform_matrix
            transform_matrix = transform_matrix.expand(B, -1, -1).clone()
            transform_matrix[:, 1, 3] = -distance  # Set camera distance
            
        # Project to image coordinates (simulate Blender projection)
        image_points, depth, valid_mask = project_points_to_image_batch(
            grid_points, transform_matrix, camera_angle_x, self.image_resolution
        )
        
        # Normalize to [-1, 1] for grid_sample
        image_points_norm = (image_points + 0.5) / self.image_resolution * 2 - 1
        
        if BHWC:
            features_map = features_map.permute(0, 3, 1, 2)  # [B, C, H, W]
            
        # Sample features from DINOv3 patch feature map
        x = sample_features(features_map, image_points_norm)  # [B, C, K]
        x = x.permute(0, 2, 1)  # [B, K, C]
   
        return x
    
    def visualize_projection(
        self,
        image: torch.Tensor,
        camera_angle_x: torch.Tensor,
        distance: torch.Tensor,
        mesh_scale: torch.Tensor,
        transform_matrix: Optional[torch.Tensor] = None,
        save_dir: Optional[str] = None,
        prefix: str = "proj_vis",
    ) -> List[Image.Image]:
        """
        Visualize the projected 3D grid points on the input image.
        
        Args:
            image: Input image tensor [B, C, H, W], assumed to be in [0, 1] range
            camera_angle_x: Camera FOV angle, shape [B]
            distance: Camera distance, shape [B]
            mesh_scale: Mesh scale factor, shape [B]
            transform_matrix: Optional camera transform matrix, shape [B, 4, 4]
            save_dir: Directory to save visualizations (optional)
            prefix: Prefix for saved files
            
        Returns:
            List of PIL Images with projected points overlaid
        """
        B = image.shape[0]
        
        # Get projected points
        grid_points = self.grid_points.expand(B, -1, -1)
        grid_points = grid_points / mesh_scale.unsqueeze(-1).unsqueeze(-1) / 2
        assert transform_matrix is None, "transform_matrix is not None"
        if transform_matrix is None:
            transform_matrix = self.front_view_transform_matrix
            transform_matrix = transform_matrix.expand(B, -1, -1).clone()
            transform_matrix[:, 1, 3] = -distance
            
        image_points, depth, valid_mask = project_points_to_image_batch(
            grid_points, transform_matrix, camera_angle_x, self.image_resolution
        )
        
        # Convert image to PIL for visualization
        vis_images = []
        for b in range(B):
            # Convert tensor to PIL image
            img_np = image[b].cpu().permute(1, 2, 0).numpy()
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
            
            # Resize to image_resolution if needed
            pil_img = Image.fromarray(img_np)
            if pil_img.size != (self.image_resolution, self.image_resolution):
                pil_img = pil_img.resize((self.image_resolution, self.image_resolution), Image.LANCZOS)
            
            # Create a copy for drawing
            vis_img = pil_img.copy()
            draw = ImageDraw.Draw(vis_img)
            
            # Get points for this batch
            pts = image_points[b].cpu().numpy()  # [K, 2]
            depths = depth[b].cpu().numpy()  # [K]
            mask = valid_mask[b].cpu().numpy()  # [K]
            
            # Normalize depth for coloring
            valid_depths = depths[mask]
            if len(valid_depths) > 0:
                d_min, d_max = valid_depths.min(), valid_depths.max()
                if d_max - d_min > 1e-6:
                    depths_norm = (depths - d_min) / (d_max - d_min)
                else:
                    depths_norm = np.ones_like(depths) * 0.5
            else:
                depths_norm = np.ones_like(depths) * 0.5
            
            # Draw projected points
            R = self.grid_resolution
            for i, (pt, d, m, dn) in enumerate(zip(pts, depths, mask, depths_norm)):
                if not m:
                    continue
                    
                x, y = pt
                
                # Color by depth (blue=near, red=far)
                r = int(255 * dn)
                g = int(255 * (1 - abs(2 * dn - 1)))
                b_color = int(255 * (1 - dn))
                color = (r, g, b_color)
                
                # Draw small circle
                radius = 2
                draw.ellipse(
                    [x - radius, y - radius, x + radius, y + radius],
                    fill=color,
                    outline=color
                )
            
            vis_images.append(vis_img)
            
            # Save if directory is specified
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{prefix}_batch{b}.png")
                vis_img.save(save_path)
                print(f"Saved projection visualization to: {save_path}")
        
        return vis_images


# =============================================================================
# DINOv3 Feature Extractor with Projection
# =============================================================================

class DinoV3ProjFeatureExtractor(nn.Module):
    """
    DINOv3 Feature Extractor with View-Aligned Projection.
    
    This extractor produces both:
    1. Global features (CLS token + register tokens) in embed_dim
    2. View-aligned projected features (3D grid projected to 2D and sampled)
       - Without NAF: [B, R³, embed_dim]
       - With NAF:    [B, R³, embed_dim * 2]  (concat of lr and hr features)
    
    NOTE: proj_linear has been moved to per-block ProjectAttention / SparseProjectAttention.
    This module now outputs raw DINOv3 features for proj (optionally concatenated with NAF-upsampled features).
    
    Args:
        model_name: Name of the pretrained DINOv3 model
        image_size: Input image size (default: 512)
        grid_resolution: Resolution of the 3D projection grid (default: 16)
        use_naf_upsample: Whether to use NAF to upsample features (default: False)
        naf_target_size: Target spatial size for NAF upsampling (default: [128, 128])
    """
    def __init__(
        self, 
        model_name: str,
        image_size: int = 512,
        grid_resolution: int = 16,
        use_naf_upsample: bool = False,
        naf_target_size: Optional[List[int]] = None
    ):
        super().__init__()
        self.model_name = model_name
        self.image_size = image_size
        self.grid_resolution = grid_resolution
        self.use_naf_upsample = use_naf_upsample
        if naf_target_size is None:
            self.naf_target_size = (128, 128)
        elif isinstance(naf_target_size, int):
            self.naf_target_size = (naf_target_size, naf_target_size)
        else:
            self.naf_target_size = tuple(naf_target_size)
        
        # Load DINOv3 model (frozen, no trainable params in this module)
        self.model = DINOv3ViTModel.from_pretrained(model_name)
        self.model.eval()
        self.model.requires_grad_(False)
        
        # Image transform (only normalize, no resize - assume already resized)
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Get patch info
        self.patch_size = self.model.config.patch_size
        self.patch_number = image_size // self.patch_size
        self.embed_dim = self.model.config.hidden_size
        
        # Projection grid for view-aligned features
        self.proj_grid = ProjGrid(
            grid_resolution=grid_resolution,
            image_resolution=image_size
        )
        
        # NAF upsampler (frozen, no trainable params)
        self.naf_model = None  # lazy; _load_naf decides

        # Per-axis spatial tile factor for the NAF + proj_grid streaming path. 1 = un-tiled
        # (current behavior, default). >1 enables a streaming wrapper that tiles NAF and
        # the projection together, capping the transient HR-feature-map peak. Set transiently
        # by the pipeline (see Pixal3DImageTo3DPipeline.get_proj_cond_shape).
        self.naf_tile_factor: int = 1

        # proj_channels: the output dimension of proj features
        # Without NAF: embed_dim (e.g. 1024)
        # With NAF: embed_dim * 2 (e.g. 2048, concat of lr and hr)
        self.proj_channels = self.embed_dim * 2 if use_naf_upsample else self.embed_dim
        
        # NOTE: proj_linear removed — now lives in each denoiser block's ProjectAttention

        # ROCm 1024 DINO isolation state.  gfx1030 can occasionally leave
        # persistent model storage corrupted after a long 4101-token pass.
        # Keep a pristine CPU state_dict so every multiview extraction starts
        # from known-good weights instead of reusing potentially damaged VRAM.
        self._rocm_dino_cpu_master = None
        self._rocm_dino_restore_count = 0

        # Capture the master while DINO is still freshly loaded on CPU.
        # Waiting until the second view is too late if a prior ROCm kernel has
        # already scribbled over persistent GPU storage.
        if (
            is_amd_gpu
            and int(self.image_size) >= 1024
            and os.environ.get(
                "TRELLIS2_DINO_RESTORE_MODEL",
                "1",
            ) == "1"
        ):
            self._ensure_rocm_dino_cpu_master()
    
    def _cached_naf_repo(self) -> Optional[str]:
        import torch.hub

        hub_dir = torch.hub.get_dir()
        if not os.path.isdir(hub_dir):
            return None
        for name in os.listdir(hub_dir):
            repo_dir = os.path.join(hub_dir, name)
            if name.startswith("valeoai_NAF_") and os.path.exists(os.path.join(repo_dir, "hubconf.py")):
                return repo_dir
        return None    
    
    def _load_naf(self, device=None):
        # DINO 1024 may intentionally stay on CPU for ROCm correctness while
        # NAF still consumes GPU-resident image/features.  Therefore NAF's
        # execution device must come from its actual input, not from
        # next(self.model.parameters()).device.
        if self.naf_model is not None:
            if device is not None:
                self.naf_model.to(device)
            return
        _real = is_amd_gpu and os.environ.get("TRELLIS2_REAL_NAF", "0") == "1"
        if is_amd_gpu and not _real:
            self.naf_model = DummyNAFModel()
            print("[Trellis2] AMD -> DummyNAFModel (bilinear fallback). Set TRELLIS2_REAL_NAF=1 for real NAF.")
            return
        if _real:
            try:
                import sys as _sys
                _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                import _natten_shim as _nshim
                _nshim.install()
                print("[Trellis2] real-NAF: pure-torch natten shim installed; loading NAF via torch.hub...")
            except Exception as _e:
                print("[Trellis2] shim install failed -> DummyNAFModel:", repr(_e))
                self.naf_model = DummyNAFModel()
                return
        if self.naf_model is None:
            try:
                self.naf_model = torch.hub.load("valeoai/NAF", "naf", pretrained=True, trust_repo=True)
            except Exception as _he:
                print("[Trellis2] hub.load(github) failed:", repr(_he))
                try:
                    _ld = os.path.expanduser("~/.cache/torch/hub/valeoai_NAF_main")
                    if os.path.isdir(_ld):
                        self.naf_model = torch.hub.load(_ld, "naf", source="local", pretrained=True)
                except Exception as _he2:
                    print("[Trellis2] hub.load(local) failed:", repr(_he2))
            if self.naf_model is not None:
                self.naf_model.eval()
                self.naf_model.requires_grad_(False)
        if self.naf_model is None:
            print("[Trellis2] NAF load failed -> DummyNAFModel fallback")
            self.naf_model = DummyNAFModel()
        else:
            try:
                _naf_dtype = os.environ.get(
                    "TRELLIS2_NAF_DTYPE",
                    "fp32" if is_amd_gpu else "bf16",
                ).lower()
                _naf_device = (
                    torch.device(device)
                    if device is not None
                    else next(self.model.parameters()).device
                )

                if _naf_dtype in ("fp32", "float32"):
                    # gfx1030: keep real NAF in native FP32.
                    # Do not route it through the custom BF16 wrapper.
                    self.naf_model = self.naf_model.float()
                    self.naf_model.to(_naf_device)
                    print(
                        "[Trellis2] real-NAF dtype=fp32 "
                        "(BF16 wrapper disabled)"
                    )

                elif _naf_dtype in ("bf16", "bfloat16"):
                    self.naf_model = _NAFBf16Wrapper(self.naf_model)
                    self.naf_model.to(_naf_device)
                    print("[Trellis2] real-NAF dtype=bf16")

                else:
                    raise ValueError(
                        "TRELLIS2_NAF_DTYPE must be fp32 or bf16, got "
                        + repr(_naf_dtype)
                    )
            except Exception as _e:
                print("[Trellis2] naf device-move skipped:", repr(_e))
            print("[Trellis2] NAF ready: type=%s" % type(self.naf_model).__name__)
            _naf_disable_miopen(self.naf_model)


    def _cpu1024_pin_dino_on_cpu(self, device) -> bool:
        """Return True when the 1024 ROCm conditioner must keep DINO on CPU.

        The Pixal3D pipeline calls ``image_cond_model.to(cuda)`` before each
        conditioning stage.  A normal ``nn.Module.to`` recursively moves every
        child module, which would move the 1.1 GiB DINO backbone to ROCm only
        for ``extract_features`` to copy it straight back to CPU.  Besides the
        wasted transfer/VRAM, that reintroduces the exact GPU residency this
        correctness mode is meant to avoid.
        """
        if not is_amd_gpu or int(self.image_size) < 1024:
            return False
        if os.environ.get("TRELLIS2_DINO_1024_CPU", "1") != "1":
            return False
        try:
            return torch.device(device).type in ("cuda", "hip")
        except Exception:
            return False

    def to(self, device):
        if self._cpu1024_pin_dino_on_cpu(device):
            # Do NOT call super().to(device) here: nn.Module.to recursively
            # descends into self.model and would move DINO to ROCm first.
            # This wrapper has no trainable tensors of its own; the relevant
            # children are DINO, projection grid, and optional NAF.
            self.model.float().cpu()
            self.proj_grid.to(device)
            if self.naf_model is not None:
                self.naf_model.to(device)
            if not getattr(self, "_trellis2_cpu1024_selective_to_logged", False):
                print(
                    "[Pixal3D DINO CPU1024] selective .to(): keeping DINO on CPU; "
                    f"projection/NAF -> {torch.device(device)}"
                )
                self._trellis2_cpu1024_selective_to_logged = True
            return self

        super().to(device)
        self.model.to(device)
        self.proj_grid.to(device)
        if self.naf_model is not None:
            self.naf_model.to(device)
        return self

    def cuda(self):
        # Route through the selective ``to`` implementation so a 1024 ROCm
        # conditioner cannot accidentally be pushed onto the GPU by callers
        # that use .cuda() instead of .to("cuda").
        return self.to(torch.device("cuda"))

    def cpu(self):
        super().cpu()
        self.model.cpu()
        self.proj_grid.cpu()
        if self.naf_model is not None:
            self.naf_model.cpu()
        return self

    def _ensure_rocm_dino_cpu_master(self):
        """Snapshot a pristine CPU copy of the complete DINO state once.

        The previous workaround only protected LayerNorm affine parameters.
        That cannot recover if a long ROCm kernel damages patch embedding,
        attention, MLP, layer-scale, or another persistent tensor between
        multiview chunks.  A full state snapshot gives each view a clean start.
        """
        if self._rocm_dino_cpu_master is not None:
            return

        master = {}
        total_bytes = 0

        with torch.no_grad():
            for name, tensor in self.model.state_dict().items():
                src = tensor.detach().to(device="cpu").clone()

                # DINO is forced to FP32 on the ROCm safe path.  Keep floating
                # masters in FP32 as well so a later restore cannot reintroduce
                # reduced-precision state.
                if src.is_floating_point() and src.dtype != torch.float32:
                    src = src.float()

                if src.is_floating_point() and not bool(torch.isfinite(src).all().item()):
                    raise RuntimeError(
                        "[Pixal3D DINO STATE] refusing to snapshot non-finite "
                        f"parameter/buffer {name!r}"
                    )

                master[name] = src
                total_bytes += src.numel() * src.element_size()

        self._rocm_dino_cpu_master = master

        print(
            "[Pixal3D DINO STATE] pristine full-model CPU master cached: "
            f"{len(master)} tensors, {total_bytes / (1024 ** 2):.1f} MiB"
        )

    def _restore_rocm_dino_cpu_master(self):
        """Restore the full DINO state in-place from the CPU master."""
        self._ensure_rocm_dino_cpu_master()
        current = self.model.state_dict()

        with torch.no_grad():
            for name, src in self._rocm_dino_cpu_master.items():
                dst = current.get(name)
                if dst is None:
                    raise RuntimeError(
                        "[Pixal3D DINO STATE] state key disappeared: "
                        + name
                    )
                dst.copy_(src, non_blocking=False)

        self.model.eval()
        self._rocm_dino_restore_count += 1

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        print(
            "[Pixal3D DINO STATE] full model restored from CPU master "
            f"(view pass {self._rocm_dino_restore_count})"
        )
    
    def _extract_features_rocm_cpu_1024(
        self,
        image: torch.Tensor,
        output_device: torch.device,
        debug: bool = True,
    ) -> torch.Tensor:
        """Run the unstable 1024 DINO pass entirely on CPU.

        gfx1030/ROCm has now demonstrated corruption at *multiple* GEMM sites:
        Q/K/V projection, attention score/value products, and the final
        attention output can all become NaN/Inf or silently enormous while
        their inputs are finite.  Once Q/K/V themselves are wrong, retrying
        only the attention matmul on CPU cannot recover the layer because the
        CPU receives already-corrupted projected tensors.

        The correctness path therefore keeps the 1024 DINO transformer on
        CPU from embeddings through final LayerNorm.  Only the completed,
        validated feature tensor is copied back to the caller's device.
        512-stage DINO and the rest of TRELLIS/Pixal3D remain on the GPU.
        """
        import contextlib

        print(
            "[Pixal3D DINO CPU1024] gfx1030 correctness mode: "
            "running the complete 1024 DINO transformer on CPU"
        )

        # Move the potentially-corrupted GPU copy out of the way first, then
        # overwrite every parameter/buffer from the pristine CPU master.  The
        # master was captured immediately after model load, before any long
        # ROCm kernel ran.
        self.model = self.model.float().cpu()
        self._restore_rocm_dino_cpu_master()
        self.model.eval()

        # Hard verification: no DINO parameter may remain on ROCm in this
        # path.  Failing loudly here is better than silently re-entering the
        # exact GPU kernels that corrupted the previous runs.
        _non_cpu = {str(p.device) for p in self.model.parameters() if p.device.type != "cpu"}
        if _non_cpu:
            raise RuntimeError(
                "[Pixal3D DINO CPU1024] model is not fully on CPU: "
                + ", ".join(sorted(_non_cpu))
            )

        image_cpu = image.detach().to(
            device="cpu",
            dtype=torch.float32,
        ).contiguous()

        def _cpu_validate(name: str, tensor: torch.Tensor) -> None:
            if not tensor.is_floating_point():
                return
            finite = torch.isfinite(tensor)
            bad = tensor.numel() - int(finite.sum().item())
            absmax = (
                float(tensor.detach().abs().amax().item())
                if tensor.numel()
                else 0.0
            )
            if debug:
                print(
                    "[Pixal3D DINO CPU1024] "
                    f"{name}: shape={tuple(tensor.shape)}, "
                    f"dtype={tensor.dtype}, bad={bad}, "
                    f"absmax={absmax:.6g}"
                )
            if bad:
                raise RuntimeError(
                    "[Pixal3D DINO CPU1024] "
                    f"{name} produced {bad} NaN/Inf values on CPU"
                )

            # IMPORTANT: do not reject large-but-finite intermediate DINO
            # residuals.  The original Pixal3D extractor intentionally lets
            # those residual streams grow very large and applies an explicit
            # F.layer_norm() only after the final transformer block.  The
            # user's CPU run reproduced the exact same ~1.39588e5 layer-0
            # magnitude with both SDPA and eager attention, proving that the
            # old 1e4 cutoff was a false-positive rather than ROCm corruption.
            # NaN/Inf remains fatal; magnitude sanity is applied only to the
            # *final normalized* features below.

        requested_impl = os.environ.get(
            "TRELLIS2_DINO_1024_CPU_ATTN",
            "sdpa",
        ).strip().lower()
        if requested_impl not in ("sdpa", "eager"):
            requested_impl = "sdpa"

        previous_impl = getattr(
            self.model.config,
            "_attn_implementation",
            None,
        )

        def _run_once(attn_impl: str) -> torch.Tensor:
            self.model.config._attn_implementation = attn_impl
            print(
                "[Pixal3D DINO CPU1024] attention backend="
                + repr(attn_impl)
            )

            with torch.inference_mode(), contextlib.nullcontext():
                hidden_states = self.model.embeddings(
                    image_cpu,
                    bool_masked_pos=None,
                )
                _cpu_validate("embeddings", hidden_states)

                position_embeddings = self.model.rope_embeddings(
                    image_cpu
                )
                _cpu_validate("rope_cos", position_embeddings[0])
                _cpu_validate("rope_sin", position_embeddings[1])

                for layer_idx, layer_module in enumerate(self.model.layer):
                    hidden_states = layer_module(
                        hidden_states,
                        position_embeddings=position_embeddings,
                    )
                    _cpu_validate(
                        f"layer_{layer_idx}",
                        hidden_states,
                    )

                # Pixal3D's released projected-image conditioner does NOT use
                # DINO's learned affine final norm.  The official extractor
                # returns F.layer_norm(hidden_states, hidden_states.shape[-1:])
                # and the released *_mv denoisers were trained against exactly
                # that feature distribution.  Keep the generic Trellis2 path
                # unchanged, but make the Pixal3D MV subclass use the official
                # parameterless normalization on this CPU1024 correctness path.
                if getattr(self, "_pixal3d_official_final_norm", False):
                    hidden_states = F.layer_norm(
                        hidden_states,
                        hidden_states.shape[-1:],
                    )
                elif hasattr(self.model, "norm"):
                    hidden_states = self.model.norm(hidden_states)
                else:
                    hidden_states = F.layer_norm(
                        hidden_states,
                        hidden_states.shape[-1:],
                    )
                _cpu_validate(
                    "final_layer_norm",
                    hidden_states,
                )

                # Do not add another arbitrary magnitude cutoff here.  The
                # production extractor validates finiteness, not an ad-hoc
                # absmax threshold.  Shape is checked immediately below.

                _num_reg = int(getattr(self.model.config, "num_register_tokens", 4))
                _expected_tokens = 1 + _num_reg + self.patch_number * self.patch_number
                if (
                    hidden_states.ndim != 3
                    or hidden_states.shape[1] != _expected_tokens
                    or hidden_states.shape[2] != self.embed_dim
                ):
                    raise RuntimeError(
                        "[Pixal3D DINO CPU1024] unexpected final feature shape "
                        f"{tuple(hidden_states.shape)}; expected "
                        f"[B,{_expected_tokens},{self.embed_dim}]"
                    )

                return hidden_states.contiguous()

        try:
            try:
                hidden_cpu = _run_once(requested_impl)
            except Exception as first_error:
                if requested_impl == "eager":
                    raise
                print(
                    "[Pixal3D DINO CPU1024] CPU SDPA pass failed; "
                    "restoring pristine state and retrying with eager: "
                    + repr(first_error)
                )
                self._restore_rocm_dino_cpu_master()
                hidden_cpu = _run_once("eager")
        finally:
            # Keep the 1024 model on CPU by default.  Moving 1.1 GiB of DINO
            # weights back to gfx1030 after every view both wastes time and
            # reintroduces the storage we are deliberately isolating from.
            # This extractor instance is the 1024 conditioner; the separate
            # 512 conditioner remains GPU resident.
            if os.environ.get(
                "TRELLIS2_DINO_1024_CPU_KEEP_MODEL",
                "1",
            ) != "1":
                self.model.to(output_device)

            if previous_impl is not None:
                self.model.config._attn_implementation = previous_impl

        result = hidden_cpu.to(
            device=output_device,
            dtype=torch.float32,
            non_blocking=False,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        print(
            "[Pixal3D DINO CPU1024] completed; validated features copied "
            f"back to {output_device}"
        )
        return result

    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extract features using DINOv3."""

        is_rocm = (
            getattr(torch.version, "hip", None)
            is not None
        )

        debug_1024 = (
            is_rocm
            and image.shape[-2] >= 1024
            and image.shape[-1] >= 1024
            and os.environ.get(
                "TRELLIS2_DINO_LAYER_DEBUG",
                "1",
            ) == "1"
        )

        if is_rocm:
            # [Pixal3D DINO ROCm FP32/eager]
            #
            # DINOv3 attention dispatches through
            # model.config._attn_implementation.
            #
            # gfx1030 becomes non-finite on the 64x64-patch
            # (1024 image) path when the optimized attention
            # backend is used. Keep DINO itself in FP32 and
            # use the explicit eager attention implementation.
            self.model = self.model.float()

            if (
                getattr(
                    self.model.config,
                    "_attn_implementation",
                    None,
                )
                != "eager"
            ):
                old_impl = getattr(
                    self.model.config,
                    "_attn_implementation",
                    None,
                )

                self.model.config._attn_implementation = (
                    "eager"
                )

                print(
                    "[Pixal3D DINO ROCm FP32/eager] "
                    f"attention {old_impl!r} -> 'eager'"
                )

            image = image.float()

        else:
            image = image.to(
                self.model
                .embeddings
                .patch_embeddings
                .weight
                .dtype
            )

        # v4 correctness path: the latest real run proves that the fault is
        # earlier than the attention score GEMM.  CPU-recomputing QK^T from
        # already-corrupted GPU Q/K cannot fix the layer.  For the 1024 ROCm
        # conditioner, default to a complete CPU DINO forward and copy only
        # the validated final features back to the GPU.
        force_cpu_1024 = (
            is_rocm
            and image.shape[-2] >= 1024
            and image.shape[-1] >= 1024
            and os.environ.get(
                "TRELLIS2_DINO_1024_CPU",
                "1",
            ) == "1"
        )

        if force_cpu_1024:
            return self._extract_features_rocm_cpu_1024(
                image,
                output_device=image.device,
                debug=(
                    os.environ.get(
                        "TRELLIS2_DINO_LAYER_DEBUG",
                        "1",
                    ) == "1"
                ),
            )

        self.model.to(image.device)

        # Full model-state isolation for the unstable 1024 ROCm multiview
        # path.  This is intentionally enabled by default on ROCm and can be
        # disabled only for A/B testing with TRELLIS2_DINO_RESTORE_MODEL=0.
        restore_full_model = (
            is_rocm
            and image.shape[-2] >= 1024
            and image.shape[-1] >= 1024
            and os.environ.get(
                "TRELLIS2_DINO_RESTORE_MODEL",
                "1",
            ) == "1"
        )

        if restore_full_model:
            self._restore_rocm_dino_cpu_master()

        def check_finite(name, tensor):
            if not debug_1024:
                return

            mask_cpu = (
                torch.isfinite(tensor)
                .detach()
                .cpu()
            )

            finite_count = int(
                mask_cpu.sum().item()
            )

            bad = (
                tensor.numel()
                - finite_count
            )

            # Finite-only checks missed the actual signature seen on the
            # second view: values can remain finite yet be so large that the
            # naive variance calculation overflows.  Always log absmax too.
            absmax = float(
                tensor.detach().float().abs().amax().cpu().item()
            ) if tensor.numel() else 0.0

            print(
                "[Pixal3D DINO 1024] "
                f"{name}: "
                f"shape={tuple(tensor.shape)}, "
                f"dtype={tensor.dtype}, "
                f"finite={finite_count}/"
                f"{tensor.numel()}, "
                f"bad={bad}, absmax={absmax:.6g}"
            )

            if bad:
                raise RuntimeError(
                    "[Pixal3D DINO 1024] "
                    f"{name} produced {bad} "
                    "real NaN/Inf values"
                )

        ctx = (
            torch.autocast(
                device_type="cuda",
                enabled=False,
            )
            if is_rocm
            else __import__("contextlib").nullcontext()
        )

        with ctx:
            hidden_states = self.model.embeddings(
                image,
                bool_masked_pos=None,
            )

            check_finite(
                "embeddings",
                hidden_states,
            )

            position_embeddings = (
                self.model.rope_embeddings(image)
            )

            check_finite(
                "rope_cos",
                position_embeddings[0],
            )

            check_finite(
                "rope_sin",
                position_embeddings[1],
            )

            # ----------------------------------------------------
            # gfx1030/ROCm recovery path for long DINO sequences.
            #
            # Run the normal FP32/eager layer first. If a 1024-stage
            # layer produces NaN/Inf, rerun ONLY that layer using
            # query-chunked FP32 attention and token-chunked MLP.
            #
            # This avoids a single enormous [N,N] eager attention
            # operation without slowing down every healthy layer.
            # ----------------------------------------------------

            def _cpu_bad_count(t):
                finite_cpu = (
                    torch.isfinite(t)
                    .detach()
                    .cpu()
                )
                return (
                    t.numel()
                    - int(finite_cpu.sum().item())
                )

            def _require_safe(name, t):
                bad = _cpu_bad_count(t)
                if bad:
                    raise RuntimeError(
                        "[Pixal3D DINO SAFE FALLBACK] "
                        f"{name} produced {bad} "
                        "NaN/Inf values"
                    )
                return t

            # [Pixal3D DINO SAFE NORM]
            #
            # gfx1030 / ROCm:
            # avoid the native large LayerNorm kernel for the
            # 4101 x 1024 DINO sequence.
            #
            # Also keep pristine CPU master copies of LayerNorm
            # affine parameters. This protects the safe path from
            # accidental GPU-side parameter corruption across views.
            def _ensure_norm_master(norm):
                if getattr(
                    norm,
                    "_trellis2_rocm_norm_master_ready",
                    False,
                ):
                    return

                weight = getattr(
                    norm,
                    "weight",
                    None,
                )

                bias = getattr(
                    norm,
                    "bias",
                    None,
                )

                weight_cpu = (
                    weight
                    .detach()
                    .cpu()
                    .float()
                    .clone()
                    if weight is not None
                    else None
                )

                bias_cpu = (
                    bias
                    .detach()
                    .cpu()
                    .float()
                    .clone()
                    if bias is not None
                    else None
                )

                object.__setattr__(
                    norm,
                    "_trellis2_rocm_norm_weight_cpu",
                    weight_cpu,
                )

                object.__setattr__(
                    norm,
                    "_trellis2_rocm_norm_bias_cpu",
                    bias_cpu,
                )

                object.__setattr__(
                    norm,
                    "_trellis2_rocm_norm_master_ready",
                    True,
                )

            def _safe_layer_norm_token_chunks(
                norm,
                x,
                token_chunk,
                name,
            ):
                _ensure_norm_master(norm)

                token_chunk = max(
                    1,
                    int(token_chunk),
                )

                eps = float(
                    getattr(
                        norm,
                        "eps",
                        1e-5,
                    )
                )

                weight_cpu = getattr(
                    norm,
                    "_trellis2_rocm_norm_weight_cpu",
                    None,
                )

                bias_cpu = getattr(
                    norm,
                    "_trellis2_rocm_norm_bias_cpu",
                    None,
                )

                weight = (
                    weight_cpu.to(
                        device=x.device,
                        dtype=torch.float32,
                    )
                    if weight_cpu is not None
                    else None
                )

                bias = (
                    bias_cpu.to(
                        device=x.device,
                        dtype=torch.float32,
                    )
                    if bias_cpu is not None
                    else None
                )

                out_parts = []

                for start in range(
                    0,
                    x.shape[1],
                    token_chunk,
                ):
                    end = min(
                        start + token_chunk,
                        x.shape[1],
                    )

                    src = (
                        x[:, start:end]
                        .contiguous()
                        .float()
                    )

                    _require_safe(
                        f"{name}.input."
                        f"{start}:{end}",
                        src,
                    )

                    # Explicit *scaled* FP32 LayerNorm.
                    #
                    # Do NOT call nn.LayerNorm/F.layer_norm here.  Also do not
                    # use mean((x-mean)^2) directly: a tensor may contain only
                    # finite values while square() still overflows FP32.
                    # Normalising each token by its max magnitude first keeps
                    # all intermediate variance math bounded.
                    scale = src.abs().amax(
                        dim=-1,
                        keepdim=True,
                    )

                    _require_safe(
                        f"{name}.scale.{start}:{end}",
                        scale,
                    )

                    safe_scale = scale.clamp_min(1.0)
                    src_scaled = src / safe_scale

                    mean_scaled = src_scaled.mean(
                        dim=-1,
                        keepdim=True,
                    )

                    centered_scaled = (
                        src_scaled - mean_scaled
                    )

                    var_scaled = (
                        centered_scaled
                        .square()
                        .mean(
                            dim=-1,
                            keepdim=True,
                        )
                    )

                    # eps / scale^2, written via reciprocal to avoid an
                    # overflow in scale.square() when scale itself is huge.
                    inv_scale = safe_scale.reciprocal()
                    eps_scaled = (
                        eps * inv_scale.square()
                    )

                    inv_std_scaled = torch.rsqrt(
                        var_scaled.clamp_min(0.0)
                        + eps_scaled
                    )

                    part = (
                        centered_scaled
                        * inv_std_scaled
                    )

                    if weight is not None:
                        part = (
                            part
                            * weight.view(
                                1,
                                1,
                                -1,
                            )
                        )

                    if bias is not None:
                        part = (
                            part
                            + bias.view(
                                1,
                                1,
                                -1,
                            )
                        )

                    _require_safe(
                        f"{name}.output."
                        f"{start}:{end}",
                        part,
                    )

                    out_parts.append(part)

                out = torch.cat(
                    out_parts,
                    dim=1,
                )

                _require_safe(
                    name,
                    out,
                )

                return out

            def _linear_token_chunks(
                linear,
                x,
                token_chunk,
            ):
                parts = []

                for start in range(
                    0,
                    x.shape[1],
                    token_chunk,
                ):
                    end = min(
                        start + token_chunk,
                        x.shape[1],
                    )

                    part = linear(
                        x[:, start:end]
                    )

                    _require_safe(
                        "linear chunk "
                        f"{start}:{end}",
                        part,
                    )

                    parts.append(part)

                return torch.cat(
                    parts,
                    dim=1,
                )

            def _rotate_half_local(x):
                half = x.shape[-1] // 2

                return torch.cat(
                    (
                        -x[..., half:],
                        x[..., :half],
                    ),
                    dim=-1,
                )

            def _apply_rope_local(
                q,
                k,
                pos,
            ):
                cos, sin = pos

                num_tokens = q.shape[-2]
                num_patches = sin.shape[-2]
                num_prefix = (
                    num_tokens - num_patches
                )

                q_prefix = q[..., :num_prefix, :]
                q_patch = q[..., num_prefix:, :]

                k_prefix = k[..., :num_prefix, :]
                k_patch = k[..., num_prefix:, :]

                cos = cos.float()
                sin = sin.float()

                # [patches, head_dim]
                # ->
                # [1, 1, patches, head_dim]
                while cos.ndim < q_patch.ndim:
                    cos = cos.unsqueeze(0)
                    sin = sin.unsqueeze(0)

                q_patch = (
                    q_patch * cos
                    + _rotate_half_local(q_patch) * sin
                )

                k_patch = (
                    k_patch * cos
                    + _rotate_half_local(k_patch) * sin
                )

                q = torch.cat(
                    (q_prefix, q_patch),
                    dim=-2,
                )

                k = torch.cat(
                    (k_prefix, k_patch),
                    dim=-2,
                )

                return q, k

            def _rocm_block_matmul(
                a,
                b,
                name,
                scale=1.0,
            ):
                """
                Small-block GEMM with a correctness fallback for gfx1030.

                The failure in the real four-view run is not an input NaN:
                finite Q/K produced NaNs *inside* the ROCm score GEMM.  We
                therefore validate each deliberately small attention block.
                A failed (or implausibly huge finite) GPU block is recomputed
                on CPU and copied back.  CPU is a rare recovery path, not the
                normal attention implementation.
                """
                out = torch.matmul(a, b)
                if scale != 1.0:
                    out.mul_(float(scale))

                finite_ok = bool(
                    torch.isfinite(out).all().item()
                )

                if finite_ok and out.numel():
                    absmax = float(
                        out.detach().abs().amax().item()
                    )
                else:
                    absmax = float("inf")

                max_abs = float(
                    os.environ.get(
                        "TRELLIS2_DINO_SAFE_GEMM_ABSMAX",
                        "10000",
                    )
                )

                if finite_ok and absmax <= max_abs:
                    return out

                print(
                    "[Pixal3D DINO GEMM CPU FALLBACK] "
                    f"{name}: gpu_finite={finite_ok}, "
                    f"gpu_absmax={absmax:.6g}; "
                    "recomputing this block on CPU"
                )

                out_cpu = torch.matmul(
                    a.detach().float().cpu(),
                    b.detach().float().cpu(),
                )
                if scale != 1.0:
                    out_cpu.mul_(float(scale))

                _require_safe(
                    f"{name}.cpu",
                    out_cpu,
                )

                cpu_absmax = (
                    float(out_cpu.abs().amax().item())
                    if out_cpu.numel()
                    else 0.0
                )
                if cpu_absmax > max_abs:
                    print(
                        "[Pixal3D DINO GEMM CPU FALLBACK] "
                        f"{name}: CPU result is finite but large "
                        f"(absmax={cpu_absmax:.6g}); keeping the "
                        "CPU result because it is numerically valid"
                    )

                return out_cpu.to(
                    device=a.device,
                    dtype=torch.float32,
                )

            def _safe_chunked_layer(
                layer_module,
                layer_input,
                pos,
                layer_idx,
            ):
                q_chunk = int(
                    os.environ.get(
                        "TRELLIS2_DINO_SAFE_QCHUNK",
                        "256",
                    )
                )

                token_chunk = int(
                    os.environ.get(
                        "TRELLIS2_DINO_SAFE_TOKEN_CHUNK",
                        "512",
                    )
                )

                # gfx1030/ROCm can return NaN/Inf (or silently enormous
                # finite garbage) from the 4101-wide attention GEMMs even
                # when Q/K/V are all finite.  Query chunking alone does not
                # change that K dimension.  Split the key/value axis too and
                # use an online-softmax reduction so no GEMM ever spans all
                # 4101 tokens.
                k_chunk = int(
                    os.environ.get(
                        "TRELLIS2_DINO_SAFE_KCHUNK",
                        "512",
                    )
                )

                norm_chunk = int(
                    os.environ.get(
                        "TRELLIS2_DINO_SAFE_NORM_CHUNK",
                        "64",
                    )
                )

                print(
                    "[Pixal3D DINO SAFE FALLBACK] "
                    f"layer={layer_idx}, "
                    f"q_chunk={q_chunk}, "
                    f"k_chunk={k_chunk}, "
                    f"token_chunk={token_chunk}, "
                    f"norm_chunk={norm_chunk}"
                )

                # ------------------------------
                # Attention half of DINO block
                # ------------------------------

                residual = layer_input.float()

                x = _safe_layer_norm_token_chunks(
                    layer_module.norm1,
                    residual,
                    norm_chunk,
                    f"layer_{layer_idx}.norm1",
                )

                attn = layer_module.attention

                q = _linear_token_chunks(
                    attn.q_proj,
                    x,
                    token_chunk,
                )

                k = _linear_token_chunks(
                    attn.k_proj,
                    x,
                    token_chunk,
                )

                v = _linear_token_chunks(
                    attn.v_proj,
                    x,
                    token_chunk,
                )

                B, N, C = q.shape

                H = attn.num_heads
                D = attn.head_dim

                q = (
                    q.view(B, N, H, D)
                    .transpose(1, 2)
                    .contiguous()
                )

                k = (
                    k.view(B, N, H, D)
                    .transpose(1, 2)
                    .contiguous()
                )

                v = (
                    v.view(B, N, H, D)
                    .transpose(1, 2)
                    .contiguous()
                )

                q, k = _apply_rope_local(
                    q,
                    k,
                    pos,
                )

                _require_safe(
                    f"layer_{layer_idx}.q_rope",
                    q,
                )

                _require_safe(
                    f"layer_{layer_idx}.k_rope",
                    k,
                )

                # Clamp chunks once we know the sequence length.  The K/V
                # chunk is the critical part of the gfx1030 workaround: both
                # QK^T and P@V now operate on at most k_chunk keys at a time.
                q_chunk = max(1, min(int(q_chunk), N))
                k_chunk = max(1, min(int(k_chunk), N))

                attn_parts = []

                scaling = float(
                    getattr(
                        attn,
                        "scaling",
                        D ** -0.5,
                    )
                )

                for start in range(
                    0,
                    N,
                    q_chunk,
                ):
                    end = min(
                        start + q_chunk,
                        N,
                    )

                    q_part = q[:, :, start:end, :]
                    q_len = int(end - start)

                    # Numerically stable online softmax over key blocks.
                    # This is mathematically equivalent to
                    # softmax(QK^T * scale) @ V, but never launches the
                    # problematic [Q, 4101] ROCm GEMM and never materializes
                    # the full attention matrix.
                    running_max = torch.full(
                        (B, H, q_len, 1),
                        -float("inf"),
                        device=q.device,
                        dtype=torch.float32,
                    )
                    running_sum = torch.zeros(
                        (B, H, q_len, 1),
                        device=q.device,
                        dtype=torch.float32,
                    )
                    running_out = torch.zeros(
                        (B, H, q_len, D),
                        device=q.device,
                        dtype=torch.float32,
                    )

                    for k_start in range(0, N, k_chunk):
                        k_end = min(k_start + k_chunk, N)

                        k_part = k[:, :, k_start:k_end, :]
                        v_part = v[:, :, k_start:k_end, :]

                        _score_name = (
                            "layer_"
                            f"{layer_idx}.scores."
                            f"q{start}:{end}.k{k_start}:{k_end}"
                        )
                        scores = _rocm_block_matmul(
                            q_part,
                            k_part.transpose(-2, -1).contiguous(),
                            _score_name,
                            scale=scaling,
                        )

                        _require_safe(
                            _score_name,
                            scores,
                        )

                        block_max = scores.amax(
                            dim=-1,
                            keepdim=True,
                        )
                        new_max = torch.maximum(
                            running_max,
                            block_max,
                        )

                        # exp(-inf - finite) is exactly zero on the first
                        # block, which is the desired contribution from the
                        # empty running accumulator.
                        old_scale = torch.exp(
                            running_max - new_max
                        )
                        block_exp = torch.exp(
                            scores - new_max
                        )

                        _require_safe(
                            "layer_"
                            f"{layer_idx}.exp."
                            f"q{start}:{end}.k{k_start}:{k_end}",
                            block_exp,
                        )

                        block_sum = block_exp.sum(
                            dim=-1,
                            keepdim=True,
                        )
                        _value_name = (
                            "layer_"
                            f"{layer_idx}.attn_block."
                            f"q{start}:{end}.k{k_start}:{k_end}"
                        )
                        block_out = _rocm_block_matmul(
                            block_exp,
                            v_part,
                            _value_name,
                        )

                        _require_safe(
                            _value_name,
                            block_out,
                        )

                        running_out = (
                            running_out * old_scale
                            + block_out
                        )
                        running_sum = (
                            running_sum * old_scale
                            + block_sum
                        )
                        running_max = new_max

                        del (
                            scores,
                            block_max,
                            new_max,
                            old_scale,
                            block_exp,
                            block_sum,
                            block_out,
                            k_part,
                            v_part,
                        )

                    part = running_out / running_sum.clamp_min(
                        torch.finfo(torch.float32).tiny
                    )

                    _require_safe(
                        "layer_"
                        f"{layer_idx}.attn."
                        f"{start}:{end}",
                        part,
                    )

                    attn_parts.append(part)

                    del running_max, running_sum, running_out, q_part

                x = torch.cat(
                    attn_parts,
                    dim=2,
                )

                del attn_parts
                del q, k, v

                x = (
                    x.transpose(1, 2)
                    .contiguous()
                    .view(B, N, C)
                )

                x = _linear_token_chunks(
                    attn.o_proj,
                    x,
                    token_chunk,
                )

                x = layer_module.layer_scale1(
                    x
                )

                x = (
                    layer_module.drop_path(x)
                    + residual
                )

                _require_safe(
                    f"layer_{layer_idx}.attn_residual",
                    x,
                )

                # ------------------------------
                # MLP half of DINO block
                # ------------------------------

                residual = x

                x = _safe_layer_norm_token_chunks(
                    layer_module.norm2,
                    x,
                    norm_chunk,
                    f"layer_{layer_idx}.norm2",
                )

                mlp_parts = []

                for start in range(
                    0,
                    N,
                    token_chunk,
                ):
                    end = min(
                        start + token_chunk,
                        N,
                    )

                    part = layer_module.mlp(
                        x[:, start:end]
                    ).float()

                    _require_safe(
                        "layer_"
                        f"{layer_idx}.mlp."
                        f"{start}:{end}",
                        part,
                    )

                    mlp_parts.append(part)

                x = torch.cat(
                    mlp_parts,
                    dim=1,
                )

                del mlp_parts

                x = layer_module.layer_scale2(
                    x
                )

                x = (
                    layer_module.drop_path(x)
                    + residual
                )

                _require_safe(
                    f"layer_{layer_idx}.final",
                    x,
                )

                return x

            safe_fallback_enabled = (
                is_rocm
                and debug_1024
                and os.environ.get(
                    "TRELLIS2_DINO_SAFE_FALLBACK",
                    "1",
                ) == "1"
            )

            # [Pixal3D DINO SAFE ALWAYS]
            #
            # gfx1030/ROCm: do not execute the full 4101-token
            # eager-attention kernel first. We have observed that
            # kernel corrupt different layers/views nondeterministically.
            #
            # Run the query-chunked FP32 implementation directly for
            # every 1024 DINO transformer layer.
            safe_always = (
                is_rocm
                and debug_1024
                and os.environ.get(
                    "TRELLIS2_DINO_SAFE_ALWAYS",
                    "1",
                ) == "1"
            )

            if safe_always:
                print(
                    "[Pixal3D DINO SAFE ALWAYS] "
                    "using chunked FP32 attention for all "
                    "1024 DINO layers"
                )

                # Snapshot every LayerNorm parameter BEFORE
                # running any long-sequence GPU kernels.
                for _layer in self.model.layer:
                    _ensure_norm_master(
                        _layer.norm1
                    )
                    _ensure_norm_master(
                        _layer.norm2
                    )

                if hasattr(
                    self.model,
                    "norm",
                ):
                    _ensure_norm_master(
                        self.model.norm
                    )

                print(
                    "[Pixal3D DINO SAFE NORM] "
                    "cached pristine CPU LayerNorm "
                    "parameters"
                )

            for layer_idx, layer_module in enumerate(
                self.model.layer
            ):
                if safe_always:
                    hidden_states = _safe_chunked_layer(
                        layer_module,
                        hidden_states,
                        position_embeddings,
                        layer_idx,
                    )

                    check_finite(
                        f"layer_{layer_idx}",
                        hidden_states,
                    )

                    continue

                # Reactive fallback retained for experimentation,
                # but preserve an independent copy because a failed
                # GPU kernel must not be allowed to poison recovery.
                layer_input = (
                    hidden_states.clone()
                    if safe_fallback_enabled
                    else hidden_states
                )

                hidden_states = layer_module(
                    hidden_states,
                    position_embeddings=position_embeddings,
                )

                layer_bad = (
                    _cpu_bad_count(hidden_states)
                    if debug_1024
                    else 0
                )

                if (
                    layer_bad
                    and safe_fallback_enabled
                ):
                    print(
                        "[Pixal3D DINO SAFE FALLBACK] "
                        f"layer_{layer_idx} normal path "
                        f"produced {layer_bad} bad values; "
                        "rerunning this layer safely"
                    )

                    torch.cuda.synchronize()

                    del hidden_states
                    torch.cuda.empty_cache()

                    hidden_states = _safe_chunked_layer(
                        layer_module,
                        layer_input,
                        position_embeddings,
                        layer_idx,
                    )

                    recovered_bad = _cpu_bad_count(
                        hidden_states
                    )

                    print(
                        "[Pixal3D DINO SAFE FALLBACK] "
                        f"layer_{layer_idx} recovery "
                        f"bad={recovered_bad}"
                    )

                    if recovered_bad:
                        raise RuntimeError(
                            "[Pixal3D DINO SAFE FALLBACK] "
                            f"layer_{layer_idx} could not "
                            "be recovered"
                        )

                check_finite(
                    f"layer_{layer_idx}",
                    hidden_states,
                )

            final_norm_chunk = int(
                os.environ.get(
                    "TRELLIS2_DINO_SAFE_NORM_CHUNK",
                    "64",
                )
            )

            if getattr(self, "_pixal3d_official_final_norm", False):
                # Match TencentARC/Pixal3D exactly.  This is intentionally
                # parameterless: no self.model.norm.weight/bias are applied.
                hidden_states = F.layer_norm(
                    hidden_states.float(),
                    hidden_states.shape[-1:],
                )
            elif hasattr(
                self.model,
                "norm",
            ):
                hidden_states = (
                    _safe_layer_norm_token_chunks(
                        self.model.norm,
                        hidden_states,
                        final_norm_chunk,
                        "final_layer_norm",
                    )
                )
            else:
                # Compatibility fallback only.
                final_parts = []

                for _s in range(
                    0,
                    hidden_states.shape[1],
                    final_norm_chunk,
                ):
                    _e = min(
                        _s + final_norm_chunk,
                        hidden_states.shape[1],
                    )

                    _x = (
                        hidden_states[:, _s:_e]
                        .contiguous()
                        .float()
                    )

                    _scale = _x.abs().amax(
                        dim=-1,
                        keepdim=True,
                    ).clamp_min(1.0)
                    _xs = _x / _scale
                    _mean = _xs.mean(
                        dim=-1,
                        keepdim=True,
                    )

                    _centered = (
                        _xs - _mean
                    )

                    _var = (
                        _centered
                        .square()
                        .mean(
                            dim=-1,
                            keepdim=True,
                        )
                    )

                    _inv_scale = _scale.reciprocal()
                    _part = (
                        _centered
                        * torch.rsqrt(
                            _var.clamp_min(0.0)
                            + 1e-5 * _inv_scale.square()
                        )
                    )

                    _require_safe(
                        "final_layer_norm."
                        f"{_s}:{_e}",
                        _part,
                    )

                    final_parts.append(
                        _part
                    )

                hidden_states = torch.cat(
                    final_parts,
                    dim=1,
                )

            check_finite(
                "final_layer_norm",
                hidden_states,
            )

        return hidden_states
    
    def forward(
        self,
        image: Union[torch.Tensor, List[Image.Image]],
        camera_angle_x: Optional[torch.Tensor] = None,
        distance: Optional[torch.Tensor] = None,
        mesh_scale: Optional[torch.Tensor] = None,
        transform_matrix: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract view-aligned features from the image.
        
        Args:
            image: Input image tensor [B, C, H, W] or list of PIL images
            camera_angle_x: Camera FOV angle in radians [B]
            distance: Camera distance [B]
            mesh_scale: Mesh scale factor [B]
            transform_matrix: Optional camera transform matrix [B, 4, 4]
        
        Returns:
            Tuple of (global_features, proj_features):
            - global_features: [B, num_global_tokens, embed_dim]
            - proj_features: [B, grid_resolution³, proj_channels]
              where proj_channels = embed_dim (no NAF) or embed_dim*2 (with NAF)
        """
        # Handle input types
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((self.image_size, self.image_size), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).cuda()
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        B = image.shape[0]
        
        # Keep a copy of the unnormalized image for NAF guide
        if self.use_naf_upsample:
            image_for_naf = image.clone()  # [B, 3, H, W], in [0, 1] range
        
        # Apply transform (ImageNet normalization)
        image = self.transform(image)
        
        # Extract DINOv3 features (frozen, no gradients)
        with torch.no_grad():
            z = self.extract_features(image)
            
            # Split into CLS token, register tokens, and patch tokens
            z_clstoken = z[:, 0:1]  # [B, 1, D]
            num_reg = getattr(self.model.config, 'num_register_tokens', 4)
            z_regtokens = z[:, 1:1+num_reg]  # [B, num_reg, D]
            z_patchtokens = z[:, 1+num_reg:]  # [B, num_patches, D]
            
            # Reshape patch tokens to spatial grid: [B, h, w, D]
            z_patchtokens_spatial = z_patchtokens.reshape(
                B, self.patch_number, self.patch_number, -1
            )  # [B, h, w, D]
            
            if camera_angle_x is None or distance is None or mesh_scale is None:
                raise ValueError("camera_angle_x, distance, and mesh_scale must be provided")
            
            # --- Low-resolution branch: sample from DINOv3 patch feature map ---
            z_proj_lr = self.proj_grid(
                z_patchtokens_spatial, 
                camera_angle_x, 
                distance, 
                mesh_scale,
                transform_matrix
            )  # [B, grid_res³, D]
            
            # --- High-resolution branch (NAF): upsample then sample ---
            if self.use_naf_upsample:
                self._load_naf(device=image_for_naf.device)
                # NAF expects: guide [B, 3, H, W], lr_features [B, C, h, w], target_size (H', W')
                lr_features_bchw = z_patchtokens_spatial.permute(0, 3, 1, 2)  # [B, D, h, w]

                K = getattr(self, 'naf_tile_factor', 1) or 1
                if K <= 1:
                    # Un-tiled path (default, unchanged behavior).
                    hr_features = self.naf_model(
                        image_for_naf, lr_features_bchw, self.naf_target_size
                    )  # [B, D, H', W']

                    # Sample from high-res feature map using same projection coordinates
                    z_proj_hr = self.proj_grid(
                        hr_features,
                        camera_angle_x,
                        distance,
                        mesh_scale,
                        transform_matrix,
                        BHWC=False  # hr_features is [B, C, H', W']
                    )  # [B, grid_res³, D]
                    del hr_features
                else:
                    # Tiled streaming path: bound the transient HR feature map by running
                    # NAF and the projection sample one image-space tile at a time.
                    z_proj_hr = self._proj_naf_tiled(
                        image_for_naf, lr_features_bchw,
                        camera_angle_x, distance, mesh_scale, transform_matrix,
                        tile_factor=int(K),
                    )

                # Concatenate lr and hr: [B, grid_res³, D*2]
                z_proj = torch.cat([z_proj_lr, z_proj_hr], dim=-1)
            else:
                z_proj = z_proj_lr  # [B, grid_res³, D]
                
            # Combine global tokens
            z_global = torch.cat([z_clstoken, z_regtokens], dim=1)  # [B, 1+num_reg, D]
        
        # proj_linear has been moved to per-block ProjectAttention
        # z_proj stays in proj_channels, each block will project independently
        
        return z_global, z_proj

    @torch.no_grad()
    def _proj_naf_tiled(
        self,
        image_for_naf: torch.Tensor,
        lr_features_bchw: torch.Tensor,
        camera_angle_x: torch.Tensor,
        distance: torch.Tensor,
        mesh_scale: torch.Tensor,
        transform_matrix: Optional[torch.Tensor],
        tile_factor: int,
    ) -> torch.Tensor:
        self._load_naf(device=image_for_naf.device)  # ensure NAF is colocated with inputs
        """Streaming NAF + grid_sample. Materialises only one image-space tile of the NAF
        output at a time, capping the transient HR-feature-map peak. Returns the same
        [B, grid_res^3, embed_dim] tensor that proj_grid(naf_model(...)) would produce in
        the un-tiled path.

        Each 3D grid point is routed to exactly one tile (the tile its projection falls in).
        Halo at the NAF output scale = kernel_size//2 (4 for kernel=9) + 2 px safety for the
        2-layer image encoder receptive field. Off-image grid points are clamped to the
        boundary tile so grid_sample's padding_mode='border' recovers the un-tiled border
        behaviour for them.

        adaptive_avg_pool inside NAF re-buckets when called per-tile, so the output is
        near-equivalent to (not bit-equivalent with) the un-tiled call. Boundary diffs are
        small in practice; if they manifest, fall back to naf_tile_factor=1.
        """
        K = int(tile_factor)
        naf_h, naf_w = self.naf_target_size  # always a 2-tuple of ints
        img_size = self.image_size
        img_res = self.proj_grid.image_resolution  # equals image_size for the configured stages

        # Halo at output (NAF) scale, then converted to image scale by the resolution ratio.
        # NAF: single CrossAttention with kernel_size=9 -> 4 px, plus 2 px safety for the
        # 2-conv image encoder. Negligible relative to tile size at K<=8.
        HALO_OUT = 6
        halo_img = max(1, int(round(HALO_OUT * img_size / max(naf_h, naf_w))))

        B = image_for_naf.shape[0]
        device = image_for_naf.device

        # Project all 3D grid points once. The output tensor is small (R^3 * 2 floats).
        grid_pts = self.proj_grid.grid_points.to(device)  # [R^3, 3]
        grid_pts_b = grid_pts.unsqueeze(0).expand(B, -1, -1)
        grid_pts_b = grid_pts_b / mesh_scale.unsqueeze(-1).unsqueeze(-1) / 2

        if transform_matrix is None:
            tm = self.proj_grid.front_view_transform_matrix.to(device)
            tm = tm.unsqueeze(0).expand(B, -1, -1).clone()
            tm[:, 1, 3] = -distance
        else:
            tm = transform_matrix

        image_points, _, _ = project_points_to_image_batch(
            grid_pts_b, tm, camera_angle_x, img_res
        )  # [B, R^3, 2] in pixel coords
        image_points_norm = (image_points + 0.5) / img_res * 2 - 1  # [B, R^3, 2] in [-1, 1]

        # Clamp off-image points to the boundary so they get routed to a boundary tile and
        # produce the same border-feature value as padding_mode='border' would in the
        # un-tiled call.
        image_points_norm = image_points_norm.clamp(-1.0, 1.0)

        u_global = image_points_norm[..., 0]  # [B, R^3]
        v_global = image_points_norm[..., 1]

        R3 = grid_pts.shape[0]
        embed_dim = lr_features_bchw.shape[1]
        z_proj_hr: Optional[torch.Tensor] = None

        naf_core = _real_naf_core(self.naf_model)
        if naf_core is None:
            # Bilinear stand-in (DummyNAFModel): one untiled call is cheap and exact.
            hr = self.naf_model(image_for_naf, lr_features_bchw, (naf_h, naf_w))
            return sample_features(hr, image_points_norm).transpose(1, 2).contiguous()

        # Exact row bands (see _naf_exact_bands). Each grid point goes to the one
        # band whose core rows contain its projection.
        for keep_lo, keep_hi, core_lo, core_hi, band_hr in _naf_exact_bands(
            naf_core, image_for_naf, lr_features_bchw, (naf_h, naf_w), K
        ):
            if os.environ.get("TRELLIS2_PIXAL3D_MV_STRICT_NAF", "1") == "1":
                # Counted on the host: the bytes that reach the conditioner.
                band_bad = int((~torch.isfinite(band_hr.cpu())).sum())
                if band_bad:
                    raise RuntimeError(
                        "[Pixal3D MV STRICT NAF] exact banded NAF produced "
                        f"{band_bad} NaN/Inf values in rows {keep_lo}-{keep_hi}; "
                        "refusing to substitute bilinear features because "
                        "that changes the conditioning semantics"
                    )

            v_lo = 2.0 * core_lo / naf_h - 1.0
            v_hi = 2.0 * core_hi / naf_h - 1.0
            last = core_hi == naf_h
            core_mask = (v_global >= v_lo) & ((v_global <= v_hi) if last else (v_global < v_hi))
            if not core_mask.any():
                continue

            if z_proj_hr is None:
                z_proj_hr = torch.zeros(B, R3, embed_dim, device=band_hr.device, dtype=band_hr.dtype)

            # Band bounds in the FULL feature map's [-1, 1] coords; full width.
            band_v_lo_n = 2.0 * keep_lo / naf_h - 1.0
            band_v_hi_n = 2.0 * keep_hi / naf_h - 1.0
            for b in range(B):
                mask_b = core_mask[b]
                if not mask_b.any():
                    continue
                pts = image_points_norm[b][mask_b]
                local_v = 2.0 * (pts[..., 1] - band_v_lo_n) / (band_v_hi_n - band_v_lo_n) - 1.0
                local_pts = torch.stack([pts[..., 0], local_v], dim=-1).unsqueeze(0)
                feats = sample_features(band_hr[b:b + 1], local_pts)
                z_proj_hr[b][mask_b] = feats.squeeze(0).transpose(0, 1).to(z_proj_hr.dtype)
            del band_hr

        if z_proj_hr is None:
            # No grid point projected on-image at all (degenerate). Return zeros — same shape
            # as the un-tiled call would produce.
            z_proj_hr = torch.zeros(
                B, R3, embed_dim,
                device=lr_features_bchw.device, dtype=lr_features_bchw.dtype,
            )
        return z_proj_hr

    @torch.no_grad()
    def visualize_projection(
        self,
        image: torch.Tensor,
        camera_angle_x: torch.Tensor,
        distance: torch.Tensor,
        mesh_scale: torch.Tensor,
        transform_matrix: Optional[torch.Tensor] = None,
        save_dir: Optional[str] = None,
        prefix: str = "proj_vis",
    ) -> List[Image.Image]:
        """
        Visualize the projected 3D grid points on the input image.
        
        This is a convenience method that delegates to ProjGrid.visualize_projection.
        
        Args:
            image: Input image tensor [B, C, H, W], in [0, 1] range (before ImageNet normalization)
            camera_angle_x: Camera FOV angle, shape [B]
            distance: Camera distance, shape [B]
            mesh_scale: Mesh scale factor, shape [B]
            transform_matrix: Optional camera transform matrix, shape [B, 4, 4]
            save_dir: Directory to save visualizations (optional)
            prefix: Prefix for saved files
            
        Returns:
            List of PIL Images with projected points overlaid
        """
        return self.proj_grid.visualize_projection(
            image=image,
            camera_angle_x=camera_angle_x,
            distance=distance,
            mesh_scale=mesh_scale,
            transform_matrix=transform_matrix,
            save_dir=save_dir,
            prefix=prefix,
        )


# =============================================================================
# DINOv3 + VAE Gated Feature Extractor with Projection
# =============================================================================

class DinoV3VaeProjFeatureExtractor(nn.Module):
    """
    DINOv3 + Flux VAE Feature Extractor with Gated Fusion and View-Aligned Projection.
    
    Produces three outputs for GatedProjectAttention:
    1. Global features (CLS + register tokens from DINOv3) for cross-attention
    2. Semantic proj features (DINOv3 patch tokens projected to 3D grid)
    3. Color proj features (Flux VAE latent projected to 3D grid)
    
    Both DINOv3 and VAE are frozen. The gated fusion happens inside each
    denoiser block's GatedProjectAttention module (trainable gate + proj_linears).
    
    Args:
        dino_model_name: Pretrained DINOv3 model name
        vae_model_name: Pretrained Flux VAE model name
        image_size: Input image size (default: 512)
        grid_resolution: Resolution of the 3D projection grid (default: 16)
    """
    def __init__(
        self,
        dino_model_name: str,
        vae_model_name: str = "black-forest-labs/FLUX.1-dev",
        image_size: int = 512,
        grid_resolution: int = 16,
    ):
        super().__init__()
        self.image_size = image_size
        self.grid_resolution = grid_resolution
        
        # --- DINOv3 backbone (frozen) ---
        self.dino_model = DINOv3ViTModel.from_pretrained(dino_model_name)
        self.dino_model.eval()
        self.dino_model.requires_grad_(False)
        
        self.dino_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.patch_size = self.dino_model.config.patch_size
        self.patch_number = image_size // self.patch_size
        self.embed_dim = self.dino_model.config.hidden_size  # e.g. 1024
        
        # --- Flux VAE encoder (frozen, lazy-loaded) ---
        self.vae_model_name = vae_model_name
        self._vae = None
        self.vae_channels = 16  # Flux VAE outputs 16 channels
        self.vae_downsample = 8  # Flux VAE downsamples by 8x
        
        # --- Projection grid (shared) ---
        self.proj_grid = ProjGrid(
            grid_resolution=grid_resolution,
            image_resolution=image_size,
        )
        
        # Expose dimensions for denoiser block construction
        self.dino_proj_channels = self.embed_dim      # e.g. 1024
        self.vae_proj_channels = self.vae_channels     # 16
        # proj_channels is kept for backward compat with _proj_channels in mixin
        self.proj_channels = self.embed_dim
        
    def _load_vae(self):
        """Lazy-load Flux VAE encoder."""
        if self._vae is not None:
            return
        from diffusers import AutoencoderKL
        device = next(self.dino_model.parameters()).device
        vae = AutoencoderKL.from_pretrained(
            self.vae_model_name,
            subfolder="vae",
            torch_dtype=torch.float32,
        )
        vae.eval()
        vae.requires_grad_(False)
        vae.to(device)
        self._vae = vae
    
    def to(self, device):
        super().to(device)
        self.dino_model.to(device)
        self.proj_grid.to(device)
        if self._vae is not None:
            self._vae.to(device)
        return self
    
    def cuda(self):
        super().cuda()
        self.dino_model.cuda()
        self.proj_grid.cuda()
        if self._vae is not None:
            self._vae.cuda()
        return self
    
    def cpu(self):
        super().cpu()
        self.dino_model.cpu()
        self.proj_grid.cpu()
        if self._vae is not None:
            self._vae.cpu()
        return self
    
    def _extract_dino_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extract DINOv3 features from normalized image."""
        image = image.to(self.dino_model.embeddings.patch_embeddings.weight.dtype)
        hidden_states = self.dino_model.embeddings(image, bool_masked_pos=None)
        position_embeddings = self.dino_model.rope_embeddings(image)
        for layer_module in self.dino_model.layer:
            hidden_states = layer_module(
                hidden_states,
                position_embeddings=position_embeddings,
            )
        return F.layer_norm(hidden_states, hidden_states.shape[-1:])
    
    @torch.no_grad()
    def _extract_vae_latent(self, image: torch.Tensor) -> torch.Tensor:
        """Extract Flux VAE latent from unnormalized image [0,1]."""
        self._load_vae()
        image_normalized = image * 2.0 - 1.0
        image_normalized = image_normalized.to(self._vae.dtype)
        posterior = self._vae.encode(image_normalized)
        latent = posterior.latent_dist.mode()
        latent = latent * self._vae.config.scaling_factor
        return latent.float()  # [B, 16, H/8, W/8]
    
    def forward(
        self,
        image: Union[torch.Tensor, List[Image.Image]],
        camera_angle_x: Optional[torch.Tensor] = None,
        distance: Optional[torch.Tensor] = None,
        mesh_scale: Optional[torch.Tensor] = None,
        transform_matrix: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract gated features from the image.
        
        Returns:
            Tuple of (global_features, proj_semantic, proj_color):
            - global_features: [B, num_global_tokens, embed_dim] (DINOv3 CLS + registers)
            - proj_semantic: [B, grid_res³, embed_dim] (DINOv3 projected features)
            - proj_color: [B, grid_res³, vae_channels] (VAE projected features)
        """
        # Handle input types
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image)
            image = [i.resize((self.image_size, self.image_size), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).cuda()
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        B = image.shape[0]
        image_raw = image.clone()  # Keep unnormalized copy for VAE
        
        if camera_angle_x is None or distance is None or mesh_scale is None:
            raise ValueError("camera_angle_x, distance, and mesh_scale must be provided")
        
        with torch.no_grad():
            # --- DINOv3 branch ---
            dino_input = self.dino_transform(image)
            z = self._extract_dino_features(dino_input)
            
            z_clstoken = z[:, 0:1]
            num_reg = getattr(self.dino_model.config, 'num_register_tokens', 4)
            z_regtokens = z[:, 1:1+num_reg]
            z_patchtokens = z[:, 1+num_reg:]
            
            z_patchtokens_spatial = z_patchtokens.reshape(
                B, self.patch_number, self.patch_number, -1
            )  # [B, h, w, D]
            
            proj_semantic = self.proj_grid(
                z_patchtokens_spatial,
                camera_angle_x, distance, mesh_scale, transform_matrix,
            )  # [B, R³, embed_dim]
            
            z_global = torch.cat([z_clstoken, z_regtokens], dim=1)  # [B, 1+num_reg, D]
            
            # --- VAE branch ---
            vae_latent = self._extract_vae_latent(image_raw)  # [B, 16, H/8, W/8]
            
            proj_color = self.proj_grid(
                vae_latent,
                camera_angle_x, distance, mesh_scale, transform_matrix,
                BHWC=False,  # VAE latent is [B, C, H, W]
            )  # [B, R³, 16]
        
        return z_global, proj_semantic, proj_color


# =============================================================================
# Image Conditioned Mixin with Projection Support
# =============================================================================

class ImageConditionedProjMixin:
    """
    Mixin for image-conditioned models with view-aligned projection.
    
    This mixin adds support for extracting view-aligned features from images
    using camera parameters.
    
    Args:
        image_cond_model: Configuration for the image conditioning model.
    """
    def __init__(self, *args, image_cond_model: dict, **kwargs):
        # Store config before super().__init__ which calls init_models_and_more
        self.image_cond_model_config = image_cond_model
        self.image_cond_model = None  # Will be initialized in init_models_and_more
        self.image_attn_mode = image_cond_model.get('image_attn_mode', 
                                image_cond_model.get('args', {}).get('image_attn_mode', 'cross'))
        super().__init__(*args, **kwargs)
        
    def _init_image_cond_model(self):
        """Initialize the image conditioning model."""
        with dist_utils.local_master_first():
            model_name = self.image_cond_model_config['name']
            model_args = self.image_cond_model_config.get('args', {})
            
            if model_name == 'DinoV3ProjFeatureExtractor':
                self.image_cond_model = DinoV3ProjFeatureExtractor(**model_args)
            elif model_name == 'DinoV3VaeProjFeatureExtractor':
                self.image_cond_model = DinoV3VaeProjFeatureExtractor(**model_args)
            else:
                # Fallback to standard extractors
                from . import image_conditioned
                self.image_cond_model = getattr(image_conditioned, model_name)(**model_args)
            
            self.image_cond_model.cuda()
            
            # Expose proj_channels for denoiser to know the correct proj_in_channels
            if hasattr(self.image_cond_model, 'proj_channels'):
                self._proj_channels = self.image_cond_model.proj_channels
            else:
                self._proj_channels = getattr(self.image_cond_model, 'embed_dim', None)
            # Expose vae_proj_channels for gated_proj mode
            self._vae_proj_channels = getattr(self.image_cond_model, 'vae_proj_channels', None)
    
    def init_models_and_more(self, **kwargs):
        """
        Override to handle image_cond_model initialization.
        
        Since proj_linear has been moved to per-block ProjectAttention in the denoiser,
        image_cond_model no longer has any trainable parameters (DINOv3 backbone is frozen,
        ProjGrid only has register_buffers). Therefore we do NOT add it to self.models
        (which would trigger DDP wrapping and fail). We just initialize it and keep it
        as a standalone module for inference.
        """
        # Initialize image_cond_model first
        if self.image_cond_model is None:
            self._init_image_cond_model()
        
        # Keep a reference to the unwrapped module for attribute access
        self._image_cond_module = self.image_cond_model  # for .grid_resolution etc.
        
        # Log that image_cond has no trainable params
        proj_params = [p for p in self.image_cond_model.parameters() if p.requires_grad]
        if self.is_master:
            if proj_params:
                print(f'\nWARNING: image_cond_model has {len(proj_params)} trainable params, '
                      f'but is NOT registered in self.models. These will NOT be trained!')
            else:
                print(f'\nimage_cond_model has no trainable parameters, skipping DDP/optimizer registration.')
        
        # Call base class to set up DDP, optimizer, EMA, etc. (without image_cond)
        super().init_models_and_more(**kwargs)

    # ------------------------------------------------------------------
    # Checkpoint save/load overrides: skip DINOv3 backbone weights
    # ------------------------------------------------------------------

    # Keys in image_cond state_dict that belong to the frozen DINOv3 backbone.
    # Everything under "model." is DINOv3; we only keep proj_grid.*
    _IMAGE_COND_BACKBONE_PREFIX = 'model.'

    def _filter_image_cond_state_dict(self, state_dict: dict) -> dict:
        """Keep only non-backbone keys (proj_grid, etc.) from image_cond state_dict."""
        return {k: v for k, v in state_dict.items()
                if not k.startswith(self._IMAGE_COND_BACKBONE_PREFIX)}

    def _fill_denoiser_proj_linear_from_image_cond(
        self,
        denoiser_ckpt: dict,
        denoiser_state_dict: dict,
        image_cond_ckpt_path: Optional[str] = None,
    ) -> dict:
        """
        Fill missing per-block proj_linear weights in denoiser checkpoint
        from the old-style image_cond proj_linear (broadcast to all blocks).
        
        Also handles shape mismatch when NAF is enabled: old proj_linear has shape
        [model_ch, embed_dim] but new model expects [model_ch, embed_dim*2].
        In this case, the old weights are placed in the lr half and the hr half is zero-padded.
        
        Compatibility strategy:
        1. If denoiser_ckpt already contains per-block proj_linear keys with correct shape -> do nothing.
        2. If shape mismatch (embed_dim vs embed_dim*2) -> zero-pad the weight.
        3. If keys missing, try to load proj_linear from image_cond checkpoint -> broadcast (with optional pad).
        
        Args:
            denoiser_ckpt: The loaded denoiser state dict
            denoiser_state_dict: The model's current state dict (to find expected keys)
            image_cond_ckpt_path: Path to image_cond checkpoint file (optional)
            
        Returns:
            Updated denoiser_ckpt with proj_linear keys filled if needed
        """
        if self.image_attn_mode != 'proj':
            return denoiser_ckpt
        
        # Find all per-block proj_linear keys expected by the model
        proj_linear_keys = [k for k in denoiser_state_dict.keys()
                            if '.cross_attn.proj_linear.' in k]
        if not proj_linear_keys:
            return denoiser_ckpt
        
        # --- Phase 1: Handle shape mismatch for existing keys (NAF upgrade) ---
        for k in proj_linear_keys:
            if k in denoiser_ckpt:
                expected_shape = denoiser_state_dict[k].shape
                actual_shape = denoiser_ckpt[k].shape
                if expected_shape != actual_shape:
                    if k.endswith('.weight') and len(expected_shape) == 2:
                        # Weight shape: [out_features, in_features]
                        # Old: [model_ch, embed_dim], New: [model_ch, embed_dim*2]
                        out_f, new_in_f = expected_shape
                        _, old_in_f = actual_shape
                        if new_in_f > old_in_f and out_f == actual_shape[0]:
                            if self.is_master:
                                print(f'\n  [NAF Compat] Padding proj_linear weight {k}: '
                                      f'{actual_shape} -> {expected_shape} (zero-pad hr half)')
                            new_w = torch.zeros(expected_shape, dtype=denoiser_ckpt[k].dtype,
                                                device=denoiser_ckpt[k].device)
                            new_w[:, :old_in_f] = denoiser_ckpt[k]
                            denoiser_ckpt[k] = new_w
                        else:
                            if self.is_master:
                                print(f'\n  Warning: proj_linear {k} shape mismatch '
                                      f'{actual_shape} vs {expected_shape}, using model init')
                            denoiser_ckpt[k] = denoiser_state_dict[k]
                    # bias shape should match (out_features only), no padding needed
        
        # --- Phase 2: Handle completely missing keys ---
        missing_proj_keys = [k for k in proj_linear_keys if k not in denoiser_ckpt]
        if not missing_proj_keys:
            return denoiser_ckpt
        
        if self.is_master:
            print(f'\n  [Compat] Denoiser ckpt missing {len(missing_proj_keys)} per-block proj_linear keys.')
            print(f'           Attempting to load from image_cond proj_linear: {image_cond_ckpt_path}')
        
        # Try to find proj_linear weights from image_cond checkpoint
        old_proj_linear_w = None
        old_proj_linear_b = None
        
        if image_cond_ckpt_path is not None:
            import os as _os
            if _os.path.exists(image_cond_ckpt_path):
                try:
                    ic_ckpt = torch.load(image_cond_ckpt_path, map_location=self.device, weights_only=True)
                    old_proj_linear_w = ic_ckpt.get('proj_linear.weight')
                    old_proj_linear_b = ic_ckpt.get('proj_linear.bias')
                except Exception as e:
                    if self.is_master:
                        print(f'           Warning: Failed to load image_cond ckpt: {e}')
        
        if old_proj_linear_w is None:
            raise RuntimeError(
                f'Denoiser checkpoint is missing per-block proj_linear keys '
                f'(e.g. {missing_proj_keys[0]}), and no image_cond proj_linear '
                f'was found to broadcast from. Cannot proceed.'
            )
        
        if self.is_master:
            print(f'           Found image_cond proj_linear: weight {old_proj_linear_w.shape}, bias {old_proj_linear_b.shape}')
            print(f'           Broadcasting to {len(missing_proj_keys)} per-block keys...')
        
        for k in missing_proj_keys:
            if k.endswith('.weight'):
                expected_shape = denoiser_state_dict[k].shape
                if expected_shape != old_proj_linear_w.shape:
                    # Pad for NAF: [model_ch, embed_dim] -> [model_ch, embed_dim*2]
                    out_f, new_in_f = expected_shape
                    _, old_in_f = old_proj_linear_w.shape
                    if new_in_f > old_in_f and out_f == old_proj_linear_w.shape[0]:
                        new_w = torch.zeros(expected_shape, dtype=old_proj_linear_w.dtype)
                        new_w[:, :old_in_f] = old_proj_linear_w
                        denoiser_ckpt[k] = new_w
                    else:
                        denoiser_ckpt[k] = denoiser_state_dict[k]
                else:
                    denoiser_ckpt[k] = old_proj_linear_w.clone()
            elif k.endswith('.bias'):
                denoiser_ckpt[k] = old_proj_linear_b.clone()
        
        return denoiser_ckpt

    def _master_params_to_state_dicts(self, master_params):
        """Override to skip image_cond checkpoint entirely.
        
        image_cond model no longer has trainable parameters:
        - proj_linear has been moved to per-block ProjectAttention in the denoiser
        - DINOv3 backbone is frozen and loaded from pretrained weights
        - ProjGrid only contains fixed register_buffers (grid_points, front_view_transform_matrix)
        So there is nothing worth saving for image_cond.
        """
        state_dicts = super()._master_params_to_state_dicts(master_params)
        state_dicts.pop('image_cond', None)
        return state_dicts

    def load(self, load_dir, step=0):
        """
        Override to handle:
        1. Old checkpoints that don't have image_cond_step*.pt
        2. Partial image_cond checkpoints (only proj_linear + proj_grid, no DINOv3 backbone)
        """
        import os as _os

        if self.is_master:
            print(f'\nLoading checkpoint from step {step}...', end='')

        model_ckpts = {}
        for name, model in self.models.items():
            ckpt_path = _os.path.join(load_dir, 'ckpts', f'{name}_step{step:07d}.pt')

            if name == 'image_cond':
                # --- handle missing or partial image_cond checkpoint ---
                if not _os.path.exists(ckpt_path):
                    if self.is_master:
                        print(f'\n  image_cond checkpoint not found at {ckpt_path}, using freshly initialised weights.')
                    model_ckpts[name] = model.state_dict()
                    continue

                try:
                    model_ckpt = torch.load(
                        read_file_dist(ckpt_path),
                        map_location=self.device, weights_only=True)
                except Exception as e:
                    if self.is_master:
                        print(f'\n  Failed to load image_cond checkpoint: {e}. Using freshly initialised weights.')
                    model_ckpts[name] = model.state_dict()
                    continue

                # Partial ckpt (no backbone) → load with strict=False
                missing, unexpected = model.load_state_dict(model_ckpt, strict=False)
                # All missing keys should be the frozen DINOv3 backbone; verify
                non_backbone_missing = [k for k in missing
                                        if not k.startswith(self._IMAGE_COND_BACKBONE_PREFIX)]
                if non_backbone_missing and self.is_master:
                    print(f'\n  Warning: unexpected missing keys in image_cond ckpt: {non_backbone_missing}')
                if unexpected and self.is_master:
                    print(f'\n  Warning: unexpected keys in image_cond ckpt: {unexpected}')

                # Build a full state_dict for master_params sync
                full_sd = model.state_dict()
                full_sd.update(model_ckpt)
                model_ckpts[name] = full_sd
            else:
                model_ckpt = torch.load(
                    read_file_dist(ckpt_path),
                    map_location=self.device, weights_only=True)
                # For denoiser: handle old ckpts missing per-block proj_linear
                if name == 'denoiser':
                    ic_ckpt_path = _os.path.join(load_dir, 'ckpts', f'image_cond_step{step:07d}.pt')
                    model_ckpt = self._fill_denoiser_proj_linear_from_image_cond(
                        model_ckpt, model.state_dict(), ic_ckpt_path)
                model_ckpts[name] = model_ckpt
                model.load_state_dict(model_ckpt)

        self._state_dicts_to_master_params(self.master_params, model_ckpts)
        del model_ckpts

        if self.is_master:
            for i, ema_rate in enumerate(self.ema_rate):
                ema_ckpts = {}
                for name, model in self.models.items():
                    ema_path = _os.path.join(
                        load_dir, 'ckpts',
                        f'{name}_ema{ema_rate}_step{step:07d}.pt')
                    if name == 'image_cond':
                        if not _os.path.exists(ema_path):
                            ema_ckpts[name] = model.state_dict()
                            continue
                        try:
                            ema_ckpt = torch.load(ema_path, map_location=self.device, weights_only=True)
                        except Exception:
                            ema_ckpts[name] = model.state_dict()
                            continue
                        full_sd = model.state_dict()
                        full_sd.update(ema_ckpt)
                        ema_ckpts[name] = full_sd
                    else:
                        ema_ckpt = torch.load(ema_path, map_location=self.device, weights_only=True)
                        if name == 'denoiser':
                            ic_ema_path = _os.path.join(
                                load_dir, 'ckpts',
                                f'image_cond_ema{ema_rate}_step{step:07d}.pt')
                            ema_ckpt = self._fill_denoiser_proj_linear_from_image_cond(
                                ema_ckpt, model.state_dict(), ic_ema_path)
                        ema_ckpts[name] = ema_ckpt
                self._state_dicts_to_master_params(self.ema_params[i], ema_ckpts)
                del ema_ckpts

        misc_ckpt = torch.load(
            read_file_dist(_os.path.join(load_dir, 'ckpts', f'misc_step{step:07d}.pt')),
            map_location=torch.device('cpu'), weights_only=False)
        # Optimizer state may mismatch when loading old checkpoints that were
        # saved before image_cond was added to self.models, or when the number
        # of trainable parameters changed (e.g. backbone freeze, NAF upgrade).
        # In that case we skip restoring optimizer state and let it re-initialise.
        try:
            self.optimizer.load_state_dict(misc_ckpt['optimizer'])
            # Verify optimizer state shapes match parameters.
            # load_state_dict may succeed even when shapes mismatch (keys are
            # integer indices), causing a crash later in optimizer.step().
            _shape_ok = True
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    state = self.optimizer.state.get(p)
                    if state is not None:
                        for sv in state.values():
                            if isinstance(sv, torch.Tensor) and sv.shape != () and sv.shape != p.shape:
                                _shape_ok = False
                                break
                    if not _shape_ok:
                        break
                if not _shape_ok:
                    break
            if not _shape_ok:
                if self.is_master:
                    print(f'\n  Warning: optimizer state shape mismatch (likely NAF upgrade). '
                          f'Optimizer will start fresh.')
                self.optimizer.state.clear()
        except (ValueError, RuntimeError) as e:
            if self.is_master:
                print(f'\n  Warning: could not load optimizer state ({e}). '
                      f'Optimizer will start fresh.')
        self.step = misc_ckpt['step']
        self.data_sampler.load_state_dict(misc_ckpt['data_sampler'])
        if self.mix_precision_mode == 'amp' and self.mix_precision_dtype == torch.float16:
            self.scaler.load_state_dict(misc_ckpt['scaler'])
        elif self.mix_precision_mode == 'inflat_all' and self.mix_precision_dtype == torch.float16:
            self.log_scale = misc_ckpt['log_scale']
        if self.lr_scheduler_config is not None:
            self.lr_scheduler.load_state_dict(misc_ckpt['lr_scheduler'])
        if self.elastic_controller_config is not None:
            self.elastic_controller.load_state_dict(misc_ckpt['elastic_controller'])
        if self.grad_clip is not None and not isinstance(self.grad_clip, float):
            self.grad_clip.load_state_dict(misc_ckpt['grad_clip'])
        del misc_ckpt

        if self.world_size > 1:
            dist.barrier()
        if self.is_master:
            print(' Done.')

        if self.world_size > 1:
            self.check_ddp()

    def finetune_from(self, finetune_ckpt):
        """
        Override to tolerate DINOv3 backbone keys missing from image_cond checkpoint.
        For image_cond, the checkpoint only stores proj_linear + proj_grid (no backbone),
        so we treat all backbone keys as allowed-missing.
        """
        ALLOWED_MISSING_KEYS = {'rope_phases'}

        if self.is_master:
            print('\nFinetuning from:')
            for name, path in finetune_ckpt.items():
                print(f'  - {name}: {path}')

        model_ckpts = {}
        for name, model in self.models.items():
            model_state_dict = model.state_dict()
            if name in finetune_ckpt:
                model_ckpt = torch.load(
                    read_file_dist(finetune_ckpt[name]),
                    map_location=self.device, weights_only=True)

                model_ckpt = self._remap_checkpoint_keys(model_ckpt, model_state_dict)

                for k, v in model_ckpt.items():
                    if k not in model_state_dict:
                        if self.is_master:
                            print(f'Warning: {k} not found in model_state_dict, skipped.')
                        model_ckpt[k] = None
                    elif model_ckpt[k].shape != model_state_dict[k].shape:
                        # For proj_linear weights, try zero-pad instead of skipping
                        # This handles NAF upgrade: [model_ch, embed_dim] -> [model_ch, embed_dim*2]
                        if '.cross_attn.proj_linear.weight' in k and len(model_ckpt[k].shape) == 2:
                            old_shape = model_ckpt[k].shape
                            new_shape = model_state_dict[k].shape
                            if new_shape[0] == old_shape[0] and new_shape[1] > old_shape[1]:
                                if self.is_master:
                                    print(f'Info: Zero-padding proj_linear weight {k}: {old_shape} -> {new_shape}')
                                new_w = torch.zeros(new_shape, dtype=model_ckpt[k].dtype)
                                new_w[:, :old_shape[1]] = model_ckpt[k]
                                model_ckpt[k] = new_w
                            else:
                                if self.is_master:
                                    print(f'Warning: {k} shape mismatch, {old_shape} vs {new_shape}, skipped.')
                                model_ckpt[k] = model_state_dict[k]
                        else:
                            if self.is_master:
                                print(f'Warning: {k} shape mismatch, {model_ckpt[k].shape} vs {model_state_dict[k].shape}, skipped.')
                            model_ckpt[k] = model_state_dict[k]
                model_ckpt = {k: v for k, v in model_ckpt.items() if v is not None}

                missing_keys = set(model_state_dict.keys()) - set(model_ckpt.keys())

                # For denoiser: fill per-block proj_linear from image_cond if missing
                if name == 'denoiser':
                    ic_path = finetune_ckpt.get('image_cond')
                    proj_linear_missing = {k for k in missing_keys if '.cross_attn.proj_linear.' in k}
                    if proj_linear_missing:
                        model_ckpt = self._fill_denoiser_proj_linear_from_image_cond(
                            model_ckpt, model_state_dict, ic_path)
                        # Recalculate missing_keys after filling
                        missing_keys = set(model_state_dict.keys()) - set(model_ckpt.keys())

                # For image_cond, DINOv3 backbone keys are expected to be missing
                allowed = set(ALLOWED_MISSING_KEYS)
                if name == 'image_cond':
                    backbone_missing = {k for k in missing_keys
                                        if k.startswith(self._IMAGE_COND_BACKBONE_PREFIX)}
                    allowed |= backbone_missing
                    if backbone_missing and self.is_master:
                        print(f'Info: image_cond: {len(backbone_missing)} DINOv3 backbone keys '
                              f'not in ckpt (expected, using pretrained weights)')
                    # Old ckpts may have proj_linear.* which has moved to denoiser blocks
                    proj_linear_missing = {k for k in missing_keys if k.startswith('proj_linear.')}
                    allowed |= proj_linear_missing

                unexpected_missing = missing_keys - allowed
                if unexpected_missing and self.is_master:
                    print(f'Error: Missing keys in checkpoint: {unexpected_missing}')
                    raise RuntimeError(f'Missing keys in checkpoint: {unexpected_missing}')
                if missing_keys & ALLOWED_MISSING_KEYS and self.is_master:
                    print(f'Info: Using model initialized values for: {missing_keys & ALLOWED_MISSING_KEYS}')

                for k in missing_keys:
                    model_ckpt[k] = model_state_dict[k]

                model_ckpts[name] = model_ckpt
                model.load_state_dict(model_ckpt)
            else:
                if self.is_master:
                    print(f'Warning: {name} not found in finetune_ckpt, skipped.')
                model_ckpts[name] = model_state_dict

        self._state_dicts_to_master_params(self.master_params, model_ckpts)
        if self.is_master:
            for i, ema_rate in enumerate(self.ema_rate):
                self._state_dicts_to_master_params(self.ema_params[i], model_ckpts)
        del model_ckpts

        if self.world_size > 1:
            dist.barrier()
        if self.is_master:
            print('Done.')

        if self.world_size > 1:
            self.check_ddp()

    def encode_image_proj(
        self,
        image: torch.Tensor,
        camera_angle_x: Optional[torch.Tensor] = None,
        distance: Optional[torch.Tensor] = None,
        mesh_scale: Optional[torch.Tensor] = None,
        transform_matrix: Optional[torch.Tensor] = None,
        coords: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Encode the image with view-aligned projection.
        
        Supports both 'proj' mode (DINOv3 only, 2 outputs) and 
        'gated_proj' mode (DINOv3 + VAE, 3 outputs).
        """
        if self.image_cond_model is None:
            self._init_image_cond_model()
        
        outputs = self.image_cond_model(
            image,
            camera_angle_x=camera_angle_x,
            distance=distance,
            mesh_scale=mesh_scale,
            transform_matrix=transform_matrix,
        )
        
        is_gated = self.image_attn_mode == 'gated_proj'
        
        if is_gated:
            cond_global, cond_proj_semantic, cond_proj_color = outputs
        else:
            cond_global, cond_proj = outputs
        
        # If coords provided, extract features at sparse positions
        if coords is not None:
            B = cond_global.shape[0]
            module = getattr(self, '_image_cond_module', self.image_cond_model)
            grid_res = module.grid_resolution
            batch_indices = coords[:, 0].long()
            x_coords = coords[:, 1].long()
            y_coords = coords[:, 2].long()
            z_coords = coords[:, 3].long()
            
            if is_gated:
                cond_proj_semantic = cond_proj_semantic.reshape(B, grid_res, grid_res, grid_res, -1)
                cond_proj_semantic = cond_proj_semantic[batch_indices, x_coords, y_coords, z_coords]
                cond_proj_color = cond_proj_color.reshape(B, grid_res, grid_res, grid_res, -1)
                cond_proj_color = cond_proj_color[batch_indices, x_coords, y_coords, z_coords]
            else:
                cond_proj = cond_proj.reshape(B, grid_res, grid_res, grid_res, -1)
                cond_proj = cond_proj[batch_indices, x_coords, y_coords, z_coords]
        
        if is_gated:
            cond = {
                'global': cond_global,
                'proj_semantic': cond_proj_semantic,
                'proj_color': cond_proj_color,
            }
            uncond = {
                'global': torch.zeros_like(cond_global),
                'proj_semantic': torch.zeros_like(cond_proj_semantic),
                'proj_color': torch.zeros_like(cond_proj_color),
            }
        else:
            cond = {'global': cond_global, 'proj': cond_proj}
            uncond = {'global': torch.zeros_like(cond_global), 'proj': torch.zeros_like(cond_proj)}
        
        return cond, uncond
    
    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        """
        Encode the image (standard mode without projection).
        """
        if self.image_cond_model is None:
            self._init_image_cond_model()
        
        if self.image_attn_mode == 'proj':
            # For proj mode, return dict
            global_feat, proj_feat = self.image_cond_model(image)
            return {'global': global_feat, 'proj': proj_feat}
        else:
            # Standard mode
            features = self.image_cond_model(image)
            return features
        
    def _extract_camera_info(self, kwargs):
        """
        Extract camera info from kwargs.
        
        Supports two formats:
        1. 'camera_info' dict: {'camera_angle_x': ..., 'distance': ..., 'mesh_scale': ..., 'transform_matrix': ..., 'coords': ...}
        2. Flat fields: 'camera_angle_x', 'camera_distance', 'mesh_scale', 'transform_matrix', 'coords' in kwargs
        
        Returns:
            camera_info dict or None if not available
        """
        if 'camera_info' in kwargs:
            return kwargs.pop('camera_info')
        
        # Try to extract from flat fields (as returned by ViewImageConditionedMixin)
        camera_angle_x = kwargs.pop('camera_angle_x', None)
        camera_distance = kwargs.pop('camera_distance', None)
        mesh_scale = kwargs.pop('mesh_scale', None)
        transform_matrix = kwargs.pop('transform_matrix', None)
        coords = kwargs.pop('coords', None)
        
        if camera_angle_x is not None and camera_distance is not None and mesh_scale is not None:
            return {
                'camera_angle_x': camera_angle_x,
                'distance': camera_distance,
                'mesh_scale': mesh_scale,
                'transform_matrix': transform_matrix,
                'coords': coords,
            }
        
        return None
        
    def get_cond(self, cond, **kwargs):
        """Get the conditioning data."""
        kwargs.pop('view_idx', None)
        
        if self.image_attn_mode in ('proj', 'gated_proj'):
            # Handle projection mode (both standard proj and gated_proj)
            camera_info = self._extract_camera_info(kwargs)
            if camera_info is not None:
                coords = camera_info.get('coords')
                cond, neg_cond = self.encode_image_proj(
                    cond,
                    camera_angle_x=camera_info.get('camera_angle_x'),
                    distance=camera_info.get('distance'),
                    mesh_scale=camera_info.get('mesh_scale'),
                    transform_matrix=camera_info.get('transform_matrix'),
                    coords=coords,
                )
                
                # For sparse mode (coords provided), handle CFG dropout ourselves
                if coords is not None and hasattr(self, 'p_uncond') and self.p_uncond > 0:
                    import numpy as np
                    B = cond['global'].shape[0]
                    mask = np.random.rand(B) < self.p_uncond
                    
                    global_tensor = cond['global']
                    global_mask_shape = [B] + [1] * (global_tensor.ndim - 1)
                    global_mask = torch.tensor(mask, device=global_tensor.device).reshape(global_mask_shape)
                    cond['global'] = torch.where(global_mask, neg_cond['global'], cond['global'])
                    
                    batch_indices = coords[:, 0].long()
                    # Handle all sparse proj keys (proj, or proj_semantic + proj_color)
                    for key in list(cond.keys()):
                        if key.startswith('proj'):
                            device = cond[key].device
                            sparse_mask = torch.tensor(mask, device=device)[batch_indices].reshape(-1, 1)
                            cond[key] = torch.where(sparse_mask, neg_cond[key], cond[key])
                    
                    return cond
                else:
                    kwargs['neg_cond'] = neg_cond
            else:
                cond = self.encode_image(cond)
                if isinstance(cond, dict) and 'global' in cond:
                    kwargs['neg_cond'] = {k: torch.zeros_like(v) for k, v in cond.items()}
                else:
                    kwargs['neg_cond'] = torch.zeros_like(cond)
        else:
            cond = self.encode_image(cond)
            kwargs['neg_cond'] = torch.zeros_like(cond)
            
        cond = super().get_cond(cond, **kwargs)
        return cond
    
    def get_inference_cond(self, cond, **kwargs):
        """Get the conditioning data for inference."""
        kwargs.pop('view_idx', None)
        
        if self.image_attn_mode in ('proj', 'gated_proj'):
            camera_info = self._extract_camera_info(kwargs)
            if camera_info is not None:
                cond, neg_cond = self.encode_image_proj(
                    cond,
                    camera_angle_x=camera_info.get('camera_angle_x'),
                    distance=camera_info.get('distance'),
                    mesh_scale=camera_info.get('mesh_scale'),
                    transform_matrix=camera_info.get('transform_matrix'),
                    coords=camera_info.get('coords'),
                )
                kwargs['neg_cond'] = neg_cond
            else:
                cond = self.encode_image(cond)
                if isinstance(cond, dict) and 'global' in cond:
                    kwargs['neg_cond'] = {k: torch.zeros_like(v) for k, v in cond.items()}
                else:
                    kwargs['neg_cond'] = torch.zeros_like(cond)
        else:
            cond = self.encode_image(cond)
            kwargs['neg_cond'] = torch.zeros_like(cond)
            
        cond = super().get_inference_cond(cond, **kwargs)
        return cond

    def vis_cond(self, cond, **kwargs):
        """Visualize the conditioning data."""
        return {'image': {'value': cond, 'type': 'image'}}

    @torch.no_grad()
    def visualize_projection_test(
        self,
        cond: torch.Tensor,
        save_dir: str,
        prefix: str = "proj_vis",
        **kwargs
    ) -> Optional[List[Image.Image]]:
        """
        Visualize projection points on the condition images.
        
        This should be called once before training starts to verify the projection is correct.
        
        Args:
            cond: Condition image tensor [B, C, H, W], in [0, 1] range
            save_dir: Directory to save visualizations
            prefix: Prefix for saved files
            **kwargs: Should contain camera_angle_x, camera_distance, mesh_scale, transform_matrix
            
        Returns:
            List of PIL Images with projected points overlaid, or None if not in proj mode
        """
        if self.image_attn_mode != 'proj':
            return None
            
        if self.image_cond_model is None:
            self._init_image_cond_model()
        
        # Use _image_cond_module for attribute access (image_cond_model may be DDP-wrapped)
        module = getattr(self, '_image_cond_module', self.image_cond_model)
        
        # Check if the model has visualization capability
        if not hasattr(module, 'visualize_projection'):
            print("Warning: image_cond_model does not support visualize_projection")
            return None
        
        # Extract camera info
        camera_info = self._extract_camera_info(kwargs)
        if camera_info is None:
            print("Warning: No camera info available for projection visualization")
            return None
        
        return module.visualize_projection(
            image=cond,
            camera_angle_x=camera_info.get('camera_angle_x'),
            distance=camera_info.get('distance'),
            mesh_scale=camera_info.get('mesh_scale'),
            transform_matrix=camera_info.get('transform_matrix'),
            save_dir=save_dir,
            prefix=prefix,
        )


# ============================================================================
# Pixal3D MultiView compatibility
# Backported from current ComfyUI-Trellis2 / Pixal3D MultiView implementation.
# ============================================================================

class ProjGridMV(ProjGrid):
    """
    Multi-view version of ProjGrid.

    Unlike the old ProjGrid implementation this allows an explicit camera
    transform for every view.
    """

    def forward(
        self,
        features_map: torch.Tensor,
        camera_angle_x: torch.Tensor,
        distance: torch.Tensor,
        mesh_scale: torch.Tensor,
        transform_matrix: Optional[torch.Tensor] = None,
        BHWC: bool = True,
    ) -> torch.Tensor:
        """
        Pixal3D MV projection.

        On ROCm/gfx1030, bypass F.grid_sample for the huge
        R^3 projected point set. We observed sporadic NaN
        values from that path even when DINO/NAF features
        and camera data were finite.

        Manual bilinear sampling below implements
        align_corners=False + padding_mode='border'.
        """
        if BHWC:
            B, H, W, C = features_map.shape
        else:
            B, C, H, W = features_map.shape

        is_rocm = (
            getattr(torch.version, "hip", None)
            is not None
        )

        # Keep projection math in FP32.
        features_map = features_map.float()

        # [Pixal3D pre-sampler CPU check]
        fmap_finite = torch.isfinite(
            features_map
        )

        fmap_finite_count = int(
            fmap_finite
            .detach()
            .cpu()
            .sum()
            .item()
        )

        fmap_bad = (
            features_map.numel()
            - fmap_finite_count
        )

        if fmap_bad:
            raise RuntimeError(
                "[Pixal3D MV ROCm sampler] "
                f"INPUT FEATURE MAP has {fmap_bad} "
                "real NaN/Inf values BEFORE projection "
                "(CPU-verified)"
            )

        grid_points = (
            self.grid_points
            .float()
            .expand(B, -1, -1)
        )

        grid_points = (
            grid_points
            / mesh_scale.float()
                .unsqueeze(-1)
                .unsqueeze(-1)
            / 2.0
        )

        if transform_matrix is None:
            transform_matrix = (
                self.front_view_transform_matrix
                .float()
                .expand(B, -1, -1)
                .clone()
            )

            transform_matrix[:, 1, 3] = (
                -distance.float()
            )
        else:
            transform_matrix = (
                transform_matrix.float()
            )

        image_points, depth, valid_mask = (
            project_points_to_image_batch(
                grid_points,
                transform_matrix,
                camera_angle_x.float(),
                self.image_resolution,
            )
        )

        image_points_norm = (
            (image_points + 0.5)
            / float(self.image_resolution)
            * 2.0
            - 1.0
        )

        coord_finite = torch.isfinite(
            image_points_norm
        )

        coord_finite_count = int(
            coord_finite
            .detach()
            .cpu()
            .sum()
            .item()
        )

        coord_bad = (
            image_points_norm.numel()
            - coord_finite_count
        )

        if coord_bad:
            print(
                "[Pixal3D MV ROCm sampler] "
                f"projection coordinates contain "
                f"{coord_bad} REAL NaN/Inf values "
                "(CPU-verified); sanitizing"
            )

        # Always execute this. It is a no-op for finite values,
        # and removes dependence on another ROCm reduction result.
        image_points_norm = torch.nan_to_num(
            image_points_norm,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )

        if BHWC:
            fmap = (
                features_map
                .permute(0, 3, 1, 2)
                .contiguous()
            )
        else:
            fmap = features_map.contiguous()

        if not is_rocm:
            # Preserve upstream implementation elsewhere.
            return sample_features(
                fmap,
                image_points_norm,
            ).permute(0, 2, 1)

        # =====================================================
        # ROCm-safe manual bilinear sampler.
        #
        # Equivalent to:
        #
        # F.grid_sample(
        #     fmap,
        #     grid,
        #     mode="bilinear",
        #     align_corners=False,
        #     padding_mode="border",
        # )
        #
        # but avoids gfx1030 grid_sample instability.
        # =====================================================

        q = image_points_norm.float()

        # padding_mode='border'
        q = q.clamp(
            min=-1.0,
            max=1.0,
        )

        # align_corners=False conversion:
        #
        # pixel_x = ((ndc_x + 1) * W - 1) / 2
        # pixel_y = ((ndc_y + 1) * H - 1) / 2
        #
        px = (
            ((q[..., 0] + 1.0) * W - 1.0)
            * 0.5
        )

        py = (
            ((q[..., 1] + 1.0) * H - 1.0)
            * 0.5
        )

        # Border behavior is equivalent to clamping the
        # continuous pixel location.
        px = px.clamp(
            0.0,
            float(W - 1),
        )

        py = py.clamp(
            0.0,
            float(H - 1),
        )

        x0 = torch.floor(px).long()
        y0 = torch.floor(py).long()

        x1 = (x0 + 1).clamp(
            max=W - 1,
        )

        y1 = (y0 + 1).clamp(
            max=H - 1,
        )

        wx = px - x0.float()
        wy = py - y0.float()

        fmap_flat = fmap.reshape(
            B,
            C,
            H * W,
        )

        K = q.shape[1]

        chunk = int(
            os.environ.get(
                "TRELLIS2_MV_PROJ_CHUNK",
                "32768",
            )
        )

        outputs = []

        for start in range(0, K, chunk):
            stop = min(
                start + chunk,
                K,
            )

            x0c = x0[:, start:stop]
            x1c = x1[:, start:stop]
            y0c = y0[:, start:stop]
            y1c = y1[:, start:stop]

            wxc = wx[:, start:stop]
            wyc = wy[:, start:stop]

            i00 = y0c * W + x0c
            i10 = y0c * W + x1c
            i01 = y1c * W + x0c
            i11 = y1c * W + x1c

            def gather(index):
                return torch.gather(
                    fmap_flat,
                    2,
                    index.unsqueeze(1).expand(
                        -1,
                        C,
                        -1,
                    ),
                )

            v00 = gather(i00)
            v10 = gather(i10)
            v01 = gather(i01)
            v11 = gather(i11)

            wxc = wxc.unsqueeze(1)
            wyc = wyc.unsqueeze(1)

            top = (
                v00 * (1.0 - wxc)
                + v10 * wxc
            )

            bottom = (
                v01 * (1.0 - wxc)
                + v11 * wxc
            )

            out_chunk = (
                top * (1.0 - wyc)
                + bottom * wyc
            )

            outputs.append(
                out_chunk
            )

        sampled = torch.cat(
            outputs,
            dim=2,
        )

        out = sampled.permute(
            0,
            2,
            1,
        ).contiguous()

        # [Pixal3D finite CPU]
        #
        # Do NOT reduce a huge bool tensor on gfx1030/ROCm here.
        # We observed two consecutive finite.sum().item() calls
        # disagreeing for the exact same mask.
        #
        # Compute isfinite elementwise on GPU, copy the mask, then
        # perform the reduction once on CPU.
        finite = torch.isfinite(out)
        finite_cpu = finite.detach().cpu()

        finite_count = int(
            finite_cpu.sum().item()
        )

        bad = (
            out.numel()
            - finite_count
        )

        print(
            "[Pixal3D MV ROCm sampler] "
            f"manual bilinear: "
            f"B={B}, C={C}, H={H}, W={W}, "
            f"K={K}, chunk={chunk}, "
            f"finite={finite_count}/"
            f"{out.numel()}, "
            f"bad={bad} "
            f"[CPU-reduced]"
        )

        if bad:
            raise RuntimeError(
                "[Pixal3D MV ROCm sampler] "
                f"manual projection has {bad} "
                "real non-finite values "
                "(CPU-verified)"
            )

        del finite_cpu

        return out

def compute_relative_calc_mat(
    transform_matrix: torch.Tensor,
    distance: torch.Tensor,
    front_view_transform_matrix: torch.Tensor,
) -> torch.Tensor:
    """
    calc_mat_i = F @ inv(C_0) @ C_i

    This converts every camera into the coordinate frame of the main/front view.
    """

    B, V = transform_matrix.shape[:2]
    device = transform_matrix.device

    F_mat = (
        front_view_transform_matrix
        .to(device)
        .unsqueeze(0)
        .expand(B, -1, -1)
        .clone()
    )

    F_mat[:, 1, 3] = -distance[:, 0]
    F_mat = F_mat.unsqueeze(1)

    C0 = transform_matrix[:, 0:1]

    # ROCm linalg.inv should remain FP32.
    with torch.amp.autocast("cuda", enabled=False):

        C0f = (
            C0.float()
            .expand(B, V, 4, 4)
            .reshape(B * V, 4, 4)
        )

        Cif = (
            transform_matrix.float()
            .reshape(B * V, 4, 4)
        )

        Ff = (
            F_mat.float()
            .expand(B, V, 4, 4)
            .reshape(B * V, 4, 4)
        )

        relative = torch.bmm(
            torch.linalg.inv(C0f),
            Cif,
        )

        calc = torch.bmm(
            Ff,
            relative,
        )

    return calc.reshape(B, V, 4, 4)


class DinoV3ProjMultiViewFeatureExtractor(
    DinoV3ProjFeatureExtractor
):
    """
    Pixal3D camera-aware multi-view DINOv3 projected conditioner.

    Inputs:
        image            [B,V,3,H,W]
        camera_angle_x   [B,V]
        distance         [B,V]
        mesh_scale       [B]
        transform_matrix [B,V,4,4]

    average fusion keeps the projected feature layout compatible with the
    existing Pixal3D denoisers/GGUF models.
    """

    def __init__(
        self,
        *args,
        multiview_fusion: str = "average",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if multiview_fusion not in (
            "average",
            "attention",
        ):
            raise ValueError(
                "multiview_fusion must be "
                "'average' or 'attention'"
            )

        self.multiview_fusion = multiview_fusion

        # The official Pixal3D MV checkpoints were trained with the projected
        # DINO extractor from TencentARC/Pixal3D, whose extract_features() ends
        # with parameterless F.layer_norm rather than DINO model.norm.  Earlier
        # ROCm patches accidentally matched this fork's learned-norm behavior
        # instead of the released Pixal3D conditioner semantics.
        self._pixal3d_official_final_norm = (
            os.environ.get(
                "TRELLIS2_PIXAL3D_OFFICIAL_DINO_NORM",
                "1",
            ) == "1"
        )
        if self._pixal3d_official_final_norm:
            print(
                "[Pixal3D MV DINO] official final normalization: "
                "parameterless F.layer_norm (no learned DINO affine)"
            )

        self.proj_grid = ProjGridMV(
            grid_resolution=self.grid_resolution,
            image_resolution=self.image_size,
        )


    def _project_naf_cpu_bilinear_fallback(
        self,
        lr_features_bchw,
        camera_angle_x_v,
        distance_v,
        mesh_scale_v,
        calc_mat_v,
    ):
        """ROCm correctness fallback for a corrupted real-NAF output.

        The real NAF/NATTEN path on gfx1030 can return a 1 GiB 512x512 feature
        map with millions of NaN/Inf values.  Repairing that map in-place on
        the GPU proved unsafe: later ROCm allocations/kernels can corrupt an
        already-verified channel block again.

        Instead, never construct or repair the giant fallback map on the GPU.
        Upsample the finite DINO LR features on CPU in channel chunks, project
        each chunk on CPU, concatenate the much smaller projected tensor, then
        copy only that projected tensor back to the caller's device.
        """
        import torch.nn.functional as F

        output_device = lr_features_bchw.device
        target = self.naf_target_size
        if isinstance(target, (tuple, list)):
            H_hr, W_hr = int(target[0]), int(target[1])
        else:
            H_hr = W_hr = int(target)

        channel_chunk = max(
            1,
            int(
                os.environ.get(
                    "TRELLIS2_MV_NAF_CPU_FALLBACK_CHUNK",
                    os.environ.get("TRELLIS2_MV_NAF_REPAIR_CHUNK", "32"),
                )
            ),
        )

        lr_cpu = lr_features_bchw.detach().float().cpu().contiguous()
        if not bool(torch.isfinite(lr_cpu).all().item()):
            bad = lr_cpu.numel() - int(torch.isfinite(lr_cpu).sum().item())
            raise RuntimeError(
                "[Pixal3D MV NAF CPU fallback] LR DINO feature map contains "
                f"{bad} NaN/Inf values"
            )

        # ProjGridMV only owns small coordinate buffers. Move those to CPU so
        # projection and manual bilinear sampling cannot touch ROCm while the
        # fallback is being built.
        try:
            proj_device = self.proj_grid.grid_points.device
        except Exception:
            proj_device = output_device

        self.proj_grid.cpu()
        projected_parts = []

        try:
            camera_cpu = camera_angle_x_v.detach().float().cpu()
            distance_cpu = distance_v.detach().float().cpu()
            mesh_scale_cpu = mesh_scale_v.detach().float().cpu()
            calc_cpu = (
                calc_mat_v.detach().float().cpu()
                if calc_mat_v is not None
                else None
            )

            C = int(lr_cpu.shape[1])
            for c0 in range(0, C, channel_chunk):
                c1 = min(c0 + channel_chunk, C)

                fallback_cpu = F.interpolate(
                    lr_cpu[:, c0:c1],
                    size=(H_hr, W_hr),
                    mode="bilinear",
                    align_corners=False,
                ).contiguous()

                fallback_bad = fallback_cpu.numel() - int(
                    torch.isfinite(fallback_cpu).sum().item()
                )
                if fallback_bad:
                    raise RuntimeError(
                        "[Pixal3D MV NAF CPU fallback] interpolated channels "
                        f"{c0}:{c1} contain {fallback_bad} NaN/Inf values"
                    )

                projected_cpu = self.proj_grid(
                    fallback_cpu,
                    camera_cpu,
                    distance_cpu,
                    mesh_scale_cpu,
                    calc_cpu,
                    BHWC=False,
                ).float().contiguous()

                projected_bad = projected_cpu.numel() - int(
                    torch.isfinite(projected_cpu).sum().item()
                )
                if projected_bad:
                    raise RuntimeError(
                        "[Pixal3D MV NAF CPU fallback] projected channels "
                        f"{c0}:{c1} contain {projected_bad} NaN/Inf values"
                    )

                projected_parts.append(projected_cpu)
                print(
                    "[Pixal3D MV NAF CPU fallback] "
                    f"channels={c0}:{c1}, projected finite"
                )

                del fallback_cpu

            z_proj_hr_cpu = torch.cat(projected_parts, dim=-1).contiguous()
            del projected_parts

            final_bad = z_proj_hr_cpu.numel() - int(
                torch.isfinite(z_proj_hr_cpu).sum().item()
            )
            if final_bad:
                raise RuntimeError(
                    "[Pixal3D MV NAF CPU fallback] final CPU projection contains "
                    f"{final_bad} NaN/Inf values"
                )

            print(
                "[Pixal3D MV NAF CPU fallback] complete: "
                f"shape={tuple(z_proj_hr_cpu.shape)}, bad=0; "
                f"copying projected tensor -> {output_device}"
            )

            return z_proj_hr_cpu.to(
                device=output_device,
                dtype=torch.float32,
                non_blocking=False,
            )
        finally:
            self.proj_grid.to(proj_device)

    def _project_single_view(
        self,
        z_patchtokens_spatial,
        image_for_naf,
        camera_angle_x_v,
        distance_v,
        mesh_scale_v,
        calc_mat_v,
    ):
        """
        Project one Pixal3D MV view.

        ROCm/gfx1030 fix:
        - keep DINO projection in FP32
        - force NAF model/input to FP32
        - disable autocast around NAF
        - if the pure-torch NATTEN fallback still creates
          non-finite HR features, substitute the corresponding
          upsampled finite DINO feature instead of propagating
          NaN into the 1024 shape-flow model.
        """
        import torch.nn.functional as F

        z_patchtokens_spatial = (
            z_patchtokens_spatial.float()
        )

        z_proj_lr = self.proj_grid(
            z_patchtokens_spatial,
            camera_angle_x_v.float(),
            distance_v.float(),
            mesh_scale_v.float(),
            calc_mat_v.float(),
        ).float()

        lr_finite = torch.isfinite(
            z_proj_lr
        )

        lr_finite_count = int(
            lr_finite
            .detach()
            .cpu()
            .sum()
            .item()
        )

        lr_bad = (
            z_proj_lr.numel()
            - lr_finite_count
        )

        if lr_bad:
            raise RuntimeError(
                "[Pixal3D MV NAF finite] "
                f"DINO LR projection has {lr_bad} "
                "real non-finite values "
                "(CPU-verified)"
            )

        if not self.use_naf_upsample:
            return z_proj_lr

        lr_features_bchw = (
            z_patchtokens_spatial
            .permute(0, 3, 1, 2)
            .contiguous()
            .float()
        )

        image_for_naf = (
            image_for_naf
            .contiguous()
            .float()
        )

        is_rocm = (
            getattr(torch.version, "hip", None)
            is not None
        )

        strict_official_naf = (
            os.environ.get(
                "TRELLIS2_PIXAL3D_MV_STRICT_NAF",
                "1",
            ) == "1"
        )
        strict_rocm_1024_naf = (
            strict_official_naf
            and is_rocm
            and int(getattr(self, "image_size", 0)) >= 1024
        )

        force_cpu_fallback = (
            is_rocm
            and not strict_rocm_1024_naf
            and (
                getattr(self, "_rocm_naf_force_cpu_fallback", False)
                or (
                    int(getattr(self, "image_size", 0)) >= 1024
                    and os.environ.get(
                        "TRELLIS2_MV_NAF_1024_CPU_FALLBACK",
                        "1",
                    ) == "1"
                )
            )
        )

        if strict_rocm_1024_naf and not getattr(
            self, "_trellis2_strict_naf1024_logged", False
        ):
            print(
                "[Pixal3D MV STRICT NAF] 1024 ROCm conditioner: "
                "bilinear semantic fallback DISABLED; running real NAF "
                "in spatial tiles and failing hard on NaN/Inf"
            )
            self._trellis2_strict_naf1024_logged = True

        # gfx1030 correctness mode: the real NAF/NATTEN 1024 path has been
        # observed to return ~49% NaN/Inf while its DINO input is fully finite.
        # Skip that known-bad GPU kernel by default for 1024 conditioners.
        if (
            force_cpu_fallback
            and int(getattr(self, "image_size", 0)) >= 1024
            and not getattr(self, "_trellis2_naf1024_cpu_logged", False)
        ):
            print(
                "[Pixal3D MV NAF CPU fallback] gfx1030 1024 correctness mode: "
                "skipping real NAF/NATTEN; using CPU bilinear projection"
            )
            self._trellis2_naf1024_cpu_logged = True

        # Once real NAF has emitted a corrupt feature map, do not move it back
        # to the GPU on subsequent views.  The CPU fallback does not need NAF.
        if not force_cpu_fallback:
            self._load_naf(device=image_for_naf.device)

        if is_rocm and not force_cpu_fallback:
            # RX 6800 / gfx1030:
            # do not leave NAF parameters in BF16.
            self.naf_model.float()

        requested_K = getattr(
            self,
            "naf_tile_factor",
            1,
        ) or 1

        # Strict Pixal3D MV correctness mode must not replace the learned NAF
        # conditioner with a bilinear resize.  On ROCm, run the real NAF in
        # spatial tiles instead.  The tile factor is deliberately independent
        # of the old TRELLIS2_MV_NAF_TILING=0 launcher knob so old launchers
        # cannot silently re-enable the geometry-destroying approximation.
        if strict_rocm_1024_naf:
            K = max(
                2,
                int(os.environ.get(
                    "TRELLIS2_MV_NAF_1024_TILE_FACTOR",
                    "4",
                )),
            )
            self.naf_tile_factor = K
        elif (
            is_rocm
            and os.environ.get(
                "TRELLIS2_MV_NAF_TILING",
                "0",
            ) == "0"
        ):
            K = 1
            self.naf_tile_factor = 1
        else:
            K = requested_K

        # CPU fallback is intentionally non-tiled.  It already bounds peak
        # memory by channel chunks, and must not enter _proj_naf_tiled() when
        # real NAF was never loaded.
        if force_cpu_fallback:
            K = 1
            self.naf_tile_factor = 1

        print(
            "[Pixal3D MV NAF] "
            f"tile_factor requested={requested_K}, "
            f"effective={K}"
        )

        if K <= 1:
            if force_cpu_fallback:
                print(
                    "[Pixal3D MV NAF CPU fallback] "
                    "real NAF was disabled after an earlier non-finite output; "
                    "using deterministic CPU bilinear projection"
                )
                z_proj_hr = self._project_naf_cpu_bilinear_fallback(
                    lr_features_bchw,
                    camera_angle_x_v,
                    distance_v,
                    mesh_scale_v,
                    calc_mat_v,
                )
            else:
                # Explicitly suppress autocast around real NAF.
                with torch.autocast(
                    device_type="cuda",
                    enabled=False,
                ):
                    hr_features = self.naf_model(
                        image_for_naf,
                        lr_features_bchw,
                        self.naf_target_size,
                    )

                hr_features = hr_features.float()

                # Count on CPU in channel chunks.  Do not trust a giant ROCm
                # reduction when the NAF output itself may already be corrupt.
                repair_chunk = max(
                    1,
                    int(
                        os.environ.get(
                            "TRELLIS2_MV_NAF_REPAIR_CHUNK",
                            "32",
                        )
                    ),
                )

                B_hr, C_hr, H_hr, W_hr = hr_features.shape
                hr_finite_count = 0
                for c0 in range(0, C_hr, repair_chunk):
                    c1 = min(c0 + repair_chunk, C_hr)
                    chunk_mask_cpu = (
                        torch.isfinite(hr_features[:, c0:c1])
                        .detach()
                        .cpu()
                    )
                    hr_finite_count += int(chunk_mask_cpu.sum().item())
                    del chunk_mask_cpu

                hr_bad = hr_features.numel() - hr_finite_count
                print(
                    "[Pixal3D MV NAF finite] "
                    f"NAF output shape={tuple(hr_features.shape)}, "
                    f"finite={hr_finite_count}/{hr_features.numel()}, "
                    f"bad={hr_bad} [CPU-reduced]"
                )

                auto_cpu_fallback = (
                    is_rocm
                    and os.environ.get(
                        "TRELLIS2_MV_NAF_AUTO_CPU_FALLBACK",
                        "1",
                    ) == "1"
                )

                if hr_bad and auto_cpu_fallback:
                    bad_ratio = float(hr_bad) / float(hr_features.numel())
                    print(
                        "[Pixal3D MV NAF CPU fallback] "
                        f"real NAF returned {hr_bad} NaN/Inf values "
                        f"({bad_ratio:.2%}); discarding the corrupt 512x512 map "
                        "instead of repairing it in-place on ROCm"
                    )

                    # Once a fresh real-NAF invocation has produced a corrupt
                    # result, do not run it again for the remaining views of
                    # this conditioner instance.  The source LR DINO features
                    # are finite, so the deterministic bilinear projection is
                    # a safe semantic fallback already supported by this file.
                    self._rocm_naf_force_cpu_fallback = True

                    del hr_features
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    try:
                        self.naf_model.cpu()
                    except Exception as _naf_offload_error:
                        print(
                            "[Pixal3D MV NAF CPU fallback] "
                            "NAF CPU offload warning: "
                            + repr(_naf_offload_error)
                        )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    z_proj_hr = self._project_naf_cpu_bilinear_fallback(
                        lr_features_bchw,
                        camera_angle_x_v,
                        distance_v,
                        mesh_scale_v,
                        calc_mat_v,
                    )
                elif hr_bad:
                    raise RuntimeError(
                        "[Pixal3D MV NAF finite] real NAF returned "
                        f"{hr_bad} NaN/Inf values and automatic CPU fallback "
                        "is disabled (TRELLIS2_MV_NAF_AUTO_CPU_FALLBACK=0)"
                    )
                else:
                    z_proj_hr = self.proj_grid(
                        hr_features,
                        camera_angle_x_v.float(),
                        distance_v.float(),
                        mesh_scale_v.float(),
                        calc_mat_v.float(),
                        BHWC=False,
                    ).float()
                    del hr_features

        else:
            # Retain existing tiled implementation for non-ROCm
            # or explicit experimentation. Your launcher uses K=1.
            z_proj_hr = self._proj_naf_tiled(
                image_for_naf,
                lr_features_bchw,
                camera_angle_x_v.float(),
                distance_v.float(),
                mesh_scale_v.float(),
                calc_mat_v.float(),
                tile_factor=int(K),
            ).float()

        proj_mask = torch.isfinite(
            z_proj_hr
        )

        proj_finite_count = int(
            proj_mask
            .detach()
            .cpu()
            .sum()
            .item()
        )

        proj_bad = (
            z_proj_hr.numel()
            - proj_finite_count
        )

        print(
            "[Pixal3D MV NAF finite] "
            f"HR projection finite="
            f"{proj_finite_count}/"
            f"{z_proj_hr.numel()}, "
            f"bad={proj_bad} "
            f"[CPU-reduced]"
        )

        if proj_bad:
            # Both branches use the same semantic channels and
            # projection coordinates. Use LR projection only at
            # corrupted HR elements.
            z_proj_hr = torch.where(
                proj_mask,
                z_proj_hr,
                z_proj_lr,
            )

            print(
                "[Pixal3D MV NAF finite] "
                f"repaired {proj_bad} projected "
                "HR values from LR projection"
            )

        out = torch.cat(
            [
                z_proj_lr,
                z_proj_hr,
            ],
            dim=-1,
        )

        combined_mask = torch.isfinite(
            out
        )

        combined_finite_count = int(
            combined_mask
            .detach()
            .cpu()
            .sum()
            .item()
        )

        combined_bad = (
            out.numel()
            - combined_finite_count
        )

        print(
            "[Pixal3D MV NAF finite] "
            f"combined finite="
            f"{combined_finite_count}/"
            f"{out.numel()}, "
            f"bad={combined_bad} "
            f"[CPU-reduced]"
        )

        if combined_bad:
            raise RuntimeError(
                "[Pixal3D MV NAF finite] "
                f"combined projection has "
                f"{combined_bad} real non-finite "
                "values (CPU-verified)"
            )

        return out

    def forward(
        self,
        image: torch.Tensor,
        camera_angle_x: Optional[torch.Tensor] = None,
        distance: Optional[torch.Tensor] = None,
        mesh_scale: Optional[torch.Tensor] = None,
        transform_matrix: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if image.ndim != 5:
            raise ValueError(
                "Pixal3D MultiView expects "
                f"[B,V,3,H,W], got {tuple(image.shape)}"
            )

        B, V = image.shape[:2]

        if (
            camera_angle_x is None
            or distance is None
            or mesh_scale is None
            or transform_matrix is None
        ):
            raise ValueError(
                "Pixal3D MultiView requires "
                "camera_angle_x, distance, mesh_scale "
                "and transform_matrix"
            )

        calc_mat = compute_relative_calc_mat(
            transform_matrix,
            distance,
            self.proj_grid.front_view_transform_matrix,
        )


        # ========================================================
        # Pixal3D MV runtime camera validation.
        # ========================================================
        import os as _os

        if _os.environ.get(
            "TRELLIS2_MV_CAMERA_DEBUG", "0"
        ) == "1":

            _tm = transform_matrix.detach().float().cpu()
            _dist = distance.detach().float().cpu()
            _fov = camera_angle_x.detach().float().cpu()
            _calc = calc_mat.detach().float().cpu()

            print(
                "[Pixal3D MV camera-debug] "
                f"B={B}, V={V}"
            )

            print(
                "[Pixal3D MV camera-debug] FOV deg="
                + str(torch.rad2deg(_fov[0]).tolist())
            )

            print(
                "[Pixal3D MV camera-debug] distance="
                + str(_dist[0].tolist())
            )

            try:
                _ms = mesh_scale.detach().float().cpu()
                print(
                    "[Pixal3D MV camera-debug] mesh_scale="
                    + str(_ms.tolist())
                )
            except Exception:
                print(
                    "[Pixal3D MV camera-debug] mesh_scale="
                    + str(mesh_scale)
                )

            _eyes = _tm[0, :, :3, 3]

            print(
                "[Pixal3D MV camera-debug] camera positions:"
            )

            for _i in range(V):
                print(
                    f"  C{_i} eye={_eyes[_i].tolist()}"
                )

            for _i in range(V):
                _R = _tm[0, _i, :3, :3]

                _orth_err = float(
                    (
                        _R.T @ _R
                        - torch.eye(3)
                    )
                    .abs()
                    .max()
                    .item()
                )

                _det = float(
                    torch.det(_R).item()
                )

                print(
                    "[Pixal3D MV camera-debug] "
                    f"C{_i}: det={_det:.6f}, "
                    f"orth_err={_orth_err:.3e}"
                )

            _F = (
                self.proj_grid
                .front_view_transform_matrix
                .detach()
                .float()
                .cpu()
                .clone()
            )

            _F[1, 3] = -_dist[0, 0]

            _front_err = float(
                (_tm[0, 0] - _F)
                .abs()
                .max()
                .item()
            )

            print(
                "[Pixal3D MV camera-debug] "
                f"canonical front max_err={_front_err:.6e}"
            )

            print(
                "[Pixal3D MV camera-debug] "
                "transform matrices:"
            )

            for _i in range(V):
                print(f"  C[{_i}]=")
                print(_tm[0, _i])

            print(
                "[Pixal3D MV camera-debug] "
                "relative calc matrices:"
            )

            for _i in range(V):
                print(f"  calc[{_i}]=")
                print(_calc[0, _i])

            if V == 4:
                _d = float(_dist[0, 0])

                _expected = torch.tensor(
                    [
                        [0.0, -_d, 0.0],
                        [_d, 0.0, 0.0],
                        [0.0, _d, 0.0],
                        [-_d, 0.0, 0.0],
                    ],
                    dtype=torch.float32,
                )

                _eye_err = (
                    _eyes - _expected
                ).abs().max(dim=1).values

                print(
                    "[Pixal3D MV camera-debug] "
                    "expected orbit order=az0,az90,az180,az270"
                )

                print(
                    "[Pixal3D MV camera-debug] "
                    "orbit eye max_err="
                    + str(_eye_err.tolist())
                )

            if _os.environ.get(
                "TRELLIS2_MV_CAMERA_DEBUG_STOP", "0"
            ) == "1":
                raise RuntimeError(
                    "[Pixal3D MV camera-debug] "
                    "debug-only stop requested"
                )

        # Run DINO in one B*V batch.
        img_flat = image.reshape(
            B * V,
            *image.shape[2:],
        )

        if self.use_naf_upsample:
            image_for_naf = img_flat.clone()

        img_norm = self.transform(img_flat)

        with torch.no_grad():

            # Low-VRAM MultiView DINO extraction.
            #
            # At 1024 resolution, flattening all V views into B*V and
            # evaluating DINO attention in one call can require several GiB
            # of transient SDPA workspace on ROCm.  Extract the same features
            # in small view batches, then concatenate them.  This does not
            # change MV fusion or projection semantics.
            _mv_total = int(img_norm.shape[0])
            _default_chunk = 1 if is_amd_gpu else _mv_total
            _dino_chunk = int(
                os.environ.get(
                    "TRELLIS2_MV_DINO_CHUNK",
                    str(_default_chunk),
                )
            )
            _dino_chunk = max(1, min(_dino_chunk, _mv_total))

            # gfx1030 only needs DINO chunking for the 1024 stages.
            # Keep 512 SS / Shape-512 on the original full-view batch.
            if int(self.image_size) < 1024:
                if _dino_chunk != _mv_total:
                    print(
                        "[Pixal3D MV DINO] 512 stage: "
                        "using full-view batch"
                    )
                _dino_chunk = _mv_total

            if _dino_chunk < _mv_total:
                print(
                    f"[Pixal3D MV DINO] chunked extraction: "
                    f"{_mv_total} view(s), chunk={_dino_chunk}, "
                    f"image={tuple(img_norm.shape[-2:])}"
                )

                # On ROCm 1024, keep completed view features on CPU while the
                # next view runs.  This both reduces VRAM pressure and prevents
                # a bad later kernel from silently overwriting a previous
                # view's output allocation.
                _isolate_parts_cpu = (
                    getattr(torch.version, "hip", None) is not None
                    and int(self.image_size) >= 1024
                    and os.environ.get(
                        "TRELLIS2_DINO_ISOLATE_VIEW_OUTPUTS",
                        "1",
                    ) == "1"
                )

                if _isolate_parts_cpu:
                    print(
                        "[Pixal3D MV DINO] ROCm view isolation: "
                        "completed DINO outputs are stashed on CPU"
                    )

                _z_parts = []
                for _mv_start in range(0, _mv_total, _dino_chunk):
                    _mv_end = min(
                        _mv_start + _dino_chunk,
                        _mv_total,
                    )

                    _retry_count = max(
                        0,
                        int(
                            os.environ.get(
                                "TRELLIS2_DINO_VIEW_RETRIES",
                                "1",
                            )
                        ),
                    )

                    for _attempt in range(_retry_count + 1):
                        try:
                            _z_part = self.extract_features(
                                img_norm[_mv_start:_mv_end]
                            )
                            break
                        except RuntimeError as _exc:
                            _is_dino_failure = (
                                "[Pixal3D DINO" in str(_exc)
                            )
                            if (
                                not _is_dino_failure
                                or _attempt >= _retry_count
                            ):
                                raise

                            print(
                                "[Pixal3D MV DINO] view chunk "
                                f"{_mv_start}:{_mv_end} failed on "
                                f"attempt {_attempt + 1}; restoring the "
                                "pristine model and retrying once: "
                                f"{_exc}"
                            )

                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                                torch.cuda.empty_cache()

                    if _isolate_parts_cpu:
                        _z_parts.append(
                            _z_part.detach().to(
                                device="cpu",
                                dtype=torch.float32,
                            ).clone()
                        )
                        del _z_part
                    else:
                        _z_parts.append(_z_part)

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()

                z = torch.cat(_z_parts, dim=0)

                if _isolate_parts_cpu:
                    z = z.to(
                        device=img_norm.device,
                        dtype=torch.float32,
                    )

                del _z_parts
            else:
                z = self.extract_features(img_norm)

            z_clstoken = z[:, 0:1]

            num_reg = getattr(
                self.model.config,
                "num_register_tokens",
                4,
            )

            z_regtokens = z[:, 1:1 + num_reg]
            z_patchtokens = z[:, 1 + num_reg:]

            z_patchtokens_spatial = (
                z_patchtokens.reshape(
                    B * V,
                    self.patch_number,
                    self.patch_number,
                    -1,
                )
            )

            camera_angle_x_flat = (
                camera_angle_x.reshape(B * V)
            )

            distance_flat = (
                distance.reshape(B * V)
            )

            mesh_scale_flat = (
                mesh_scale[:, None]
                .expand(B, V)
                .reshape(B * V)
            )

            calc_mat_flat = (
                calc_mat.reshape(
                    B * V,
                    4,
                    4,
                )
            )

            average = (
                self.multiview_fusion
                == "average"
            )

            z_proj_acc = None

            per_view_proj = (
                [] if not average else None
            )

            # Do projection per view to keep VRAM peak bounded.
            for v in range(V):

                idx = torch.arange(
                    v,
                    B * V,
                    V,
                    device=z.device,
                )

                img_naf_v = (
                    image_for_naf[idx]
                    if self.use_naf_upsample
                    else None
                )

                z_view = self._project_single_view(
                    z_patchtokens_spatial[idx],
                    img_naf_v,
                    camera_angle_x_flat[idx],
                    distance_flat[idx],
                    mesh_scale_flat[idx],
                    calc_mat_flat[idx],
                )

                if average:
                    if z_proj_acc is None:
                        z_proj_acc = (
                            z_view.contiguous()
                        )
                    else:
                        z_proj_acc.add_(z_view)
                        del z_view
                else:
                    per_view_proj.append(
                        z_view
                    )

            z_global_all = torch.cat(
                [
                    z_clstoken,
                    z_regtokens,
                ],
                dim=1,
            )

            z_global_all = (
                z_global_all.reshape(
                    B,
                    V,
                    z_global_all.shape[-2],
                    z_global_all.shape[-1],
                )
            )

            if average:
                z_proj = z_proj_acc.div_(V)

                z_global = (
                    z_global_all.mean(
                        dim=1
                    )
                )

            else:
                z_proj = torch.stack(
                    per_view_proj,
                    dim=2,
                )

                z_global = (
                    z_global_all.reshape(
                        B,
                        V * z_global_all.shape[2],
                        z_global_all.shape[3],
                    )
                )

        return z_global, z_proj


