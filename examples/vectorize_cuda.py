#!/usr/bin/env python3
"""
CurveSplat — GPU-Accelerated Multi-Curve Image Vectorization (v2).

Key improvements over v1:
  1. Hybrid initialization: color-region FILL curves (wide) + contour EDGE
     curves (thin). Fill curves cover the interior of color regions via
     k-means clustering; edge curves trace structural boundaries.
  2. Coarse-to-fine resolution schedule: trains at 64→128→full resolution so
     coarse structure is locked in before detail is attempted.
  3. Shuffled curve render order every step so gradients reach ALL curves,
     avoiding the (1-alpha)^N vanishing-gradient problem in serial compositing.
  4. Wider sigma_short (width/2 instead of width/4) for better area coverage.
  5. Adaptive LR schedule: high LR at coarse resolution, lower at fine.
  6. L1 + MS-SSIM loss; optional LPIPS if installed.

Usage:
    python examples/vectorize_cuda.py --image my_photo.png --steps 500 --num_curves 200
"""

import os
import sys
import math
import argparse
import time
import random
import contextlib

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add src/python to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "python"))

from curvesplat.scene import Curve, Scene, FilledShape
from curvesplat.sdf_init import initialize_curves_from_image, initialize_filled_regions
from curvesplat.renderer import CurveSplatRenderer, evaluate_bezier, build_covariance, invert_2x2, render_curves_batched, render_curves_tiled
from curvesplat.fill_renderer import render_filled_curve_fn
from curvesplat.cuda_splat import HAS_CUDA_SPLAT, render_cuda
from curvesplat.losses import compute_topology_loss, compute_width_loss, compute_edge_loss
from curvesplat.segmentation import generate_semantic_masks

try:
    from curvesplat.losses import LPIPSLoss
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False


# ---------------------------------------------------------------------------
# Region-fill initialization via k-means color clustering
# ---------------------------------------------------------------------------

def _kmeans_colors(image_np: np.ndarray, k: int = 12, iters: int = 20) -> tuple:
    """
    Simple k-means on pixel colors. Returns (centers, labels).
    image_np: (H, W, 3) float in [0,1].
    centers: (k, 3) cluster color centers.
    labels:  (H*W,) int cluster assignments.
    """
    H, W, C = image_np.shape
    pixels = image_np.reshape(-1, C).astype(np.float32)  # (N, 3)

    # Subsample pixels for centroid computation to avoid O(N * k) huge memory/time
    rng = np.random.default_rng(42)
    num_samples = min(len(pixels), 20000)
    subsample_indices = rng.choice(len(pixels), size=num_samples, replace=False)
    sub_pixels = pixels[subsample_indices]

    # Initialize centers using k-means++ style (spread out) on sub_pixels
    idx0 = rng.integers(0, len(sub_pixels))
    centers = [sub_pixels[idx0]]
    for _ in range(k - 1):
        dists = np.min(
            np.array([np.sum((sub_pixels - c) ** 2, axis=1) for c in centers]),
            axis=0,
        )
        total = dists.sum()
        if total <= 1e-12:
            chosen = rng.integers(0, len(sub_pixels))  # all points coincide with a center
        else:
            probs = dists / total
            probs = probs / probs.sum()  # guard float drift (must sum to 1 for rng.choice)
            chosen = rng.choice(len(sub_pixels), p=probs)
        centers.append(sub_pixels[chosen])
    centers = np.array(centers)  # (k, 3)

    sub_labels = np.zeros(len(sub_pixels), dtype=np.int32)
    for _ in range(iters):
        # Assign
        dists = np.sum(
            (sub_pixels[:, None, :] - centers[None, :, :]) ** 2, axis=2
        )  # (sub_N, k)
        sub_labels = dists.argmin(axis=1)
        # Update
        for j in range(k):
            mask = sub_labels == j
            if mask.sum() > 0:
                centers[j] = sub_pixels[mask].mean(axis=0)

    # Assign all pixels to the final centers
    final_dists = np.sum(
        (pixels[:, None, :] - centers[None, :, :]) ** 2, axis=2
    )
    labels = final_dists.argmin(axis=1)

    return centers, labels.reshape(H, W)


def initialize_region_fill_curves(
    image_np: np.ndarray,
    n_region_curves: int = 80,
    n_color_clusters: int = 12,
    device: str = "cpu",
    prompt: str = None,
    style_palette: np.ndarray = None,
) -> list:
    """
    Create wide fill curves that cover color regions of the image.

    For each color cluster:
      - Find the centroid and bounding box of the cluster's pixels
      - Create one or more wide Bezier curves spanning that region
      - Sample the cluster's average color
      - If style_palette is provided, match to closest style color

    These curves get very wide strokes (30-80px) to fill large areas.
    """
    H, W = image_np.shape[:2]
    img3 = image_np[:, :, :3] if image_np.ndim == 3 else np.stack([image_np]*3, axis=2)

    centers_rgb, label_map = _kmeans_colors(img3, k=n_color_clusters)

    curves = []
    target_count = n_region_curves
    if prompt is not None:
        n_region_curves = 2 * n_region_curves
    curves_per_cluster = max(1, n_region_curves // n_color_clusters)

    for cluster_id in range(n_color_clusters):
        mask = (label_map == cluster_id)
        if mask.sum() < 50:
            continue

        ys, xs = np.where(mask)
        color_rgb = centers_rgb[cluster_id].clip(0, 1)
        if style_palette is not None:
            # Map color to nearest color in style palette
            dists = np.sum((style_palette - color_rgb) ** 2, axis=1)
            color_rgb = style_palette[dists.argmin()]

        # Skip near-white clusters ONLY if they touch the image corners (acting as background)
        if color_rgb.mean() > 0.92:
            margin_y = max(1, int(H * 0.05))
            margin_x = max(1, int(W * 0.05))
            touches_corner = (
                mask[:margin_y, :margin_x].any() or
                mask[:margin_y, -margin_x:].any() or
                mask[-margin_y:, :margin_x].any() or
                mask[-margin_y:, -margin_x:].any()
            )
            if touches_corner:
                continue

        # Pre-sigmoid color
        color_raw = np.log(
            color_rgb.clip(0.02, 0.98) / (1.0 - color_rgb.clip(0.02, 0.98))
        )

        # Compute region extent
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()
        region_h = max(y_max - y_min, 1)
        region_w = max(x_max - x_min, 1)

        # Width: proportional to region size, capped sensibly (min(H, W) * 0.08)
        region_diag = math.sqrt(region_h**2 + region_w**2)
        stroke_width = float(np.clip(region_diag * 0.15, 6.0, min(H, W) * 0.08))
        # Pre-softplus: softplus^-1(w) = ln(exp(w)-1) for w>0.5, guarded against overflow
        width_raw = math.log(math.exp(stroke_width) - 1) if 0.5 < stroke_width < 20 else stroke_width

        for _ in range(curves_per_cluster):
            if len(curves) >= n_region_curves:
                break

            # Local structured stroke (NOT 4 random pixels across the whole
            # cluster — that produced canvas-spanning zig-zags and a noise init).
            # Seed a pixel, take its nearest cluster neighbors, lay 4 control
            # points ordered along the local principal axis → a coherent stroke.
            seed = np.random.randint(0, len(ys))
            d2 = (xs - xs[seed]) ** 2 + (ys - ys[seed]) ** 2
            pool_n = int(max(8, min(len(ys), len(ys) // max(1, curves_per_cluster))))
            pool = np.argpartition(d2, pool_n - 1)[:pool_n]  # pool_n nearest to seed
            pick = np.random.choice(pool, size=4, replace=len(pool) < 4)
            px = xs[pick].astype(float)
            py = ys[pick].astype(float)

            # Order along the local principal axis so the cubic is monotonic
            # (avoids self-crossing control polygons).
            centered = np.stack([px - px.mean(), py - py.mean()], axis=1)
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            order = np.argsort(centered @ vt[0])
            px, py = px[order], py[order]

            jitter = np.random.randn(4, 2) * region_diag * 0.02
            pts = np.column_stack([px, py]) + jitter
            pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)

            cp = torch.tensor(pts, dtype=torch.float32, device=device)

            # High opacity for fill curves
            opacity_raw = math.log(0.92 / 0.08)  # sigmoid^-1(0.92) ≈ 2.44

            curve = Curve(
                control_points=cp,
                color=torch.tensor(color_raw, dtype=torch.float32, device=device),
                width=torch.tensor(width_raw, dtype=torch.float32, device=device),
                opacity=torch.tensor(opacity_raw, dtype=torch.float32, device=device),
                z_index=len(curves),
            )
            curves.append(curve)

    if prompt is not None and len(curves) > target_count:
        print(f"  [CLIP Init] Scoring {len(curves)} candidate curves against prompt: '{prompt}'")
        try:
            from transformers import CLIPProcessor, CLIPModel
            
            # Load CLIP Vit-B/32
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            # Helper to extract tensor from various huggingface output formats
            def extract_tensor(features):
                if isinstance(features, torch.Tensor):
                    return features
                if hasattr(features, "text_embeds"):
                    return features.text_embeds
                if hasattr(features, "image_embeds"):
                    return features.image_embeds
                if hasattr(features, "pooler_output"):
                    return features.pooler_output
                if hasattr(features, "last_hidden_state"):
                    return features.last_hidden_state[:, 0]
                if isinstance(features, dict):
                    for k in ["text_embeds", "image_embeds", "pooler_output", "last_hidden_state"]:
                        if k in features:
                            val = features[k]
                            if k == "last_hidden_state":
                                return val[:, 0]
                            return val
                if isinstance(features, (list, tuple)):
                    return features[0]
                return features

            # Encode text prompt
            inputs_text = clip_processor(text=[prompt], return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                text_features_raw = clip_model.get_text_features(**inputs_text)
                text_features = extract_tensor(text_features_raw)
                text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
                
            # Score each candidate curve
            scored_curves = []
            for curve in curves:
                pts = curve.control_points.detach().cpu().numpy()
                x_min = int(np.floor(pts[:, 0].min()))
                x_max = int(np.ceil(pts[:, 0].max()))
                y_min = int(np.floor(pts[:, 1].min()))
                y_max = int(np.ceil(pts[:, 1].max()))
                
                x_min = max(0, x_min)
                x_max = min(W, x_max)
                y_min = max(0, y_min)
                y_max = min(H, y_max)
                
                if x_max - x_min < 32:
                    pad = (32 - (x_max - x_min)) // 2
                    x_min = max(0, x_min - pad)
                    x_max = min(W, x_max + pad + 1)
                if y_max - y_min < 32:
                    pad = (32 - (y_max - y_min)) // 2
                    y_min = max(0, y_min - pad)
                    y_max = min(H, y_max + pad + 1)
                    
                crop = img3[y_min:y_max, x_min:x_max]
                
                inputs_img = clip_processor(images=crop, return_tensors="pt").to(device)
                with torch.no_grad():
                    image_features_raw = clip_model.get_image_features(**inputs_img)
                    image_features = extract_tensor(image_features_raw)
                    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                    similarity = (image_features @ text_features.T).item()
                    
                scored_curves.append((similarity, curve))
                
            scored_curves.sort(reverse=True, key=lambda x: x[0])
            curves = [item[1] for item in scored_curves[:target_count]]
            # Reset z-index
            for idx, c in enumerate(curves):
                c.z_index = idx
                
            del clip_model
            del clip_processor
            if device == "cuda":
                torch.cuda.empty_cache()
            print(f"  [CLIP Init] Done. Kept top {len(curves)} curves with highest CLIP similarity.")
        except Exception as e:
            print(f"  [CLIP Init] Error loading CLIP or scoring: {e}. Falling back to default top {target_count} candidates.")
            curves = curves[:target_count]
            # Reset z-index
            for idx, c in enumerate(curves):
                c.z_index = idx

    return curves


# ---------------------------------------------------------------------------
# Multi-curve differentiable renderer (shuffled order for gradient balance)
# ---------------------------------------------------------------------------

def render_single_curve_fn(
    control_points: torch.Tensor,
    color_raw: torch.Tensor,
    width_raw: torch.Tensor,
    opacity_raw: torch.Tensor,
    canvas: torch.Tensor,
    num_samples: int,
    width_scale: float = 1.0,
    max_width: float = 35.0,
) -> torch.Tensor:
    """Renders a single curve and composites it onto the canvas."""
    # Force float32 for the entire rendering and compositing pipeline to avoid
    # mixed-precision (Float/Half) type mismatches during TorchInductor backward compile.
    with torch.cuda.amp.autocast(enabled=False):
        control_points = control_points.float()
        color_raw = color_raw.float()
        width_raw = width_raw.float()
        opacity_raw = opacity_raw.float()
        canvas_f32 = canvas.float()

        t = torch.linspace(0.0, 1.0, num_samples, device=control_points.device, dtype=torch.float32)
        positions, tangents = evaluate_bezier(control_points, t)

        # Scale the width and clamp to min=0.5, max=max_width for numerical stability and preventing bloat
        width = (F.softplus(width_raw) * width_scale).clamp(min=0.5, max=max_width)
        opacity = torch.sigmoid(opacity_raw)
        color = torch.sigmoid(color_raw)

        covariances = build_covariance(tangents, width)
        cov_inv = invert_2x2(covariances)

        H, W = canvas.shape[1], canvas.shape[2]
        device = canvas.device

        py = torch.arange(H, device=device, dtype=torch.float32) + 0.5
        px = torch.arange(W, device=device, dtype=torch.float32) + 0.5
        grid_y, grid_x = torch.meshgrid(py, px, indexing="ij")

        # Sum-mode Gaussian accumulation for smooth area coverage
        accum_weight = torch.zeros(H, W, device=device, dtype=torch.float32)
        for i in range(num_samples):
            dx = grid_x - positions[i, 0]
            dy = grid_y - positions[i, 1]

            a = cov_inv[i, 0, 0]
            b = cov_inv[i, 0, 1]
            c = cov_inv[i, 1, 0]
            d = cov_inv[i, 1, 1]

            mahal_sq = a * dx * dx + (b + c) * dx * dy + d * dy * dy
            # Clamp mahal_sq to a safe maximum to prevent exp(-inf) gradient NaNs in backprop
            mahal_sq_clipped = torch.clamp(mahal_sq, max=50.0)

            within_cutoff = (mahal_sq <= 9.0)
            weight = torch.exp(-0.5 * mahal_sq_clipped) * within_cutoff.float()
            accum_weight = accum_weight + weight

        # Normalize by num_samples → [0,1]
        final_weight = (accum_weight / num_samples).clamp(0.0, 1.0)

        alpha = (final_weight * opacity).clamp(0.0, 1.0)
        alpha_3d = alpha.unsqueeze(0)  # (1, H, W)
        color_expanded = color.view(3, 1, 1)

        out_canvas = alpha_3d * color_expanded + (1.0 - alpha_3d) * canvas_f32

    # Cast back to original canvas dtype (e.g. float16 under AMP) at the function boundary
    return out_canvas.to(canvas.dtype)


# Compile the single-curve renderer if running on GPU to eliminate Python loop and kernel launch overheads.
if torch.cuda.is_available():
    try:
        render_single_curve_fn = torch.compile(render_single_curve_fn)
        print("✓ Enabled torch.compile on curve renderer for GPU acceleration")
    except Exception as e:
        print(f"  → Warning: Could not compile renderer: {e}")


def render_scene_shuffled(
    scene: Scene,
    renderer: CurveSplatRenderer,
    shuffle: bool = True,
    render_size: tuple = None,
    use_checkpoint: bool = False,
    crop_box: tuple = None,
) -> torch.Tensor:
    """
    Differentiable multi-curve compositor with optional curve shuffling and bounding-box cropping.

    Shuffling ensures all curves receive meaningful gradients each step,
    preventing the (1-alpha)^N vanishing-gradient problem in deep compositing.

    Args:
        scene: Scene with curves and background.
        renderer: CurveSplatRenderer (used for num_samples).
        shuffle: If True, randomize the render order each call.
        render_size: Optional (H, W) to render at a different resolution
                     than scene.canvas_size (for coarse-to-fine schedule).
        use_checkpoint: If True, use gradient checkpointing to save GPU memory.
        crop_box: Optional (ymin, ymax, xmin, xmax) for cropped rendering.
    """
    orig_H, orig_W = scene.canvas_size
    device = scene.background_raw.device
    dtype = scene.background_raw.dtype

    if crop_box is not None:
        ymin, ymax, xmin, xmax = crop_box
        crop_h = ymax - ymin
        crop_w = xmax - xmin
        
        H, W = render_size if render_size is not None else (crop_h, crop_w)
        scale_x = W / crop_w
        scale_y = H / crop_h
        width_scale = min(scale_x, scale_y)
        
        # Start with background cropped
        canvas = scene.background.view(3, 1, 1).expand(3, H, W).clone()
        crop_offset = torch.tensor([[xmin, ymin]], device=device, dtype=dtype)
    else:
        H, W = render_size if render_size is not None else scene.canvas_size
        scale_x = W / orig_W
        scale_y = H / orig_H
        width_scale = min(scale_x, scale_y)
        canvas = scene.background.view(3, 1, 1).expand(3, H, W).clone()
        crop_offset = None

    is_training = any(p.requires_grad for p in scene.parameters()) and torch.is_grad_enabled()

    # Split filled shapes (crisp base layer) from painterly strokes.
    # Filled shapes composite FIRST, in fixed z-order — shuffling them would break
    # opaque paint-over ordering. Only strokes are shuffled (for gradient balance).
    filled_shapes = [c for c in scene.curves if getattr(c, "is_closed", False)]
    stroke_curves = [c for c in scene.curves if not getattr(c, "is_closed", False)]
    filled_shapes = sorted(filled_shapes, key=lambda c: getattr(c, "z_index", 0))

    fill_k = float(getattr(scene, "fill_k", 1.0))
    for shape in filled_shapes:
        seg = shape.segments.clone()
        if crop_offset is not None:
            seg = seg - crop_offset
        seg_scaled = seg * torch.tensor([[[scale_x, scale_y]]], device=device, dtype=dtype)
        canvas = render_filled_curve_fn(
            seg_scaled,
            shape.color_raw,
            shape.opacity_raw,
            canvas,
            samples_per_segment=int(getattr(scene, "fill_samples", 24)),
            k=fill_k,
            width_scale=width_scale,
        )

    curve_list = list(stroke_curves)
    if shuffle and is_training:
        random.shuffle(curve_list)

    if curve_list:
        # Batched stroke render: vectorize the splat math over ALL curves in one
        # kernel (chunked) instead of a Python loop — this is what actually uses
        # the GPU. Composite order = curve_list order (shuffled above if training).
        scale_t = torch.tensor([[scale_x, scale_y]], device=device, dtype=dtype)
        cps, colors, widths, opacities, maxws = [], [], [], [], []
        for curve in curve_list:
            cp = curve.control_points
            if crop_offset is not None:
                cp = cp - crop_offset
            cps.append(cp * scale_t)
            colors.append(curve.color_raw)
            widths.append(curve.width_raw.reshape(()))
            opacities.append(curve.opacity_raw.reshape(()))
            maxws.append(float(getattr(curve, "max_width", 35.0)))

        cp_b = torch.stack(cps, dim=0)
        color_b = torch.stack(colors, dim=0)
        width_b = torch.stack(widths, dim=0)
        opacity_b = torch.stack(opacities, dim=0)
        maxw_b = torch.tensor(maxws, device=device, dtype=torch.float32)

        # CUDA fast path: custom kernel computes per-curve alpha (the O(C·N·H·W)
        # bottleneck), PyTorch composites. Falls back to the dense PyTorch renderer
        # (the numerical reference) when the kernel isn't built or shapes don't fit.
        use_cuda = (HAS_CUDA_SPLAT and canvas.is_cuda and crop_box is None
                    and not use_checkpoint)
        if use_cuda:
            # Activate params here (kernel takes activated inputs; autograd handles
            # these ops). Must match render_curves_batched's internal activation.
            width_act = (F.softplus(width_b) * width_scale).clamp(min=0.5)
            width_act = torch.minimum(width_act, maxw_b)
            color_act = torch.sigmoid(color_b)
            opacity_act = torch.sigmoid(opacity_b)
            canvas = render_cuda(cp_b, color_act, width_act, opacity_act,
                                 canvas, renderer.num_samples)
        elif is_training and use_checkpoint:
            canvas = torch.utils.checkpoint.checkpoint(
                render_curves_batched,
                cp_b, color_b, width_b, opacity_b, maxw_b, canvas,
                renderer.num_samples, width_scale, 24,
                use_reentrant=False,
            )
        else:
            canvas = render_curves_batched(
                cp_b, color_b, width_b, opacity_b, maxw_b, canvas,
                renderer.num_samples, width_scale,
            )

    return canvas



# ---------------------------------------------------------------------------
# SVG export
# ---------------------------------------------------------------------------

def export_to_svg(scene: Scene, path: str, quantize: bool = False,
                  min_opacity: float = 0.1, min_area: float = 8.0):
    """
    Export the scene curves to a standard SVG file with snapping and merging.

    Prunes near-invisible scratch strokes that clutter the output (faint and/or
    thin-and-short curves contribute nothing visually but litter the SVG):
      - min_opacity: drop strokes fainter than this.
      - min_area:    drop strokes whose coverage (width × length) is below this.
    Raise both for a cleaner, more intentional (abstraction-style) export.
    """
    H, W = scene.canvas_size

    # 1. Gather active curves and filter out dormant/low-opacity/negligible ones
    active_curves = []
    for curve in scene.curves:
        if getattr(curve, "is_dormant", False):
            continue
        if getattr(curve, "is_closed", False):
            continue  # filled shapes are exported separately (see below)
        color = curve.color.detach().cpu().numpy()
        w = curve.width.item()
        o = curve.opacity.item()
        if o < min_opacity:
            continue
        cp = curve.control_points.detach().cpu().numpy()
        # Coverage = width × control-polygon length; thin short scratches → ~0.
        length = float(np.linalg.norm(np.diff(cp, axis=0), axis=1).sum())
        if w * length < min_area:
            continue
        active_curves.append({
            "cp": cp,
            "color": color,
            "width": w,
            "opacity": o,
            "used": False
        })

    # 2. Greedy merge pass
    merged_paths = []
    for i, c_i in enumerate(active_curves):
        if c_i["used"]:
            continue
        c_i["used"] = True
        
        chain = [c_i["cp"]]
        color = c_i["color"]
        width = c_i["width"]
        opacity = c_i["opacity"]
        
        while True:
            appended = False
            last_pt = chain[-1][3]
            
            for j, c_j in enumerate(active_curves):
                if c_j["used"]:
                    continue
                
                # Check compatibility (similar color, width, opacity)
                color_dist = np.linalg.norm(color - c_j["color"])
                width_dist = abs(width - c_j["width"])
                opacity_dist = abs(opacity - c_j["opacity"])
                
                if color_dist < 0.05 and width_dist < 1.5 and opacity_dist < 0.1:
                    # Check connection distance
                    dist = np.linalg.norm(last_pt - c_j["cp"][0])
                    if dist < 3.0:
                        # Check tangent alignment (angle < 30 degrees)
                        t_last = last_pt - chain[-1][2]
                        t_next = c_j["cp"][1] - c_j["cp"][0]
                        norm_l = np.linalg.norm(t_last)
                        norm_n = np.linalg.norm(t_next)
                        aligned = True
                        if norm_l > 1e-4 and norm_n > 1e-4:
                            cos_theta = np.dot(t_last, t_next) / (norm_l * norm_n)
                            cos_theta = np.clip(cos_theta, -1.0, 1.0)
                            if cos_theta < 0.86:  # cos(30 deg)
                                aligned = False
                        
                        if aligned:
                            chain.append(c_j["cp"])
                            c_j["used"] = True
                            appended = True
                            break
            if not appended:
                break
                
        merged_paths.append({
            "segments": chain,
            "color": color,
            "width": width,
            "opacity": opacity
        })

    # 3. Write SVG
    with open(path, "w") as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">\n')

        bg_rgb = scene.background.detach().cpu().numpy()
        bg_r, bg_g, bg_b = int(bg_rgb[0]*255), int(bg_rgb[1]*255), int(bg_rgb[2]*255)
        f.write(f'  <rect width="{W}" height="{H}" fill="rgb({bg_r},{bg_g},{bg_b})" />\n')

        def fmt(val):
            return f"{round(val):d}" if quantize else f"{val:.1f}"

        # --- Filled shapes first (SVG paint order = document order) ---
        # Crisp tracer-style flat regions form the base layer, painted under the
        # painterly strokes. Sorted by z_index so larger/background regions go first.
        filled = [c for c in scene.curves if getattr(c, "is_closed", False)
                  and not getattr(c, "is_dormant", False)]
        filled.sort(key=lambda c: getattr(c, "z_index", 0))
        for shape in filled:
            seg = shape.segments.detach().cpu().numpy()  # (S, 4, 2)
            col = shape.color.detach().cpu().numpy()
            r, g, b = int(col[0] * 255), int(col[1] * 255), int(col[2] * 255)
            o = float(shape.opacity.item())
            d_parts = [f"M {fmt(seg[0, 0, 0])} {fmt(seg[0, 0, 1])}"]
            for s in range(seg.shape[0]):
                d_parts.append(
                    f"C {fmt(seg[s, 1, 0])} {fmt(seg[s, 1, 1])} "
                    f"{fmt(seg[s, 2, 0])} {fmt(seg[s, 2, 1])} "
                    f"{fmt(seg[s, 3, 0])} {fmt(seg[s, 3, 1])}"
                )
            d_parts.append("Z")
            f.write(f'  <path d="{" ".join(d_parts)}" fill="rgb({r},{g},{b})" '
                    f'fill-opacity="{o:.2f}" stroke="none" />\n')

        for p in merged_paths:
            chain = p["segments"]
            r = int(p["color"][0] * 255)
            g = int(p["color"][1] * 255)
            b = int(p["color"][2] * 255)
            w = p["width"]
            o = p["opacity"]
            
            d_parts = []
            d_parts.append(f"M {fmt(chain[0][0,0])} {fmt(chain[0][0,1])}")
            for seg in chain:
                d_parts.append(f"C {fmt(seg[1,0])} {fmt(seg[1,1])} {fmt(seg[2,0])} {fmt(seg[2,1])} {fmt(seg[3,0])} {fmt(seg[3,1])}")
            d_str = " ".join(d_parts)
            
            f.write(f'  <path d="{d_str}" stroke="rgb({r},{g},{b})" stroke-width="{w:.2f}" '
                    f'fill="none" opacity="{o:.2f}" stroke-linecap="round" />\n')

        f.write('</svg>\n')


def adapt_curve_density(
    scene: Scene,
    renderer: CurveSplatRenderer,
    current_render: torch.Tensor,
    phase_target: torch.Tensor,
    phase_lr: float,
    target_rgb: np.ndarray,
    orig_H: int,
    orig_W: int,
    scale_x: float,
    scale_y: float,
    optimizer: torch.optim.Optimizer,
    render_size: tuple,
    lr_cp_mult: float = 2.0,
    max_width_fill: float = 35.0,
    max_width_edge: float = 8.0,
    crop_box: tuple = None,
) -> torch.optim.Optimizer:
    """
    Spawns 10 curves in highest-error regions by splitting worst curves there,
    removes the 10 lowest-contributing curves, and updates the optimizer.
    Supports crop_box for semantic layer decomposition.
    """
    # 1. Evaluate contributions of all curves.
    # Filled shapes (crisp base layer) are NEVER pruned or jittered — they form
    # the tracer-quality scaffold. Keep them present during every ablation render
    # (so contributions are measured against the real composite) but adapt only
    # the painterly stroke curves.
    all_curves = list(scene.curves)
    filled_shapes = [c for c in all_curves if getattr(c, "is_closed", False)]
    original_curves = [c for c in all_curves if not getattr(c, "is_closed", False)]
    contributions = []

    # Use torch.no_grad() to prevent building huge computation graphs during ablation, which causes OOM
    with torch.no_grad():
        for i in range(len(original_curves)):
            temp_curves = filled_shapes + original_curves[:i] + original_curves[i+1:]
            scene.curves = torch.nn.ModuleList(temp_curves)
            rendered_without = render_scene_shuffled(
                scene, renderer, shuffle=False, render_size=render_size, crop_box=crop_box
            )
            diff = torch.abs(current_render - rendered_without).mean().item()
            contributions.append(diff)
            
    scene.curves = torch.nn.ModuleList(filled_shapes + original_curves)

    # Sort curves by contribution, prioritizing dormant ones (Task 1.2)
    adjusted_contributions = []
    for idx, curve in enumerate(original_curves):
        contrib = contributions[idx]
        # Dormant curves are given a negative contribution score to prune them first
        contrib = -1.0 - (1.0 / (1.0 + contrib))
        adjusted_contributions.append(contrib)
    curve_indices_sorted = np.argsort(adjusted_contributions)
    
    # Keep top N-10 curves
    num_to_adapt = min(10, len(original_curves) // 2)
    if num_to_adapt <= 0:
        return optimizer
        
    remove_indices = set(curve_indices_sorted[:num_to_adapt])
    remaining_curves = [original_curves[idx] for idx in range(len(original_curves)) if idx not in remove_indices]
    
    # 2. connected component error map for spawning (Task 1.3)
    error_map_np = torch.abs(current_render - phase_target).mean(dim=0).cpu().numpy()
    
    # Smooth error map
    from scipy.ndimage import gaussian_filter, label
    smoothed = gaussian_filter(error_map_np, sigma=2.0)
    
    # Threshold at mean + 0.5 * std
    thresh = smoothed.mean() + 0.5 * smoothed.std()
    binary_mask = (smoothed > thresh)
    
    # Find components
    labeled_array, num_features = label(binary_mask)
    
    # Compute component stats (area, centroid, major axis)
    components = []
    device = current_render.device
    
    if crop_box is not None:
        ymin, ymax, xmin, xmax = crop_box
    else:
        ymin, ymax, xmin, xmax = 0, orig_H, 0, orig_W
        
    for comp_idx in range(1, num_features + 1):
        comp_mask = (labeled_array == comp_idx)
        area = comp_mask.sum()
        if area < 10:
            continue
            
        ys, xs = np.where(comp_mask)
        cy, cx = ys.mean(), xs.mean()
        
        # Orient using spatial moments
        dy = ys - cy
        dx = xs - cx
        mu_20 = (dx**2).mean()
        mu_02 = (dy**2).mean()
        mu_11 = (dx*dy).mean()
        
        # Major axis angle
        diff_mom = mu_20 - mu_02
        denom = math.sqrt(diff_mom**2 + 4 * mu_11**2)
        if denom > 1e-6:
            cos_2t = diff_mom / denom
            sin_2t = (2 * mu_11) / denom
            angle = 0.5 * math.atan2(sin_2t, cos_2t)
        else:
            angle = 0.0
            
        r_diag = math.sqrt(area)
        components.append({
            "cx": cx,
            "cy": cy,
            "angle": angle,
            "r_diag": r_diag,
            "area": area
        })
        
    # Sort components by area descending
    components.sort(key=lambda x: x["area"], reverse=True)
    
    new_curves = []
    H, W = render_size
    for i in range(num_to_adapt):
        # Fallback to worst-curve split if we don't have enough components
        if i >= len(components):
            worst_curve = remaining_curves[-1]
            new_cp = worst_curve.control_points.clone().detach()
            noise = torch.randn_like(new_cp) * min(orig_H, orig_W) * 0.02
            # Clamp to crop boundaries if crop_box is active
            new_cp[:, 0] = (new_cp[:, 0] + noise[:, 0]).clamp(xmin, xmax)
            new_cp[:, 1] = (new_cp[:, 1] + noise[:, 1]).clamp(ymin, ymax)
            
            new_color = worst_curve.color_raw.clone().detach() + torch.randn_like(worst_curve.color_raw) * 0.1
            new_width = worst_curve.width_raw.clone().detach() + torch.randn_like(worst_curve.width_raw) * 0.1
            new_opacity = worst_curve.opacity_raw.clone().detach() + torch.randn_like(worst_curve.opacity_raw) * 0.1
            
            new_curve = Curve(
                control_points=new_cp,
                color=new_color,
                width=new_width,
                opacity=new_opacity,
                z_index=worst_curve.z_index,
                max_width=getattr(worst_curve, "max_width", max_width_fill),
            )
        else:
            comp = components[i]
            center_x, center_y = comp["cx"], comp["cy"]
            angle, r_diag = comp["angle"], comp["r_diag"]
            
            # Map back to global coordinates using crop_box offset
            orig_center_x = center_x / scale_x + xmin
            orig_center_y = center_y / scale_y + ymin
            orig_r_diag = r_diag / min(scale_x, scale_y)
            
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            L = orig_r_diag * 0.25
            
            pts = np.array([
                [orig_center_x - 0.5 * L * cos_a, orig_center_y - 0.5 * L * sin_a],
                [orig_center_x - 0.16 * L * cos_a, orig_center_y - 0.16 * L * sin_a],
                [orig_center_x + 0.16 * L * cos_a, orig_center_y + 0.16 * L * sin_a],
                [orig_center_x + 0.5 * L * cos_a, orig_center_y + 0.5 * L * sin_a]
            ])
            pts[:, 0] = np.clip(pts[:, 0], xmin, xmax - 1)
            pts[:, 1] = np.clip(pts[:, 1], ymin, ymax - 1)
            
            # Sample color from the global target image using global coords
            global_img_y = int(np.clip(orig_center_y, 0, orig_H - 1))
            global_img_x = int(np.clip(orig_center_x, 0, orig_W - 1))
            color_rgb = target_rgb[global_img_y, global_img_x].clip(0.02, 0.98)
            color_raw = np.log(color_rgb / (1.0 - color_rgb))
            
            stroke_width = float(np.clip(orig_r_diag * 0.15, 4.0, 30.0))
            width_raw = math.log(math.exp(stroke_width) - 1) if 0.5 < stroke_width < 20 else stroke_width
            opacity_raw = math.log(0.8 / 0.2)
            
            new_curve = Curve(
                control_points=torch.tensor(pts, dtype=torch.float32, device=device),
                color=torch.tensor(color_raw, dtype=torch.float32, device=device),
                width=torch.tensor(width_raw, dtype=torch.float32, device=device),
                opacity=torch.tensor(opacity_raw, dtype=torch.float32, device=device),
                z_index=len(remaining_curves) + len(new_curves),
                max_width=max_width_edge,
            )
            
        new_curves.append(new_curve)
        
    all_new_curves = remaining_curves + new_curves
    for idx, c in enumerate(all_new_curves):
        c.z_index = idx

    # Re-attach the untouched filled base layer underneath the painterly strokes.
    scene.curves = torch.nn.ModuleList(filled_shapes + all_new_curves)
    
    # Recreate optimizer since parameters changed (using parameter groups for learning rates)
    control_points_params = []
    other_params = []
    for name, param in scene.named_parameters():
        if "control_points" in name:
            control_points_params.append(param)
        else:
            other_params.append(param)

    new_optimizer = torch.optim.AdamW(
        [
            {"params": control_points_params, "lr": phase_lr * lr_cp_mult},
            {"params": other_params, "lr": phase_lr},
        ],
        betas=(0.9, 0.999),
        weight_decay=0.005,
    )
    
    print(f"  [Adaptive Curves] Spawned {num_to_adapt} new curves in high-error zones, removed {num_to_adapt} weak ones. Optimizer reset.")
    return new_optimizer

PROFILES = {
    "background": {
        "n_fill": 40,
        "n_edge": 10,
        "max_width_fill": 30.0,
        "max_width_edge": 6.0,
        "lr_scale": 1.2,
        "steps_scale": 0.8,
    },
    "mid-ground": {
        "n_fill": 30,
        "n_edge": 30,
        "max_width_fill": 15.0,
        "max_width_edge": 4.0,
        "lr_scale": 1.0,
        "steps_scale": 1.0,
    },
    "foreground": {
        "n_fill": 20,
        "n_edge": 40,
        "max_width_fill": 10.0,
        "max_width_edge": 3.0,
        "lr_scale": 1.0,
        "steps_scale": 1.0,
    },
    "details": {
        "n_fill": 5,
        "n_edge": 50,
        "max_width_fill": 6.0,
        "max_width_edge": 2.0,
        "lr_scale": 0.8,
        "steps_scale": 1.2,
    }
}

def get_layer_profile(prompt: str, index: int, total_prompts: int) -> dict:
    prompt_lower = prompt.lower()
    if "background" in prompt_lower or "bg" in prompt_lower:
        return PROFILES["background"]
    elif any(kw in prompt_lower for kw in ["detail", "feather", "hair", "eye", "face", "mouth", "nose", "edge", "line", "text", "character", "sketch"]):
        return PROFILES["details"]
    
    if total_prompts == 1:
        return PROFILES["foreground"]
    elif total_prompts == 2:
        return PROFILES["background"] if index == 0 else PROFILES["foreground"]
    else:
        if index == 0:
            return PROFILES["background"]
        elif index == total_prompts - 1:
            return PROFILES["details"]
        elif index == total_prompts - 2:
            return PROFILES["foreground"]
        else:
            return PROFILES["mid-ground"]

def run_semantic_decomposition(
    args,
    device: str,
    target_rgb: np.ndarray,
    target_tensor: torch.Tensor,
    lpips_loss_fn,
    style_palette: np.ndarray = None,
    style_mean_width: float = None,
    style_std_width: float = None,
    style_vgg_extractor = None,
    style_gram_matrices = None,
):
    """
    Executes Phase 3 - Task 3.1: Semantic Layer Decomposition.
    Segments image, fits each layer independently, composites them,
    and runs a global refinement pass.
    """
    import os
    import time
    import torch.nn.functional as F
    from pytorch_msssim.ssim import _ssim, _fspecial_gauss_1d

    H, W = target_rgb.shape[:2]
    
    # 1. Parse prompts
    prompts = [p.strip() for p in args.segment_prompts.split(",") if p.strip()]
    print("=" * 65)
    print(f"✓ Starting Semantic Layer Decomposition with {len(prompts)} layers:")
    for idx, p in enumerate(prompts):
        print(f"  Layer {idx + 1}: '{p}'")
    print("=" * 65)
    
    # 2. Generate masks
    masks_np = generate_semantic_masks(target_rgb, prompts, device=device)
    
    # Save the masks to output directory
    for idx, mask in enumerate(masks_np):
        mask_path = os.path.join(args.output_dir, f"mask_layer_{idx}_{prompts[idx].replace(' ', '_')}.png")
        plt.imsave(mask_path, mask.astype(np.float32), cmap="gray")
        print(f"  → Saved mask: {mask_path}")

    global_curves = []
    
    # Initialize background color (sample corners)
    corner_pixels = [
        target_rgb[0, 0], target_rgb[0, -1], target_rgb[-1, 0], target_rgb[-1, -1]
    ]
    bg_rgb = np.mean(corner_pixels, axis=0).clip(0.02, 0.98)
    bg_raw = np.log(bg_rgb / (1.0 - bg_rgb)).astype(np.float32)

    # Initialize renderer
    renderer = CurveSplatRenderer(num_samples=args.num_samples)
    
    # Mixed Precision (AMP) setup
    is_cuda = (device == "cuda")
    use_amp = is_cuda and not getattr(args, "no_amp", False)
    autocast_context = torch.cuda.amp.autocast() if use_amp else contextlib.nullcontext()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Determine MS-SSIM status
    try:
        from pytorch_msssim.ssim import _ssim, _fspecial_gauss_1d
        has_msssim = True
    except ImportError:
        has_msssim = False

    # Helper function for stable MS-SSIM
    def stable_ms_ssim(X: torch.Tensor, Y: torch.Tensor, data_range: float = 1.0, size_average: bool = True, K = (0.01, 0.03)) -> torch.Tensor:
        win_size = 11
        win_sigma = 1.5
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        weights_tensor = X.new_tensor(weights)
        levels = len(weights)
        mcs = []
        for i in range(levels):
            ssim_per_channel, cs = _ssim(X, Y, win=win, data_range=data_range, size_average=False, K=K)
            if i < levels - 1:
                mcs.append(torch.clamp(cs, min=1e-4))
                padding = [s % 2 for s in X.shape[2:]]
                X = F.avg_pool2d(X, kernel_size=2, padding=padding)
                Y = F.avg_pool2d(Y, kernel_size=2, padding=padding)
        ssim_per_channel = torch.clamp(ssim_per_channel, min=1e-4)
        mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)
        ms_ssim_val = torch.prod(mcs_and_ssim ** weights_tensor.view(-1, 1, 1), dim=0)
        if size_average:
            return ms_ssim_val.mean()
        else:
            return ms_ssim_val.mean(1)

    # 3. Layer-by-layer fitting
    for idx, (p_k, M_k) in enumerate(zip(prompts, masks_np)):
        profile = get_layer_profile(p_k, idx, len(prompts))
        print("\n" + "-" * 65)
        print(f"  Optimizing Layer {idx + 1}/{len(prompts)}: '{p_k}'")
        print(f"  Profile: {profile}")
        print("-" * 65)
        
        # Get bounding box with margin
        ys, xs = np.where(M_k)
        if len(ys) == 0:
            ymin, ymax, xmin, xmax = 0, H, 0, W
        else:
            margin = 8
            ymin = max(0, int(ys.min()) - margin)
            ymax = min(H, int(ys.max()) + margin + 1)
            xmin = max(0, int(xs.min()) - margin)
            xmax = min(W, int(xs.max()) + margin + 1)
            
        crop_h = ymax - ymin
        crop_w = xmax - xmin
        print(f"  Crop bounding box: y[{ymin}:{ymax}], x[{xmin}:{xmax}] (Size: {crop_w}x{crop_h})")
        
        target_rgb_crop = target_rgb[ymin:ymax, xmin:xmax]
        M_k_crop = M_k[ymin:ymax, xmin:xmax]
        
        # Initialize curves inside the crop
        # A. Region fill curves
        print(f"  → Initializing {profile['n_fill']} region fill curves for '{p_k}'...")
        layer_fill_curves = initialize_region_fill_curves(
            target_rgb_crop,
            n_region_curves=profile["n_fill"],
            n_color_clusters=min(16, profile["n_fill"] // 4 + 2),
            device=device,
            prompt=p_k,
            style_palette=style_palette,
        )
        for c in layer_fill_curves:
            c.control_points.data[:, 0] += xmin
            c.control_points.data[:, 1] += ymin
            c.max_width = min(profile["max_width_fill"], style_mean_width * 1.5) if style_mean_width is not None else profile["max_width_fill"]
            
        # B. Edge curves
        print(f"  → Initializing {profile['n_edge']} edge curves for '{p_k}'...")
        target_rgb_crop_masked = target_rgb_crop * M_k_crop[:, :, None]
        layer_edge_scene = initialize_curves_from_image(
            target_rgb_crop_masked,
            canvas_size=(crop_h, crop_w),
            is_binary=False,
            min_curves=5,
            max_curves=profile["n_edge"],
            default_width=args.width,
            simplify_epsilon=1.5,
            corner_angle_threshold=30.0,
            device=device,
        )
        layer_edge_curves = list(layer_edge_scene.curves)
        for c in layer_edge_curves:
            c.control_points.data[:, 0] += xmin
            c.control_points.data[:, 1] += ymin
            c.max_width = min(profile["max_width_edge"], style_mean_width * 0.8) if style_mean_width is not None else profile["max_width_edge"]
            
        layer_curves = layer_fill_curves + layer_edge_curves
        for c_idx, c in enumerate(layer_curves):
            c.z_index = c_idx
            
        print(f"  → Total active curves for this layer: {len(layer_curves)}")
        if len(layer_curves) == 0:
            print("  → Skipping layer (no curves initialized).")
            continue
            
        # Create layer Scene
        layer_scene = Scene(
            curves=layer_curves,
            canvas_size=(H, W),
            background=torch.tensor(bg_raw, dtype=torch.float32, device=device),
            device=device,
        )
        
        # Setup layer optimizer
        control_points_params = []
        other_params = []
        for name, param in layer_scene.named_parameters():
            if "control_points" in name:
                control_points_params.append(param)
            else:
                other_params.append(param)
                
        layer_optimizer = torch.optim.AdamW(
            [
                {"params": control_points_params, "lr": args.lr * profile["lr_scale"] * args.lr_cp_mult},
                {"params": other_params, "lr": args.lr * profile["lr_scale"]},
            ],
            betas=(0.9, 0.999),
            weight_decay=0.005,
        )
        
        # Setup crop-aware coarse-to-fine schedule
        layer_steps = args.steps
        if args.no_coarse:
            layer_phases = [
                {"name": "Full", "size": (crop_h, crop_w), "lr": args.lr * profile["lr_scale"], "max_steps": layer_steps}
            ]
        else:
            coarse_h, coarse_w = max(32, crop_h // 8), max(32, crop_w // 8)
            mid_h, mid_w = max(64, crop_h // 4), max(64, crop_w // 4)
            coarse_steps = int(layer_steps // 4 * profile["steps_scale"])
            mid_steps = int(layer_steps // 4 * profile["steps_scale"])
            fine_steps = int((layer_steps - coarse_steps - mid_steps) * profile["steps_scale"])
            layer_phases = [
                {"name": "Coarse", "size": (coarse_h, coarse_w), "lr": args.lr * 1.5 * profile["lr_scale"], "max_steps": coarse_steps},
                {"name": "Mid", "size": (mid_h, mid_w), "lr": args.lr * profile["lr_scale"], "max_steps": mid_steps},
                {"name": "Fine", "size": (crop_h, crop_w), "lr": args.lr * 0.4 * profile["lr_scale"], "max_steps": fine_steps}
            ]
            
        current_phase_idx = 0
        steps_in_phase = 0
        loss_history = []
        
        current_phase = layer_phases[current_phase_idx]
        phase_name = current_phase["name"]
        render_size = current_phase["size"]
        phase_lr = current_phase["lr"]
        rH, rW = render_size
        
        # Crop target tensors
        target_crop = target_tensor[:, ymin:ymax, xmin:xmax]
        mask_crop = torch.from_numpy(M_k_crop).float().to(device)
        
        # Interpolate crop target and crop mask to phase size
        phase_target = F.interpolate(target_crop.unsqueeze(0), size=(rH, rW), mode="bilinear", align_corners=False).squeeze(0)
        phase_mask = F.interpolate(mask_crop.unsqueeze(0).unsqueeze(0), size=(rH, rW), mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
        
        scale_x = rW / crop_w
        scale_y = rH / crop_h
        
        layer_optimizer.state.clear()
        layer_optimizer.param_groups[0]["lr"] = phase_lr * args.lr_cp_mult
        layer_optimizer.param_groups[1]["lr"] = phase_lr
        
        layer_step = 0
        max_total_steps = sum(p["max_steps"] for p in layer_phases)
        
        def compute_slope(history):
            n = len(history)
            if n < 2:
                return 0.0
            x = np.arange(n)
            y = np.array(history)
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            numerator = np.sum((x - x_mean) * (y - y_mean))
            denominator = np.sum((x - x_mean) ** 2)
            if denominator < 1e-8:
                return 0.0
            return float(numerator / denominator)

        while layer_step < max_total_steps:
            layer_step += 1
            steps_in_phase += 1
            
            if device == "cuda":
                torch.cuda.synchronize()
            t_step_start = time.time()
            
            layer_optimizer.zero_grad()
            
            with autocast_context:
                # Render with crop_box
                rendered = render_scene_shuffled(
                    layer_scene, renderer, shuffle=True, render_size=render_size, use_checkpoint=args.checkpoint, crop_box=(ymin, ymax, xmin, xmax)
                )
                
                if phase_name == "Coarse":
                    w_l1, w_ssim, w_lpips = 1.0, 0.0, 0.0
                elif phase_name == "Mid":
                    w_l1, w_ssim, w_lpips = 0.5, 0.5, 0.0
                else:  # Fine or Full
                    if lpips_loss_fn is not None:
                        w_l1, w_ssim, w_lpips = 0.3, 0.4, 0.3
                    else:
                        w_l1, w_ssim, w_lpips = 0.6, 0.4, 0.0
                        
                # Apply mask (continuous/soft)
                rendered_masked = rendered * phase_mask.unsqueeze(0)
                target_masked = phase_target * phase_mask.unsqueeze(0)
                
                # Compute normalized L1 loss
                l1 = F.l1_loss(rendered_masked, target_masked, reduction='sum') / (phase_mask.sum() * 3.0 + 1e-8)
                if torch.isnan(l1):
                    raise RuntimeError(f"NaN detected in L1 loss at step {layer_step} of layer {p_k}!")
                    
                loss = w_l1 * l1
                
                # MS-SSIM Loss (requires resolution > 160)
                use_ssim = w_ssim > 0 and has_msssim and rH > 160 and rW > 160
                if use_ssim:
                    autocast_disable = torch.cuda.amp.autocast(enabled=False) if use_amp else contextlib.nullcontext()
                    with autocast_disable:
                        ssim_val = stable_ms_ssim(
                            rendered_masked.unsqueeze(0), target_masked.unsqueeze(0),
                            data_range=1.0, size_average=True,
                        )
                    if torch.isnan(ssim_val):
                        raise RuntimeError(f"NaN detected in MS-SSIM loss at step {layer_step} of layer {p_k}!")
                    loss = loss + w_ssim * (1.0 - ssim_val)
                elif w_ssim > 0:
                    loss = loss + w_ssim * l1
                    
                # LPIPS Loss
                if w_lpips > 0 and lpips_loss_fn is not None:
                    autocast_disable = torch.cuda.amp.autocast(enabled=False) if use_amp else contextlib.nullcontext()
                    with autocast_disable:
                        lpips_val = lpips_loss_fn(rendered_masked, target_masked)
                    if torch.isnan(lpips_val):
                        raise RuntimeError(f"NaN detected in LPIPS loss at step {layer_step} of layer {p_k}!")
                    loss = loss + w_lpips * lpips_val
                    
                # Regularization
                if getattr(args, "lambda_topo", 0.0) > 0.0:
                    topo_loss = compute_topology_loss(layer_scene)
                    if not torch.isnan(topo_loss):
                        loss = loss + args.lambda_topo * topo_loss
                        
                if getattr(args, "lambda_width", 0.0) > 0.0:
                    width_loss = compute_width_loss(layer_scene)
                    if not torch.isnan(width_loss):
                        loss = loss + args.lambda_width * width_loss
                        
                # Style-aware losses for layer decomposition (Task 3.2)
                if args.style_image:
                    if style_vgg_extractor is not None and style_vgg_extractor.has_vgg and getattr(args, "lambda_style", 0.0) > 0.0 and phase_name != "Coarse":
                        if rH >= 32 and rW >= 32:
                            autocast_disable = torch.cuda.amp.autocast(enabled=False) if use_amp else contextlib.nullcontext()
                            with autocast_disable:
                                rendered_batch = rendered_masked.float().unsqueeze(0)
                                rendered_features = style_vgg_extractor(rendered_batch)
                                from curvesplat.style import gram_matrix
                                style_loss = torch.tensor(0.0, device=device)
                                for r_feat, s_gram in zip(rendered_features, style_gram_matrices):
                                    r_gram = gram_matrix(r_feat)
                                    style_loss = style_loss + F.mse_loss(r_gram, s_gram)
                            loss = loss + args.lambda_style * (style_loss / len(rendered_features))

                    if style_palette is not None and getattr(args, "lambda_palette", 0.0) > 0.0:
                        from curvesplat.style import compute_palette_loss
                        style_palette_tensor = torch.from_numpy(style_palette).float().to(device)
                        palette_loss = compute_palette_loss(layer_scene, style_palette_tensor)
                        loss = loss + args.lambda_palette * palette_loss

                    if style_mean_width is not None and getattr(args, "lambda_width_dist", 0.0) > 0.0:
                        from curvesplat.style import compute_stroke_width_loss
                        width_dist_loss = compute_stroke_width_loss(layer_scene, style_mean_width, style_std_width)
                        loss = loss + args.lambda_width_dist * width_dist_loss
                        
            # Backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(layer_optimizer)
                torch.nn.utils.clip_grad_norm_(layer_scene.parameters(), max_norm=10.0)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(layer_scene.parameters(), max_norm=10.0)
                
            # Track EMA opacity and dormancy of each curve
            with torch.no_grad():
                for curve in layer_scene.curves:
                    current_opacity = torch.sigmoid(curve.opacity_raw).item()
                    if not hasattr(curve, "ema_opacity"):
                        curve.ema_opacity = current_opacity
                        curve.low_opacity_steps = 0
                        curve.is_dormant = False
                    else:
                        curve.ema_opacity = 0.9 * curve.ema_opacity + 0.1 * current_opacity
                        
                    if curve.ema_opacity < 0.05:
                        curve.low_opacity_steps += 1
                    else:
                        curve.low_opacity_steps = 0
                        
                    if curve.is_dormant:
                        if current_opacity > 0.1:
                            curve.is_dormant = False
                            curve.low_opacity_steps = 0
                    else:
                        if curve.low_opacity_steps > 50:
                            curve.is_dormant = True
                            
            # Zero gradients for dormant curves
            for curve in layer_scene.curves:
                if getattr(curve, "is_dormant", False):
                    for param in curve.parameters():
                        param.grad = None
                        
            # Optimizer step
            if scaler is not None:
                scaler.step(layer_optimizer)
                scaler.update()
            else:
                layer_optimizer.step()
                
            if device == "cuda":
                torch.cuda.synchronize()
            t_step_end = time.time()
            t_total = t_step_end - t_step_start
            fps = 1.0 / max(t_total, 1e-6)
            
            if layer_step % 50 == 0 or layer_step == 1:
                print(f"    {phase_name:<8}  Step {layer_step:>4}/{max_total_steps}  "
                      f"Loss: {loss.item():>8.5f}  L1: {l1.item():>8.5f}  FPS: {fps:>5.1f}")
                # Export live preview for the frontend
                export_to_svg(layer_scene, os.path.join(args.output_dir, "live_preview.svg"), quantize=True)
                      
            # Adaptive Curve Density inside layer
            if not getattr(args, "no_adaptive", False) and layer_step % 100 == 0 and layer_step < max_total_steps:
                layer_optimizer = adapt_curve_density(
                    scene=layer_scene,
                    renderer=renderer,
                    current_render=rendered.detach(),
                    phase_target=phase_target,
                    phase_lr=phase_lr,
                    target_rgb=target_rgb,
                    orig_H=H,
                    orig_W=W,
                    scale_x=scale_x,
                    scale_y=scale_y,
                    optimizer=layer_optimizer,
                    render_size=render_size,
                    lr_cp_mult=args.lr_cp_mult,
                    max_width_fill=profile["max_width_fill"],
                    max_width_edge=profile["max_width_edge"],
                    crop_box=(ymin, ymax, xmin, xmax),
                )
                
            # Transition phase if plateau or budget limit
            loss_history.append(loss.item())
            loss_history = loss_history[-30:]
            
            trigger_transition = False
            if steps_in_phase >= 50 and len(loss_history) >= 30:
                slope = compute_slope(loss_history)
                if slope > -0.0001:
                    print(f"    [Learned Scheduler] Loss plateaued (slope = {slope:.6f} > -0.0001). Transitioning...")
                    trigger_transition = True
                    
            if steps_in_phase >= current_phase["max_steps"]:
                trigger_transition = True
                
            if trigger_transition and current_phase_idx < len(layer_phases) - 1:
                current_phase_idx += 1
                steps_in_phase = 0
                loss_history.clear()
                
                current_phase = layer_phases[current_phase_idx]
                phase_name = current_phase["name"]
                render_size = current_phase["size"]
                phase_lr = current_phase["lr"]
                rH, rW = render_size
                
                phase_target = F.interpolate(target_crop.unsqueeze(0), size=(rH, rW), mode="bilinear", align_corners=False).squeeze(0)
                phase_mask = F.interpolate(mask_crop.unsqueeze(0).unsqueeze(0), size=(rH, rW), mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
                
                scale_x = rW / crop_w
                scale_y = rH / crop_h
                
                layer_optimizer.state.clear()
                layer_optimizer.param_groups[0]["lr"] = phase_lr * args.lr_cp_mult
                layer_optimizer.param_groups[1]["lr"] = phase_lr
        
        # Save optimized curves of this layer
        print(f"  → Layer '{p_k}' optimization complete.")
        for c in layer_scene.curves:
            global_curves.append(c)

    print("\n" + "=" * 65)
    print(f"✓ Layer-by-layer optimization complete. Total curves: {len(global_curves)}")
    print("=" * 65)
    
    # 4. Create global scene and assign z-indices
    for idx, curve in enumerate(global_curves):
        curve.z_index = idx
        
    scene = Scene(
        curves=global_curves,
        canvas_size=(H, W),
        background=torch.tensor(bg_raw, dtype=torch.float32, device=device),
        device=device,
    )
    
    # Render initial composited scene
    with torch.no_grad():
        before_image = render_scene_shuffled(scene, renderer, shuffle=False)
    before_np = before_image.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
    
    # 5. Global refinement pass
    if args.refinement_steps > 0:
        print("\n" + "=" * 65)
        print(f"✓ Running Global Refinement Pass for {args.refinement_steps} steps at full resolution...")
        print("=" * 65)
        
        # Setup optimizer for all parameters
        control_points_params = []
        other_params = []
        for name, param in scene.named_parameters():
            if "control_points" in name:
                control_points_params.append(param)
            else:
                other_params.append(param)
                
        ref_lr = args.lr * 0.4
        ref_optimizer = torch.optim.AdamW(
            [
                {"params": control_points_params, "lr": ref_lr * args.lr_cp_mult},
                {"params": other_params, "lr": ref_lr},
            ],
            betas=(0.9, 0.999),
            weight_decay=0.005,
        )
        
        phase_target = target_tensor
        render_size = (H, W)
        
        ref_optimizer.state.clear()
        
        for ref_step in range(1, args.refinement_steps + 1):
            if device == "cuda":
                torch.cuda.synchronize()
            t_step_start = time.time()
            
            ref_optimizer.zero_grad()
            
            with autocast_context:
                rendered = render_scene_shuffled(
                    scene, renderer, shuffle=True, render_size=render_size, use_checkpoint=args.checkpoint
                )
                
                if lpips_loss_fn is not None:
                    w_l1, w_ssim, w_lpips = 0.3, 0.4, 0.3
                else:
                    w_l1, w_ssim, w_lpips = 0.6, 0.4, 0.0
                    
                l1 = F.l1_loss(rendered, phase_target)
                loss = w_l1 * l1
                
                if w_ssim > 0 and has_msssim and H > 160 and W > 160:
                    autocast_disable = torch.cuda.amp.autocast(enabled=False) if use_amp else contextlib.nullcontext()
                    with autocast_disable:
                        ssim_val = stable_ms_ssim(
                            rendered.unsqueeze(0), phase_target.unsqueeze(0),
                            data_range=1.0, size_average=True,
                        )
                    loss = loss + w_ssim * (1.0 - ssim_val)
                elif w_ssim > 0:
                    loss = loss + w_ssim * l1
                    
                if w_lpips > 0 and lpips_loss_fn is not None:
                    autocast_disable = torch.cuda.amp.autocast(enabled=False) if use_amp else contextlib.nullcontext()
                    with autocast_disable:
                        lpips_val = lpips_loss_fn(rendered, phase_target)
                    loss = loss + w_lpips * lpips_val
                    
                if getattr(args, "lambda_topo", 0.0) > 0.0:
                    topo_loss = compute_topology_loss(scene)
                    if not torch.isnan(topo_loss):
                        loss = loss + args.lambda_topo * topo_loss
                        
                if getattr(args, "lambda_width", 0.0) > 0.0:
                    width_loss = compute_width_loss(scene)
                    if not torch.isnan(width_loss):
                        loss = loss + args.lambda_width * width_loss

                # Style-aware losses for global refinement (Task 3.2)
                if args.style_image:
                    if style_vgg_extractor is not None and style_vgg_extractor.has_vgg and getattr(args, "lambda_style", 0.0) > 0.0:
                        autocast_disable = torch.cuda.amp.autocast(enabled=False) if use_amp else contextlib.nullcontext()
                        with autocast_disable:
                            rendered_batch = rendered.float().unsqueeze(0)
                            rendered_features = style_vgg_extractor(rendered_batch)
                            from curvesplat.style import gram_matrix
                            style_loss = torch.tensor(0.0, device=device)
                            for r_feat, s_gram in zip(rendered_features, style_gram_matrices):
                                r_gram = gram_matrix(r_feat)
                                style_loss = style_loss + F.mse_loss(r_gram, s_gram)
                        loss = loss + args.lambda_style * (style_loss / len(rendered_features))

                    if style_palette is not None and getattr(args, "lambda_palette", 0.0) > 0.0:
                        from curvesplat.style import compute_palette_loss
                        style_palette_tensor = torch.from_numpy(style_palette).float().to(device)
                        palette_loss = compute_palette_loss(scene, style_palette_tensor)
                        loss = loss + args.lambda_palette * palette_loss

                    if style_mean_width is not None and getattr(args, "lambda_width_dist", 0.0) > 0.0:
                        from curvesplat.style import compute_stroke_width_loss
                        width_dist_loss = compute_stroke_width_loss(scene, style_mean_width, style_std_width)
                        loss = loss + args.lambda_width_dist * width_dist_loss
                        
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(ref_optimizer)
                torch.nn.utils.clip_grad_norm_(scene.parameters(), max_norm=10.0)
                scaler.step(ref_optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(scene.parameters(), max_norm=10.0)
                ref_optimizer.step()
                
            if device == "cuda":
                torch.cuda.synchronize()
            t_step_end = time.time()
            fps = 1.0 / max(t_step_end - t_step_start, 1e-6)
            
            if ref_step % 20 == 0 or ref_step == 1:
                print(f"    Refine Step {ref_step:>3}/{args.refinement_steps}  "
                      f"Loss: {loss.item():>8.5f}  L1: {l1.item():>8.5f}  FPS: {fps:>5.1f}")
        
        print(f"✓ Global refinement complete.")

    return scene, before_np

def main():
    parser = argparse.ArgumentParser(description="CurveSplat GPU Differentiable Vectorization v2")
    parser.add_argument("--image", type=str, required=True, help="Path to input target image")
    parser.add_argument("--output_dir", type=str, default="examples/output", help="Output directory")
    parser.add_argument("--steps", type=int, default=500, help="Total optimization steps")
    parser.add_argument("--lr", type=float, default=0.03, help="Peak learning rate")
    parser.add_argument("--num_curves", type=int, default=200,
                        help="Total curves (region fill + edge contours)")
    parser.add_argument("--n_fill", type=int, default=100,
                        help="Number of wide color-region fill curves")
    parser.add_argument("--n_edge", type=int, default=100,
                        help="Max edge contour curves from SDF init")
    parser.add_argument("--width", type=float, default=2.0, help="Default edge stroke width")
    parser.add_argument("--max_dim", type=int, default=512, help="Max image dimension")
    parser.add_argument("--num_samples", type=int, default=16,
                        help="Bezier sample count per curve (default: 16 for speed/quality balance)")
    parser.add_argument("--no_coarse", action="store_true",
                        help="Skip coarse-to-fine schedule (recommended default is coarse-to-fine)")
    parser.add_argument("--save_every", type=int, default=100,
                        help="Save intermediate comparison every N steps (0=disable)")
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable Mixed Precision (AMP) training")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt to guide curve initialization via CLIP similarity")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run 10 warmup + 20 benchmark steps and report FPS, then exit")
    parser.add_argument("--no_adaptive", action="store_true",
                        help="Disable adaptive curve density (spawn/remove curves)")
    parser.add_argument("--checkpoint", action="store_true",
                        help="Enable gradient checkpointing (trade-off speed for GPU memory)")
    parser.add_argument("--max_width_fill", type=float, default=15.0,
                        help="Maximum stroke width limit for region fill curves")
    parser.add_argument("--max_width_edge", type=float, default=3.0,
                        help="Maximum stroke width limit for edge contour curves")
    parser.add_argument("--lr_cp_mult", type=float, default=2.0,
                        help="Learning rate multiplier for control points (reduced to prevent desert drift)")
    parser.add_argument("--lambda_topo", type=float, default=0.01,
                        help="Topology regularization loss weight to prevent self-intersections")
    parser.add_argument("--lambda_width", type=float, default=0.002,
                        help="L2 regularization weight on stroke widths to prevent bloating")
    parser.add_argument("--segment_prompts", type=str, default=None,
                        help="Comma-separated text prompts to guide semantic layer decomposition (e.g. 'background, body, wings')")
    parser.add_argument("--refinement_steps", type=int, default=100,
                        help="Number of steps for final global joint refinement")
    parser.add_argument("--quantize_svg", action="store_true",
                        help="Quantize SVG output coordinates to integers (default: 1 decimal place)")
    parser.add_argument("--style_image", type=str, default=None,
                        help="Path to style reference image for style-aware vectorization")
    parser.add_argument("--lambda_style", type=float, default=50.0,
                        help="Gram matrix loss weight for style guidance")
    parser.add_argument("--lambda_palette", type=float, default=1.0,
                        help="Color palette mapping loss weight for style guidance")
    parser.add_argument("--lambda_width_dist", type=float, default=1.0,
                        help="Stroke width distribution matching loss weight for style guidance")
    parser.add_argument("--proxy_weights", type=str, default=None,
                        help="Path to trained neural proxy weights (.pt file)")
    parser.add_argument("--preview_only", action="store_true",
                        help="Run quick neural preview mode using proxy and exit immediately after initialization")
    parser.add_argument("--fixed_scheduler", action="store_true",
                        help="Disable learned resolution transitions and rely strictly on static step limits")
    # --- Crisp / hybrid output controls ---
    parser.add_argument("--mode", type=str, default="hybrid", choices=["hybrid", "painterly"],
                        help="hybrid: crisp filled-region base layer + painterly strokes (default). "
                             "painterly: original stroke-only behavior.")
    parser.add_argument("--n_regions", type=int, default=12,
                        help="Number of flat color regions for the filled base layer (hybrid mode)")
    parser.add_argument("--lambda_edge", type=float, default=0.5,
                        help="Weight of the Sobel edge-sharpness loss (Mid/Fine phases)")
    parser.add_argument("--polish_steps", type=int, default=150,
                        help="Stage-6 exact-boundary polish steps at full res (0 to disable)")
    parser.add_argument("--ssim_gate", type=float, default=0.95,
                        help="Skip polish if MS-SSIM already exceeds this")
    parser.add_argument("--log_every", type=int, default=10,
                        help="Print a metrics line every N steps (1 = every step)")
    parser.add_argument("--svg_min_opacity", type=float, default=0.1,
                        help="SVG export: drop strokes fainter than this (raise for cleaner output)")
    parser.add_argument("--svg_min_area", type=float, default=8.0,
                        help="SVG export: drop strokes with width×length below this (prunes scratches)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 65)
    print("  CurveSplat v2 — Improved GPU Multi-Curve Optimization")
    print("=" * 65)
    print(f"  Target Image:  {args.image}")
    if args.style_image:
        print(f"  Style Image:   {args.style_image}")
    print(f"  Device:        {device.upper()}")
    if device == "cuda":
        print(f"  GPU:           {torch.cuda.get_device_name(0)}")
    print(f"  Steps:         {args.steps}")
    print(f"  Fill curves:   {args.n_fill}  |  Edge curves: {args.n_edge}")
    print(f"  Peak LR:       {args.lr}")
    print(f"  Bezier samps:  {args.num_samples}")
    if args.prompt:
        print(f"  Prompt:        {args.prompt}")
    print(f"  AMP Enabled:   {not args.no_amp and device == 'cuda'}")
    print(f"  Adaptive:      {not args.no_adaptive}")
    print(f"  Checkpoint:    {args.checkpoint}")
    if args.style_image:
        print(f"  Style Weights: Style={args.lambda_style} | Palette={args.lambda_palette} | Width={args.lambda_width_dist}")
    print("=" * 65)

    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Load and (optionally) resize image
    # -----------------------------------------------------------------------
    try:
        target_np = plt.imread(args.image)
        if target_np.dtype == np.uint8:
            target_np = target_np.astype(np.float64) / 255.0
        else:
            target_np = target_np.astype(np.float64)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Ensure 3 channels
    if target_np.ndim == 2:
        target_np = np.stack([target_np] * 3, axis=2)
    elif target_np.shape[2] == 4:
        # Alpha-composite on white
        rgb = target_np[:, :, :3]
        alpha = target_np[:, :, 3:4]
        target_np = rgb * alpha + (1.0 - alpha)

    H, W = target_np.shape[:2]
    if max(H, W) > args.max_dim:
        scale = args.max_dim / max(H, W)
        new_H, new_W = int(H * scale), int(W * scale)
        t_tensor = torch.from_numpy(target_np[:, :, :3]).permute(2, 0, 1).unsqueeze(0).float()
        t_resized = F.interpolate(t_tensor, size=(new_H, new_W), mode="bilinear", align_corners=False)
        target_np = t_resized.squeeze(0).permute(1, 2, 0).numpy().clip(0, 1)
        H, W = new_H, new_W
        print(f"✓ Resized to {W}×{H}")

    target_rgb = target_np[:, :, :3].astype(np.float32)
    target_tensor = torch.from_numpy(target_rgb).permute(2, 0, 1).float().to(device)

    # -----------------------------------------------------------------------
    # 1.5 Load Style Reference (if provided)
    # -----------------------------------------------------------------------
    style_palette = None
    style_mean_width = None
    style_std_width = None
    style_vgg_extractor = None
    style_gram_matrices = None

    if args.style_image:
        print("✓ Loading and analyzing style reference image...")
        try:
            from curvesplat.style import (
                extract_dominant_palette,
                analyze_stroke_widths,
                VGGStyleExtractor,
                gram_matrix,
            )
            style_np = plt.imread(args.style_image)
            if style_np.dtype == np.uint8:
                style_np = style_np.astype(np.float32) / 255.0
            else:
                style_np = style_np.astype(np.float32)
            if style_np.ndim == 2:
                style_np = np.stack([style_np] * 3, axis=2)
            elif style_np.shape[2] == 4:
                style_np = style_np[:, :, :3]
            
            # Analyze dominant palette and stroke widths
            style_palette = extract_dominant_palette(style_np, k=8)
            style_mean_width, style_std_width = analyze_stroke_widths(style_np)
            print(f"  → Style palette extracted ({len(style_palette)} colors)")
            print(f"  → Style stroke widths analyzed: mean={style_mean_width:.2f}px, std={style_std_width:.2f}px")

            # Set up VGG feature extraction
            style_vgg_extractor = VGGStyleExtractor(device=device)
            if style_vgg_extractor.has_vgg:
                # Downsample style image for faster VGG pass (max size 256)
                s_H, s_W = style_np.shape[:2]
                s_max = 256
                if max(s_H, s_W) > s_max:
                    s_scale = s_max / max(s_H, s_W)
                    new_s_H, new_s_W = int(s_H * s_scale), int(s_W * s_scale)
                else:
                    new_s_H, new_s_W = s_H, s_W

                style_tensor = torch.from_numpy(style_np[:, :, :3]).permute(2, 0, 1).unsqueeze(0).float().to(device)
                style_resized = F.interpolate(
                    style_tensor, size=(new_s_H, new_s_W), mode="bilinear", align_corners=False
                )
                
                with torch.no_grad():
                    style_features = style_vgg_extractor(style_resized)
                    style_gram_matrices = [gram_matrix(feat) for feat in style_features]
                print(f"  → VGG style Gram matrices pre-computed at resolution {new_s_W}x{new_s_H}")
            else:
                print("  → VGG style Gram matrix matching disabled (using palette/width matching only)")
        except Exception as e:
            print(f"  → Warning: Failed to load/analyze style image: {e}")

    # -----------------------------------------------------------------------
    # 1.6 Load Neural Proxy (if weights provided)
    # -----------------------------------------------------------------------
    proxy_model = None
    if args.proxy_weights:
        print("✓ Loading Neural Proxy network...")
        try:
            from curvesplat.proxy import LightweightUNet
            proxy_model = LightweightUNet(in_channels=6, out_channels=3).to(device)
            state_dict = torch.load(args.proxy_weights, map_location=device)
            proxy_model.load_state_dict(state_dict)
            proxy_model.eval()
            print(f"  → Loaded proxy weights from {args.proxy_weights}")
        except Exception as e:
            print(f"  → Warning: Failed to load neural proxy weights: {e}")
            proxy_model = None

    if args.segment_prompts:
        # Load LPIPS perceptual loss model
        lpips_loss_fn = None
        if HAS_LPIPS:
            try:
                print("✓ Loading LPIPS perceptual loss model...")
                lpips_loss_fn = LPIPSLoss(net="vgg", device=device)
                print("  → LPIPS loaded successfully")
            except Exception as e:
                print(f"  → Warning: Failed to initialize LPIPS: {e}")

        scene, before_np = run_semantic_decomposition(
            args, device, target_rgb, target_tensor, lpips_loss_fn,
            style_palette=style_palette,
            style_mean_width=style_mean_width,
            style_std_width=style_std_width,
            style_vgg_extractor=style_vgg_extractor,
            style_gram_matrices=style_gram_matrices,
        )
        
        # Render final image and save outputs
        renderer = CurveSplatRenderer(num_samples=args.num_samples)
        with torch.no_grad():
            after_image = render_scene_shuffled(scene, renderer, shuffle=False)
        after_np = after_image.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(target_rgb)
        axes[0].set_title("Target Image", color="#D7FE3D", fontweight="bold", fontsize=14)
        axes[0].axis("off")

        axes[1].imshow(before_np)
        axes[1].set_title("Initialized Curves", color="#D7FE3D", fontweight="bold", fontsize=14)
        axes[1].axis("off")

        axes[2].imshow(after_np)
        axes[2].set_title(f"Optimized Curves ({args.steps} steps)", color="#D7FE3D",
                          fontweight="bold", fontsize=14)
        axes[2].axis("off")

        fig.patch.set_facecolor("#0A0A0A")
        for ax in axes:
            ax.set_facecolor("#0A0A0A")
        plt.tight_layout()

        comparison_path = os.path.join(args.output_dir, "cuda_vectorization_comparison.png")
        plt.savefig(comparison_path, dpi=150, facecolor="#0A0A0A", bbox_inches="tight")
        plt.close()

        svg_path = os.path.join(args.output_dir, "optimized_artwork.svg")
        export_to_svg(scene, svg_path, quantize=args.quantize_svg, min_opacity=args.svg_min_opacity, min_area=args.svg_min_area)

        print(f"✓ Comparison: {comparison_path}")
        print(f"✓ SVG export: {svg_path}")
        print("=" * 65)
        return

    # -----------------------------------------------------------------------
    # 2. Hybrid initialization: region fill + edge contour curves
    # -----------------------------------------------------------------------
    print("✓ Running k-means color clustering for region fill curves...")
    fill_curves = initialize_region_fill_curves(
        target_rgb,
        n_region_curves=args.n_fill,
        n_color_clusters=min(16, args.n_fill // 4 + 2),
        device=device,
        prompt=args.prompt,
        style_palette=style_palette,
    )
    for c in fill_curves:
        c.max_width = min(args.max_width_fill, style_mean_width * 1.5) if style_mean_width is not None else args.max_width_fill
    print(f"  → Created {len(fill_curves)} region fill curves (max width clamped to {fill_curves[0].max_width if fill_curves else args.max_width_fill:.1f}px)")

    print("✓ Running SDF contour extraction for edge curves...")
    edge_scene = initialize_curves_from_image(
        target_rgb,
        canvas_size=(H, W),
        is_binary=False,
        min_curves=10,
        max_curves=args.n_edge,
        default_width=args.width,
        simplify_epsilon=1.5,
        corner_angle_threshold=30.0,
        device=device,
    )
    edge_curves = list(edge_scene.curves)
    for c in edge_curves:
        c.max_width = min(args.max_width_edge, style_mean_width * 0.8) if style_mean_width is not None else args.max_width_edge
    print(f"  → Extracted {len(edge_curves)} edge contour curves (max width clamped to {edge_curves[0].max_width if edge_curves else args.max_width_edge:.1f}px)")

    # Hybrid mode: crisp filled-region base layer (tracer-style) underneath the
    # painterly strokes. This is what closes most of the gap with contour tracers
    # on flat / clean art while keeping the splat engine for gradients & texture.
    filled_shapes = []
    if args.mode == "hybrid":
        print("✓ Extracting crisp filled regions (hybrid base layer)...")
        regions = initialize_filled_regions(target_rgb, n_colors=args.n_regions)
        filled_shapes = [FilledShape.from_region(r, device=device) for r in regions]
        print(f"  → Created {len(filled_shapes)} filled regions (z-ordered by area)")

    # Combine: filled base (lowest z) → region-fill strokes → edge details
    all_curves = filled_shapes + fill_curves + edge_curves
    # Re-assign z_indices
    for i, c in enumerate(all_curves):
        c.z_index = i

    # Background: sample the image corners
    corner_pixels = [
        target_rgb[0, 0], target_rgb[0, -1], target_rgb[-1, 0], target_rgb[-1, -1]
    ]
    bg_rgb = np.mean(corner_pixels, axis=0).clip(0.02, 0.98)
    if style_palette is not None:
        # Map bg color to nearest color in style palette
        dists = np.sum((style_palette - bg_rgb) ** 2, axis=1)
        bg_rgb = style_palette[dists.argmin()]
    bg_raw = np.log(bg_rgb / (1.0 - bg_rgb)).astype(np.float32)

    scene = Scene(
        curves=all_curves,
        canvas_size=(H, W),
        background=torch.tensor(bg_raw, dtype=torch.float32, device=device),
        device=device,
    )
    # Fill-edge sharpness state (annealed coarse→fine in the optimization loop).
    scene.fill_k = 0.5
    scene.fill_samples = 12  # boundary points/segment; fill autograd mem ∝ this·#seg·H·W
    print(f"✓ Scene ready: {len(scene.curves)} total curves "
          f"({len(filled_shapes)} filled + {len(fill_curves) + len(edge_curves)} strokes) on {device.upper()}")

    # -----------------------------------------------------------------------
    # 3. Render "before" image (initialized, no optimization)
    # -----------------------------------------------------------------------
    renderer = CurveSplatRenderer(num_samples=args.num_samples)
    with torch.no_grad():
        before_image = render_scene_shuffled(scene, renderer, shuffle=False)
    before_np = before_image.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)

    # -----------------------------------------------------------------------
    # 3.5 Quick Neural Preview Mode (if preview_only)
    # -----------------------------------------------------------------------
    if args.preview_only:
        if proxy_model is None:
            print("  [ERROR] Cannot run --preview_only because neural proxy weights were not successfully loaded!")
            return
        
        print("✓ Running quick neural preview...")
        t_preview_start = time.time()
        with torch.no_grad():
            proxy_input = torch.cat([target_tensor, before_image], dim=0).unsqueeze(0)
            orig_H, orig_W = H, W
            if (H, W) != (256, 256):
                proxy_input_resized = F.interpolate(proxy_input, size=(256, 256), mode="bilinear", align_corners=False)
                predicted_resized = proxy_model(proxy_input_resized)
                predicted_tensor = F.interpolate(predicted_resized, size=(orig_H, orig_W), mode="bilinear", align_corners=False)
            else:
                predicted_tensor = proxy_model(proxy_input)
                
            predicted_np = predicted_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1)
            
        t_preview_end = time.time()
        print(f"  → Neural proxy inference completed in {(t_preview_end - t_preview_start)*1000:.2f}ms")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(target_rgb)
        axes[0].set_title("Target Image", color="#D7FE3D", fontweight="bold", fontsize=14)
        axes[0].axis("off")

        axes[1].imshow(before_np)
        axes[1].set_title("Initialized Curves", color="#D7FE3D", fontweight="bold", fontsize=14)
        axes[1].axis("off")

        axes[2].imshow(predicted_np)
        axes[2].set_title("Neural Proxy Preview", color="#D7FE3D", fontweight="bold", fontsize=14)
        axes[2].axis("off")

        fig.patch.set_facecolor("#0A0A0A")
        for ax in axes:
            ax.set_facecolor("#0A0A0A")
        plt.tight_layout()

        comparison_path = os.path.join(args.output_dir, "neural_preview_comparison.png")
        plt.savefig(comparison_path, dpi=150, facecolor="#0A0A0A", bbox_inches="tight")
        plt.close()

        predicted_path = os.path.join(args.output_dir, "predicted_artwork.png")
        plt.imsave(predicted_path, predicted_np)

        svg_path = os.path.join(args.output_dir, "initialized_artwork.svg")
        export_to_svg(scene, svg_path, quantize=args.quantize_svg, min_opacity=args.svg_min_opacity, min_area=args.svg_min_area)

        print(f"✓ Neural Preview Comparison saved: {comparison_path}")
        print(f"✓ Predicted Artwork saved: {predicted_path}")
        print(f"✓ SVG export (initialized): {svg_path}")
        print("=" * 65)
        return

    # -----------------------------------------------------------------------
    # 4. Coarse-to-fine resolution schedule
    # -----------------------------------------------------------------------
    if args.no_coarse:
        # Single resolution: full size for all steps
        schedule = [(args.steps, (H, W), args.lr)]
    else:
        # 3-tier coarse-to-fine schedule
        coarse_H, coarse_W = max(32, H // 8), max(32, W // 8)
        mid_H, mid_W = max(64, H // 4), max(64, W // 4)
        fine_H, fine_W = H, W

        total = args.steps
        coarse_steps = total // 4       # 25%: very coarse structure
        mid_steps    = total // 4       # 25%: mid detail
        fine_steps   = total - coarse_steps - mid_steps  # 50%: full detail

        schedule = [
            (coarse_steps, (coarse_H, coarse_W), args.lr * 1.5),
            (mid_steps,    (mid_H, mid_W),        args.lr),
            (fine_steps,   (fine_H, fine_W),       args.lr * 0.4),
        ]

    # -----------------------------------------------------------------------
    # 5. Optimizer and AMP Setup
    # -----------------------------------------------------------------------
    # Mixed Precision (AMP) setup
    is_cuda = (device == "cuda")
    use_amp = is_cuda and not getattr(args, "no_amp", False)
    autocast_context = torch.cuda.amp.autocast() if use_amp else contextlib.nullcontext()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    lpips_loss_fn = None
    if HAS_LPIPS:
        try:
            print("✓ Loading LPIPS perceptual loss model...")
            lpips_loss_fn = LPIPSLoss(net="vgg", device=device)
            print("  → LPIPS loaded successfully")
        except Exception as e:
            print(f"  → Warning: Failed to initialize LPIPS: {e}")

    # Separate control points from other parameters to use a higher learning rate
    control_points_params = []
    other_params = []
    for name, param in scene.named_parameters():
        if "control_points" in name:
            control_points_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": control_points_params, "lr": args.lr * args.lr_cp_mult},
            {"params": other_params, "lr": args.lr},
        ],
        betas=(0.9, 0.999),
        weight_decay=0.005,
    )

    # -----------------------------------------------------------------------
    # 5.5 Benchmark mode (if requested)
    # -----------------------------------------------------------------------
    if getattr(args, "benchmark", False):
        print("=" * 65)
        print("  Running CurveSplat Benchmark (10 Warmup + 20 Timed Steps)")
        print("=" * 65)
        
        bench_size = schedule[0][1]
        bench_rH, bench_rW = bench_size
        if (bench_rH, bench_rW) != (H, W):
            bench_target = F.interpolate(target_tensor.unsqueeze(0), size=(bench_rH, bench_rW), mode="bilinear", align_corners=False).squeeze(0)
        else:
            bench_target = target_tensor

        # Warmup
        for _ in range(10):
            optimizer.zero_grad()
            with autocast_context:
                rendered = render_scene_shuffled(scene, renderer, shuffle=True, render_size=bench_size, use_checkpoint=args.checkpoint)
                loss = F.l1_loss(rendered, bench_target)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                
        if device == "cuda":
            torch.cuda.synchronize()
            
        # Timed steps
        t_start = time.time()
        for _ in range(20):
            optimizer.zero_grad()
            with autocast_context:
                rendered = render_scene_shuffled(scene, renderer, shuffle=True, render_size=bench_size, use_checkpoint=args.checkpoint)
                loss = F.l1_loss(rendered, bench_target)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                
        if device == "cuda":
            torch.cuda.synchronize()
        t_end = time.time()
        
        total_time = t_end - t_start
        avg_fps = 20 / max(total_time, 1e-6)
        print(f"✓ Benchmark Complete!")
        print(f"  Average FPS:  {avg_fps:.2f}")
        print(f"  Total Time:   {total_time:.3f}s for 20 steps")
        print("=" * 65)
        return

    # -----------------------------------------------------------------------
    # 6. Training loop with coarse-to-fine
    # -----------------------------------------------------------------------
    try:
        from pytorch_msssim.ssim import _ssim, _fspecial_gauss_1d
        
        def stable_ms_ssim(X: torch.Tensor, Y: torch.Tensor, data_range: float = 1.0, size_average: bool = True, K = (0.01, 0.03)) -> torch.Tensor:
            win_size = 11
            win_sigma = 1.5
            win = _fspecial_gauss_1d(win_size, win_sigma)
            win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))
            weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
            weights_tensor = X.new_tensor(weights)
            levels = len(weights)
            mcs = []
            for i in range(levels):
                ssim_per_channel, cs = _ssim(X, Y, win=win, data_range=data_range, size_average=False, K=K)
                if i < levels - 1:
                    # Clamp to min=1e-4 to prevent gradient NaN/explosion during backpropagation when cs is negative/zero
                    mcs.append(torch.clamp(cs, min=1e-4))
                    padding = [s % 2 for s in X.shape[2:]]
                    X = F.avg_pool2d(X, kernel_size=2, padding=padding)
                    Y = F.avg_pool2d(Y, kernel_size=2, padding=padding)
            ssim_per_channel = torch.clamp(ssim_per_channel, min=1e-4)
            mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)
            ms_ssim_val = torch.prod(mcs_and_ssim ** weights_tensor.view(-1, 1, 1), dim=0)
            if size_average:
                return ms_ssim_val.mean()
            else:
                return ms_ssim_val.mean(1)

        HAS_MSSSIM = True
    except ImportError:
        HAS_MSSSIM = False
        print("  (pytorch_msssim not found — using L1+L2 only)")

    print(f"\n{'Phase':<10} {'Step':>6}  {'Loss':>10}  {'L1':>8}  {'FPS':>8}")
    print("-" * 50)

    # Setup learned resolution scheduler (Task 2.1)
    if args.no_coarse:
        phases = [
            {"name": "Full", "size": (H, W), "lr": args.lr, "max_steps": args.steps}
        ]
    else:
        coarse_H, coarse_W = max(32, H // 8), max(32, W // 8)
        mid_H, mid_W = max(64, H // 4), max(64, W // 4)
        coarse_steps = args.steps // 4
        mid_steps = args.steps // 4
        fine_steps = args.steps - coarse_steps - mid_steps
        phases = [
            {"name": "Coarse", "size": (coarse_H, coarse_W), "lr": args.lr * 1.5, "max_steps": coarse_steps},
            {"name": "Mid", "size": (mid_H, mid_W), "lr": args.lr, "max_steps": mid_steps},
            {"name": "Fine", "size": (H, W), "lr": args.lr * 0.4, "max_steps": fine_steps}
        ]

    def compute_slope(history):
        n = len(history)
        if n < 2:
            return 0.0
        x = np.arange(n)
        y = np.array(history)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        if denominator < 1e-8:
            return 0.0
        return float(numerator / denominator)

    current_phase_idx = 0
    steps_in_phase = 0
    loss_history = []

    current_phase = phases[current_phase_idx]
    phase_name = current_phase["name"]
    render_size = current_phase["size"]
    phase_lr = current_phase["lr"]
    rH, rW = render_size

    # Target and scaling initialization
    if (rH, rW) != (H, W):
        t4d = target_tensor.unsqueeze(0)
        phase_target = F.interpolate(t4d, size=(rH, rW), mode="bilinear", align_corners=False).squeeze(0)
    else:
        phase_target = target_tensor
    scale_x = rW / W
    scale_y = rH / H

    # Update optimizer learning rates
    optimizer.state.clear()
    optimizer.param_groups[0]["lr"] = phase_lr * args.lr_cp_mult
    optimizer.param_groups[1]["lr"] = phase_lr

    global_step = 0
    start_time = time.time()

    while global_step < args.steps:
        global_step += 1
        steps_in_phase += 1
        
        if device == "cuda":
            torch.cuda.synchronize()
        t_step_start = time.time()

        optimizer.zero_grad()

        with autocast_context:
            rendered = render_scene_shuffled(
                scene, renderer, shuffle=True, render_size=render_size, use_checkpoint=args.checkpoint
            )
            
            # Determine loss weights dynamically based on phase
            if phase_name == "Coarse":
                w_l1, w_ssim, w_lpips = 1.0, 0.0, 0.0
            elif phase_name == "Mid":
                w_l1, w_ssim, w_lpips = 0.5, 0.5, 0.0
            else:  # Fine or Full
                if lpips_loss_fn is not None:
                    w_l1, w_ssim, w_lpips = 0.3, 0.4, 0.3
                else:
                    w_l1, w_ssim, w_lpips = 0.6, 0.4, 0.0

            # Anneal filled-region edge sharpness coarse→fine (soft early, crisp late).
            scene.fill_k = {"Coarse": 0.4, "Mid": 1.0}.get(phase_name, 2.5)

            l1 = F.l1_loss(rendered, phase_target)
            if torch.isnan(l1):
                raise RuntimeError(f"NaN detected in L1 loss at step {global_step}!")

            loss = w_l1 * l1

            # MS-SSIM Loss (requires resolution > 160)
            if w_ssim > 0 and HAS_MSSSIM and rH > 160 and rW > 160:
                autocast_disable = torch.cuda.amp.autocast(enabled=False) if use_amp else contextlib.nullcontext()
                with autocast_disable:
                    ssim_val = stable_ms_ssim(
                        rendered.float().unsqueeze(0), phase_target.float().unsqueeze(0),
                        data_range=1.0, size_average=True,
                    )
                if torch.isnan(ssim_val):
                    raise RuntimeError(f"NaN detected in MS-SSIM loss at step {global_step}!")
                loss = loss + w_ssim * (1.0 - ssim_val)
            elif w_ssim > 0:
                loss = loss + w_ssim * l1

            # LPIPS Loss
            if w_lpips > 0 and lpips_loss_fn is not None:
                autocast_disable = torch.cuda.amp.autocast(enabled=False) if use_amp else contextlib.nullcontext()
                with autocast_disable:
                    lpips_val = lpips_loss_fn(rendered, phase_target)
                if torch.isnan(lpips_val):
                    raise RuntimeError(f"NaN detected in LPIPS loss at step {global_step}!")
                loss = loss + w_lpips * lpips_val

            # Edge-sharpness Loss — push crisp boundaries (Mid/Fine only).
            if getattr(args, "lambda_edge", 0.0) > 0.0 and phase_name != "Coarse":
                autocast_disable = torch.cuda.amp.autocast(enabled=False) if use_amp else contextlib.nullcontext()
                with autocast_disable:
                    edge_l = compute_edge_loss(rendered.float(), phase_target.float())
                if not torch.isnan(edge_l):
                    loss = loss + args.lambda_edge * edge_l

            # Topology Regularization Loss (Task 2.2)
            if getattr(args, "lambda_topo", 0.0) > 0.0:
                topo_loss = compute_topology_loss(scene)
                if not torch.isnan(topo_loss):
                    loss = loss + args.lambda_topo * topo_loss

            # Width Regularization Loss (Task 2.5)
            if getattr(args, "lambda_width", 0.0) > 0.0:
                width_loss = compute_width_loss(scene)
                if not torch.isnan(width_loss):
                    loss = loss + args.lambda_width * width_loss

            # Style-aware losses for standard optimization (Task 3.2)
            if args.style_image:
                if style_vgg_extractor is not None and style_vgg_extractor.has_vgg and getattr(args, "lambda_style", 0.0) > 0.0 and phase_name != "Coarse":
                    if rH >= 32 and rW >= 32:
                        autocast_disable = torch.cuda.amp.autocast(enabled=False) if use_amp else contextlib.nullcontext()
                        with autocast_disable:
                            rendered_batch = rendered.float().unsqueeze(0)
                            rendered_features = style_vgg_extractor(rendered_batch)
                            from curvesplat.style import gram_matrix
                            style_loss = torch.tensor(0.0, device=device)
                            for r_feat, s_gram in zip(rendered_features, style_gram_matrices):
                                r_gram = gram_matrix(r_feat)
                                style_loss = style_loss + F.mse_loss(r_gram, s_gram)
                        loss = loss + args.lambda_style * (style_loss / len(rendered_features))

                if style_palette is not None and getattr(args, "lambda_palette", 0.0) > 0.0:
                    from curvesplat.style import compute_palette_loss
                    style_palette_tensor = torch.from_numpy(style_palette).float().to(device)
                    palette_loss = compute_palette_loss(scene, style_palette_tensor)
                    loss = loss + args.lambda_palette * palette_loss

                if style_mean_width is not None and getattr(args, "lambda_width_dist", 0.0) > 0.0:
                    from curvesplat.style import compute_stroke_width_loss
                    width_dist_loss = compute_stroke_width_loss(scene, style_mean_width, style_std_width)
                    loss = loss + args.lambda_width_dist * width_dist_loss

        if device == "cuda":
            torch.cuda.synchronize()
        t_fwd_end = time.time()

        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            
            # Check for NaN gradients
            has_nan_grad = False
            nan_params = []
            for name, p in scene.named_parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    has_nan_grad = True
                    nan_params.append(name)
            if has_nan_grad:
                print(f"  [ERROR] NaN detected in gradients (with scaler) at step {global_step} for parameters: {nan_params}")
                raise RuntimeError(f"NaN detected in gradients (with scaler) at step {global_step}!")
                
            torch.nn.utils.clip_grad_norm_(scene.parameters(), max_norm=10.0)
        else:
            loss.backward()
            
            # Check for NaN gradients
            has_nan_grad = False
            nan_params = []
            for name, p in scene.named_parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    has_nan_grad = True
                    nan_params.append(name)
            if has_nan_grad:
                print(f"  [ERROR] NaN detected in gradients (no scaler) at step {global_step} for parameters: {nan_params}")
                raise RuntimeError(f"NaN detected in gradients (no scaler) at step {global_step}!")
                
            torch.nn.utils.clip_grad_norm_(scene.parameters(), max_norm=10.0)

        # Track EMA opacity and dormancy of each curve (Task 1.2)
        with torch.no_grad():
            for curve in scene.curves:
                current_opacity = torch.sigmoid(curve.opacity_raw).item()
                if not hasattr(curve, "ema_opacity"):
                    curve.ema_opacity = current_opacity
                    curve.low_opacity_steps = 0
                    curve.is_dormant = False
                else:
                    curve.ema_opacity = 0.9 * curve.ema_opacity + 0.1 * current_opacity

                if curve.ema_opacity < 0.05:
                    curve.low_opacity_steps += 1
                else:
                    curve.low_opacity_steps = 0

                if curve.is_dormant:
                    if current_opacity > 0.1:
                        curve.is_dormant = False
                        curve.low_opacity_steps = 0
                else:
                    if curve.low_opacity_steps > 50:
                        curve.is_dormant = True

        # Zero out gradients for dormant curves
        for curve in scene.curves:
            if getattr(curve, "is_dormant", False):
                for p in curve.parameters():
                    p.grad = None

        if device == "cuda":
            torch.cuda.synchronize()
        t_bwd_end = time.time()

        # Optimizer step
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        if device == "cuda":
            torch.cuda.synchronize()
        t_step_end = time.time()

        t_fwd = t_fwd_end - t_step_start
        t_bwd = t_bwd_end - t_fwd_end
        t_opt = t_step_end - t_bwd_end
        t_total = t_step_end - t_step_start
        fps = 1.0 / max(t_total, 1e-6)

        # Performance logging every 100 steps
        if device == "cuda" and global_step % 100 == 0:
            mem_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
            print(f"  [GPU Mem] Step {global_step}: {mem_mb:.1f} MB")

        if global_step % args.log_every == 0 or global_step == 1:
            print(f"  {phase_name:<8}  {global_step:>4}  {loss.item():>10.5f}  "
                  f"{l1.item():>8.5f}  {fps:>8.1f}  "
                  f"(Fwd: {t_fwd*1000:.1f}ms, Bwd: {t_bwd*1000:.1f}ms, Opt: {t_opt*1000:.1f}ms)")

        if global_step % 50 == 0 or global_step == 1:
            # Periodic proxy preview (kept at 50-step cadence regardless of log_every)
            if proxy_model is not None:
                with torch.no_grad():
                    proxy_input = torch.cat([phase_target, rendered.detach()], dim=0).unsqueeze(0)
                    if proxy_input.shape[2:] != (256, 256):
                        proxy_input_resized = F.interpolate(proxy_input, size=(256, 256), mode="bilinear", align_corners=False)
                        predicted_resized = proxy_model(proxy_input_resized)
                        predicted_tensor = F.interpolate(predicted_resized, size=(H, W), mode="bilinear", align_corners=False)
                    else:
                        predicted_tensor = proxy_model(proxy_input)
                    predicted_np = predicted_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1)
                    preview_path = os.path.join(args.output_dir, "proxy_preview.png")
                    plt.imsave(preview_path, predicted_np)

        # Adaptive Curve Density
        if not getattr(args, "no_adaptive", False) and global_step % 100 == 0 and global_step < args.steps:
            optimizer = adapt_curve_density(
                scene=scene,
                renderer=renderer,
                current_render=rendered.detach(),
                phase_target=phase_target,
                phase_lr=phase_lr,
                target_rgb=target_rgb,
                orig_H=H,
                orig_W=W,
                scale_x=scale_x,
                scale_y=scale_y,
                optimizer=optimizer,
                render_size=render_size,
                lr_cp_mult=args.lr_cp_mult,
                max_width_fill=args.max_width_fill,
                max_width_edge=args.max_width_edge,
            )
            # Ablation renders the scene once per curve → frees peak reservations.
            if device == "cuda":
                torch.cuda.empty_cache()

        # Intermediate save
        if args.save_every > 0 and global_step % args.save_every == 0:
            with torch.no_grad():
                inter = render_scene_shuffled(scene, renderer, shuffle=False)
            inter_np = inter.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
            out = os.path.join(args.output_dir, f"step_{global_step:05d}.png")
            plt.imsave(out, inter_np)

        # -------------------------------------------------------------------
        # Resolution Phase Transitions (Learned Resolution Scheduler)
        # -------------------------------------------------------------------
        loss_history.append(loss.item())
        loss_history = loss_history[-30:]

        trigger_transition = False
        # Case A: Plateau Condition (only if phase has run at least 50 steps and there is a next phase)
        if not getattr(args, "fixed_scheduler", False) and steps_in_phase >= 50 and len(loss_history) >= 30 and current_phase_idx < len(phases) - 1:
            slope = compute_slope(loss_history)
            if slope > -0.0001:
                print(f"  [Learned Scheduler] Loss plateaued (slope = {slope:.6f} > -0.0001). Transitioning from {phase_name}...")
                trigger_transition = True

        # Case B: Maximum budget limit for phase reached (and there is a next phase)
        if steps_in_phase >= current_phase["max_steps"] and current_phase_idx < len(phases) - 1:
            print(f"  [Learned Scheduler] Reached max steps ({current_phase['max_steps']}) for phase {phase_name}. Transitioning...")
            trigger_transition = True

        if trigger_transition:
            current_phase_idx += 1
            steps_in_phase = 0
            loss_history.clear()
            
            current_phase = phases[current_phase_idx]
            phase_name = current_phase["name"]
            render_size = current_phase["size"]
            phase_lr = current_phase["lr"]
            rH, rW = render_size

            # Interpolate target for new phase
            if (rH, rW) != (H, W):
                t4d = target_tensor.unsqueeze(0)
                phase_target = F.interpolate(t4d, size=(rH, rW), mode="bilinear", align_corners=False).squeeze(0)
            else:
                phase_target = target_tensor

            scale_x = rW / W
            scale_y = rH / H

            # Reset optimizer state to clear stale momentum and update learning rates
            optimizer.state.clear()
            optimizer.param_groups[0]["lr"] = phase_lr * args.lr_cp_mult
            optimizer.param_groups[1]["lr"] = phase_lr

            # Return cached VRAM before the higher-res phase allocates: live usage
            # is tiny (~60MB) but the allocator hoards peak reservations, which
            # collide with other processes on a shared GPU at the resolution bump.
            if device == "cuda":
                torch.cuda.empty_cache()

    total_time = time.time() - start_time
    print(f"\n✓ Optimization complete in {total_time:.1f}s "
          f"({total_time/args.steps:.2f}s/step)")

    # -----------------------------------------------------------------------
    # 6.5 Stage-6 exact-boundary polish (SPEC §6) — gated by MS-SSIM.
    # Full-res steps with maximal edge sharpness to crispen boundaries.
    # -----------------------------------------------------------------------
    if getattr(args, "polish_steps", 0) > 0:
        with torch.no_grad():
            full_render = render_scene_shuffled(scene, renderer, shuffle=False)
            if HAS_MSSSIM and H > 160 and W > 160:
                cur_ssim = float(stable_ms_ssim(
                    full_render.float().unsqueeze(0), target_tensor.float().unsqueeze(0),
                    data_range=1.0, size_average=True))
            else:
                cur_ssim = 0.0
        if cur_ssim < args.ssim_gate:
            print(f"✓ Stage-6 polish: MS-SSIM {cur_ssim:.4f} < gate {args.ssim_gate} → "
                  f"{args.polish_steps} full-res steps (sharp edges)")
            scene.fill_k = 4.0  # hard edges
            phase_target = target_tensor
            render_size = (H, W)
            optimizer.state.clear()
            optimizer.param_groups[0]["lr"] = args.lr * 0.1 * args.lr_cp_mult
            optimizer.param_groups[1]["lr"] = args.lr * 0.1
            for _ in range(args.polish_steps):
                optimizer.zero_grad()
                with autocast_context:
                    rendered = render_scene_shuffled(
                        scene, renderer, shuffle=True, render_size=render_size,
                        use_checkpoint=args.checkpoint)
                    loss = F.l1_loss(rendered, phase_target)
                    autocast_disable = torch.cuda.amp.autocast(enabled=False) if use_amp else contextlib.nullcontext()
                    with autocast_disable:
                        loss = loss + max(args.lambda_edge, 1.0) * compute_edge_loss(
                            rendered.float(), phase_target.float())
                if use_amp:
                    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                else:
                    loss.backward(); optimizer.step()
            print(f"  → polish done (final L1+edge loss {float(loss):.4f})")
        else:
            print(f"✓ Stage-6 polish skipped: MS-SSIM {cur_ssim:.4f} ≥ gate {args.ssim_gate}")

    # -----------------------------------------------------------------------
    # 7. Final render and save
    # -----------------------------------------------------------------------
    with torch.no_grad():
        after_image = render_scene_shuffled(scene, renderer, shuffle=False)
    after_np = after_image.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(target_rgb)
    axes[0].set_title("Target Image", color="#D7FE3D", fontweight="bold", fontsize=14)
    axes[0].axis("off")

    axes[1].imshow(before_np)
    axes[1].set_title("Initialized Curves", color="#D7FE3D", fontweight="bold", fontsize=14)
    axes[1].axis("off")

    axes[2].imshow(after_np)
    axes[2].set_title(f"Optimized Curves ({args.steps} steps)", color="#D7FE3D",
                      fontweight="bold", fontsize=14)
    axes[2].axis("off")

    fig.patch.set_facecolor("#0A0A0A")
    for ax in axes:
        ax.set_facecolor("#0A0A0A")
    plt.tight_layout()

    comparison_path = os.path.join(args.output_dir, "cuda_vectorization_comparison.png")
    plt.savefig(comparison_path, dpi=150, facecolor="#0A0A0A", bbox_inches="tight")
    plt.close()

    # Save raw optimized render for benchmarking
    plt.imsave(os.path.join(args.output_dir, "optimized_render.png"), after_np)

    svg_path = os.path.join(args.output_dir, "optimized_artwork.svg")
    export_to_svg(scene, svg_path, quantize=args.quantize_svg, min_opacity=args.svg_min_opacity, min_area=args.svg_min_area)

    print(f"✓ Comparison: {comparison_path}")
    print(f"✓ SVG export: {svg_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
