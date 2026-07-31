"""
Differentiable Gaussian-splatting renderer for a single cubic Bezier curve.

Implements SPEC.md §7 in pure PyTorch — no CUDA, no tiling.
Numerical correctness is prioritized over performance.

Algorithm:
  1. Sample the cubic Bezier at N points: positions P(t) and tangents P'(t).
  2. At each sample, build an oriented 2D Gaussian (elongated along tangent).
  3. For every pixel, compute Mahalanobis distance to each sample's Gaussian.
  4. Take the maximum Gaussian weight across samples → per-pixel alpha.
  5. Alpha-blend the curve's color onto the background.

The entire pipeline is differentiable via PyTorch autograd.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from curvesplat.scene import Curve, Scene


def evaluate_bezier(
    control_points: torch.Tensor,
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate a cubic Bezier curve and its tangent at parameter values t.

    SPEC.md §7:
        P(t)  = (1-t)³ P₀ + 3(1-t)²t P₁ + 3(1-t)t² P₂ + t³ P₃
        P'(t) = 3(1-t)²(P₁-P₀) + 6(1-t)t(P₂-P₁) + 3t²(P₃-P₂)

    Args:
        control_points: Shape (4, 2) — P0, P1, P2, P3.
        t: Shape (N,) — parameter values in [0, 1].

    Returns:
        positions: Shape (N, 2) — points on the curve.
        tangents:  Shape (N, 2) — tangent vectors (unnormalized).
    """
    P0, P1, P2, P3 = control_points[0], control_points[1], control_points[2], control_points[3]

    # Precompute powers: t has shape (N,), we need (N, 1) for broadcasting
    t_ = t.unsqueeze(-1)  # (N, 1)
    omt = 1.0 - t_        # one minus t, (N, 1)

    # Cubic Bezier position: P(t)
    positions = (
        omt ** 3 * P0
        + 3.0 * omt ** 2 * t_ * P1
        + 3.0 * omt * t_ ** 2 * P2
        + t_ ** 3 * P3
    )  # (N, 2)

    # Cubic Bezier tangent: P'(t)
    tangents = (
        3.0 * omt ** 2 * (P1 - P0)
        + 6.0 * omt * t_ * (P2 - P1)
        + 3.0 * t_ ** 2 * (P3 - P2)
    )  # (N, 2)

    return positions, tangents


def build_covariance(
    tangents: torch.Tensor,
    width: torch.Tensor,
) -> torch.Tensor:
    """
    Build per-sample 2x2 covariance matrices from tangent directions and width.

    SPEC.md §7:
        θ = atan2(P'_y, P'_x)
        σ_long  = width
        σ_short = width / 4
        Σ = R(θ) · diag(σ_long², σ_short²) · R(θ)ᵀ

    Args:
        tangents: Shape (N, 2) — tangent vectors at each sample.
        width: Scalar tensor — activated (positive) stroke width.

    Returns:
        covariances: Shape (N, 2, 2) — per-sample covariance matrices.
    """
    # Compute tangent components and norm
    dx = tangents[:, 0]
    dy = tangents[:, 1]
    norm = torch.sqrt(dx * dx + dy * dy)
    
    is_zero = norm < 1e-6
    
    # Perturb tangents to be non-zero in the zero cases to avoid atan2(0,0) singularity/NaN backprop
    dx_safe = torch.where(is_zero, torch.ones_like(norm), dx)
    dy_safe = torch.where(is_zero, torch.zeros_like(norm), dy)
    
    theta_raw = torch.atan2(dy_safe, dx_safe)
    theta = torch.where(is_zero, torch.zeros_like(theta_raw), theta_raw)

    cos_t = torch.cos(theta)  # (N,)
    sin_t = torch.sin(theta)  # (N,)

    # Rotation matrix R(θ): shape (N, 2, 2)
    R = torch.stack([
        torch.stack([cos_t, -sin_t], dim=-1),
        torch.stack([sin_t,  cos_t], dim=-1),
    ], dim=-2)  # (N, 2, 2)

    # Variance along and across the curve
    sigma_long_sq = width ** 2                   # σ_long² = width²
    sigma_short_sq = (width / 4.0) ** 2          # σ_short² = (width/4)² — standard tangent scaling

    # Diagonal variance matrix: (N, 2, 2)
    # Broadcast scalar variances to all N samples
    D = torch.zeros(tangents.shape[0], 2, 2, device=tangents.device, dtype=tangents.dtype)
    D[:, 0, 0] = sigma_long_sq
    D[:, 1, 1] = sigma_short_sq

    # Σ = R · D · Rᵀ
    covariances = R @ D @ R.transpose(-1, -2)  # (N, 2, 2)

    return covariances


def invert_2x2(matrices: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Analytically invert a batch of 2x2 matrices.

    More numerically stable than torch.linalg.inv for 2x2, and avoids
    autograd issues at near-singular points by adding epsilon to the determinant.

    Args:
        matrices: Shape (..., 2, 2).
        eps: Small value added to |det| to prevent division by zero.

    Returns:
        inverses: Shape (..., 2, 2).
    """
    a = matrices[..., 0, 0]
    b = matrices[..., 0, 1]
    c = matrices[..., 1, 0]
    d = matrices[..., 1, 1]

    det = a * d - b * c
    
    # Protect against sign being 0 when det is exactly 0
    det_sign = det.sign()
    det_sign = torch.where(det_sign == 0, torch.ones_like(det_sign), det_sign)
    
    # Safe determinant that is guaranteed non-zero in both branches of torch.where
    det_safe = torch.where(det.abs() < eps, eps * det_sign, det)

    inv_det = 1.0 / det_safe

    inv = torch.stack([
        torch.stack([ d * inv_det, -b * inv_det], dim=-1),
        torch.stack([-c * inv_det,  a * inv_det], dim=-1),
    ], dim=-2)

    return inv


class CurveSplatRenderer(nn.Module):
    """
    Pure-PyTorch differentiable renderer for a single Bezier curve.

    Renders via Gaussian splatting: the curve is approximated as a series of
    elongated 2D Gaussians placed along the Bezier spline. Each pixel's
    contribution is the maximum Gaussian weight across all samples.

    SPEC.md §7 math, fully differentiable for autograd.
    """

    def __init__(self, num_samples: int = 64, accumulation: str = "max") -> None:
        """
        Args:
            num_samples: Number of points to sample along the Bezier curve.
                         More samples = smoother rendering, higher memory.
                         64 is good for 256x256; use 8-16 for gradcheck.
            accumulation: 'max' (original) or 'sum' — how to combine per-sample
                          Gaussian weights. 'sum' gives smoother area coverage
                          for wide strokes; 'max' is sharper for thin lines.
        """
        super().__init__()
        self.num_samples = num_samples
        self.accumulation = accumulation

    def forward(
        self,
        curve: Curve,
        canvas_size: tuple[int, int],
        background: torch.Tensor,
    ) -> torch.Tensor:
        """
        Render a single curve onto a canvas.

        Args:
            curve: The Curve module (parameters auto-registered).
            canvas_size: (H, W) canvas dimensions.
            background: Shape (3,) activated background RGB in [0, 1].

        Returns:
            image: Shape (3, H, W) rendered RGB image in [0, 1].
        """
        H, W = canvas_size
        device = curve.control_points.device
        dtype = curve.control_points.dtype

        # --- 1. Sample the Bezier curve ---
        t = torch.linspace(0.0, 1.0, self.num_samples, device=device, dtype=dtype)
        positions, tangents = evaluate_bezier(curve.control_points, t)
        # positions: (N, 2), tangents: (N, 2)

        # --- 2. Get activated parameters ---
        width = curve.width        # softplus-activated, positive scalar
        opacity = curve.opacity    # sigmoid-activated, [0, 1] scalar
        color = curve.color        # sigmoid-activated, (3,) in [0, 1]

        # --- 3. Build covariance matrices and their inverses ---
        covariances = build_covariance(tangents, width)  # (N, 2, 2)
        cov_inv = invert_2x2(covariances)                # (N, 2, 2)

        # --- 4. Create pixel grid ---
        py = torch.arange(H, device=device, dtype=dtype) + 0.5  # (H,)
        px = torch.arange(W, device=device, dtype=dtype) + 0.5  # (W,)
        grid_y, grid_x = torch.meshgrid(py, px, indexing="ij")  # (H, W) each

        # --- 5. Iterate over samples, accumulate Gaussian weight ---
        # This avoids the O(H*W*N) intermediate tensor, keeping memory at O(H*W).
        # Much faster on CPU than the fully-expanded version.
        if self.accumulation == "sum":
            accum_weight = torch.zeros(H, W, device=device, dtype=dtype)
        else:
            accum_weight = torch.zeros(H, W, device=device, dtype=dtype)  # max mode

        for i in range(self.num_samples):
            # Displacement from this sample's center: (H, W)
            dx = grid_x - positions[i, 0]
            dy = grid_y - positions[i, 1]

            # Mahalanobis² = dᵀ Σ⁻¹ d, expanded for 2x2:
            # = a*dx² + (b+c)*dx*dy + d*dy²
            a = cov_inv[i, 0, 0]
            b = cov_inv[i, 0, 1]
            c = cov_inv[i, 1, 0]
            d_val = cov_inv[i, 1, 1]

            mahal_sq = a * dx * dx + (b + c) * dx * dy + d_val * dy * dy  # (H, W)

            # 3-sigma cutoff (SPEC.md §7): skip if mahalanobis² > 9
            within_cutoff = (mahal_sq <= 9.0)
            weight = torch.exp(-0.5 * mahal_sq) * within_cutoff.to(dtype)  # (H, W)

            if self.accumulation == "sum":
                # Sum mode: accumulate weights (gives smooth area coverage)
                accum_weight = accum_weight + weight
            else:
                # Max mode (original): take per-pixel maximum across samples
                accum_weight = torch.max(accum_weight, weight)

        if self.accumulation == "sum":
            # Normalize by num_samples and clamp to [0,1]
            final_weight = (accum_weight / self.num_samples).clamp(0.0, 1.0)
        else:
            final_weight = accum_weight

        # --- 6. Alpha blending (SPEC.md §7) ---
        alpha = (final_weight * opacity).clamp(0.0, 1.0)  # (H, W)

        # C_out = alpha * color + (1 - alpha) * background
        alpha_3d = alpha.unsqueeze(-1)                        # (H, W, 1)
        color_expanded = color.unsqueeze(0).unsqueeze(0)      # (1, 1, 3)

        if background.dim() == 3:
            # background is (3, H, W), permute to (H, W, 3)
            bg_expanded = background.permute(1, 2, 0)
        else:
            bg_expanded = background.unsqueeze(0).unsqueeze(0)    # (1, 1, 3)

        image = alpha_3d * color_expanded + (1.0 - alpha_3d) * bg_expanded  # (H, W, 3)

        # Transpose to (3, H, W) — standard PyTorch image format
        image = image.permute(2, 0, 1)  # (3, H, W)

        return image


def render_curves_batched(
    control_points: torch.Tensor,   # (C, 4, 2) already scaled to render res
    color_raw: torch.Tensor,        # (C, 3)
    width_raw: torch.Tensor,        # (C,)
    opacity_raw: torch.Tensor,      # (C,)
    max_width: torch.Tensor,        # (C,)
    canvas: torch.Tensor,           # (3, H, W)
    num_samples: int,
    width_scale: float = 1.0,
    chunk: int = 24,
) -> torch.Tensor:
    """
    Batched Gaussian-splat renderer for MANY curves.

    Numerically identical to calling render_single_curve_fn per curve in order,
    but the expensive splat math is vectorized over all curves + samples (chunked
    to bound memory) instead of a Python loop. This is what actually uses the GPU
    — the per-curve loop left it ~idle (kernel-launch bound). 10–30× faster.

    Compositing is still sequential (alpha-over is order-dependent): per-curve
    alpha/color are computed batched, then blended in array order with cheap
    elementwise ops. Pass curves pre-sorted/shuffled in the desired paint order.

    Runs in float32 inside autocast(False) for AMP / dtype safety, mirroring
    render_single_curve_fn.
    """
    orig_dtype = canvas.dtype
    with torch.cuda.amp.autocast(enabled=False):
        control_points = control_points.float()
        color_raw = color_raw.float()
        width_raw = width_raw.float()
        opacity_raw = opacity_raw.float()
        max_width = max_width.float()
        canvas = canvas.float()

        C = control_points.shape[0]
        H, W = canvas.shape[1], canvas.shape[2]
        device = canvas.device

        if C == 0:
            return canvas.to(orig_dtype)

        # Activations (per curve).
        width = (F.softplus(width_raw) * width_scale).clamp(min=0.5)      # (C,)
        width = torch.minimum(width, max_width)
        opacity = torch.sigmoid(opacity_raw)                              # (C,)
        color = torch.sigmoid(color_raw)                                  # (C, 3)

        t = torch.linspace(0.0, 1.0, num_samples, device=device, dtype=torch.float32)
        py = torch.arange(H, device=device, dtype=torch.float32) + 0.5
        px = torch.arange(W, device=device, dtype=torch.float32) + 0.5
        grid_y, grid_x = torch.meshgrid(py, px, indexing="ij")            # (H, W)

        # --- per-curve alpha, computed in chunks over curves ---
        alphas = []
        for s in range(0, C, chunk):
            cp = control_points[s:s + chunk]                             # (c, 4, 2)
            w = width[s:s + chunk]                                       # (c,)
            o = opacity[s:s + chunk]                                     # (c,)
            c = cp.shape[0]

            # Batched cubic Bezier position + tangent at t. cp[:, i] is (c, 2).
            tt = t.view(1, num_samples, 1)
            omt = 1.0 - tt
            P0, P1, P2, P3 = cp[:, 0:1], cp[:, 1:2], cp[:, 2:3], cp[:, 3:4]  # (c,1,2)
            pos = (omt**3) * P0 + 3*(omt**2)*tt*P1 + 3*omt*(tt**2)*P2 + (tt**3)*P3  # (c,N,2)
            tan = 3*(omt**2)*(P1-P0) + 6*omt*tt*(P2-P1) + 3*(tt**2)*(P3-P2)         # (c,N,2)

            # Oriented covariance per (curve, sample), inverse analytically.
            dx_t, dy_t = tan[..., 0], tan[..., 1]                        # (c,N)
            norm = torch.sqrt(dx_t*dx_t + dy_t*dy_t + 1e-12)
            is_zero = norm < 1e-6
            dxs = torch.where(is_zero, torch.ones_like(norm), dx_t)
            dys = torch.where(is_zero, torch.zeros_like(norm), dy_t)
            theta = torch.where(is_zero, torch.zeros_like(norm), torch.atan2(dys, dxs))
            cos_t, sin_t = torch.cos(theta), torch.sin(theta)            # (c,N)
            sl = (w**2).view(c, 1)                                       # σ_long²  (c,1)
            ss = ((w/4.0)**2).view(c, 1)                                 # σ_short² (c,1)
            # Σ = R diag(sl,ss) Rᵀ  → closed form 2x2 entries (c,N)
            Sxx = cos_t*cos_t*sl + sin_t*sin_t*ss
            Sxy = cos_t*sin_t*(sl - ss)
            Syy = sin_t*sin_t*sl + cos_t*cos_t*ss
            det = Sxx*Syy - Sxy*Sxy
            det = torch.where(det.abs() < 1e-4, torch.full_like(det, 1e-4), det)
            iXX = (Syy/det).view(c, num_samples, 1, 1)
            iXY = (-Sxy/det).view(c, num_samples, 1, 1)
            iYY = (Sxx/det).view(c, num_samples, 1, 1)

            ddx = grid_x.view(1, 1, H, W) - pos[..., 0].view(c, num_samples, 1, 1)
            ddy = grid_y.view(1, 1, H, W) - pos[..., 1].view(c, num_samples, 1, 1)
            mahal = iXX*ddx*ddx + 2.0*iXY*ddx*ddy + iYY*ddy*ddy          # (c,N,H,W)
            within = (mahal <= 9.0)
            weight = torch.exp(-0.5 * torch.clamp(mahal, max=50.0)) * within.float()
            acc = weight.sum(dim=1)                                      # (c,H,W)
            a = (acc / num_samples).clamp(0.0, 1.0) * o.view(c, 1, 1)    # (c,H,W)
            alphas.append(a.clamp(0.0, 1.0))

        alpha = torch.cat(alphas, dim=0)                                 # (C,H,W)

        # Vectorized alpha-over composite (no Python loop → shallow backward graph).
        # Sequential `canvas = a_i·c_i + (1-a_i)·canvas` over i=0..C-1 unrolls to:
        #   out = Σ_i a_i·c_i·∏_{j>i}(1-a_j)  +  canvas·∏_j(1-a_j)
        # i.e. each curve is occluded by the curves painted AFTER it (reverse cumprod).
        one_minus = 1.0 - alpha                                          # (C,H,W)
        # T_incl[i] = ∏_{j>=i}(1-a_j) via reversed cumprod.
        T_incl = torch.flip(torch.cumprod(torch.flip(one_minus, [0]), dim=0), [0])
        # T_after[i] = ∏_{j>i}(1-a_j): shift up, last curve sees nothing in front.
        ones = torch.ones(1, H, W, device=device, dtype=torch.float32)
        T_after = torch.cat([T_incl[1:], ones], dim=0)                   # (C,H,W)
        prod_all = T_incl[0]                                             # (H,W)

        contrib = (alpha * T_after).unsqueeze(1)                         # (C,1,H,W)
        fg = (contrib * color.view(C, 3, 1, 1)).sum(dim=0)              # (3,H,W)
        canvas = fg + canvas * prod_all.unsqueeze(0)

    return canvas.to(orig_dtype)


def render_curves_tiled(
    control_points: torch.Tensor,   # (C, 4, 2) already scaled to render res
    color_raw: torch.Tensor,        # (C, 3)
    width_raw: torch.Tensor,        # (C,)
    opacity_raw: torch.Tensor,      # (C,)
    max_width: torch.Tensor,        # (C,)
    canvas: torch.Tensor,           # (3, H, W)
    num_samples: int,
    width_scale: float = 1.0,
) -> torch.Tensor:
    """
    Bounding-box (tiled) Gaussian-splat renderer — the FAST path.

    A curve's Gaussian support is local: beyond 3σ the weight is hard-zero (the
    same 3σ cutoff the dense renderer uses). So instead of evaluating every curve
    over the whole H×W canvas (most pixels are far-away background contributing
    nothing), each curve is rendered only on its own bounding box (sample span ±
    3·σ_long), then composited in order. On a sparse image (e.g. watercolour on
    white) this skips the overwhelming majority of (curve, pixel) pairs.

    Numerically matches render_curves_batched / the per-curve renderer within the
    3σ cutoff: bbox padding (3·σ_long) covers the entire non-zero support, so no
    contributing pixel is dropped. Composite order = array order (alpha-over).

    Runs float32 inside autocast(False), returns canvas dtype — AMP-safe.
    """
    orig_dtype = canvas.dtype
    with torch.cuda.amp.autocast(enabled=False):
        control_points = control_points.float()
        color_raw = color_raw.float()
        width_raw = width_raw.float()
        opacity_raw = opacity_raw.float()
        max_width = max_width.float()
        canvas = canvas.float()

        C = control_points.shape[0]
        H, W = canvas.shape[1], canvas.shape[2]
        device = canvas.device
        if C == 0:
            return canvas.to(orig_dtype)

        width = (F.softplus(width_raw) * width_scale).clamp(min=0.5)
        width = torch.minimum(width, max_width)                          # (C,)
        opacity = torch.sigmoid(opacity_raw)                             # (C,)
        color = torch.sigmoid(color_raw)                                 # (C, 3)

        t = torch.linspace(0.0, 1.0, num_samples, device=device, dtype=torch.float32)

        for i in range(C):
            cp = control_points[i]                                       # (4,2)
            positions, tangents = evaluate_bezier(cp, t)                 # (N,2),(N,2)
            w = width[i]

            # Detached integer bbox: sample span padded by 3·σ_long (full support).
            with torch.no_grad():
                pad = 3.0 * w + 1.0
                x0 = int(torch.clamp(torch.floor(positions[:, 0].min() - pad), 0, W - 1).item())
                x1 = int(torch.clamp(torch.ceil(positions[:, 0].max() + pad), 1, W).item())
                y0 = int(torch.clamp(torch.floor(positions[:, 1].min() - pad), 0, H - 1).item())
                y1 = int(torch.clamp(torch.ceil(positions[:, 1].max() + pad), 1, H).item())
            if x1 <= x0 or y1 <= y0:
                continue

            covs = build_covariance(tangents, w)                         # (N,2,2)
            cov_inv = invert_2x2(covs)                                   # (N,2,2)

            gy = torch.arange(y0, y1, device=device, dtype=torch.float32) + 0.5
            gx = torch.arange(x0, x1, device=device, dtype=torch.float32) + 0.5
            grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")       # (bh,bw)

            ddx = grid_x.unsqueeze(0) - positions[:, 0].view(num_samples, 1, 1)  # (N,bh,bw)
            ddy = grid_y.unsqueeze(0) - positions[:, 1].view(num_samples, 1, 1)
            a = cov_inv[:, 0, 0].view(num_samples, 1, 1)
            b = cov_inv[:, 0, 1].view(num_samples, 1, 1)
            c = cov_inv[:, 1, 0].view(num_samples, 1, 1)
            d = cov_inv[:, 1, 1].view(num_samples, 1, 1)
            mahal = a*ddx*ddx + (b + c)*ddx*ddy + d*ddy*ddy              # (N,bh,bw)
            within = (mahal <= 9.0)
            weight = torch.exp(-0.5 * torch.clamp(mahal, max=50.0)) * within.float()
            acc = weight.sum(dim=0)                                      # (bh,bw)
            alpha = ((acc / num_samples).clamp(0.0, 1.0) * opacity[i]).clamp(0.0, 1.0)

            sub = canvas[:, y0:y1, x0:x1]                                # (3,bh,bw)
            blended = alpha.unsqueeze(0) * color[i].view(3, 1, 1) + (1.0 - alpha.unsqueeze(0)) * sub
            # Out-of-place region update (clone keeps alpha-over autograd correct).
            canvas = canvas.clone()
            canvas[:, y0:y1, x0:x1] = blended

    return canvas.to(orig_dtype)


def render_scene(
    scene: Scene,
    renderer: CurveSplatRenderer,
) -> torch.Tensor:
    """
    Render a full scene by compositing all curves in order (Z-index order).

    Args:
        scene: Scene with curves and background.
        renderer: CurveSplatRenderer instance.

    Returns:
        image: Shape (3, H, W) rendered RGB image.
    """
    H, W = scene.canvas_size
    if len(scene.curves) == 0:
        # Empty scene: just return background
        bg = scene.background  # (3,)
        return bg.view(3, 1, 1).expand(3, H, W)

    # Start with background canvas
    canvas = scene.background.view(3, 1, 1).expand(3, H, W)

    # Composite each curve in order (Z-index order)
    sorted_curves = sorted(scene.curves, key=lambda c: getattr(c, "z_index", 0))

    for curve in sorted_curves:
        canvas = renderer(
            curve=curve,
            canvas_size=scene.canvas_size,
            background=canvas,
        )
    return canvas
