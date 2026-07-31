"""
Differentiable filled-shape renderer for closed cubic-Bezier loops.

Where `renderer.py` strokes a curve as soft elongated Gaussians (great for
painterly detail, but inherently blurry for large flat regions), this module
renders a *closed loop* as a crisp flat-filled region with a sharp ~1px
anti-aliased boundary — the tracer-quality primitive.

Approach (signed-distance coverage band, DiffVG-style):
  1. A loop is a list of S cubic segments, shape (S, 4, 2). Densely sample each
     segment to get a closed boundary polyline (M points).
  2. For every pixel, accumulate the minimum distance to the polyline edges
     (running min, O(H*W) memory — same trick as the Gaussian loop).
  3. Determine inside/outside via the winding number (sum of signed edge angles
     ≈ ±2π inside, 0 outside). The sign is computed under no_grad and treated as
     piecewise-constant: it carries NO gradient. The differentiable *distance*
     carries the geometric gradient, exactly where a soft boundary band needs it.
  4. signed distance  sdf = dist * sign      (sign ∈ {-1 inside, +1 outside})
     coverage         alpha = sigmoid(-k * sdf) * opacity
     A larger k → sharper edge. Anneal k upward coarse→fine to crispen.
  5. Composite alpha·color + (1-alpha)·canvas — identical blend to the splat path.

Autograd-safety mirrors the idioms already trusted in this codebase:
  - sqrt(x² + eps) to avoid the infinite gradient of sqrt at 0 (cf. build_covariance)
  - eps-padded denominators (cf. invert_2x2's det_safe)
  - the winding atan2 (which can hit atan2(0,0)) is fully inside torch.no_grad,
    so the documented atan2(0,0) NaN-in-backward problem cannot occur here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from curvesplat.renderer import evaluate_bezier

_EPS = 1e-8


def sample_loop_polyline(segments: torch.Tensor, samples_per_segment: int) -> torch.Tensor:
    """
    Sample a closed loop of cubic Bezier segments into a boundary polyline.

    Args:
        segments: (S, 4, 2) — S cubic segments forming a closed loop.
        samples_per_segment: points sampled per segment (endpoint excluded to
            avoid duplicate vertices at segment joins; the polyline is treated
            as cyclic so the loop closes automatically).

    Returns:
        polyline: (M, 2) where M = S * samples_per_segment.
    """
    S = segments.shape[0]
    device, dtype = segments.device, segments.dtype
    # Exclude t=1.0 so consecutive segments don't emit a duplicate shared vertex.
    t = torch.linspace(0.0, 1.0, samples_per_segment + 1, device=device, dtype=dtype)[:-1]
    pts = []
    for s in range(S):
        positions, _ = evaluate_bezier(segments[s], t)  # (samples_per_segment, 2)
        pts.append(positions)
    return torch.cat(pts, dim=0)  # (M, 2)


def _point_segment_distance_sq(
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
    ax: torch.Tensor,
    ay: torch.Tensor,
    bx: torch.Tensor,
    by: torch.Tensor,
) -> torch.Tensor:
    """
    Squared distance from every pixel (grid_x, grid_y) to the segment a→b.

    All of grid_x/grid_y are (H, W); ax..by are scalar tensors. Returns (H, W).
    Uses an eps-padded denominator for the projection parameter so coincident
    endpoints (a == b) stay finite and differentiable.
    """
    abx = bx - ax
    aby = by - ay
    apx = grid_x - ax
    apy = grid_y - ay
    seg_len_sq = abx * abx + aby * aby + _EPS
    t = ((apx * abx + apy * aby) / seg_len_sq).clamp(0.0, 1.0)
    proj_x = ax + t * abx
    proj_y = ay + t * aby
    dx = grid_x - proj_x
    dy = grid_y - proj_y
    return dx * dx + dy * dy


def polyline_sdf(
    polyline: torch.Tensor,
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
) -> torch.Tensor:
    """
    Signed distance field of a closed polyline, evaluated on a pixel grid.

    Negative inside, positive outside. Differentiable through the distance
    magnitude; the inside/outside sign is computed under no_grad and applied as
    a piecewise-constant multiplier (no gradient through the winding atan2).

    Args:
        polyline: (M, 2) cyclic boundary vertices.
        grid_x, grid_y: (H, W) pixel-center coordinates.

    Returns:
        sdf: (H, W) signed distance in pixels.
    """
    M = polyline.shape[0]
    H, W = grid_x.shape
    device, dtype = grid_x.device, grid_x.dtype

    min_dist_sq = torch.full((H, W), float("inf"), device=device, dtype=dtype)
    with torch.no_grad():
        winding = torch.zeros((H, W), device=device, dtype=dtype)

    gx = grid_x.unsqueeze(0)  # (1,H,W)
    gy = grid_y.unsqueeze(0)

    # Vectorize over edges, chunked to bound memory (avoids the per-edge Python
    # loop that made this O(M) GPU↔Python round-trips — the fill-path bottleneck).
    EDGE_CHUNK = 64
    for s in range(0, M, EDGE_CHUNK):
        e = min(s + EDGE_CHUNK, M)
        idx = torch.arange(s, e, device=device)
        a = polyline[idx]                          # (k,2)
        b = polyline[(idx + 1) % M]                # (k,2)
        ax = a[:, 0].view(-1, 1, 1); ay = a[:, 1].view(-1, 1, 1)
        bx = b[:, 0].view(-1, 1, 1); by = b[:, 1].view(-1, 1, 1)

        abx = bx - ax; aby = by - ay
        apx = gx - ax; apy = gy - ay
        seg_len_sq = abx * abx + aby * aby + _EPS
        t = ((apx * abx + apy * aby) / seg_len_sq).clamp(0.0, 1.0)
        dx = gx - (ax + t * abx); dy = gy - (ay + t * aby)
        dist_sq = dx * dx + dy * dy                # (k,H,W)
        min_dist_sq = torch.minimum(min_dist_sq, dist_sq.min(dim=0).values)

        with torch.no_grad():
            v1x = ax - gx; v1y = ay - gy
            v2x = bx - gx; v2y = by - gy
            cross = v1x * v2y - v1y * v2x
            dot = v1x * v2x + v1y * v2y
            winding = winding + torch.atan2(cross, dot).sum(dim=0)

    dist = torch.sqrt(min_dist_sq + _EPS)

    with torch.no_grad():
        # |winding| ≈ 2π inside (any orientation / multiplicity), ≈ 0 outside.
        inside = winding.abs() >= 3.14159265358979
        sign = torch.where(inside, -torch.ones_like(dist), torch.ones_like(dist))

    return dist * sign  # sign detached → only `dist` backpropagates


def render_filled_curve_fn(
    segments: torch.Tensor,
    color_raw: torch.Tensor,
    opacity_raw: torch.Tensor,
    canvas: torch.Tensor,
    samples_per_segment: int = 24,
    k: float = 1.0,
    width_scale: float = 1.0,
) -> torch.Tensor:
    """
    Render a closed filled loop onto a canvas and return the composited canvas.

    Mirrors the structure / dtype handling of render_single_curve_fn in
    vectorize_cuda.py: runs in float32 inside autocast(enabled=False) and casts
    back to the canvas dtype at the boundary, for AMP / torch.compile safety.

    Args:
        segments: (S, 4, 2) cubic segments forming the loop, already scaled to
            the current render resolution by the caller.
        color_raw: (3,) pre-sigmoid fill color.
        opacity_raw: scalar pre-sigmoid fill opacity.
        canvas: (3, H, W) current canvas (background or prior layers).
        samples_per_segment: boundary sampling density.
        k: edge sharpness; larger = harder edge. Annealed coarse→fine.
        width_scale: scales k with resolution so the ~1px band stays ~1px at the
            current render size (a band of 1/k source-pixels → 1/(k·scale) here).

    Returns:
        out_canvas: (3, H, W) in the canvas's original dtype.
    """
    with torch.cuda.amp.autocast(enabled=False):
        segments = segments.float()
        color_raw = color_raw.float()
        opacity_raw = opacity_raw.float()
        canvas_f32 = canvas.float()

        H, W = canvas_f32.shape[1], canvas_f32.shape[2]
        device = canvas_f32.device

        polyline = sample_loop_polyline(segments, samples_per_segment)

        py = torch.arange(H, device=device, dtype=torch.float32) + 0.5
        px = torch.arange(W, device=device, dtype=torch.float32) + 0.5
        grid_y, grid_x = torch.meshgrid(py, px, indexing="ij")

        sdf = polyline_sdf(polyline, grid_x, grid_y)  # (H, W)

        # Sharper edge as resolution drops would alias; scale k by width_scale so
        # the transition band stays ~1 rendered-pixel wide regardless of phase.
        k_eff = k / max(width_scale, _EPS)
        coverage = torch.sigmoid(-k_eff * sdf)  # (H, W) in [0,1]

        opacity = torch.sigmoid(opacity_raw)
        color = torch.sigmoid(color_raw)

        alpha = (coverage * opacity).clamp(0.0, 1.0)
        alpha_3d = alpha.unsqueeze(0)               # (1, H, W)
        color_expanded = color.view(3, 1, 1)        # (3, 1, 1)

        out_canvas = alpha_3d * color_expanded + (1.0 - alpha_3d) * canvas_f32

    return out_canvas.to(canvas.dtype)
