"""
Tests for the differentiable filled-shape renderer (fill_renderer.py).

GATE: the float64 gradcheck must pass before the filled primitive is wired into
the optimization pipeline — it is the highest-risk piece (autograd stability of
the polyline distance field).
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "python"))

from curvesplat.fill_renderer import (  # noqa: E402
    polyline_sdf,
    sample_loop_polyline,
    render_filled_curve_fn,
)


def _square_segments(lo: float, hi: float, dtype=torch.float64) -> torch.Tensor:
    """Axis-aligned square loop as 4 cubic segments (straight edges)."""
    corners = [(lo, lo), (hi, lo), (hi, hi), (lo, hi)]
    segs = []
    for i in range(4):
        ax, ay = corners[i]
        bx, by = corners[(i + 1) % 4]
        p0 = (ax, ay)
        p1 = (ax + (bx - ax) / 3.0, ay + (by - ay) / 3.0)
        p2 = (ax + 2.0 * (bx - ax) / 3.0, ay + 2.0 * (by - ay) / 3.0)
        p3 = (bx, by)
        segs.append([p0, p1, p2, p3])
    return torch.tensor(segs, dtype=dtype)  # (4, 4, 2)


def _grid(H: int, W: int, dtype=torch.float64):
    py = torch.arange(H, dtype=dtype) + 0.5
    px = torch.arange(W, dtype=dtype) + 0.5
    grid_y, grid_x = torch.meshgrid(py, px, indexing="ij")
    return grid_x, grid_y


def _render_from_params(segments, color_raw, opacity_raw, canvas_size, samples_per_segment, k):
    """Pure float64 functional render for gradcheck (mirrors production math)."""
    H, W = canvas_size
    grid_x, grid_y = _grid(H, W, dtype=segments.dtype)
    polyline = sample_loop_polyline(segments, samples_per_segment)
    sdf = polyline_sdf(polyline, grid_x, grid_y)
    coverage = torch.sigmoid(-k * sdf)
    alpha = (coverage * torch.sigmoid(opacity_raw)).clamp(0.0, 1.0)
    color = torch.sigmoid(color_raw)
    bg = torch.zeros(3, H, W, dtype=segments.dtype)
    return alpha.unsqueeze(0) * color.view(3, 1, 1) + (1.0 - alpha.unsqueeze(0)) * bg


class TestGradcheck:
    def test_gradcheck_control_points(self):
        """Gradients through the loop control points match finite differences."""
        # Jitter so no pixel is exactly equidistant to two edges: a perfect square
        # has tie points where min-over-edges is non-differentiable (measure-zero,
        # irrelevant in optimization) which makes strict float64 gradcheck flaky.
        g = torch.Generator().manual_seed(7)
        segments = _square_segments(4.0, 12.0)
        segments = (segments + (torch.rand(segments.shape, generator=g, dtype=torch.float64) - 0.5) * 0.3)
        segments = segments.requires_grad_(True)
        color_raw = torch.tensor([0.2, -0.1, 0.4], dtype=torch.float64)
        opacity_raw = torch.tensor(2.0, dtype=torch.float64)

        def fn(seg):
            return _render_from_params(seg, color_raw, opacity_raw, (16, 16), 4, k=1.0)

        assert torch.autograd.gradcheck(fn, (segments,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_gradcheck_color(self):
        """Gradients through fill color match finite differences."""
        segments = _square_segments(4.0, 12.0)
        color_raw = torch.tensor([0.2, -0.1, 0.4], dtype=torch.float64, requires_grad=True)
        opacity_raw = torch.tensor(2.0, dtype=torch.float64)

        def fn(col):
            return _render_from_params(segments, col, opacity_raw, (16, 16), 4, k=1.0)

        assert torch.autograd.gradcheck(fn, (color_raw,), eps=1e-6, atol=1e-4, rtol=1e-3)


class TestCoverage:
    def test_inside_is_filled_outside_is_empty(self):
        """A sharp-k square fills its interior and leaves the exterior empty."""
        segments = _square_segments(8.0, 24.0, dtype=torch.float32)
        grid_x, grid_y = _grid(32, 32, dtype=torch.float32)
        sdf = polyline_sdf(sample_loop_polyline(segments, 16), grid_x, grid_y)
        coverage = torch.sigmoid(-8.0 * sdf)
        assert coverage[16, 16] > 0.95, "center of square should be filled"
        assert coverage[1, 1] < 0.05, "far corner should be empty"
        # Sign is correct: inside has negative sdf, outside positive.
        assert sdf[16, 16] < 0
        assert sdf[1, 1] > 0

    def test_render_composites_onto_canvas(self):
        segments = _square_segments(8.0, 24.0, dtype=torch.float32)
        canvas = torch.zeros(3, 32, 32)
        color_raw = torch.tensor([5.0, -5.0, -5.0])  # ~pure red
        opacity_raw = torch.tensor(5.0)
        out = render_filled_curve_fn(segments, color_raw, opacity_raw, canvas, k=8.0)
        assert out.shape == (3, 32, 32)
        assert out[0, 16, 16] > 0.9 and out[1, 16, 16] < 0.1  # red interior

    def test_nan_safety_pixel_on_vertex(self):
        """A pixel coincident with a boundary vertex stays finite fwd and bwd."""
        # Put a square vertex exactly on a pixel center (x.5, y.5).
        segments = _square_segments(4.5, 12.5, dtype=torch.float64).requires_grad_(True)
        out = _render_from_params(segments, torch.zeros(3, dtype=torch.float64),
                                  torch.tensor(2.0, dtype=torch.float64), (16, 16), 6, k=2.0)
        loss = out.sum()
        loss.backward()
        assert torch.isfinite(out).all()
        assert torch.isfinite(segments.grad).all()
