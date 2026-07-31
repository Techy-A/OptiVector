"""
Equivalence test: render_curves_batched must match the per-curve production
renderer (render_single_curve_fn) in BOTH forward output and gradients.

If this passes, swapping the Python per-curve loop for the batched kernel is a
pure speed win with no change to results.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

from curvesplat.renderer import render_curves_batched, render_curves_tiled  # noqa: E402
import vectorize_cuda as vc  # noqa: E402  (provides render_single_curve_fn)


def _make_params(C, device="cpu", seed=0):
    g = torch.Generator().manual_seed(seed)
    cp = (torch.rand(C, 4, 2, generator=g) * 24 + 4)          # in a 32x32 canvas
    color = torch.randn(C, 3, generator=g) * 0.5
    width = torch.randn(C, generator=g) * 0.5 + 1.5
    opacity = torch.randn(C, generator=g) * 0.5 + 1.0
    maxw = torch.full((C,), 15.0)
    return cp, color, width, opacity, maxw


def _reference(cp, color, width, opacity, maxw, canvas, N, ws):
    """Sequential per-curve render — the production path."""
    out = canvas
    for i in range(cp.shape[0]):
        out = vc.render_single_curve_fn(
            cp[i], color[i], width[i], opacity[i], out, N, ws, float(maxw[i])
        )
    return out


class TestBatchedEquivalence:
    def test_forward_matches(self):
        C, N = 6, 12
        cp, color, width, opacity, maxw = _make_params(C)
        canvas = torch.zeros(3, 32, 32)

        ref = _reference(cp, color, width, opacity, maxw, canvas, N, 1.0)
        bat = render_curves_batched(cp, color, width, opacity, maxw, canvas, N, 1.0)

        assert torch.allclose(ref, bat, atol=1e-4, rtol=1e-3), \
            f"max diff {(ref - bat).abs().max().item():.2e}"

    def test_gradients_match(self):
        C, N = 5, 12
        cp0, color0, width0, opacity0, maxw = _make_params(C, seed=1)
        canvas = torch.zeros(3, 32, 32)

        # Reference grads
        cp_a = cp0.clone().requires_grad_(True)
        col_a = color0.clone().requires_grad_(True)
        w_a = width0.clone().requires_grad_(True)
        o_a = opacity0.clone().requires_grad_(True)
        _reference(cp_a, col_a, w_a, o_a, maxw, canvas, N, 1.0).pow(2).sum().backward()

        # Batched grads
        cp_b = cp0.clone().requires_grad_(True)
        col_b = color0.clone().requires_grad_(True)
        w_b = width0.clone().requires_grad_(True)
        o_b = opacity0.clone().requires_grad_(True)
        render_curves_batched(cp_b, col_b, w_b, o_b, maxw, canvas, N, 1.0).pow(2).sum().backward()

        for name, ga, gb in [("cp", cp_a.grad, cp_b.grad), ("color", col_a.grad, col_b.grad),
                             ("width", w_a.grad, w_b.grad), ("opacity", o_a.grad, o_b.grad)]:
            assert torch.allclose(ga, gb, atol=1e-3, rtol=1e-2), \
                f"{name} grad mismatch, max diff {(ga - gb).abs().max().item():.2e}"


class TestTiledEquivalence:
    """Tiled (bbox) renderer must match the dense renderer fwd + gradients."""

    def test_forward_matches_dense(self):
        C, N = 6, 12
        cp, color, width, opacity, maxw = _make_params(C, seed=3)
        canvas = torch.zeros(3, 32, 32)
        dense = render_curves_batched(cp, color, width, opacity, maxw, canvas, N, 1.0)
        tiled = render_curves_tiled(cp, color, width, opacity, maxw, canvas, N, 1.0)
        assert torch.allclose(dense, tiled, atol=1e-4, rtol=1e-3), \
            f"max diff {(dense - tiled).abs().max().item():.2e}"

    def test_gradients_match_dense(self):
        C, N = 5, 12
        cp0, color0, width0, opacity0, maxw = _make_params(C, seed=4)
        canvas = torch.zeros(3, 32, 32)

        def grads(fn):
            cp = cp0.clone().requires_grad_(True)
            col = color0.clone().requires_grad_(True)
            w = width0.clone().requires_grad_(True)
            o = opacity0.clone().requires_grad_(True)
            fn(cp, col, w, o, maxw, canvas, N, 1.0).pow(2).sum().backward()
            return cp.grad, col.grad, w.grad, o.grad

        for name, ga, gb in zip(["cp", "color", "width", "opacity"],
                                grads(render_curves_batched), grads(render_curves_tiled)):
            assert torch.allclose(ga, gb, atol=1e-3, rtol=1e-2), \
                f"{name} grad mismatch, max diff {(ga - gb).abs().max().item():.2e}"
