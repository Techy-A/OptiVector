"""
GATE: CUDA splat kernel must match the PyTorch reference (render_curves_batched)
in forward output AND gradients. Skipped unless CUDA + the JIT kernel are present
(so it's inert on CPU/Mac, runs on the A6000). Do NOT trust the CUDA path until
this passes.
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "python"))

from curvesplat.renderer import render_curves_batched  # noqa: E402
from curvesplat import cuda_splat  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and cuda_splat.HAS_CUDA_SPLAT),
    reason="CUDA splat kernel not available",
)

DEV = "cuda"


def _raw_params(C, seed=0):
    g = torch.Generator().manual_seed(seed)
    cp = (torch.rand(C, 4, 2, generator=g) * 48 + 8).to(DEV)          # in 64x64
    color_raw = (torch.randn(C, 3, generator=g) * 0.5).to(DEV)
    # width_raw chosen so softplus(width_raw) lands well inside (0.5, max_width)
    width_raw = (torch.randn(C, generator=g) * 0.3 + 1.5).to(DEV)
    opacity_raw = (torch.randn(C, generator=g) * 0.5 + 0.5).to(DEV)
    maxw = torch.full((C,), 15.0, device=DEV)
    return cp, color_raw, width_raw, opacity_raw, maxw


def _activate(color_raw, width_raw, opacity_raw, maxw):
    width = F.softplus(width_raw).clamp(min=0.5)
    width = torch.minimum(width, maxw)
    return torch.sigmoid(color_raw), width, torch.sigmoid(opacity_raw)


def test_forward_matches_reference():
    C, N, H, W = 6, 12, 64, 64
    cp, color_raw, width_raw, opacity_raw, maxw = _raw_params(C, seed=1)
    canvas = torch.zeros(3, H, W, device=DEV)

    ref = render_curves_batched(cp, color_raw, width_raw, opacity_raw, maxw, canvas, N, 1.0)
    col, wid, opa = _activate(color_raw, width_raw, opacity_raw, maxw)
    cu = cuda_splat.render_cuda(cp, col, wid, opa, canvas, N)

    diff = (ref - cu).abs().max().item()
    assert torch.allclose(ref, cu, atol=1e-3, rtol=1e-2), f"forward max diff {diff:.2e}"


def test_gradients_match_reference():
    C, N, H, W = 5, 12, 64, 64
    cp0, color0, width0, opacity0, maxw = _raw_params(C, seed=2)
    canvas = torch.zeros(3, H, W, device=DEV)

    # Reference grads (raw params).
    cp_a = cp0.clone().requires_grad_(True)
    col_a = color0.clone().requires_grad_(True)
    w_a = width0.clone().requires_grad_(True)
    o_a = opacity0.clone().requires_grad_(True)
    render_curves_batched(cp_a, col_a, w_a, o_a, maxw, canvas, N, 1.0).pow(2).sum().backward()

    # CUDA grads (activate in torch, then CUDA alpha + torch composite).
    cp_b = cp0.clone().requires_grad_(True)
    col_b = color0.clone().requires_grad_(True)
    w_b = width0.clone().requires_grad_(True)
    o_b = opacity0.clone().requires_grad_(True)
    col, wid, opa = _activate(col_b, w_b, o_b, maxw)
    cuda_splat.render_cuda(cp_b, col, wid, opa, canvas, N).pow(2).sum().backward()

    for name, ga, gb in [("cp", cp_a.grad, cp_b.grad), ("color", col_a.grad, col_b.grad),
                         ("width", w_a.grad, w_b.grad), ("opacity", o_a.grad, o_b.grad)]:
        diff = (ga - gb).abs().max().item()
        assert torch.allclose(ga, gb, atol=1e-2, rtol=1e-2), f"{name} grad max diff {diff:.2e}"
