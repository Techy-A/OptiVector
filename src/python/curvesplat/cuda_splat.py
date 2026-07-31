"""
CUDA fast path for CurveSplat stroke rendering.

Wraps ONLY the expensive geometry (curve params -> per-curve alpha [C,H,W]) in a
custom CUDA kernel via an autograd.Function. The ordered alpha-over composite
stays in plain PyTorch (cheap reverse-cumprod, identical to render_curves_batched),
so only the O(C·N·H·W) Gaussian — the actual bottleneck — moves to CUDA.

Build is JIT (torch.utils.cpp_extension.load): compiles src/cuda/splat_cuda.cu on
first import (needs `ninja` + CUDA toolkit). Any failure -> HAS_CUDA_SPLAT=False and
callers fall back to the pure-PyTorch render_curves_batched. The PyTorch path stays
the numerical reference (tests/test_cuda_splat.py gates equivalence).
"""

from __future__ import annotations

import os
import torch

HAS_CUDA_SPLAT = False
_EXT = None

_CU_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "cuda", "splat_cuda.cu")

if torch.cuda.is_available():
    try:
        from torch.utils.cpp_extension import load
        _EXT = load(
            name="curvesplat_cuda",
            sources=[os.path.abspath(_CU_PATH)],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
        HAS_CUDA_SPLAT = True
        print("✓ CUDA splat kernel compiled & loaded")
    except Exception as e:  # pragma: no cover — depends on host toolchain
        print(f"  → CUDA splat kernel unavailable (using PyTorch fallback): {e}")
        HAS_CUDA_SPLAT = False


class CudaAlpha(torch.autograd.Function):
    """params -> alpha (C,H,W). Forward/backward both in CUDA."""

    @staticmethod
    def forward(ctx, cp, width, opacity, H, W, num_samples):
        cp = cp.contiguous().float()
        width = width.contiguous().float()
        opacity = opacity.contiguous().float()
        alpha = _EXT.alpha_forward(cp, width, opacity, int(H), int(W), int(num_samples))
        ctx.save_for_backward(cp, width, opacity)
        ctx.num_samples = int(num_samples)
        return alpha

    @staticmethod
    def backward(ctx, grad_alpha):
        cp, width, opacity = ctx.saved_tensors
        grad_cp, grad_width, grad_opacity = _EXT.alpha_backward(
            grad_alpha.contiguous().float(), cp, width, opacity, ctx.num_samples
        )
        # Match forward arg order: cp, width, opacity, H, W, num_samples
        return grad_cp, grad_width, grad_opacity, None, None, None


def render_cuda(
    control_points: torch.Tensor,   # (C,4,2) scaled to render res
    color: torch.Tensor,            # (C,3) ACTIVATED [0,1]
    width: torch.Tensor,            # (C,)  ACTIVATED >0
    opacity: torch.Tensor,          # (C,)  ACTIVATED [0,1]
    canvas: torch.Tensor,           # (3,H,W)
    num_samples: int,
) -> torch.Tensor:
    """
    CUDA stroke render: kernel computes per-curve alpha, PyTorch composites it
    over the canvas in array order (reverse-cumprod alpha-over). Returns canvas
    dtype. Inputs must already be activated (sigmoid/softplus done by caller) —
    keeps the kernel pure geometry, autograd handles activations outside.
    """
    orig_dtype = canvas.dtype
    H, W = canvas.shape[1], canvas.shape[2]
    C = control_points.shape[0]
    if C == 0:
        return canvas

    alpha = CudaAlpha.apply(
        control_points.float(), width.float(), opacity.float(), H, W, num_samples
    )  # (C,H,W), float32

    canvas_f = canvas.float()
    color_f = color.float()

    one_minus = 1.0 - alpha                                           # (C,H,W)
    T_incl = torch.flip(torch.cumprod(torch.flip(one_minus, [0]), dim=0), [0])
    ones = torch.ones(1, H, W, device=alpha.device, dtype=alpha.dtype)
    T_after = torch.cat([T_incl[1:], ones], dim=0)                    # (C,H,W)
    prod_all = T_incl[0]                                              # (H,W)

    contrib = (alpha * T_after).unsqueeze(1)                          # (C,1,H,W)
    fg = (contrib * color_f.view(C, 3, 1, 1)).sum(dim=0)            # (3,H,W)
    out = fg + canvas_f * prod_all.unsqueeze(0)
    return out.to(orig_dtype)
