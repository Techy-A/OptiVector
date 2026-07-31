"""
Region-fill init must produce LOCAL strokes, not canvas-spanning zig-zags.

The old init sampled 4 random pixels from the whole color cluster; for a spatially
spread cluster that drew strokes across the entire image (noise init). This test
uses red blobs in all four corners (one spread cluster) — a local-stroke init keeps
each curve's control points compact; the old random init would span the canvas.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import vectorize_cuda as vc  # noqa: E402


def _four_corner_reds(H=64, W=64):
    img = np.ones((H, W, 3), np.float32)  # white bg
    s = 12
    for (y, x) in [(0, 0), (0, W - s), (H - s, 0), (H - s, W - s)]:
        img[y:y + s, x:x + s] = [0.9, 0.15, 0.15]
    return img


def test_fill_curves_are_local_not_spanning():
    np.random.seed(0)
    img = _four_corner_reds(64, 64)
    curves = vc.initialize_region_fill_curves(
        img, n_region_curves=12, n_color_clusters=3, device="cpu"
    )
    assert len(curves) > 0, "no fill curves created"

    # Each curve's control-point bbox should be local (well under canvas size).
    # A canvas-spanning zig-zag (old behaviour) would be ~full width/height.
    for c in curves:
        cp = c.control_points.detach().numpy()
        span = max(cp[:, 0].ptp(), cp[:, 1].ptp())
        assert span < 0.6 * 64, f"curve spans {span:.1f}px — not a local stroke"
