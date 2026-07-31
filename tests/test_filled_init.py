"""
Tests for the tracer-style filled-region scaffold and crisp SVG export.

Covers:
  - initialize_filled_regions extracts closed Bezier loops from flat regions
  - FilledShape round-trips region dicts (inverse-sigmoid color)
  - export_to_svg emits filled `fill="rgb()" … Z` paths + stroke paths, valid XML
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

from curvesplat.sdf_init import initialize_filled_regions  # noqa: E402
from curvesplat.scene import Curve, Scene, FilledShape  # noqa: E402

SVG_NS = "{http://www.w3.org/2000/svg}"


def _square_image(H=64, W=64):
    img = np.full((H, W, 3), 0.1, np.float32)
    img[16:48, 16:48] = [0.9, 0.2, 0.2]  # red square on dark background
    return img


class TestFilledInit:
    def test_extracts_closed_loops(self):
        loops = initialize_filled_regions(_square_image(), n_colors=3)
        assert len(loops) >= 2, "should find at least background + square"
        for lp in loops:
            assert lp["segments"].ndim == 3 and lp["segments"].shape[1:] == (4, 2)
            # exact closure: last segment P3 == first segment P0
            assert np.allclose(lp["segments"][-1, 3], lp["segments"][0, 0])

    def test_z_order_largest_first(self):
        loops = initialize_filled_regions(_square_image(), n_colors=3)
        areas = [lp["area"] for lp in loops]
        assert areas == sorted(areas, reverse=True)
        assert [lp["z_index"] for lp in loops] == list(range(len(loops)))

    def test_red_region_color_recovered(self):
        loops = initialize_filled_regions(_square_image(), n_colors=3)
        colors = np.stack([lp["color"] for lp in loops])
        # some region should be reddish (R high, G/B low)
        reddish = (colors[:, 0] > 0.6) & (colors[:, 1] < 0.4) & (colors[:, 2] < 0.4)
        assert reddish.any(), f"no red region recovered; colors={np.round(colors,2)}"

    def test_filledshape_from_region(self):
        loops = initialize_filled_regions(_square_image(), n_colors=3)
        shape = FilledShape.from_region(loops[0])
        assert shape.is_closed is True
        assert shape.segments.shape[1:] == (4, 2)
        # color round-trips through inverse-sigmoid within tolerance
        recovered = shape.color.detach().numpy()
        assert np.allclose(recovered, np.clip(loops[0]["color"], 0.01, 0.99), atol=0.02)


class TestSvgExport:
    def test_export_mixed_scene(self, tmp_path):
        import vectorize_cuda  # noqa: E402  (examples/ on sys.path)

        loops = initialize_filled_regions(_square_image(), n_colors=3)
        filled = [FilledShape.from_region(lp) for lp in loops]
        stroke = Curve.create(canvas_size=(64, 64))
        scene = Scene(curves=filled + [stroke], canvas_size=(64, 64))

        out = tmp_path / "out.svg"
        vectorize_cuda.export_to_svg(scene, str(out))

        root = ET.parse(str(out)).getroot()  # raises if invalid XML
        paths = root.findall(f".//{SVG_NS}path")
        fills = [p for p in paths if p.get("fill", "none") not in ("none", None)]
        strokes = [p for p in paths if p.get("fill") == "none"]

        assert len(fills) == len(filled)
        assert len(strokes) >= 1
        # filled paths are closed
        for p in fills:
            assert p.get("d").strip().endswith("Z")
        # color quantization sanity: at most 256 distinct fill colors
        assert len({p.get("fill") for p in fills}) <= 256
