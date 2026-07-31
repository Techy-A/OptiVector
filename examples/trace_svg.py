"""
Filled-region tracer — CurveSplat's path to tracer-quality flat-color SVG.

Reuses initialize_filled_regions (k-means color quantize → per-region marching-
squares → closed cubic-Bezier loops) but with a HIGH color count, then exports
the closed regions as filled <path fill Z>. No optimization, no GPU — the trace
IS the init. This is the representation that maps cleanly to SVG (fills, not
soft splats), so the output is crisp flat-color, like a contour tracer.

Usage (runs on CPU):
    python3 examples/trace_svg.py --image phoenix.png --n_colors 32 --out examples/output/trace
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "python"))

from curvesplat.sdf_init import initialize_filled_regions  # noqa: E402
from curvesplat.scene import Scene, FilledShape  # noqa: E402
from curvesplat.fill_renderer import render_filled_curve_fn  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--n_colors", type=int, default=32)
    ap.add_argument("--min_region_frac", type=float, default=0.0008)
    ap.add_argument("--simplify", type=float, default=1.0)
    ap.add_argument("--max_dim", type=int, default=512)
    ap.add_argument("--out", default="examples/output/trace")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    im = Image.open(args.image).convert("RGB")
    im.thumbnail((args.max_dim, args.max_dim))
    img = np.asarray(im, np.float32) / 255.0
    H, W = img.shape[:2]
    print(f"✓ {args.image} → {W}×{H}, quantizing to {args.n_colors} colors...")

    loops = initialize_filled_regions(
        img, n_colors=args.n_colors,
        min_region_frac=args.min_region_frac, simplify_epsilon=args.simplify,
    )
    print(f"✓ Traced {len(loops)} filled regions")
    if not loops:
        print("  (no regions — try higher --n_colors or lower --min_region_frac)")
        return

    shapes = [FilledShape.from_region(lp) for lp in loops]
    bg = img.reshape(-1, 3).mean(0).clip(0.02, 0.98)
    bg_raw = torch.tensor(np.log(bg / (1 - bg)), dtype=torch.float32)
    scene = Scene(curves=shapes, canvas_size=(H, W), background=bg_raw)

    # Raster preview: composite the filled regions (sharp k, forward-only).
    scene.fill_k = 6.0
    scene.fill_samples = 16
    canvas = scene.background.view(3, 1, 1).expand(3, H, W).clone()
    with torch.no_grad():
        for shape in sorted(scene.curves, key=lambda c: c.z_index):
            seg = shape.segments
            canvas = render_filled_curve_fn(
                seg, shape.color_raw, shape.opacity_raw, canvas,
                samples_per_segment=scene.fill_samples, k=scene.fill_k,
            )
    preview = canvas.permute(1, 2, 0).numpy().clip(0, 1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.imsave(os.path.join(args.out, "trace_preview.png"), preview)

    # SVG: filled closed paths only (the tracer output).
    from importlib import import_module
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    vc = import_module("vectorize_cuda")
    svg_path = os.path.join(args.out, "traced.svg")
    vc.export_to_svg(scene, svg_path, quantize=False, min_opacity=0.0, min_area=0.0)
    print(f"✓ preview: {os.path.join(args.out, 'trace_preview.png')}")
    print(f"✓ svg:     {svg_path} ({os.path.getsize(svg_path)//1024} KB)")


if __name__ == "__main__":
    main()
