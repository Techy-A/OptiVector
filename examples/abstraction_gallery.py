"""
Abstraction gallery — CurveSplat's unique trick a tracer can't do: render the
SAME image at increasing stroke budgets, from a bold few-stroke abstraction to a
detailed one. The progression is the point: controllable, intentional vector art,
not a faithful copy.

Runs the optimizer at 3 budgets (each emits its own SVG) and stitches the
optimized renders into one labelled gallery PNG.

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 examples/abstraction_gallery.py --image phoenix.png
Outputs (under --out):
    gallery.png                  — the 3-up progression
    abstract/optimized_artwork.svg, stylized/..., detailed/...  — the SVGs
"""

import argparse
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
VECTORIZE = os.path.join(HERE, "vectorize_cuda.py")

# Bold few strokes → more, thinner strokes. Low counts read as intentional
# brushwork; the export prune (svg_min_*) keeps only the strokes that matter.
BUDGETS = [
    {"name": "abstract", "label": "Abstract · ~33 strokes",
     "n_fill": 8, "n_edge": 25, "max_width_edge": 6.0, "steps": 300},
    {"name": "stylized", "label": "Stylized · ~100 strokes",
     "n_fill": 25, "n_edge": 75, "max_width_edge": 4.0, "steps": 400},
    {"name": "detailed", "label": "Detailed · ~260 strokes",
     "n_fill": 60, "n_edge": 200, "max_width_edge": 2.5, "steps": 500},
]


def run_budget(image, out_dir, b, max_dim):
    cmd = [
        sys.executable, "-u", VECTORIZE,
        "--image", image, "--output_dir", out_dir,
        "--mode", "painterly", "--no_adaptive",
        "--n_fill", str(b["n_fill"]), "--n_edge", str(b["n_edge"]),
        "--max_width_edge", str(b["max_width_edge"]), "--steps", str(b["steps"]),
        "--max_dim", str(max_dim), "--polish_steps", "0",
        "--svg_min_opacity", "0.2", "--svg_min_area", "30", "--log_every", "50",
    ]
    print(f"\n=== {b['label']} ===")
    subprocess.run(cmd, check=True)
    return os.path.join(out_dir, "optimized_render.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="examples/output/gallery")
    ap.add_argument("--max_dim", type=int, default=384)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    renders = []
    for b in BUDGETS:
        out_dir = os.path.join(args.out, b["name"])
        os.makedirs(out_dir, exist_ok=True)
        renders.append((b["label"], run_budget(args.image, out_dir, b, args.max_dim)))

    target = np.asarray(Image.open(args.image).convert("RGB")) / 255.0
    panels = [("Target", target)] + [
        (label, np.asarray(Image.open(p).convert("RGB")) / 255.0) for label, p in renders
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 6))
    for ax, (label, img) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(label, color="#D7FE3D", fontweight="bold", fontsize=15)
        ax.axis("off")
        ax.set_facecolor("#0A0A0A")
    fig.patch.set_facecolor("#0A0A0A")
    plt.tight_layout()
    gallery_path = os.path.join(args.out, "gallery.png")
    plt.savefig(gallery_path, dpi=150, facecolor="#0A0A0A", bbox_inches="tight")
    plt.close()
    print(f"\n✓ Gallery: {gallery_path}")
    print(f"✓ SVGs under: {args.out}/<abstract|stylized|detailed>/optimized_artwork.svg")


if __name__ == "__main__":
    main()
