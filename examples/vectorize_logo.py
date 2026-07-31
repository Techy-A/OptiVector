#!/usr/bin/env python3
"""
CurveSplat Phase 1 — Example: Vectorize a Filled Circle

Creates a 256x256 target image of a filled white circle on a black background,
initializes a single Bezier curve with sensible starting parameters, and
optimizes it for 500 steps using AdamW with L1 loss.

Saves before/after/comparison PNGs to examples/output/.

Usage:
    cd SVG_AI
    python examples/vectorize_logo.py

Success criteria (SPEC.md §12):
    - Loss decreases monotonically
    - The rendered curve visibly approximates the circle after 500 steps
    - Before/after PNGs are saved

Note: A single open cubic Bezier cannot perfectly match a filled circle,
but it should converge to an arc that covers a significant portion of the
circle's boundary, demonstrating correct gradient flow.
"""

import os
import sys
import math

import torch
import torch.nn.functional as F

# Add src/python to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "python"))

from curvesplat.scene import Curve, Scene
from curvesplat.renderer import CurveSplatRenderer, render_scene
import numpy as np
from curvesplat.sdf_init import initialize_curves_from_image


def make_star_mask(size: int = 256, n_points: int = 5) -> np.ndarray:
    """
    Create a binary mask of a 5-pointed star centered on the canvas.
    Returns a (size, size) bool array.
    """
    cx, cy = size / 2.0, size / 2.0
    outer_r = size * 0.4
    inner_r = size * 0.16

    angles = []
    radii = []
    for i in range(n_points * 2):
        angle = (i * math.pi / n_points) - math.pi / 2
        angles.append(angle)
        radii.append(outer_r if i % 2 == 0 else inner_r)

    vertices = np.array([
        [cx + r * math.cos(a), cy + r * math.sin(a)]
        for a, r in zip(angles, radii)
    ])

    mask = np.zeros((size, size), dtype=bool)
    n_verts = len(vertices)

    for y in range(size):
        intersections = []
        for i in range(n_verts):
            j = (i + 1) % n_verts
            y1, y2 = vertices[i, 1], vertices[j, 1]
            x1, x2 = vertices[i, 0], vertices[j, 0]

            if (y1 <= y < y2) or (y2 <= y < y1):
                t = (y - y1) / (y2 - y1 + 1e-12)
                x_intersect = x1 + t * (x2 - x1)
                intersections.append(x_intersect)

        intersections.sort()
        for k in range(0, len(intersections) - 1, 2):
            x_start = max(0, int(math.ceil(intersections[k])))
            x_end = min(size, int(math.floor(intersections[k + 1])) + 1)
            mask[y, x_start:x_end] = True

    return mask


def run_phase2_demo(image_path: str = None) -> None:
    """Run the Phase 2 initialization demo (star-shape or custom image)."""
    from pytorch_msssim import ms_ssim
    from matplotlib.path import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    if image_path is not None:
        print("=" * 60)
        print(f"  CurveSplat Phase 2 — Custom Image Vectorization: {os.path.basename(image_path)}")
        print("=" * 60)

        # 1. Load target image
        try:
            target_np = plt.imread(image_path)
            # Normalise to [0, 1] float
            if target_np.dtype == np.uint8:
                target_np = target_np.astype(np.float64) / 255.0
            else:
                target_np = target_np.astype(np.float64)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return

        H, W = target_np.shape[:2]
        print(f"✓ Loaded image: {W}x{H}")

        # 2. Run initialization (is_binary=False for general photos)
        scene = initialize_curves_from_image(
            target_np,
            canvas_size=(H, W),
            is_binary=False,
            min_curves=20,
            max_curves=200,
            default_width=3.0,
            simplify_epsilon=1.5,
            corner_angle_threshold=30.0,
        )
        num_curves = len(scene.curves)
        print(f"✓ Extracted and fitted {num_curves} curves from the image")

        # 3. Render curves using the stroke renderer (pre-multiplied alpha composite)
        renderer = CurveSplatRenderer(num_samples=32)
        with torch.no_grad():
            # Backup original background and set to black for rendering
            orig_bg_raw = scene.background_raw.clone()
            scene.background_raw.data.fill_(-10.0)  # sigmoid(-10) ≈ 0
            canvas = render_scene(scene, renderer)
            # Restore original background
            scene.background_raw.data.copy_(orig_bg_raw)

        rendered_np = canvas.permute(1, 2, 0).numpy().clip(0, 1)

        # Save comparison image
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        # Handle grayscale vs color target image
        if target_np.ndim == 2:
            axes[0].imshow(target_np, cmap="gray")
        else:
            axes[0].imshow(target_np[:, :, :3])
        axes[0].set_title("Original Image", color="#D7FE3D", fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(rendered_np)
        axes[1].set_title(f"Initialized Strokes ({num_curves} curves)", color="#D7FE3D", fontweight="bold")
        axes[1].axis("off")

        fig.patch.set_facecolor("#0A0A0A")
        for ax in axes:
            ax.set_facecolor("#0A0A0A")
        plt.tight_layout()
        out_path = os.path.join(output_dir, "custom_vectorized.png")
        plt.savefig(out_path, dpi=150, facecolor="#0A0A0A", bbox_inches="tight")
        plt.close()
        print(f"✓ Saved visual comparison: examples/output/custom_vectorized.png")
        print("=" * 60)
        return

    # Default star mask demo
    size = 256
    print("=" * 60)
    print("  CurveSplat Phase 2 — SDF Initialization Demo (Star)")
    print("=" * 60)

    # 1. Create star mask
    star_mask = make_star_mask(size, n_points=5)
    print("✓ Created 5-pointed star target mask")

    # 2. Run initialization
    scene = initialize_curves_from_image(
        star_mask.astype(np.float64),
        canvas_size=(size, size),
        is_binary=True,
        min_curves=5,
        default_width=5.0,
        simplify_epsilon=1.5,
        corner_angle_threshold=25.0,
    )
    num_curves = len(scene.curves)
    print(f"✓ Initialized scene with {num_curves} curves using SDF pipeline")

    # 3. Render silhouette using polygon fill
    poly_points = []
    for curve in scene.curves:
        cp = curve.control_points.detach().cpu().numpy()
        t = np.linspace(0, 1, 32)
        omt = 1.0 - t
        pts = (
            omt[:, None]**3 * cp[0]
            + 3 * omt[:, None]**2 * t[:, None] * cp[1]
            + 3 * omt[:, None] * t[:, None]**2 * cp[2]
            + t[:, None]**3 * cp[3]
        )
        poly_points.extend(pts[:-1])
    poly_points = np.array(poly_points)

    y, x = np.mgrid[:size, :size]
    grid_points = np.column_stack([x.ravel(), y.ravel()])
    path = Path(poly_points)
    inside_mask = path.contains_points(grid_points).reshape(size, size)
    canvas = torch.from_numpy(inside_mask.astype(np.float32)).unsqueeze(0).expand(3, -1, -1)

    target_tensor = torch.from_numpy(star_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    rendered_gray = canvas.mean(dim=0, keepdim=True).unsqueeze(0).clamp(0, 1)

    # 4. Compute and print MS-SSIM
    score = ms_ssim(target_tensor, rendered_gray, data_range=1.0, size_average=True)
    score_val = score.item()
    print(f"✓ MS-SSIM score of initialized silhouette vs. target: {score_val:.4f}")

    # Save comparison image
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(star_mask, cmap="gray")
    axes[0].set_title("Target Star", color="#D7FE3D", fontweight="bold")
    axes[0].axis("off")

    rendered_np = canvas.permute(1, 2, 0).numpy().clip(0, 1)
    axes[1].imshow(rendered_np)
    axes[1].set_title(f"Rendered ({num_curves} curves, MS-SSIM={score_val:.3f})",
                    color="#D7FE3D", fontweight="bold")
    axes[1].axis("off")

    fig.patch.set_facecolor("#0A0A0A")
    for ax in axes:
        ax.set_facecolor("#0A0A0A")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "phase2_star_demo.png"),
               dpi=150, facecolor="#0A0A0A", bbox_inches="tight")
    plt.close()
    print(f"✓ Saved comparison: examples/output/phase2_star_demo.png")
    print("=" * 60)



def make_circle_target(
    size: int = 256,
    radius: float = 80.0,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Create a 256x256 antialiased filled circle target image.

    Uses a signed distance field for smooth, antialiased edges.

    Args:
        size: Image dimension (square).
        radius: Circle radius in pixels.
        device: Torch device.

    Returns:
        target: Shape (3, H, W) RGB image in [0, 1].
                White circle on black background.
    """
    cy, cx = size / 2.0, size / 2.0

    y = torch.arange(size, dtype=torch.float32, device=device) + 0.5
    x = torch.arange(size, dtype=torch.float32, device=device) + 0.5
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")

    # Signed distance from circle edge (negative inside, positive outside)
    dist = torch.sqrt((grid_x - cx) ** 2 + (grid_y - cy) ** 2) - radius

    # Smooth antialiasing: sigmoid transition over ~1.5 pixels
    alpha = torch.sigmoid(-dist * 2.0)  # (H, W)

    # White circle, black background
    target = alpha.unsqueeze(0).expand(3, -1, -1)  # (3, H, W)
    return target


def create_initial_curve(
    canvas_size: tuple[int, int] = (256, 256),
    device: str = "cpu",
) -> Curve:
    """
    Create a single curve with initialization that gives it a fighting chance
    to learn the circle shape.

    We initialize control points as a rough circular arc (quarter circle)
    centered on the canvas, with wide stroke width and high opacity.
    """
    H, W = canvas_size
    cx, cy = W / 2.0, H / 2.0
    r = 70.0  # slightly less than target radius

    # Initialize as a quarter-circle-ish arc using standard Bezier circle approx.
    # For a quarter circle, the magic number is k ≈ 0.5522847498
    k = 0.5522847498 * r

    control_points = torch.tensor(
        [
            [cx, cy - r],        # P0: top
            [cx + k, cy - r],    # P1: control toward right
            [cx + r, cy - k],    # P2: control toward bottom
            [cx + r, cy],        # P3: right
        ],
        dtype=torch.float32,
        device=device,
    )

    # Pre-sigmoid color: inverse_sigmoid(0.9) ≈ 2.2 → bright white-ish
    color = torch.tensor([2.2, 2.2, 2.2], dtype=torch.float32, device=device)

    # Pre-softplus width: we want a wide stroke to cover the circle
    # softplus(4.0) ≈ 4.02, which gives sigma_long=4, sigma_short=1
    # We need wider: softplus(30) ≈ 30, giving substantial coverage
    width = torch.tensor(30.0, dtype=torch.float32, device=device)

    # Pre-sigmoid opacity: inverse_sigmoid(0.95) ≈ 2.94
    opacity = torch.tensor(2.94, dtype=torch.float32, device=device)

    return Curve(
        control_points=control_points,
        color=color,
        width=width,
        opacity=opacity,
    )


def save_image_tensor(tensor: torch.Tensor, path: str) -> None:
    """
    Save a (3, H, W) float tensor as a PNG image.

    Uses raw file I/O to avoid requiring PIL/matplotlib at minimum.
    Falls back to matplotlib if available for nicer output.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt

        img = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        plt.imsave(path, img)
        return
    except ImportError:
        pass

    # Fallback: save as raw PPM (always works, no dependencies)
    img = tensor.detach().cpu().clamp(0, 1)
    img = (img * 255).byte()
    C, H, W = img.shape
    ppm_path = path.replace(".png", ".ppm")
    with open(ppm_path, "wb") as f:
        f.write(f"P6\n{W} {H}\n255\n".encode())
        # PPM expects HWC byte order
        f.write(img.permute(1, 2, 0).contiguous().numpy().tobytes())
    print(f"  (saved as PPM: {ppm_path} — install matplotlib for PNG)")


def save_comparison(
    target: torch.Tensor,
    before: torch.Tensor,
    after: torch.Tensor,
    path: str,
) -> None:
    """
    Save a side-by-side comparison: [target | before | after].

    Args:
        target: (3, H, W) target image.
        before: (3, H, W) initial render.
        after:  (3, H, W) optimized render.
        path:   Output file path.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        titles = ["Target", "Before (init)", "After (500 steps)"]
        images = [target, before, after]

        for ax, img, title in zip(axes, images, titles):
            img_np = img.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
            ax.imshow(img_np)
            ax.set_title(title, fontsize=14, fontweight="bold", color="white")
            ax.axis("off")

        fig.patch.set_facecolor("#0A0A0A")
        for ax in axes:
            ax.set_facecolor("#0A0A0A")
            ax.title.set_color("#D7FE3D")  # Neon lime accent per brand guidelines

        plt.tight_layout()
        plt.savefig(path, dpi=150, facecolor="#0A0A0A", bbox_inches="tight")
        plt.close()
    except ImportError:
        # Fallback: save individual images
        save_image_tensor(target, path.replace("comparison", "target"))
        save_image_tensor(before, path.replace("comparison", "before"))
        save_image_tensor(after, path.replace("comparison", "after"))


def main() -> None:
    """Run the Phase 1 optimization example."""

    # --- Configuration ---
    canvas_size = (256, 256)
    num_steps = 500
    lr = 0.01
    weight_decay = 0.01
    num_samples = 64
    device = "cpu"

    print("=" * 60)
    print("  CurveSplat Phase 1 — Single Curve → Circle")
    print("=" * 60)
    print(f"  Canvas:       {canvas_size[0]}x{canvas_size[1]}")
    print(f"  Steps:        {num_steps}")
    print(f"  LR:           {lr}")
    print(f"  Samples/curve: {num_samples}")
    print(f"  Device:       {device}")
    print("=" * 60)

    # --- Setup output directory ---
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    # --- Create target ---
    target = make_circle_target(size=canvas_size[0], radius=80.0, device=device)
    print(f"\n✓ Target: filled circle (radius=80px)")

    # --- Initialize curve and scene ---
    curve = create_initial_curve(canvas_size=canvas_size, device=device)
    scene = Scene(
        curves=[curve],
        canvas_size=canvas_size,
        # Black background: pre-sigmoid → sigmoid(-5) ≈ 0.007
        background=torch.tensor([-5.0, -5.0, -5.0], dtype=torch.float32, device=device),
        device=device,
    )
    renderer = CurveSplatRenderer(num_samples=num_samples)

    # --- Render "before" image ---
    with torch.no_grad():
        before_image = render_scene(scene, renderer)
    save_image_tensor(before_image, os.path.join(output_dir, "before.png"))
    print(f"✓ Saved: examples/output/before.png")

    # --- Optimizer (SPEC.md §8) ---
    optimizer = torch.optim.AdamW(
        scene.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
    )

    # --- Training loop ---
    print(f"\n{'Step':>6}  {'L1 Loss':>10}  {'Δ':>10}")
    print("-" * 32)

    prev_loss = float("inf")
    for step in range(1, num_steps + 1):
        optimizer.zero_grad()

        # Forward render
        rendered = render_scene(scene, renderer)

        # L1 loss
        loss = F.l1_loss(rendered, target)

        # Backward
        loss.backward()

        # Gradient clipping (SPEC.md §11: clip to 1.0)
        torch.nn.utils.clip_grad_norm_(scene.parameters(), max_norm=1.0)

        # Step
        optimizer.step()

        loss_val = loss.item()
        delta = prev_loss - loss_val
        prev_loss = loss_val

        if step % 50 == 0 or step == 1:
            print(f"  {step:>4}  {loss_val:>10.6f}  {delta:>+10.6f}")

    # --- Render "after" image ---
    with torch.no_grad():
        after_image = render_scene(scene, renderer)
    save_image_tensor(after_image, os.path.join(output_dir, "after.png"))
    print(f"\n✓ Saved: examples/output/after.png")

    # --- Save comparison ---
    with torch.no_grad():
        save_comparison(
            target, before_image, after_image,
            os.path.join(output_dir, "comparison.png"),
        )
    print(f"✓ Saved: examples/output/comparison.png")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print(f"  Final L1 loss: {prev_loss:.6f}")
    print(f"  Optimized curve parameters:")
    with torch.no_grad():
        print(f"    Color (RGB):   {curve.color.tolist()}")
        print(f"    Width:         {curve.width.item():.3f}")
        print(f"    Opacity:       {curve.opacity.item():.3f}")
        print(f"    Control pts:   {curve.control_points.tolist()}")
    print(f"{'=' * 60}")
    print(f"\n✓ Phase 1 complete. Check examples/output/ for results.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CurveSplat Examples")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2], help="Which phase demo to run (1 or 2)")
    parser.add_argument("--image", type=str, default=None, help="Path to custom image to run Phase 2 vectorization on")
    args = parser.parse_args()

    if args.phase == 1:
        main()
    else:
        run_phase2_demo(args.image)
