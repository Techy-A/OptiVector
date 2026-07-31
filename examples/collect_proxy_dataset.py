"""
Dataset collection script for training the real-time Neural Proxy in CurveSplat.
Generates (target, intermediate_render, final_render) pairs from images in a directory.
"""

import sys
import os
import argparse
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# Add src/python to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "python"))
sys.path.insert(0, os.path.dirname(__file__))

from curvesplat.scene import Scene
from curvesplat.renderer import CurveSplatRenderer
from curvesplat.proxy import save_dataset_sample
from vectorize_cuda import (
    initialize_region_fill_curves,
    initialize_curves_from_image,
    render_scene_shuffled,
)


def collect_dataset(
    image_dir: str,
    output_dir: str,
    steps: int = 300,
    n_fill: int = 80,
    n_edge: int = 80,
    capture_steps: list = None,
    device: str = "cpu",
):
    if capture_steps is None:
        capture_steps = [0, 5, 15, 30, 50, 100, 150, 200, 250]

    os.makedirs(output_dir, exist_ok=True)
    image_patterns = ["*.jpg", "*.jpeg", "*.png"]
    image_paths = []
    for pattern in image_patterns:
        image_paths.extend(glob.glob(os.path.join(image_dir, pattern)))
        image_paths.extend(glob.glob(os.path.join(image_dir, pattern.upper())))

    # Remove duplicates
    image_paths = list(set(image_paths))

    if not image_paths:
        print(f"No images found in {image_dir}! Please put some images (.jpg/.png) there.")
        return

    print("=" * 65)
    print(f"  CurveSplat — Dataset Collection (Neural Proxy)")
    print("=" * 65)
    print(f"  Input Directory:  {image_dir}")
    print(f"  Output Directory: {output_dir}")
    print(f"  Images Found:     {len(image_paths)}")
    print(f"  Steps per Image:  {steps}")
    print(f"  Capture Steps:    {capture_steps}")
    print(f"  Device:           {device.upper()}")
    print("=" * 65)

    sample_counter = 0
    renderer = CurveSplatRenderer(num_samples=8)

    for img_idx, img_path in enumerate(image_paths):
        print(f"\n[{img_idx + 1}/{len(image_paths)}] Processing: {os.path.basename(img_path)}")
        try:
            target_np = plt.imread(img_path)
            if target_np.dtype == np.uint8:
                target_np = target_np.astype(np.float32) / 255.0
            else:
                target_np = target_np.astype(np.float32)

            if target_np.ndim == 2:
                target_np = np.stack([target_np] * 3, axis=2)
            elif target_np.shape[2] == 4:
                rgb = target_np[:, :, :3]
                alpha = target_np[:, :, 3:4]
                target_np = rgb * alpha + (1.0 - alpha)

            target_rgb = target_np[:, :, :3]
            H, W = target_rgb.shape[:2]

            # Downscale if image is too large for fast dataset collection
            max_dim = 256
            if max(H, W) > max_dim:
                scale = max_dim / max(H, W)
                new_H, new_W = int(H * scale), int(W * scale)
                t_tensor = torch.from_numpy(target_rgb).permute(2, 0, 1).unsqueeze(0).float()
                t_resized = F.interpolate(t_tensor, size=(new_H, new_W), mode="bilinear", align_corners=False)
                target_rgb = t_resized.squeeze(0).permute(1, 2, 0).numpy().clip(0, 1)
                H, W = new_H, new_W

            target_tensor = torch.from_numpy(target_rgb).permute(2, 0, 1).float().to(device)

            # Initialize Scene
            fill_curves = initialize_region_fill_curves(
                target_rgb, n_region_curves=n_fill, n_color_clusters=12, device=device
            )
            for c in fill_curves:
                c.max_width = 35.0

            edge_scene = initialize_curves_from_image(
                target_rgb,
                canvas_size=(H, W),
                is_binary=False,
                min_curves=10,
                max_curves=n_edge,
                default_width=2.0,
                device=device,
            )
            edge_curves = list(edge_scene.curves)
            for c in edge_curves:
                c.max_width = 8.0

            all_curves = fill_curves + edge_curves
            for idx_c, c in enumerate(all_curves):
                c.z_index = idx_c

            corner_pixels = [
                target_rgb[0, 0], target_rgb[0, -1], target_rgb[-1, 0], target_rgb[-1, -1]
            ]
            bg_rgb = np.mean(corner_pixels, axis=0).clip(0.02, 0.98)
            bg_raw = np.log(bg_rgb / (1.0 - bg_rgb)).astype(np.float32)

            scene = Scene(
                curves=all_curves,
                canvas_size=(H, W),
                background=torch.tensor(bg_raw, dtype=torch.float32, device=device),
                device=device,
            )

            # Optimizer Setup
            cp_params = []
            other_params = []
            for name, param in scene.named_parameters():
                if "control_points" in name:
                    cp_params.append(param)
                else:
                    other_params.append(param)

            optimizer = torch.optim.Adam(
                [
                    {"params": cp_params, "lr": 0.02 * 2.0},
                    {"params": other_params, "lr": 0.02},
                ]
            )

            # Optimization Loop
            captured_renders = {}
            for step in range(steps + 1):
                optimizer.zero_grad()

                rendered = render_scene_shuffled(
                    scene, renderer, shuffle=True, render_size=(H, W)
                )
                loss = F.l1_loss(rendered, target_tensor)

                if step in capture_steps:
                    captured_renders[step] = rendered.detach().cpu()

                if step < steps:
                    loss.backward()
                    optimizer.step()

            final_render = rendered.detach().cpu()

            # Save dataset samples
            for step, current_render in captured_renders.items():
                save_dataset_sample(
                    output_dir,
                    sample_counter,
                    target_tensor.cpu(),
                    current_render,
                    final_render,
                )
                sample_counter += 1

            print(f"  → Saved {len(captured_renders)} sample pairs (Total samples: {sample_counter})")

        except Exception as e:
            print(f"  → Error vectorizing {os.path.basename(img_path)}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n✓ Dataset collection completed. Saved {sample_counter} total samples to {output_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect CurveSplat neural proxy training dataset")
    parser.add_argument("--image_dir", type=str, default="examples",
                        help="Directory containing source images to vectorize")
    parser.add_argument("--output_dir", type=str, default=".tmp/proxy_dataset",
                        help="Output directory to save dataset .npz samples")
    parser.add_argument("--steps", type=int, default=150,
                        help="Optimization steps per image")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Hardware device (cpu or cuda)")
    args = parser.parse_args()

    collect_dataset(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        steps=args.steps,
        device=args.device,
    )
