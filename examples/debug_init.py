import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add src/python to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "python"))

from curvesplat.sdf_init import detect_edges, otsu_threshold, _to_grayscale, _image_to_float
from curvesplat.scene import Scene
from curvesplat.renderer import CurveSplatRenderer, render_scene

def main():
    image_path = "phoenix_black.png"
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found in the current working directory.")
        return

    # Load image
    print("Loading image...")
    target_np = plt.imread(image_path)
    img_float = _image_to_float(target_np)
    gray = _to_grayscale(img_float)
    
    # 1. Detect edges
    print("Detecting edges...")
    edges = detect_edges(gray, sigmas=(1.0, 2.0, 4.0))
    print(f"Total edge pixels: {np.sum(edges)} out of {edges.size}")
    
    # 2. Otsu threshold
    print("Computing Otsu interior...")
    interior, threshold = otsu_threshold(gray)
    print(f"Otsu threshold: {threshold:.4f}, interior pixels: {np.sum(interior)} out of {interior.size}")
    
    # Save debug maps
    os.makedirs("examples/output", exist_ok=True)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Original Grayscale")
    plt.imshow(gray, cmap="gray")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.title(f"Detected Edges ({np.sum(edges)} px)")
    plt.imshow(edges, cmap="gray")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.title("Otsu Interior")
    plt.imshow(interior, cmap="gray")
    plt.axis("off")
    
    plt.tight_layout()
    plt.savefig("examples/output/debug_init_maps.png", dpi=150)
    print("✓ Saved debug maps to examples/output/debug_init_maps.png")

    # Let's check max value and thresholding details
    # We do multi-scale Sobel manually here to print metrics
    from curvesplat.sdf_init import _gaussian_blur_2d, _sobel_gradients_pure, _non_maximum_suppression, _hysteresis_threshold
    all_magnitudes = []
    for sigma in (1.0, 2.0, 4.0):
        blurred = _gaussian_blur_2d(gray, sigma)
        gx, gy = _sobel_gradients_pure(blurred)
        mag = np.sqrt(gx**2 + gy**2)
        all_magnitudes.append(mag.max())
    print(f"Max gradient magnitudes per scale: {all_magnitudes}")
    
    # Check max gradient overall
    all_mags = []
    all_dirs = []
    for sigma in (1.0, 2.0, 4.0):
        blurred = _gaussian_blur_2d(gray, sigma)
        gx, gy = _sobel_gradients_pure(blurred)
        mag = np.sqrt(gx**2 + gy**2)
        direction = np.arctan2(gy, gx)
        all_mags.append(mag)
        all_dirs.append(direction)
    mags_stack = np.stack(all_mags, axis=0)
    best_scale = mags_stack.argmax(axis=0)
    H, W = gray.shape
    magnitude = np.zeros((H, W), dtype=np.float64)
    direction = np.zeros((H, W), dtype=np.float64)
    for i in range(3):
        mask = best_scale == i
        magnitude[mask] = all_mags[i][mask]
        direction[mask] = all_dirs[i][mask]
    thin = _non_maximum_suppression(magnitude, direction)
    print(f"Max thin edge gradient: {thin.max():.4f}")
    print(f"99.5th percentile of thin edge: {np.percentile(thin, 99.5):.4f}")
    print(f"98th percentile of thin edge: {np.percentile(thin, 98):.4f}")
    print(f"95th percentile of thin edge: {np.percentile(thin, 95):.4f}")
    print(f"90th percentile of thin edge: {np.percentile(thin, 90):.4f}")

if __name__ == "__main__":
    main()
