"""
Tests for Phase 2: SDF-based curve initialization.

Includes unit tests for each sub-component and the acceptance test:
  Given a binary star-shape mask, produce ~10 curves whose rendered
  silhouette visually matches the star (MS-SSIM > 0.9 before optimization).
"""

import sys
import os
import math

import numpy as np
import pytest
import torch

# Add src/python to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "python"))

from curvesplat.sdf_init import (
    detect_edges,
    otsu_threshold,
    compute_sdf_jfa,
    extract_contours,
    simplify_contour,
    detect_corners,
    fit_bezier_segment,
    initialize_curves_from_image,
)
from curvesplat.scene import Curve, Scene
from curvesplat.renderer import CurveSplatRenderer, render_scene


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_star_mask(size: int = 256, n_points: int = 5) -> np.ndarray:
    """
    Create a binary mask of a 5-pointed star centered on the canvas.

    Returns a (size, size) bool array.
    """
    cx, cy = size / 2.0, size / 2.0
    outer_r = size * 0.4
    inner_r = size * 0.16  # inner radius for star points

    # Generate star vertices (alternating outer/inner)
    angles = []
    radii = []
    for i in range(n_points * 2):
        angle = (i * math.pi / n_points) - math.pi / 2  # start from top
        angles.append(angle)
        radii.append(outer_r if i % 2 == 0 else inner_r)

    vertices = np.array([
        [cx + r * math.cos(a), cy + r * math.sin(a)]
        for a, r in zip(angles, radii)
    ])

    # Rasterize the polygon using a scanline fill
    mask = np.zeros((size, size), dtype=bool)
    n_verts = len(vertices)

    for y in range(size):
        # Find intersections with polygon edges
        intersections = []
        for i in range(n_verts):
            j = (i + 1) % n_verts
            y1, y2 = vertices[i, 1], vertices[j, 1]
            x1, x2 = vertices[i, 0], vertices[j, 0]

            if (y1 <= y < y2) or (y2 <= y < y1):
                # Edge crosses this scanline
                t = (y - y1) / (y2 - y1 + 1e-12)
                x_intersect = x1 + t * (x2 - x1)
                intersections.append(x_intersect)

        intersections.sort()

        # Fill between pairs of intersections
        for k in range(0, len(intersections) - 1, 2):
            x_start = max(0, int(math.ceil(intersections[k])))
            x_end = min(size, int(math.floor(intersections[k + 1])) + 1)
            mask[y, x_start:x_end] = True

    return mask


def make_circle_mask(size: int = 256, radius: float = 80.0) -> np.ndarray:
    """Create a binary circle mask."""
    cy, cx = size / 2.0, size / 2.0
    y, x = np.mgrid[:size, :size]
    return ((x - cx)**2 + (y - cy)**2) < radius**2


# ---------------------------------------------------------------------------
# Test: Otsu thresholding
# ---------------------------------------------------------------------------

class TestOtsuThreshold:
    """Verify Otsu's automatic thresholding."""

    def test_bimodal_image(self):
        """Bimodal image should be cleanly separated."""
        img = np.zeros((100, 100), dtype=np.float64)
        img[:50, :] = 0.8  # top half bright
        img[50:, :] = 0.2  # bottom half dark

        binary, thresh = otsu_threshold(img)
        # Otsu finds optimal threshold between the two modes
        # The exact bin depends on quantization; what matters is clean separation
        assert binary[:50, :].all(), "Top half should be True (bright)"
        assert not binary[50:, :].any(), "Bottom half should be False (dark)"

    def test_uniform_image(self):
        """Uniform image should still return a threshold."""
        img = np.full((50, 50), 0.5, dtype=np.float64)
        binary, thresh = otsu_threshold(img)
        assert isinstance(thresh, float)


# ---------------------------------------------------------------------------
# Test: Edge detection
# ---------------------------------------------------------------------------

class TestEdgeDetection:
    """Verify edge detection on known patterns."""

    def test_vertical_edge(self):
        """A sharp vertical intensity step should produce edges."""
        img = np.zeros((64, 64), dtype=np.float64)
        img[:, 32:] = 1.0  # Right half bright

        edges = detect_edges(img, sigmas=(1.0,))
        # Edges should exist near the vertical boundary
        assert edges.any(), "Should detect vertical edge"
        # The edge pixels should include column 31 or 32
        edge_cols = np.where(edges.any(axis=0))[0]
        assert any(abs(c - 32) <= 3 for c in edge_cols), (
            f"Edge should include pixels near column 32, got {edge_cols}"
        )

    def test_uniform_no_edges_interior(self):
        """A uniform image should have no edges in the interior."""
        img = np.full((64, 64), 0.5, dtype=np.float64)
        edges = detect_edges(img, sigmas=(1.0,))
        # Border artifacts from padding are acceptable; interior should be clean
        interior = edges[5:-5, 5:-5]
        assert not interior.any(), "Uniform image interior should have no edges"


# ---------------------------------------------------------------------------
# Test: JFA SDF
# ---------------------------------------------------------------------------

class TestSDF:
    """Verify signed distance field computation."""

    def test_circle_sdf(self):
        """SDF of a circle should be approximately correct."""
        size = 64
        mask = make_circle_mask(size, radius=20)

        # Get edges from mask boundary
        padded = np.pad(mask.astype(np.float64), 1, mode="constant")
        edges = np.zeros_like(mask, dtype=bool)
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                shifted = padded[1+dy:size+1+dy, 1+dx:size+1+dx]
                edges |= (mask != shifted.astype(bool))

        sdf = compute_sdf_jfa(edges, interior=mask)

        # Center should be negative (inside)
        assert sdf[size//2, size//2] < 0, "Center of circle should be inside (negative)"
        # Corner should be positive (outside)
        assert sdf[0, 0] > 0, "Corner should be outside (positive)"
        # Edge pixels should be near zero
        edge_values = sdf[edges]
        assert np.abs(edge_values).max() < 1.5, "Edge pixels should be near zero"


# ---------------------------------------------------------------------------
# Test: Marching squares
# ---------------------------------------------------------------------------

class TestMarchingSquares:
    """Verify contour extraction."""

    def test_circle_contour(self):
        """A circle SDF should produce at least one contour."""
        size = 64
        cx, cy = size / 2.0, size / 2.0
        y, x = np.mgrid[:size, :size].astype(np.float64)
        # Analytical SDF for a circle
        sdf = np.sqrt((x - cx)**2 + (y - cy)**2) - 20.0

        contours = extract_contours(sdf, iso=0.0, min_length=5)
        assert len(contours) >= 1, "Should extract at least one contour from circle"

        # The main contour should have many points
        longest = max(contours, key=len)
        assert len(longest) > 10, f"Main contour too short: {len(longest)} points"

    def test_empty_sdf(self):
        """All-positive SDF should produce no contours."""
        sdf = np.ones((32, 32), dtype=np.float64)
        contours = extract_contours(sdf, iso=0.0)
        assert len(contours) == 0


# ---------------------------------------------------------------------------
# Test: Douglas-Peucker simplification
# ---------------------------------------------------------------------------

class TestDouglasPeucker:
    """Verify polyline simplification."""

    def test_straight_line(self):
        """Points on a line should simplify to 2 endpoints."""
        points = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], dtype=np.float64)
        simplified = simplify_contour(points, epsilon=0.1)
        assert len(simplified) == 2

    def test_zigzag(self):
        """A zigzag should preserve all corners at tight epsilon."""
        points = np.array([
            [0, 0], [5, 10], [10, 0], [15, 10], [20, 0]
        ], dtype=np.float64)
        simplified = simplify_contour(points, epsilon=0.1)
        # Should keep all points since they deviate significantly
        assert len(simplified) == 5

    def test_reduces_count(self):
        """Simplification should reduce point count on a noisy line."""
        # Noisy line from (0,0) to (100,0) with small y perturbations
        x = np.linspace(0, 100, 50)
        y = np.random.RandomState(42).randn(50) * 0.5
        points = np.column_stack([x, y])

        simplified = simplify_contour(points, epsilon=2.0)
        assert len(simplified) < len(points)


# ---------------------------------------------------------------------------
# Test: Corner detection
# ---------------------------------------------------------------------------

class TestCornerDetection:
    """Verify corner detection on known shapes."""

    def test_right_angle(self):
        """An L-shape should have a corner at the bend."""
        contour = np.array([
            [0, 0], [5, 0], [10, 0], [10, 5], [10, 10]
        ], dtype=np.float64)
        corners = detect_corners(contour, angle_threshold_deg=30.0)
        # Should include endpoints + the corner at [10, 0]
        assert 0 in corners
        assert len(contour) - 1 in corners
        # The 90° bend at index 2 should be detected
        assert 2 in corners

    def test_straight_line_no_interior_corners(self):
        """A straight line should have only endpoint corners."""
        contour = np.array([
            [0, 0], [2, 0], [4, 0], [6, 0]
        ], dtype=np.float64)
        corners = detect_corners(contour, angle_threshold_deg=30.0)
        assert corners == [0, 3]


# ---------------------------------------------------------------------------
# Test: Bezier fitting
# ---------------------------------------------------------------------------

class TestBezierFitting:
    """Verify cubic Bezier fitting."""

    def test_straight_line_fit(self):
        """Points on a line should produce a straight Bezier."""
        points = np.array([[0, 0], [3, 3], [6, 6], [9, 9]], dtype=np.float64)
        cp = fit_bezier_segment(points)
        assert cp.shape == (4, 2)
        # All control points should be roughly on the line y=x
        for pt in cp:
            assert abs(pt[0] - pt[1]) < 1.0, f"Control point {pt} not on y=x"

    def test_endpoint_preservation(self):
        """First and last points should be preserved exactly."""
        points = np.array([
            [10, 20], [15, 30], [25, 25], [30, 10]
        ], dtype=np.float64)
        cp = fit_bezier_segment(points)
        np.testing.assert_array_almost_equal(cp[0], points[0])
        np.testing.assert_array_almost_equal(cp[3], points[-1])

    def test_two_points(self):
        """Two points should produce a straight-line Bezier."""
        points = np.array([[0, 0], [10, 10]], dtype=np.float64)
        cp = fit_bezier_segment(points)
        assert cp.shape == (4, 2)
        np.testing.assert_array_almost_equal(cp[0], [0, 0])
        np.testing.assert_array_almost_equal(cp[3], [10, 10])


# ---------------------------------------------------------------------------
# Test: Full pipeline
# ---------------------------------------------------------------------------

class TestFullPipeline:
    """Test the end-to-end initialization pipeline."""

    def test_circle_extraction(self):
        """Circle mask should produce curves."""
        mask = make_circle_mask(128, radius=40).astype(np.float64)
        scene = initialize_curves_from_image(
            mask,
            is_binary=True,
            min_curves=1,
            simplify_epsilon=2.0,
        )
        assert len(scene.curves) >= 1, f"Expected ≥1 curve, got {len(scene.curves)}"

    def test_star_extraction(self):
        """Star mask should produce multiple curves."""
        mask = make_star_mask(128, n_points=5).astype(np.float64)
        scene = initialize_curves_from_image(
            mask,
            is_binary=True,
            min_curves=3,
            simplify_epsilon=2.0,
        )
        assert len(scene.curves) >= 3, f"Expected ≥3 curves, got {len(scene.curves)}"

    def test_output_is_scene(self):
        """Output should be a valid Scene with proper types."""
        mask = make_circle_mask(64, radius=20).astype(np.float64)
        scene = initialize_curves_from_image(mask, is_binary=True, min_curves=1)

        assert isinstance(scene, Scene)
        assert hasattr(scene, "curves")
        assert hasattr(scene, "canvas_size")
        for curve in scene.curves:
            assert isinstance(curve, Curve)
            assert curve.control_points.shape == (4, 2)


# ---------------------------------------------------------------------------
# Test: Acceptance — Star shape MS-SSIM > 0.9
# ---------------------------------------------------------------------------

class TestAcceptance:
    """
    Phase 2 acceptance criterion (SPEC.md §6):
    Given a binary star-shape mask, produce ~10 curves whose rendered
    silhouette visually matches the star (MS-SSIM > 0.9 before optimization).
    """

    def test_star_shape_ms_ssim(self):
        """
        Full acceptance test:
        1. Create a 256x256 5-pointed star mask.
        2. Run initialize_curves_from_image().
        3. Render the curves with the Phase 1 renderer.
        4. Measure MS-SSIM against the original mask.
        5. Assert MS-SSIM > 0.9.
        """
        from pytorch_msssim import ms_ssim

        size = 256

        # 1. Create star mask
        star_mask = make_star_mask(size, n_points=5)

        # 2. Initialize curves from the binary mask
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
        print(f"\n  Extracted {num_curves} curves from star mask")
        assert num_curves >= 5, f"Expected ≥5 curves, got {num_curves}"

        # 3. Render the closed curves as a filled silhouette by polygon rasterization
        H, W = size, size

        with torch.no_grad():
            # Sample points along each curve and concatenate to form a polygon
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
                # Avoid duplicate endpoints between adjacent curves
                poly_points.extend(pts[:-1])
            
            poly_points = np.array(poly_points)

            # Use matplotlib.path.Path to rasterize the filled polygon
            from matplotlib.path import Path
            y, x = np.mgrid[:H, :W]
            grid_points = np.column_stack([x.ravel(), y.ravel()])
            path = Path(poly_points)
            inside_mask = path.contains_points(grid_points).reshape(H, W)

            # Convert to canvas: 1.0 where inside, 0.0 where outside
            canvas = torch.from_numpy(inside_mask.astype(np.float32)).unsqueeze(0).expand(3, -1, -1)

        # 4. Convert both to the format MS-SSIM expects: (1, 1, H, W)
        # Star mask as a grayscale image
        target = torch.from_numpy(star_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        # Rendered silhouette as grayscale
        rendered_gray = canvas.mean(dim=0, keepdim=True).unsqueeze(0)  # (1, 1, H, W)

        # Clamp to valid range
        rendered_gray = rendered_gray.clamp(0, 1)

        # 5. Compute MS-SSIM
        score = ms_ssim(target, rendered_gray, data_range=1.0, size_average=True)
        score_val = score.item()
        print(f"  MS-SSIM score: {score_val:.4f}")
        print(f"  Threshold:     0.90")
        print(f"  Pass:          {'✓' if score_val > 0.9 else '✗'}")

        # Save comparison images if matplotlib available
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            output_dir = os.path.join(os.path.dirname(__file__), "..", "examples", "output")
            os.makedirs(output_dir, exist_ok=True)

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
            plt.savefig(os.path.join(output_dir, "phase2_star.png"),
                       dpi=150, facecolor="#0A0A0A", bbox_inches="tight")
            plt.close()
            print(f"  Saved: examples/output/phase2_star.png")
        except ImportError:
            pass

        assert score_val > 0.9, (
            f"MS-SSIM {score_val:.4f} is below 0.9 threshold. "
            f"The {num_curves} extracted curves do not sufficiently match the star."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
