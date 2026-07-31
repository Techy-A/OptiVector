"""
Unit tests for Phase 3: Semantic Layer Decomposition.
"""

import sys
import os
import numpy as np
import pytest
import torch
import torch.nn.functional as F

# Add src/python to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "python"))

from curvesplat.segmentation import generate_semantic_masks, generate_kmeans_fallback_masks
from curvesplat.scene import Scene, Curve
from curvesplat.renderer import CurveSplatRenderer

# ---------------------------------------------------------------------------
# Test: Segmentation Modules
# ---------------------------------------------------------------------------

class TestSegmentation:
    """Verify semantic mask generation and fallbacks."""

    def test_kmeans_fallback_masks(self):
        """K-means fallback should generate the expected number of binary masks."""
        H, W = 64, 64
        image_np = np.random.rand(H, W, 3).astype(np.float32)
        
        num_masks = 3
        masks = generate_kmeans_fallback_masks(image_np, num_masks=num_masks)
        
        assert len(masks) == num_masks
        for mask in masks:
            assert mask.shape == (H, W)
            assert mask.dtype == bool
            
        # Ensure masks are non-overlapping and partition the image
        combined = np.zeros((H, W), dtype=int)
        for i, mask in enumerate(masks):
            combined[mask] += 1
        assert np.all(combined == 1), "Masks should partition the pixel space exactly once."

    def test_generate_semantic_masks(self):
        """generate_semantic_masks entry point should handle inputs cleanly."""
        H, W = 64, 64
        image_np = np.zeros((H, W, 3), dtype=np.float32)
        # Draw a simple white square in center
        image_np[16:48, 16:48] = 1.0
        
        prompts = ["background", "square"]
        masks = generate_semantic_masks(image_np, prompts, device="cpu")
        
        assert len(masks) == len(prompts)
        for mask in masks:
            assert mask.shape == (H, W)
            assert mask.dtype == bool


# ---------------------------------------------------------------------------
# Test: Crop-Aware Rendering & Masked Losses
# ---------------------------------------------------------------------------

class TestCropAwareRendering:
    """Verify that renderer correctly handles crop boxes and propagates gradients."""

    def test_crop_box_render_dimensions(self):
        """render_scene_shuffled should render directly onto the cropped canvas size."""
        # Need to import it from vectorize_cuda
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
        from vectorize_cuda import render_scene_shuffled
        
        device = "cpu"
        H, W = 128, 128
        
        # Create a single curve
        cp = torch.tensor([[10, 10], [20, 30], [30, 20], [40, 40]], dtype=torch.float32)
        curve = Curve(
            control_points=cp,
            color=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
            width=torch.tensor(2.0, dtype=torch.float32),
            opacity=torch.tensor(1.0, dtype=torch.float32),
            z_index=0
        )
        
        scene = Scene(
            curves=[curve],
            canvas_size=(H, W),
            background=torch.tensor([-2.0, -2.0, -2.0], dtype=torch.float32),
            device=device
        )
        
        renderer = CurveSplatRenderer(num_samples=8)
        
        # Test with crop_box: y[20:60], x[30:80]
        crop_box = (20, 60, 30, 80)
        crop_h = 60 - 20
        crop_w = 80 - 30
        
        rendered = render_scene_shuffled(
            scene, renderer, shuffle=False, render_size=(crop_h, crop_w), crop_box=crop_box
        )
        
        assert rendered.shape == (3, crop_h, crop_w)
        
    def test_masked_loss_gradients(self):
        """Masked loss should propagate gradients back to curve control points."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
        from vectorize_cuda import render_scene_shuffled
        
        device = "cpu"
        H, W = 64, 64
        
        # Create curve and enable gradients
        cp = torch.tensor([[10, 10], [12, 15], [15, 12], [20, 20]], dtype=torch.float32, requires_grad=True)
        curve = Curve(
            control_points=cp,
            color=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
            width=torch.tensor(2.0, dtype=torch.float32),
            opacity=torch.tensor(1.0, dtype=torch.float32),
            z_index=0
        )
        
        scene = Scene(
            curves=[curve],
            canvas_size=(H, W),
            background=torch.tensor([-2.0, -2.0, -2.0], dtype=torch.float32),
            device=device
        )
        
        renderer = CurveSplatRenderer(num_samples=8)
        
        # Bounding box of the active region
        crop_box = (5, 25, 5, 25)
        crop_h = 25 - 5
        crop_w = 25 - 5
        
        # Target crop and mask
        target_crop = torch.ones(3, crop_h, crop_w, dtype=torch.float32)
        mask_crop = torch.zeros(crop_h, crop_w, dtype=torch.float32)
        mask_crop[5:15, 5:15] = 1.0  # active mask region
        
        # Render scene in cropped coordinates
        rendered = render_scene_shuffled(
            scene, renderer, shuffle=False, render_size=(crop_h, crop_w), crop_box=crop_box
        )
        
        # Apply mask
        rendered_masked = rendered * mask_crop.unsqueeze(0)
        target_masked = target_crop * mask_crop.unsqueeze(0)
        
        # Compute loss
        loss = F.l1_loss(rendered_masked, target_masked, reduction='sum') / (mask_crop.sum() * 3.0 + 1e-8)
        loss.backward()
        
        # Check that control points received gradient
        assert curve.control_points.grad is not None
        assert not torch.isnan(curve.control_points.grad).any()
        assert curve.control_points.grad.abs().sum() > 0.0, "Control points should receive non-zero gradients from masked loss."


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
