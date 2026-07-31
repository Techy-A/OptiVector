"""
Unit tests for Phase 3: Style-Aware Vectorization (Task 3.2).
"""

import sys
import os
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add src/python to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "python"))

from curvesplat.style import (
    VGGStyleExtractor,
    gram_matrix,
    extract_dominant_palette,
    analyze_stroke_widths,
    compute_palette_loss,
    compute_stroke_width_loss,
)
from curvesplat.scene import Scene, Curve


class TestStyleAnalysis:
    """Verify style extraction helper functions."""

    def test_gram_matrix(self):
        """gram_matrix should compute correct spatial correlation tensors."""
        B, C, H, W = 2, 8, 16, 16
        features = torch.randn(B, C, H, W)
        gram = gram_matrix(features)
        
        # Output shape should be (B, C, C)
        assert gram.shape == (B, C, C)
        
        # Verify manually for batch 0
        feat0 = features[0].view(C, H * W)
        expected0 = torch.mm(feat0, feat0.t()) / (C * H * W)
        assert torch.allclose(gram[0], expected0, atol=1e-6)

    def test_extract_dominant_palette(self):
        """extract_dominant_palette should return k colors within [0, 1]."""
        H, W = 32, 32
        # Create an image with distinct colored blocks
        img = np.zeros((H, W, 3), dtype=np.float32)
        img[:16, :16] = [1.0, 0.0, 0.0]  # Red
        img[:16, 16:] = [0.0, 1.0, 0.0]  # Green
        img[16:, :16] = [0.0, 0.0, 1.0]  # Blue
        img[16:, 16:] = [1.0, 1.0, 1.0]  # White

        k = 4
        palette = extract_dominant_palette(img, k=k)
        
        assert palette.shape == (k, 3)
        assert np.all(palette >= 0.0)
        assert np.all(palette <= 1.0)
        
        # Verify that we extracted the dominant colors approximately
        # Since it's k-means, they should be close to the original colors
        for c in [[1,0,0], [0,1,0], [0,0,1], [1,1,1]]:
            dists = np.sum((palette - c) ** 2, axis=1)
            assert np.min(dists) < 0.05, f"Dominant color {c} not found in palette"

    def test_analyze_stroke_widths(self):
        """analyze_stroke_widths should return positive mean and std values."""
        H, W = 64, 64
        # Create a synthetic image with a thick white line
        img = np.zeros((H, W, 3), dtype=np.float32)
        img[20:44, :] = 1.0  # 24px wide stroke
        
        mean_w, std_w = analyze_stroke_widths(img)
        
        assert mean_w > 0.0
        assert std_w > 0.0
        # Given a 24px wide white block, EDT distance to boundary (edges) should yield a reasonable estimate
        # Note that the function limits/filters widths, but it should be positive and non-zero
        assert isinstance(mean_w, float)
        assert isinstance(std_w, float)


class TestVGGStyleExtractor:
    """Verify pre-trained VGG feature extraction slices."""

    def test_vgg_extractor_initialization(self):
        """Extractor should initialize and handle has_vgg flag cleanly."""
        device = "cpu"
        extractor = VGGStyleExtractor(device=device)
        
        # It shouldn't crash regardless of torchvision availability
        assert hasattr(extractor, "has_vgg")
        
        if extractor.has_vgg:
            # Test forward pass
            B, C, H, W = 1, 3, 64, 64
            x = torch.randn(B, C, H, W).to(device)
            feats = extractor(x)
            
            # Should output 4 feature maps for conv1_2, conv2_2, conv3_3, conv4_3
            assert len(feats) == 4
            for f in feats:
                assert f.device.type == device
        else:
            # If VGG is not loaded, it should return an empty list
            x = torch.randn(1, 3, 64, 64).to(device)
            feats = extractor(x)
            assert feats == []


class TestStyleLossGradients:
    """Verify that style losses propagate gradients to curve parameters."""

    def test_palette_loss_gradients(self):
        """Palette loss should propagate gradients to color parameters."""
        device = "cpu"
        # Setup simple scene with 1 curve
        cp = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=torch.float32)
        # Pass a regular tensor to Curve constructor. Curve wraps it in nn.Parameter inside init.
        color_raw_init = torch.tensor([1.0, -1.0, 0.5], dtype=torch.float32)
        
        curve = Curve(
            control_points=cp,
            color=color_raw_init,
            width=torch.tensor(2.0, dtype=torch.float32),
            opacity=torch.tensor(1.0, dtype=torch.float32),
            z_index=0
        )
        
        scene = Scene(
            curves=[curve],
            canvas_size=(32, 32),
            background=torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float32),
            device=device
        )
        
        # Style palette with 2 distinct colors
        style_palette = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
        
        loss = compute_palette_loss(scene, style_palette)
        loss.backward()
        
        assert curve.color_raw.grad is not None
        assert curve.color_raw.grad.abs().sum() > 0.0

    def test_stroke_width_loss_gradients(self):
        """Stroke width distribution loss should propagate gradients to width parameters."""
        device = "cpu"
        cp = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=torch.float32)
        
        curve = Curve(
            control_points=cp,
            color=torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32),
            width=torch.tensor(1.5, dtype=torch.float32),
            opacity=torch.tensor(1.0, dtype=torch.float32),
            z_index=0
        )
        
        scene = Scene(
            curves=[curve],
            canvas_size=(32, 32),
            background=torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float32),
            device=device
        )
        
        loss = compute_stroke_width_loss(scene, target_mean=8.0, target_std=2.0)
        loss.backward()
        
        assert curve.width_raw.grad is not None
        assert curve.width_raw.grad.abs().sum() > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
