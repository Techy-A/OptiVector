"""
Unit tests for Phase 3: Neural Proxy Preview (Task 3.3).
"""

import sys
import os
import tempfile
import numpy as np
import pytest
import torch
import torch.nn as nn

# Add src/python to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "python"))

from curvesplat.proxy import (
    LightweightUNet,
    save_dataset_sample,
)


class TestProxyArchitecture:
    """Verify proxy model shape compliance, dynamic resolutions, and gradient propagation."""

    def test_forward_shape_compliance(self):
        """LightweightUNet should accept 6-channel input and yield 3-channel output of identical resolution."""
        model = LightweightUNet(in_channels=6, out_channels=3)
        model.eval()

        # Batch=2, Channels=6, Height=128, Width=128
        x = torch.randn(2, 6, 128, 128)
        with torch.no_grad():
            out = model(x)

        assert out.shape == (2, 3, 128, 128)
        assert torch.all(out >= 0.0)
        assert torch.all(out <= 1.0)

    def test_dynamic_resolutions(self):
        """Model should accept odd resolutions and crop/pad internally to return correct output shape."""
        model = LightweightUNet(in_channels=6, out_channels=3)
        model.eval()

        # Input dimensions not divisible by 4 (e.g., 61x65)
        x = torch.randn(1, 6, 61, 65)
        with torch.no_grad():
            out = model(x)

        assert out.shape == (1, 3, 61, 65)

    def test_gradient_flow(self):
        """Model should propagate gradients from loss functions back to input features and weights."""
        model = LightweightUNet(in_channels=6, out_channels=3)
        model.train()

        x = torch.randn(1, 6, 64, 64, requires_grad=True)
        target = torch.rand(1, 3, 64, 64)

        pred = model(x)
        loss = nn.L1Loss()(pred, target)
        loss.backward()

        # Verify gradients exist and are non-zero
        assert x.grad is not None
        assert x.grad.abs().sum() > 0.0

        for name, p in model.named_parameters():
            assert p.grad is not None, f"Parameter {name} has no gradient flow!"
            assert p.grad.abs().sum() > 0.0, f"Parameter {name} has zero gradient!"


class TestProxyDatasetUtilities:
    """Verify dataset saving and loading routines."""

    def test_dataset_saving_loading(self):
        """save_dataset_sample utility should write compressed tensors correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            target = torch.rand(3, 128, 128)
            current_render = torch.rand(3, 128, 128)
            final_render = torch.rand(3, 128, 128)

            save_dataset_sample(
                dir_path=tmpdir,
                idx=42,
                target=target,
                current_render=current_render,
                final_render=final_render,
            )

            expected_file = os.path.join(tmpdir, "sample_000042.npz")
            assert os.path.exists(expected_file)

            with np.load(expected_file) as data:
                assert "target" in data
                assert "current_render" in data
                assert "final_render" in data

                # Check shapes (should be channel-first)
                assert data["target"].shape == (3, 128, 128)
                assert data["current_render"].shape == (3, 128, 128)
                assert data["final_render"].shape == (3, 128, 128)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
