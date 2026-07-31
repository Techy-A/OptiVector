"""
Neural Proxy Module for real-time previewing in CurveSplat.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class LightweightUNet(nn.Module):
    """
    A lightweight convolutional U-Net to predict the final optimized vector render
    from the target image and the current intermediate/initial curve render.
    """

    def __init__(self, in_channels: int = 6, out_channels: int = 3) -> None:
        super().__init__()
        # Encoder (Downsampling path)
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(32, 64),
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(64, 128),
        )

        # Decoder (Upsampling path with skip connections)
        self.conv_up1 = DoubleConv(192, 64)  # Concatenated: 128 (upsampled) + 64 (skip)
        self.conv_up2 = DoubleConv(96, 32)   # Concatenated: 64 (upsampled) + 32 (skip)

        # Output layer mapping to RGB range [0, 1]
        self.outc = nn.Sequential(
            nn.Conv2d(32, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 6, H, W) tensor containing concatenated target image (3ch)
           and current render (3ch).
        Returns: (B, 3, H, W) predicted final render.
        """
        # Save original height and width
        _, _, h, w = x.shape

        # Pad height and width to multiples of 4 for clean MaxPool/Interpolate transitions
        pad_h = (4 - (h % 4)) % 4
        pad_w = (4 - (w % 4)) % 4
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        # Decoder path
        u1 = F.interpolate(x3, size=x2.shape[2:], mode="bilinear", align_corners=False)
        u1 = torch.cat([u1, x2], dim=1)
        u1 = self.conv_up1(u1)

        u2 = F.interpolate(u1, size=x1.shape[2:], mode="bilinear", align_corners=False)
        u2 = torch.cat([u2, x1], dim=1)
        u2 = self.conv_up2(u2)

        out = self.outc(u2)

        # Crop back to original dimensions if padded
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :h, :w]

        return out


def save_dataset_sample(
    dir_path: str,
    idx: int,
    target: torch.Tensor,
    current_render: torch.Tensor,
    final_render: torch.Tensor,
) -> None:
    """
    Save target, current_render, and final_render tensors as a single compressed sample.
    Inputs can be tensors or numpy arrays.
    """
    os.makedirs(dir_path, exist_ok=True)

    # Convert tensors to numpy arrays
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    if isinstance(current_render, torch.Tensor):
        current_render = current_render.detach().cpu().numpy()
    if isinstance(final_render, torch.Tensor):
        final_render = final_render.detach().cpu().numpy()

    # If the inputs have a batch dimension, squeeze it out
    if target.ndim == 4:
        target = target.squeeze(0)
    if current_render.ndim == 4:
        current_render = current_render.squeeze(0)
    if final_render.ndim == 4:
        final_render = final_render.squeeze(0)

    # Convert channel first/last if necessary. We store as channel-first (C, H, W)
    if target.shape[0] != 3 and target.shape[2] == 3:
        target = target.transpose(2, 0, 1)
    if current_render.shape[0] != 3 and current_render.shape[2] == 3:
        current_render = current_render.transpose(2, 0, 1)
    if final_render.shape[0] != 3 and final_render.shape[2] == 3:
        final_render = final_render.transpose(2, 0, 1)

    np.savez_compressed(
        os.path.join(dir_path, f"sample_{idx:06d}.npz"),
        target=target.astype(np.float32),
        current_render=current_render.astype(np.float32),
        final_render=final_render.astype(np.float32),
    )
