"""
Differentiable loss functions for CurveSplat.

Implements TASK 1.1: wraps the lpips perceptual loss library for use in
differentiable optimization.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import lpips
    HAS_LPIPS_LIB = True
except ImportError:
    HAS_LPIPS_LIB = False


class LPIPSLoss(nn.Module):
    """
    Perceptual Loss wrapper using the `lpips` library.
    Evaluates similarity between two images based on human visual perception.
    """

    def __init__(self, net: str = "vgg", device: str = "cpu") -> None:
        super().__init__()
        if not HAS_LPIPS_LIB:
            raise ImportError("The 'lpips' library is not installed. Run 'pip install lpips' first.")
        
        # Load the LPIPS model and map to device
        self.loss_fn = lpips.LPIPS(net=net, verbose=False).to(device)
        self.loss_fn.eval()
        
        # Freeze model parameters to prevent optimization
        for p in self.loss_fn.parameters():
            p.requires_grad = False

    def forward(self, rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Evaluate LPIPS distance.

        Args:
            rendered: Tensor of shape (3, H, W) or (B, 3, H, W) in [0, 1].
            target:   Tensor of shape (3, H, W) or (B, 3, H, W) in [0, 1].

        Returns:
            Scalar loss value.
        """
        # Ensure inputs have batch dimension (B, 3, H, W)
        if rendered.dim() == 3:
            rendered = rendered.unsqueeze(0)
        if target.dim() == 3:
            target = target.unsqueeze(0)

        # Scale from [0, 1] to [-1, 1] as required by LPIPS
        rendered_scaled = rendered * 2.0 - 1.0
        target_scaled = target * 2.0 - 1.0

        # Compute perceptual distance
        loss = self.loss_fn(rendered_scaled, target_scaled)
        return loss.mean()


def compute_topology_loss(scene: "Scene") -> torch.Tensor:
    """
    Compute a penalty for self-intersecting and looping curves based on 
    turning sign transitions in the control polygon (Task 2.2).
    Coordinates are normalized to [0, 1] relative to the canvas dimensions
    to make the turning sign penalty scale-invariant and prevent it from dominating.
    """
    loss = torch.tensor(0.0, device=scene.background_raw.device)
    H, W = scene.canvas_size
    scale = torch.tensor([W, H], dtype=torch.float32, device=scene.background_raw.device)
    count = 0
    for curve in scene.curves:
        if getattr(curve, "is_dormant", False):
            continue
        if getattr(curve, "is_closed", False):
            continue  # filled shapes have no single control polygon
        cp = curve.control_points  # (4, 2)
        # Normalize to prevent the cross product penalty scaling with coordinates^4
        cp_norm = cp / scale
        v1 = cp_norm[1] - cp_norm[0]
        v2 = cp_norm[2] - cp_norm[1]
        v3 = cp_norm[3] - cp_norm[2]
        
        c1 = v1[0] * v2[1] - v1[1] * v2[0]
        c2 = v2[0] * v3[1] - v2[1] * v3[0]
        
        # We penalize opposite signs of c1 and c2.
        # Opposite signs mean c1 * c2 < 0.
        # Loss is ReLU(-c1 * c2)
        penalty = torch.relu(-c1 * c2)
        loss = loss + penalty
        count += 1
    
    if count > 0:
        return loss / count
    return loss


def compute_width_loss(scene: "Scene") -> torch.Tensor:
    """Compute L2 penalty on curve widths to prevent stroke bloating."""
    loss = torch.tensor(0.0, device=scene.background_raw.device)
    count = 0
    for curve in scene.curves:
        if getattr(curve, "is_dormant", False):
            continue
        if getattr(curve, "is_closed", False):
            continue  # filled shapes have no stroke width
        loss = loss + (curve.width ** 2)
        count += 1
    if count > 0:
        return loss / count
    return loss



def compute_edge_loss(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Sobel-gradient L1 loss — rewards sharp, correctly-placed edges.

    The painterly Gaussian splats blur boundaries; matching the target's image
    gradients (not just its pixels) directly penalizes that softness, pushing
    the optimizer toward crisp, tracer-like edges. Pure-torch and differentiable
    (a fixed Sobel conv), so gradients flow back to control points and fill `k`.

    Args:
        rendered: (3, H, W) or (B, 3, H, W) in [0, 1].
        target:   same shape as rendered.

    Returns:
        scalar L1 distance between Sobel gradient magnitudes.
    """
    if rendered.dim() == 3:
        rendered = rendered.unsqueeze(0)
        target = target.unsqueeze(0)

    device, dtype = rendered.device, rendered.dtype
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                      device=device, dtype=dtype)
    ky = kx.t()
    # One kernel per channel (depthwise via groups=C).
    C = rendered.shape[1]
    wx = kx.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    wy = ky.view(1, 1, 3, 3).repeat(C, 1, 1, 1)

    def grad_mag(img: torch.Tensor) -> torch.Tensor:
        img = torch.nn.functional.pad(img, (1, 1, 1, 1), mode="reflect")
        gx = torch.nn.functional.conv2d(img, wx, groups=C)
        gy = torch.nn.functional.conv2d(img, wy, groups=C)
        return torch.sqrt(gx * gx + gy * gy + 1e-8)

    return torch.nn.functional.l1_loss(grad_mag(rendered), grad_mag(target))
