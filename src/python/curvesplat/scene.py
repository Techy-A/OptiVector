"""
Core data structures for CurveSplat: Curve and Scene.

Matches SPEC.md §3 exactly:
  - Curve: 13 learnable parameters (8 control-point coords + 3 color + 1 width + 1 opacity)
  - Scene: list of Curves + learnable background color + canvas size

Activations are applied at render time, not stored:
  - color:   sigmoid  → [0, 1] RGB
  - opacity: sigmoid  → [0, 1]
  - width:   softplus → positive stroke width
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Curve(nn.Module):
    """
    A single cubic Bezier curve with stroke parameters.

    Stores raw (pre-activation) parameters as nn.Parameters so gradients
    flow naturally through sigmoid/softplus at render time.

    Fields per SPEC.md §3:
        control_points: float[4][2]  — P0..P3 cubic Bezier
        color:          float[3]     — RGB, sigmoid-activated at render time
        width:          float        — softplus-activated stroke width
        opacity:        float        — sigmoid-activated, [0,1]
        is_closed:      bool         — filled shape vs stroke (not used in Phase 1)
        z_index:        int          — layer ordering (not used in Phase 1)
    """

    def __init__(
        self,
        control_points: torch.Tensor,
        color: torch.Tensor,
        width: torch.Tensor,
        opacity: torch.Tensor,
        is_closed: bool = False,
        z_index: int = 0,
        max_width: float = 35.0,
    ) -> None:
        """
        Args:
            control_points: Shape (4, 2), raw control point positions.
            color: Shape (3,), raw pre-sigmoid RGB values.
            width: Scalar, raw pre-softplus width value.
            opacity: Scalar, raw pre-sigmoid opacity value.
            is_closed: Whether the curve forms a filled shape (Phase 3+).
            z_index: Layer ordering for compositing (Phase 3+).
            max_width: Maximum allowed stroke width after softplus activation.
        """
        super().__init__()

        # Learnable parameters (raw, pre-activation)
        self.control_points = nn.Parameter(control_points.clone().detach().float())
        self.color_raw = nn.Parameter(color.clone().detach().float())
        self.width_raw = nn.Parameter(width.clone().detach().float())
        self.opacity_raw = nn.Parameter(opacity.clone().detach().float())

        # Non-learnable attributes
        self.is_closed = is_closed
        self.z_index = z_index
        self.max_width = max_width

    @property
    def color(self) -> torch.Tensor:
        """Activated color: sigmoid → [0, 1] RGB. Shape (3,)."""
        return torch.sigmoid(self.color_raw)

    @property
    def width(self) -> torch.Tensor:
        """Activated width: softplus → positive. Scalar."""
        return F.softplus(self.width_raw).clamp(min=0.5, max=self.max_width)

    @property
    def opacity(self) -> torch.Tensor:
        """Activated opacity: sigmoid → [0, 1]. Scalar."""
        return torch.sigmoid(self.opacity_raw)

    @staticmethod
    def create(
        control_points: torch.Tensor | None = None,
        canvas_size: Tuple[int, int] = (256, 256),
        is_closed: bool = False,
        z_index: int = 0,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "Curve":
        """
        Factory: create a Curve with sensible defaults.

        If control_points is None, generates a small "C"-shaped curve
        centered on the canvas — a reasonable starting point for optimization.
        """
        H, W = canvas_size
        cx, cy = W / 2.0, H / 2.0
        radius = min(H, W) * 0.2

        if control_points is None:
            # Default: a small arc near center (quarter-circle-ish)
            control_points = torch.tensor(
                [
                    [cx - radius, cy - radius],  # P0: top-left
                    [cx + radius, cy - radius],  # P1: top-right
                    [cx + radius, cy + radius],  # P2: bottom-right
                    [cx - radius, cy + radius],  # P3: bottom-left
                ],
                dtype=dtype,
                device=device,
            )

        # Pre-sigmoid color: 0.0 → sigmoid(0) = 0.5 (mid-gray)
        color = torch.zeros(3, dtype=dtype, device=device)
        # Pre-softplus width: ~2.0 → softplus(2.0) ≈ 2.13 pixels
        width = torch.tensor(2.0, dtype=dtype, device=device)
        # Pre-sigmoid opacity: 2.0 → sigmoid(2.0) ≈ 0.88
        opacity = torch.tensor(2.0, dtype=dtype, device=device)

        return Curve(
            control_points=control_points,
            color=color,
            width=width,
            opacity=opacity,
            is_closed=is_closed,
            z_index=z_index,
        )


class FilledShape(nn.Module):
    """
    A closed loop of cubic Bezier segments rendered as a CRISP flat-filled region.

    Complements `Curve` (soft painterly stroke). A filled region's boundary needs
    many cubic segments, so this stores a (S, 4, 2) parameter — unlike Curve's
    single (4, 2). It is dispatched by the filled renderer (fill_renderer.py) and
    composited in fixed z-order (no shuffle) so opaque base layers paint correctly.

    Fields:
        segments:   (S, 4, 2) raw control points of the S cubic boundary segments
        color:      (3,) RGB, sigmoid-activated at render time
        opacity:    scalar, sigmoid-activated
        is_closed:  always True (marks this as a fill primitive for dispatch)
        z_index:    layer ordering (lower = painted first / underneath)
    """

    def __init__(
        self,
        segments: torch.Tensor,
        color: torch.Tensor,
        opacity: torch.Tensor,
        z_index: int = 0,
    ) -> None:
        super().__init__()
        self.segments = nn.Parameter(segments.clone().detach().float())  # (S, 4, 2)
        self.color_raw = nn.Parameter(color.clone().detach().float())     # (3,)
        self.opacity_raw = nn.Parameter(opacity.clone().detach().float())  # scalar
        self.is_closed = True
        self.z_index = z_index

    @property
    def color(self) -> torch.Tensor:
        """Activated fill color: sigmoid → [0, 1] RGB. Shape (3,)."""
        return torch.sigmoid(self.color_raw)

    @property
    def opacity(self) -> torch.Tensor:
        """Activated fill opacity: sigmoid → [0, 1]. Scalar."""
        return torch.sigmoid(self.opacity_raw)

    @staticmethod
    def from_region(region: dict, device: torch.device | str = "cpu") -> "FilledShape":
        """
        Build a FilledShape from one dict produced by
        sdf_init.initialize_filled_regions: keys 'segments' (S,4,2 in [0,1]?),
        'color' (3,) in [0,1], 'z_index' (int).
        """
        seg = torch.as_tensor(region["segments"], dtype=torch.float32, device=device)
        col = torch.as_tensor(region["color"], dtype=torch.float32, device=device).clamp(0.01, 0.99)
        color_raw = torch.log(col / (1.0 - col))                  # inverse sigmoid
        opacity_raw = torch.tensor(4.0, dtype=torch.float32, device=device)  # ~0.98 opaque
        return FilledShape(seg, color_raw, opacity_raw, z_index=int(region.get("z_index", 0)))


class Scene(nn.Module):
    """
    A scene containing curves and a learnable background, per SPEC.md §3.

    Fields:
        curves:      list[Curve]  — ModuleList of Curve nn.Modules
        background:  float[3]    — learnable background RGB (sigmoid-activated)
        canvas_size: (H, W)      — canvas dimensions
    """

    def __init__(
        self,
        curves: list[Curve],
        canvas_size: Tuple[int, int] = (256, 256),
        background: torch.Tensor | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        """
        Args:
            curves: List of Curve modules.
            canvas_size: (H, W) canvas dimensions.
            background: Shape (3,), raw pre-sigmoid background color.
                        Defaults to a very negative value → sigmoid ≈ 0 (black).
            device: Torch device.
        """
        super().__init__()
        self.curves = nn.ModuleList(curves)
        self.canvas_size = canvas_size

        if background is None:
            # Default: black background (sigmoid(-5) ≈ 0.007 ≈ black)
            background = torch.full((3,), -5.0, device=device)
        self.background_raw = nn.Parameter(background.clone().detach().float())

    @property
    def background(self) -> torch.Tensor:
        """Activated background: sigmoid → [0, 1] RGB. Shape (3,)."""
        return torch.sigmoid(self.background_raw)
