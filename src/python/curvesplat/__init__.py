"""
CurveSplat — Differentiable image-to-SVG vectorizer.

Phase 1: Pure PyTorch Gaussian-splatting renderer for cubic Bezier curves.
Phase 2: SDF-based curve initialization from raster images.
No CUDA kernels, no tiling — validates the math and gradient flow.
"""

from curvesplat.scene import Curve, Scene
from curvesplat.renderer import CurveSplatRenderer
from curvesplat.sdf_init import initialize_curves_from_image
from curvesplat.losses import compute_topology_loss

try:
    from curvesplat.losses import LPIPSLoss
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False

__version__ = "0.2.0"
__all__ = ["Curve", "Scene", "CurveSplatRenderer", "initialize_curves_from_image", "compute_topology_loss"]
if HAS_LPIPS:
    __all__.append("LPIPSLoss")
