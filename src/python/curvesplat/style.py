"""
Style-Aware Vectorization losses and feature extraction for CurveSplat.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

# Try to import torchvision for pre-trained VGG model
try:
    from torchvision.models import vgg16, VGG16_Weights
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


class VGGStyleExtractor(nn.Module):
    """
    Slices a pre-trained VGG16 network to extract feature maps for Gram matrices.
    Using layers conv1_2, conv2_2, conv3_3, conv4_3.
    """

    def __init__(self, device: str = "cpu") -> None:
        super().__init__()
        self.device = device
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        
        if HAS_TORCHVISION:
            try:
                # Load pre-trained VGG16 weights
                vgg = vgg16(weights=VGG16_Weights.DEFAULT).features.to(device).eval()
                for param in vgg.parameters():
                    param.requires_grad = False
                
                # Slices based on standard style transfer layer indices
                self.slice1 = nn.Sequential(*vgg[:4])   # conv1_2 ReLU
                self.slice2 = nn.Sequential(*vgg[4:9])  # conv2_2 ReLU
                self.slice3 = nn.Sequential(*vgg[9:16]) # conv3_3 ReLU
                self.slice4 = nn.Sequential(*vgg[16:23])# conv4_3 ReLU
                self.has_vgg = True
            except Exception as e:
                print(f"  [Style Module Warning] Failed to load pre-trained VGG16: {e}. Style texture Gram loss will be disabled.")
                self.has_vgg = False
        else:
            print("  [Style Module Warning] torchvision is not available. Style texture Gram loss will be disabled.")
            self.has_vgg = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract features from intermediate VGG layers.
        x: Input image tensor of shape (B, 3, H, W) normalized to [0, 1].
        """
        if not self.has_vgg:
            return []
            
        # VGG expects normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x_norm = (x - mean) / std
        
        h1 = self.slice1(x_norm)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        return [h1, h2, h3, h4]


def gram_matrix(y: torch.Tensor) -> torch.Tensor:
    """Compute the Gram matrix of a batch of activation maps."""
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, h * w)
    features_t = features.transpose(1, 2)
    # Batch matrix multiplication: (B, ch, HW) x (B, HW, ch) -> (B, ch, ch)
    # Scale by channel dimension * spatial dimension to normalize Gram matrix magnitude
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def extract_dominant_palette(image_np: np.ndarray, k: int = 8) -> np.ndarray:
    """
    Extract k dominant colors from the image in [0, 1] RGB space using K-Means++.
    image_np: (H, W, 3) float32 in [0, 1].
    """
    H, W, C = image_np.shape
    pixels = image_np.reshape(-1, 3).astype(np.float32)
    
    # Subsample pixels to speed up processing
    rng = np.random.default_rng(42)
    num_samples = min(len(pixels), 10000)
    subsample_indices = rng.choice(len(pixels), size=num_samples, replace=False)
    sub_pixels = pixels[subsample_indices]
    
    # K-Means++ initialization
    idx0 = rng.integers(0, len(sub_pixels))
    centroids = [sub_pixels[idx0]]
    for _ in range(k - 1):
        dists = np.min(
            np.array([np.sum((sub_pixels - c) ** 2, axis=1) for c in centroids]),
            axis=0,
        )
        probs = dists / (dists.sum() + 1e-12)
        chosen = rng.choice(len(sub_pixels), p=probs)
        centroids.append(sub_pixels[chosen])
    centroids = np.array(centroids)
    
    # K-Means iterations
    for _ in range(15):
        # Compute squared Euclidean distance
        dists = np.sum(
            (sub_pixels[:, None, :] - centroids[None, :, :]) ** 2, axis=2
        )  # (N, k)
        labels = dists.argmin(axis=1)
        
        # Update centroids
        for j in range(k):
            mask = (labels == j)
            if mask.sum() > 0:
                centroids[j] = sub_pixels[mask].mean(axis=0)
                
    return centroids.clip(0, 1)


def analyze_stroke_widths(image_np: np.ndarray) -> Tuple[float, float]:
    """
    Estimate the mean and standard deviation of stroke widths in the style image.
    Uses Sobel gradients and distance transform to measure stroke thickness.
    """
    from scipy.ndimage import distance_transform_edt, sobel
    
    # Convert to grayscale
    if image_np.ndim == 3:
        gray = 0.299 * image_np[:, :, 0] + 0.587 * image_np[:, :, 1] + 0.114 * image_np[:, :, 2]
    else:
        gray = image_np
        
    # Simple Sobel filters for edge detection
    sx = sobel(gray, axis=1)
    sy = sobel(gray, axis=0)
    grad = np.sqrt(sx**2 + sy**2)
    
    # Threshold gradient to identify boundaries of style stroke/texture regions
    grad_max = grad.max()
    thresh = 0.15 * grad_max if grad_max > 1e-4 else 0.1
    edges = grad > thresh
    
    # Distance transform from boundaries (find center-lines of strokes)
    dt = distance_transform_edt(~edges)
    
    # Widths are double the distance to the nearest boundary/edge
    valid_widths = dt[dt > 0.8] * 2.0
    
    # Filter out extremely large background components (cap at 45.0)
    valid_widths = valid_widths[valid_widths < 45.0]
    
    if len(valid_widths) > 50:
        return float(np.mean(valid_widths)), float(np.std(valid_widths) + 1e-3)
        
    # Defaults: moderate/thin strokes
    return 6.0, 3.0


def compute_palette_loss(scene: nn.Module, style_palette: torch.Tensor) -> torch.Tensor:
    """
    Penalizes curves whose activated colors deviate from the style palette.
    style_palette: (K, 3) tensor of dominant style colors.
    """
    loss = torch.tensor(0.0, device=style_palette.device)
    count = 0
    for curve in scene.curves:
        if getattr(curve, "is_dormant", False):
            continue
        color = torch.sigmoid(curve.color_raw)  # (3,)
        
        # Distance to each color in style_palette
        diffs = color.unsqueeze(0) - style_palette  # (K, 3)
        sq_dists = torch.sum(diffs ** 2, dim=1)  # (K,)
        
        loss = loss + sq_dists.min()
        count += 1
        
    if count > 0:
        return loss / count
    return loss


def compute_stroke_width_loss(
    scene: nn.Module, target_mean: float, target_std: float
) -> torch.Tensor:
    """
    Penalizes curves whose stroke widths deviate from the style reference width distribution.
    """
    loss = torch.tensor(0.0, device=scene.background_raw.device)
    count = 0
    for curve in scene.curves:
        if getattr(curve, "is_dormant", False):
            continue
            
        # Get activated width of the curve
        width = curve.width
        
        # Penalize squared normalized deviation
        deviation = (width - target_mean) / target_std
        loss = loss + (deviation ** 2)
        count += 1
        
    if count > 0:
        return loss / count
    return loss
