import os
import numpy as np
import torch
from typing import List, Tuple, Optional

try:
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    HAS_CLIPSEG = True
except ImportError:
    HAS_CLIPSEG = False

from scipy.ndimage import binary_dilation, binary_closing

class CLIPSegWrapper:
    """Wrapper around HuggingFace CLIPSeg for text-guided image segmentation."""
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.processor = None
        self.model = None
        
    def load_model(self):
        if not HAS_CLIPSEG:
            raise ImportError("transformers library is not installed. Please install transformers to use CLIPSeg.")
        
        if self.model is None:
            # Load CLIPSeg model and processor
            model_id = "CIDAS/clipseg-rd64-refined"
            self.processor = CLIPSegProcessor.from_pretrained(model_id)
            self.model = CLIPSegForImageSegmentation.from_pretrained(model_id).to(self.device)
            self.model.eval()

    def segment(self, image_np: np.ndarray, prompts: List[str], threshold: float = 0.4) -> List[np.ndarray]:
        """
        Segment image into binary masks using text prompts.
        image_np: (H, W, 3) float32 in [0, 1]
        prompts: list of strings
        Returns list of boolean masks of shape (H, W)
        """
        self.load_model()
        H, W = image_np.shape[:2]
        
        # Format image for CLIPSeg (needs uint8 in [0, 255])
        img_uint8 = (image_np * 255.0).clip(0, 255).astype(np.uint8)
        
        # Prepare inputs
        inputs = self.processor(
            text=prompts, 
            images=[img_uint8] * len(prompts), 
            padding="max_length", 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # CLIPSeg outputs raw logits. For multiple prompts, outputs.logits has shape (len(prompts), H_out, W_out)
        # We need to resize to original image dimensions.
        logits = outputs.logits
        if len(prompts) == 1 and logits.dim() == 2:
            logits = logits.unsqueeze(0)
            
        masks = []
        for i in range(len(prompts)):
            logit = logits[i].unsqueeze(0).unsqueeze(0)  # (1, 1, H_out, W_out)
            # Resize using bilinear interpolation
            resized = torch.nn.functional.interpolate(
                logit, size=(H, W), mode="bilinear", align_corners=False
            ).squeeze(0).squeeze(0)  # (H, W)
            
            # Sigmoid activation to get probabilities
            probs = torch.sigmoid(resized).cpu().numpy()
            
            # Threshold to get binary mask
            binary = probs > threshold
            
            # Post-process: apply closing then dilation to clean up details
            binary = binary_closing(binary, structure=np.ones((5, 5)))
            binary = binary_dilation(binary, structure=np.ones((3, 3)))
            
            masks.append(binary)
            
        return masks

def generate_kmeans_fallback_masks(image_np: np.ndarray, num_masks: int = 3) -> List[np.ndarray]:
    """Fallback method using basic K-Means on pixel positions and colors to generate masks."""
    H, W = image_np.shape[:2]
    # Simple color + position features
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    features = np.stack([
        image_np[:, :, 0], image_np[:, :, 1], image_np[:, :, 2],
        yy / float(H), xx / float(W)
    ], axis=-1).reshape(-1, 5).astype(np.float32)
    
    # Simple K-Means++ implementation
    rng = np.random.default_rng(42)
    indices = rng.choice(len(features), size=num_masks, replace=False)
    centroids = features[indices]
    
    for _ in range(10):
        # Compute distances
        dists = np.linalg.norm(features[:, None, :] - centroids[None, :, :], axis=-1)
        labels = dists.argmin(axis=1)
        # Update centroids
        for k in range(num_masks):
            mask = (labels == k)
            if mask.sum() > 0:
                centroids[k] = features[mask].mean(axis=0)
                
    labels = labels.reshape(H, W)
    masks = []
    for k in range(num_masks):
        masks.append(labels == k)
    return masks

def generate_semantic_masks(
    image_np: np.ndarray, 
    prompts: List[str], 
    device: str = "cpu",
    threshold: float = 0.4
) -> List[np.ndarray]:
    """
    Main entry point for generating binary semantic segmentation masks.
    Attempts to use CLIPSeg first; falls back to K-Means if model fails or imports missing.
    """
    if HAS_CLIPSEG:
        try:
            wrapper = CLIPSegWrapper(device=device)
            return wrapper.segment(image_np, prompts, threshold=threshold)
        except Exception as e:
            print(f"  [Segmentation Warning] CLIPSeg failed: {e}. Falling back to K-Means.")
            
    # Fallback
    return generate_kmeans_fallback_masks(image_np, num_masks=len(prompts))
