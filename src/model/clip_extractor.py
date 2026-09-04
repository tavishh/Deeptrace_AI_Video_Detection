"""
clip_extractor.py - Frozen CLIP ViT-B/32 feature extractor

Owner: Zihao/Tavish
Week:  1 (verify loads) / Week 3 (used in training)

Loads CLIP ViT-B/32 via OpenCLIP and extracts 512-dim feature vectors
from preprocessed frames. The backbone is FROZEN throughout training -
only the MLP head is updated.

Architectural choice: freezing CLIP isolates the benchmark contribution
and ensures performance differences reflect dataset generalization rather
than backbone capacity.
"""

import torch
import torch.nn as nn
import open_clip
from pathlib import Path


CLIP_MODEL_NAME  = "ViT-B-32"
CLIP_PRETRAINED  = "openai"
FEATURE_DIM      = 512


class CLIPExtractor(nn.Module):
    """
    Frozen CLIP ViT-B/32 image encoder.
    Outputs 512-dimensional L2-normalized feature vectors.
    """
 
    def __init__(self, model_name: str = CLIP_MODEL_NAME,
                 pretrained: str = CLIP_PRETRAINED):
        super().__init__()
 
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        # Keep only the visual encoder
        self.visual = model.visual
        self.preprocess = preprocess
 
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Preprocessed image tensor (B, 3, 224, 224)
 
        Returns:
            L2-normalized feature tensor (B, 512)
        """
        features = self.visual(x)
        # L2 normalize
        features = features / features.norm(dim=-1, keepdim=True)
        return features

def load_extractor(device: str = "cuda") -> CLIPExtractor:
    """Load the CLIP extractor and move to device."""
    extractor = CLIPExtractor()
    extractor = extractor.to(device)
    extractor.eval()
    return extractor