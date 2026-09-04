"""
mlp_head.py - 2-layer MLP classification head

Owner: Zihao/Tavish
Week:  3

Trainable classification head on top of frozen CLIP features.
Architecture: 512 -> 128 -> 2 with ReLU, BatchNorm, and Dropout.

Only this module is updated during training - CLIP backbone is frozen.

Clip-level aggregation: per-frame probabilities are averaged across
all sampled frames (mean aggregation). Majority vote available as
secondary comparison.
"""

import torch
import torch.nn as nn


class MLPHead(nn.Module):
    """
    2-layer MLP: 512 -> 128 -> 2 (real / fake)
    """
 
    def __init__(self, input_dim: int = 512, hidden_dim: int = 128,
                 num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
 
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: CLIP feature tensor (B, 512)
 
        Returns:
            Logits (B, 2)
        """
        return self.network(x)


class DeepTrace(nn.Module):
    """
    Full DeepTrace model: CLIPExtractor (frozen) + MLPHead (trainable).
    Also handles clip-level aggregation of per-frame predictions.
    """
 
    def __init__(self, clip_extractor, mlp_head, aggregation: str = "mean"):
        super().__init__()
        self.extractor   = clip_extractor
        self.head        = mlp_head
        self.aggregation = aggregation
 
    def forward_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Forward pass over individual frames.
 
        Args:
            frames: (B, 3, 224, 224)
 
        Returns:
            Per-frame logits (B, 2)
        """
        with torch.no_grad():
            features = self.extractor(frames)
        logits = self.head(features)
        return logits
 
    def predict_clip(self, frames: torch.Tensor) -> dict:
        """
        Aggregate frame predictions into a clip-level verdict.
 
        Args:
            frames: (N_frames, 3, 224, 224) - all frames from one clip
 
        Returns:
            dict with keys: fake_prob, label, per_frame_probs
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward_frames(frames)
            probs  = torch.softmax(logits, dim=-1)  # (N_frames, 2)
            per_frame_fake_probs = probs[:, 1].cpu().numpy()
 
        if self.aggregation == "mean":
            clip_fake_prob = float(per_frame_fake_probs.mean())
        elif self.aggregation == "majority_vote":
            frame_preds    = (per_frame_fake_probs > 0.5).astype(int)
            clip_fake_prob = float(frame_preds.mean())
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
 
        return {
            "fake_prob":        clip_fake_prob,
            "real_prob":        1.0 - clip_fake_prob,
            "label":            "fake" if clip_fake_prob > 0.5 else "real",
            "per_frame_probs":  per_frame_fake_probs.tolist(),
        }