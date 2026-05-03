"""Fusion module for reliability-aware multimodal integration.

Soft gating allows sample-wise and feature-wise balancing between
physiological and visual evidence when one modality is noisy.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class SoftGatingFusion(nn.Module):
    """Per-dimension soft gating over projected modality embeddings."""

    def __init__(self, vid_dim: int = 128, sig_dim: int = 256, fused_dim: int = 384, modality_dropout: float = 0.15) -> None:
        super().__init__()
        self.modality_dropout = modality_dropout
        self.proj_vid = nn.Linear(vid_dim, fused_dim)
        self.proj_sig = nn.Linear(sig_dim, fused_dim)
        self.gate = nn.Sequential(
            nn.Linear(vid_dim + sig_dim, fused_dim),
            nn.Sigmoid(),
        )

    def forward(self, enhanced_vid: torch.Tensor, enhanced_sig: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse modality embeddings with soft reliability gating.

        Args:
            enhanced_vid: Tensor of shape (B, 128).
            enhanced_sig: Tensor of shape (B, 256).

        Returns:
            fused: Tensor of shape (B, 384).
            g: Tensor of shape (B, 384).
        """
        combined = torch.cat([enhanced_vid, enhanced_sig], dim=1)  # (B, 128)+(B, 256) -> (B, 384)
        g = self.gate(combined)  # (B, 384) -> (B, 384)

        proj_vid = self.proj_vid(enhanced_vid)  # (B, 128) -> (B, 384)
        proj_sig = self.proj_sig(enhanced_sig)  # (B, 256) -> (B, 384)

        # Modality Dropout (Masked Fusion)
        if self.training and self.modality_dropout > 0:
            mask_vid = (torch.rand(enhanced_vid.shape[0], 1, device=enhanced_vid.device) > self.modality_dropout).float()
            mask_sig = (torch.rand(enhanced_sig.shape[0], 1, device=enhanced_sig.device) > self.modality_dropout).float()
            proj_vid = proj_vid * mask_vid
            proj_sig = proj_sig * mask_sig

        # WHY gated interpolation: suppresses unreliable modality features instead
        # of hard-discarding an entire stream.
        fused = g * proj_sig + (1.0 - g) * proj_vid  # (B, 384) -> (B, 384)
        return fused, g
