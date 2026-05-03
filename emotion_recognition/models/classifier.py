"""Final classifier head for 7-class emotion prediction.

The head is intentionally compact and strongly regularized to reduce overfitting
on participant-limited data.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EmotionClassifier(nn.Module):
    """MLP classifier producing 7-way emotion and 3-way valence log-probabilities."""

    def __init__(self, input_dim: int = 384, num_classes: int = 7) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            # WHY dropout=0.4: strongest regularizer requested for 58 participants.
            nn.Dropout(p=0.4),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )
        self.valence_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(64, 3) # 3-class Valence
        )
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """Predict class log-probabilities.

        Args:
            fused: Tensor of shape (B, 384).

        Returns:
            log_probs: Tensor of shape (B, 7) containing emotion log-probabilities.
            val_log_probs: Tensor of shape (B, 3) containing valence log-probabilities.
        """
        logits = self.net(fused)  # (B, 384) -> (B, 7)
        val_logits = self.valence_net(fused) # (B, 384) -> (B, 3)
        log_probs = self.log_softmax(logits)  # (B, 7) -> (B, 7)
        val_log_probs = self.log_softmax(val_logits) # (B, 3) -> (B, 3)
        return log_probs, val_log_probs
