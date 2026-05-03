"""Physiological signal encoder for multimodal emotion recognition.

This module processes 6-channel Empatica E4 signals:
BVP, EDA, TEMP, ACC_X, ACC_Y, ACC_Z.

Architecture: Multi-Scale CNN -> BiLSTM -> Transformer -> AttentionPool
- Multi-Scale CNN captures patterns at different temporal resolutions
- BiLSTM captures local sequential dependencies
- Transformer captures long-range global relationships
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .attention_module import TemporalAttentionPool, PositionalEncoding


class ChannelAttention(nn.Module):
    """Lightweight channel-wise attention over physiological modalities."""

    def __init__(self, channels: int = 6) -> None:
        super().__init__()
        self.fc1 = nn.Linear(channels, channels)
        self.fc2 = nn.Linear(channels, channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel gates.

        Args:
            x: Tensor of shape (B, T_s, 6).

        Returns:
            Tensor of shape (B, T_s, 6).
        """
        pooled = x.mean(dim=1)  # (B, T_s, 6) -> (B, 6)
        gates = self.fc1(pooled)  # (B, 6) -> (B, 6)
        gates = self.relu(gates)  # (B, 6) -> (B, 6)
        gates = self.fc2(gates)  # (B, 6) -> (B, 6)
        gates = self.sigmoid(gates)  # (B, 6) -> (B, 6)
        out = x * gates.unsqueeze(1)  # (B, T_s, 6) * (B, 1, 6) -> (B, T_s, 6)
        return out


class MultiScaleSignalCNN(nn.Module):
    """Multi-scale inception-style Conv1D encoder.

    WHY multi-scale: Different biosignal patterns operate at different
    temporal resolutions. BVP peaks are fast (small kernel), EDA trends
    are slow (large kernel). Parallel branches capture all scales
    simultaneously, unlike sequential convolutions.
    """

    def __init__(self, in_channels: int = 6) -> None:
        super().__init__()
        # Branch 1: Fine-grained patterns (heartbeat peaks)
        self.branch_fine = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )
        # Branch 2: Medium patterns (breathing, movement bursts)
        self.branch_medium = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )
        # Branch 3: Coarse patterns (EDA trends, temperature drift)
        self.branch_coarse = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=15, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )
        # Merge: 48 channels -> 64 with downsampling
        self.merge = nn.Sequential(
            nn.Conv1d(48, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode signal windows at multiple temporal scales.

        Args:
            x: Tensor of shape (B, T_s, 6).

        Returns:
            Tensor of shape (B, T_s//4, 64).
        """
        conv_in = x.transpose(1, 2)  # (B, T_s, 6) -> (B, 6, T_s)
        b_fine = self.branch_fine(conv_in)      # (B, 16, T_s)
        b_medium = self.branch_medium(conv_in)  # (B, 16, T_s)
        b_coarse = self.branch_coarse(conv_in)  # (B, 16, T_s)
        # WHY concat: preserves information from all scales rather than
        # forcing a single receptive field like sequential convolutions.
        multi = torch.cat([b_fine, b_medium, b_coarse], dim=1)  # (B, 48, T_s)
        merged = self.merge(multi)  # (B, 64, T_s//4)
        out = merged.transpose(1, 2)  # (B, T_s//4, 64)
        return out


# Keep original for backward compatibility with saved checkpoints
class SignalCNNBlocks(nn.Module):
    """Two-block Conv1D encoder over channel-attended signals."""

    def __init__(self, in_channels: int = 6) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode signal windows.

        Args:
            x: Tensor of shape (B, T_s, 6).

        Returns:
            Tensor of shape (B, T_s//4, 64).
        """
        conv_in = x.transpose(1, 2)  # (B, T_s, 6) -> (B, 6, T_s)
        block1_out = self.block1(conv_in)  # (B, 6, T_s) -> (B, 32, T_s//2)
        block2_out = self.block2(block1_out)  # (B, 32, T_s//2) -> (B, 64, T_s//4)
        out = block2_out.transpose(1, 2)  # (B, 64, T_s//4) -> (B, T_s//4, 64)
        return out


class SignalModule(nn.Module):
    """Full signal pipeline: ChannelAttention + MultiScaleCNN + BiLSTM + Transformer + AttentionPool.

    WHY hybrid BiLSTM-Transformer: BiLSTM captures ordered, local temporal
    dynamics (e.g., rising EDA response) while the Transformer captures
    global, non-causal relationships across the entire window. This
    combination outperforms either approach alone on physiological data.
    """

    def __init__(self, channels: int = 6, use_multiscale: bool = True) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels=channels)

        # WHY multi-scale: captures fast BVP peaks AND slow EDA trends
        if use_multiscale:
            self.cnn_blocks = MultiScaleSignalCNN(in_channels=channels)
        else:
            self.cnn_blocks = SignalCNNBlocks(in_channels=channels)

        # Project CNN output to model dimension
        self.sig_proj = nn.Linear(64, 256)

        # WHY BiLSTM: captures local sequential patterns that Transformer
        # self-attention may miss due to position-invariant dot products.
        self.bilstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        self.lstm_norm = nn.LayerNorm(256)

        # WHY Transformer after BiLSTM: the Transformer operates on
        # BiLSTM-enriched features that already encode local context,
        # enabling more effective global attention.
        self.pos_encoder = PositionalEncoding(d_model=256)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=512, batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.temporal_attention = TemporalAttentionPool(input_dim=256)

    def forward(self, signal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode physiological windows.

        Args:
            signal: Tensor with shape (B, T_s, 6).

        Returns:
            sig_emb: Tensor with shape (B, 256).
            seq_out: Tensor with shape (B, T_s//4, 256).
            attn_weights: Tensor with shape (B, T_s//4).
        """
        attended = self.channel_attention(signal)      # (B, T_s, 6)
        cnn_out = self.cnn_blocks(attended)            # (B, T_s//4, 64)
        proj_out = self.sig_proj(cnn_out)              # (B, T_s//4, 256)

        # BiLSTM for local sequential encoding
        lstm_out, _ = self.bilstm(proj_out)            # (B, T_s//4, 256)
        lstm_out = self.lstm_norm(lstm_out + proj_out) # residual + norm

        # Transformer for global attention
        pos_encoded = self.pos_encoder(lstm_out)
        transformer_out = self.temporal_transformer(pos_encoded)  # (B, T_s//4, 256)

        sig_emb, attn_weights = self.temporal_attention(transformer_out)
        return sig_emb, transformer_out, attn_weights

    def freeze_cnn_blocks(self) -> None:
        """Freeze only convolutional blocks for Stage 3 fine-tuning."""
        for param in self.cnn_blocks.parameters():
            param.requires_grad = False

    def set_stage3_policy(self) -> None:
        """Stage 3 policy: frozen CNN, trainable recurrent-attention head."""
        self.freeze_cnn_blocks()
        for param in self.channel_attention.parameters():
            param.requires_grad = True
        for param in self.sig_proj.parameters():
            param.requires_grad = True
        for param in self.bilstm.parameters():
            param.requires_grad = True
        for param in self.lstm_norm.parameters():
            param.requires_grad = True
        for param in self.temporal_transformer.parameters():
            param.requires_grad = True
        for param in self.temporal_attention.parameters():
            param.requires_grad = True
