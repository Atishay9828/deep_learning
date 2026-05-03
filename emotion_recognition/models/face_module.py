"""Face stream module for multimodal emotion recognition.

Architecture: FaceNet -> Projection -> BiLSTM -> Transformer -> AttentionPool

Pipeline:
1) Shared FaceNet backbone over each frame -> 512-d embeddings
2) Projection to compact 128-d frame tokens
3) BiLSTM for local temporal dynamics (micro-expressions)
4) Transformer for global temporal attention
5) Attention pooling to focus on highly expressive moments
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .attention_module import TemporalAttentionPool, PositionalEncoding
from .facenet_backbone import FaceNetBackbone
from .projection_head import ProjectionHead


class FaceModule(nn.Module):
    """Video encoder with hybrid BiLSTM-Transformer temporal modeling."""

    def __init__(
        self,
        backbone: FaceNetBackbone | None = None,
        projection_head: ProjectionHead | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone if backbone is not None else FaceNetBackbone(pretrained="vggface2")
        self.projection_head = projection_head if projection_head is not None else ProjectionHead()

        # WHY BiLSTM before Transformer: micro-expressions are inherently
        # sequential (onset -> apex -> offset). BiLSTM captures this
        # ordered temporal structure that position-invariant Transformer
        # attention may overlook.
        self.bilstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        self.lstm_norm = nn.LayerNorm(128)

        # WHY Transformer after BiLSTM: operates on BiLSTM-enriched
        # features to capture global dependencies (e.g., emotional build-up
        # across non-adjacent frames).
        self.pos_encoder = PositionalEncoding(d_model=128)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256, batch_first=True)
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.temporal_attention = TemporalAttentionPool(input_dim=128)

    def forward(self, video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode a video window.

        Args:
            video: Tensor with shape (B, T_v, 3, 160, 160).

        Returns:
            frame_emb: Tensor with shape (B, T_v, 128).
            vid_emb: Tensor with shape (B, 128).
            attn_weights: Tensor with shape (B, T_v).
        """
        bsz, t_v, channels, height, width = video.shape

        frames = video.reshape(bsz * t_v, channels, height, width)  # (B*T_v, 3, 160, 160)
        face_512 = self.backbone(frames)           # (B*T_v, 512)
        proj_128 = self.projection_head(face_512)  # (B*T_v, 128)
        frame_emb = proj_128.reshape(bsz, t_v, 128)  # (B, T_v, 128)

        # BiLSTM for local sequential encoding of expression dynamics
        lstm_out, _ = self.bilstm(frame_emb)       # (B, T_v, 128)
        lstm_out = self.lstm_norm(lstm_out + frame_emb)  # residual + norm

        # Transformer for global temporal attention
        pos_encoded = self.pos_encoder(lstm_out)
        transformer_out = self.temporal_transformer(pos_encoded)  # (B, T_v, 128)

        vid_emb, attn_weights = self.temporal_attention(transformer_out)

        return frame_emb, vid_emb, attn_weights

    def set_stage3_policy(self) -> None:
        """Stage 3 policy: frozen FaceNet backbone, trainable temporal head."""
        self.backbone.set_stage3_policy()
        for param in self.projection_head.parameters():
            param.requires_grad = True
        for param in self.bilstm.parameters():
            param.requires_grad = True
        for param in self.lstm_norm.parameters():
            param.requires_grad = True
        for param in self.temporal_transformer.parameters():
            param.requires_grad = True
        for param in self.temporal_attention.parameters():
            param.requires_grad = True
