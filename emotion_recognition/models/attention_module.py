"""Attention modules used by face, signal, and fusion pipelines.

Includes:
- PositionalEncoding for Transformer layers
- TemporalAttentionPool for sequence compression
- CrossModalAttention for bidirectional video-signal refinement (pooled)
- SequenceCrossModalAttention for temporal cross-modal attention (sequence-level)
"""

from __future__ import annotations

from math import sqrt
from typing import Tuple

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer sequence layers."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [B, T, D]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalAttentionPool(nn.Module):
    """Temporal attention pooling over sequence features."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.score_layer = nn.Linear(input_dim, 1)

    def forward(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pool sequence features with learned temporal salience.

        Args:
            seq: Tensor of shape (B, T, D).

        Returns:
            pooled: Tensor of shape (B, D).
            weights: Tensor of shape (B, T).
        """
        scores = self.score_layer(seq).squeeze(-1)  # (B, T, D) -> (B, T, 1) -> (B, T)
        weights = torch.softmax(scores, dim=1)  # (B, T) -> (B, T)
        pooled = (weights.unsqueeze(-1) * seq).sum(dim=1)  # (B, T, 1)*(B, T, D) -> (B, D)
        return pooled, weights


class CrossModalAttention(nn.Module):
    """Bidirectional cross-modal attention between video and signal embeddings."""

    def __init__(self, vid_dim: int = 128, sig_dim: int = 256, attn_dim: int = 128) -> None:
        super().__init__()
        self.scale = sqrt(attn_dim)

        self.vid_q = nn.Linear(vid_dim, attn_dim)
        self.vid_k = nn.Linear(vid_dim, attn_dim)
        self.vid_v = nn.Linear(vid_dim, attn_dim)

        self.sig_q = nn.Linear(sig_dim, attn_dim)
        self.sig_k = nn.Linear(sig_dim, attn_dim)
        self.sig_v = nn.Linear(sig_dim, attn_dim)

        # WHY projection: signal embedding is 256-d, but cross-attention operates
        # in a shared 128-d latent space for symmetric interactions.
        self.sig_residual_proj = nn.Linear(sig_dim, attn_dim)
        self.sig_back_proj = nn.Linear(attn_dim, sig_dim)

    def forward(self, vid_emb: torch.Tensor, sig_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run bidirectional cross-modal attention.

        Args:
            vid_emb: Tensor of shape (B, 128).
            sig_emb: Tensor of shape (B, 256).

        Returns:
            enhanced_vid: Tensor of shape (B, 128).
            enhanced_sig: Tensor of shape (B, 256).
        """
        vid_q = self.vid_q(vid_emb).unsqueeze(1)  # (B, 128) -> (B, 1, 128)
        sig_k = self.sig_k(sig_emb).unsqueeze(1)  # (B, 256) -> (B, 1, 128)
        sig_v = self.sig_v(sig_emb).unsqueeze(1)  # (B, 256) -> (B, 1, 128)

        vid_k = self.vid_k(vid_emb).unsqueeze(1)  # (B, 128) -> (B, 1, 128)
        vid_v = self.vid_v(vid_emb).unsqueeze(1)  # (B, 128) -> (B, 1, 128)
        sig_q = self.sig_q(sig_emb).unsqueeze(1)  # (B, 256) -> (B, 1, 128)

        # WHY video->signal attention: visual branch can query physiological context
        # to reduce ambiguity when facial expressions are subtle.
        logits_vid = torch.matmul(vid_q, sig_k.transpose(1, 2)) / self.scale  # (B, 1, 128)x(B, 128, 1) -> (B, 1, 1)
        weights_vid = torch.softmax(logits_vid, dim=-1)  # (B, 1, 1) -> (B, 1, 1)
        attn_vid = torch.matmul(weights_vid, sig_v)  # (B, 1, 1)x(B, 1, 128) -> (B, 1, 128)
        enhanced_vid = vid_emb + attn_vid.squeeze(1)  # (B, 128) + (B, 128) -> (B, 128)

        # WHY signal->video attention: physiology can focus on expression-relevant
        # visual content, improving robustness to sensor noise.
        logits_sig = torch.matmul(sig_q, vid_k.transpose(1, 2)) / self.scale  # (B, 1, 128)x(B, 128, 1) -> (B, 1, 1)
        weights_sig = torch.softmax(logits_sig, dim=-1)  # (B, 1, 1) -> (B, 1, 1)
        attn_sig = torch.matmul(weights_sig, vid_v)  # (B, 1, 1)x(B, 1, 128) -> (B, 1, 128)

        sig_base = self.sig_residual_proj(sig_emb)  # (B, 256) -> (B, 128)
        enhanced_sig_128 = sig_base + attn_sig.squeeze(1)  # (B, 128) + (B, 128) -> (B, 128)
        enhanced_sig = self.sig_back_proj(enhanced_sig_128)  # (B, 128) -> (B, 256)

        return enhanced_vid, enhanced_sig


class SequenceCrossModalAttention(nn.Module):
    """Temporal sequence-level cross-modal attention.

    WHY sequence-level: Unlike pooled cross-attention which operates on
    single summary vectors, this module lets each time step in one modality
    attend to ALL time steps in the other. This captures fine-grained
    temporal correspondences (e.g., a facial micro-expression at t=3
    correlating with an EDA spike at t=5).
    """

    def __init__(self, vid_dim: int = 128, sig_dim: int = 256, attn_dim: int = 128, num_heads: int = 4) -> None:
        super().__init__()
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        self.scale = sqrt(self.head_dim)

        # Video queries signal sequence
        self.vid_to_q = nn.Linear(vid_dim, attn_dim)
        self.sig_to_kv = nn.Linear(sig_dim, attn_dim * 2)
        self.vid_out_proj = nn.Linear(attn_dim, vid_dim)
        self.vid_norm = nn.LayerNorm(vid_dim)

        # Signal queries video sequence
        self.sig_to_q = nn.Linear(sig_dim, attn_dim)
        self.vid_to_kv = nn.Linear(vid_dim, attn_dim * 2)
        self.sig_out_proj = nn.Linear(attn_dim, sig_dim)
        self.sig_norm = nn.LayerNorm(sig_dim)

        self.dropout = nn.Dropout(0.1)

    def _multi_head_attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Multi-head attention helper.

        Args:
            q: (B, T_q, attn_dim)
            k: (B, T_k, attn_dim)
            v: (B, T_k, attn_dim)

        Returns:
            (B, T_q, attn_dim)
        """
        B, T_q, _ = q.shape
        T_k = k.shape[1]

        # Reshape to multi-head: (B, T, D) -> (B, H, T, D_h)
        q = q.view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, H, T_q, T_k)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # (B, H, T_q, D_h)

        # Merge heads
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.attn_dim)
        return out

    def forward(
        self,
        vid_seq: torch.Tensor,
        sig_seq: torch.Tensor,
        vid_emb: torch.Tensor,
        sig_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run bidirectional sequence-level cross-modal attention.

        Args:
            vid_seq: Video temporal features (B, T_v, 128).
            sig_seq: Signal temporal features (B, T_s, 256).
            vid_emb: Pooled video embedding (B, 128) - used as fallback residual.
            sig_emb: Pooled signal embedding (B, 256) - used as fallback residual.

        Returns:
            enhanced_vid: (B, 128).
            enhanced_sig: (B, 256).
        """
        # Video queries signal: each video frame attends to all signal steps
        q_vid = self.vid_to_q(vid_seq)                    # (B, T_v, attn_dim)
        kv_sig = self.sig_to_kv(sig_seq)                  # (B, T_s, attn_dim*2)
        k_sig, v_sig = kv_sig.chunk(2, dim=-1)            # each (B, T_s, attn_dim)
        vid_cross = self._multi_head_attn(q_vid, k_sig, v_sig)  # (B, T_v, attn_dim)
        vid_cross = self.vid_out_proj(vid_cross)           # (B, T_v, vid_dim)
        vid_enhanced_seq = self.vid_norm(vid_seq + vid_cross)    # residual
        enhanced_vid = vid_enhanced_seq.mean(dim=1)        # (B, vid_dim) global pool

        # Signal queries video: each signal step attends to all video frames
        q_sig = self.sig_to_q(sig_seq)                    # (B, T_s, attn_dim)
        kv_vid = self.vid_to_kv(vid_seq)                  # (B, T_v, attn_dim*2)
        k_vid, v_vid = kv_vid.chunk(2, dim=-1)            # each (B, T_v, attn_dim)
        sig_cross = self._multi_head_attn(q_sig, k_vid, v_vid)  # (B, T_s, attn_dim)
        sig_cross = self.sig_out_proj(sig_cross)           # (B, T_s, sig_dim)
        sig_enhanced_seq = self.sig_norm(sig_seq + sig_cross)    # residual
        enhanced_sig = sig_enhanced_seq.mean(dim=1)        # (B, sig_dim) global pool

        return enhanced_vid, enhanced_sig
