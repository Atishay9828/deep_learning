"""End-to-end multimodal emotion recognition model.

This module assembles:
- Face pipeline (FaceNet + projection + BiLSTM + Transformer + attention pool)
- Signal pipeline (channel attention + Multi-Scale CNN + BiLSTM + Transformer + attention pool)
- Sequence-level cross-modal attention (temporal, not just pooled)
- Soft-gating fusion
- Final classifier

Architectural contributions beyond baseline:
1. Hybrid BiLSTM-Transformer in both face and signal modules
2. Multi-scale inception-style CNN for physiological signals
3. Sequence-level bidirectional cross-modal attention with multi-head support
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .attention_module import SequenceCrossModalAttention
from .classifier import EmotionClassifier
from .face_module import FaceModule
from .fusion_module import SoftGatingFusion
from .signal_module import SignalModule


class MultimodalEmotionModel(nn.Module):
    """Research-level multimodal architecture for NeuroBioSense."""

    def __init__(self, num_classes: int = 7) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.face_module = FaceModule()
        self.signal_module = SignalModule(channels=6, use_multiscale=True)
        # WHY sequence-level: allows each video frame to attend to each
        # signal time step, capturing fine-grained temporal correspondences.
        self.cross_modal_attention = SequenceCrossModalAttention(
            vid_dim=128, sig_dim=256, attn_dim=128, num_heads=4
        )
        self.fusion = SoftGatingFusion(vid_dim=128, sig_dim=256, fused_dim=384)
        self.classifier = EmotionClassifier(input_dim=384, num_classes=self.num_classes)

    def apply_stage3_freezing(self, unfreeze_backbone: bool = False) -> None:
        """Apply the Stage 3 trainable/frozen policy, optionally unfreezing the visual backbone."""
        self.face_module.set_stage3_policy()
        if unfreeze_backbone:
            self.face_module.backbone.set_stage1_policy()
        self.signal_module.set_stage3_policy()

    def forward(
        self,
        video: torch.Tensor,
        signal: torch.Tensor,
        use_face: bool = True,
        use_signal: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            video: Tensor of shape (B, T_v, 3, 160, 160).
            signal: Tensor of shape (B, T_s, 6).
            use_face: Whether to execute face branch.
            use_signal: Whether to execute signal branch.

        Returns:
            output: Log-probabilities tensor of shape (B, C).
            val_output: Valence Log-probabilities tensor of shape (B, 3).
            confidence: Scalar for B=1 else tensor shape (B,).
        """
        bsz, t_v = video.shape[0], video.shape[1]

        if use_face:
            facenet_emb, vid_emb, _ = self.face_module(video)
            vid_seq = facenet_emb  # (B, T_v, 128) temporal sequence
        else:
            vid_seq = torch.zeros((bsz, t_v, 128), device=video.device, dtype=video.dtype)
            vid_emb = torch.zeros((bsz, 128), device=video.device, dtype=video.dtype)

        if use_signal:
            sig_emb, sig_seq, _ = self.signal_module(signal)
            # sig_seq: (B, T_s//4, 256) temporal sequence
        else:
            sig_emb = torch.zeros((bsz, 256), device=video.device, dtype=video.dtype)
            sig_seq = torch.zeros((bsz, 1, 256), device=video.device, dtype=video.dtype)

        # Sequence-level cross-modal attention: each modality's time steps
        # attend to all time steps of the other modality
        enhanced_vid, enhanced_sig = self.cross_modal_attention(
            vid_seq, sig_seq, vid_emb, sig_emb
        )

        fused, _ = self.fusion(enhanced_vid, enhanced_sig)

        output, val_output = self.classifier(fused)

        probs = torch.exp(output)
        confidence_vec = probs.max(dim=1).values
        confidence = confidence_vec.squeeze(0) if confidence_vec.numel() == 1 else confidence_vec

        return output, val_output, confidence


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = MultimodalEmotionModel().to(device)

    B, T_v, T_s, C = 1, 15, 64, 6
    video = torch.randn(B, T_v, 3, 160, 160).to(device)
    signal = torch.randn(B, T_s, C).to(device)

    with torch.inference_mode():
        output, val_output, confidence = model(video, signal)

    print(f"Output shape : {output.shape}")
    print(f"Valence shape: {val_output.shape}")
    print(f"Confidence   : {float(confidence):.4f}")
    print(f"Predicted    : {output.argmax(dim=1).item()}")
    assert output.shape == (1, 7), "Shape mismatch!"
    print("All assertions passed.")
