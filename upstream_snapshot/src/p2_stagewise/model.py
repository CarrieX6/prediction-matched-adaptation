"""Shared student used by every P2-F0 comparison arm."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class StagewiseMammographyStudent(nn.Module):
    """Image-only view encoder with auxiliary localization and patient pooling.

    Every arm instantiates the same modules. The comparison changes only loss
    weights; masks never enter ``forward`` and are unavailable at inference.
    """

    def __init__(self, *, imagenet_weights: bool = True) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if imagenet_weights else None
        backbone = efficientnet_b0(weights=weights)
        self.features = backbone.features
        channels = int(backbone.classifier[1].in_features)
        self.view_head = nn.Linear(channels, 1)
        self.localization_head = nn.Conv2d(channels, 1, kernel_size=1)

    @staticmethod
    def _patient_mean(view_logits: torch.Tensor, patient_index: torch.Tensor) -> torch.Tensor:
        if view_logits.ndim != 1:
            raise ValueError("view_logits must have shape [V]")
        if patient_index.dtype != torch.long or patient_index.shape != view_logits.shape:
            raise ValueError("patient_index must be int64 with shape [V]")
        if patient_index.numel() == 0 or int(patient_index.min()) != 0:
            raise ValueError("patient_index must be non-empty and start at zero")
        patient_count = int(patient_index.max()) + 1
        expected = torch.arange(patient_count, device=patient_index.device)
        if not torch.equal(torch.unique(patient_index), expected):
            raise ValueError("patient_index must contain contiguous patient IDs")
        sums = view_logits.new_zeros(patient_count)
        counts = view_logits.new_zeros(patient_count)
        sums.scatter_add_(0, patient_index, view_logits)
        counts.scatter_add_(0, patient_index, torch.ones_like(view_logits))
        return sums / counts.clamp_min(1.0)

    def forward(self, images: torch.Tensor, patient_index: torch.Tensor) -> dict[str, torch.Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [V,3,H,W]")
        feature_map = self.features(images)
        pooled = feature_map.mean(dim=(-2, -1))
        view_logits = self.view_head(pooled).squeeze(1)
        patient_logits = self._patient_mean(view_logits, patient_index)
        localization_logits = self.localization_head(feature_map)
        return {
            "view_logits": view_logits,
            "patient_logits": patient_logits,
            "localization_logits": localization_logits,
        }


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Mean soft Dice loss for binary localization maps."""

    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have identical shapes")
    probabilities = torch.sigmoid(logits)
    axes = tuple(range(1, logits.ndim))
    intersection = (probabilities * targets).sum(dim=axes)
    denominator = probabilities.sum(dim=axes) + targets.sum(dim=axes)
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()

