from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al. 2017) with optional class weights and label smoothing."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma, self.weight, self.ls = gamma, weight, label_smoothing

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none", label_smoothing=self.ls)
        pt = torch.exp(-F.cross_entropy(logits, target, reduction="none"))
        return ((1 - pt) ** self.gamma * ce).mean()


def build_loss(name: str, class_weights: torch.Tensor | None, label_smoothing: float, gamma: float = 2.0):
    if name == "focal":
        return FocalLoss(gamma, class_weights, label_smoothing)
    return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
