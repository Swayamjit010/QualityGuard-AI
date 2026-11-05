"""Supervised defect segmentation: U-Net decoder on an ImageNet ResNet-18 encoder.

Trained on pixel-annotated data (KolektorSDD2 ``*_GT.png`` masks). Unlike anomaly maps and Grad-CAM, its
output is a learned per-pixel defect probability, so its mask is the *precise* boundary estimate shown
in the UI - but only for the surface type it was trained on.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


class _Up(nn.Module):
    def __init__(self, cin: int, cskip: int, cout: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin + cskip, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], 1))


class ResNetUNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        r = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)      # /2, 64
        self.pool = r.maxpool
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4  # /4 64, /8 128, /16 256, /32 512
        self.u4 = _Up(512, 256, 256)
        self.u3 = _Up(256, 128, 128)
        self.u2 = _Up(128, 64, 64)
        self.u1 = _Up(64, 64, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def encoder_params(self):
        for m in (self.stem, self.l1, self.l2, self.l3, self.l4):
            yield from m.parameters()

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.l1(self.pool(s0))
        s2 = self.l2(s1)
        s3 = self.l3(s2)
        s4 = self.l4(s3)
        d = self.u4(s4, s3)
        d = self.u3(d, s2)
        d = self.u2(d, s1)
        d = self.u1(d, s0)
        return F.interpolate(self.head(d), size=x.shape[-2:], mode="bilinear", align_corners=False)


def image_score(prob: torch.Tensor, k: int = 0) -> torch.Tensor:
    """Image-level defect score = max of the 5x5-average-pooled probability map (suppresses single-pixel noise)."""
    return F.avg_pool2d(prob, 5, 1, 2).flatten(1).max(1).values


def seg_loss(logits: torch.Tensor, target: torch.Tensor, pos_weight: float = 5.0) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=torch.tensor(pos_weight))
    p = torch.sigmoid(logits)
    has = target.flatten(1).sum(1) > 0
    if has.any():
        inter = (p * target).flatten(1).sum(1)[has]
        dice = 1 - (2 * inter + 1) / (p.flatten(1).sum(1)[has] + target.flatten(1).sum(1)[has] + 1)
        return bce + dice.mean()
    return bce
