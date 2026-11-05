"""Class-activation explanations computed from the real model forward/backward pass.

* Grad-CAM (Selvaraju et al. 2017)
* Grad-CAM++ (Chattopadhay et al. 2018)
* Score-CAM (Wang et al. 2020) - gradient-free; costs one forward pass per selected channel,
  so channels are limited to the top-k by activation mean.

IMPORTANT: these maps show *what influenced the classifier*, not defect boundaries. The UI
labels them accordingly; defect boundaries come from anomaly maps / segmentation.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ml.architectures.registry import ARCHITECTURES, get_module


def _to_spatial(t: torch.Tensor) -> torch.Tensor:
    """ViT token tensors (N, 1+HW, C) -> (N, C, H, W), dropping the class token."""
    if t.dim() == 3:
        n, l, c = t.shape
        hw = int(round((l - 1) ** 0.5))
        t = t[:, 1:, :] if hw * hw == l - 1 else t
        hw = int(round(t.shape[1] ** 0.5))
        return t.transpose(1, 2).reshape(n, c, hw, hw)
    return t


class CAMExplainer:
    def __init__(self, model: torch.nn.Module, arch: str):
        self.model = model.eval()
        self.layer = get_module(model, ARCHITECTURES[arch].cam_layer)
        self.acts = self.grads = None
        self._h = [self.layer.register_forward_hook(self._fwd)]

    def _fwd(self, m, i, o):
        self.acts = o
        if o.requires_grad:
            o.register_hook(lambda g: setattr(self, "grads", g))

    def remove(self):
        for h in self._h:
            h.remove()

    def _forward_backward(self, x: torch.Tensor, class_idx: int | None):
        x = x.clone().requires_grad_(True)
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(x)
            c = int(logits.argmax(1)) if class_idx is None else class_idx
            logits[0, c].backward()
        return logits.detach(), c

    @staticmethod
    def _norm(cam: torch.Tensor, size) -> np.ndarray:
        cam = F.relu(cam)
        cam = F.interpolate(cam[None, None], size=size, mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        return (cam / cam.max().clamp_min(1e-8)).cpu().numpy()

    def gradcam(self, x, class_idx=None):
        logits, c = self._forward_backward(x, class_idx)
        A, G = _to_spatial(self.acts.detach())[0], _to_spatial(self.grads.detach())[0]
        w = G.mean((1, 2))
        return self._norm((w[:, None, None] * A).sum(0), x.shape[-2:]), c, logits

    def gradcam_pp(self, x, class_idx=None):
        logits, c = self._forward_backward(x, class_idx)
        A, G = _to_spatial(self.acts.detach())[0], _to_spatial(self.grads.detach())[0]
        g2, g3 = G ** 2, G ** 3
        denom = 2 * g2 + (A * g3).sum((1, 2), keepdim=True)
        alpha = g2 / torch.where(denom != 0, denom, torch.ones_like(denom))
        w = (alpha * F.relu(G)).sum((1, 2))
        return self._norm((w[:, None, None] * A).sum(0), x.shape[-2:]), c, logits

    @torch.no_grad()
    def scorecam(self, x, class_idx=None, top_k: int = 32):
        logits = self.model(x)
        c = int(logits.argmax(1)) if class_idx is None else class_idx
        A = _to_spatial(self.acts)[0]                     # C,h,w
        idx = A.mean((1, 2)).topk(min(top_k, A.shape[0])).indices
        ups = F.interpolate(A[idx][None], size=x.shape[-2:], mode="bilinear", align_corners=False)[0]
        mn = ups.flatten(1).min(1).values[:, None, None]
        mx = ups.flatten(1).max(1).values[:, None, None]
        masks = (ups - mn) / (mx - mn).clamp_min(1e-8)
        base = x.mean((2, 3), keepdim=True)
        scores = []
        for chunk in masks.split(8):
            xb = x * chunk[:, None] + base * (1 - chunk[:, None])
            scores.append(F.softmax(self.model(xb), 1)[:, c])
        w = F.softmax(torch.cat(scores), 0)
        cam = (w[:, None, None] * masks).sum(0)
        return self._norm(cam, x.shape[-2:]), c, logits


def overlay(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45, cmap=cv2.COLORMAP_JET) -> np.ndarray:
    heat = cv2.applyColorMap((np.clip(cam, 0, 1) * 255).astype(np.uint8), cmap)[..., ::-1]
    if heat.shape[:2] != rgb.shape[:2]:
        heat = cv2.resize(heat, (rgb.shape[1], rgb.shape[0]))
    return ((1 - alpha) * rgb + alpha * heat).astype(np.uint8)


def heatmap(cam: np.ndarray, cmap=cv2.COLORMAP_JET) -> np.ndarray:
    return cv2.applyColorMap((np.clip(cam, 0, 1) * 255).astype(np.uint8), cmap)[..., ::-1]
