"""Probability calibration (temperature scaling) and out-of-distribution scoring.

Why: a softmax probability of 0.99 says nothing about whether the input resembles the training
data. We therefore keep two independent signals:

1. **Calibrated confidence** - temperature T fitted on the validation split (Guo et al. 2017).
2. **Feature-space novelty** - Mahalanobis distance of the penultimate embedding to the nearest
   class-conditional Gaussian (shared covariance, Lee et al. 2018), plus the energy score
   (Liu et al. 2020). Thresholds are set on *in-distribution validation data* (e.g. 99th percentile)
   so that, by construction, ~1% of genuine in-distribution images are flagged for review.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def fit_temperature(logits: np.ndarray, y: np.ndarray, iters: int = 300) -> float:
    lg = torch.tensor(logits, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=iters)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lg / log_t.exp(), yt)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp().clamp(0.05, 20).item())


def softmax_np(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = logits / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def energy_score(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    """Negative free energy; HIGHER = more in-distribution. We return -E so higher = more novel."""
    z = logits / T
    m = z.max(1, keepdims=True)
    return -(T * (m.squeeze(1) + np.log(np.exp(z - m).sum(1))))


class MahalanobisOOD:
    def __init__(self):
        self.means: np.ndarray | None = None
        self.prec: np.ndarray | None = None
        self.threshold: float | None = None
        self.energy_threshold: float | None = None

    def fit(self, emb: np.ndarray, y: np.ndarray, num_classes: int, shrinkage: float = 0.01):
        emb = emb.astype(np.float64)
        self.means = np.stack([emb[y == c].mean(0) for c in range(num_classes)])
        centered = emb - self.means[y]
        cov = centered.T @ centered / len(emb)
        cov += shrinkage * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
        self.prec = np.linalg.inv(cov)
        return self

    def score(self, emb: np.ndarray) -> np.ndarray:
        """Min Mahalanobis distance over classes (higher = more novel)."""
        emb = emb.astype(np.float64)
        d = []
        for mu in self.means:
            diff = emb - mu
            d.append(np.einsum("ij,jk,ik->i", diff, self.prec, diff))
        return np.sqrt(np.maximum(np.min(np.stack(d, 1), 1), 0))

    def calibrate(self, val_scores: np.ndarray, val_energy: np.ndarray, quantile: float = 0.99):
        self.threshold = float(np.quantile(val_scores, quantile))
        self.energy_threshold = float(np.quantile(val_energy, quantile))
        return self

    def state(self) -> dict:
        return {"means": self.means, "prec": self.prec, "threshold": self.threshold,
                "energy_threshold": self.energy_threshold}

    @classmethod
    def from_state(cls, s: dict) -> "MahalanobisOOD":
        o = cls()
        o.means, o.prec = np.asarray(s["means"]), np.asarray(s["prec"])
        o.threshold, o.energy_threshold = s["threshold"], s["energy_threshold"]
        return o
