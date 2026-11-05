"""Unsupervised anomaly detection trained on defect-free images only.

* PatchCore (Roth et al., CVPR 2022): mid-level patch features (layer2+layer3) with 3x3 local
  aggregation, a greedy k-center coreset memory bank, and nearest-neighbour patch scoring.
* PaDiM (Defard et al., ICPR 2021): per-position multivariate Gaussian on randomly-selected
  embedding dims from layer1..3; Mahalanobis distance map.
* GlobalNovelty: global pooled embedding kNN distance to the normal set. Used as Engine C to flag
  inputs that are not even the right *product* (unfamiliar pattern), distinct from defects.

All models expose ``predict(batch) -> (image_scores[N], anomaly_maps[N,H,W])`` where maps are at
the input resolution and are produced by the actual forward pass.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


def gaussian_blur(maps: torch.Tensor, sigma: float = 4.0) -> torch.Tensor:
    k = int(2 * math.ceil(3 * sigma) + 1)
    x = torch.arange(k, dtype=torch.float32) - k // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    m = maps.unsqueeze(1)
    m = F.conv2d(F.pad(m, (k // 2, k // 2, 0, 0), mode="reflect"), g.view(1, 1, 1, k))
    m = F.conv2d(F.pad(m, (0, 0, k // 2, k // 2), mode="reflect"), g.view(1, 1, k, 1))
    return m.squeeze(1)


class FeatureExtractor(nn.Module):
    def __init__(self, backbone: str = "resnet18", layers=("layer1", "layer2", "layer3")):
        super().__init__()
        weights = {"resnet18": tvm.ResNet18_Weights.DEFAULT,
                   "wide_resnet50_2": tvm.Wide_ResNet50_2_Weights.DEFAULT}[backbone]
        net = getattr(tvm, backbone)(weights=weights).eval()
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2, self.layer3 = net.layer1, net.layer2, net.layer3
        self.layers = layers
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x) -> dict[str, torch.Tensor]:
        out = {}
        x = self.stem(x)
        x = self.layer1(x); out["layer1"] = x
        x = self.layer2(x); out["layer2"] = x
        x = self.layer3(x); out["layer3"] = x
        return out


def greedy_coreset(feats: torch.Tensor, ratio: float, proj_dim: int = 128, seed: int = 0) -> torch.Tensor:
    """Approximate greedy k-center selection with a Johnson-Lindenstrauss random projection."""
    n = feats.shape[0]
    k = max(1, int(n * ratio))
    g = torch.Generator().manual_seed(seed)
    proj = torch.randn(feats.shape[1], proj_dim, generator=g) / math.sqrt(proj_dim)
    z = feats @ proj
    idx = [int(torch.randint(n, (1,), generator=g))]
    mind = torch.cdist(z, z[idx[-1]:idx[-1] + 1]).squeeze(1)
    for it in range(k - 1):
        if it % 500 == 0:
            print(f"  coreset {it}/{k}", flush=True)
        i = int(torch.argmax(mind))
        idx.append(i)
        mind = torch.minimum(mind, torch.cdist(z, z[i:i + 1]).squeeze(1))
    return torch.tensor(idx)


class PatchCore:
    name = "patchcore"

    def __init__(self, extractor: FeatureExtractor, coreset_ratio: float = 0.1, k: int = 1):
        self.ex, self.ratio, self.k = extractor, coreset_ratio, k
        self.bank: torch.Tensor | None = None
        self.fmap_hw: tuple[int, int] | None = None

    def _patch_features(self, x: torch.Tensor) -> torch.Tensor:
        f = self.ex(x)
        f2 = F.avg_pool2d(f["layer2"], 3, 1, 1)
        f3 = F.avg_pool2d(f["layer3"], 3, 1, 1)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        feat = torch.cat([f2, f3], 1)                       # N, C, h, w
        self.fmap_hw = feat.shape[-2:]
        return feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1])

    def fit(self, batches) -> dict:
        feats = torch.cat([self._patch_features(x) for x in batches])
        idx = greedy_coreset(feats, self.ratio)
        self.bank = feats[idx].contiguous()
        return {"patches_total": int(feats.shape[0]), "memory_bank": int(self.bank.shape[0]),
                "feature_dim": int(self.bank.shape[1])}

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        n = x.shape[0]
        f = self._patch_features(x)
        d = []
        for chunk in f.split(4096):
            dist = torch.cdist(chunk, self.bank)
            d.append(dist.topk(self.k, largest=False).values.mean(1))
        d = torch.cat(d).view(n, *self.fmap_hw)
        scores = d.view(n, -1).max(1).values
        maps = F.interpolate(d.unsqueeze(1), size=x.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        return scores.numpy(), gaussian_blur(maps, 4).numpy()

    def state(self):
        return {"bank": self.bank.half(), "fmap_hw": list(self.fmap_hw), "ratio": self.ratio, "k": self.k}

    def load(self, s):
        self.bank, self.fmap_hw, self.ratio, self.k = s["bank"].float(), tuple(s["fmap_hw"]), s["ratio"], s["k"]
        return self


class PaDiM:
    name = "padim"

    def __init__(self, extractor: FeatureExtractor, d: int = 100, seed: int = 0):
        self.ex, self.d = extractor, d
        g = torch.Generator().manual_seed(seed)
        self.sel = torch.randperm(64 + 128 + 256, generator=g)[:d]
        self.mean = self.prec = None

    def _emb(self, x):
        f = self.ex(x)
        e = f["layer1"]
        for l in ("layer2", "layer3"):
            e = torch.cat([e, F.interpolate(f[l], size=e.shape[-2:], mode="nearest")], 1)
        return e[:, self.sel]  # N, d, H, W

    def fit(self, batches) -> dict:
        embs = torch.cat([self._emb(x) for x in batches])  # N,d,H,W
        n, d, h, w = embs.shape
        e = embs.permute(2, 3, 0, 1).reshape(h * w, n, d).double()
        self.mean = e.mean(1)                                   # HW, d
        c = e - self.mean[:, None]
        cov = torch.einsum("pnd,pne->pde", c, c) / (n - 1) + 0.01 * torch.eye(d, dtype=torch.float64)
        self.prec = torch.linalg.inv(cov).float()
        self.mean = self.mean.float()
        self.hw = (h, w)
        return {"positions": h * w, "dims": d, "train_images": n}

    @torch.no_grad()
    def predict(self, x):
        e = self._emb(x)
        n, d, h, w = e.shape
        e = e.permute(2, 3, 0, 1).reshape(h * w, n, d) - self.mean[:, None]
        m = torch.einsum("pnd,pde,pne->pn", e, self.prec, e).clamp_min(0).sqrt()  # HW, n
        m = m.T.reshape(n, h, w)
        maps = F.interpolate(m.unsqueeze(1), size=x.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        maps = gaussian_blur(maps, 4)
        return maps.view(n, -1).max(1).values.numpy(), maps.numpy()

    def state(self):
        return {"mean": self.mean.half(), "prec": self.prec.half(), "sel": self.sel, "hw": list(self.hw), "d": self.d}

    def load(self, s):
        self.mean, self.prec, self.sel = s["mean"].float(), s["prec"].float(), s["sel"]
        self.hw, self.d = tuple(s["hw"]), s["d"]
        return self


class GlobalNovelty:
    """Engine C for anomaly profiles: is this image the expected product at all?"""

    name = "global_novelty"

    def __init__(self, extractor: FeatureExtractor, k: int = 3):
        self.ex, self.k = extractor, k
        self.ref: torch.Tensor | None = None
        self.threshold: float | None = None

    def _g(self, x):
        f = self.ex(x)
        g = torch.cat([F.adaptive_avg_pool2d(f["layer2"], 1).flatten(1), F.adaptive_avg_pool2d(f["layer3"], 1).flatten(1)], 1)
        return F.normalize(g, dim=1)

    def fit(self, batches):
        self.ref = torch.cat([self._g(x) for x in batches])
        return {"reference_images": int(self.ref.shape[0])}

    @torch.no_grad()
    def score(self, x) -> np.ndarray:
        d = torch.cdist(self._g(x), self.ref)
        return d.topk(min(self.k, self.ref.shape[0]), largest=False).values.mean(1).numpy()

    def state(self):
        return {"ref": self.ref.half(), "k": self.k, "threshold": self.threshold}

    def load(self, s):
        self.ref, self.k, self.threshold = s["ref"].float(), s["k"], s["threshold"]
        return self
