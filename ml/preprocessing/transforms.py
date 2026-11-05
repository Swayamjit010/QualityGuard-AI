"""Preprocessing + augmentation. The *same* eval transform is used for training-time validation,
offline evaluation and online inference, driven by a serialisable ``PreprocessConfig``.

Augmentation policy is declared per dataset so that only meaning-preserving transforms are used.
E.g. for NEU steel textures, flips/90-degree rotations are physically plausible (strip orientation
is arbitrary in the crops) but Cutout is disabled because it can erase the very defect that
defines the label.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import albumentations as A
import cv2
import numpy as np
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class PreprocessConfig:
    input_size: int = 224
    mean: tuple = IMAGENET_MEAN
    std: tuple = IMAGENET_STD
    color_mode: str = "rgb"           # 'rgb' or 'gray3' (grayscale replicated to 3 channels)
    clahe: bool = False               # contrast-limited histogram equalisation
    denoise: bool = False             # mild bilateral filtering
    augment: dict = field(default_factory=lambda: {
        "hflip": True, "vflip": True, "rot90": True, "rotate_deg": 0, "zoom": 0.1,
        "translate": 0.05, "brightness": 0.15, "contrast": 0.15, "noise": True, "cutout": False,
    })

    def to_dict(self) -> dict:
        d = asdict(self)
        d["mean"], d["std"] = list(self.mean), list(self.std)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PreprocessConfig":
        d = dict(d)
        d["mean"], d["std"] = tuple(d.get("mean", IMAGENET_MEAN)), tuple(d.get("std", IMAGENET_STD))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def base_prepare(bgr: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Colour conversion + optional enhancement. Input BGR uint8 -> RGB uint8."""
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    if cfg.color_mode == "gray3":
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if cfg.clahe:
            g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
        if cfg.denoise:
            g = cv2.bilateralFilter(g, 5, 25, 25)
        return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if cfg.clahe:
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    if cfg.denoise:
        rgb = cv2.bilateralFilter(rgb, 5, 25, 25)
    return rgb


def build_train_aug(cfg: PreprocessConfig) -> A.Compose:
    a = cfg.augment
    t = [A.Resize(cfg.input_size, cfg.input_size)]
    if a.get("hflip"):
        t.append(A.HorizontalFlip(p=0.5))
    if a.get("vflip"):
        t.append(A.VerticalFlip(p=0.5))
    if a.get("rot90"):
        t.append(A.RandomRotate90(p=0.5))
    if a.get("zoom") or a.get("translate") or a.get("rotate_deg"):
        t.append(A.Affine(scale=(1 - a.get("zoom", 0), 1 + a.get("zoom", 0)),
                          translate_percent=(-a.get("translate", 0), a.get("translate", 0)),
                          rotate=(-a.get("rotate_deg", 0), a.get("rotate_deg", 0)),
                          border_mode=cv2.BORDER_REFLECT_101, p=0.7))
    if a.get("brightness") or a.get("contrast"):
        t.append(A.RandomBrightnessContrast(a.get("brightness", 0), a.get("contrast", 0), p=0.7))
    if a.get("noise"):
        t.append(A.GaussNoise(std_range=(0.01, 0.04), p=0.3))
    if a.get("cutout"):
        t.append(A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(0.05, 0.12),
                                 hole_width_range=(0.05, 0.12), p=0.3))
    return A.Compose(t)


def to_tensor(rgb: np.ndarray, cfg: PreprocessConfig) -> torch.Tensor:
    x = rgb.astype(np.float32) / 255.0
    x = (x - np.array(cfg.mean, dtype=np.float32)) / np.array(cfg.std, dtype=np.float32)
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


def preprocess_for_inference(bgr: np.ndarray, cfg: PreprocessConfig) -> tuple[torch.Tensor, np.ndarray]:
    """Returns (normalised CHW tensor, resized RGB uint8 for visualisation)."""
    rgb = base_prepare(bgr, cfg)
    rgb = cv2.resize(rgb, (cfg.input_size, cfg.input_size), interpolation=cv2.INTER_AREA)
    return to_tensor(rgb, cfg), rgb


def denormalize(t: torch.Tensor, cfg: PreprocessConfig) -> np.ndarray:
    x = t.detach().cpu().numpy().transpose(1, 2, 0)
    x = x * np.array(cfg.std) + np.array(cfg.mean)
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)
