"""PyTorch Dataset over a manifest split."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from ml.config import absp
from ml.datasets.analysis import read_image
from ml.preprocessing.transforms import PreprocessConfig, base_prepare, build_train_aug, to_tensor

import cv2


class ManifestDataset(Dataset):
    def __init__(self, rows: list[dict], classes: list[str], cfg: PreprocessConfig, train: bool):
        self.rows = rows
        self.classes = classes
        self.cidx = {c: i for i, c in enumerate(classes)}
        self.cfg = cfg
        self.train = train
        self.aug = build_train_aug(cfg) if train else None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = read_image(absp(r["path"]))
        if img is None:
            raise RuntimeError(f"Unreadable image {r['path']}")
        rgb = base_prepare(img, self.cfg)
        if self.aug is not None:
            rgb = self.aug(image=rgb)["image"]
        else:
            rgb = cv2.resize(rgb, (self.cfg.input_size, self.cfg.input_size), interpolation=cv2.INTER_AREA)
        return to_tensor(rgb, self.cfg), torch.tensor(self.cidx[r["label"]], dtype=torch.long)

    def targets(self) -> np.ndarray:
        return np.array([self.cidx[r["label"]] for r in self.rows])
