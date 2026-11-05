"""Shared paths, device selection and reproducibility helpers for the ML package."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
REGISTRY_DIR = ARTIFACTS_DIR / "model_registry"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
EXPERIMENT_DIR = ARTIFACTS_DIR / "experiments"
VIS_DIR = ARTIFACTS_DIR / "visualizations"
CONFIG_DIR = ROOT / "configs"

for _d in (PROCESSED_DIR, CHECKPOINT_DIR, REGISTRY_DIR, METRICS_DIR, EXPERIMENT_DIR, VIS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def get_device(prefer: str = "auto"):
    """Return a torch.device. 'auto' picks CUDA, then Apple MPS, then CPU."""
    import torch

    if prefer == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_info() -> dict:
    import torch

    info = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(get_device()),
        "cpu_threads": torch.get_num_threads(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    return info


def seed_everything(seed: int = 42) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rel(p: Path | str) -> str:
    """Path relative to repo root, with forward slashes (portable in manifests)."""
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def absp(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p
