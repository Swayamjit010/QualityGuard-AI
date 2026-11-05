"""Fit + evaluate anomaly engines (PatchCore, PaDiM, unfamiliar-pattern) per MVTec category.

    python scripts/train_anomaly.py bottle metal_nut [--backbone resnet18] [--coreset 0.1]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.anomaly_detection.train import train_category  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("categories", nargs="+")
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--coreset", type=float, default=0.01)
    a = ap.parse_args()
    for c in a.categories:
        train_category(c, a.backbone, a.coreset)
