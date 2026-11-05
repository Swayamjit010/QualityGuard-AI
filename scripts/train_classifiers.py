"""Train and evaluate one or more classification architectures on the same split.

    python scripts/train_classifiers.py --archs custom_cnn mobilenet_v2 efficientnet_b0 --epochs 10
    python scripts/train_classifiers.py --archs resnet50 --resume --run-id neu_cls-resnet50-xxxx

Failed runs are logged (status.json + stderr) and the loop continues with the next architecture.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.training.trainer import TrainConfig, train  # noqa: E402

DEFAULTS = {
    # From-scratch baseline needs a higher LR and more epochs.
    "custom_cnn": {"lr": 2e-3, "epochs": 25, "patience": 5, "freeze_epochs": 0},
    "vit_tiny": {"lr": 5e-4},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", required=True)
    ap.add_argument("--dataset", default="neu_cls")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--loss", default="ce")
    ap.add_argument("--max-minutes", type=float, default=40)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--run-id", default=None)
    a = ap.parse_args()
    summary = []
    for arch in a.archs:
        kw = {"epochs": a.epochs, **DEFAULTS.get(arch, {})}
        if a.lr:
            kw["lr"] = a.lr
        cfg = TrainConfig(arch=arch, dataset=a.dataset, batch_size=a.batch_size, loss=a.loss,
                          max_minutes=a.max_minutes, resume=a.resume, run_id=a.run_id or f"{a.dataset}-{arch}-v1", **kw)
        try:
            r = train(cfg)
            summary.append({"arch": arch, "run_id": r["run_id"], "state": r.get("state", "completed"),
                            "acc": r.get("test", {}).get("accuracy")})
        except Exception as exc:
            traceback.print_exc()
            summary.append({"arch": arch, "state": "failed", "error": str(exc)})
    print("SUMMARY", json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
