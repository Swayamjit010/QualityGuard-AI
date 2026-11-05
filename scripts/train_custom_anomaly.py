"""Train the anomaly model of a user-defined product (created on the "Add a product" page or by hand).

    python scripts/train_custom_anomaly.py <slug> --run-id custom_<slug>-anomaly-resnet18-v1

Expects data/raw/custom_<slug>/product.json, good/*.png|jpg (>= 10, ideally 30+) and optionally defect/*.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.anomaly_detection.custom import _status, train_custom  # noqa: E402
from ml.config import CHECKPOINT_DIR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("--run-id", default=None)
    a = ap.parse_args()
    run_id = a.run_id or f"custom_{a.slug}-anomaly-resnet18-v1"
    try:
        res = train_custom(a.slug, run_id)
        print(json.dumps({k: res[k] for k in ("run_id", "split_sizes", "warnings")}, indent=1))
    except SystemExit:
        raise
    except Exception as e:  # recorded for the UI, then re-raised for the log
        _status(CHECKPOINT_DIR / run_id, state="failed", run_id=run_id, message=f"{type(e).__name__}: {e}"[:300])
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
