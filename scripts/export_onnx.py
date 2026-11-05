"""Edge-deployment study (research direction 11): ONNX export + INT8 quantisation, measured on the test split.

    pip install onnx onnxruntime
    python scripts/export_onnx.py                                   # MobileNetV3-Small and MobileNetV2
    python scripts/export_onnx.py --runs neu_cls-mobilenet_v2-v1 --threads 2

Writes artifacts/edge/<run>.onnx, <run>.int8.onnx and artifacts/metrics/edge_export.{json,md}.
The ONNX files are not used by the web application (which runs PyTorch); they are the hand-off format for
edge runtimes (ONNX Runtime, TensorRT, OpenVINO).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.benchmarking.edge import study  # noqa: E402
from ml.config import ARTIFACTS_DIR, CHECKPOINT_DIR, METRICS_DIR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["neu_cls-mobilenet_v3_small-v1", "neu_cls-mobilenet_v2-v1"])
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    reports = []
    for r in a.runs:
        if not (CHECKPOINT_DIR / r / "model.pt").exists():
            print(f"skip {r}: no checkpoint")
            continue
        print(f"{r} ...", flush=True)
        reports.append(study(r, ARTIFACTS_DIR / "edge", threads=a.threads))
    (METRICS_DIR / "edge_export.json").write_text(json.dumps(reports, indent=2))
    md = ["# Edge export (ONNX / INT8)", "",
          f"Held-out test split; INT8 static quantisation calibrated on validation images; CPU, {a.threads} thread(s).", "",
          "| Model | Variant | Accuracy | Macro F1 | Agreement with PyTorch | Max abs logit diff | Latency (ms) | Size (MB) |",
          "|---|---|---|---|---|---|---|---|"]
    for rep in reports:
        for k, v in rep["variants"].items():
            md.append(f"| {rep['arch']} | {k} | {v['accuracy']:.4f} | {v['f1_macro']:.4f} | "
                      f"{v['agreement_with_pytorch']:.4f} | {v['max_abs_logit_diff']:.4f} | {v['latency_ms']} | {v['size_mb']} |")
    (METRICS_DIR / "edge_export.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
