"""Edge deployment study: export a trained classifier to ONNX, quantise it to INT8, and MEASURE what that costs.

For each variant (PyTorch FP32, ONNX FP32, ONNX INT8 static QDQ) on the held-out test split:
accuracy, macro F1, agreement with the PyTorch model, max |logit difference|, file size and CPU latency.
INT8 calibration uses validation images only (the test split is never used for tuning).
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from ml.config import CHECKPOINT_DIR, PROCESSED_DIR, ROOT


def _split_images(dataset: str, split: str, classes: list[str], limit: int | None = None):
    m = json.loads((PROCESSED_DIR / dataset / "manifest_latest.json").read_text())
    rows = [s for s in m["samples"] if s["split"] == split]
    if limit:
        rows = rows[:limit]
    return [(ROOT / r["path"], classes.index(r["label"]) if r["label"] in classes else -1) for r in rows]


def _prep(path: Path, pp) -> np.ndarray:
    from ml.preprocessing.transforms import preprocess_for_inference
    x, _ = preprocess_for_inference(cv2.imread(str(path), cv2.IMREAD_COLOR), pp)
    return x.numpy()[None].astype(np.float32)


def export_onnx(model: torch.nn.Module, input_size: int, out: Path, opset: int = 17) -> Path:
    model.eval()
    dummy = torch.zeros(1, 3, input_size, input_size)
    torch.onnx.export(model, dummy, str(out), input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=opset,
                      dynamo=False)
    import onnx
    onnx.checker.check_model(onnx.load(str(out)))
    return out


def quantize_int8(fp32: Path, out: Path, calib: list[np.ndarray], method: str = "MinMax") -> Path:
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter([{"input": x} for x in calib])

        def get_next(self):
            return next(self.it, None)

    pre = out.with_name(out.stem + "_pre.onnx")
    quant_pre_process(str(fp32), str(pre), skip_symbolic_shape=True)
    quantize_static(str(pre), str(out), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    calibrate_method=getattr(CalibrationMethod, method))
    pre.unlink(missing_ok=True)
    return out


def _metrics(pred: np.ndarray, y: np.ndarray, n_classes: int) -> dict:
    from sklearn.metrics import accuracy_score, f1_score
    ok = y >= 0
    return {"accuracy": float(accuracy_score(y[ok], pred[ok])),
            "f1_macro": float(f1_score(y[ok], pred[ok], average="macro", labels=list(range(n_classes))))}


def _latency(fn, x, runs=30) -> float:
    for _ in range(3):
        fn(x)
    t = []
    for _ in range(runs):
        s = time.perf_counter()
        fn(x)
        t.append((time.perf_counter() - s) * 1000)
    return round(statistics.median(t), 2)


def study(run_id: str, out_dir: Path, calib_n: int = 100, threads: int = 1,
          calib_methods: tuple[str, ...] = ("MinMax",)) -> dict:
    """Run the export/quantisation study for one classifier run. ``threads`` pins ORT and torch to the same
    thread count so latencies are comparable (edge devices typically have 1-4 cores)."""
    import onnxruntime as ort

    from ml.architectures.registry import build_model
    from ml.preprocessing.transforms import PreprocessConfig

    ck = torch.load(CHECKPOINT_DIR / run_id / "model.pt", map_location="cpu", weights_only=False)
    classes, arch = ck["classes"], ck["arch"]
    pp = PreprocessConfig.from_dict(ck["preprocess"])
    model = build_model(arch, len(classes), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    dataset = json.loads((CHECKPOINT_DIR / run_id / "config.json").read_text())["dataset"]
    out_dir.mkdir(parents=True, exist_ok=True)

    fp32 = export_onnx(model, pp.input_size, out_dir / f"{run_id}.onnx")
    calib = [_prep(p, pp) for p, _ in _split_images(dataset, "val", classes, calib_n)]
    # MinMax calibration. (Entropy/Percentile were tried with onnxruntime 1.26 and produced byte-identical
    # models, i.e. the option had no effect in that version, so only MinMax is reported.)
    int8s = {m: quantize_int8(fp32, out_dir / f"{run_id}.int8{'' if m == 'MinMax' else '-' + m.lower()}.onnx",
                              calib, m) for m in calib_methods}

    test = _split_images(dataset, "test", classes)
    X = [_prep(p, pp) for p, _ in test]
    y = np.array([c for _, c in test])
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    paths = {"onnx_fp32": fp32, **{("onnx_int8" if m == "MinMax" else f"onnx_int8_{m.lower()}"): p for m, p in int8s.items()}}
    sess = {k: ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"]) for k, p in paths.items()}
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        with torch.no_grad():
            ref = np.concatenate([model(torch.from_numpy(x)).numpy() for x in X])
            lat_torch = _latency(lambda x: model(torch.from_numpy(x)), X[0])
    finally:
        torch.set_num_threads(prev_threads)
    res = {"pytorch_fp32": {**_metrics(ref.argmax(1), y, len(classes)), "latency_ms": lat_torch,
                            "size_mb": round((CHECKPOINT_DIR / run_id / "model.pt").stat().st_size / 2**20, 2),
                            "agreement_with_pytorch": 1.0, "max_abs_logit_diff": 0.0}}
    for k, s in sess.items():
        logits = np.concatenate([s.run(None, {"input": x})[0] for x in X])
        path = paths[k]
        res[k] = {**_metrics(logits.argmax(1), y, len(classes)),
                  "latency_ms": _latency(lambda x, s=s: s.run(None, {"input": x}), X[0]),
                  "size_mb": round(path.stat().st_size / 2**20, 2),
                  "agreement_with_pytorch": float((logits.argmax(1) == ref.argmax(1)).mean()),
                  "max_abs_logit_diff": float(np.abs(logits - ref).max())}
    report = {"run_id": run_id, "arch": arch, "dataset": dataset, "test_images": len(test),
              "calibration_images_val": len(calib), "threads": threads, "onnxruntime": ort.__version__,
              "variants": res, "files": {k: p.relative_to(ROOT).as_posix() for k, p in paths.items()}}
    return report
