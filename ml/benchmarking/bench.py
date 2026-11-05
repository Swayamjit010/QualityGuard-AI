"""Measured (never estimated) performance benchmarks.

Every number produced here is a wall-clock or RSS measurement taken on the machine that runs it:

* model load time (fresh load from disk, cache bypassed)
* single-image forward latency (median / p95 over N runs, after warm-up)
* batch throughput (images per second) for several batch sizes
* process memory: resident-set-size increase caused by loading the model, and peak RSS during inference
* full-pipeline latency per product profile (validation -> models -> rules -> visualisations), with and
  without explanation maps
* optional HTTP API response times against a running server
"""
from __future__ import annotations

import gc
import json
import platform
import statistics
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import psutil
import torch

from ml.config import CHECKPOINT_DIR

_PROC = psutil.Process()


def _rss_mb() -> float:
    return _PROC.memory_info().rss / 2**20


class _PeakRSS:
    """Samples RSS on a background thread so short spikes during inference are captured."""

    def __init__(self, every_s: float = 0.005):
        self.every_s, self.peak, self._stop = every_s, _rss_mb(), threading.Event()

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, _rss_mb())
            time.sleep(self.every_s)

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, _rss_mb())


def _stats(ms: list[float]) -> dict:
    s = sorted(ms)
    return {"median_ms": round(statistics.median(s), 2), "p95_ms": round(s[min(len(s) - 1, int(0.95 * len(s)))], 2),
            "min_ms": round(s[0], 2), "runs": len(s)}


def environment() -> dict:
    return {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform(),
            "cpu": platform.processor() or platform.machine(), "logical_cpus": psutil.cpu_count(),
            "torch_threads": torch.get_num_threads(), "ram_total_gb": round(psutil.virtual_memory().total / 2**30, 1),
            "cuda_available": torch.cuda.is_available(),
            "cpu_load_pct_at_start": psutil.cpu_percent(interval=0.5)}


def bench_classifier(run_id: str, image_bgr: np.ndarray, runs: int = 30, batch_sizes=(1, 8, 16)) -> dict:
    from ml.inference.pipeline import LoadedClassifier
    from ml.preprocessing.transforms import preprocess_for_inference

    gc.collect()
    rss0 = _rss_mb()
    t = time.perf_counter()
    clf = LoadedClassifier(run_id)  # fresh load, not the shared cache
    load_ms = (time.perf_counter() - t) * 1000
    rss_loaded = _rss_mb()
    x, _ = preprocess_for_inference(image_bgr, clf.pp)
    x = x[None]
    with torch.no_grad():
        for _ in range(3):
            clf.model(x)
        lat = []
        with _PeakRSS() as peak:
            for _ in range(runs):
                t = time.perf_counter()
                clf.model(x)
                lat.append((time.perf_counter() - t) * 1000)
        thr = {}
        for bs in batch_sizes:
            xb = x.repeat(bs, 1, 1, 1)
            clf.model(xb)
            n_iter = max(2, 32 // bs)
            t = time.perf_counter()
            for _ in range(n_iter):
                clf.model(xb)
            thr[str(bs)] = round(bs * n_iter / (time.perf_counter() - t), 1)
    size_mb = (CHECKPOINT_DIR / run_id / "model.pt").stat().st_size / 2**20
    out = {"run_id": run_id, "arch": clf.arch, "input_size": clf.pp.input_size, "load_ms": round(load_ms, 1),
           "checkpoint_mb": round(size_mb, 1),
           "params_m": round(sum(p.numel() for p in clf.model.parameters()) / 1e6, 2),
           "single_image": _stats(lat), "throughput_img_per_s": thr,
           "rss_increase_on_load_mb": round(max(0.0, rss_loaded - rss0), 1),
           "peak_rss_during_inference_mb": round(peak.peak, 1)}
    del clf
    gc.collect()
    return out


def bench_anomaly(run_id: str, image_bgr: np.ndarray, runs: int = 10) -> dict:
    from ml.inference.pipeline import LoadedAnomaly
    from ml.preprocessing.transforms import preprocess_for_inference

    gc.collect()
    rss0 = _rss_mb()
    t = time.perf_counter()
    an = LoadedAnomaly(run_id)
    load_ms = (time.perf_counter() - t) * 1000
    rss_loaded = _rss_mb()
    x, _ = preprocess_for_inference(image_bgr, an.pp)
    x = x[None]
    res = {}
    with torch.no_grad():
        for name, eng in an.engines.items():
            eng.predict(x)
            lat = []
            with _PeakRSS() as peak:
                for _ in range(runs):
                    t = time.perf_counter()
                    eng.predict(x)
                    lat.append((time.perf_counter() - t) * 1000)
            res[name] = {**_stats(lat), "peak_rss_mb": round(peak.peak, 1)}
    size_mb = (CHECKPOINT_DIR / run_id / "anomaly.pt").stat().st_size / 2**20
    out = {"run_id": run_id, "load_ms": round(load_ms, 1), "checkpoint_mb": round(size_mb, 1),
           "input_size": an.pp.input_size, "rss_increase_on_load_mb": round(max(0.0, rss_loaded - rss0), 1), "engines": res}
    del an
    gc.collect()
    return out


def bench_pipeline(profile: dict, image_bgr: np.ndarray, runs: int = 5) -> dict:
    """Full hybrid pipeline as used by the API (shared model cache, visualisations included)."""
    from ml.inference.pipeline import inspect

    out = {"profile_id": profile["id"]}
    for label, opts in (("without_explanations", {"skip_explain": True}), ("with_explanations", {})):
        inspect(image_bgr, profile, dict(opts))  # warm-up + model load
        lat, stages = [], []
        for _ in range(runs):
            t = time.perf_counter()
            r = inspect(image_bgr, profile, dict(opts))
            lat.append((time.perf_counter() - t) * 1000)
            stages.append(r.get("timings_ms") or {})
        out[label] = _stats(lat)
        if stages and isinstance(stages[-1], dict):
            out[label]["stage_ms_last_run"] = {k: round(float(v), 1) for k, v in stages[-1].items()
                                               if isinstance(v, (int, float))}
        out[label]["decision_last_run"] = (r.get("decision") or {}).get("status")
    return out


def bench_api(base_url: str, api_key: str, profile_id: str, image_path: Path, runs: int = 5) -> dict:
    import httpx

    h = {"X-API-Key": api_key}
    with httpx.Client(base_url=base_url, timeout=120) as c:
        health = []
        for _ in range(10):
            t = time.perf_counter()
            c.get("/api/health").raise_for_status()
            health.append((time.perf_counter() - t) * 1000)
        insp = []
        for _ in range(runs):
            with open(image_path, "rb") as f:
                t = time.perf_counter()
                r = c.post("/api/inspections", headers=h, data={"profile_id": profile_id, "explain": "false"},
                           files={"file": (image_path.name, f, "image/png")})
                insp.append((time.perf_counter() - t) * 1000)
            r.raise_for_status()
    return {"base_url": base_url, "health": _stats(health), "inspection_post": _stats(insp)}


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def to_markdown(rep: dict) -> str:
    env = rep["environment"]
    md = ["# Benchmark results", "",
          f"Measured {rep['measured_at']} on {env['cpu']} ({env['logical_cpus']} logical CPUs, "
          f"{env['ram_total_gb']} GB RAM, torch {env['torch']}, CUDA: {env['cuda_available']}). "
          f"CPU load at start: {env['cpu_load_pct_at_start']} %.", "",
          "## Classifiers (forward pass only)", "",
          "| Model | Input | Params (M) | Checkpoint (MB) | Load (ms) | Median (ms) | p95 (ms) | img/s @1 | img/s @8 | img/s @16 | RSS on load (MB) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in rep.get("classifiers", []):
        t = c["throughput_img_per_s"]
        md.append(f"| {c['arch']} | {c['input_size']} | {c['params_m']} | {c['checkpoint_mb']} | {c['load_ms']} | "
                  f"{c['single_image']['median_ms']} | {c['single_image']['p95_ms']} | {t.get('1', '')} | "
                  f"{t.get('8', '')} | {t.get('16', '')} | {c['rss_increase_on_load_mb']} |")
    md += ["", "## Anomaly engines (scoring only)", "",
           "| Run | Engine | Median (ms) | p95 (ms) | Load (ms) | Checkpoint (MB) |", "|---|---|---|---|---|---|"]
    for a in rep.get("anomaly", []):
        for e, s in a["engines"].items():
            md.append(f"| {a['run_id']} | {e} | {s['median_ms']} | {s['p95_ms']} | {a['load_ms']} | {a['checkpoint_mb']} |")
    md += ["", "## Full pipeline per profile", "",
           "| Profile | Without explanations, median (ms) | With explanations, median (ms) |", "|---|---|---|"]
    for p in rep.get("pipeline", []):
        if "error" in p:
            md.append(f"| {p['profile_id']} | not run: {p['error']} | |")
            continue
        md.append(f"| {p['profile_id']} | {p['without_explanations']['median_ms']} | {p['with_explanations']['median_ms']} |")
    if rep.get("api"):
        a = rep["api"]
        md += ["", "## HTTP API", "", f"- `GET /api/health`: median {a['health']['median_ms']} ms",
               f"- `POST /api/inspections` (no explanations): median {a['inspection_post']['median_ms']} ms"]
    md += ["", f"Peak process RSS during the run: {rep['peak_rss_mb']} MB.", ""]
    return "\n".join(md)


def save(rep: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark.json").write_text(json.dumps(rep, indent=2))
    (out_dir / "benchmark.md").write_text(to_markdown(rep), encoding="utf-8")
