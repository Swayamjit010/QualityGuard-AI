"""Measure inference performance on this machine (proposal Part 22: no estimated values).

    python scripts/benchmark.py                       # every trained classifier + anomaly model + profiles
    python scripts/benchmark.py --quick               # fewer repetitions
    python scripts/benchmark.py --api http://127.0.0.1:8000 --api-key qg_...   # also time the HTTP API

Writes artifacts/metrics/benchmark.json and artifacts/metrics/benchmark.md.
Run it on an otherwise idle machine: concurrent training jobs inflate every latency.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.benchmarking.bench import (_PeakRSS, bench_anomaly, bench_api, bench_classifier,  # noqa: E402
                                   bench_pipeline, environment, load_image, save, to_markdown)
from ml.config import CHECKPOINT_DIR, METRICS_DIR  # noqa: E402


def first_test_image(dataset: str, prefer_label: str | None = None) -> Path:
    m = json.loads((ROOT / "data" / "processed" / dataset / "manifest_latest.json").read_text())
    rows = [s for s in m["samples"] if s["split"] == "test"]
    if prefer_label:
        rows = [s for s in rows if s["label"] == prefer_label] or rows
    return ROOT / rows[0]["path"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="*", help="run ids to include (default: all completed runs)")
    ap.add_argument("--api", default=None, help="base URL of a running server")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--api-profile", default="steel_strip")
    a = ap.parse_args()
    runs_clf, runs_an = (10, 4) if a.quick else (30, 10)

    rep = {"measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "environment": environment(),
           "classifiers": [], "anomaly": [], "pipeline": []}
    with _PeakRSS(every_s=0.05) as peak:
        for d in sorted(CHECKPOINT_DIR.iterdir()):
            if a.only and d.name not in a.only:
                continue
            if (d / "model.pt").exists():
                ds = json.loads((d / "config.json").read_text()).get("dataset", "neu_cls")
                print(f"classifier {d.name} ...", flush=True)
                rep["classifiers"].append(bench_classifier(d.name, load_image(first_test_image(ds)), runs_clf))
            elif (d / "anomaly.pt").exists():
                cat = json.loads((d / "metrics.json").read_text()).get("category")
                print(f"anomaly {d.name} ...", flush=True)
                rep["anomaly"].append(bench_anomaly(d.name, load_image(first_test_image(f"mvtec_{cat}")), runs_an))
        for pf in sorted((ROOT / "configs" / "profiles").glob("*.json")):
            prof = json.loads(pf.read_text())
            print(f"pipeline {prof['id']} ...", flush=True)
            try:
                rep["pipeline"].append(bench_pipeline(prof, load_image(first_test_image(prof["dataset"])),
                                                      3 if a.quick else 5))
            except Exception as e:  # a profile whose models are not trained is reported, not hidden
                rep["pipeline"].append({"profile_id": prof["id"], "error": str(e)})
        if a.api:
            if not a.api_key:
                sys.exit("--api requires --api-key (create one on the Admin page)")
            prof = json.loads((ROOT / "configs" / "profiles" / f"{a.api_profile}.json").read_text())
            rep["api"] = bench_api(a.api, a.api_key, a.api_profile, first_test_image(prof["dataset"]))
    rep["peak_rss_mb"] = round(peak.peak, 1)
    rep["pipeline"] = [p for p in rep["pipeline"] if "error" not in p] + \
                      [p for p in rep["pipeline"] if "error" in p]
    save(rep, METRICS_DIR)
    print(to_markdown(rep))


if __name__ == "__main__":
    main()
