"""End-to-end evaluation of the DEPLOYED system on held-out test images.

    python scripts/evaluate_profiles.py [--auto | --no-auto] [--max-per-profile N]

Each test image goes through exactly what /api/inspect does: component recognition (auto mode by default) ->
the routed profile's engines -> fusion -> decision rules -> verdict. Ground truth: 'good' -> NORMAL, otherwise
DEFECTIVE. Explanations (Grad-CAM) are skipped because they do not influence the decision; latency is therefore
the decision latency.

Also runs the foreign probe sets (images of unsupported things) and reports how often they end up as
MANUAL INSPECTION REQUIRED.

Outputs: artifacts/metrics/end_to_end_evaluation.json, reports/end_to_end_evaluation.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from ml.config import CONFIG_DIR, METRICS_DIR, ROOT, absp  # noqa: E402
from ml.datasets.analysis import read_image  # noqa: E402
from ml.datasets.splitting import load_manifest  # noqa: E402
from ml.inference.pipeline import CACHE, RECOGNIZER_RUN, ModelUnavailable, inspect  # noqa: E402
from ml.recognition.router import gate  # noqa: E402

V = ("NORMAL", "DEFECTIVE", "MANUAL INSPECTION REQUIRED")


def run_one(img, requested, profiles, covered):
    t = time.perf_counter()
    rec = None
    if img is not None:
        try:
            r = CACHE.get("recognizer", RECOGNIZER_RUN).recognize(img)
            pid, rec = gate(r, requested, covered)
        except ModelUnavailable:
            pid = requested
    else:
        pid = requested
    prof = profiles.get(pid)
    if prof is None:
        return None, pid, rec, (time.perf_counter() - t) * 1000
    res = inspect(img, prof, {"skip_explain": True, "recognition": rec})
    return res["decision"]["verdict"], pid, rec, (time.perf_counter() - t) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-auto", action="store_true", help="use the profile's own id instead of auto-recognition")
    ap.add_argument("--max-per-profile", type=int, default=0)
    ap.add_argument("--split", default="test", choices=["val", "test"],
                    help="'val' for choosing policies; 'test' for the final, reported numbers")
    ap.add_argument("--profiles", nargs="*", default=None)
    ap.add_argument("--profile-override", default=None,
                    help="JSON merged into each evaluated profile's 'pipeline' (policy comparison on val)")
    a = ap.parse_args()
    profiles = {p.stem: json.loads(p.read_text()) for p in (CONFIG_DIR / "profiles").glob("*.json")}
    if a.profile_override:
        for pr in profiles.values():
            pr.setdefault("pipeline", {}).update(json.loads(a.profile_override))
    try:
        covered = set(CACHE.get("recognizer", RECOGNIZER_RUN).rec.domain_profiles.values())
    except ModelUnavailable:
        covered = set()
        print("!! recogniser not trained - auto mode unavailable; evaluating with explicit profiles")
        a.no_auto = True
    out = {"mode": "explicit profile" if a.no_auto else "auto-recognition (as /api/inspect default)",
           "split": a.split, "profile_override": a.profile_override,
           "completed_at": None, "profiles": {}, "foreign": {}}
    rng = np.random.default_rng(0)
    for pid, prof in sorted(profiles.items()):
        if a.profiles and pid not in a.profiles:
            continue
        try:
            man = load_manifest(prof["dataset"])
        except (FileNotFoundError, KeyError):
            continue
        rows = [r for r in man["samples"] if r["split"] == a.split]
        if a.max_per_profile and len(rows) > a.max_per_profile:
            rows = [rows[i] for i in sorted(rng.choice(len(rows), a.max_per_profile, replace=False))]
        cm = {g: Counter() for g in ("NORMAL", "DEFECTIVE")}
        routed, lat, errors = Counter(), [], []
        for i, r in enumerate(rows):
            gt = "NORMAL" if r["label"] == "good" else "DEFECTIVE"
            img = read_image(absp(r["path"]))
            try:
                v, used, rec, ms = run_one(img, None if not a.no_auto else pid, profiles, covered)
            except ModelUnavailable as exc:
                errors.append(str(exc)); break
            cm[gt][v or "ERROR"] += 1
            routed[used] += 1
            lat.append(ms)
            print(f"  {pid}: {i + 1}/{len(rows)}", end="\r", flush=True)
        print()
        if errors:
            out["profiles"][pid] = {"error": errors[0]}
            print(f"{pid}: {errors[0]}")
            continue
        n_def, n_norm = sum(cm["DEFECTIVE"].values()), sum(cm["NORMAL"].values())
        auto = {g: sum(c for k, c in cm[g].items() if k != "MANUAL INSPECTION REQUIRED") for g in cm}
        correct = cm["DEFECTIVE"]["DEFECTIVE"] + cm["NORMAL"]["NORMAL"]
        res = {
            "dataset": prof["dataset"], "dataset_version": man["version"], "test_images": len(rows),
            "ground_truth": {"DEFECTIVE": n_def, "NORMAL": n_norm},
            "verdicts_by_ground_truth": {g: dict(c) for g, c in cm.items()},
            "routed_to": dict(routed), "routing_accuracy": routed.get(pid, 0) / len(rows),
            "manual_inspection_rate": sum(cm[g]["MANUAL INSPECTION REQUIRED"] for g in cm) / len(rows),
            "auto_decided": sum(auto.values()),
            "auto_decided_accuracy": correct / sum(auto.values()) if sum(auto.values()) else None,
            "missed_defects (DEFECTIVE->NORMAL)": cm["DEFECTIVE"]["NORMAL"],
            "false_alarms (NORMAL->DEFECTIVE)": cm["NORMAL"]["DEFECTIVE"],
            "defect_recall_auto": cm["DEFECTIVE"]["DEFECTIVE"] / n_def if n_def else None,
            "latency_ms": {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95))},
        }
        out["profiles"][pid] = res
        print(pid, json.dumps({k: res[k] for k in ("verdicts_by_ground_truth", "routing_accuracy",
                                                   "manual_inspection_rate", "auto_decided_accuracy")}))
    final = a.split == "test" and not a.profile_override and not a.profiles
    if not final:
        stem = f"end_to_end_{a.split}" + ("_override" if a.profile_override else "")
        (METRICS_DIR / f"{stem}.json").write_text(json.dumps(out, indent=2))
        print(f"(policy-selection run; written to artifacts/metrics/{stem}.json only)")
        return
    for name, pattern in (("imagenette_natural_photos", "data/raw/foreign_probe/imagenette/*.JPEG"),
                          ("mvtec_unsupported_categories", "data/raw/foreign_probe/mvtec_unsupported/*.png")):
        paths = sorted(ROOT.glob(pattern))
        if not paths or not covered:
            continue
        c = Counter()
        for p in paths:
            v, *_ = run_one(read_image(p), None, profiles, covered)
            c[v] += 1
        out["foreign"][name] = {"n": len(paths), "verdicts": dict(c),
                                "manual_rate": c["MANUAL INSPECTION REQUIRED"] / len(paths)}
        print(name, out["foreign"][name])
    out["completed_at"] = datetime.now(timezone.utc).isoformat()
    (METRICS_DIR / "end_to_end_evaluation.json").write_text(json.dumps(out, indent=2))
    lines = ["# End-to-end evaluation (deployed decision path, held-out test images)", "",
             f"Mode: **{out['mode']}**. Generated {out['completed_at']} by `scripts/evaluate_profiles.py`.",
             "Ground truth: `good` → NORMAL, any defect label → DEFECTIVE. Explanations skipped (no effect on decisions).", "",
             "| Profile | Test images (def/normal) | Routed correctly | Manual rate | Auto-decided accuracy | Missed defects | False alarms | p50 latency |",
             "|---|---|---|---|---|---|---|---|"]
    for pid, r in out["profiles"].items():
        if "error" in r:
            lines.append(f"| {pid} | error: {r['error']} |||||||"); continue
        acc = "—" if r["auto_decided_accuracy"] is None else f"{r['auto_decided_accuracy']:.3f}"
        lines.append(f"| {pid} | {r['test_images']} ({r['ground_truth']['DEFECTIVE']}/{r['ground_truth']['NORMAL']}) | "
                     f"{r['routing_accuracy']:.3f} | {r['manual_inspection_rate']:.3f} | {acc} | "
                     f"{r['missed_defects (DEFECTIVE->NORMAL)']} | {r['false_alarms (NORMAL->DEFECTIVE)']} | "
                     f"{r['latency_ms']['p50']:.0f} ms |")
    if out["foreign"]:
        lines += ["", "## Unsupported images (should become MANUAL INSPECTION REQUIRED)", "",
                  "| Probe set | Images | MANUAL rate | Verdicts |", "|---|---|---|---|"]
        for k, r in out["foreign"].items():
            lines.append(f"| {k} | {r['n']} | {r['manual_rate']:.3f} | {r['verdicts']} |")
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "end_to_end_evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.stdout.buffer.write(("\n".join(lines) + "\n").encode("utf-8", "replace"))  # Windows consoles: cp1252


if __name__ == "__main__":
    main()
