"""Anomaly model for a user-defined product, trained from photos of GOOD items (defective photos optional).

Protocol (documented in docs/QUALITY_RULES.md):
  * out-of-fold calibration: the good photos are split into K folds (K <= 5); for each fold the engines are fitted
    on the other folds and score the held-out fold, so every good photo gets a score from a model that never saw
    it. The deployed model is then fitted on all good photos;
  * image threshold per engine = max out-of-fold good score x THRESHOLD_MARGIN (no defects are needed);
  * pixel threshold per engine = 99.5th percentile of out-of-fold good anomaly-map values;
  * unfamiliar-input threshold sits between the good photos and images of OTHER products:
    max(out-of-fold good distance x 2, half the 5th percentile of foreign-image distances), so "unfamiliar" means
    "a different kind of item" while defective items of this product are still judged as defects;
  * if defective photos are supplied they are used ONLY for reporting (detection rate / AUROC at the chosen
    threshold) - never to tune it - so the report is an honest estimate.
Writes the same artefacts as the MVTec trainer (anomaly.pt, metrics.json, status.json), so the inference pipeline,
registry and UI treat custom products exactly like built-in ones.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ml.anomaly_detection.models import FeatureExtractor, GlobalNovelty, PaDiM, PatchCore
from ml.anomaly_detection.train import ANOMALY_PP, foreign_rows
from ml.config import CHECKPOINT_DIR, RAW_DIR, absp, seed_everything
from ml.datasets.analysis import read_image
from ml.evaluation import metrics as M
from ml.preprocessing.transforms import preprocess_for_inference

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
# PatchCore (primary) threshold = max out-of-fold good score: the final model, fitted on all good photos, scores
# unseen good items lower than the out-of-fold models did, so this is already conservative. PaDiM only escalates.
THRESHOLD_MARGIN = {"patchcore": 1.0, "padim": 1.10}
NOVELTY_GOOD_FACTOR = 2.0
NOVELTY_FOREIGN_FRACTION = 0.5
REVIEW_MARGIN = 0.05
FUSION_MODE = "primary"
MIN_GOOD = 10


def product_dir(slug: str) -> Path:
    return RAW_DIR / f"custom_{slug}"


def list_images(d: Path) -> list[Path]:
    return sorted(p for p in d.glob("*") if p.suffix.lower() in IMAGE_EXTS) if d.exists() else []


def _status(run_dir: Path, **kw):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "status.json").write_text(json.dumps({**kw, "updated_at": datetime.now(timezone.utc).isoformat()}))


def _batch(paths: list[Path]) -> torch.Tensor:
    return torch.stack([preprocess_for_inference(read_image(p), ANOMALY_PP)[0] for p in paths])


def _chunks(paths, bs=8):
    for i in range(0, len(paths), bs):
        yield paths[i:i + bs]


def _scores(model, paths):
    s, m = [], []
    for c in _chunks(paths):
        a, b = model.predict(_batch(c))
        s.append(a)
        m.append(b)
    return (np.concatenate(s), np.concatenate(m)) if s else (np.array([]), np.zeros((0, 1, 1)))


def train_custom(slug: str, run_id: str, backbone: str = "resnet18", seed: int = 42) -> dict:
    seed_everything(seed)
    torch.set_num_threads(max(2, torch.get_num_threads() // 2))
    pdir = product_dir(slug)
    meta = json.loads((pdir / "product.json").read_text(encoding="utf-8"))
    good, defect = list_images(pdir / "good"), list_images(pdir / "defect")
    run_dir = CHECKPOINT_DIR / run_id
    if len(good) < MIN_GOOD:
        _status(run_dir, state="failed", run_id=run_id, message=f"need at least {MIN_GOOD} good photos, got {len(good)}")
        raise SystemExit(f"need at least {MIN_GOOD} good photos")
    t0 = time.time()
    k = max(2, min(5, len(good) // 4))
    folds = [good[i::k] for i in range(k)]
    coreset = 0.10 if len(good) < 60 else 0.02
    ex = FeatureExtractor(backbone)
    oof = {"patchcore": [], "padim": []}
    oof_maps = {"patchcore": [], "padim": []}
    oof_g = []
    for fi in range(k):  # out-of-fold calibration: every good photo is scored by a model that never saw it
        _status(run_dir, state="running", run_id=run_id, stage="calibrating", progress=0.05 + 0.6 * fi / k,
                message=f"calibration fold {fi + 1} of {k}")
        train = [p for j, f in enumerate(folds) if j != fi for p in f]
        bt = [_batch(c) for c in _chunks(train, 16)]
        engines = {"patchcore": PatchCore(ex, coreset), "padim": PaDiM(ex)}
        g_f = GlobalNovelty(ex)
        for m_ in engines.values():
            m_.fit(bt)
        g_f.fit(bt)
        del bt
        for n_, m_ in engines.items():
            sc_, mc_ = _scores(m_, folds[fi])
            oof[n_].append(sc_)
            oof_maps[n_].append(mc_)
        oof_g.append(np.concatenate([g_f.score(_batch(c)) for c in _chunks(folds[fi])]))
    _status(run_dir, state="running", run_id=run_id, stage="fitting", progress=0.7,
            message=f"fitting the final model on all {len(good)} good photos")
    pc, pd, gn = PatchCore(ex, coreset), PaDiM(ex), GlobalNovelty(ex)
    batches = [_batch(c) for c in _chunks(good, 16)]
    fit_info = {"patchcore": pc.fit(batches), "padim": pd.fit(batches), "global_novelty": gn.fit(batches)}
    del batches
    _status(run_dir, state="running", run_id=run_id, stage="evaluating", progress=0.85, message="evaluating")

    res = {"run_id": run_id, "category": f"custom_{slug}", "dataset": f"custom_{slug}", "product": meta,
           "backbone": backbone, "dataset_version": f"{len(good)}g{len(defect)}d",
           "split_sizes": {"good": len(good), "calibration_folds": k, "defect_examples": len(defect)},
           "fit": fit_info, "engines": {}, "calibration_protocol": {
               "image_threshold": f"max out-of-fold good score x {THRESHOLD_MARGIN} ({k}-fold)",
               "fusion": "PatchCore decides; within +/-5 % of its threshold or PaDiM flagging a PatchCore PASS -> REVIEW",
               "pixel_threshold": "99.5th percentile of out-of-fold good anomaly-map values",
               "unfamiliar_threshold": f"max(out-of-fold good distance x {NOVELTY_GOOD_FACTOR}, "
                                       f"{NOVELTY_FOREIGN_FRACTION} x 5th percentile of foreign-image distances)",
               "defect_examples_use": "reporting only (never used to set thresholds)"}}
    thr, px_thr, d_scores = {}, {}, {}
    for model in (pc, pd):
        sc, mc = np.concatenate(oof[model.name]), np.concatenate(oof_maps[model.name])
        thr[model.name] = float(sc.max() * THRESHOLD_MARGIN[model.name])
        px_thr[model.name] = float(np.quantile(mc.ravel(), 0.995))
        e = {"deployed_threshold": thr[model.name], "threshold_rule": res["calibration_protocol"]["image_threshold"],
             "pixel_threshold_val": px_thr[model.name],
             "score_stats_val_good": {"mean": float(sc.mean()), "max": float(sc.max())},
             "test": {"num_samples": len(sc), "num_anomalous": 0, "image_auroc": None,
                      # the threshold is placed at/above the out-of-fold maximum, so count strictly-above
                      "false_positive_rate": float((sc > thr[model.name]).mean())},
             "test_pixel": {"pixel_auroc": None}}
        if defect:
            sd, _ = _scores(model, defect)
            d_scores[model.name] = sd
            y = np.r_[np.zeros(len(sc)), np.ones(len(sd))]
            e["test"] = M.binary_anomaly_report(y, np.r_[sc, sd], thr[model.name])
            e["test"]["false_positive_rate"] = float((sc > thr[model.name]).mean())  # strict, as above
            e["defect_detection_rate"] = float((sd >= thr[model.name]).mean())
        res["engines"][model.name] = e

    # unfamiliar = a different kind of item: threshold between good photos and images of other products
    gv = np.concatenate(oof_g)
    fr = foreign_rows(meta.get("exclude_from_foreign") or "none")
    gf = np.concatenate([gn.score(_batch([absp(r["path"]) for r in c])) for c in _chunks(fr, 16)]) if fr else np.array([])
    t_good = float(gv.max() * NOVELTY_GOOD_FACTOR)
    t_foreign = float(np.quantile(gf, 0.05) * NOVELTY_FOREIGN_FRACTION) if len(gf) else 0.0
    gn.threshold = max(t_good, t_foreign)
    res["unfamiliar_pattern"] = {"threshold": gn.threshold, "threshold_rule": res["calibration_protocol"]["unfamiliar_threshold"],
                                 "good_out_of_fold_max": float(gv.max()),
                                 "good_flag_rate": float((gv > gn.threshold).mean()), "foreign_images": len(fr),
                                 "foreign_detection_rate": float((gf > gn.threshold).mean()) if len(gf) else None}
    rule = ("PatchCore decides: FAIL above its threshold, PASS below; REVIEW within +/-"
            f"{REVIEW_MARGIN:.0%} of the threshold or when PaDiM flags an item PatchCore passed")
    if defect:
        r_pc = d_scores["patchcore"] / thr["patchcore"]
        near = np.abs(r_pc - 1) < REVIEW_MARGIN
        fail = (r_pc >= 1) & ~near
        review = near | (~fail & (d_scores["padim"] >= thr["padim"]))
        res["fusion"] = {"rule": rule, "review_margin": REVIEW_MARGIN, "mode": FUSION_MODE,
                         "defects_auto_failed": int(fail.sum()), "defects_sent_to_review": int((review & ~fail).sum()),
                         "defects_auto_passed": int((~fail & ~review).sum())}
    else:
        res["fusion"] = {"rule": rule, "review_margin": REVIEW_MARGIN, "mode": FUSION_MODE}
    x1 = _batch(good[:1])
    lat = {}
    for model in (pc, pd):
        model.predict(x1)
        ts = []
        for _ in range(3):
            t = time.perf_counter()
            model.predict(x1)
            ts.append(time.perf_counter() - t)
        lat[model.name] = float(np.median(ts) * 1000)
    res["latency_ms_median"] = lat
    res["total_time_s"] = time.time() - t0
    res["completed_at"] = datetime.now(timezone.utc).isoformat()
    res["preprocess"] = ANOMALY_PP.to_dict()
    warnings = []
    if len(good) < 20:
        warnings.append(f"Only {len(good)} good photos. Add more (30+, taken the same way) for more reliable "
                        "decisions; until then expect more items to go to REVIEW.")
    for n, e in res["engines"].items():
        if n == "patchcore" and defect and e.get("defect_detection_rate", 1) < 0.8:
            warnings.append(f"{n} caught only {e['defect_detection_rate']:.0%} of the supplied defective photos: "
                            "small or subtle defects may be missed - such items will often go to REVIEW.")
    res["warnings"] = warnings
    torch.save({"category": f"custom_{slug}", "backbone": backbone, "preprocess": ANOMALY_PP.to_dict(),
                "patchcore": pc.state(), "padim": pd.state(), "global_novelty": gn.state(), "thresholds": thr,
                "pixel_thresholds": px_thr, "review_margin": REVIEW_MARGIN, "run_id": run_id,
                "fusion_mode": FUSION_MODE, "primary_engine": "patchcore",
                "dataset_version": res["dataset_version"],
                "val_good_max": {n: res["engines"][n]["score_stats_val_good"]["max"] for n in thr}},
               run_dir / "anomaly.pt")
    M.save_json(res, run_dir / "metrics.json")
    _status(run_dir, state="completed", run_id=run_id, stage="done", progress=1.0, message="completed")
    return res
