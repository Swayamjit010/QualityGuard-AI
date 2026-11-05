"""Fit, calibrate and evaluate anomaly-detection engines for one MVTec-style product category.

Protocol (no test-set tuning):
  * fit PatchCore / PaDiM / GlobalNovelty on ``train`` (defect-free) images
  * choose image thresholds, pixel threshold, OOD threshold on ``val`` (calibration half)
  * report everything on ``test`` (held-out half)
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

from ml.anomaly_detection.models import FeatureExtractor, GlobalNovelty, PaDiM, PatchCore
from ml.config import CHECKPOINT_DIR, VIS_DIR, absp, seed_everything
from ml.datasets.analysis import read_image
from ml.datasets.splitting import load_manifest
from ml.evaluation import metrics as M
from ml.preprocessing.transforms import PreprocessConfig, preprocess_for_inference

ANOMALY_PP = PreprocessConfig(input_size=256, color_mode="rgb", augment={})


def load_batch(rows, pp=None, with_masks=False):
    pp = pp or ANOMALY_PP
    xs, masks = [], []
    for r in rows:
        x, _ = preprocess_for_inference(read_image(absp(r["path"])), pp)
        xs.append(x)
        if with_masks:
            if r.get("mask_path"):
                m = cv2.imread(str(absp(r["mask_path"])), cv2.IMREAD_GRAYSCALE)
                m = cv2.resize(m, (pp.input_size, pp.input_size), interpolation=cv2.INTER_NEAREST) > 127
            else:
                m = np.zeros((pp.input_size, pp.input_size), bool)
            masks.append(m)
    return torch.stack(xs), (np.stack(masks) if with_masks else None)


def batched(rows, bs=16):
    for i in range(0, len(rows), bs):
        yield rows[i:i + bs]


def run_model(model, rows, bs=8):
    s, m = [], []
    for chunk in batched(rows, bs):
        x, _ = load_batch(chunk)
        a, b = model.predict(x)
        s.append(a); m.append(b)
    return np.concatenate(s), np.concatenate(m)


def best_pixel_threshold(maps, masks):
    vals = maps[masks].ravel() if masks.any() else maps.ravel()
    cands = np.quantile(maps.ravel(), np.linspace(0.80, 0.999, 60))
    best, bt = -1, cands[0]
    defect = masks.reshape(len(masks), -1).any(1)
    for t in cands:
        pred = maps[defect] >= t
        gt = masks[defect]
        inter = (pred & gt).sum((1, 2)); s = pred.sum((1, 2)) + gt.sum((1, 2))
        dice = (2 * inter / np.maximum(s, 1)).mean()
        if dice > best:
            best, bt = dice, t
    return float(bt), float(best)


def fuse(scores: dict[str, np.ndarray], thr: dict[str, float], margin: float) -> np.ndarray:
    """Decision fusion: 0=PASS, 1=FAIL, 2=REVIEW.
    Each engine votes with its own validated threshold; disagreement or a score within the
    relative ``margin`` of any threshold yields REVIEW."""
    names = list(scores)
    votes = np.stack([(scores[n] >= thr[n]).astype(int) for n in names], 1)
    near = np.stack([np.abs(scores[n] / thr[n] - 1) < margin for n in names], 1).any(1)
    out = np.where(votes.all(1), 1, np.where((1 - votes).all(1), 0, 2))
    out[near] = 2
    return out


def save_vis(rows, maps, thr_px, out_dir: Path, n=6):
    out_dir.mkdir(parents=True, exist_ok=True)
    picked = [i for i, r in enumerate(rows) if r["label"] != "good"][:n - 2] + \
             [i for i, r in enumerate(rows) if r["label"] == "good"][:2]
    for i in picked:
        _, rgb = preprocess_for_inference(read_image(absp(rows[i]["path"])), ANOMALY_PP)
        m = maps[i]
        norm = np.clip(m / (thr_px * 2), 0, 1)
        heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
        over = (0.55 * rgb + 0.45 * heat).astype(np.uint8)
        cnts, _ = cv2.findContours((m >= thr_px).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(over, cnts, -1, (0, 255, 0), 2)
        tile = np.concatenate([rgb, heat, over], 1)
        cv2.imwrite(str(out_dir / f"{rows[i]['label']}_{Path(rows[i]['path']).stem}.png"), tile[..., ::-1])


def foreign_rows(category: str, limit=60, own_manifest: str | None = None) -> list[dict]:
    """Real images of *other* products/surfaces used as an OOD probe set."""
    out = []
    own = own_manifest or f"mvtec_{category}"
    from ml.config import PROCESSED_DIR
    names = sorted(d.name for d in PROCESSED_DIR.iterdir() if (d / "manifest_latest.json").exists())
    for name in names:
        if name == own:
            continue
        try:
            m = load_manifest(name)
        except FileNotFoundError:
            continue
        rows = [r for r in m["samples"] if r["split"] == "test"]
        rng = np.random.default_rng(0)
        out += [rows[i] for i in rng.choice(len(rows), min(max(5, limit // len(names)), len(rows)), replace=False)]
    return out


def train_category(category: str, backbone="resnet18", coreset_ratio=0.01, review_margin=0.08, seed=42,
                   manifest: str | None = None, max_train: int | None = None, input_size: int = 256) -> dict:
    """Fit on defect-free ``train`` images only (defective train images, e.g. in KolektorSDD2, are excluded),
    calibrate on ``val``, report on ``test``."""
    global ANOMALY_PP
    seed_everything(seed)
    torch.set_num_threads(max(2, torch.get_num_threads() // 2))
    manifest = manifest or f"mvtec_{category}"
    ANOMALY_PP = PreprocessConfig(input_size=input_size, color_mode="rgb", augment={})
    man = load_manifest(manifest)
    rows = {s: [r for r in man["samples"] if r["split"] == s] for s in ("train", "val", "test")}
    rows["train"] = [r for r in rows["train"] if r["label"] == "good"]
    if max_train and len(rows["train"]) > max_train:
        pick = np.random.default_rng(seed).choice(len(rows["train"]), max_train, replace=False)
        rows["train"] = [rows["train"][i] for i in sorted(pick)]
    run_id = f"{manifest}-anomaly-{backbone}-v1"
    run_dir = CHECKPOINT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ex = FeatureExtractor(backbone)
    pc, pd, gn = PatchCore(ex, coreset_ratio), PaDiM(ex), GlobalNovelty(ex)
    fit_info = {}
    train_batches = [load_batch(c)[0] for c in batched(rows["train"], 16)]
    t = time.time(); fit_info["patchcore"] = pc.fit(train_batches); fit_info["patchcore"]["fit_s"] = time.time() - t
    t = time.time(); fit_info["padim"] = pd.fit(train_batches); fit_info["padim"]["fit_s"] = time.time() - t
    fit_info["global_novelty"] = gn.fit(train_batches)
    del train_batches
    print(f"[{run_id}] fitted in {time.time() - t0:.0f}s {fit_info}", flush=True)

    y = {s: np.array([int(r["label"] != "good") for r in rows[s]]) for s in ("val", "test")}
    masks = {s: load_batch(rows[s], with_masks=True)[1] for s in ("val", "test")}
    res = {"run_id": run_id, "category": category, "backbone": backbone, "dataset": manifest,
           "dataset_version": man["version"],
           "split_sizes": {k: len(v) for k, v in rows.items()}, "fit": fit_info, "engines": {}}
    thr, scores_test, maps_test = {}, {}, {}
    for model in (pc, pd):
        sv, mv = run_model(model, rows["val"])
        st, mt = run_model(model, rows["test"])
        t_rec = M.threshold_for_recall(y["val"], sv, 0.95)
        t_f1 = M.threshold_for_f1(y["val"], sv)
        thr[model.name] = t_rec
        px_t, px_dice_val = best_pixel_threshold(mv, masks["val"])
        per_type = {}
        for lbl in sorted({r["label"] for r in rows["test"]}):
            idx = [i for i, r in enumerate(rows["test"]) if r["label"] == lbl]
            per_type[lbl] = {"n": len(idx), "flagged_anomalous": int((st[idx] >= t_rec).sum())}
        res["engines"][model.name] = {
            "threshold_recall95_val": t_rec, "threshold_f1_val": t_f1,
            "val": M.binary_anomaly_report(y["val"], sv, t_rec),
            "test": M.binary_anomaly_report(y["test"], st, t_rec),
            "test_at_f1_threshold": M.binary_anomaly_report(y["test"], st, t_f1),
            "pixel_threshold_val": px_t, "pixel_dice_val": px_dice_val,
            "test_pixel": M.pixel_metrics(masks["test"], mt, px_t),
            "test_per_defect_type": per_type,
            "score_stats_val_good": {"mean": float(sv[y["val"] == 0].mean()), "max": float(sv[y["val"] == 0].max())},
        }
        scores_test[model.name], maps_test[model.name] = st, mt
        print(f"[{run_id}] {model.name}: test image AUROC={res['engines'][model.name]['test']['image_auroc']:.4f} "
              f"pixel AUROC={res['engines'][model.name]['test_pixel']['pixel_auroc']:.4f}", flush=True)

    # ---- engine selection on VALIDATION only: an engine that cannot separate normal from defective on val
    # (image AUROC < 0.90) is still reported but does not vote in the deployed decision.
    val_auc = {n: res["engines"][n]["val"]["image_auroc"] for n in res["engines"]}
    deployed = [n for n, a in val_auc.items() if a >= 0.90] or [max(val_auc, key=val_auc.get)]
    res["deployed_engines"] = deployed
    res["engine_selection"] = {"rule": "engines with validation image AUROC >= 0.90 vote; others are reported only",
                               "val_image_auroc": val_auc}
    # ---- fused decision on held-out test
    dec = fuse({n: scores_test[n] for n in deployed}, {n: thr[n] for n in deployed}, review_margin)
    auto = dec != 2
    yt = y["test"]
    res["fusion"] = {
        "rule": f"Deployed engines {deployed}: FAIL if all >= own threshold; PASS if all below; else REVIEW. "
                f"REVIEW also if any score within +/-{review_margin:.0%} of its threshold.",
        "review_margin": review_margin, "review_rate": float((dec == 2).mean()),
        "auto_decided": int(auto.sum()),
        "auto_accuracy": float((dec[auto] == yt[auto]).mean()) if auto.any() else None,
        "missed_defects_auto_pass": int(((dec == 0) & (yt == 1)).sum()),
        "false_rejects_auto_fail": int(((dec == 1) & (yt == 0)).sum()),
        "defects_sent_to_review": int(((dec == 2) & (yt == 1)).sum()),
    }

    # ---- Engine C: unfamiliar-pattern detection
    gv = np.concatenate([gn.score(load_batch(c)[0]) for c in batched(rows["val"], 16)])
    gn.threshold = float(gv.max() * 1.05)
    gt_ = np.concatenate([gn.score(load_batch(c)[0]) for c in batched(rows["test"], 16)])
    fr = foreign_rows(category, own_manifest=manifest)
    gf = np.concatenate([gn.score(load_batch(c)[0]) for c in batched(fr, 16)]) if fr else np.array([])
    res["unfamiliar_pattern"] = {
        "threshold": gn.threshold, "threshold_rule": "1.05 x max global kNN distance on calibration images",
        "test_in_distribution_flag_rate": float((gt_ > gn.threshold).mean()),
        "foreign_images": len(fr), "foreign_detection_rate": float((gf > gn.threshold).mean()) if len(gf) else None,
        "foreign_sources": sorted({r["path"].split("/")[2] for r in fr}),
        "train_images_used": len(rows["train"]),
    }

    # ---- latency (measured, single image, current CPU threads)
    x1, _ = load_batch(rows["test"][:1])
    lat = {}
    for model in (pc, pd):
        model.predict(x1)
        ts = []
        for _ in range(5):
            t = time.perf_counter(); model.predict(x1); ts.append(time.perf_counter() - t)
        lat[model.name] = float(np.median(ts) * 1000)
    res["latency_ms_median"] = lat
    res["threads"] = torch.get_num_threads()
    res["total_time_s"] = time.time() - t0
    res["completed_at"] = datetime.now(timezone.utc).isoformat()
    res["preprocess"] = ANOMALY_PP.to_dict()

    torch.save({"category": category, "backbone": backbone, "preprocess": ANOMALY_PP.to_dict(),
                "patchcore": pc.state(), "padim": pd.state(), "global_novelty": gn.state(),
                "thresholds": {k: float(v) for k, v in thr.items()},
                "pixel_thresholds": {n: res["engines"][n]["pixel_threshold_val"] for n in ("patchcore", "padim")},
                "review_margin": review_margin, "run_id": run_id, "dataset_version": man["version"],
                "deployed_engines": deployed,
                # reference normal score statistics for map normalisation in the UI
                "val_good_max": {n: res["engines"][n]["score_stats_val_good"]["max"] for n in ("patchcore", "padim")}},
               run_dir / "anomaly.pt")
    M.save_json(res, run_dir / "metrics.json")
    (run_dir / "status.json").write_text(json.dumps({"state": "completed", "run_id": run_id}))
    save_vis(rows["test"], maps_test["patchcore"], res["engines"]["patchcore"]["pixel_threshold_val"],
             VIS_DIR / run_id)
    print(f"[{run_id}] fusion={res['fusion']} unfamiliar={res['unfamiliar_pattern']} lat={lat}", flush=True)
    return res
