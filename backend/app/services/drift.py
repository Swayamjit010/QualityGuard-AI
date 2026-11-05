"""Drift monitoring.

Distinguishes three things that are often conflated:
  * data drift        - input image statistics differ from the reference (training/calibration) images
  * prediction drift  - predicted class / anomaly-score distributions differ from the reference
  * performance degradation - verified (inspector-confirmed) accuracy declines. Only this one says the model is
    actually getting worse; drift alone only says conditions changed.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sqlalchemy.orm import Session

from app.models.entities import DriftCheck, Inspection, ProductProfile
from app.services import alerts
from ml.config import absp
from ml.datasets.analysis import read_image
from ml.datasets.splitting import load_manifest
from ml.inference.validation import validate_image

_REF_CACHE: dict[str, dict] = {}


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    r = np.histogram(ref, edges)[0] / len(ref) + 1e-4
    c = np.histogram(np.clip(cur, edges[0], edges[-1]), edges)[0] / len(cur) + 1e-4
    return float(np.sum((c - r) * np.log(c / r)))


def reference_stats(profile: dict, limit: int = 300) -> dict:
    ds = profile.get("dataset")
    if ds in _REF_CACHE:
        return _REF_CACHE[ds]
    man = load_manifest(ds)
    rows = [r for r in man["samples"] if r["split"] in ("train", "val")]
    rng = np.random.default_rng(0)
    rows = [rows[i] for i in rng.choice(len(rows), min(limit, len(rows)), replace=False)]
    feats = {"brightness": [], "contrast": [], "sharpness": []}
    labels = []
    for r in rows:
        _, _, info = validate_image(read_image(absp(r["path"])), None)
        for k in feats:
            feats[k].append(info[k])
        labels.append(r["label"])
    out = {"features": {k: np.array(v) for k, v in feats.items()}, "labels": labels, "n": len(rows)}
    _REF_CACHE[ds] = out
    return out


def check(db: Session, profile: ProductProfile, window: int = 200) -> DriftCheck:
    cfg = profile.config
    recent = db.query(Inspection).filter(Inspection.profile_id == profile.id, Inspection.input_brightness.isnot(None)) \
        .order_by(Inspection.created_at.desc()).limit(window).all()
    details: dict = {"window": window, "n_recent": len(recent)}
    if len(recent) < 20:
        details["interpretation"] = f"Insufficient data: {len(recent)} recent inspections (need >= 20)."
        chk = DriftCheck(profile_id=profile.id, window=window, data_drift=False, prediction_drift=False,
                         performance_degradation=None, details=details)
        db.add(chk); db.commit()
        return chk
    ref = reference_stats(cfg)
    details["reference"] = {"source": f"{cfg.get('dataset')} train+val images", "n": ref["n"]}
    cur = {"brightness": np.array([r.input_brightness for r in recent]),
           "contrast": np.array([r.input_contrast for r in recent]),
           "sharpness": np.array([r.input_sharpness for r in recent])}
    feat = {}
    data_drift = False
    for k, v in cur.items():
        ks = stats.ks_2samp(ref["features"][k], v)
        p = psi(ref["features"][k], v)
        drift = bool(ks.pvalue < 0.01 and p > 0.2)
        data_drift |= drift
        feat[k] = {"ks": float(ks.statistic), "p_value": float(ks.pvalue), "psi": p, "drift": drift,
                   "ref_mean": float(ref["features"][k].mean()), "cur_mean": float(v.mean())}
    details["input_features"] = feat

    pred_drift = False
    classes = [r.defect_class for r in recent if r.defect_class]
    if cfg.get("models", {}).get("classifiers") and classes:
        labels = sorted(set(ref["labels"]))
        rc = np.array([ref["labels"].count(c) for c in labels], float)
        cc = np.array([classes.count(c) for c in labels], float)
        exp = rc / rc.sum() * cc.sum()
        chi = stats.chisquare(cc + 0.5, exp + 0.5 * len(labels) * exp / exp.sum())
        pred_drift = bool(chi.pvalue < 0.01)
        details["class_frequency"] = {"labels": labels, "reference": rc.tolist(), "recent": cc.tolist(),
                                      "chi2_p_value": float(chi.pvalue)}
        conf = np.array([r.confidence for r in recent if r.confidence is not None])
        details["confidence"] = {"mean": float(conf.mean()), "p10": float(np.percentile(conf, 10))}
    ratios = np.array([r.anomaly_score_ratio for r in recent if r.anomaly_score_ratio is not None])
    if len(ratios):
        details["anomaly_score_ratio"] = {"mean": float(ratios.mean()), "median": float(np.median(ratios)),
                                          "share_above_threshold": float((ratios >= 1).mean())}
    unf = np.mean([r.unfamiliar for r in recent])
    details["unfamiliar_rate"] = float(unf)
    pred_drift |= bool(unf > 0.15)

    verified = [r for r in recent if r.feedback]
    perf = None
    if len(verified) >= 20:
        half = len(verified) // 2
        acc = lambda rs: float(np.mean([r.feedback[-1].verdict == "confirm" for r in rs]))  # noqa: E731
        newer, older = acc(verified[:half]), acc(verified[half:])
        perf = bool(newer < older - 0.1)
        details["verified_performance"] = {"n": len(verified), "recent_agreement": newer, "earlier_agreement": older}
    interp = []
    interp.append("Input statistics shifted vs reference images." if data_drift else "No significant input drift.")
    interp.append("Prediction distribution shifted." if pred_drift else "Prediction distribution stable.")
    interp.append("Insufficient verified feedback to assess performance." if perf is None else
                  ("Verified agreement declined — investigate model performance." if perf else
                   "Verified agreement stable — drift (if any) has not reduced verified accuracy."))
    details["interpretation"] = " ".join(interp)
    chk = DriftCheck(profile_id=profile.id, window=window, data_drift=data_drift, prediction_drift=pred_drift,
                     performance_degradation=perf, details=details)
    db.add(chk)
    db.commit()
    if data_drift or pred_drift or perf:
        alerts.raise_system_alert(db, "model_drift", "warning", f"Drift detected on {profile.id}",
                                  details["interpretation"], {"profile_id": profile.id, "drift_check_id": chk.id})
    return chk
