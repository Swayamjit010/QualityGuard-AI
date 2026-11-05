"""Model registry: discovers checkpoints under artifacts/checkpoints and keeps DB records in sync.

Lifecycle: development -> training -> evaluation -> validated -> approved -> deployed -> retired
(failed/stopped for unsuccessful runs).  Rules enforced here:
  * a run with held-out test metrics becomes 'validated' - never 'approved' or 'deployed' automatically;
  * only 'approved' (or previously deployed) versions can be deployed to a profile;
  * deploying records a ProfileRevision so it can be rolled back.
The only exception is first-time bootstrap: versions referenced by the seed product profiles are marked
'deployed' (audited as 'bootstrap deployment from seed configuration') so the platform is usable out of the box.
"""
from __future__ import annotations

import json
import logging

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.entities import EvaluationMetric, MLModel, ModelVersion, ProductProfile, User, utcnow
from app.services.audit import audit
from app.services.profiles import update_profile
from ml.config import CHECKPOINT_DIR, rel

log = logging.getLogger("qualityguard.registry")


def _read(p):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _flat_metrics(task: str, m: dict) -> dict[str, float]:
    out = {}
    if task == "classification":
        t = m.get("test", {})
        for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted", "roc_auc_macro_ovr",
                  "pr_auc_macro", "ece", "ece_uncalibrated", "temperature"):
            if t.get(k) is not None:
                out[k] = float(t[k])
        b = m.get("benchmark", {})
        for k in ("latency_ms_median", "latency_ms_p95", "throughput_img_s_batch16", "param_size_mb",
                  "rss_delta_mb_inference"):
            if b.get(k) is not None:
                out[k] = float(b[k])
        out["train_time_s"] = float(m.get("train_time_s", 0))
        out["checkpoint_size_mb"] = float(m.get("checkpoint_size_mb", 0))
        # worst-class recall = the most relevant "missed defect" indicator for multi-class
        pc = t.get("per_class", {})
        if pc:
            out["min_class_recall"] = min(v["recall"] for v in pc.values())
    elif task == "segmentation":
        for k in ("image_auroc", "image_ap", "accuracy", "precision", "recall", "f1", "false_negative_rate",
                  "false_positive_rate"):
            if m.get("test", {}).get(k) is not None:
                out[f"image_{k}" if not k.startswith("image_") else k] = float(m["test"][k])
        tp = m.get("test_pixel", {})
        for k in ("pixel_auroc", "pixel_ap", "mean_dice_defective", "mean_iou_defective"):
            if tp.get(k) is not None:
                out[k] = float(tp[k])
        if m.get("latency_ms_median") is not None:
            out["latency_ms_median"] = float(m["latency_ms_median"])
        out["train_time_s"] = float(m.get("train_time_s", 0))
    elif task == "component_recognition":
        t = m.get("test", {})
        out["top1_accuracy"] = float(t.get("top1_accuracy", 0))
        out["false_reject_rate"] = float(t.get("false_reject_rate", 0))
        out["accepted_but_wrong_domain"] = float(t.get("accepted_but_wrong_domain", 0))
        for k, v in m.get("foreign_probes", {}).items():
            out[f"rejected_rate_{k}"] = float(v["rejected_rate"])
    else:
        for eng, e in m.get("engines", {}).items():
            t = e.get("test", {})
            for k, name in (("image_auroc", "image_auroc"), ("recall", "recall"), ("precision", "precision"),
                            ("false_negative_rate", "false_negative_rate"),
                            ("false_positive_rate", "false_positive_rate")):
                if t.get(k) is not None:  # custom products without defect photos have no AUROC/recall
                    out[f"{eng}_{name}"] = float(t[k])
            if e.get("defect_detection_rate") is not None:
                out[f"{eng}_defect_detection_rate"] = float(e["defect_detection_rate"])
            if e.get("test_pixel", {}).get("pixel_auroc") is not None:
                out[f"{eng}_pixel_auroc"] = e["test_pixel"]["pixel_auroc"]
            if e.get("test_pixel", {}).get("mean_dice_defective") is not None:
                out[f"{eng}_dice"] = e["test_pixel"]["mean_dice_defective"]
        f = m.get("fusion", {})
        if f.get("auto_accuracy") is not None:
            out["fusion_auto_accuracy"] = f["auto_accuracy"]
        out["fusion_review_rate"] = f.get("review_rate", 0)
        for k, v in m.get("latency_ms_median", {}).items():
            out[f"{k}_latency_ms"] = v
    return out


def sync(db: Session) -> dict:
    added, updated = [], []
    for d in sorted(p for p in CHECKPOINT_DIR.iterdir() if p.is_dir()):
        status = _read(d / "status.json") or {}
        metrics = _read(d / "metrics.json")
        cfg = _read(d / "config.json") or {}
        is_anom = (d / "anomaly.pt").exists() or "-anomaly-" in d.name
        is_seg = (d / "segmentation.pt").exists() or "-segmentation-" in d.name
        is_rec = (d / "recognizer.pt").exists() or d.name.startswith("component_recognizer")
        task = ("segmentation" if is_seg else "component_recognition" if is_rec else
                "anomaly_detection" if is_anom else "classification")
        ck_name = {"segmentation": "segmentation.pt", "component_recognition": "recognizer.pt",
                   "anomaly_detection": "anomaly.pt"}.get(task, "model.pt")
        arch = (metrics or {}).get("backbone") if is_anom else ((metrics or {}).get("arch") or cfg.get("arch"))
        arch = (f"patchcore+padim/{arch or 'resnet18'}" if is_anom else "unet/resnet18" if is_seg else
                "logreg+knn-reject/resnet18" if is_rec else (arch or "unknown"))
        name = d.name.rsplit("-v", 1)[0]
        ver = d.name.rsplit("-v", 1)[1] if "-v" in d.name else "1"
        m = db.query(MLModel).filter_by(name=name).first()
        if not m:
            m = MLModel(name=name, task=task, architecture=arch)
            db.add(m)
            db.flush()
        mv = db.query(ModelVersion).filter_by(run_id=d.name).first()
        state = status.get("state")
        new_status = {"running": "training", "failed": "failed", "stopped": "failed"}.get(state)
        if metrics and (d / ck_name).exists():
            new_status = "validated"
        new_status = new_status or "development"
        if not mv:
            ck = d / ck_name
            mv = ModelVersion(model_id=m.id, run_id=d.name, version=ver, status=new_status,
                              dataset_name=(metrics or {}).get("dataset") or (f"mvtec_{metrics['category']}"
                                                                                if metrics and is_anom and metrics.get("category")
                                                                                else cfg.get("dataset", "")),
                              dataset_version=(metrics or {}).get("dataset_version", cfg.get("dataset_version", "")),
                              input_size=((metrics or {}).get("preprocess") or {}).get("input_size",
                                                                                     _read(d / "preprocess.json") and
                                                                                     _read(d / "preprocess.json")["input_size"] or 224),
                              class_labels=(metrics or {}).get("classes", cfg.get("classes", ["good", "anomalous"]
                                                                                   if is_anom else [])),
                              checkpoint_path=rel(ck), checkpoint_size_mb=ck.stat().st_size / 1e6 if ck.exists() else 0,
                              preprocess=_read(d / "preprocess.json") or (metrics or {}).get("preprocess", {}),
                              training_config=cfg)
            db.add(mv)
            db.flush()
            added.append(d.name)
        elif mv.status in ("development", "training", "evaluation", "failed") and new_status != mv.status:
            mv.status = new_status
            ck = d / ck_name
            if ck.exists():
                mv.checkpoint_size_mb = ck.stat().st_size / 1e6
            if metrics:
                mv.class_labels = metrics.get("classes", mv.class_labels)
                mv.dataset_version = metrics.get("dataset_version", mv.dataset_version)
            updated.append(d.name)
        if metrics and not mv.metrics:
            for k, v in _flat_metrics(task, metrics).items():
                db.add(EvaluationMetric(model_version_id=mv.id, split="test", name=k, value=float(v)))
    db.commit()
    return {"added": added, "updated": updated}


def bootstrap_deployments(db: Session):
    for p in db.query(ProductProfile).all():
        mc = p.config.get("models", {})
        ids = list(mc.get("classifiers", [])) + [mc[k] for k in ("anomaly", "segmentation") if mc.get(k)]
        for rid in ids:
            mv = db.query(ModelVersion).filter_by(run_id=rid).first()
            if mv and mv.status == "validated" and not mv.approved_at:
                mv.status, mv.approved_at = "deployed", utcnow()
                mv.notes = (mv.notes + "\n" if mv.notes else "") + f"Bootstrap deployment from seed profile '{p.id}'."
                audit(db, None, "model.bootstrap_deploy", "model_version", rid, {"profile": p.id}, commit=False)
    db.commit()


def metrics_file(mv: ModelVersion) -> dict | None:
    return _read(CHECKPOINT_DIR / mv.run_id / "metrics.json")


def set_status(db: Session, mv: ModelVersion, status: str, user: User, note: str = ""):
    allowed = {
        "approved": {"validated", "retired"},
        "retired": {"validated", "approved", "deployed", "failed", "development"},
        "validated": {"approved", "retired"},
    }
    if status not in allowed or mv.status not in allowed[status]:
        raise HTTPException(409, f"Cannot move model version from '{mv.status}' to '{status}'")
    if status == "retired" and _in_use(db, mv.run_id):
        raise HTTPException(409, f"Model is active in profile(s) {_in_use(db, mv.run_id)}; deploy a replacement first")
    prev = mv.status
    mv.status = status
    if status == "approved":
        mv.approved_by, mv.approved_at = user.id, utcnow()
    if note:
        mv.notes = (mv.notes + "\n" if mv.notes else "") + note
    audit(db, user, f"model.{status}", "model_version", mv.run_id, {"from": prev, "note": note})
    return mv


def _in_use(db: Session, run_id: str) -> list[str]:
    out = []
    for p in db.query(ProductProfile).all():
        m = p.config.get("models", {})
        if run_id in m.get("classifiers", []) or run_id in (m.get("anomaly"), m.get("segmentation")):
            out.append(p.id)
    return out


def deploy(db: Session, mv: ModelVersion, profile: ProductProfile, user: User, mode: str = "primary") -> ProductProfile:
    """mode: primary (classifier becomes first ensemble member) | ensemble (append) | replace (sole member)."""
    if mv.status not in ("approved", "deployed"):
        raise HTTPException(409, "Only approved model versions can be deployed. Approve it first.")
    want = profile.config.get("dataset")
    if want and mv.dataset_name and mv.dataset_name != want:
        raise HTTPException(409, f"Model was trained on '{mv.dataset_name}' but profile expects '{want}'")
    cfg = json.loads(json.dumps(profile.config))
    models = cfg.setdefault("models", {})
    if mv.model.task == "anomaly_detection":
        models["anomaly"] = mv.run_id
    else:
        cl = [c for c in models.get("classifiers", []) if c != mv.run_id]
        models["classifiers"] = [mv.run_id] if mode == "replace" else ([mv.run_id] + cl if mode == "primary"
                                                                       else cl + [mv.run_id])
    p = update_profile(db, profile.id, cfg, user, f"deploy {mv.run_id} ({mode})")
    mv.status = "deployed"
    _refresh_deployed_flags(db)
    audit(db, user, "model.deploy", "model_version", mv.run_id, {"profile": profile.id, "mode": mode,
                                                                 "profile_version": p.version})
    return p


def _refresh_deployed_flags(db: Session):
    used = set()
    for p in db.query(ProductProfile).all():
        m = p.config.get("models", {})
        used |= set(m.get("classifiers", [])) | ({m["anomaly"]} if m.get("anomaly") else set())
    for mv in db.query(ModelVersion).filter(ModelVersion.status == "deployed").all():
        if mv.run_id not in used:
            mv.status = "approved"
    db.commit()
