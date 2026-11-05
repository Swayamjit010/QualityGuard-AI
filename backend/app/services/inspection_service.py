"""Runs the ML pipeline for one image and persists every traceable artefact."""
from __future__ import annotations

import hashlib
import logging
import threading

import numpy as np
from sqlalchemy.orm import Session

from app.models.entities import (AnomalyResult, ImageAsset, Inspection, Prediction, SegmentationResult,
                                 SeverityAssessment, User)
from app.services import alerts
from app.services.events import bus
from app.services.profiles import get_profile
from app.services.storage import inspection_dir, safe_name, storage
from ml.inference.pipeline import CACHE, RECOGNIZER_RUN, ModelUnavailable, inspect
from ml.inference.validation import decode_image
from ml.quality.rules import VERDICT_TEXT, verdict_for
from ml.recognition.router import gate

log = logging.getLogger("qualityguard.inspection")
_PIPE_LOCK = threading.Semaphore(2)  # bound concurrent CPU-heavy inferences

IMAGE_KEYS_ORDER = ["original", "bounding_boxes", "segmentation_overlay", "segmentation_mask",
                    "anomaly_patchcore_overlay", "anomaly_patchcore_heatmap", "anomaly_padim_overlay",
                    "anomaly_padim_heatmap", "segmentation_unet_overlay", "segmentation_unet_heatmap", "gradcam_overlay", "gradcam_heatmap", "gradcam_pp_overlay",
                    "gradcam_pp_heatmap", "scorecam_overlay", "scorecam_heatmap", "defect_zoom", "comparison"]


AUTO = "auto"


def recognize(img, requested: str | None, db: Session) -> tuple[str, dict]:
    """Component recognition + routing.

    * ``requested`` is None/"auto": the recogniser picks the profile. Unsupported -> inspection is still run with the
      closest category's models (shown for reference) but the verdict is forced to MANUAL INSPECTION REQUIRED.
    * explicit profile: the recogniser checks that the image plausibly shows that category; a clear mismatch or an
      unsupported image also forces MANUAL INSPECTION REQUIRED. Custom products (not covered by the recogniser) are
      not checked.
    """
    from app.models.entities import ProductProfile
    auto = requested in (None, "", AUTO)
    if img is None:
        if not auto:
            return requested, {"mode": "manual", "available": True, "note": "Image could not be decoded; not recognised."}
        fallback = next((p.id for p in db.query(ProductProfile).filter_by(is_active=True).order_by(ProductProfile.id)), None)
        return fallback, {"mode": "auto", "available": True, "supported": False,
                          "note": "Image could not be decoded, so no category could be recognised."}
    try:
        rec = CACHE.get("recognizer", RECOGNIZER_RUN).recognize(img)
    except ModelUnavailable as exc:
        if auto:
            raise
        return requested, {"mode": "manual", "available": False, "note": f"Component recogniser unavailable: {exc}"}
    rec["available"] = True
    covered = set(CACHE.get("recognizer", RECOGNIZER_RUN).rec.domain_profiles.values())
    pid, rec = gate(rec, requested, covered)
    if auto:
        prof = db.get(ProductProfile, pid)
        if prof is None or not prof.is_active:
            raise ModelUnavailable(f"Recognised category '{rec['title']}' maps to profile '{pid}', which is not installed.")
    return pid, rec


def verdict_of(i: Inspection) -> str:
    return verdict_for(i.status, i.defect_present)


def run_inspection(db: Session, *, data: bytes, filename: str, profile_id: str, user: User | None,
                   source: str = "single", batch_id: str | None = None, line_id: str | None = None,
                   product_serial: str = "", options: dict | None = None, ground_truth: str | None = None,
                   is_simulated: bool = False, video_timestamp_s: float | None = None,
                   raise_alerts: bool = True) -> Inspection:
    img = decode_image(data)
    options = dict(options or {})
    t_rec = __import__("time").perf_counter()
    if options.pop("recognize", True):
        profile_id, recog = recognize(img, profile_id, db)
        recog["latency_ms"] = (__import__("time").perf_counter() - t_rec) * 1000
        options["recognition"] = recog
    if not profile_id:
        raise ModelUnavailable("No active product profile is installed.")
    prof = get_profile(db, profile_id)
    cfg = prof.config
    with _PIPE_LOCK:
        try:
            res = inspect(img, cfg, options)
        except ModelUnavailable:
            raise
        except Exception as exc:
            log.exception("inspection pipeline failure")
            alerts.raise_system_alert(db, "inspection_failure", "critical", "Inspection pipeline failure",
                                      f"{profile_id}/{safe_name(filename)}: {exc}", {"profile_id": profile_id})
            raise

    insp = Inspection(profile_id=profile_id, profile_version=prof.version, user_id=user.id if user else None,
                      batch_id=batch_id, line_id=line_id, product_serial=product_serial[:128], source=source,
                      filename=safe_name(filename), ground_truth=ground_truth, is_simulated=is_simulated,
                      video_timestamp_s=video_timestamp_s, status="PENDING")
    db.add(insp)
    db.flush()
    adir = inspection_dir(insp.id)
    insp.artifact_dir = adir
    ext = (safe_name(filename).rsplit(".", 1)[-1].lower() if "." in filename else "bin")[:5]
    raw_key = storage.put_bytes(f"{adir}/upload.{ext}", data)
    images = res.pop("_images", {})
    emb = res.pop("_embedding", None)
    saved = {}
    for k in IMAGE_KEYS_ORDER + [k for k in images if k not in IMAGE_KEYS_ORDER]:
        if k in images:
            saved[k] = storage.put_image(f"{adir}/{k}.png", images[k])
    res["visualizations"] = list(saved)

    dec = res["decision"]
    info = res.get("input", {})
    asset = ImageAsset(path=raw_key, sha256=hashlib.sha256(data).hexdigest(), width=info.get("width", 0),
                       height=info.get("height", 0), original_filename=safe_name(filename))
    db.add(asset)
    db.flush()
    clf = res.get("engines", {}).get("classification")
    an = res.get("engines", {}).get("anomaly")
    nov = res.get("engines", {}).get("novelty")
    insp.image_id = asset.id
    insp.status = dec["status"]
    insp.defect_present = bool(res.get("defect_present"))
    insp.defect_class = res.get("defect_class")
    insp.severity = res.get("severity", {}).get("level", "none")
    insp.quality_index = dec.get("quality_index")
    insp.confidence = clf["confidence"] if clf else None
    insp.anomaly_score_ratio = an["score_ratio"][an["primary_engine"]] if an else None
    insp.reliability = res.get("reliability", {}).get("level", "")
    insp.unfamiliar = bool(res.get("unfamiliar"))
    insp.manual_review_required = bool(dec.get("manual_review_required"))
    insp.review_status = "pending" if insp.manual_review_required else "not_required"
    insp.latency_ms = res.get("timings_ms", {}).get("total", 0)
    insp.models_used = res.get("models_used", [])
    insp.input_brightness, insp.input_contrast = info.get("brightness"), info.get("contrast")
    insp.input_sharpness = info.get("sharpness")
    if emb is not None:
        insp.embedding = np.asarray(emb, np.float32).tobytes()
        insp.embedding_model = res.get("embedding_model")
    insp.result = _jsonable(res)

    if clf:
        for m in clf["ensemble"]:
            db.add(Prediction(inspection_id=insp.id, model_run_id=m["run_id"], predicted_class=m["top_class"],
                              confidence=m["confidence"], probabilities=m["probabilities"],
                              ood_score=m["mahalanobis"], is_ood=m["ood"]))
    if an:
        db.add(AnomalyResult(inspection_id=insp.id, model_run_id=an["model"], scores=an["scores"],
                             thresholds=an["thresholds"], is_anomalous=an["is_anomalous"],
                             engines_disagree=an["engines_disagree"], novelty_score=nov and nov["score"],
                             novelty_threshold=nov and nov["threshold"]))
    if res.get("measurements"):
        m = res["measurements"]
        db.add(SegmentationResult(inspection_id=insp.id, method=f"{an['primary_engine']} anomaly-map threshold",
                                  num_regions=m["num_regions"], total_area_px=m["total_defect_area_px"],
                                  area_ratio=m["defect_area_ratio"], calibrated=m["calibrated"],
                                  measurements=_jsonable(m)))
    db.add(SeverityAssessment(inspection_id=insp.id, ai_level=insp.severity,
                              basis=res.get("severity", {}).get("basis", []), rule_status=insp.status,
                              rules_applied=dec.get("rules_applied", [])))
    if raise_alerts:
        alerts.evaluate_inspection(db, insp)
    db.commit()
    bus.publish("inspection", summary(insp))
    return insp


def summary(i: Inspection) -> dict:
    v = verdict_of(i)
    return {"id": i.id, "created_at": i.created_at.isoformat() + "Z", "profile_id": i.profile_id, "status": i.status,
            "verdict": v, "verdict_text": VERDICT_TEXT[v],
            "defect_present": i.defect_present, "defect_class": i.defect_class, "severity": i.severity,
            "quality_index": i.quality_index, "confidence": i.confidence,
            "anomaly_score_ratio": i.anomaly_score_ratio, "reliability": i.reliability, "unfamiliar": i.unfamiliar,
            "manual_review_required": i.manual_review_required, "review_status": i.review_status,
            "human_decision": i.human_decision, "latency_ms": i.latency_ms, "source": i.source,
            "filename": i.filename, "batch_id": i.batch_id, "line_id": i.line_id, "product_serial": i.product_serial,
            "is_simulated": i.is_simulated, "ground_truth": i.ground_truth, "video_timestamp_s": i.video_timestamp_s,
            "models_used": i.models_used}


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o
