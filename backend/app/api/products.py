"""Products: everything the system can inspect, and "Add a product" - train an anomaly model for a new kind of item
from photos of good examples (defective photos optional, used only to report detection rate).

Lifecycle of a custom product:
  upload photos -> training job (out-of-fold calibrated PatchCore + PaDiM + novelty) -> model version registered
  as *validated* -> a user with ``approve_models`` approves & activates it, which creates the product profile ->
  the product appears in Inspect. Nothing is deployed automatically.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.api.inspections import IMAGE_EXT, read_upload
from app.core.config import get_settings
from app.database.session import get_db
from app.models.entities import (ModelVersion, Product, ProductProfile, ProfileRevision, TrainingJob, User, utcnow)
from app.security.auth import require
from app.services import model_registry
from app.services.audit import audit
from app.services.profiles import validate_profile
from app.workers.jobs import start_training
from ml.anomaly_detection.custom import MIN_GOOD, list_images, product_dir
from ml.config import CHECKPOINT_DIR, RAW_DIR
from ml.inference.validation import decode_image

router = APIRouter(prefix="/api/products", tags=["products"])
settings = get_settings()
SLUG_RE = re.compile(r"^[a-z0-9_]{2,40}$")
MAX_GOOD, MAX_DEFECT = 500, 200


def _slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:40]
    return s if len(s) >= 2 else "product"


def _profile_id(slug: str) -> str:
    return f"custom_{slug}"


def _runs(slug: str) -> list[Path]:
    runs = sorted(CHECKPOINT_DIR.glob(f"custom_{slug}-anomaly-resnet18-v*"),
                  key=lambda p: int(p.name.rsplit("-v", 1)[1]) if p.name.rsplit("-v", 1)[1].isdigit() else 0)
    return runs


def _read(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def custom_out(slug: str, db: Session) -> dict:
    d = product_dir(slug)
    meta = _read(d / "product.json")
    runs = _runs(slug)
    run = runs[-1] if runs else None
    st = _read(run / "status.json") if run else {}
    m = _read(run / "metrics.json") if run else {}
    prof = db.get(ProductProfile, _profile_id(slug))
    active_run = (prof.config.get("models") or {}).get("anomaly") if prof else None
    mv = db.query(ModelVersion).filter_by(run_id=run.name).first() if run else None
    job = db.query(TrainingJob).filter(TrainingJob.kind == "anomaly_custom", TrainingJob.status.in_(("queued", "running")),
                                       TrainingJob.run_id.like(f"custom_{slug}-anomaly-%")).first()
    if job and (not run or run.name != job.run_id):  # the new run folder may not exist yet
        run, st, m = None, {"state": "running", "progress": 0.0, "message": "starting…"}, (_read(runs[-1] / "metrics.json") if runs else {})
    state = st.get("state") or ("not started" if not run else "unknown")
    if state == "completed":
        state = "ready" if prof and active_run == run.name else "trained"
    summary = None
    if m:
        summary = {"good_photos": m.get("split_sizes", {}).get("good"), "defect_photos": m.get("split_sizes", {}).get("defect_examples"),
                   "calibration_folds": m.get("split_sizes", {}).get("calibration_folds"),
                   "defect_detection_rate": {n: e.get("defect_detection_rate") for n, e in m.get("engines", {}).items()},
                   "good_false_reject_rate": {n: e.get("test", {}).get("false_positive_rate") for n, e in m.get("engines", {}).items()},
                   "fusion": m.get("fusion"), "unfamiliar": m.get("unfamiliar_pattern"), "warnings": m.get("warnings", []),
                   "training_seconds": round(m.get("total_time_s", 0)), "latency_ms": m.get("latency_ms_median")}
    good = list_images(d / "good")
    return {"id": _profile_id(slug), "slug": slug, "kind": "custom", "name": meta.get("name", slug),
            "description": meta.get("description", ""), "created_at": meta.get("created_at"),
            "state": state, "stage": st.get("stage"), "progress": st.get("progress"), "message": st.get("message"),
            "run_id": run.name if run else (job.run_id if job else None), "model_version_id": mv.id if mv else None,
            "inspectable": bool(prof and active_run),
            "model_status": mv.status if mv else None, "active_run": active_run, "profile_id": prof.id if prof else None,
            "counts": {"good": len(good), "defect": len(list_images(d / "defect"))},
            "thumbnails": [f"/api/products/{slug}/image/good/{p.name}" for p in good[:6]], "summary": summary}


@router.get("")
def list_products(_: User = Depends(require("view")), db: Session = Depends(get_db)):
    out = []
    custom_ids = set()
    for d in sorted(RAW_DIR.glob("custom_*")):
        if (d / "product.json").exists():
            slug = d.name[len("custom_"):]
            custom_ids.add(_profile_id(slug))
            out.append(custom_out(slug, db))
    for p in db.query(ProductProfile).filter_by(is_active=True).order_by(ProductProfile.id):
        if p.id in custom_ids:
            continue
        c = p.config or {}
        m = c.get("models", {})
        out.append({"id": p.id, "kind": "built-in", "name": c.get("name", p.id), "description": c.get("description", ""),
                    "state": "ready", "profile_id": p.id, "defect_classes": c.get("defect_classes", []),
                    "engines": [x for x in ("classifier" if m.get("classifiers") else "", "anomaly" if m.get("anomaly") else "") if x]})
    return out


@router.get("/{slug}")
def get_product(slug: str, _: User = Depends(require("view")), db: Session = Depends(get_db)):
    if not SLUG_RE.match(slug) or not (product_dir(slug) / "product.json").exists():
        raise HTTPException(404, "Product not found")
    return custom_out(slug, db)


@router.get("/{slug}/image/{kind}/{name}")
def product_image(slug: str, kind: str, name: str, _: User = Depends(require("view"))):
    if not SLUG_RE.match(slug) or kind not in ("good", "defect"):
        raise HTTPException(404, "Not found")
    base = (product_dir(slug) / kind).resolve()
    p = (base / name).resolve()
    if base not in p.parents or not p.is_file():
        raise HTTPException(404, "Not found")
    return FileResponse(p)


async def _save_images(files: list[UploadFile], dest: Path, start: int = 0) -> tuple[int, list[str]]:
    dest.mkdir(parents=True, exist_ok=True)
    saved, rejected = 0, []
    for i, f in enumerate(files):
        try:
            data = await read_upload(f, settings.max_upload_mb, IMAGE_EXT)
            img = decode_image(data)
            if img is None or min(img.shape[:2]) < 32:
                raise ValueError("unreadable or tiny")
        except Exception:
            rejected.append(f.filename or f"file {i + 1}")
            continue
        ext = (f.filename or "x.png").rsplit(".", 1)[-1].lower()
        (dest / f"{start + saved:04d}.{ext}").write_bytes(data)
        saved += 1
    return saved, rejected


def _start_job(db: Session, slug: str, user: User) -> str:
    if db.query(TrainingJob).filter(TrainingJob.status.in_(("queued", "running"))).count():
        raise HTTPException(409, "Another training job is running (one at a time on CPU). Try again when it finishes.")
    n = len(_runs(slug))
    run_id = f"custom_{slug}-anomaly-resnet18-v{n + 1}"
    tj = TrainingJob(run_id=run_id, kind="anomaly_custom", config={"slug": slug}, created_by=user.id)
    db.add(tj)
    db.commit()
    start_training(tj.id, "anomaly_custom", {"slug": slug})
    return run_id


@router.post("", status_code=201)
async def create_product(request: Request, name: str = Form(...), description: str = Form(""),
                         good: list[UploadFile] = File(...), defect: list[UploadFile] = File(default=[]),
                         user: User = Depends(require("train")), db: Session = Depends(get_db)):
    """Create a product from photos of good items (>= 10, ideally 30+) and optional defective photos, then start
    training its anomaly model in the background."""
    name = name.strip()
    if not (2 <= len(name) <= 64):
        raise HTTPException(422, "Name must be 2-64 characters")
    if len(good) > MAX_GOOD or len(defect) > MAX_DEFECT:
        raise HTTPException(413, f"At most {MAX_GOOD} good and {MAX_DEFECT} defective photos per upload")
    slug = _slugify(name)
    base, i = slug, 2
    while (product_dir(slug)).exists() or db.get(ProductProfile, slug) or db.get(ProductProfile, _profile_id(slug)):
        slug = f"{base[:36]}_{i}"
        i += 1
    if db.query(TrainingJob).filter(TrainingJob.status.in_(("queued", "running"))).count():
        raise HTTPException(409, "Another training job is running (one at a time on CPU). Try again when it finishes.")
    d = product_dir(slug)
    try:
        n_good, bad_good = await _save_images(good, d / "good")
        n_def, bad_def = await _save_images(defect, d / "defect")
        if n_good < MIN_GOOD:
            raise HTTPException(422, f"Need at least {MIN_GOOD} readable photos of good items (got {n_good}"
                                     f"{', ' + str(len(bad_good)) + ' unreadable' if bad_good else ''}). 30+ is better.")
        (d / "product.json").write_text(json.dumps({
            "slug": slug, "name": name, "description": description.strip()[:500], "created_by": user.email,
            "created_at": utcnow().isoformat() + "Z"}, indent=1), encoding="utf-8")
    except HTTPException:
        import shutil
        shutil.rmtree(d, ignore_errors=True)
        raise
    run_id = _start_job(db, slug, user)
    audit(db, user, "product.create", "product", slug, {"name": name, "good": n_good, "defect": n_def,
                                                         "run_id": run_id}, request)
    return {**custom_out(slug, db), "rejected_files": bad_good + bad_def}


@router.post("/{slug}/photos", status_code=202)
async def add_photos(slug: str, request: Request, good: list[UploadFile] = File(default=[]),
                     defect: list[UploadFile] = File(default=[]), user: User = Depends(require("train")),
                     db: Session = Depends(get_db)):
    """Add more photos and retrain as a new model version (the active version keeps serving until the new one is
    approved and activated)."""
    d = product_dir(slug)
    if not SLUG_RE.match(slug) or not (d / "product.json").exists():
        raise HTTPException(404, "Product not found")
    n_good, bad_good = await _save_images(good, d / "good", start=len(list_images(d / "good")))
    n_def, bad_def = await _save_images(defect, d / "defect", start=len(list_images(d / "defect")))
    run_id = _start_job(db, slug, user)
    audit(db, user, "product.retrain", "product", slug, {"good_added": n_good, "defect_added": n_def, "run_id": run_id},
          request)
    return {**custom_out(slug, db), "rejected_files": bad_good + bad_def}


def _profile_config(slug: str, meta: dict, run_id: str) -> dict:
    return {
        "id": _profile_id(slug), "version": 1, "name": meta.get("name", slug), "description": meta.get("description", ""),
        "product_type": "object", "dataset": f"custom_{slug}",
        "configuration_note": "Created from photos of good items on the 'Add a product' page. Severity bands and the "
                              "zero-tolerance rule are defaults - adjust them to your own acceptance criteria.",
        "assumes_defect_present": False, "defect_classes": [], "normal_classes": ["good"],
        "image": {"min_side_px": 64, "min_contrast": 2.0, "expected_color_mode": "color"},
        "calibration": {"mm_per_pixel": None}, "min_region_area_px": 60,
        "models": {"classifiers": [], "anomaly": run_id},
        "pipeline": {"engines": ["anomaly", "novelty"], "primary_anomaly_engine": "patchcore", "cam_methods": []},
        "severity_rules": {"default_level": "medium", "area_metric": "defect_area_ratio",
                           "area_levels": [{"level": "low", "max": 0.005}, {"level": "medium", "max": 0.02},
                                           {"level": "high", "max": 0.05}, {"level": "critical", "max": None}],
                           "count_escalation": {"regions_ge": 3, "escalate_by": 1},
                           "intensity_escalation": {"score_ratio_ge": 2.0, "escalate_by": 1}},
        "acceptance_rules": {"fail_at_severity": "low", "accept_below_fail_level": True, "rejected_classes": [],
                             "unfamiliar_action": "INCONCLUSIVE", "disagreement_action": "REVIEW REQUIRED"},
        "quality_index": {"deductions": {"none": 0, "low": 10, "medium": 35, "high": 65, "critical": 100}, "review_cap": 60},
    }


@router.post("/{slug}/activate")
def activate(slug: str, request: Request, user: User = Depends(require("approve_models")), db: Session = Depends(get_db)):
    """Approve the latest trained model of a custom product and make the product available for inspection
    (creates or updates its profile - recorded as a profile revision, so it can be rolled back)."""
    d = product_dir(slug)
    if not SLUG_RE.match(slug) or not (d / "product.json").exists():
        raise HTTPException(404, "Product not found")
    runs = [r for r in _runs(slug) if (r / "anomaly.pt").exists() and _read(r / "status.json").get("state") == "completed"]
    if not runs:
        raise HTTPException(409, "No trained model yet - wait for training to finish")
    run_id = runs[-1].name
    model_registry.sync(db)
    mv = db.query(ModelVersion).filter_by(run_id=run_id).first()
    if not mv:
        raise HTTPException(500, "Model was trained but could not be registered")
    if mv.status == "validated":
        model_registry.set_status(db, mv, "approved", user, "approved from the Add-a-product page")
    meta = _read(d / "product.json")
    pid = _profile_id(slug)
    prof = db.get(ProductProfile, pid)
    if prof is None:
        cfg = _profile_config(slug, meta, run_id)
        validate_profile(cfg)
        if not db.get(Product, pid):
            db.add(Product(id=pid, name=cfg["name"], description=cfg["description"]))
            db.flush()
        db.add(ProductProfile(id=pid, product_id=pid, version=1, config=cfg))
        db.flush()
        db.add(ProfileRevision(profile_id=pid, version=1, config=cfg, reason=f"created with {run_id}", user_id=user.id))
        mv.status = "deployed"
        db.commit()
        audit(db, user, "product.activate", "product", pid, {"run_id": run_id, "profile_version": 1}, request)
    else:
        model_registry.deploy(db, mv, prof, user)
        db.commit()
        audit(db, user, "product.activate", "product", pid, {"run_id": run_id}, request)
    from ml.inference.pipeline import CACHE
    CACHE.evict(run_id)
    return custom_out(slug, db)
