from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.models.entities import ModelVersion, ProductProfile, ProfileRevision, TrainingJob, User
from app.security.auth import require
from app.services import model_registry
from app.services.audit import audit
from app.services.profiles import get_profile, update_profile
from app.workers.jobs import MANAGER, start_training, stop_training
from ml.architectures.registry import ARCHITECTURES
from ml.config import CHECKPOINT_DIR
from ml.inference.pipeline import CACHE

router = APIRouter(prefix="/api", tags=["models & training"])


def mv_out(mv: ModelVersion, full: bool = False, db: Session | None = None) -> dict:
    d = {"id": mv.id, "run_id": mv.run_id, "name": mv.model.name, "task": mv.model.task,
         "architecture": mv.model.architecture, "version": mv.version, "status": mv.status,
         "dataset": mv.dataset_name, "dataset_version": mv.dataset_version, "input_size": mv.input_size,
         "classes": mv.class_labels, "checkpoint_path": mv.checkpoint_path, "checkpoint_size_mb": mv.checkpoint_size_mb,
         "created_at": mv.created_at.isoformat() + "Z", "approved_at": mv.approved_at.isoformat() + "Z" if mv.approved_at else None,
         "notes": mv.notes, "metrics": {m.name: m.value for m in mv.metrics}}
    if full:
        d["preprocess"] = mv.preprocess
        d["training_config"] = mv.training_config
        d["evaluation"] = model_registry.metrics_file(mv)
        hist = CHECKPOINT_DIR / mv.run_id / "history.json"
        d["history"] = json.loads(hist.read_text()) if hist.exists() else []
        d["plots"] = [f for f in ("confusion_matrix.png", "history.png", "roc_pr.png") if (CHECKPOINT_DIR / mv.run_id / f).exists()]
        d["used_by_profiles"] = model_registry._in_use(db, mv.run_id) if db is not None else []
    return d


def _mv(db, mid: int) -> ModelVersion:
    mv = db.get(ModelVersion, mid)
    if not mv:
        raise HTTPException(404, "Model version not found")
    return mv


@router.get("/models")
def list_models(_: User = Depends(require("view")), db: Session = Depends(get_db)):
    return [mv_out(m) for m in db.query(ModelVersion).order_by(ModelVersion.created_at).all()]


@router.post("/models/sync")
def sync(request: Request, user: User = Depends(require("manage_models")), db: Session = Depends(get_db)):
    r = model_registry.sync(db)
    audit(db, user, "model.sync", "registry", "", r, request)
    return r


@router.get("/models/architectures")
def architectures(_: User = Depends(require("view"))):
    return [{"name": a.name, "family": a.family, "default_input": a.default_input, "edge_friendly": a.edge_friendly,
             "cam_layer": a.cam_layer, "notes": a.notes} for a in ARCHITECTURES.values()]


@router.get("/models/{mid}")
def get_model(mid: int, _: User = Depends(require("view")), db: Session = Depends(get_db)):
    return mv_out(_mv(db, mid), full=True, db=db)


@router.get("/models/{mid}/metrics")
def metrics(mid: int, _: User = Depends(require("view")), db: Session = Depends(get_db)):
    mv = _mv(db, mid)
    return {"flat": {m.name: m.value for m in mv.metrics}, "evaluation": model_registry.metrics_file(mv)}


@router.get("/models/{mid}/plots/{name}")
def plot(mid: int, name: str, _: User = Depends(require("view")), db: Session = Depends(get_db)):
    mv = _mv(db, mid)
    if name not in ("confusion_matrix.png", "history.png", "roc_pr.png"):
        raise HTTPException(404, "Unknown plot")
    p = CHECKPOINT_DIR / mv.run_id / name
    if not p.exists():
        raise HTTPException(404, "Plot not available")
    return FileResponse(p)


class StatusIn(BaseModel):
    note: str = Field("", max_length=1000)


@router.post("/models/{mid}/approve")
def approve(mid: int, body: StatusIn, request: Request, user: User = Depends(require("approve_models")),
            db: Session = Depends(get_db)):
    mv = model_registry.set_status(db, _mv(db, mid), "approved", user, body.note)
    return mv_out(mv)


@router.post("/models/{mid}/retire")
def retire(mid: int, body: StatusIn, request: Request, user: User = Depends(require("manage_models")),
           db: Session = Depends(get_db)):
    mv = model_registry.set_status(db, _mv(db, mid), "retired", user, body.note)
    CACHE.evict(mv.run_id)
    return mv_out(mv)


class ActivateIn(BaseModel):
    profile_id: str
    mode: str = Field("primary", pattern="^(primary|ensemble|replace)$")


@router.post("/models/{mid}/activate")
def activate(mid: int, body: ActivateIn, request: Request, user: User = Depends(require("approve_models")),
             db: Session = Depends(get_db)):
    """Deploy an APPROVED model version to a product profile (records a profile revision for rollback)."""
    mv = _mv(db, mid)
    p = model_registry.deploy(db, mv, get_profile(db, body.profile_id), user, body.mode)
    return {"profile_id": p.id, "profile_version": p.version, "models": p.config["models"]}


class RollbackIn(BaseModel):
    profile_id: str
    to_version: int | None = None


@router.post("/models/rollback")
def rollback(body: RollbackIn, request: Request, user: User = Depends(require("approve_models")),
             db: Session = Depends(get_db)):
    p = get_profile(db, body.profile_id)
    revs = db.query(ProfileRevision).filter_by(profile_id=p.id).order_by(ProfileRevision.version.desc()).all()
    cur_models = p.config.get("models")
    target = None
    for r in revs:
        if body.to_version is not None:
            if r.version == body.to_version:
                target = r
                break
        elif r.version < p.version and r.config.get("models") != cur_models:
            target = r
            break
    if not target:
        raise HTTPException(404, "No earlier revision with a different model configuration")
    cfg = {**p.config, "models": target.config["models"]}
    p2 = update_profile(db, p.id, cfg, user, f"rollback models to revision v{target.version}")
    model_registry._refresh_deployed_flags(db)
    for rid in target.config["models"].get("classifiers", []) + [target.config["models"].get("anomaly")]:
        mv = db.query(ModelVersion).filter_by(run_id=rid).first() if rid else None
        if mv and mv.status in ("approved", "validated"):
            mv.status = "deployed"
    db.commit()
    audit(db, user, "model.rollback", "profile", p.id, {"to_revision": target.version, "models": target.config["models"]},
          request)
    return {"profile_id": p.id, "profile_version": p2.version, "models": p2.config["models"]}


@router.get("/profiles/{pid}/revisions")
def revisions(pid: str, _: User = Depends(require("view")), db: Session = Depends(get_db)):
    return [{"version": r.version, "reason": r.reason, "models": r.config.get("models"), "created_at": r.created_at.isoformat() + "Z"}
            for r in db.query(ProfileRevision).filter_by(profile_id=pid).order_by(ProfileRevision.version.desc()).all()]


# ---------------------------------------------------------------- training
class TrainIn(BaseModel):
    kind: str = Field("classification", pattern="^(classification|anomaly)$")
    arch: str | None = None
    dataset: str = "neu_cls"
    category: str | None = None
    epochs: int = Field(8, ge=1, le=100)
    batch_size: int = Field(32, ge=4, le=256)
    lr: float | None = Field(None, gt=0, lt=1)
    loss: str = Field("ce", pattern="^(ce|focal)$")
    max_minutes: float = Field(30, gt=0, le=600)
    resume_run_id: str | None = None
    coreset: float = Field(0.01, gt=0, le=0.5)


@router.post("/training/start", status_code=202)
def start(body: TrainIn, request: Request, user: User = Depends(require("train")), db: Session = Depends(get_db)):
    running = db.query(TrainingJob).filter(TrainingJob.status.in_(("queued", "running"))).count()
    if running:
        raise HTTPException(409, "A training job is already running (CPU budget: one at a time)")
    if body.kind == "classification":
        if body.arch not in ARCHITECTURES:
            raise HTTPException(422, f"arch must be one of {list(ARCHITECTURES)}")
        if body.resume_run_id:
            run_id = body.resume_run_id
        else:
            n = sum(1 for p in CHECKPOINT_DIR.glob(f"{body.dataset}-{body.arch}-v*"))
            run_id = f"{body.dataset}-{body.arch}-v{n + 1}"
        cfg = {**body.model_dump(), "resume": bool(body.resume_run_id)}
    else:
        if not body.category:
            raise HTTPException(422, "category required for anomaly training (e.g. bottle)")
        n = sum(1 for p in CHECKPOINT_DIR.glob(f"mvtec_{body.category}-anomaly-resnet18-v*"))
        run_id = f"mvtec_{body.category}-anomaly-resnet18-v{n + 1}"
        if n:
            raise HTTPException(409, "Anomaly runs are versioned by category; existing run found — retrain via CLI with a new run id.")
        cfg = body.model_dump()
    tj = TrainingJob(run_id=run_id, kind=body.kind, config=cfg, created_by=user.id)
    db.add(tj)
    db.commit()
    start_training(tj.id, body.kind, cfg)
    audit(db, user, "training.start", "training_job", tj.id, {"run_id": run_id, **cfg}, request)
    return {"id": tj.id, "run_id": run_id, "note": "New versions register as 'validated'; approval is required before deployment."}


def tj_out(tj: TrainingJob) -> dict:
    st_p = CHECKPOINT_DIR / tj.run_id / "status.json"
    live = json.loads(st_p.read_text()) if st_p.exists() else {}
    hist_p = CHECKPOINT_DIR / tj.run_id / "history.json"
    return {"id": tj.id, "run_id": tj.run_id, "kind": tj.kind, "status": tj.status, "message": tj.message,
            "config": tj.config, "created_at": tj.created_at.isoformat() + "Z",
            "finished_at": tj.finished_at.isoformat() + "Z" if tj.finished_at else None, "live": live,
            "history": json.loads(hist_p.read_text()) if hist_p.exists() else []}


@router.get("/training/jobs")
def jobs(_: User = Depends(require("view")), db: Session = Depends(get_db)):
    return [tj_out(t) for t in db.query(TrainingJob).order_by(TrainingJob.created_at.desc()).limit(50).all()]


@router.get("/training/jobs/{jid}")
def job(jid: str, _: User = Depends(require("view")), db: Session = Depends(get_db)):
    tj = db.get(TrainingJob, jid)
    if not tj:
        raise HTTPException(404, "Training job not found")
    out = tj_out(tj)
    try:
        with open(tj.log_path, encoding="utf-8", errors="replace") as f:
            out["log_tail"] = [l for l in f.read().splitlines() if "%|" not in l][-60:]
    except OSError:
        out["log_tail"] = []
    return out


@router.post("/training/jobs/{jid}/stop")
def stop(jid: str, request: Request, user: User = Depends(require("train")), db: Session = Depends(get_db)):
    tj = db.get(TrainingJob, jid)
    if not tj or tj.status not in ("queued", "running"):
        raise HTTPException(409, "Job is not running")
    stop_training(tj.run_id)
    audit(db, user, "training.stop", "training_job", jid, request=request)
    return {"ok": True, "note": "Stop requested; the run stops after the current batch and can be resumed."}
