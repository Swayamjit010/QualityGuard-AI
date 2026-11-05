"""Datasets, product profiles and the defect knowledge base."""
from __future__ import annotations

import copy
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.database.session import get_db
from app.models.entities import Dataset, KnowledgeEntry, ProductProfile, User
from app.security.auth import require
from app.services import datasets as ds_service
from app.services.audit import audit
from app.services.profiles import get_profile, update_profile
from ml.config import absp
from ml.datasets.splitting import load_manifest

router = APIRouter(prefix="/api", tags=["datasets, profiles & knowledge"])
settings = get_settings()


# ---------------------------------------------------------------- datasets
def ds_out(d: Dataset) -> dict:
    return {"id": d.id, "name": d.name, "title": d.title, "task": d.task, "layout": d.layout, "source": d.source,
            "license": d.license, "path": d.path, "description": d.description,
            "created_at": d.created_at.isoformat() + "Z",
            "versions": [{"id": v.id, "version": v.version, "num_samples": v.num_samples, "split_counts": v.split_counts,
                          "split_policy": v.split_policy, "created_at": v.created_at.isoformat() + "Z"}
                         for v in d.versions]}


def _ds(db: Session, name: str) -> Dataset:
    d = db.query(Dataset).filter_by(name=name).first()
    if not d:
        raise HTTPException(404, "Dataset not found")
    return d


@router.get("/datasets")
def list_datasets(_: User = Depends(require("view")), db: Session = Depends(get_db)):
    return [ds_out(d) for d in db.query(Dataset).order_by(Dataset.name).all()]


@router.get("/datasets/{name}")
def get_dataset(name: str, _: User = Depends(require("view")), db: Session = Depends(get_db)):
    d = _ds(db, name)
    out = ds_out(d)
    out["statistics"] = d.versions[-1].statistics if d.versions else None
    return out


@router.get("/datasets/{name}/samples")
def samples(name: str, split: str | None = Query(None, pattern="^(train|val|test)$"), label: str | None = None,
            page: int = Query(1, ge=1), page_size: int = Query(48, ge=1, le=200), _: User = Depends(require("view")),
            db: Session = Depends(get_db)):
    _ds(db, name)
    try:
        rows = load_manifest(name)["samples"]
    except FileNotFoundError:
        raise HTTPException(404, "Dataset has no manifest; run analysis + split first")
    if split:
        rows = [r for r in rows if r["split"] == split]
    if label:
        rows = [r for r in rows if r["label"] == label]
    labels = sorted({r["label"] for r in load_manifest(name)["samples"]})
    page_rows = rows[(page - 1) * page_size: page * page_size]
    return {"total": len(rows), "labels": labels, "page": page, "page_size": page_size,
            "items": [{"index": (page - 1) * page_size + i, "label": r["label"], "split": r["split"],
                       "file": r["path"].rsplit("/", 1)[-1],
                       "url": f"/api/datasets/{name}/image?path={quote(r['path'])}"} for i, r in enumerate(page_rows)]}


@router.get("/datasets/{name}/image")
def sample_image(name: str, path: str, _: User = Depends(require("view")), db: Session = Depends(get_db)):
    _ds(db, name)
    known = {r["path"] for r in load_manifest(name)["samples"]}
    if path not in known:  # only manifest paths are served (no arbitrary file reads)
        raise HTTPException(404, "Not a sample of this dataset")
    if not absp(path).is_file():
        raise HTTPException(404, "Image file is missing (dataset not downloaded on this server)")
    return FileResponse(absp(path), headers={"Cache-Control": "private, max-age=86400"})


@router.post("/datasets/import", status_code=201)
async def import_dataset(request: Request, file: UploadFile = File(...), name: str = Form(...),
                         task: str = Form("classification"), user: User = Depends(require("manage_datasets")),
                         db: Session = Depends(get_db)):
    if not (file.filename or "").lower().endswith(".zip"):
        raise HTTPException(415, "Upload a .zip archive (one folder per class, or images + labels.csv)")
    limit = 2048 * 1024 * 1024
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, "Archive exceeds 2 GB")
    d = ds_service.import_zip(db, name, data, task)
    audit(db, user, "dataset.import", "dataset", d.name, {"bytes": len(data)}, request)
    return ds_out(d)


class SplitIn(BaseModel):
    train: float = Field(0.7, gt=0, lt=1)
    val: float = Field(0.15, gt=0, lt=1)
    test: float = Field(0.15, gt=0, lt=1)
    seed: int = 42


@router.post("/datasets/{name}/analyze")
def analyze(name: str, body: SplitIn, request: Request, user: User = Depends(require("manage_datasets")),
            db: Session = Depends(get_db)):
    if abs(body.train + body.val + body.test - 1) > 1e-6:
        raise HTTPException(422, "Split ratios must sum to 1")
    v = ds_service.analyze_and_split(db, _ds(db, name), (body.train, body.val, body.test), body.seed)
    audit(db, user, "dataset.split", "dataset", name, {"version": v.version, **body.model_dump()}, request)
    return {"version": v.version, "num_samples": v.num_samples, "split_counts": v.split_counts,
            "statistics": v.statistics}


@router.post("/datasets/feedback-export")
def feedback_export(request: Request, only_new: bool = True, user: User = Depends(require("export")),
                    db: Session = Depends(get_db)):
    r = ds_service.export_feedback(db, only_new)
    audit(db, user, "dataset.feedback_export", "dataset", r.get("path") or "", r, request)
    return r


# ---------------------------------------------------------------- product profiles
def profile_out(p: ProductProfile) -> dict:
    c = p.config
    return {"id": p.id, "name": c.get("name", p.id), "description": c.get("description", ""), "version": p.version,
            "dataset": c.get("dataset"), "product_type": c.get("product_type"),
            "defect_classes": c.get("defect_classes", []), "models": c.get("models", {}),
            "assumes_defect_present": c.get("assumes_defect_present", False),
            "updated_at": p.updated_at.isoformat() + "Z", "config": c}


@router.get("/profiles")
def list_profiles(_: User = Depends(require("view")), db: Session = Depends(get_db)):
    return [profile_out(p) for p in db.query(ProductProfile).filter_by(is_active=True).order_by(ProductProfile.id)]


@router.get("/profiles/{pid}")
def get_profile_ep(pid: str, _: User = Depends(require("view")), db: Session = Depends(get_db)):
    return profile_out(get_profile(db, pid))


class ProfileUpdateIn(BaseModel):
    config: dict
    reason: str = Field("edited via UI", max_length=255)


@router.put("/profiles/{pid}")
def put_profile(pid: str, body: ProfileUpdateIn, request: Request, user: User = Depends(require("manage_profiles")),
                db: Session = Depends(get_db)):
    old = copy.deepcopy(get_profile(db, pid).config)
    new = copy.deepcopy(body.config)
    # model deployment goes through the approval workflow (/api/models/{id}/activate), not profile edits
    if new.get("models") != old.get("models") and "approve_models" not in user.permission_codes:
        raise HTTPException(403, "Changing deployed models requires the approve_models permission")
    p = update_profile(db, pid, new, user, body.reason)
    audit(db, user, "profile.update", "profile", pid, {"version": p.version, "reason": body.reason}, request)
    return profile_out(p)


# ---------------------------------------------------------------- knowledge base
def kb_out(k: KnowledgeEntry) -> dict:
    return {"id": k.id, "defect_class": k.defect_class, "profile_id": k.profile_id, "title": k.title,
            "visual_description": k.visual_description, "characteristics": k.characteristics,
            "inspection_guidance": k.inspection_guidance, "corrective_actions": k.corrective_actions,
            "source": k.source, "updated_at": k.updated_at.isoformat() + "Z"}


@router.get("/knowledge")
def list_kb(q: str | None = None, defect_class: str | None = None, _: User = Depends(require("view")),
            db: Session = Depends(get_db)):
    qr = db.query(KnowledgeEntry)
    if defect_class:
        qr = qr.filter(KnowledgeEntry.defect_class == defect_class)
    if q:
        like = f"%{q[:100]}%"
        qr = qr.filter(KnowledgeEntry.title.like(like) | KnowledgeEntry.visual_description.like(like) |
                       KnowledgeEntry.defect_class.like(like))
    return [kb_out(k) for k in qr.order_by(KnowledgeEntry.defect_class).all()]


class KbIn(BaseModel):
    defect_class: str = Field(max_length=128)
    profile_id: str | None = None
    title: str = Field(max_length=255)
    visual_description: str = ""
    characteristics: list[str] = []
    inspection_guidance: list[str] = []
    corrective_actions: list[dict] = []
    source: str = ""


@router.post("/knowledge", status_code=201)
def create_kb(body: KbIn, request: Request, user: User = Depends(require("manage_profiles")),
              db: Session = Depends(get_db)):
    k = KnowledgeEntry(**body.model_dump())
    db.add(k)
    db.commit()
    audit(db, user, "knowledge.create", "knowledge", k.id, request=request)
    return kb_out(k)


@router.put("/knowledge/{kid}")
def update_kb(kid: int, body: KbIn, request: Request, user: User = Depends(require("manage_profiles")),
              db: Session = Depends(get_db)):
    k = db.get(KnowledgeEntry, kid)
    if not k:
        raise HTTPException(404, "Entry not found")
    for f, v in body.model_dump().items():
        setattr(k, f, v)
    db.commit()
    audit(db, user, "knowledge.update", "knowledge", k.id, request=request)
    return kb_out(k)
