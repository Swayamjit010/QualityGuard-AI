from __future__ import annotations

import json
from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.database.session import get_db
from app.models.entities import ImageAnnotation, Inspection, ProductionBatch, User
from app.security.auth import limiter, require
from app.services import reports, similarity
from app.services.audit import audit
from app.services.inspection_service import IMAGE_KEYS_ORDER, run_inspection, summary
from app.services.profiles import get_profile
from app.services.storage import safe_name, storage
from app.workers.jobs import MANAGER, batch_job, video_job
from ml.inference.pipeline import ModelUnavailable
from ml.inference.validation import decode_image

router = APIRouter(prefix="/api/inspections", tags=["inspections"])
settings = get_settings()
IMAGE_EXT = {"png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"}
VIDEO_EXT = {"mp4", "avi", "mov", "mkv", "webm"}
SOURCES = {"single", "camera", "api", "folder", "batch"}


async def read_upload(f: UploadFile, max_mb: int, exts: set[str]) -> bytes:
    name = f.filename or ""
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in exts:
        raise HTTPException(415, f"Unsupported file type '.{ext}'. Allowed: {', '.join(sorted(exts))}")
    data = await f.read(max_mb * 1024 * 1024 + 1)
    if len(data) > max_mb * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {max_mb} MB limit")
    if not data:
        raise HTTPException(422, "Empty file")
    return data


def _opts(options: str | None) -> dict:
    if not options:
        return {}
    try:
        o = json.loads(options)
    except json.JSONDecodeError:
        raise HTTPException(422, "options must be JSON")
    allowed = {"engines", "cam_methods", "classifiers", "anomaly_model", "skip_explain"}
    bad = set(o) - allowed
    if bad:
        raise HTTPException(422, f"Unknown options: {sorted(bad)}")
    if "cam_methods" in o and not set(o["cam_methods"]) <= {"gradcam", "gradcam_pp", "scorecam"}:
        raise HTTPException(422, "cam_methods must be subset of gradcam, gradcam_pp, scorecam")
    return o


def detail(i: Inspection) -> dict:
    d = summary(i)
    d["result"] = i.result
    d["visualizations"] = {k: f"/api/inspections/{i.id}/images/{k}" for k in (i.result or {}).get("visualizations", [])}
    d["feedback"] = [{"id": f.id, "reviewer": f.reviewer.email, "verdict": f.verdict, "corrected_class": f.corrected_class,
                      "corrected_severity": f.corrected_severity, "final_decision": f.final_decision,
                      "location_confirmed": f.location_confirmed, "difficult": f.difficult, "comment": f.comment,
                      "created_at": f.created_at.isoformat() + "Z"} for f in i.feedback]
    d["annotations"] = [annotation_out(a) for a in _annotations(i)]
    d["traceability"] = {"image_id": i.image_id, "profile_version": i.profile_version, "batch_id": i.batch_id,
                         "line_id": i.line_id, "product_serial": i.product_serial, "models": i.models_used,
                         "user_id": i.user_id}
    return d


@router.post("", status_code=201)
async def create(request: Request, file: UploadFile = File(...), profile_id: str = Form("auto"),
                 source: str = Form("single"), product_serial: str = Form(""), batch_id: str | None = Form(None),
                 options: str | None = Form(None), user: User = Depends(require("inspect")),
                 db: Session = Depends(get_db)):
    limiter.check(f"inspect:{user.id}", settings.rate_limit_per_minute)
    if getattr(request.state, "auth", "") == "api_key":
        source = "api"
    if source not in SOURCES:
        raise HTTPException(422, f"source must be one of {sorted(SOURCES)}")
    if profile_id != "auto":
        get_profile(db, profile_id)
    data = await read_upload(file, settings.max_upload_mb, IMAGE_EXT)
    try:
        insp = run_inspection(db, data=data, filename=file.filename or "upload", profile_id=profile_id, user=user,
                              source=source, product_serial=product_serial, batch_id=batch_id, options=_opts(options))
    except ModelUnavailable as exc:
        raise HTTPException(503, f"Model unavailable: {exc}")
    audit(db, user, "inspection.run", "inspection", insp.id, {"profile": insp.profile_id, "requested": profile_id,
                                                              "source": source,
                                                              "status": insp.status}, request)
    return detail(insp)


@router.post("/batch", status_code=202)
async def batch(request: Request, files: list[UploadFile] = File(...), profile_id: str = Form(...),
                name: str = Form(""), source: str = Form("batch"), options: str | None = Form(None),
                user: User = Depends(require("inspect")), db: Session = Depends(get_db)):
    """Batch and folder inspection (the browser sends a selected folder's files with source=folder).
    ``profile_id=auto`` recognises the component category of each image separately."""
    if profile_id != "auto":
        get_profile(db, profile_id)
    if len(files) > settings.max_batch_files:
        raise HTTPException(413, f"At most {settings.max_batch_files} files per batch")
    opts = _opts(options)
    b = ProductionBatch(name=safe_name(name or f"batch-{datetime.utcnow():%Y%m%d-%H%M%S}"),
                        profile_id=None if profile_id == "auto" else profile_id,
                        source="folder" if source == "folder" else "batch", created_by=user.id)
    db.add(b)
    db.commit()
    staged, skipped = [], []
    for f in files:
        ext = (f.filename or "").rsplit(".", 1)[-1].lower()
        if ext not in IMAGE_EXT:
            skipped.append(f.filename)
            continue
        data = await read_upload(f, settings.max_upload_mb, IMAGE_EXT)
        key = storage.put_bytes(f"staging/{b.id}/{len(staged):05d}.{ext}", data)
        staged.append((key, safe_name(f.filename or "file")))
    if not staged:
        raise HTTPException(422, "No supported image files in upload")
    job = MANAGER.create("batch", {"batch_id": b.id, "profile_id": profile_id, "files": len(staged),
                                   "skipped": skipped}, user)
    MANAGER.submit(job, batch_job, staged, profile_id, b.id, user.id, b.source, opts)
    audit(db, user, "inspection.batch", "batch", b.id, {"files": len(staged), "skipped": len(skipped)}, request)
    return {"job_id": job, "batch_id": b.id, "accepted": len(staged), "skipped_unsupported": skipped}


@router.post("/video", status_code=202)
async def video(request: Request, file: UploadFile = File(...), profile_id: str = Form(...),
                every_seconds: float = Form(1.0), max_frames: int = Form(60), user: User = Depends(require("inspect")),
                db: Session = Depends(get_db)):
    get_profile(db, profile_id)
    if not (0.05 <= every_seconds <= 60) or not (1 <= max_frames <= 600):
        raise HTTPException(422, "every_seconds in [0.05, 60], max_frames in [1, 600]")
    data = await read_upload(file, settings.max_video_mb, VIDEO_EXT)
    b = ProductionBatch(name=safe_name(file.filename or "video"), profile_id=profile_id, source="video",
                        created_by=user.id)
    db.add(b)
    db.commit()
    key = storage.put_bytes(f"videos/{b.id}/source.{file.filename.rsplit('.', 1)[-1].lower()}", data)
    job = MANAGER.create("video", {"batch_id": b.id, "profile_id": profile_id, "every_seconds": every_seconds,
                                   "max_frames": max_frames}, user)
    MANAGER.submit(job, video_job, key, profile_id, b.id, user.id, every_seconds, max_frames, {})
    audit(db, user, "inspection.video", "batch", b.id, {"every_seconds": every_seconds}, request)
    return {"job_id": job, "batch_id": b.id}


@router.post("/compare-reference")
async def compare_reference(reference: UploadFile = File(...), test: UploadFile = File(...),
                            user: User = Depends(require("inspect"))):
    from ml.inference.reference import compare
    import base64
    import cv2
    r = decode_image(await read_upload(reference, settings.max_upload_mb, IMAGE_EXT))
    t = decode_image(await read_upload(test, settings.max_upload_mb, IMAGE_EXT))
    if r is None or t is None:
        raise HTTPException(422, "Could not decode one of the images")
    res = compare(r, t)
    imgs = res.pop("_images")
    res["images"] = {k: "data:image/png;base64," + base64.b64encode(cv2.imencode(".png", v[..., ::-1])[1]).decode()
                     for k, v in imgs.items()}
    res["note"] = "Classical registration + difference analysis; flags any visual difference, including benign ones."
    return res


@router.get("")
def list_inspections(page: int = Query(1, ge=1), page_size: int = Query(25, ge=1, le=200), q: str | None = None,
                     profile_id: str | None = None, status: str | None = None, defect_class: str | None = None,
                     severity: str | None = None, source: str | None = None, batch_id: str | None = None,
                     review_status: str | None = None, simulated: bool | None = None, date_from: datetime | None = None,
                     verdict: str | None = Query(None, pattern="^(NORMAL|DEFECTIVE|MANUAL INSPECTION REQUIRED)$"),
                     date_to: datetime | None = None, _: User = Depends(require("view")), db: Session = Depends(get_db)):
    qr = db.query(Inspection)
    for col, val in ((Inspection.profile_id, profile_id), (Inspection.status, status),
                     (Inspection.defect_class, defect_class), (Inspection.severity, severity),
                     (Inspection.source, source), (Inspection.batch_id, batch_id),
                     (Inspection.review_status, review_status)):
        if val:
            qr = qr.filter(col == val)
    if simulated is not None:
        qr = qr.filter(Inspection.is_simulated.is_(simulated))
    if verdict == "MANUAL INSPECTION REQUIRED":
        qr = qr.filter(Inspection.status.in_(["REVIEW REQUIRED", "INCONCLUSIVE"]))
    elif verdict == "DEFECTIVE":
        qr = qr.filter(Inspection.status.in_(["PASS", "FAIL"]),
                       or_(Inspection.status == "FAIL", Inspection.defect_present.is_(True)))
    elif verdict == "NORMAL":
        qr = qr.filter(Inspection.status == "PASS", Inspection.defect_present.is_(False))
    if date_from:
        qr = qr.filter(Inspection.created_at >= date_from)
    if date_to:
        qr = qr.filter(Inspection.created_at < date_to)
    if q:
        like = f"%{q[:100]}%"
        qr = qr.filter(or_(Inspection.id.like(like), Inspection.filename.like(like), Inspection.product_serial.like(like),
                           Inspection.defect_class.like(like)))
    total = qr.count()
    rows = qr.order_by(Inspection.created_at.desc()).offset((page - 1) * page_size).limit(page_size).all()
    return {"total": total, "page": page, "page_size": page_size, "items": [summary(r) for r in rows]}


@router.get("/export")
def export(fmt: str = Query("csv", pattern="^(csv|json)$"), profile_id: str | None = None, status: str | None = None,
           batch_id: str | None = None, days: int | None = None, request: Request = None,
           user: User = Depends(require("export")), db: Session = Depends(get_db)):
    from app.services.analytics import filtered
    q = filtered(db, days=days, profile_id=profile_id, status=status)
    if batch_id:
        q = q.filter(Inspection.batch_id == batch_id)
    rows = q.order_by(Inspection.created_at).limit(100_000).all()
    data = reports.export_rows(rows, fmt)
    audit(db, user, "export.inspections", "inspection", "", {"fmt": fmt, "rows": len(rows)}, request)
    return Response(data, media_type="text/csv" if fmt == "csv" else "application/json",
                    headers={"Content-Disposition": f'attachment; filename="inspections.{fmt}"'})


def _get(db, iid) -> Inspection:
    i = db.get(Inspection, iid)
    if not i:
        raise HTTPException(404, "Inspection not found")
    return i


@router.get("/{iid}")
def get(iid: str, _: User = Depends(require("view")), db: Session = Depends(get_db)):
    return detail(_get(db, iid))


@router.delete("/{iid}")
def delete(iid: str, request: Request, user: User = Depends(require("manage_settings")), db: Session = Depends(get_db)):
    i = _get(db, iid)
    storage.delete_tree(i.artifact_dir)
    db.delete(i)
    db.commit()
    audit(db, user, "inspection.delete", "inspection", iid, request=request)
    return {"ok": True}


@router.get("/{iid}/visualizations")
def visualizations(iid: str, _: User = Depends(require("view")), db: Session = Depends(get_db)):
    i = _get(db, iid)
    kinds = {"gradcam": "explanation (classifier attention, not a defect boundary)",
             "anomaly": "anomaly map (deviation from defect-free references)",
             "segmentation": "thresholded anomaly map (estimated defect region)",
             "bounding": "regions from the segmentation mask", "defect_zoom": "magnified largest region",
             "comparison": "original vs. result", "original": "input image", "scorecam": "explanation",
             "gradcam_pp": "explanation"}
    out = []
    for k in (i.result or {}).get("visualizations", []):
        desc = next((v for p, v in kinds.items() if k.startswith(p)), "")
        out.append({"key": k, "url": f"/api/inspections/{iid}/images/{k}", "description": desc})
    return out


@router.get("/{iid}/images/{key}")
def image(iid: str, key: str, _: User = Depends(require("view")), db: Session = Depends(get_db)):
    i = _get(db, iid)
    if key == "upload":
        p = next(storage.resolve(i.artifact_dir).glob("upload.*"), None)
    else:
        if key not in (i.result or {}).get("visualizations", []) or key not in IMAGE_KEYS_ORDER + list(
                (i.result or {}).get("visualizations", [])):
            raise HTTPException(404, "Visualization not available")
        p = storage.resolve(f"{i.artifact_dir}/{safe_name(key)}.png")
    if not p or not p.exists():
        raise HTTPException(404, "File missing")
    return FileResponse(p, headers={"Cache-Control": "private, max-age=3600"})


@router.get("/{iid}/images")
def images(iid: str, u: User = Depends(require("view")), db: Session = Depends(get_db)):
    return visualizations(iid, u, db)


@router.get("/{iid}/report")
def report(iid: str, request: Request, user: User = Depends(require("reports")), db: Session = Depends(get_db)):
    i = _get(db, iid)
    r = reports.inspection_pdf(db, i, user)
    audit(db, user, "report.generate", "report", r.id, {"kind": "inspection", "inspection": iid}, request)
    return FileResponse(storage.resolve(r.path), media_type="application/pdf",
                        filename=f"QualityGuard_inspection_{iid[:8]}.pdf")


@router.get("/{iid}/similar")
def similar(iid: str, k: int = Query(8, ge=1, le=50), only_defective: bool = False,
            _: User = Depends(require("view")), db: Session = Depends(get_db)):
    i = _get(db, iid)
    return {"query": iid, "embedding_model": i.embedding_model, "results": similarity.similar(db, i, k, only_defective),
            "caveat": "Visual similarity is not proof of an identical defect cause."}


# ---------------------------------------------------------------- inspector annotations (human-in-the-loop)
def _annotations(i: Inspection) -> list:
    from app.database.session import SessionLocal
    if not i.image_id:
        return []
    with SessionLocal() as db:
        return db.query(ImageAnnotation).filter_by(image_id=i.image_id, source="inspector")             .order_by(ImageAnnotation.created_at).all()


def annotation_out(a: ImageAnnotation) -> dict:
    return {"id": a.id, "kind": a.kind, "label": a.label, "data": a.data, "created_by": a.created_by,
            "created_at": a.created_at.isoformat() + "Z",
            "mask_url": f"/api/inspections/annotations/{a.id}/mask" if a.mask_path else None}


@router.post("/{iid}/annotations", status_code=201)
async def add_annotation(iid: str, request: Request, kind: str = Form(...), label: str = Form(""),
                         data: str = Form("{}"), file: UploadFile | None = File(None),
                         user: User = Depends(require("review")), db: Session = Depends(get_db)):
    """Attach an inspector annotation to the inspected image: a binary defect ``mask`` (PNG, same aspect as the
    image; non-zero = defect), ``bbox`` boxes (data = {"boxes": [[x, y, w, h], ...]} in original pixels),
    a ``polygon`` (data = {"points": [[x, y], ...]}) or a plain ``label``. Stored as ImageAnnotation rows with
    source=inspector so curated datasets can include verified localisation."""
    import cv2
    import numpy as np

    i = _get(db, iid)
    if not i.image_id:
        raise HTTPException(409, "Inspection has no stored image record")
    if kind not in {"mask", "bbox", "polygon", "label"}:
        raise HTTPException(422, "kind must be mask, bbox, polygon or label")
    try:
        payload = json.loads(data or "{}")
    except json.JSONDecodeError:
        raise HTTPException(422, "data must be JSON")
    if not isinstance(payload, dict) or len(data) > 200_000:
        raise HTTPException(422, "data must be a JSON object (<= 200 kB)")
    w, h = ((i.result or {}).get("input") or {}).get("width"), ((i.result or {}).get("input") or {}).get("height")
    mask_key = None
    if kind == "mask":
        if file is None:
            raise HTTPException(422, "A mask PNG file is required for kind=mask")
        raw = await read_upload(file, settings.max_upload_mb, {"png"})
        m = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise HTTPException(422, "Mask is not a readable PNG")
        if w and h and abs(m.shape[1] / m.shape[0] - w / h) > 0.02:
            raise HTTPException(422, f"Mask aspect ratio {m.shape[1]}x{m.shape[0]} does not match the image {w}x{h}")
        binary = (m > 0).astype(np.uint8) * 255
        payload = {**payload, "mask_hw": [int(m.shape[0]), int(m.shape[1])],
                   "defect_fraction": float((binary > 0).mean())}
        ok, png = cv2.imencode(".png", binary)
        mask_key = storage.put_bytes(f"{i.artifact_dir}/annotations/mask_{len(_annotations(i)) + 1}.png",
                                     png.tobytes())
    elif kind == "bbox":
        boxes = payload.get("boxes")
        if not isinstance(boxes, list) or not boxes or not all(
                isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) and v >= 0 for v in b)
                and b[2] > 0 and b[3] > 0 for b in boxes):
            raise HTTPException(422, "bbox data must be {\"boxes\": [[x, y, w, h], ...]} with positive sizes")
        if w and h and any(b[0] + b[2] > w * 1.01 or b[1] + b[3] > h * 1.01 for b in boxes):
            raise HTTPException(422, f"A box lies outside the {w}x{h} image")
    elif kind == "polygon":
        pts = payload.get("points")
        if not isinstance(pts, list) or len(pts) < 3 or not all(isinstance(p, list) and len(p) == 2 for p in pts):
            raise HTTPException(422, "polygon data must be {\"points\": [[x, y], ...]} with >= 3 points")
    if label:
        allowed = set(get_profile(db, i.profile_id).config.get("defect_classes") or []) | {"good", "normal"}
        if label not in allowed:
            raise HTTPException(422, f"label must be one of {sorted(allowed)}")
    a = ImageAnnotation(image_id=i.image_id, kind=kind, label=label, data=payload, mask_path=mask_key,
                        source="inspector", created_by=user.id)
    db.add(a)
    db.commit()
    audit(db, user, "inspection.annotate", "inspection", i.id, {"kind": kind, "label": label}, request=request)
    return annotation_out(a)


@router.get("/annotations/{aid}/mask")
def annotation_mask(aid: int, _: User = Depends(require("view")), db: Session = Depends(get_db)):
    a = db.get(ImageAnnotation, aid)
    if not a or not a.mask_path:
        raise HTTPException(404, "No mask for this annotation")
    return FileResponse(storage.resolve(a.mask_path), media_type="image/png")


@router.delete("/{iid}/annotations/{aid}")
def delete_annotation(iid: str, aid: int, request: Request, user: User = Depends(require("review")),
                      db: Session = Depends(get_db)):
    i = _get(db, iid)
    a = db.get(ImageAnnotation, aid)
    if not a or a.image_id != i.image_id or a.source != "inspector":
        raise HTTPException(404, "Annotation not found")
    db.delete(a)
    db.commit()
    audit(db, user, "inspection.annotation_delete", "inspection", i.id, {"annotation": aid}, request=request)
    return {"deleted": aid}
