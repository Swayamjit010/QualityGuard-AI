"""Report generation (PDF) and report archive."""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.models.entities import DriftCheck, ModelVersion, ProductionBatch, Report, User
from app.security.auth import require
from app.services import reports
from app.services.audit import audit
from app.services.storage import storage
from ml.config import CHECKPOINT_DIR, METRICS_DIR

router = APIRouter(prefix="/api/reports", tags=["reports"])


def report_out(r: Report) -> dict:
    return {"id": r.id, "kind": r.kind, "title": r.title, "format": r.format, "status": r.status,
            "size_bytes": r.size_bytes, "params": r.params, "inspection_id": r.inspection_id,
            "created_at": r.created_at.isoformat() + "Z", "download_url": f"/api/reports/{r.id}/download"}


@router.get("")
def list_reports(kind: str | None = None, limit: int = Query(100, ge=1, le=1000), _: User = Depends(require("reports")),
                 db: Session = Depends(get_db)):
    q = db.query(Report)
    if kind:
        q = q.filter(Report.kind == kind)
    return [report_out(r) for r in q.order_by(Report.created_at.desc()).limit(limit).all()]


@router.get("/{rid}/download")
def download(rid: str, _: User = Depends(require("reports")), db: Session = Depends(get_db)):
    r = db.get(Report, rid)
    if not r or not r.path or not storage.exists(r.path):
        raise HTTPException(404, "Report not found")
    return FileResponse(storage.resolve(r.path), media_type="application/pdf" if r.format == "pdf" else None,
                        filename=f"QualityGuard_{r.kind}_{r.id[:8]}.{r.format}")


class GenerateIn(BaseModel):
    kind: str = Field(pattern="^(batch|period|model|comparison|dataset|drift|review)$")
    batch_id: str | None = None
    period: str = Field("daily", pattern="^(daily|weekly|monthly)$")
    end: datetime | None = None
    profile_id: str | None = None
    include_simulated: bool = True
    model_id: int | None = None
    model_ids: list[int] = []
    dataset: str | None = Field(None, pattern=r"^[a-z0-9_]{2,64}$")
    drift_check_id: int | None = None
    days: int = Field(30, ge=1, le=365)


@router.post("/generate", status_code=201)
def generate(body: GenerateIn, request: Request, user: User = Depends(require("reports")),
             db: Session = Depends(get_db)):
    k = body.kind
    if k == "batch":
        b = db.get(ProductionBatch, body.batch_id or "")
        if not b:
            raise HTTPException(404, "Batch not found")
        r = reports.batch_pdf(db, b, user)
    elif k == "period":
        r = reports.period_pdf(db, body.period, user, body.profile_id, body.end, body.include_simulated)
    elif k == "model":
        mv = db.get(ModelVersion, body.model_id or 0)
        if not mv:
            raise HTTPException(404, "Model version not found")
        if not (CHECKPOINT_DIR / mv.run_id / "metrics.json").exists():
            raise HTTPException(422, "No evaluation metrics for this model")
        r = reports.model_eval_pdf(db, mv, user)
    elif k == "comparison":
        mvs = [m for m in (db.get(ModelVersion, i) for i in body.model_ids) if m]
        if not mvs:
            mvs = db.query(ModelVersion).join(ModelVersion.model).filter_by(task="classification").all()
        if len(mvs) < 1:
            raise HTTPException(422, "No classification models to compare")
        r = reports.model_comparison_pdf(db, mvs, user)
    elif k == "dataset":
        if not body.dataset or not (METRICS_DIR / f"dataset_report_{body.dataset}.json").exists():
            raise HTTPException(404, "No dataset analysis report for that dataset")
        r = reports.dataset_pdf(db, body.dataset, user)
    elif k == "drift":
        q = db.query(DriftCheck)
        chk = db.get(DriftCheck, body.drift_check_id) if body.drift_check_id else \
            q.filter_by(profile_id=body.profile_id).order_by(DriftCheck.created_at.desc()).first() if body.profile_id \
            else q.order_by(DriftCheck.created_at.desc()).first()
        if not chk:
            raise HTTPException(404, "No drift check found; run one first")
        r = reports.drift_pdf(db, chk, user)
    else:
        r = reports.review_pdf(db, user, body.days)
    audit(db, user, "report.generate", "report", r.id, {"kind": k}, request)
    return report_out(r)
