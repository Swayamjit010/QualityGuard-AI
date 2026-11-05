"""Short, task-oriented endpoints named in the project specification.

POST /api/inspect             one image -> NORMAL / DEFECTIVE / MANUAL INSPECTION REQUIRED (+ full result)
POST /api/inspect/batch       several images (category recognised per image by default)
GET  /api/dashboard/summary   verdict counts, defect distribution, recent inspections, average inference time
GET  /api/recognizer          component-recogniser status, supported categories and measured performance

They reuse the same services as /api/inspections/* (no duplicated logic).
"""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from sqlalchemy.orm import Session

from app.api import inspections as insp_api
from app.database.session import get_db
from app.models.entities import User
from app.security.auth import require
from app.services import analytics
from ml.config import CHECKPOINT_DIR
from ml.inference.pipeline import RECOGNIZER_RUN

router = APIRouter(prefix="/api", tags=["inspect (short API)"])


@router.post("/inspect", status_code=201)
async def inspect(request: Request, file: UploadFile = File(...), profile_id: str = Form("auto"),
                  product_serial: str = Form(""), options: str | None = Form(None),
                  user: User = Depends(require("inspect")), db: Session = Depends(get_db)):
    """Inspect one image. ``profile_id`` defaults to ``auto``: the component recogniser selects the category; an
    unsupported image yields MANUAL INSPECTION REQUIRED. The response's ``verdict`` is the primary output."""
    return await insp_api.create(request, file=file, profile_id=profile_id, source="single",
                                 product_serial=product_serial, batch_id=None, options=options, user=user, db=db)


@router.post("/inspect/batch", status_code=202)
async def inspect_batch(request: Request, files: list[UploadFile] = File(...), profile_id: str = Form("auto"),
                        name: str = Form(""), options: str | None = Form(None),
                        user: User = Depends(require("inspect")), db: Session = Depends(get_db)):
    return await insp_api.batch(request, files=files, profile_id=profile_id, name=name, source="batch",
                                options=options, user=user, db=db)


@router.get("/dashboard/summary")
def dashboard_summary(days: int | None = None, profile_id: str | None = None, _: User = Depends(require("view")),
                      db: Session = Depends(get_db)):
    d = analytics.dashboard(db, days=days, profile_id=profile_id)
    return {
        "total_inspections": d["total"],
        "normal": d["verdicts"]["NORMAL"], "defective": d["verdicts"]["DEFECTIVE"],
        "manual_inspection_required": d["verdicts"]["MANUAL INSPECTION REQUIRED"],
        "verdicts": d["verdicts"], "by_status": d["by_status"],
        "defect_distribution": d["defect_distribution"], "recent": d["recent"],
        "average_inference_time_ms": d["avg_latency_ms"], "pending_reviews": d["pending_reviews"],
        "simulated_records": d["simulated"],
        "note": "Counts come from stored inspection records. Simulated records are included unless filtered.",
    }


@router.get("/recognizer")
def recognizer(_: User = Depends(require("view"))):
    d = CHECKPOINT_DIR / RECOGNIZER_RUN
    if not (d / "recognizer.pt").exists():
        return {"available": False, "run_id": RECOGNIZER_RUN,
                "setup": "Train it with: python scripts/train_recognizer.py (after preparing the datasets)."}
    m = json.loads((d / "metrics.json").read_text())
    return {"available": True, "run_id": RECOGNIZER_RUN,
            "categories": [{"domain": k, "profile_id": v.get("profile_id"), "title": v.get("title")}
                           for k, v in m.get("domains", {}).items()],
            "rejection_rule": m.get("rejection_rule"), "test": {k: v for k, v in m["test"].items()
                                                                if k != "confusion_matrix"},
            "foreign_probes": m.get("foreign_probes"), "limitations": m.get("limitations")}
