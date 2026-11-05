"""Human review (feedback), alerts, production batches, background jobs and production-line simulation."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.models.entities import (Alert, HumanFeedback, Inspection, Job, ProductionBatch, ProductProfile, User,
                                 utcnow)
from app.security.auth import require
from app.services.audit import audit
from app.services.events import bus
from app.services.inspection_service import summary
from app.services.profiles import get_profile
from app.services.storage import storage
from app.workers.jobs import MANAGER, simulation_job

router = APIRouter(prefix="/api", tags=["operations"])
SEV = "^(none|low|medium|high|critical)$"


# ---------------------------------------------------------------- human review
class FeedbackIn(BaseModel):
    verdict: str = Field(pattern="^(confirm|reject)$")
    final_decision: str = Field(pattern="^(accept|reject)$")
    corrected_class: str | None = Field(None, max_length=128)
    corrected_severity: str | None = Field(None, pattern=SEV)
    location_confirmed: bool | None = None
    difficult: bool = False
    comment: str = Field("", max_length=4000)


@router.post("/inspections/{iid}/feedback", status_code=201)
def submit_feedback(iid: str, body: FeedbackIn, request: Request, user: User = Depends(require("review")),
                    db: Session = Depends(get_db)):
    i = db.get(Inspection, iid)
    if not i:
        raise HTTPException(404, "Inspection not found")
    if body.corrected_class:
        allowed = set((get_profile(db, i.profile_id).config.get("defect_classes") or [])) | {"good", "normal"}
        if allowed != {"good", "normal"} and body.corrected_class not in allowed:
            raise HTTPException(422, f"corrected_class must be one of {sorted(allowed)}")
    fb = HumanFeedback(inspection_id=i.id, reviewer_id=user.id, **body.model_dump())
    db.add(fb)
    i.review_status = "reviewed"
    i.human_decision = body.final_decision
    if i.severity_assessment and body.corrected_severity:
        i.severity_assessment.human_level = body.corrected_severity
    # resolve the matching review alert(s)
    for a in db.query(Alert).filter(Alert.inspection_id == i.id, Alert.kind == "review_required",
                                    Alert.status != "resolved").all():
        a.status, a.resolved_by = "resolved", user.id
    db.commit()
    audit(db, user, "inspection.review", "inspection", i.id, body.model_dump(), request)
    bus.publish("review", {"inspection_id": i.id, "final_decision": body.final_decision})
    return {"id": fb.id, "inspection": summary(i)}


def review_priority(i: Inspection, class_counts: dict[str, int], n_classes: int) -> tuple[float, list[str]]:
    """Active-learning priority of a pending inspection (higher = more informative to label first).

    Transparent additive score in [0, 5]; each term is in [0, 1]:
      uncertainty   1 - calibrated confidence (classifier profiles)
      disagreement  ensemble members or anomaly engines disagree
      borderline    anomaly score close to its threshold: 1 - |score/threshold - 1| / 0.5, clipped
      unfamiliar    input flagged as unlike the training data
      rarity        predicted class is under-represented among already-verified labels
    """
    r = i.result or {}
    eng = r.get("engines") or {}
    terms: dict[str, float] = {}
    if i.confidence is not None and "classification" in eng:
        terms["uncertainty"] = max(0.0, min(1.0, 1.0 - float(i.confidence)))
    clf, an = eng.get("classification") or {}, eng.get("anomaly") or {}
    if (clf and clf.get("members_agree") is False) or an.get("engines_disagree"):
        terms["disagreement"] = 1.0
    if i.anomaly_score_ratio is not None:
        terms["borderline"] = max(0.0, 1.0 - abs(float(i.anomaly_score_ratio) - 1.0) / 0.5)
    if i.unfamiliar:
        terms["unfamiliar"] = 1.0
    if i.defect_class and n_classes:
        seen = class_counts.get(i.defect_class, 0)
        mean = sum(class_counts.values()) / n_classes
        terms["rarity"] = max(0.0, min(1.0, 1.0 - seen / (mean + 1e-9))) if mean else 1.0
    reasons = [f"{k} {v:.2f}" for k, v in sorted(terms.items(), key=lambda kv: -kv[1]) if v >= 0.05]
    return round(sum(terms.values()), 3), reasons


@router.get("/reviews/queue")
def review_queue(page: int = Query(1, ge=1), page_size: int = Query(25, ge=1, le=200), profile_id: str | None = None,
                 order: str = Query("priority", pattern="^(priority|oldest|newest)$"),
                 _: User = Depends(require("view")), db: Session = Depends(get_db)):
    """Pending reviews. ``order=priority`` ranks by active-learning value (see review_priority);
    ``oldest``/``newest`` by time. Every item carries its priority score and the reasons for it."""
    from sqlalchemy import func

    q = db.query(Inspection).filter(Inspection.review_status == "pending")
    if profile_id:
        q = q.filter(Inspection.profile_id == profile_id)
    total = q.count()
    verified = db.query(Inspection.profile_id, func.coalesce(HumanFeedback.corrected_class, Inspection.defect_class),
                        func.count(HumanFeedback.id)).join(HumanFeedback, HumanFeedback.inspection_id == Inspection.id)
    counts: dict[str, dict[str, int]] = {}
    for pid, cls, n in verified.group_by(Inspection.profile_id, func.coalesce(
            HumanFeedback.corrected_class, Inspection.defect_class)).all():
        if cls:
            counts.setdefault(pid, {})[cls] = counts.setdefault(pid, {}).get(cls, 0) + n
    ncls = {p.id: len((p.config or {}).get("defect_classes", [])) for p in
            db.query(ProductProfile).filter(ProductProfile.id.in_({i for (i,) in q.with_entities(
                Inspection.profile_id).distinct()}))}

    def item(r: Inspection) -> dict:
        score, reasons = review_priority(r, counts.get(r.profile_id, {}), ncls.get(r.profile_id, 0))
        return {**summary(r), "priority": score, "priority_reasons": reasons}

    if order == "priority":
        rows = [item(r) for r in q.order_by(Inspection.created_at.desc()).limit(5000).all()]
        rows.sort(key=lambda d: (-d["priority"], d["created_at"]))
        items = rows[(page - 1) * page_size: page * page_size]
    else:
        col = Inspection.created_at.asc() if order == "oldest" else Inspection.created_at.desc()
        items = [item(r) for r in q.order_by(col).offset((page - 1) * page_size).limit(page_size).all()]
    return {"total": total, "page": page, "page_size": page_size, "order": order, "items": items}


# ---------------------------------------------------------------- alerts
def alert_out(a: Alert) -> dict:
    return {"id": a.id, "created_at": a.created_at.isoformat() + "Z", "kind": a.kind, "level": a.level,
            "title": a.title, "message": a.message, "profile_id": a.profile_id, "inspection_id": a.inspection_id,
            "status": a.status, "data": a.data, "updated_at": a.updated_at.isoformat() + "Z"}


@router.get("/alerts")
def list_alerts(status: str | None = Query(None, pattern="^(open|acknowledged|resolved|active)$"),
                level: str | None = None, kind: str | None = None, limit: int = Query(100, ge=1, le=1000),
                _: User = Depends(require("view")), db: Session = Depends(get_db)):
    q = db.query(Alert)
    if status == "active":
        q = q.filter(Alert.status != "resolved")
    elif status:
        q = q.filter(Alert.status == status)
    if level:
        q = q.filter(Alert.level == level)
    if kind:
        q = q.filter(Alert.kind == kind)
    return [alert_out(a) for a in q.order_by(Alert.created_at.desc()).limit(limit).all()]


@router.post("/alerts/{aid}/{action}")
def alert_action(aid: int, action: str, request: Request, user: User = Depends(require("manage_alerts")),
                 db: Session = Depends(get_db)):
    a = db.get(Alert, aid)
    if not a:
        raise HTTPException(404, "Alert not found")
    if action == "acknowledge":
        if a.status == "open":
            a.status, a.acknowledged_by = "acknowledged", user.id
    elif action == "resolve":
        a.status, a.resolved_by = "resolved", user.id
    else:
        raise HTTPException(404, "Unknown action (acknowledge|resolve)")
    db.commit()
    audit(db, user, f"alert.{action}", "alert", a.id, request=request)
    return alert_out(a)


# ---------------------------------------------------------------- batches
def batch_out(b: ProductionBatch, db: Session) -> dict:
    n = db.query(Inspection).filter_by(batch_id=b.id).count()
    return {"id": b.id, "name": b.name, "profile_id": b.profile_id, "line_id": b.line_id, "source": b.source,
            "created_at": b.created_at.isoformat() + "Z", "inspections": n, "summary": b.summary,
            "is_simulated": b.is_simulated}


@router.get("/batches")
def list_batches(limit: int = Query(50, ge=1, le=500), _: User = Depends(require("view")),
                 db: Session = Depends(get_db)):
    return [batch_out(b, db) for b in db.query(ProductionBatch).order_by(ProductionBatch.created_at.desc())
            .limit(limit).all()]


@router.get("/batches/{bid}")
def get_batch(bid: str, _: User = Depends(require("view")), db: Session = Depends(get_db)):
    b = db.get(ProductionBatch, bid)
    if not b:
        raise HTTPException(404, "Batch not found")
    out = batch_out(b, db)
    out["items"] = [summary(i) for i in db.query(Inspection).filter_by(batch_id=bid)
                    .order_by(Inspection.created_at).limit(1000).all()]
    return out


@router.get("/batches/{bid}/video")
def batch_video(bid: str, _: User = Depends(require("view")), db: Session = Depends(get_db)):
    b = db.get(ProductionBatch, bid)
    key = (b.summary or {}).get("annotated_video") if b else None
    if not key or not storage.exists(key):
        raise HTTPException(404, "Annotated video not available")
    return FileResponse(storage.resolve(key), media_type="video/webm")


# ---------------------------------------------------------------- jobs
def job_out(j: Job) -> dict:
    return {"id": j.id, "kind": j.kind, "status": j.status, "progress": j.progress, "message": j.message,
            "params": j.params, "result": j.result, "created_at": j.created_at.isoformat() + "Z",
            "updated_at": j.updated_at.isoformat() + "Z"}


@router.get("/jobs")
def list_jobs(kind: str | None = None, limit: int = Query(50, ge=1, le=500), _: User = Depends(require("view")),
              db: Session = Depends(get_db)):
    q = db.query(Job)
    if kind:
        q = q.filter(Job.kind == kind)
    return [job_out(j) for j in q.order_by(Job.created_at.desc()).limit(limit).all()]


@router.get("/jobs/{jid}")
def get_job(jid: str, _: User = Depends(require("view")), db: Session = Depends(get_db)):
    j = db.get(Job, jid)
    if not j:
        raise HTTPException(404, "Job not found")
    return job_out(j)


@router.post("/jobs/{jid}/stop")
def stop_job(jid: str, request: Request, user: User = Depends(require("inspect")), db: Session = Depends(get_db)):
    j = db.get(Job, jid)
    if not j or j.status not in ("queued", "running"):
        raise HTTPException(409, "Job is not running")
    if not MANAGER.stop(jid):
        raise HTTPException(409, "Job is not owned by this server process")
    audit(db, user, "job.stop", "job", jid, request=request)
    return {"ok": True}


# ---------------------------------------------------------------- production-line simulation
class SimulationIn(BaseModel):
    profile_id: str
    interval_s: float = Field(1.0, ge=0.0, le=60)
    count: int = Field(30, ge=1, le=2000)
    defect_rate: float | None = Field(None, ge=0, le=1)
    line_id: str = Field("SIM-LINE-1", pattern=r"^[A-Za-z0-9_-]{1,64}$")


@router.post("/simulation/start", status_code=202)
def start_simulation(body: SimulationIn, request: Request, user: User = Depends(require("inspect")),
                     db: Session = Depends(get_db)):
    prof = get_profile(db, body.profile_id)
    if not prof.config.get("dataset"):
        raise HTTPException(422, "Profile has no dataset to draw simulation images from")
    running = db.query(Job).filter(Job.kind == "simulation", Job.status.in_(("queued", "running"))).count()
    if running:
        raise HTTPException(409, "A simulation is already running")
    b = ProductionBatch(name=f"simulation-{utcnow():%Y%m%d-%H%M%S}", profile_id=prof.id, source="simulation",
                        created_by=user.id, is_simulated=True, line_id=None)
    db.add(b)
    db.commit()
    job = MANAGER.create("simulation", {**body.model_dump(), "batch_id": b.id}, user)
    MANAGER.submit(job, simulation_job, prof.id, b.id, user.id, body.interval_s, body.count, body.defect_rate,
                   body.line_id)
    audit(db, user, "simulation.start", "batch", b.id, body.model_dump(), request)
    return {"job_id": job, "batch_id": b.id,
            "note": "Simulation replays held-out dataset images through the real pipeline; records are flagged simulated."}
