"""System health, logs, audit trail, settings, drift monitoring, assistant and the live event WebSocket."""
from __future__ import annotations

import asyncio
import platform
import time
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app import integrations
from app.core.logging import RING
from app.database.session import SessionLocal, get_db
from app.models.entities import (AuditEvent, DriftCheck, Inspection, ProductProfile, RevokedToken, SystemSetting,
                                 User)
from app.security.auth import decode_token, get_current_user, require
from app.services import assistant, drift
from app.services.audit import audit
from app.services.events import bus
from app.services.profiles import get_profile
from app.services.storage import storage

router = APIRouter(prefix="/api", tags=["system"])
settings = get_settings()
STARTED = time.time()
EDITABLE_SETTINGS = {"alert_rules", "report_preferences", "notification_preferences", "camera_settings",
                     "storage_settings"}


@router.get("/health")
def health(db: Session = Depends(get_db)):
    """Unauthenticated liveness/readiness probe."""
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False
    return {"status": "ok" if db_ok else "degraded", "database": db_ok, "uptime_s": round(time.time() - STARTED, 1),
            "app": settings.app_name, "environment": settings.environment}


@router.get("/system/info")
def info(_: User = Depends(require("view_logs")), db: Session = Depends(get_db)):
    import torch

    from ml.config import device_info
    from ml.inference.pipeline import CACHE
    return {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
            "device": device_info(), "loaded_models": CACHE.loaded(), "storage": storage.usage(),
            "inspections": db.query(Inspection).count(), "uptime_s": round(time.time() - STARTED, 1),
            "assistant_llm": assistant.llm_available()}


def _resources() -> dict:
    import psutil

    vm, proc = psutil.virtual_memory(), psutil.Process()
    return {"cpu_percent": psutil.cpu_percent(interval=0.2), "cpu_count": psutil.cpu_count(),
            "memory_used_bytes": vm.total - vm.available, "memory_total_bytes": vm.total,
            "memory_percent": vm.percent, "process_rss_bytes": proc.memory_info().rss,
            "process_threads": proc.num_threads()}


def _model_status(db: Session) -> list[dict]:
    """For every active profile: does each configured checkpoint exist on disk, and is it loaded in memory?"""
    from ml.config import CHECKPOINT_DIR
    from ml.inference.pipeline import CACHE

    loaded = {m["key"].split(":", 1)[1] for m in CACHE.loaded()}
    out = []
    for p in db.query(ProductProfile).filter_by(is_active=True).order_by(ProductProfile.id):
        cfg = (p.config or {}).get("models", {})
        runs = [("classifier", r) for r in cfg.get("classifiers", [])] +                ([("anomaly", cfg["anomaly"])] if cfg.get("anomaly") else [])
        for kind, rid in runs:
            f = CHECKPOINT_DIR / rid / ("model.pt" if kind == "classifier" else "anomaly.pt")
            out.append({"profile_id": p.id, "kind": kind, "run_id": rid, "checkpoint_present": f.exists(),
                        "loaded": rid in loaded})
    return out


@router.get("/system/status")
def status(_: User = Depends(require("view_logs")), db: Session = Depends(get_db)):
    """System health page: API, database, models, compute resources, storage and recent errors."""
    from ml.config import device_info

    t = time.perf_counter()
    try:
        db.execute(text("SELECT 1"))
        db_ok, db_err = True, None
    except Exception as e:  # pragma: no cover - reported, not raised
        db_ok, db_err = False, str(e)[:200]
    db_ms = (time.perf_counter() - t) * 1000
    models = _model_status(db)
    errors = [r for r in RING if r["level"] in ("ERROR", "CRITICAL")][-20:][::-1]
    warnings = sum(1 for r in RING if r["level"] == "WARNING")
    ok = db_ok and all(m["checkpoint_present"] for m in models)
    return {"status": "ok" if ok else "degraded", "api": {"uptime_s": round(time.time() - STARTED, 1),
                                                          "environment": settings.environment},
            "database": {"ok": db_ok, "latency_ms": round(db_ms, 2), "error": db_err,
                         "dialect": settings.database_url.split(":", 1)[0], "schema_mode": settings.schema_mode},
            "models": models, "device": device_info(), "resources": _resources(), "storage": storage.usage(),
            "recent_errors": errors, "warning_count": warnings, "integrations": integrations.status()}


@router.get("/integrations")
def integrations_status(_: User = Depends(require("view_logs"))):
    """Status of outbound adapters (WebSocket, REST, webhook, MQTT; OPC UA / Modbus reported as not implemented)."""
    return integrations.status()


@router.get("/system/logs")
def logs(level: str | None = Query(None, pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"),
         q: str | None = None, limit: int = Query(300, ge=1, le=2000), _: User = Depends(require("view_logs"))):
    rows = list(RING)
    if level:
        order = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        rows = [r for r in rows if order.index(r["level"]) >= order.index(level)]
    if q:
        rows = [r for r in rows if q.lower() in r["message"].lower()]
    return rows[-limit:][::-1]


@router.get("/system/audit")
def audit_log(action: str | None = None, user_email: str | None = None, page: int = Query(1, ge=1),
              page_size: int = Query(50, ge=1, le=500), _: User = Depends(require("view_logs")),
              db: Session = Depends(get_db)):
    q = db.query(AuditEvent)
    if action:
        q = q.filter(AuditEvent.action.like(f"{action[:64]}%"))
    if user_email:
        q = q.filter(AuditEvent.user_email == user_email)
    total = q.count()
    rows = q.order_by(AuditEvent.created_at.desc()).offset((page - 1) * page_size).limit(page_size).all()
    return {"total": total, "items": [{"id": e.id, "created_at": e.created_at.isoformat() + "Z",
                                       "user_email": e.user_email, "action": e.action, "entity_type": e.entity_type,
                                       "entity_id": e.entity_id, "details": e.details, "ip": e.ip} for e in rows]}


@router.get("/settings")
def get_settings_ep(_: User = Depends(require("view")), db: Session = Depends(get_db)):
    return {s.key: s.value for s in db.query(SystemSetting).all() if s.key in EDITABLE_SETTINGS}


@router.put("/settings/{key}")
def put_setting(key: str, body: dict, request: Request, user: User = Depends(require("manage_settings")),
                db: Session = Depends(get_db)):
    if key not in EDITABLE_SETTINGS:
        raise HTTPException(404, "Unknown setting")
    if key == "alert_rules":
        rd = body.get("repeated_defect", {})
        if rd and (int(rd.get("count", 1)) < 1 or int(rd.get("window_minutes", 1)) < 1):
            raise HTTPException(422, "repeated_defect count and window_minutes must be >= 1")
    s = db.get(SystemSetting, key) or SystemSetting(key=key, value={})
    s.value = {**(s.value or {}), **body}
    db.merge(s)
    db.commit()
    audit(db, user, "settings.update", "setting", key, body, request)
    return s.value


# ---------------------------------------------------------------- drift monitoring
def drift_out(c: DriftCheck) -> dict:
    return {"id": c.id, "created_at": c.created_at.isoformat() + "Z", "profile_id": c.profile_id, "window": c.window,
            "data_drift": c.data_drift, "prediction_drift": c.prediction_drift,
            "performance_degradation": c.performance_degradation, "details": c.details}


@router.post("/drift/check")
def drift_check(profile_id: str, window: int = Query(200, ge=20, le=5000), request: Request = None,
                user: User = Depends(require("view")), db: Session = Depends(get_db)):
    c = drift.check(db, get_profile(db, profile_id), window)
    audit(db, user, "drift.check", "profile", profile_id, {"id": c.id}, request)
    return drift_out(c)


@router.get("/drift/checks")
def drift_checks(profile_id: str | None = None, limit: int = Query(50, ge=1, le=500),
                 _: User = Depends(require("view")), db: Session = Depends(get_db)):
    q = db.query(DriftCheck)
    if profile_id:
        q = q.filter(DriftCheck.profile_id == profile_id)
    return [drift_out(c) for c in q.order_by(DriftCheck.created_at.desc()).limit(limit).all()]


# ---------------------------------------------------------------- assistant
class ChatIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[dict] = Field(default_factory=list, max_length=40)


@router.post("/assistant/chat")
def chat(body: ChatIn, user: User = Depends(require("view")), db: Session = Depends(get_db)):
    hist = [{"role": h.get("role"), "content": str(h.get("content", ""))[:4000]} for h in body.history
            if h.get("role") in ("user", "assistant")]
    return assistant.chat(db, user, body.question, hist)


@router.get("/assistant/status")
def assistant_status(_: User = Depends(get_current_user)):
    return {"llm_available": assistant.llm_available(), "model": settings.assistant_model if assistant.llm_available()
            else None, "mode": "llm" if assistant.llm_available() else "offline (rule-based, database-grounded)"}


# ---------------------------------------------------------------- live events
@router.websocket("/ws/events")
async def events(ws: WebSocket):
    """Pushes inspection/alert/job/simulation/training events. Auth: ?token=<JWT>."""
    token = ws.query_params.get("token", "")
    try:
        payload = decode_token(token)
    except HTTPException:
        await ws.close(code=4401)
        return
    db = SessionLocal()
    try:
        u = db.get(User, int(payload["sub"]))
        ok = u and u.is_active and "view" in u.permission_codes and not db.get(RevokedToken, payload.get("jti", ""))
    finally:
        db.close()
    if not ok:
        await ws.close(code=4403)
        return
    await ws.accept()
    q = bus.subscribe()
    try:
        await ws.send_text('{"type":"hello","data":{"time":"%s"}}' % datetime.utcnow().isoformat())
        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=25)
                await ws.send_text(msg)
            except asyncio.TimeoutError:
                await ws.send_text('{"type":"ping","data":{}}')
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        bus.unsubscribe(q)
