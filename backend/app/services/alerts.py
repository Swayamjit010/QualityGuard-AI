"""Alert rules evaluated after each inspection. Thresholds live in system settings ('alert_rules')."""
from __future__ import annotations

from datetime import timedelta

from sqlalchemy.orm import Session

from app.models.entities import Alert, Inspection, SystemSetting, utcnow
from app.services.events import bus

DEFAULT_RULES = {
    "critical_severity_alert": True,
    "unfamiliar_alert": True,
    "review_alert": True,
    "repeated_defect": {"enabled": True, "count": 5, "window_minutes": 60},
}


def rules(db: Session) -> dict:
    s = db.get(SystemSetting, "alert_rules")
    return {**DEFAULT_RULES, **(s.value if s else {})}


def _raise(db: Session, kind, level, title, message, insp: Inspection | None = None, data=None, dedupe_minutes=0):
    if dedupe_minutes and insp is not None:
        since = utcnow() - timedelta(minutes=dedupe_minutes)
        exists = db.query(Alert).filter(Alert.kind == kind, Alert.profile_id == insp.profile_id,
                                        Alert.status == "open", Alert.created_at >= since,
                                        Alert.title == title).first()
        if exists:
            return None
    a = Alert(kind=kind, level=level, title=title, message=message,
              profile_id=insp.profile_id if insp else (data or {}).get("profile_id"),
              inspection_id=insp.id if insp else None, data=data or {})
    db.add(a)
    db.flush()
    bus.publish("alert", {"id": a.id, "kind": kind, "level": level, "title": title, "message": message,
                          "inspection_id": a.inspection_id, "profile_id": a.profile_id})
    return a


def evaluate_inspection(db: Session, insp: Inspection):
    r = rules(db)
    sim = " [SIMULATED]" if insp.is_simulated else ""
    if r["critical_severity_alert"] and insp.severity == "critical":
        _raise(db, "critical_defect", "critical", f"Critical defect detected{sim}",
               f"{insp.profile_id}: {insp.defect_class or 'anomaly'} with critical severity (inspection {insp.id[:8]}).",
               insp)
    if r["unfamiliar_alert"] and insp.unfamiliar:
        _raise(db, "unfamiliar_pattern", "warning", f"Unfamiliar pattern{sim}",
               f"{insp.profile_id}: input outside the supported distribution (inspection {insp.id[:8]}).", insp)
    elif r["review_alert"] and insp.status == "REVIEW REQUIRED":
        _raise(db, "review_required", "info", f"Manual review required{sim}",
               f"{insp.profile_id}: inspection {insp.id[:8]} needs inspector confirmation.", insp)
    rd = r["repeated_defect"]
    if rd.get("enabled") and insp.status == "FAIL":
        since = utcnow() - timedelta(minutes=rd["window_minutes"])
        q = db.query(Inspection).filter(Inspection.profile_id == insp.profile_id, Inspection.status == "FAIL",
                                        Inspection.created_at >= since)
        if insp.defect_class:
            q = q.filter(Inspection.defect_class == insp.defect_class)
        n = q.count()
        if n >= rd["count"]:
            what = insp.defect_class or "anomaly"
            _raise(db, "repeated_defect", "warning", f"Repeated defect: {what} on {insp.profile_id}{sim}",
                   f"{n} rejected items with '{what}' in the last {rd['window_minutes']} min.", insp,
                   {"count": n, "window_minutes": rd["window_minutes"]}, dedupe_minutes=rd["window_minutes"])


def raise_system_alert(db: Session, kind: str, level: str, title: str, message: str, data: dict | None = None):
    a = _raise(db, kind, level, title, message, None, data)
    db.commit()
    return a
