"""Dashboard + analytics endpoints. All figures are SQL aggregates (see services/analytics.py)."""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.models.entities import User
from app.security.auth import require
from app.services import analytics

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


def filters(days: int | None = Query(None, ge=1, le=3650), start: datetime | None = None,
            end: datetime | None = None, profile_id: str | None = None, defect_class: str | None = None,
            severity: str | None = None, model: str | None = None, status: str | None = None,
            include_simulated: bool = True, source: str | None = None) -> dict:
    return {"days": days, "start": start, "end": end, "profile_id": profile_id, "defect_class": defect_class,
            "severity": severity, "model": model, "status": status, "include_simulated": include_simulated,
            "source": source}


@router.get("/dashboard")
def dashboard(f: dict = Depends(filters), _: User = Depends(require("view")), db: Session = Depends(get_db)):
    return analytics.dashboard(db, **f)


@router.get("/trend")
def trend(unit: str = Query("day", pattern="^(hour|day|week|month)$"), f: dict = Depends(filters),
          _: User = Depends(require("view")), db: Session = Depends(get_db)):
    return analytics.trend(db, unit, **f)


@router.get("/defects")
def defects(f: dict = Depends(filters), _: User = Depends(require("view")), db: Session = Depends(get_db)):
    return analytics.defects(db, **f)


@router.get("/quality")
def quality(f: dict = Depends(filters), _: User = Depends(require("view")), db: Session = Depends(get_db)):
    return analytics.quality(db, **f)


@router.get("/forecast")
def forecast(horizon: int = Query(7, ge=1, le=60), f: dict = Depends(filters), _: User = Depends(require("view")),
             db: Session = Depends(get_db)):
    return analytics.forecast(db, horizon, **f)


@router.get("/insights")
def insights(f: dict = Depends(filters), _: User = Depends(require("view")), db: Session = Depends(get_db)):
    return analytics.insights(db, **f)


@router.get("/models")
def model_distribution(f: dict = Depends(filters), _: User = Depends(require("view")), db: Session = Depends(get_db)):
    return analytics.model_prediction_distribution(db, **f)


@router.get("/feedback")
def feedback(_: User = Depends(require("view")), db: Session = Depends(get_db)):
    return analytics.feedback_stats(db)


@router.get("/anomaly-overview")
def anomaly_overview(profile_id: str = Query(...), days: int | None = Query(None, ge=1, le=3650),
                     _: User = Depends(require("view")), db: Session = Depends(get_db)):
    """Anomaly Detection Center: the deployed thresholds and validated metrics of the profile's anomaly model
    (read from its training record), plus the distribution of anomaly scores in stored inspections."""
    import json
    from datetime import timedelta

    from app.models.entities import Inspection
    from app.services.inspection_service import summary
    from app.services.profiles import get_profile
    from ml.config import CHECKPOINT_DIR

    prof = get_profile(db, profile_id)
    cfg = prof.config or {}
    run_id = (cfg.get("models") or {}).get("anomaly")
    model = None
    if run_id and (CHECKPOINT_DIR / run_id / "metrics.json").exists():
        m = json.loads((CHECKPOINT_DIR / run_id / "metrics.json").read_text())
        engines = {}
        for name, e in m.get("engines", {}).items():
            engines[name] = {"deployed_threshold": e.get("deployed_threshold", e.get("threshold_recall95_val")),
                             "threshold_rule": e.get("threshold_rule") or
                             "lowest validation score that keeps >= 95 % recall on defective items",
                             "f1_threshold_val": e.get("threshold_f1_val"), "pixel_threshold": e.get("pixel_threshold_val"),
                             "good_val_score_mean": (e.get("score_stats_val_good") or {}).get("mean"),
                             "good_val_score_max": (e.get("score_stats_val_good") or {}).get("max"),
                             "test": e.get("test"), "test_pixel": e.get("test_pixel"),
                             "per_defect_type": e.get("test_per_defect_type")}
        model = {"run_id": run_id, "category": m.get("category"), "backbone": m.get("backbone"),
                 "split_sizes": m.get("split_sizes"), "engines": engines, "fusion": m.get("fusion"),
                 "novelty": m.get("unfamiliar_pattern"), "primary_engine": (cfg.get("pipeline") or {}).get(
                     "primary_anomaly_engine"), "completed_at": m.get("completed_at")}

    q = db.query(Inspection).filter(Inspection.profile_id == profile_id, Inspection.anomaly_score_ratio.isnot(None))
    if days:
        q = q.filter(Inspection.created_at >= datetime.utcnow() - timedelta(days=days))
    rows = q.with_entities(Inspection.anomaly_score_ratio, Inspection.status, Inspection.unfamiliar).all()
    edges = [round(0.1 * i, 1) for i in range(0, 31)]  # score / threshold, 0.0 .. 3.0 (last bin open-ended)
    hist = [{"from": edges[i], "to": edges[i + 1] if i + 1 < len(edges) else None, "PASS": 0, "FAIL": 0,
             "REVIEW REQUIRED": 0, "INCONCLUSIVE": 0} for i in range(len(edges))]
    for ratio, status, _unf in rows:
        b = hist[min(len(edges) - 1, int(max(0.0, ratio) / 0.1))]
        b[status if status in b else "INCONCLUSIVE"] += 1
    top = q.order_by(Inspection.anomaly_score_ratio.desc()).limit(12).all()
    return {"profile_id": profile_id, "profile_name": cfg.get("name"), "model": model,
            "inspections": len(rows), "above_threshold": sum(1 for r in rows if r[0] >= 1.0),
            "unfamiliar": sum(1 for r in rows if r[2]), "score_ratio_histogram": hist,
            "most_anomalous": [summary(i) for i in top]}
