"""Database-backed analytics and grounded insights. Every figure is a SQL aggregate over inspections."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta

import numpy as np
from sqlalchemy import or_, func
from sqlalchemy.orm import Query, Session

from app.models.entities import Alert, HumanFeedback, Inspection, ModelVersion, utcnow

STATUSES = ["PASS", "FAIL", "REVIEW REQUIRED", "INCONCLUSIVE"]


def filtered(db: Session, *, days: int | None = None, start: datetime | None = None, end: datetime | None = None,
             profile_id: str | None = None, defect_class: str | None = None, severity: str | None = None,
             model: str | None = None, status: str | None = None, include_simulated: bool = True,
             source: str | None = None) -> Query:
    q = db.query(Inspection)
    if days:
        q = q.filter(Inspection.created_at >= utcnow() - timedelta(days=days))
    if start:
        q = q.filter(Inspection.created_at >= start)
    if end:
        q = q.filter(Inspection.created_at < end)
    if profile_id:
        q = q.filter(Inspection.profile_id == profile_id)
    if defect_class:
        q = q.filter(Inspection.defect_class == defect_class)
    if severity:
        q = q.filter(Inspection.severity == severity)
    if status:
        q = q.filter(Inspection.status == status)
    if source:
        q = q.filter(Inspection.source == source)
    if not include_simulated:
        q = q.filter(Inspection.is_simulated.is_(False))
    if model:
        q = q.filter(Inspection.models_used.contains(model)) if db.bind.dialect.name == "postgresql" else \
            q.filter(func.json_extract(Inspection.models_used, "$").like(f"%{model}%"))
    return q


def dashboard(db: Session, **f) -> dict:
    q = filtered(db, **f)
    total = q.count()
    by_status = dict(q.with_entities(Inspection.status, func.count()).group_by(Inspection.status).all())
    defective = q.filter(Inspection.defect_present.is_(True)).count()
    avg_lat = q.with_entities(func.avg(Inspection.latency_ms)).scalar() or 0
    recent = q.order_by(Inspection.created_at.desc()).limit(10).all()
    defects = q.filter(Inspection.defect_class.isnot(None)).with_entities(Inspection.defect_class, func.count()) \
        .group_by(Inspection.defect_class).all()
    anomalies = q.filter(Inspection.defect_present.is_(True), Inspection.defect_class.is_(None)).count()
    sim = q.filter(Inspection.is_simulated.is_(True)).count()
    open_alerts = db.query(Alert).filter(Alert.status != "resolved").order_by(Alert.created_at.desc()).limit(8).all()
    models = db.query(ModelVersion).filter(ModelVersion.status == "deployed").all()
    manual = by_status.get("REVIEW REQUIRED", 0) + by_status.get("INCONCLUSIVE", 0)
    defective_v = q.filter(Inspection.status.in_(["PASS", "FAIL"]),
                           or_(Inspection.status == "FAIL", Inspection.defect_present.is_(True))).count()
    return {
        "total": total, "by_status": {s: by_status.get(s, 0) for s in STATUSES},
        # verdict counts (NORMAL / DEFECTIVE / MANUAL INSPECTION REQUIRED); manual cases are never counted as normal
        "verdicts": {"NORMAL": total - manual - defective_v, "DEFECTIVE": defective_v,
                     "MANUAL INSPECTION REQUIRED": manual},
        "normal": total - manual - defective_v, "defective": defective_v, "manual_inspection": manual,
        "defect_present_count": defective, "simulated": sim,
        "review_required": by_status.get("REVIEW REQUIRED", 0) + by_status.get("INCONCLUSIVE", 0),
        "pending_reviews": q.filter(Inspection.review_status == "pending").count(),
        "defect_detection_rate": defective / total if total else None,
        "acceptance_rate": by_status.get("PASS", 0) / total if total else None,
        "avg_latency_ms": float(avg_lat),
        "defect_distribution": sorted([{"name": k, "count": v} for k, v in defects] +
                                      ([{"name": "anomaly (localised)", "count": anomalies}] if anomalies else []),
                                      key=lambda x: -x["count"]),
        "recent": [_row(r) for r in recent],
        "alerts": [{"id": a.id, "kind": a.kind, "level": a.level, "title": a.title, "message": a.message,
                    "status": a.status, "created_at": a.created_at.isoformat() + "Z"} for a in open_alerts],
        "deployed_models": [{"run_id": m.run_id, "task": m.model.task, "architecture": m.model.architecture}
                            for m in models],
        "trend": trend(db, "day", **{**f, "days": f.get("days") or 30}),
        "insights": insights(db, **f),
    }


def _row(r: Inspection) -> dict:
    from ml.quality.rules import verdict_for
    return {"id": r.id, "created_at": r.created_at.isoformat() + "Z", "profile_id": r.profile_id, "status": r.status,
            "verdict": verdict_for(r.status, r.defect_present), "confidence": r.confidence,
            "anomaly_score_ratio": r.anomaly_score_ratio,
            "defect_class": r.defect_class, "severity": r.severity, "quality_index": r.quality_index,
            "latency_ms": r.latency_ms, "is_simulated": r.is_simulated, "source": r.source, "filename": r.filename}


def _bucket(dt: datetime, unit: str) -> str:
    if unit == "hour":
        return dt.strftime("%Y-%m-%d %H:00")
    if unit == "week":
        y, w, _ = dt.isocalendar()
        return f"{y}-W{w:02d}"
    if unit == "month":
        return dt.strftime("%Y-%m")
    return dt.strftime("%Y-%m-%d")


def trend(db: Session, unit: str = "day", **f) -> list[dict]:
    rows = filtered(db, **f).with_entities(Inspection.created_at, Inspection.status, Inspection.defect_present,
                                           Inspection.quality_index, Inspection.latency_ms).all()
    b: dict[str, dict] = defaultdict(lambda: {"total": 0, "PASS": 0, "FAIL": 0, "REVIEW REQUIRED": 0,
                                              "INCONCLUSIVE": 0, "defective": 0, "qi": [], "lat": []})
    for ts, st, dp, qi, lat in rows:
        k = _bucket(ts, unit)
        b[k]["total"] += 1; b[k][st] = b[k].get(st, 0) + 1; b[k]["defective"] += int(bool(dp))
        if qi is not None:
            b[k]["qi"].append(qi)
        b[k]["lat"].append(lat)
    out = []
    for k in sorted(b):
        v = b[k]
        out.append({"period": k, "total": v["total"], "pass": v["PASS"], "fail": v["FAIL"],
                    "review": v["REVIEW REQUIRED"], "inconclusive": v["INCONCLUSIVE"], "defective": v["defective"],
                    "defect_rate": v["defective"] / v["total"], "acceptance_rate": v["PASS"] / v["total"],
                    "avg_quality_index": float(np.mean(v["qi"])) if v["qi"] else None,
                    "avg_latency_ms": float(np.mean(v["lat"]))})
    return out


def defects(db: Session, **f) -> dict:
    q = filtered(db, **f)
    rows = q.with_entities(Inspection.profile_id, Inspection.defect_class, Inspection.severity, Inspection.status,
                           Inspection.defect_present).all()
    by_class = Counter((c or ("anomaly" if dp else None)) for _, c, _, _, dp in rows if dp)
    by_sev = Counter(s for _, _, s, _, _ in rows)
    by_prof = defaultdict(Counter)
    for p, _, _, st, _ in rows:
        by_prof[p][st] += 1
    return {"by_class": [{"name": k, "count": v} for k, v in by_class.most_common()],
            "by_severity": [{"name": s, "count": by_sev.get(s, 0)} for s in ["none", "low", "medium", "high", "critical"]],
            "by_profile": [{"profile_id": p, **{s: c.get(s, 0) for s in STATUSES}, "total": sum(c.values()),
                            "rejection_rate": c.get("FAIL", 0) / sum(c.values())} for p, c in by_prof.items()]}


def quality(db: Session, **f) -> dict:
    q = filtered(db, **f)
    qi = [v for (v,) in q.with_entities(Inspection.quality_index).all() if v is not None]
    hist = np.histogram(qi, bins=10, range=(0, 100))[0].tolist() if qi else [0] * 10
    lat = [v for (v,) in q.with_entities(Inspection.latency_ms).all()]
    conf = [v for (v,) in q.with_entities(Inspection.confidence).all() if v is not None]
    return {"quality_index_histogram": [{"bin": f"{i * 10}-{i * 10 + 10}", "count": c} for i, c in enumerate(hist)],
            "quality_index_mean": float(np.mean(qi)) if qi else None,
            "latency_ms": {"mean": float(np.mean(lat)), "p50": float(np.percentile(lat, 50)),
                           "p95": float(np.percentile(lat, 95))} if lat else None,
            "confidence_histogram": np.histogram(conf, bins=10, range=(0, 1))[0].tolist() if conf else [],
            "ground_truth": ground_truth_stats(db, **f), "throughput": throughput(db, **f)}


def throughput(db: Session, **f) -> dict:
    ts = sorted(t for (t,) in filtered(db, **f).with_entities(Inspection.created_at).all())
    if len(ts) < 2:
        return {"per_hour_last_24h": 0, "max_per_minute": 0}
    last = ts[-1]
    per_min = Counter(t.strftime("%Y%m%d%H%M") for t in ts)
    return {"per_hour_last_24h": sum(1 for t in ts if t >= last - timedelta(hours=24)) / 24,
            "max_per_minute": max(per_min.values())}


def ground_truth_stats(db: Session, **f) -> dict:
    """False positive / negative statistics where ground truth exists (dataset-labelled simulation images, or
    inspector final decisions). Only inspections WITH ground truth are counted."""
    rows = filtered(db, **f).all()
    tp = fp = tn = fn = 0
    n_gt = 0
    cls_correct = cls_total = 0
    for r in rows:
        truth_defect = None
        if r.feedback:
            fbk = r.feedback[-1]
            truth_defect = fbk.final_decision == "reject"
            if fbk.corrected_class or fbk.verdict == "confirm":
                cls_total += 1
                cls_correct += int(fbk.verdict == "confirm" and not fbk.corrected_class)
        elif r.ground_truth is not None:
            truth_defect = r.ground_truth not in ("good", "normal")
            if r.defect_class and r.ground_truth not in ("good", "normal"):
                cls_total += 1
                cls_correct += int(r.defect_class == r.ground_truth)
        if truth_defect is None or r.status not in ("PASS", "FAIL"):
            continue
        n_gt += 1
        pred = r.status == "FAIL"
        tp += pred and truth_defect; fp += pred and not truth_defect
        tn += (not pred) and (not truth_defect); fn += (not pred) and truth_defect
    return {"evaluated": n_gt, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "false_negative_rate": fn / (tp + fn) if tp + fn else None,
            "false_positive_rate": fp / (fp + tn) if fp + tn else None,
            "class_accuracy": cls_correct / cls_total if cls_total else None, "class_evaluated": cls_total,
            "note": "Computed only over auto-decided (PASS/FAIL) inspections that have dataset labels or inspector decisions."}


def forecast(db: Session, horizon: int = 7, **f) -> dict:
    t = trend(db, "day", **{**f, "days": f.get("days") or 90})
    if len(t) < 14:
        return {"available": False, "reason": f"Forecasting needs >= 14 days of history; {len(t)} day(s) available.",
                "history": t}
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    # fill missing days with 0-count days
    days = [datetime.strptime(x["period"], "%Y-%m-%d") for x in t]
    idx = {d: x for d, x in zip(days, t)}
    full = [days[0] + timedelta(days=i) for i in range((days[-1] - days[0]).days + 1)]
    y = np.array([idx[d]["defect_rate"] if d in idx else np.nan for d in full])
    y = np.where(np.isnan(y), np.nanmean(y), y)
    split = max(7, len(y) // 5)
    m = ExponentialSmoothing(y[:-split], trend="add", damped_trend=True).fit()
    bt = m.forecast(split)
    mae = float(np.mean(np.abs(bt - y[-split:])))
    naive = float(np.mean(np.abs(y[-split - 1:-1] - y[-split:])))
    model = ExponentialSmoothing(y, trend="add", damped_trend=True).fit()
    fc = np.clip(model.forecast(horizon), 0, 1)
    return {"available": True, "method": "Holt damped-trend exponential smoothing on daily defect rate",
            "backtest_mae": mae, "naive_mae": naive, "beats_naive": mae < naive,
            "history": [{"period": d.strftime("%Y-%m-%d"), "defect_rate": float(v)} for d, v in zip(full, y)],
            "forecast": [{"period": (full[-1] + timedelta(days=i + 1)).strftime("%Y-%m-%d"), "defect_rate": float(v)}
                         for i, v in enumerate(fc)],
            "caveat": "Statistical extrapolation of past inspection outcomes; not a prediction of manufacturing failures."}


def insights(db: Session, **f) -> list[dict]:
    """Short, grounded observations. Each includes the query facts it was derived from."""
    out = []
    q = filtered(db, **f)
    total = q.count()
    if total == 0:
        return [{"text": "No inspections recorded for the selected filters yet.", "facts": {"total": 0}}]
    now = utcnow()
    week = filtered(db, **{**f, "days": None}).filter(Inspection.created_at >= now - timedelta(days=7))
    prev = filtered(db, **{**f, "days": None}).filter(Inspection.created_at >= now - timedelta(days=14),
                                                      Inspection.created_at < now - timedelta(days=7))
    top = q.filter(Inspection.defect_class.isnot(None)).with_entities(Inspection.defect_class, func.count()) \
        .group_by(Inspection.defect_class).order_by(func.count().desc()).first()
    if top:
        out.append({"text": f"Most frequent defect category: '{top[0]}' ({top[1]} of {total} inspections).",
                    "facts": {"class": top[0], "count": top[1], "total": total}})
    cw = dict(week.filter(Inspection.defect_class.isnot(None)).with_entities(Inspection.defect_class, func.count())
              .group_by(Inspection.defect_class).all())
    cp = dict(prev.filter(Inspection.defect_class.isnot(None)).with_entities(Inspection.defect_class, func.count())
              .group_by(Inspection.defect_class).all())
    inc = sorted(((k, v - cp.get(k, 0)) for k, v in cw.items()), key=lambda x: -x[1])
    if inc and inc[0][1] > 0 and cp:
        out.append({"text": f"'{inc[0][0]}' increased by {inc[0][1]} occurrences vs. the previous 7 days "
                            f"({cp.get(inc[0][0], 0)} → {cw[inc[0][0]]}).",
                    "facts": {"class": inc[0][0], "this_week": cw[inc[0][0]], "prev_week": cp.get(inc[0][0], 0)}})
    d = defects(db, **f)["by_profile"]
    if d:
        worst = max(d, key=lambda x: x["rejection_rate"])
        out.append({"text": f"Highest rejection rate: profile '{worst['profile_id']}' "
                            f"({worst['FAIL']}/{worst['total']} = {worst['rejection_rate']:.1%} FAIL).", "facts": worst})
    t = trend(db, "day", **{**f, "days": f.get("days") or 30})
    if len(t) >= 2:
        peak = max(t, key=lambda x: x["defective"])
        out.append({"text": f"Peak defect day: {peak['period']} with {peak['defective']} defective of {peak['total']}.",
                    "facts": peak})
    rv = q.filter(Inspection.review_status == "pending").count()
    if rv:
        out.append({"text": f"{rv} inspection(s) are waiting for manual review.", "facts": {"pending": rv}})
    sim = q.filter(Inspection.is_simulated.is_(True)).count()
    if sim:
        out.append({"text": f"{sim} of {total} inspections are simulated production-line events.",
                    "facts": {"simulated": sim, "total": total}})
    gt = ground_truth_stats(db, **f)
    if gt["evaluated"]:
        out.append({"text": f"On {gt['evaluated']} inspections with ground truth: false-negative rate "
                            f"{(gt['false_negative_rate'] or 0):.1%}, false-positive rate {(gt['false_positive_rate'] or 0):.1%}.",
                    "facts": gt})
    return out


def model_prediction_distribution(db: Session, **f) -> list[dict]:
    from app.models.entities import Prediction
    q = filtered(db, **f).join(Prediction, Prediction.inspection_id == Inspection.id) \
        .with_entities(Prediction.model_run_id, Prediction.predicted_class, func.count()) \
        .group_by(Prediction.model_run_id, Prediction.predicted_class)
    out = defaultdict(dict)
    for m, c, n in q.all():
        out[m][c] = n
    return [{"model": m, **v} for m, v in out.items()]


def feedback_stats(db: Session) -> dict:
    fb = db.query(HumanFeedback).all()
    return {"total": len(fb), "confirmed": sum(f.verdict == "confirm" for f in fb),
            "rejected": sum(f.verdict == "reject" for f in fb)}


def _aggregate_rows(rows: list[Inspection]) -> dict:
    st = Counter(r.status for r in rows)
    return {"total": len(rows), "status": dict(st),
            "defects": dict(Counter(r.defect_class or "anomaly" for r in rows if r.defect_present)),
            "acceptance_rate": st.get("PASS", 0) / len(rows) if rows else None,
            "avg_latency_ms": float(np.mean([r.latency_ms for r in rows])) if rows else None}
