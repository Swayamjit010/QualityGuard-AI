"""PDF / CSV / JSON report generation. Every number comes from DB records or metric files."""
from __future__ import annotations

import csv
import io
import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from reportlab.graphics.shapes import Drawing, Line, Polygon, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)
from sqlalchemy.orm import Session

from app.models.entities import (DriftCheck, HumanFeedback, Inspection, ModelVersion, ProductionBatch, Report,
                                 User, utcnow)
from app.services.storage import storage
from ml.config import CHECKPOINT_DIR, METRICS_DIR

NAVY = colors.HexColor("#0f1b2d")
BLUE = colors.HexColor("#1d6fe0")
CYAN = colors.HexColor("#14b8c4")
GREY = colors.HexColor("#5b6675")
STATUS_COLORS = {"PASS": colors.HexColor("#16a34a"), "FAIL": colors.HexColor("#dc2626"),
                 "REVIEW REQUIRED": colors.HexColor("#d97706"), "INCONCLUSIVE": colors.HexColor("#6b7280")}

ss = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=17, textColor=NAVY, spaceAfter=4)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12.5, textColor=BLUE, spaceBefore=8, spaceAfter=4)
BODY = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=12, alignment=TA_LEFT)
SMALL = ParagraphStyle("s", parent=BODY, fontSize=7.5, textColor=GREY, leading=10)

LIMITATIONS = [
    "QualityGuard AI is an AI-assisted inspection tool. It is not certified for industrial production use and does "
    "not replace qualified inspectors or established quality-control procedures.",
    "Predictions are only valid for images resembling the training distribution of the selected product profile.",
    "Explanation maps (Grad-CAM family) show regions that influenced the classifier; they are not defect boundaries.",
    "Pixel measurements are not physical dimensions unless the profile defines a validated camera calibration.",
    "Severity levels and acceptance rules come from the configured product profile, not from a universal standard.",
    "Only visible surface/appearance defects of the trained component categories are assessed. Internal bearing "
    "failures, vibration faults, overheating, lubrication and electrical faults cannot be diagnosed from a photograph.",
]


def logo(size=28) -> Drawing:
    d = Drawing(size, size)
    s = size / 28.0
    d.add(Polygon([14 * s, 27 * s, 26 * s, 22 * s, 24 * s, 9 * s, 14 * s, 1 * s, 4 * s, 9 * s, 2 * s, 22 * s],
                  fillColor=NAVY, strokeColor=CYAN, strokeWidth=1.2 * s))
    d.add(Line(8 * s, 14 * s, 12.5 * s, 9.5 * s, strokeColor=CYAN, strokeWidth=2.4 * s))
    d.add(Line(12.5 * s, 9.5 * s, 20.5 * s, 18.5 * s, strokeColor=CYAN, strokeWidth=2.4 * s))
    return d


def _header_footer(title: str):
    def draw(c, doc):
        c.saveState()
        c.setFillColor(NAVY)
        c.rect(0, A4[1] - 18 * mm, A4[0], 18 * mm, fill=1, stroke=0)
        lg = logo(30)
        lg.drawOn(c, 14 * mm, A4[1] - 15.5 * mm)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 13)
        c.drawString(27 * mm, A4[1] - 10 * mm, "QualityGuard AI")
        c.setFont("Helvetica", 8.5)
        c.setFillColor(colors.HexColor("#9fd8ff"))
        c.drawString(27 * mm, A4[1] - 14.3 * mm, "Industrial Quality Intelligence Platform")
        c.setFillColor(colors.white)
        c.setFont("Helvetica", 9)
        c.drawRightString(A4[0] - 14 * mm, A4[1] - 11 * mm, title)
        c.setFillColor(GREY)
        c.setFont("Helvetica", 7)
        c.drawString(14 * mm, 9 * mm, f"Generated {utcnow():%Y-%m-%d %H:%M} UTC · AI-assisted inspection — "
                                      "not a certified quality decision")
        c.drawRightString(A4[0] - 14 * mm, 9 * mm, f"Page {doc.page}")
        c.restoreState()
    return draw


def _doc(buf, title):
    return SimpleDocTemplate(buf, pagesize=A4, leftMargin=14 * mm, rightMargin=14 * mm, topMargin=24 * mm,
                             bottomMargin=16 * mm, title=title, author="QualityGuard AI")


def _table(rows, widths=None, header=True, zebra=True, font=8.3):
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    st = [("FONT", (0, 0), (-1, -1), "Helvetica", font), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d5dbe3")),
          ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
               ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", font)]
    if zebra:
        for i in range(1 if header else 0, len(rows)):
            if i % 2 == 0:
                st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f3f6fa")))
    t.setStyle(TableStyle(st))
    return t


def _kv(pairs, w1=48 * mm, w2=None):
    return _table([[Paragraph(f"<b>{k}</b>", BODY), Paragraph(str(v), BODY)] for k, v in pairs],
                  [w1, w2 or (182 * mm - w1)], header=False)


def _img(key: str | None, width: float, max_h: float = 70 * mm):
    if not key:
        return None
    p = storage.resolve(key) if not Path(key).is_absolute() else Path(key)
    if not p.exists():
        return None
    from PIL import Image as PImg
    with PImg.open(p) as im:
        w, h = im.size
    sc = min(width / w, max_h / h)
    return Image(str(p), w * sc, h * sc)


def _fmt(v, nd=3):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _save(db: Session, kind: str, title: str, data: bytes, fmt: str, user: User | None, params: dict,
          inspection_id: str | None = None) -> Report:
    r = Report(kind=kind, title=title, format=fmt, params=params, created_by=user.id if user else None,
               inspection_id=inspection_id)
    db.add(r)
    db.flush()
    key = f"reports/{utcnow():%Y/%m}/{kind}_{r.id}.{fmt}"
    storage.put_bytes(key, data)
    r.path, r.size_bytes = key, len(data)
    db.commit()
    return r


# ------------------------------------------------------------------ inspection report
def inspection_pdf(db: Session, insp: Inspection, user: User | None) -> Report:
    res = insp.result or {}
    buf = io.BytesIO()
    doc = _doc(buf, f"Inspection {insp.id}")
    el = [Paragraph("Inspection Report", H1),
          Paragraph(f"Inspection ID <b>{insp.id}</b>", BODY), Spacer(1, 4)]
    from ml.quality.rules import VERDICT_TEXT, verdict_for
    verdict = verdict_for(insp.status, insp.defect_present)
    vcol = {"NORMAL": colors.HexColor("#16a34a"), "DEFECTIVE": colors.HexColor("#dc2626")}.get(
        verdict, colors.HexColor("#d97706"))
    vt = Table([[Paragraph(f"<font color='white' size='13'><b>{verdict}</b></font>", BODY),
                 Paragraph(VERDICT_TEXT[verdict], BODY)]], [62 * mm, 120 * mm])
    vt.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, 0), vcol), ("BOX", (0, 0), (-1, -1), 1, vcol),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 6),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    el += [Paragraph("Inspection result", H2), vt, Spacer(1, 4)]
    clf0, an0 = res.get("engines", {}).get("classification"), res.get("engines", {}).get("anomaly")
    el.append(_kv([
        ("Predicted defect class", insp.defect_class or ("anomaly (not classified)" if insp.defect_present else "—")),
        ("Classification confidence", _fmt(clf0["confidence"]) if clf0 else "n/a (no classifier for this category)"),
        ("Anomaly score", f"{an0['score_ratio'][an0['primary_engine']]:.3f} x validated threshold "
                          f"({an0['primary_engine']})" if an0 else "n/a"),
        ("Processing time", f"{insp.latency_ms:.0f} ms")]))
    rec = res.get("recognition")
    if rec:
        if rec.get("available") is False:
            rtxt = f"Recogniser unavailable: {rec.get('note', '')}"
        elif rec.get("title"):
            rtxt = (f"{rec['title']} (probability {rec.get('probability', 0):.2f}, appearance distance "
                    f"{rec.get('distance', 0):.3f} / limit {rec.get('distance_threshold', 0):.3f}; "
                    f"{'auto-selected' if rec.get('mode') == 'auto' else 'user-selected category checked'})")
        else:
            rtxt = rec.get("note", "—")
        el.append(_kv([("Component recognition", rtxt)] + ([("Warning", rec["block"])] if rec.get("block") else [])))
    el.append(Paragraph("Line disposition (product-profile rules)", H2))
    col = STATUS_COLORS.get(insp.status, GREY)
    badge = Table([[Paragraph(f"<font color='white'><b>{insp.status}</b></font>", BODY),
                    Paragraph(f"Severity: <b>{insp.severity.upper()}</b> &nbsp; · &nbsp; Quality index: "
                              f"<b>{_fmt(insp.quality_index, 0)}</b> &nbsp; · &nbsp; Reliability: "
                              f"<b>{(insp.reliability or '—').upper()}</b>", BODY)]], [42 * mm, 140 * mm])
    badge.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, 0), col), ("BOX", (0, 0), (-1, -1), 0.5, col),
                               ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    el += [badge, Spacer(1, 6)]
    if res.get("decision", {}).get("message"):
        el.append(Paragraph(f"<font color='#d97706'><b>{res['decision']['message']}</b></font>", BODY))
    el.append(Paragraph("Product & traceability", H2))
    el.append(_kv([("Timestamp (UTC)", f"{insp.created_at:%Y-%m-%d %H:%M:%S}"),
                   ("Product profile", f"{insp.profile_id} (rules v{insp.profile_version})"),
                   ("Source / file", f"{insp.source} / {insp.filename}"),
                   ("Batch / line / serial", f"{insp.batch_id or '—'} / {insp.line_id or '—'} / {insp.product_serial or '—'}"),
                   ("Simulated data", "YES — simulated production-line event" if insp.is_simulated else "No"),
                   ("Models", ", ".join(insp.models_used) or "—"),
                   ("Inference duration", f"{insp.latency_ms:.0f} ms"),
                   ("Image", f"{res.get('input', {}).get('width')}x{res.get('input', {}).get('height')} px")]))
    vis = res.get("visualizations", [])
    adir = insp.artifact_dir
    pics = [(k, f"{adir}/{k}.png") for k in ("original", "segmentation_overlay", "bounding_boxes",
                                              "anomaly_patchcore_overlay", "segmentation_unet_overlay", "gradcam_overlay",
                                              "gradcam_pp_overlay", "defect_zoom")
            if k in vis]
    if pics:
        el.append(Paragraph("Images (generated from the actual model outputs)", H2))
        cells, caps = [], []
        for k, key in pics[:3]:
            cells.append(_img(key, 58 * mm, 58 * mm)); caps.append(Paragraph(k.replace("_", " "), SMALL))
        el.append(Table([cells, caps], [61 * mm] * len(cells)))
        if len(pics) > 3:
            cells, caps = [], []
            for k, key in pics[3:6]:
                cells.append(_img(key, 58 * mm, 58 * mm)); caps.append(Paragraph(k.replace("_", " "), SMALL))
            el.append(Table([cells, caps], [61 * mm] * len(cells)))
        if len(pics) > 6:
            cells, caps = [], []
            for k, key in pics[6:9]:
                cells.append(_img(key, 58 * mm, 58 * mm)); caps.append(Paragraph(k.replace("_", " "), SMALL))
            el.append(Table([cells, caps], [61 * mm] * len(cells)))
        el.append(Paragraph("Grad-CAM maps show what influenced the classifier, not defect boundaries. "
                            + ("The segmentation overlay is the supervised U-Net defect mask."
                               if res.get("engines", {}).get("segmentation") else
                               "Segmentation overlays are thresholded anomaly maps (region estimates)."), SMALL))

    clf = res.get("engines", {}).get("classification")
    if clf:
        el.append(Paragraph("Classification (Engine A)", H2))
        rows = [["Class", "Ensemble probability"] + [m["arch"] for m in clf["ensemble"]]]
        for c, p in sorted(clf["probabilities"].items(), key=lambda kv: -kv[1]):
            rows.append([c, f"{p:.4f}"] + [f"{m['probabilities'].get(c, 0):.4f}" for m in clf["ensemble"]])
        el.append(_table(rows))
        el.append(Spacer(1, 3))
        el.append(_kv([("Predicted class", clf["predicted_class"]), ("Calibrated confidence", _fmt(clf["confidence"])),
                       ("Top-1/2 margin", _fmt(clf["margin"])), ("Ensemble agreement", "yes" if clf["members_agree"] else "NO"),
                       ("Out-of-distribution votes", f"{clf['ood_votes']}/{len(clf['ensemble'])}"),
                       ("Reliability notes", "; ".join(clf.get("reliability_notes", [])))]))
    an = res.get("engines", {}).get("anomaly")
    if an:
        el.append(Paragraph("Anomaly detection (Engine B) & unfamiliar-pattern check (Engine C)", H2))
        rows = [["Engine", "Score", "Validated threshold", "Score / threshold", "Vote"]]
        for k, s in an["scores"].items():
            rows.append([k, f"{s:.4f}", f"{an['thresholds'][k]:.4f}", f"{an['score_ratio'][k]:.3f}",
                         "anomalous" if an["votes"][k] else "normal"])
        el.append(_table(rows))
        nov = res.get("engines", {}).get("novelty", {})
        el.append(_kv([("Engines disagree / in margin", f"{an['engines_disagree']} / {an['within_review_margin']}"),
                       ("Novelty score", f"{_fmt(nov.get('score'))} (threshold {_fmt(nov.get('threshold'))})")]))
    m = res.get("measurements")
    if m:
        el.append(Paragraph("Defect measurements", H2))
        unit = "calibrated" if m["calibrated"] else "pixels only — no camera calibration configured"
        el.append(Paragraph(f"Units: {unit}.", SMALL))
        rows = [["#", "Area px", "BBox (x,y,w,h) px", "Perimeter px", "Aspect", "Solidity"] +
                (["Area mm²"] if m["calibrated"] else [])]
        for r in m["regions"][:12]:
            rows.append([r["id"], f"{r['area_px']:.0f}", str(r["bbox_px"]), f"{r['perimeter_px']:.0f}",
                         f"{r['aspect_ratio']:.2f}", _fmt(r["solidity"], 2)] +
                        ([f"{r['area_mm2']:.2f}"] if m["calibrated"] else []))
        el.append(_table(rows))
        el.append(Paragraph(f"Regions: {m['num_regions']} · total area {m['total_defect_area_px']:.0f} px · "
                            f"area ratio {m['defect_area_ratio']:.3%}", BODY))
    sev = res.get("severity", {})
    dec = res.get("decision", {})
    el.append(Paragraph("Severity, rules and decision", H2))
    el.append(_kv([("AI severity assessment", sev.get("level", "—")),
                   ("Severity basis", "<br/>".join(sev.get("basis", [])) or "—"),
                   ("Configured rules applied", ", ".join(dec.get("rules_applied", []))),
                   ("Rule decision", f"{dec.get('status')} — " + " ".join(dec.get("reasons", []))),
                   ("Quality index", f"{_fmt(dec.get('quality_index'), 0)} ({dec.get('quality_index_note', '')})"),
                   ("Recommendation", dec.get("recommendation", "")),
                   ("Human decision", insp.human_decision or "pending" if insp.manual_review_required else
                    (insp.human_decision or "not required"))]))
    if insp.feedback:
        el.append(Paragraph("Inspector feedback", H2))
        rows = [["When", "Reviewer", "Verdict", "Corrected class", "Corrected severity", "Final", "Comment"]]
        for f in insp.feedback:
            rows.append([f"{f.created_at:%Y-%m-%d %H:%M}", f.reviewer.email, f.verdict, f.corrected_class or "—",
                         f.corrected_severity or "—", f.final_decision, Paragraph(f.comment or "", SMALL)])
        el.append(_table(rows, [24 * mm, 34 * mm, 15 * mm, 25 * mm, 22 * mm, 14 * mm, 48 * mm]))
    el.append(Paragraph("Technical limitations", H2))
    for line in LIMITATIONS:
        el.append(Paragraph("• " + line, SMALL))
    doc.build(el, onFirstPage=_header_footer("Inspection report"), onLaterPages=_header_footer("Inspection report"))
    return _save(db, "inspection", f"Inspection {insp.id[:8]} — {verdict}", buf.getvalue(), "pdf", user,
                 {"inspection_id": insp.id}, insp.id)


# ------------------------------------------------------------------ aggregate reports
def _aggregate(rows: list[Inspection]) -> dict:
    st = Counter(r.status for r in rows)
    return {"total": len(rows), "status": dict(st), "defects": dict(Counter(r.defect_class for r in rows if r.defect_class)),
            "severity": dict(Counter(r.severity for r in rows)),
            "avg_latency_ms": sum(r.latency_ms for r in rows) / len(rows) if rows else 0,
            "acceptance_rate": st.get("PASS", 0) / len(rows) if rows else 0}


def _agg_section(el, agg):
    el.append(_kv([("Inspections", agg["total"]), ("PASS / FAIL / REVIEW / INCONCLUSIVE",
                    " / ".join(str(agg["status"].get(s, 0)) for s in ("PASS", "FAIL", "REVIEW REQUIRED", "INCONCLUSIVE"))),
                   ("Acceptance rate", f"{agg['acceptance_rate']:.1%}"),
                   ("Average inference latency", f"{agg['avg_latency_ms']:.0f} ms")]))
    if agg["defects"]:
        el.append(Paragraph("Defect categories", H2))
        el.append(_table([["Defect", "Count"]] + [[k, v] for k, v in sorted(agg["defects"].items(), key=lambda x: -x[1])]))
    el.append(Paragraph("Severity distribution", H2))
    el.append(_table([["Severity", "Count"]] + [[k, v] for k, v in agg["severity"].items()]))


def _insp_table(el, rows, limit=300):
    el.append(Paragraph("Inspections", H2))
    t = [["Time", "ID", "Profile", "File", "Status", "Defect", "Severity", "QI", "Conf/ratio"]]
    for r in rows[:limit]:
        t.append([f"{r.created_at:%m-%d %H:%M:%S}", r.id[:8], r.profile_id, Paragraph(r.filename[:40], SMALL), r.status,
                  r.defect_class or "—", r.severity, _fmt(r.quality_index, 0),
                  _fmt(r.confidence if r.confidence is not None else r.anomaly_score_ratio, 2)])
    el.append(_table(t, [22 * mm, 15 * mm, 19 * mm, 36 * mm, 24 * mm, 22 * mm, 14 * mm, 10 * mm, 18 * mm], font=7))
    if len(rows) > limit:
        el.append(Paragraph(f"... {len(rows) - limit} more rows in CSV export.", SMALL))


def batch_pdf(db: Session, batch: ProductionBatch, user: User | None) -> Report:
    rows = db.query(Inspection).filter_by(batch_id=batch.id).order_by(Inspection.created_at).all()
    buf = io.BytesIO()
    doc = _doc(buf, f"Batch {batch.name}")
    el = [Paragraph("Batch Inspection Report", H1),
          Paragraph(f"Batch <b>{batch.name}</b> ({batch.id}) · source {batch.source} · profile {batch.profile_id}"
                    + (" · <b>SIMULATED DATA</b>" if batch.is_simulated else ""), BODY), Spacer(1, 4)]
    _agg_section(el, _aggregate(rows))
    _insp_table(el, rows)
    doc.build(el, onFirstPage=_header_footer("Batch report"), onLaterPages=_header_footer("Batch report"))
    return _save(db, "batch", f"Batch {batch.name}", buf.getvalue(), "pdf", user, {"batch_id": batch.id})


def period_pdf(db: Session, period: str, user: User | None, profile_id: str | None = None,
               end: datetime | None = None, include_simulated: bool = True) -> Report:
    end = end or utcnow()
    start = end - {"daily": timedelta(days=1), "weekly": timedelta(days=7), "monthly": timedelta(days=30)}[period]
    q = db.query(Inspection).filter(Inspection.created_at >= start, Inspection.created_at < end)
    if profile_id:
        q = q.filter(Inspection.profile_id == profile_id)
    if not include_simulated:
        q = q.filter(Inspection.is_simulated.is_(False))
    rows = q.order_by(Inspection.created_at).all()
    buf = io.BytesIO()
    doc = _doc(buf, f"{period} quality report")
    sim = sum(r.is_simulated for r in rows)
    el = [Paragraph(f"{period.capitalize()} Quality Report", H1),
          Paragraph(f"Period {start:%Y-%m-%d %H:%M} → {end:%Y-%m-%d %H:%M} UTC · profile {profile_id or 'all'} · "
                    f"{sim} of {len(rows)} inspections are SIMULATED production-line events", BODY), Spacer(1, 4)]
    _agg_section(el, _aggregate(rows))
    byp = Counter((r.profile_id, r.status) for r in rows)
    profs = sorted({r.profile_id for r in rows})
    if profs:
        el.append(Paragraph("By product profile", H2))
        el.append(_table([["Profile", "PASS", "FAIL", "REVIEW", "INCONCLUSIVE"]] +
                         [[p] + [byp.get((p, s), 0) for s in ("PASS", "FAIL", "REVIEW REQUIRED", "INCONCLUSIVE")]
                          for p in profs]))
    _insp_table(el, rows, 150)
    doc.build(el, onFirstPage=_header_footer(f"{period} report"), onLaterPages=_header_footer(f"{period} report"))
    return _save(db, period, f"{period.capitalize()} quality report {end:%Y-%m-%d}", buf.getvalue(), "pdf", user,
                 {"period": period, "profile_id": profile_id, "start": start.isoformat(), "end": end.isoformat()})


def model_eval_pdf(db: Session, mv: ModelVersion, user: User | None) -> Report:
    d = CHECKPOINT_DIR / mv.run_id
    m = json.loads((d / "metrics.json").read_text())
    buf = io.BytesIO()
    doc = _doc(buf, f"Model {mv.run_id}")
    el = [Paragraph("Model Evaluation Report", H1),
          Paragraph(f"<b>{mv.run_id}</b> · status {mv.status} · dataset {mv.dataset_name}@{mv.dataset_version}", BODY)]
    if mv.model.task == "classification":
        t = m["test"]
        el.append(Paragraph("Held-out test metrics", H2))
        el.append(_kv([("Test samples", t["num_samples"]), ("Accuracy", _fmt(t["accuracy"], 4)),
                       ("Macro precision / recall / F1", f"{t['precision_macro']:.4f} / {t['recall_macro']:.4f} / {t['f1_macro']:.4f}"),
                       ("Weighted F1", _fmt(t["f1_weighted"], 4)), ("ROC-AUC (macro OvR)", _fmt(t.get("roc_auc_macro_ovr"), 4)),
                       ("PR-AUC (macro)", _fmt(t.get("pr_auc_macro"), 4)),
                       ("ECE calibrated / uncalibrated", f"{t['ece']:.4f} / {t['ece_uncalibrated']:.4f} (T={t['temperature']:.3f})"),
                       ("Epochs / training time", f"{m['epochs_run']} / {m['train_time_s'] / 60:.1f} min"),
                       ("Latency (median, 1 img)", f"{m['benchmark']['latency_ms_median']:.1f} ms on {m['benchmark']['device']}"),
                       ("Parameters / checkpoint", f"{m['benchmark']['params'] / 1e6:.2f} M / {m['checkpoint_size_mb']:.1f} MB")]))
        el.append(Paragraph("Per-class performance", H2))
        el.append(_table([["Class", "Precision", "Recall", "F1", "Support"]] +
                         [[c, f"{v['precision']:.3f}", f"{v['recall']:.3f}", f"{v['f1']:.3f}", v["support"]]
                          for c, v in t["per_class"].items()]))
        for f in ("confusion_matrix.png", "history.png", "roc_pr.png"):
            im = _img(str(d / f), 175 * mm, 80 * mm)
            if im:
                el += [Spacer(1, 4), im]
    else:
        el.append(Paragraph("Held-out test metrics (thresholds chosen on calibration split)", H2))
        rows = [["Engine", "Image AUROC", "Precision", "Recall", "FNR", "FPR", "Pixel AUROC", "Dice"]]
        for k, e in m["engines"].items():
            rows.append([k, f"{e['test']['image_auroc']:.4f}", f"{e['test']['precision']:.3f}", f"{e['test']['recall']:.3f}",
                         f"{e['test']['false_negative_rate']:.3f}", f"{e['test']['false_positive_rate']:.3f}",
                         _fmt(e["test_pixel"]["pixel_auroc"], 4), _fmt(e["test_pixel"].get("mean_dice_defective"), 3)])
        el.append(_table(rows))
        f = m["fusion"]
        el.append(_kv([("Fusion rule", f["rule"]), ("Review rate", f"{f['review_rate']:.1%}"),
                       ("Auto-decision accuracy", _fmt(f["auto_accuracy"], 4)),
                       ("Missed defects (auto PASS)", f["missed_defects_auto_pass"]),
                       ("False rejects (auto FAIL)", f["false_rejects_auto_fail"]),
                       ("Unfamiliar-pattern detection", f"{_fmt(m['unfamiliar_pattern']['foreign_detection_rate'])} on "
                                                         f"{m['unfamiliar_pattern']['foreign_images']} foreign images; "
                                                         f"{_fmt(m['unfamiliar_pattern']['test_in_distribution_flag_rate'])} false flag rate")]))
    doc.build(el, onFirstPage=_header_footer("Model evaluation"), onLaterPages=_header_footer("Model evaluation"))
    return _save(db, "model_evaluation", f"Model evaluation {mv.run_id}", buf.getvalue(), "pdf", user,
                 {"run_id": mv.run_id})


def model_comparison_pdf(db: Session, versions: list[ModelVersion], user: User | None) -> Report:
    buf = io.BytesIO()
    doc = _doc(buf, "Model comparison")
    el = [Paragraph("Model Comparison Report", H1),
          Paragraph("All classifiers evaluated on the identical held-out test split. Selection should weigh "
                    "worst-class recall (missed defects), calibration, latency and size — not accuracy alone.", BODY)]
    rows = [["Model", "Acc", "Macro F1", "Min class recall", "ECE", "Latency ms", "Params M", "Train min"]]
    for mv in versions:
        mm_ = {e.name: e.value for e in mv.metrics}
        if mv.model.task != "classification":
            continue
        rows.append([mv.model.architecture, _fmt(mm_.get("accuracy"), 4), _fmt(mm_.get("f1_macro"), 4),
                     _fmt(mm_.get("min_class_recall"), 3), _fmt(mm_.get("ece"), 4), _fmt(mm_.get("latency_ms_median"), 1),
                     _fmt(mm_.get("param_size_mb", 0) / 4, 2), _fmt(mm_.get("train_time_s", 0) / 60, 1)])
    el.append(_table(rows))
    anom = [mv for mv in versions if mv.model.task == "anomaly_detection"]
    if anom:
        el.append(Paragraph("Anomaly-detection engines", H2))
        rows = [["Run", "PatchCore AUROC", "PaDiM AUROC", "PatchCore px-AUROC", "PaDiM px-AUROC", "Fusion acc", "Review rate"]]
        for mv in anom:
            mm_ = {e.name: e.value for e in mv.metrics}
            rows.append([mv.run_id, _fmt(mm_.get("patchcore_image_auroc"), 4), _fmt(mm_.get("padim_image_auroc"), 4),
                         _fmt(mm_.get("patchcore_pixel_auroc"), 4), _fmt(mm_.get("padim_pixel_auroc"), 4),
                         _fmt(mm_.get("fusion_auto_accuracy"), 3), _fmt(mm_.get("fusion_review_rate"), 3)])
        el.append(_table(rows, font=7.5))
    doc.build(el, onFirstPage=_header_footer("Model comparison"), onLaterPages=_header_footer("Model comparison"))
    return _save(db, "model_comparison", "Model comparison", buf.getvalue(), "pdf", user,
                 {"run_ids": [v.run_id for v in versions]})


def dataset_pdf(db: Session, name: str, user: User | None) -> Report:
    rep = json.loads((METRICS_DIR / f"dataset_report_{name}.json").read_text())
    buf = io.BytesIO()
    doc = _doc(buf, f"Dataset {name}")
    el = [Paragraph("Dataset Quality Report", H1), Paragraph(f"Dataset <b>{name}</b> · manifest {rep.get('manifest')}", BODY)]
    el.append(_kv([("Samples (files found)", rep["num_samples"]), ("Readable", rep["num_readable"]),
                   ("Corrupted / unreadable", rep["num_corrupted"]), ("Classes", rep["num_classes"]),
                   ("Imbalance ratio (max/min)", rep["imbalance_ratio"]), ("Image sizes", json.dumps(rep["image_sizes"])),
                   ("Colour modes", json.dumps(rep["color_modes"])), ("Exact duplicate groups", rep["exact_duplicate_groups"]),
                   ("Confirmed near-duplicate pairs", f"{rep['near_duplicate_pairs']} (from {rep.get('near_duplicate_hash_candidates', '—')} hash candidates)"),
                   ("Quality-flagged images (blank/dark/over-exposed)", rep["num_quality_flagged"]),
                   ("Masks", rep["num_masks"]), ("Mask size mismatches", len(rep["mask_size_mismatch"]))]))
    el.append(Paragraph("Class distribution", H2))
    el.append(_table([["Class", "Images"]] + [[k, v] for k, v in rep["class_distribution"].items()]))
    q = rep["quality"]
    el.append(Paragraph("Image statistics", H2))
    el.append(_table([["Metric", "Mean", "Std", "Min", "Max"]] +
                     [[k, *(f"{q[k][s]:.2f}" for s in ("mean", "std", "min", "max"))] for k in q if q[k]]))
    doc.build(el, onFirstPage=_header_footer("Dataset report"), onLaterPages=_header_footer("Dataset report"))
    return _save(db, "dataset", f"Dataset report {name}", buf.getvalue(), "pdf", user, {"dataset": name})


def drift_pdf(db: Session, chk: DriftCheck, user: User | None) -> Report:
    buf = io.BytesIO()
    doc = _doc(buf, "Drift report")
    d = chk.details
    el = [Paragraph("Drift Monitoring Report", H1),
          Paragraph(f"Profile <b>{chk.profile_id}</b> · window {chk.window} most recent inspections · {chk.created_at:%Y-%m-%d %H:%M}", BODY),
          _kv([("Data drift", chk.data_drift), ("Prediction drift", chk.prediction_drift),
               ("Verified performance degradation", "insufficient verified feedback" if chk.performance_degradation is None
                else chk.performance_degradation),
               ("Interpretation", d.get("interpretation", ""))])]
    rows = [["Feature", "KS statistic", "p-value", "PSI", "Drift"]]
    for k, v in d.get("input_features", {}).items():
        rows.append([k, _fmt(v.get("ks")), _fmt(v.get("p_value"), 4), _fmt(v.get("psi")), v.get("drift")])
    el += [Paragraph("Input feature drift (vs. reference training/calibration images)", H2), _table(rows)]
    el.append(Paragraph("Details", H2))
    el.append(Paragraph(json.dumps({k: v for k, v in d.items() if k != "input_features"}, indent=1)[:3000]
                        .replace("\n", "<br/>").replace(" ", "&nbsp;"), SMALL))
    doc.build(el, onFirstPage=_header_footer("Drift report"), onLaterPages=_header_footer("Drift report"))
    return _save(db, "drift", f"Drift report {chk.profile_id}", buf.getvalue(), "pdf", user, {"drift_check_id": chk.id})


def review_pdf(db: Session, user: User | None, days: int = 30) -> Report:
    since = utcnow() - timedelta(days=days)
    fb = db.query(HumanFeedback).filter(HumanFeedback.created_at >= since).order_by(HumanFeedback.created_at).all()
    buf = io.BytesIO()
    doc = _doc(buf, "Human review report")
    conf = sum(f.verdict == "confirm" for f in fb)
    el = [Paragraph("Human Review Report", H1), Paragraph(f"Last {days} days · {len(fb)} reviews", BODY),
          _kv([("Model prediction confirmed", conf), ("Model prediction rejected", len(fb) - conf),
               ("Agreement rate (verified)", f"{conf / len(fb):.1%}" if fb else "—"),
               ("Flagged difficult cases", sum(f.difficult for f in fb)),
               ("Pending reviews now", db.query(Inspection).filter_by(review_status="pending").count())])]
    rows = [["When", "Inspection", "Reviewer", "Verdict", "Corrected class", "Final", "Comment"]]
    for f in fb[-200:]:
        rows.append([f"{f.created_at:%m-%d %H:%M}", f.inspection_id[:8], f.reviewer.email, f.verdict,
                     f.corrected_class or "—", f.final_decision, Paragraph((f.comment or "")[:120], SMALL)])
    el += [Paragraph("Reviews", H2), _table(rows, [20 * mm, 18 * mm, 36 * mm, 15 * mm, 26 * mm, 14 * mm, 53 * mm], font=7)]
    doc.build(el, onFirstPage=_header_footer("Review report"), onLaterPages=_header_footer("Review report"))
    return _save(db, "review", f"Human review report ({days}d)", buf.getvalue(), "pdf", user, {"days": days})


# ------------------------------------------------------------------ tabular exports
EXPORT_COLS = ["id", "created_at", "profile_id", "profile_version", "source", "filename", "batch_id", "line_id",
               "product_serial", "status", "defect_present", "defect_class", "severity", "quality_index", "confidence",
               "anomaly_score_ratio", "reliability", "unfamiliar", "manual_review_required", "review_status",
               "human_decision", "latency_ms", "ground_truth", "is_simulated", "video_timestamp_s"]


def export_rows(rows: list[Inspection], fmt: str) -> bytes:
    if fmt == "json":
        return json.dumps([{c: (getattr(r, c).isoformat() if c == "created_at" else getattr(r, c)) for c in EXPORT_COLS}
                           | {"models_used": r.models_used} for r in rows], indent=1, default=str).encode()
    s = io.StringIO()
    w = csv.writer(s)
    w.writerow(EXPORT_COLS + ["models_used"])
    for r in rows:
        w.writerow([getattr(r, c) for c in EXPORT_COLS] + [";".join(r.models_used or [])])
    return s.getvalue().encode("utf-8-sig")
