"""Lightweight Gradio inspection UI (the interface described in the project proposal).

Mounted by app.main at /gradio (login with any QualityGuard account that has the ``inspect`` permission),
or run standalone with ``python scripts/run_gradio.py``.

It calls the same service layer as the REST API, so every inspection is persisted, audited and
appears in the React dashboard.
"""
from __future__ import annotations

import os

# Inspection images are treated as confidential: no third-party usage analytics from Gradio / Hugging Face.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import json

import gradio as gr

from app.database.session import SessionLocal
from app.models.entities import Inspection, ModelVersion, ProductProfile, User
from app.security.auth import verify_password
from app.services import reports
from app.services.audit import audit
from app.services.inspection_service import run_inspection
from app.services.storage import storage
from ml.config import CHECKPOINT_DIR
from ml.inference.pipeline import ModelUnavailable

STATUS_COLORS = {"PASS": "#16a34a", "FAIL": "#dc2626", "REVIEW REQUIRED": "#d97706", "INCONCLUSIVE": "#6b7280"}
VERDICT_COLORS = {"NORMAL": "#16a34a", "DEFECTIVE": "#dc2626", "MANUAL INSPECTION REQUIRED": "#d97706"}
AUTO_CHOICE = ("Auto-detect category (recommended)", "auto")
CAPTIONS = {"original": "Input image", "gradcam_overlay": "Grad-CAM (classifier attention)",
            "gradcam_pp_overlay": "Grad-CAM++", "scorecam_overlay": "Score-CAM",
            "anomaly_patchcore_overlay": "PatchCore anomaly map", "anomaly_padim_overlay": "PaDiM anomaly map",
            "segmentation_overlay": "Estimated defect region", "bounding_boxes": "Defect regions",
            "defect_zoom": "Largest region (zoom)", "comparison": "Original vs. result"}


def authenticate(email: str, password: str) -> bool:
    db = SessionLocal()
    try:
        u = db.query(User).filter(User.email == (email or "").strip().lower()).first()
        return bool(u and u.is_active and verify_password(password or "", u.password_hash)
                    and "inspect" in u.permission_codes)
    finally:
        db.close()


def _user(db, request: gr.Request | None) -> User | None:
    name = getattr(request, "username", None) if request else None
    return db.query(User).filter(User.email == name).first() if name else None


def _profiles() -> list[tuple[str, str]]:
    db = SessionLocal()
    try:
        return [AUTO_CHOICE] + [(f"{p.config.get('name', p.id)} [{p.id}]", p.id)
                                for p in db.query(ProductProfile).filter_by(is_active=True).order_by(ProductProfile.id)]
    finally:
        db.close()


def _summary_md(i: Inspection) -> str:
    from app.services.inspection_service import verdict_of
    from ml.quality.rules import VERDICT_TEXT
    r = i.result or {}
    dec = r.get("decision", {})
    v = verdict_of(i)
    rec = r.get("recognition") or {}
    blocked = bool(rec.get("block"))
    lines = [f"<div style='font-size:1.7rem;font-weight:800;color:{VERDICT_COLORS[v]}'>{v}</div>",
             f"{VERDICT_TEXT[v]}", ""]
    if blocked:
        lines += ["### Why this image was not assessed", ""]
        if rec.get("title"):
            lines.append(f"- Closest supported category: **{rec['title']}** (recogniser probability "
                         f"{rec.get('probability', 0):.0%}).")
        if rec.get("distance") is not None:
            ratio = rec["distance"] / max(rec.get("distance_threshold") or 1e-9, 1e-9)
            lines.append(f"- Appearance distance to that category's training images: **{rec['distance']:.3f}**, "
                         f"about **{ratio:.0f}× the accepted limit** ({rec['distance_threshold']:.3f}). The models "
                         "were trained on photos taken a specific way (camera angle, lighting, background); this photo "
                         "looks different, so a defect verdict would not be reliable.")
        lines += ["", "**What you can do**", "",
                  "- Use a photo taken like the training images (see *Supported photos* below), or",
                  "- train QualityGuard on your own kind of photos: main app → **Products → Add a product**, "
                  "upload 10+ (ideally 30–100) photos of *good* items taken with your own camera set-up.", ""]
    elif dec.get("message"):
        lines.append(f"_{dec['message']}_" + chr(10))
    ref = " (reference only — not a result)" if blocked else ""
    rows = [("Category", f"{rec.get('title') or i.profile_id}" + (" — auto-detected" if rec.get("mode") == "auto" else "")),
            ("Defect class", i.defect_class or ("anomaly (unclassified)" if i.defect_present else "—")),
            ("Confidence", f"{i.confidence:.1%}{ref}" if i.confidence is not None else "—"),
            ("Anomaly score / threshold", f"{i.anomaly_score_ratio:.2f}×{ref}" if i.anomaly_score_ratio is not None else "—"),
            ("Severity", i.severity), ("Line disposition (profile rules)", i.status),
            ("Reliability", i.reliability or "—"), ("Manual review", "required" if i.manual_review_required else "no"),
            ("Latency", f"{i.latency_ms:.0f} ms"), ("Models", ", ".join(i.models_used) or "—"),
            ("Inspection ID", f"`{i.id}`")]
    lines += ["| | |", "|---|---|"] + [f"| **{k}** | {v_} |" for k, v_ in rows]
    clf = r.get("engines", {}).get("classification")
    if clf and clf.get("probabilities") and not blocked:
        top = sorted(clf["probabilities"].items(), key=lambda x: -x[1])[:5]
        lines += ["", "**Class probabilities**", ""] + [f"- {k}: {v_:.1%}" for k, v_ in top]
    notes = [n for n in (r.get("reliability", {}).get("notes") or []) if not n.startswith("UNSUPPORTED")]
    if notes:
        lines += ["", "**Reliability notes**", ""] + [f"- {n}" for n in notes]
    issues = r.get("input_issues") or []
    if issues:
        lines += ["", "**Input issues**", ""] + [f"- {x}" for x in issues]
    if blocked:
        lines += ["", "**Supported photos** (examples of what each model was trained on): "
                  "`data/samples/` and the held-out test images under `data/raw/` — e.g. MVTec screws are grayscale, "
                  "top-down, on a uniform background; KolektorSDD2 commutators are close-up surface strips."]
    return chr(10).join(lines)


def _gallery(i: Inspection) -> list[tuple[str, str]]:
    keys = (i.result or {}).get("visualizations", [])
    out = []
    for k in keys:
        if k.endswith("_heatmap") or k == "segmentation_mask":
            continue
        p = storage.resolve(f"{i.artifact_dir}/{k}.png")
        if p.exists():
            out.append((str(p), CAPTIONS.get(k, k.replace("_", " "))))
    return out


def inspect_image(image_path: str | None, profile_id: str, serial: str, explain: bool,
                  request: gr.Request = None):
    if not image_path:
        raise gr.Error("Upload an image first.")
    if not profile_id:
        raise gr.Error("Choose a product profile.")
    db = SessionLocal()
    try:
        user = _user(db, request)
        with open(image_path, "rb") as f:
            data = f.read()
        try:
            i = run_inspection(db, data=data, filename=image_path.replace("\\", "/").rsplit("/", 1)[-1],
                               profile_id=profile_id, user=user, source="single", product_serial=serial or "",
                               options={} if explain else {"skip_explain": True})
        except ModelUnavailable as exc:
            raise gr.Error(f"Model unavailable: {exc}")
        audit(db, user, "inspection.run", "inspection", i.id, {"profile": profile_id, "via": "gradio"})
        return _summary_md(i), _gallery(i), i.id, gr.update(value=None, visible=False)
    finally:
        db.close()


def make_report(iid: str, request: gr.Request = None):
    if not iid:
        raise gr.Error("Run an inspection first.")
    db = SessionLocal()
    try:
        i = db.get(Inspection, iid)
        if not i:
            raise gr.Error("Inspection not found.")
        user = _user(db, request)
        r = reports.inspection_pdf(db, i, user)
        audit(db, user, "report.generate", "report", r.id, {"kind": "inspection", "via": "gradio"})
        return gr.update(value=str(storage.resolve(r.path)), visible=True)
    finally:
        db.close()


def history(profile_id: str | None):
    db = SessionLocal()
    try:
        q = db.query(Inspection)
        if profile_id:
            q = q.filter(Inspection.profile_id == profile_id)
        rows = q.order_by(Inspection.created_at.desc()).limit(100).all()
        from app.services.inspection_service import verdict_of
        return [[i.created_at.strftime("%Y-%m-%d %H:%M:%S"), i.profile_id, i.filename, verdict_of(i),
                 i.defect_class or ("anomaly" if i.defect_present else ""), i.severity,
                 None if i.confidence is None else round(i.confidence, 3),
                 None if i.quality_index is None else round(i.quality_index), i.id[:8]] for i in rows]
    finally:
        db.close()


def model_choices() -> list[tuple[str, int]]:
    db = SessionLocal()
    try:
        return [(f"{m.run_id} ({m.status})", m.id) for m in db.query(ModelVersion).order_by(ModelVersion.created_at)]
    finally:
        db.close()


def model_details(mid: int | None):
    if not mid:
        return "", []
    db = SessionLocal()
    try:
        mv = db.get(ModelVersion, mid)
        d = CHECKPOINT_DIR / mv.run_id
        lines = [f"### {mv.run_id}", f"- **Task**: {mv.model.task}", f"- **Architecture**: {mv.model.architecture}",
                 f"- **Status**: {mv.status}", f"- **Dataset**: {mv.dataset_name} {mv.dataset_version}",
                 f"- **Classes**: {', '.join(mv.class_labels) or '—'}", "", "| Metric | Value |", "|---|---|"]
        for m in sorted(mv.metrics, key=lambda x: x.name):
            lines.append(f"| {m.name} | {m.value:.4f} |")
        imgs = [(str(d / f), f.rsplit(".", 1)[0].replace("_", " ")) for f in
                ("confusion_matrix.png", "history.png", "roc_pr.png") if (d / f).exists()]
        return "\n".join(lines), imgs
    finally:
        db.close()


def _refresh_choices():
    ps = _profiles()
    return (gr.update(choices=ps, value=ps[0][1] if ps else None),
            gr.update(choices=[("All profiles", "")] + [p for p in ps if p[1] != "auto"], value=""), gr.update(choices=model_choices()))


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="QualityGuard AI — Inspection") as ui:
        gr.Markdown("# QualityGuard AI\nDeep-learning visual inspection: upload a product image, get a "
                    "decision with confidence, severity, visual explanation and a PDF inspection report.  \n"
                    "_AI-assisted tool: results marked MANUAL INSPECTION REQUIRED need an inspector._")
        with gr.Tab("Inspect"):
            with gr.Row():
                with gr.Column(scale=1):
                    prof = gr.Dropdown([], label="Product profile")
                    img = gr.Image(type="filepath", label="Product image", sources=["upload", "webcam", "clipboard"],
                                   height=320)
                    serial = gr.Textbox(label="Product serial (optional)", max_length=128)
                    explain = gr.Checkbox(value=True, label="Generate explanations (Grad-CAM / anomaly maps)")
                    run = gr.Button("Inspect", variant="primary")
                with gr.Column(scale=1):
                    out_md = gr.Markdown("Results appear here.")
                    rep_btn = gr.Button("Generate PDF inspection report")
                    rep_file = gr.File(label="Inspection report", visible=False)
            gallery = gr.Gallery(label="Visual explanation", columns=4, height="auto", object_fit="contain")
            iid = gr.State("")
            run.click(inspect_image, [img, prof, serial, explain], [out_md, gallery, iid, rep_file])
            rep_btn.click(make_report, [iid], [rep_file])
        with gr.Tab("History"):
            hp = gr.Dropdown([("All profiles", "")], value="", label="Profile")
            table = gr.Dataframe(headers=["Time (UTC)", "Profile", "File", "Result", "Defect", "Severity",
                                          "Confidence", "Quality", "ID"], interactive=False, wrap=True)
            gr.Button("Refresh").click(history, [hp], [table])
            hp.change(history, [hp], [table])
            ui.load(history, [hp], [table])
        # choices are loaded per page view (the DB is initialised after the UI is built)
        with gr.Tab("Model performance"):
            mm = gr.Dropdown([], label="Model version")
            md = gr.Markdown()
            mg = gr.Gallery(label="Evaluation plots", columns=3, height="auto", object_fit="contain")
            mm.change(model_details, [mm], [md, mg])
        ui.load(_refresh_choices, None, [prof, hp, mm])
        gr.Markdown("Full dashboard, review workflow, analytics and model management: open the main web app.")
    return ui


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(_profiles()))
