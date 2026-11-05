"""QualityGuard AI assistant.

Tool-based architecture: the assistant can only call the read-only, permission-checked functions in TOOLS.
All numbers come from tool results (SQL queries / experiment files). The deep-learning models remain the only
source of visual predictions - the assistant never classifies images.

Two backends share the same tools:
  * LLM mode (``QG_ANTHROPIC_API_KEY`` or ANTHROPIC_API_KEY set and ``anthropic`` installed): Claude interprets the
    retrieved statistics through a manual tool-use loop.
  * Offline mode (default): an intent engine extracts entities (period, product, status, defect class, inspection
    reference) from the question and the conversation, scores intents, calls the tools and formats grounded
    answers. When it does not understand, it says so and suggests questions it can answer - it never pads with an
    unrelated summary.

Answers are Markdown. Links to application pages use relative paths (``[text](/inspections/<id>)``); the chat UI
renders them as in-app links.
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import timedelta

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.entities import (Alert, Inspection, KnowledgeEntry, ModelVersion, ProductionBatch, ProductProfile,
                                 User, utcnow)
from app.services import analytics, similarity

log = logging.getLogger("qualityguard.assistant")

STATUSES = ("PASS", "FAIL", "REVIEW REQUIRED", "INCONCLUSIVE")
PLAIN = {"PASS": "not defective (PASS)", "FAIL": "defective (FAIL)",
         "REVIEW REQUIRED": "needs human review", "INCONCLUSIVE": "inconclusive"}

CONCEPTS = {
    "confidence_vs_anomaly": (
        "**Classification confidence** is the temperature-calibrated probability that the image belongs to one of the "
        "*known* defect classes the classifier was trained on. It says nothing about inputs unlike the training data.\n\n"
        "The **anomaly score** (PatchCore / PaDiM) measures how far image patches are from defect-free reference images, "
        "compared with a threshold validated on held-out data. A high anomaly score means *unlike normal parts*, not a "
        "specific defect type.\n\nQualityGuard also runs a separate **unfamiliar-input check** (feature-space distance), "
        "because a high softmax probability is not certainty."),
    "gradcam": ("**Grad-CAM** highlights the image regions that most influenced the classifier's chosen class, computed "
                "from the gradients of the real model on that image. It explains the model's attention - it is *not* a "
                "measured defect boundary. For the estimated defect region, look at the anomaly map / segmentation."),
    "severity": ("**Severity** (none / low / medium / high / critical) is assessed by configurable product-profile rules: "
                 "defect-area bands, number of regions, anomaly intensity and per-class minimum levels. These are the "
                 "manufacturer's configured rules, not universal industrial standards. See [Product profiles](/profiles)."),
    "quality_index": ("The **quality index** (0-100) is a transparent rule-derived score: 100 minus the deduction "
                      "configured for the assessed severity, capped for review cases. It is not a scientifically "
                      "validated quality metric."),
    "decisions": ("Every inspection ends in one decision:\n"
                  "- **PASS** - not defective; the item can go on.\n"
                  "- **FAIL** - defective; reject the item.\n"
                  "- **REVIEW REQUIRED** - the models are unsure or disagree, or the score is near the threshold; "
                  "a person decides in the [review queue](/reviews).\n"
                  "- **INCONCLUSIVE** - the image is unusable (blurred, dark, too small) or looks unlike anything the "
                  "models know (for example a different product). Nothing is decided automatically."),
    "unfamiliar": ("An input is **unfamiliar** when it looks unlike the training data - typically a different product, "
                   "a very different camera setup or lighting. The models then refuse to guess and the result is "
                   "INCONCLUSIVE / manual review. To inspect a new kind of item, add it as a product: "
                   "[Add a product](/products/new) trains a detector from photos of good items."),
    "anomaly_detection": ("**Anomaly detection** learns only from photos of *good* items. At inspection time it flags "
                          "anything that deviates from them - including defect types nobody labelled - and outlines "
                          "where. This is how new products can be added with nothing but good examples."),
    "active_learning": ("The [review queue](/reviews) is ordered by **learning value**: uncertain, borderline, "
                        "disputed, unfamiliar or rare-class items first. Your reviews (and any masks you draw) can be "
                        "exported as a curated dataset for retraining under [Datasets](/datasets)."),
}

HOWTO = {
    "inspect": ("**To check whether an item is defective:**\n"
                "1. Open [Inspect](/inspect).\n2. Choose the product type at the top right.\n"
                "3. Drop a photo of the item (or click to browse) and press **Run inspection**.\n"
                "4. The verdict appears on the right: **Not defective / Defective / Needs review**, with the defect "
                "outlined and a downloadable PDF report.\n\n"
                "The product type must be one the system has a model for (see [products](/products/new)); "
                "a photo of a different kind of item is reported as *unfamiliar*."),
    "add_product": ("**To inspect a new kind of item:**\n"
                    "1. Open [Add a product](/products/new).\n2. Name it and upload at least 20 photos of *good* items "
                    "(same camera position and lighting you will use later); optionally a few defective ones.\n"
                    "3. Training runs in the background (a few minutes on CPU).\n"
                    "4. Approve and deploy it, then inspect it from [Inspect](/inspect)."),
    "batch": ("**Batch / folder inspection:** [Inspect](/inspect) → *Batch / folder* tab → select many images or a whole "
              "folder → *Start*. You get per-image results, a summary, CSV export and a batch PDF."),
    "video": ("**Video inspection:** [Inspect](/inspect) → *Video* tab → upload a video and choose how often to sample "
              "frames. You get timestamped results and an annotated video."),
    "camera": ("**Live camera:** [Inspect](/inspect) → *Camera* tab → pick a camera → *Start camera* → *Capture & "
               "inspect* (or *Continuous*). Session counters and alerts update live."),
    "review": ("**Reviewing:** open the [review queue](/reviews), click an item, confirm or correct the AI result, set the "
               "final decision and optionally upload a defect mask. Reviews resolve the related alerts."),
    "report": ("**Reports:** every inspection has a *PDF report* button on its page. Batch, daily/weekly/monthly, model, "
               "dataset, drift and review reports are generated in [Reports](/reports)."),
    "models": ("**Models:** [Models & training](/models) shows every model version with its test metrics, lets you start "
               "training, approve validated versions and deploy them to a product (with rollback)."),
    "alerts": ("**Alerts:** [Alerts](/alerts) lists critical defects, repeated defects, unfamiliar inputs, drift and "
               "failures; acknowledge or resolve them there."),
    "simulation": ("**Line simulation:** [Line simulation](/simulation) feeds held-out images through the models at a set "
                   "rate to demonstrate a production line (records are marked as simulated)."),
    "api": ("**API access:** create a key in [Administration](/admin) → *API keys*, then POST images to "
            "`/api/inspections` with header `X-API-Key`. Full reference: [/api/docs](/api/docs)."),
}

TOOLS = [
    {"name": "get_dashboard_summary", "description": "Inspection totals, status counts, acceptance/defect rates, "
     "latency, defect distribution and grounded insights over the last N days.",
     "input_schema": {"type": "object", "properties": {"days": {"type": "integer"}, "profile_id": {"type": "string"}},
                      "required": ["days"], "additionalProperties": False}},
    {"name": "search_inspections", "description": "Count and list inspection records with filters. status is one of "
     "PASS, FAIL, REVIEW REQUIRED, INCONCLUSIVE. review_pending=true gives the manual review queue. days=0 means all time.",
     "input_schema": {"type": "object", "properties": {
         "status": {"type": "string"}, "profile_id": {"type": "string"}, "defect_class": {"type": "string"},
         "review_pending": {"type": "boolean"}, "days": {"type": "integer"}, "limit": {"type": "integer"}},
         "required": [], "additionalProperties": False}},
    {"name": "get_inspection", "description": "Full stored result for one inspection: id (or 8-char prefix), or "
     "'latest' for the most recent one (optionally filtered by profile_id / status).",
     "input_schema": {"type": "object", "properties": {"inspection_id": {"type": "string"},
                                                       "profile_id": {"type": "string"}, "status": {"type": "string"}},
                      "required": ["inspection_id"], "additionalProperties": False}},
    {"name": "defect_statistics", "description": "Defect counts by class and severity, and per-product rejection rates.",
     "input_schema": {"type": "object", "properties": {"days": {"type": "integer"}, "profile_id": {"type": "string"}},
                      "required": ["days"], "additionalProperties": False}},
    {"name": "quality_trend", "description": "Compare the last N days with the N days before: volume, defect rate, "
     "acceptance rate and which defect classes increased or decreased.",
     "input_schema": {"type": "object", "properties": {"days": {"type": "integer"}, "profile_id": {"type": "string"}},
                      "required": ["days"], "additionalProperties": False}},
    {"name": "list_alerts", "description": "Alerts (critical defects, repeated defects, unfamiliar inputs, drift, "
     "failures). status: active (default), open, acknowledged, resolved, all.",
     "input_schema": {"type": "object", "properties": {"status": {"type": "string"}, "limit": {"type": "integer"}},
                      "required": [], "additionalProperties": False}},
    {"name": "compare_batches", "description": "Compare two inspection batches (defaults to the two most recent).",
     "input_schema": {"type": "object", "properties": {"batch_a": {"type": "string"}, "batch_b": {"type": "string"}},
                      "required": [], "additionalProperties": False}},
    {"name": "model_comparison", "description": "Held-out evaluation metrics for all registered model versions, and "
     "which models each product currently uses.",
     "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}},
    {"name": "list_products", "description": "Product profiles the system can inspect, their defect classes and models.",
     "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}},
    {"name": "knowledge_base", "description": "Verified knowledge-base entry for a defect class.",
     "input_schema": {"type": "object", "properties": {"defect_class": {"type": "string"}},
                      "required": ["defect_class"], "additionalProperties": False}},
    {"name": "similar_defects", "description": "Visually similar past inspections (embedding cosine similarity).",
     "input_schema": {"type": "object", "properties": {"inspection_id": {"type": "string"}},
                      "required": ["inspection_id"], "additionalProperties": False}},
    {"name": "explain_concept", "description": "Platform definitions.",
     "input_schema": {"type": "object", "properties": {"topic": {"type": "string", "enum": list(CONCEPTS)}},
                      "required": ["topic"], "additionalProperties": False}},
    {"name": "app_help", "description": "Step-by-step instructions for using the application, with page links.",
     "input_schema": {"type": "object", "properties": {"topic": {"type": "string", "enum": list(HOWTO)}},
                      "required": ["topic"], "additionalProperties": False}},
]
for _t in TOOLS:
    _t["strict"] = True

SYSTEM = (
    "You are the QualityGuard AI assistant inside an industrial visual-inspection platform. Answer questions about "
    "inspections, quality trends, models, defects and how to use the application, using ONLY the results of the "
    "provided tools. Every number you state must come from a tool result in this conversation; if the tools do not "
    "provide it, say the data is not available. Never invent industrial standards, tolerances, measurements, model "
    "metrics or inspection records. You do not classify images; the platform's deep-learning models do - if a user "
    "wants an item checked, explain how to use the Inspect page (app_help topic 'inspect'). Do not diagnose "
    "manufacturing root causes from images; you may relay verified knowledge-base guidance as suggested follow-up "
    "investigation. Mark simulated data as simulated. Link to records and pages with Markdown links using relative "
    "paths, e.g. [a1b2c3d4](/inspections/<full id>), [review queue](/reviews), [Inspect](/inspect). Be concise and use "
    "short bullet lists for figures.")


# ------------------------------------------------------------------ tools
def _resolve(db: Session, iid: str, profile_id: str | None = None, status: str | None = None) -> Inspection | None:
    if iid in ("latest", "last", "recent"):
        q = db.query(Inspection)
        if profile_id:
            q = q.filter(Inspection.profile_id == profile_id)
        if status:
            q = q.filter(Inspection.status == status.upper())
        return q.order_by(Inspection.created_at.desc()).first()
    i = db.get(Inspection, iid)
    if i is None and len(iid) >= 6:
        i = db.query(Inspection).filter(Inspection.id.like(f"{iid}%")).first()
    return i


def _since(days):
    return utcnow() - timedelta(days=days) if days else None


def run_tool(db: Session, user: User, name: str, args: dict):
    if "view" not in user.permission_codes:
        return {"error": "permission denied"}
    if name == "get_dashboard_summary":
        d = analytics.dashboard(db, days=int(args.get("days") or 30), profile_id=args.get("profile_id"))
        d.pop("recent", None)
        d.pop("trend", None)
        return d
    if name == "search_inspections":
        q = db.query(Inspection)
        if args.get("status"):
            q = q.filter(Inspection.status == args["status"].upper())
        if args.get("profile_id"):
            q = q.filter(Inspection.profile_id == args["profile_id"])
        if args.get("defect_class"):
            q = q.filter(Inspection.defect_class == args["defect_class"])
        if args.get("review_pending"):
            q = q.filter(Inspection.review_status == "pending")
        if args.get("days"):
            q = q.filter(Inspection.created_at >= _since(int(args["days"])))
        total = q.count()
        rows = q.order_by(Inspection.created_at.desc()).limit(min(int(args.get("limit") or 10), 50)).all()
        return {"count": total, "items": [analytics._row(r) for r in rows]}
    if name == "get_inspection":
        i = _resolve(db, str(args["inspection_id"]).lower(), args.get("profile_id"), args.get("status"))
        if not i:
            return {"error": "inspection not found"}
        r = i.result or {}
        return {**analytics._row(i), "decision": r.get("decision"), "severity_assessment": r.get("severity"),
                "reliability": r.get("reliability"), "classification": _slim(r.get("engines", {}).get("classification")),
                "anomaly": r.get("engines", {}).get("anomaly"), "novelty": r.get("engines", {}).get("novelty"),
                "measurements": {k: v for k, v in (r.get("measurements") or {}).items() if k != "regions"},
                "unfamiliar": i.unfamiliar, "review_status": i.review_status, "human_decision": i.human_decision,
                "ground_truth": i.ground_truth}
    if name == "defect_statistics":
        return analytics.defects(db, days=int(args.get("days") or 30), profile_id=args.get("profile_id"))
    if name == "quality_trend":
        days = int(args.get("days") or 7)
        now = utcnow()
        out = {}
        for key, start, end in (("current", now - timedelta(days=days), now),
                                ("previous", now - timedelta(days=2 * days), now - timedelta(days=days))):
            q = analytics.filtered(db, start=start, end=end, profile_id=args.get("profile_id"))
            out[key] = analytics._aggregate_rows(q.all())
        cur, prev = out["current"]["defects"], out["previous"]["defects"]
        changes = sorted(({"defect": k, "current": cur.get(k, 0), "previous": prev.get(k, 0),
                           "change": cur.get(k, 0) - prev.get(k, 0)} for k in set(cur) | set(prev)),
                         key=lambda x: -abs(x["change"]))
        return {"days": days, **out, "defect_changes": changes}
    if name == "list_alerts":
        st = (args.get("status") or "active").lower()
        q = db.query(Alert)
        if st == "active":
            q = q.filter(Alert.status != "resolved")
        elif st != "all":
            q = q.filter(Alert.status == st)
        rows = q.order_by(Alert.created_at.desc()).limit(min(int(args.get("limit") or 10), 50)).all()
        by_level = {}
        for a in q.all():
            by_level[a.level] = by_level.get(a.level, 0) + 1
        return {"count": q.count(), "by_level": by_level,
                "items": [{"id": a.id, "created_at": a.created_at.isoformat() + "Z", "kind": a.kind, "level": a.level,
                           "title": a.title, "status": a.status, "inspection_id": a.inspection_id,
                           "profile_id": a.profile_id} for a in rows]}
    if name == "compare_batches":
        bs = db.query(ProductionBatch).order_by(ProductionBatch.created_at.desc())
        a = db.get(ProductionBatch, args["batch_a"]) if args.get("batch_a") else None
        b = db.get(ProductionBatch, args["batch_b"]) if args.get("batch_b") else None
        if not (a and b):
            recent = bs.limit(2).all()
            if len(recent) < 2:
                return {"error": "fewer than two batches exist"}
            a, b = recent[1], recent[0]
        out = []
        for x in (a, b):
            rows = db.query(Inspection).filter_by(batch_id=x.id).all()
            out.append({"batch_id": x.id, "name": x.name, "created_at": x.created_at.isoformat(), "source": x.source,
                        "simulated": x.is_simulated, **analytics._aggregate_rows(rows)})
        return {"older": out[0], "newer": out[1]}
    if name == "model_comparison":
        res = []
        for mv in db.query(ModelVersion).all():
            res.append({"run_id": mv.run_id, "task": mv.model.task, "architecture": mv.model.architecture,
                        "status": mv.status, "metrics": {m.name: round(m.value, 5) for m in mv.metrics}})
        deployed = {p.id: (p.config or {}).get("models", {}) for p in
                    db.query(ProductProfile).filter_by(is_active=True).all()}
        return {"models": res, "deployed_by_product": deployed,
                "note": "Classifiers share one held-out test split; anomaly metrics use held-out test images."}
    if name == "list_products":
        out = []
        for p in db.query(ProductProfile).filter_by(is_active=True).order_by(ProductProfile.id):
            c = p.config or {}
            out.append({"id": p.id, "name": c.get("name", p.id), "defect_classes": c.get("defect_classes", []),
                        "models": c.get("models", {}), "description": c.get("description", "")})
        return {"products": out}
    if name == "knowledge_base":
        e = db.query(KnowledgeEntry).filter(KnowledgeEntry.defect_class == args["defect_class"]).first()
        if not e:
            return {"error": "no knowledge-base entry"}
        return {"defect_class": e.defect_class, "title": e.title, "visual_description": e.visual_description,
                "characteristics": e.characteristics, "inspection_guidance": e.inspection_guidance,
                "corrective_actions": [c for c in e.corrective_actions if c.get("verified")],
                "unverified_actions_count": sum(not c.get("verified") for c in e.corrective_actions), "source": e.source}
    if name == "similar_defects":
        i = _resolve(db, str(args["inspection_id"]).lower())
        if not i:
            return {"error": "inspection not found"}
        return {"query": i.id, "results": similarity.similar(db, i, 6),
                "caveat": "Visual similarity does not imply the same root cause."}
    if name == "explain_concept":
        return {"topic": args["topic"], "text": CONCEPTS.get(args["topic"], "unknown topic")}
    if name == "app_help":
        return {"topic": args["topic"], "text": HOWTO.get(args["topic"], "unknown topic")}
    return {"error": f"unknown tool {name}"}


def _slim(c):
    if not c:
        return None
    return {k: c[k] for k in ("predicted_class", "confidence", "margin", "members_agree", "ood_votes", "reliability",
                              "reliability_notes", "probabilities") if k in c}


# ------------------------------------------------------------------ LLM backend
def _llm_key() -> str | None:
    return get_settings().anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY") or None


def llm_available() -> bool:
    if not _llm_key():
        return False
    try:
        import anthropic  # noqa: F401
        return True
    except ImportError:
        return False


def chat_llm(db: Session, user: User, history: list[dict], question: str) -> dict:
    import anthropic

    client = anthropic.Anthropic(api_key=_llm_key())
    messages = [{"role": m["role"], "content": m["content"]} for m in history[-10:]
                if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str) and m["content"]]
    while messages and messages[0]["role"] != "user":  # the conversation must start with a user turn
        messages.pop(0)
    messages.append({"role": "user", "content": question})
    calls = []
    for _ in range(8):
        try:
            # Claude Opus 5.5 always thinks; effort controls depth. Server-side fallback (default routing) re-runs a
            # refused request on a suitable model inside the same call.
            resp = client.beta.messages.create(
                model=get_settings().assistant_model, max_tokens=16000, system=SYSTEM, tools=TOOLS,
                messages=messages, output_config={"effort": "medium"},
                betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        except anthropic.RateLimitError:
            return {"answer": "The language model is rate-limited right now; please retry shortly.", "tool_calls": calls,
                    "mode": "llm"}
        except anthropic.APIStatusError as exc:
            log.warning("assistant LLM error %s; falling back to offline mode", exc.status_code)
            return {**chat_offline(db, user, question, history), "note": f"LLM error {exc.status_code}; offline answer shown."}
        except anthropic.APIConnectionError:
            return {**chat_offline(db, user, question, history), "note": "LLM unreachable; offline answer shown."}
        if resp.stop_reason == "refusal":
            return {"answer": "The request was declined by the language model's safety system.", "tool_calls": calls,
                    "mode": "llm"}
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason == "pause_turn":
            continue
        uses = [b for b in resp.content if b.type == "tool_use"]
        if not uses:
            text = "".join(b.text for b in resp.content if b.type == "text")
            return {"answer": text, "tool_calls": calls, "mode": "llm", "model": resp.model}
        results = []
        for u in uses:
            try:
                out = run_tool(db, user, u.name, dict(u.input))
                err = "error" in out if isinstance(out, dict) else False
            except Exception as exc:
                out, err = {"error": str(exc)}, True
            calls.append({"tool": u.name, "input": dict(u.input)})
            results.append({"type": "tool_result", "tool_use_id": u.id, "content": json.dumps(out, default=str)[:30000],
                            "is_error": err})
        messages.append({"role": "user", "content": results})
    return {"answer": "Stopped after too many tool calls.", "tool_calls": calls, "mode": "llm"}


# ------------------------------------------------------------------ offline backend: entity extraction
def _pct(v):
    return "—" if v is None else f"{v:.1%}"


def _link(iid: str) -> str:
    return f"[{iid[:8]}](/inspections/{iid})"


def _has(text: str, *words) -> bool:
    return any(re.search(rf"\b{w}", text) for w in words)


PROFILE_ALIASES = {"steel": "steel_strip", "strip": "steel_strip", "sheet": "steel_strip", "bottle": "bottle",
                   "bottles": "bottle", "glass": "bottle", "nut": "metal_nut", "nuts": "metal_nut"}
STATUS_WORDS = [("REVIEW REQUIRED", r"\breview(s|ed)?\b|\bneeds? review\b|\bunsure\b|\buncertain\b"),
                ("INCONCLUSIVE", r"\binconclusive\b|\bunfamiliar\b|\bunknown\b"),
                ("FAIL", r"\bfail(ed|s|ing|ure)?\b|\breject(ed|s|ion)?\b|\bdefective\b|\bbad\b|\bnot ok\b"),
                ("PASS", r"\bpass(ed|es)?\b|\baccept(ed)?\b|\bgood\b|\bok\b|\bnot defective\b|\bfine\b")]


def _period(ql: str) -> tuple[int, str]:
    m = re.search(r"(?:last|past|previous)?\s*(\d+)\s*(day|week|month|hour)s?", ql)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        days = max(1, {"day": n, "week": 7 * n, "month": 30 * n, "hour": 1}[unit])
        return days, f"last {n} {unit}{'s' if n != 1 else ''}"
    for words, days, label in ((("today",), 1, "the last 24 hours"), (("yesterday",), 2, "the last 2 days"),
                               (("this week", "last week", "past week", "weekly", "a week"), 7, "the last 7 days"),
                               (("this month", "last month", "past month", "monthly", "a month"), 30, "the last 30 days"),
                               (("this year", "last year", "a year"), 365, "the last year"),
                               (("ever", "all time", "total", "overall", "so far"), 0, "all time")):
        if any(w in ql for w in words):
            return days, label
    return 30, "the last 30 days"


class Entities:
    def __init__(self, db: Session, q: str, history: list[dict]):
        self.q = q
        self.ql = ql = " " + re.sub(r"[^\w\s/-]", " ", q.lower()).replace("-", " ") + " "
        self.days, self.period = _period(ql)
        self.explicit_period = bool(re.search(r"\d+\s*(day|week|month|hour)|today|yesterday|week|month|year|ever|all time", ql))
        profiles = db.query(ProductProfile).filter_by(is_active=True).all()
        self.profile = None
        for p in profiles:
            names = {p.id, p.id.replace("_", " "), (p.config or {}).get("name", "").lower()}
            if any(n and f" {n} " in ql for n in names):
                self.profile = p.id
        if not self.profile:
            for alias, pid in PROFILE_ALIASES.items():
                if f" {alias} " in ql and any(p.id == pid for p in profiles):
                    self.profile = pid
                    break
        classes = {c for p in profiles for c in (p.config or {}).get("defect_classes", [])}
        self.defect = next((c for c in sorted(classes, key=len, reverse=True)
                            if f" {c.replace('_', ' ').replace('-', ' ')} " in ql or f" {c} " in ql), None)
        self.status = next((s for s, rx in STATUS_WORDS if re.search(rx, ql)), None)
        m = re.search(r"\b([0-9a-f]{8,32})\b", q.lower())
        self.iid = m.group(1) if m else None
        self.refers_last = bool(re.search(r"\b(last|latest|most recent|newest|previous|recent)\b", ql))
        self.refers_pronoun = bool(re.search(
            r"\b(it|this one|that one|this inspection|that inspection|this item|that item|the item|the image|the result)\b",
            ql))
        self.context_iid = None
        for h in reversed(history or []):
            ids = re.findall(r"/inspections/([0-9a-f]{32})", str(h.get("content", "")))
            if ids:
                self.context_iid = ids[0]
                break


# ------------------------------------------------------------------ offline backend: intents
SUGGEST = {
    "default": ["Summarise quality this week", "Which defect is most common?", "Why was the last bottle rejected?",
                "How do I check if an item is defective?"],
    "inspection": ["Find similar past defects", "What does Grad-CAM show?", "Show inspections waiting for review"],
    "stats": ["Is quality getting better or worse this week?", "Which product has the highest rejection rate?",
              "Show failed inspections today"],
    "help": ["How do I add a new product?", "How do I inspect a video?", "What do PASS, FAIL and REVIEW mean?"],
}


def chat_offline(db: Session, user: User, q: str, history: list[dict] | None = None) -> dict:
    e = Entities(db, q, history or [])
    ql = e.ql
    calls: list[dict] = []

    def call(name, **a):
        calls.append({"tool": name, "input": a})
        return run_tool(db, user, name, a)

    def ans(text, suggest="default"):
        return {"answer": text, "tool_calls": calls, "mode": "offline", "suggestions": SUGGEST.get(suggest, [])}

    # --- small talk / capabilities
    if re.fullmatch(r"\s*(hi|hello|hey|hii+|good (morning|afternoon|evening)|namaste|yo)\W*\s*", q.lower()):
        return ans("Hello! I can answer questions about your inspections, defects, quality trends and models, and explain "
                   "how to use QualityGuard. What would you like to know?")
    if re.fullmatch(r"\s*(thanks|thank you|thx|ok|okay|great|cool|nice)\W*\s*", q.lower()):
        return ans("You're welcome. Anything else?")
    if _has(ql, "what can you", "help me", "what do you do", "who are you", "capabilit") or ql.strip() in ("help", "?"):
        return ans("I answer from the inspection database and model records - I never make numbers up. Try:\n"
                   "- **Results:** *How many items failed today?*, *Show defective bottles this week*\n"
                   "- **One inspection:** *Why was the last steel item rejected?*, *Explain inspection a1b2c3d4*\n"
                   "- **Trends:** *Is quality getting worse?*, *Which defect is most common?*, *Which product is rejected most?*\n"
                   "- **Work queue:** *What needs review?*, *Any critical alerts?*\n"
                   "- **Models:** *How accurate is the steel model?*, *Which models are deployed?*\n"
                   "- **How-to:** *How do I check a picture?*, *How do I add a new product?*\n\n"
                   "I don't look at images myself - the deep-learning models do. To check an item, use [Inspect](/inspect).",
                   "help")

    # --- products / what can be inspected
    if _has(ql, "which products", "what products", "products? (can|do|are)", "what (items|things|objects) can",
            "which (items|things|objects)", "supported", "what can (it|you|the system) (inspect|detect|check)",
            "list products", "profiles"):
        r = call("list_products")
        lines = []
        for p in r["products"]:
            kind = "defect classifier" if p["models"].get("classifiers") else ""
            kind = " + ".join(x for x in (kind, "anomaly detector" if p["models"].get("anomaly") else "") if x)
            cls = f" — classes: {', '.join(p['defect_classes'])}" if p["defect_classes"] else ""
            lines.append(f"- **{p['name']}** (`{p['id']}`): {kind or 'no model deployed yet'}{cls}")
        return ans("Products the system can inspect:\n" + "\n".join(lines) +
                   "\n\nTo inspect another kind of item, [add a product](/products/new) with photos of good examples.",
                   "help")

    # --- user wants an image checked by the assistant
    if _has(ql, "upload", "check (this|my|an|a) (item|image|photo|picture|pic)", "is (this|my) .* defect",
            "can you (check|see|look at) (this|my|an|a|the)", "analy[sz]e (this|my|an|a) (image|photo|picture|pic)"):
        return ans(call("app_help", topic="inspect")["text"], "help")

    # --- how-to
    if _has(ql, "how (do|can|to|should)", "where (do|can|is)", "steps? to", "guide", "tutorial", "way to"):
        for topic, kws in (("add_product", ("new product", "add a product", "add product", "new item", "custom",
                                            "another product", "different item", "my own", "train on", "own item")),
                           ("batch", ("batch", "folder", "many images", "multiple")),
                           ("video", ("video",)), ("camera", ("camera", "webcam", "live")),
                           ("review", ("review", "correct", "feedback", "annotat")),
                           ("report", ("report", "pdf", "export", "download")),
                           ("models", ("model", "train", "deploy", "approve", "rollback")),
                           ("alerts", ("alert",)), ("simulation", ("simulat",)), ("api", ("api", "integrat", "key")),
                           ("inspect", ("inspect", "check", "upload", "detect", "image", "photo", "picture", "pic",
                                        "defect", "use", "start"))):
            if any(k in ql for k in kws):
                return ans(call("app_help", topic=topic)["text"], "help")

    # --- concepts
    concept = None
    if ("confidence" in ql and "anomaly" in ql) or ("difference between" in ql and "score" in ql):
        concept = "confidence_vs_anomaly"
    elif _has(ql, "grad ?cam", "heat ?map", "explanation map"):
        concept = "gradcam"
    elif _has(ql, "quality index", "quality score"):
        concept = "quality_index"
    elif "severity" in ql and _has(ql, "what", "how", "mean", "explain", "decid", "calculat"):
        concept = "severity"
    elif _has(ql, "what (do|does|is|are) .*(pass|fail|review|inconclusive)", "meaning of (pass|fail|review)",
              "pass.*fail.*mean", "decisions? mean", "what .*status"):
        concept = "decisions"
    elif "unfamiliar" in ql or ("inconclusive" in ql and _has(ql, "why", "what", "mean")):
        concept = "unfamiliar"
    elif _has(ql, "what is anomaly", "how does anomaly", "anomaly detection work", "explain anomaly"):
        concept = "anomaly_detection"
    elif _has(ql, "active learning", "learning value", "priority"):
        concept = "active_learning"
    if concept and not e.iid and not e.refers_last:
        return ans(call("explain_concept", topic=concept)["text"], "help")

    # --- one inspection: explicit id, "last <product>", or a pronoun referring to the previous answer
    target = e.iid or ("latest" if e.refers_last and _has(ql, "why", "explain", "what happened", "reason", "detail",
                                                           "result", "show", "tell", "was", "verdict")
                       else None) or (e.context_iid if e.refers_pronoun else None)
    if target and _has(ql, "similar", "like (this|it|that)", "resembl", "same kind"):
        tid = target if target != "latest" else (_resolve(db, "latest", e.profile) or Inspection(id="")).id
        r = call("similar_defects", inspection_id=tid or "none")
        if "error" in r:
            return ans(f"I couldn't find that inspection ({r['error']}).")
        lines = [f"- {_link(x['id'])} · {x['status']} · {x.get('defect_class') or '—'} · similarity {x['similarity']:.2f}"
                 for x in r["results"]]
        return ans(f"Inspections most similar to {_link(r['query'])}:\n" + "\n".join(lines or ["- none found yet"]) +
                   f"\n\n_{r['caveat']}_", "inspection")
    if target:
        st = None if e.status in ("PASS",) and not e.refers_last else e.status
        if target == "latest" and e.status == "FAIL" and _has(ql, "why .* (pass|accept)"):
            st = None
        r = call("get_inspection", inspection_id=target, **({"profile_id": e.profile} if e.profile else {}),
                 **({"status": st} if (target == "latest" and st) else {}))
        if "error" in r:
            what = f"{e.profile.replace('_', ' ')} " if e.profile else ""
            return ans(f"I couldn't find {'that inspection' if target != 'latest' else f'any {what}inspection'}"
                       f"{' with status ' + st if (st and target == 'latest') else ''}. "
                       "You can browse all records under [Inspections](/inspections).")
        return ans(_explain_inspection(r), "inspection")

    # --- review queue
    if "review" in ql and _has(ql, "pending", "waiting", "queue", "need", "awaiting", "requir", "open", "outstanding",
                               "left", "how many", "any", "show", "list", "to do", "todo"):
        r = call("search_inspections", review_pending=True, limit=10,
                 **({"profile_id": e.profile} if e.profile else {}))
        lines = [f"- {_link(x['id'])} · {x['profile_id']} · {x['status']} · {x['created_at'][:16].replace('T', ' ')}"
                 + (" · simulated" if x["is_simulated"] else "") for x in r["items"]]
        head = f"**{r['count']}** inspection{'s' if r['count'] != 1 else ''} waiting for human review"
        return ans(head + (":\n" + "\n".join(lines) + "\n\nOpen the [review queue](/reviews) to work through them "
                           "(most informative first)." if lines else ". Nothing to do."), "stats")

    # --- alerts
    if _has(ql, "alert", "alarm", "warning", "critical", "notification"):
        st = "all" if _has(ql, "all alerts", "every alert", "history") else ("resolved" if "resolved" in ql else "active")
        r = call("list_alerts", status=st, limit=8)
        if not r["count"]:
            return ans(f"No {st if st != 'all' else ''} alerts. ✅ See [Alerts](/alerts).", "stats")
        lv = ", ".join(f"{v} {k}" for k, v in sorted(r["by_level"].items()))
        lines = [f"- **{a['level']}** · {a['title']} · {a['created_at'][:16].replace('T', ' ')}" +
                 (f" · {_link(a['inspection_id'])}" if a["inspection_id"] else "") for a in r["items"]]
        return ans(f"**{r['count']}** {st} alert{'s' if r['count'] != 1 else ''} ({lv}):\n" + "\n".join(lines) +
                   "\n\nManage them in [Alerts](/alerts).", "stats")

    # --- models
    if _has(ql, "model", "classifier", "accura", "precision", "recall", "f1", "auroc", "metric", "neural",
            "architecture", "mobilenet", "efficientnet", "resnet", "convnext", "densenet", "vit", "patchcore", "padim"):
        return ans(_models_answer(call("model_comparison"), e), "stats")

    # --- batches
    if "batch" in ql and _has(ql, "compar", "last two", "difference", "versus", "vs"):
        r = call("compare_batches")
        if "error" in r:
            return ans("There are fewer than two batches to compare yet. Run a batch from [Inspect](/inspect).")
        f = lambda b: (f"**{b['name'] or b['batch_id'][:8]}** ({b['created_at'][:16].replace('T', ' ')}, {b['source']}"  # noqa: E731
                       f"{', simulated' if b['simulated'] else ''}): {b['total']} items, acceptance "
                       f"{_pct(b['acceptance_rate'])}, FAIL {b['status'].get('FAIL', 0)}, review "
                       f"{b['status'].get('REVIEW REQUIRED', 0)}, top defects {b['defects'] or '—'}")
        return ans(f"- Older: {f(r['older'])}\n- Newer: {f(r['newer'])}", "stats")

    # --- trends
    if _has(ql, "trend", "increas", "decreas", "improv", "worse", "better", "getting", "compared to", "change",
            "going up", "going down", "rising", "falling"):
        days = e.days if e.explicit_period and e.days else 7
        r = call("quality_trend", days=days, **({"profile_id": e.profile} if e.profile else {}))
        c, p = r["current"], r["previous"]
        if not c["total"] and not p["total"]:
            return ans(f"No inspections in the last {2 * days} days, so there is no trend to report.")
        dr = lambda x: (x["status"].get("FAIL", 0) / x["total"]) if x["total"] else None  # noqa: E731
        dc, dp = dr(c), dr(p)
        verdict = ("not enough data in the earlier period to compare" if dp is None or not p["total"] else
                   "**worse** (more rejects)" if dc is not None and dc > dp + 0.02 else
                   "**better** (fewer rejects)" if dc is not None and dc < dp - 0.02 else "**about the same**")
        lines = [f"- Last {days} days: {c['total']} inspections, rejection rate {_pct(dc)}, acceptance {_pct(c['acceptance_rate'])}",
                 f"- {days} days before: {p['total']} inspections, rejection rate {_pct(dp)}, acceptance {_pct(p['acceptance_rate'])}"]
        moves = [x for x in r["defect_changes"] if x["change"]][:4]
        if moves:
            lines.append("- Biggest changes: " + ", ".join(f"{x['defect']} {x['previous']}→{x['current']}" for x in moves))
        scope = f" for {e.profile.replace('_', ' ')}" if e.profile else ""
        head = (f"There are no inspections{scope} from the {days} days before, so I can't say yet whether quality is "
                f"improving. Current figures:" if not p["total"] else
                f"Quality{scope} is {verdict} compared with the previous {days} days.")
        return ans(head + "\n" + "\n".join(lines) +
                   "\n\nCharts: [Analytics](/analytics).", "stats")

    # --- rejection rate by product
    if _has(ql, "rejection rate", "reject rate", "which product", "which profile", "worst product", "most rejected",
            "highest rejection", "rejected most", "by product", "per product"):
        r = call("defect_statistics", days=e.days)
        rows = sorted(r["by_profile"], key=lambda x: -x["rejection_rate"])
        if not rows:
            return ans(f"No inspections in {e.period}.")
        return ans(f"Rejection rate by product ({e.period}):\n" +
                   "\n".join(f"- **{x['profile_id']}**: {x['FAIL']} of {x['total']} rejected = {x['rejection_rate']:.1%}"
                             for x in rows) + f"\n\nHighest: **{rows[0]['profile_id']}**.", "stats")

    # --- defect frequency
    if _has(ql, "common", "frequent", "top defect", "most defect", "which defect", "defect type", "defect categor",
            "kinds of defect", "types of defect", "defects (found|detected|seen)", "what defects"):
        r = call("defect_statistics", days=e.days, **({"profile_id": e.profile} if e.profile else {}))
        top = r["by_class"][:8]
        if not top:
            return ans(f"No defects recorded in {e.period}.")
        sev = ", ".join(f"{x['name']} {x['count']}" for x in r["by_severity"] if x["count"] and x["name"] != "none")
        note = ("\n\n_'anomaly' = defects found by the anomaly detector, which locates a defect without naming its type._"
                if any(x["name"] == "anomaly" for x in top) else "")
        return ans(f"Defects found ({e.period}{', ' + e.profile if e.profile else ''}):\n" +
                   "\n".join(f"- **{x['name']}**: {x['count']}" for x in top) +
                   (f"\n\nBy severity: {sev}." if sev else "") + note, "stats")

    # --- knowledge base for a defect class
    if e.defect and _has(ql, "what is", "what are", "what does", "explain", "cause", "fix", "prevent", "look like",
                         "guidance", "tell me about", "about", "describe", "action"):
        r = call("knowledge_base", defect_class=e.defect)
        if "error" not in r:
            acts = "\n".join(f"- {a['text']}" for a in r["corrective_actions"]) or "- none verified yet"
            return ans(f"**{r['title']}**\n\n{r['visual_description']}\n\n**Inspection guidance**\n" +
                       "\n".join(f"- {g}" for g in r["inspection_guidance"]) +
                       f"\n\n**Verified follow-up actions**\n{acts}\n\n_Source: {r['source']}. These are suggestions "
                       "for investigation, not a diagnosis._", "stats")

    # --- counts / lists of inspections
    wants_list = _has(ql, "show", "list", "find", "which (items|inspections)", "give me", "display", "see")
    wants_count = _has(ql, "how many", "count", "number of", "total", "how much")
    if wants_list or wants_count or e.status or e.defect or (e.profile and "inspect" in ql):
        args = {"days": e.days, "limit": 8}
        if e.status:
            args["status"] = e.status
        if e.profile:
            args["profile_id"] = e.profile
        if e.defect:
            args["defect_class"] = e.defect
        r = call("search_inspections", **args)
        what = " ".join(x for x in (PLAIN.get(e.status, "") if e.status else "", e.defect or "",
                                    e.profile.replace("_", " ") if e.profile else "") if x)
        noun = "inspection" if r["count"] == 1 else "inspections"
        head = f"**{r['count']}** {what + ' ' if what else ''}{noun} in {e.period}"
        if not r["count"]:
            return ans(head + ".", "stats")
        if wants_count and not wants_list:
            dash = call("get_dashboard_summary", days=e.days or 3650, **({"profile_id": e.profile} if e.profile else {}))
            share = f" — {r['count'] / dash['total']:.1%} of all {dash['total']} inspections" if dash["total"] and e.status else ""
            return ans(head + share + f". [See them](/inspections{_query(e)}).", "stats")
        lines = [f"- {_link(x['id'])} · {x['profile_id']} · **{x['status']}** · {x['defect_class'] or '—'} · "
                 f"{x['severity']} · {x['created_at'][:16].replace('T', ' ')}" + (" · simulated" if x["is_simulated"] else "")
                 for x in r["items"]]
        more = f"\n\n[See all {r['count']}](/inspections{_query(e)})" if r["count"] > len(lines) else ""
        return ans(head + ":\n" + "\n".join(lines) + more, "inspection")

    # --- overview / summary
    if _has(ql, "summar", "overview", "status", "how (are|is) (we|it|things|quality)", "report", "dashboard",
            "quality", "performance", "statistic", "stats", "kpi"):
        return ans(_summary(call("get_dashboard_summary", days=e.days or 3650, **(
            {"profile_id": e.profile} if e.profile else {})), e), "stats")

    # --- not understood: say so, suggest
    return ans("I'm not sure what you're asking. I can help with questions like:\n"
               "- *How many items failed this week?*\n- *Why was the last bottle rejected?*\n"
               "- *Which defect is most common?*  ·  *Is quality getting worse?*\n"
               "- *What needs review?*  ·  *Any critical alerts?*\n"
               "- *How accurate is the steel model?*\n- *How do I check a picture of an item?*")


def _query(e: Entities) -> str:
    parts = []
    if e.status:
        parts.append(f"status={e.status.replace(' ', '%20')}")
    if e.profile:
        parts.append(f"profile_id={e.profile}")
    if e.defect:
        parts.append(f"defect_class={e.defect}")
    return ("?" + "&".join(parts)) if parts else ""


def _explain_inspection(r: dict) -> str:
    d = r.get("decision") or {}
    when = r["created_at"][:16].replace("T", " ")
    parts = [f"Inspection {_link(r['id'])} ({r['profile_id']}, {when}{', simulated' if r['is_simulated'] else ''}) → "
             f"**{r['status']}** — {PLAIN.get(r['status'], '')}."]
    if r.get("defect_class") and r["status"] in ("FAIL", "REVIEW REQUIRED", "PASS"):
        parts.append(f"- Defect: **{r['defect_class']}**, severity **{r['severity']}**.")
    elif r.get("defect_class"):
        parts.append(f"- The classifier's best guess was '{r['defect_class']}', but the result was not accepted.")
    elif r["status"] == "FAIL":
        parts.append(f"- Defect located by the anomaly detector, severity **{r['severity']}**.")
    for x in d.get("reasons", [])[:4]:
        parts.append(f"- Reason: {x}")
    c = r.get("classification")
    if c:
        parts.append(f"- Classifier: '{c['predicted_class']}' at {c['confidence']:.1%} calibrated confidence "
                     f"(models {'agree' if c['members_agree'] else '**disagree**'}).")
    a = r.get("anomaly")
    if a:
        parts.append("- Anomaly score vs threshold: " +
                     ", ".join(f"{k} {v / a['thresholds'][k]:.2f}×" for k, v in a["scores"].items()) +
                     " (above 1.00× = abnormal).")
    mm = r.get("measurements") or {}
    if mm.get("num_regions"):
        parts.append(f"- {mm['num_regions']} defect region(s) covering {mm.get('defect_area_ratio', 0):.2%} of the image.")
    if r.get("unfamiliar"):
        parts.append("- The image looks **unlike the training data**, so the models did not decide on their own.")
    if r.get("human_decision"):
        parts.append(f"- Human decision: **{r['human_decision']}**.")
    elif r.get("review_status") == "pending":
        parts.append("- Waiting for human review in the [review queue](/reviews).")
    parts.append(f"\n[Open the full record]({'/inspections/' + r['id']}) · [explain visually](/xai?inspection={r['id']})")
    return "\n".join(parts)


def _models_answer(r: dict, e: Entities) -> str:
    deployed = r.get("deployed_by_product", {})
    dep_ids = {rid for m in deployed.values() for rid in (m.get("classifiers") or []) + ([m["anomaly"]] if m.get("anomaly") else [])}
    clf = [x for x in r["models"] if x["task"] == "classification" and x["metrics"]]
    an = [x for x in r["models"] if x["task"] == "anomaly_detection" and x["metrics"]]
    want_steel = e.profile == "steel_strip" or _has(e.ql, "classif", "steel")
    want_anom = e.profile in ("bottle", "metal_nut") or _has(e.ql, "anomaly", "bottle", "nut", "patchcore", "padim")
    if e.profile and e.profile not in ("steel_strip", "bottle", "metal_nut"):  # a custom product
        want_anom, want_steel = True, False
    if not (want_steel or want_anom):
        want_steel = want_anom = True
    lines = []
    if want_steel and clf:
        lines.append("**Steel defect classifiers** (270 held-out test images):")
        for x in sorted(clf, key=lambda x: (-x["metrics"].get("accuracy", 0), x["metrics"].get("latency_ms_median", 1e9))):
            mt = x["metrics"]
            tag = " — **in use**" if x["run_id"] in dep_ids else ""
            lines.append(f"- {x['architecture']}: accuracy {mt.get('accuracy', 0):.1%}, macro-F1 {mt.get('f1_macro', 0):.3f}, "
                         f"{mt.get('latency_ms_median', 0):.0f} ms{tag}")
    if want_anom and an:
        lines.append("\n**Anomaly detectors** (held-out test images):")
        for x in an:
            mt = x["metrics"]
            tag = " — **in use**" if x["run_id"] in dep_ids else ""
            name = x["run_id"].replace("mvtec_", "").split("-anomaly")[0].replace("custom_", "")
            auroc = mt.get("patchcore_image_auroc")
            lines.append(f"- {name}: PatchCore AUROC {auroc:.3f}" if auroc is not None else f"- {name}: trained")
            if mt.get("patchcore_false_negative_rate") is not None:
                lines[-1] += f", missed-defect rate {mt['patchcore_false_negative_rate']:.1%}"
            lines[-1] += tag
    if not lines:
        return "No evaluated models are registered yet. See [Models & training](/models)."
    return ("\n".join(lines) + f"\n\n_{r['note']} The steel benchmark is easy; re-validate on real line images._"
            "\n\nDetails: [Models & training](/models).")


def _summary(r: dict, e: Entities) -> str:
    if not r["total"]:
        return f"No inspections recorded in {e.period}. Try [Inspect](/inspect) or [Line simulation](/simulation)."
    s = r["by_status"]
    scope = f" for {e.profile.replace('_', ' ')}" if e.profile else ""
    sim = f", {r['simulated']} simulated" if r.get("simulated") else ""
    txt = (f"Quality{scope} in {e.period} ({r['total']} inspections{sim}):\n"
           f"- Not defective (PASS): **{s['PASS']}** · Defective (FAIL): **{s['FAIL']}** · Needs review: "
           f"**{s['REVIEW REQUIRED']}** · Inconclusive: **{s['INCONCLUSIVE']}**\n"
           f"- Acceptance rate {_pct(r['acceptance_rate'])}, defect detection rate {_pct(r['defect_detection_rate'])}\n"
           f"- Average inspection time {r['avg_latency_ms']:.0f} ms")
    ins = [f"- {i['text']}" for i in r.get("insights", [])]
    return txt + ("\n" + "\n".join(ins) if ins else "") + "\n\nMore in the [Dashboard](/) and [Analytics](/analytics)."


def chat(db: Session, user: User, question: str, history: list[dict] | None = None) -> dict:
    question = question.strip()[:2000]
    if not question:
        return {"answer": "Please type a question.", "tool_calls": [], "mode": "offline", "suggestions": SUGGEST["default"]}
    if llm_available():
        return chat_llm(db, user, history or [], question)
    try:
        return chat_offline(db, user, question, history or [])
    except Exception:  # never surface a stack trace to the chat
        log.exception("assistant offline error")
        return {"answer": "Sorry - something went wrong while answering. Please rephrase or try one of the suggestions.",
                "tool_calls": [], "mode": "offline", "suggestions": SUGGEST["default"]}
