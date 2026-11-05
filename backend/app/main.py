"""QualityGuard AI - FastAPI application entry point.

Run (from the repository root):
    uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000

Serves:
    /api/...      REST API (OpenAPI docs at /api/docs)
    /api/ws/events  live event WebSocket
    /gradio       lightweight Gradio inspection UI (if gradio is installed)
    /             React dashboard (frontend/dist, if built)
"""
from __future__ import annotations

import os

# Inspection images are treated as confidential: no third-party usage analytics from Gradio / Hugging Face.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import logging
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # make the top-level `ml` package importable
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.middleware.gzip import GZipMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402

settings = get_settings()
log = setup_logging()


def _housekeeping():
    """Startup cleanup: expired revoked tokens, jobs orphaned by a restart, optional data retention."""
    from datetime import timedelta

    from app.database.session import session_scope
    from app.models.entities import Inspection, Job, RevokedToken, TrainingJob, utcnow
    from app.services.storage import storage
    with session_scope() as db:
        db.query(RevokedToken).filter(RevokedToken.expires_at < utcnow()).delete()
        for j in db.query(Job).filter(Job.status.in_(("queued", "running"))).all():
            j.status, j.message = "failed", "interrupted by server restart"
        for t in db.query(TrainingJob).filter(TrainingJob.status.in_(("queued", "running"))).all():
            t.status, t.message = "stopped", "interrupted by server restart (resumable)"
        if settings.data_retention_days > 0:
            cutoff = utcnow() - timedelta(days=settings.data_retention_days)
            old = db.query(Inspection).filter(Inspection.created_at < cutoff).limit(5000).all()
            for i in old:
                storage.delete_tree(i.artifact_dir)
                db.delete(i)
            if old:
                log.info("retention: removed %d inspections older than %d days", len(old), settings.data_retention_days)


@asynccontextmanager
async def lifespan(_: FastAPI):
    from app.core.bootstrap import init_db
    if settings.torch_threads:
        import torch
        torch.set_num_threads(settings.torch_threads)
    init_db()
    _housekeeping()
    from app.integrations import start_all
    start_all()
    log.info("%s started (%s)", settings.app_name, settings.environment)
    yield
    from app.workers.jobs import MANAGER
    MANAGER.pool.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan, docs_url="/api/docs",
              redoc_url=None, openapi_url="/api/openapi.json",
              description="AI-assisted industrial visual inspection: classification, anomaly detection, "
                          "explainability, review workflow, analytics and reporting.")
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_list, allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def security_headers(request: Request, call_next):
    t0 = time.perf_counter()
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("Referrer-Policy", "same-origin")
    if not request.url.path.startswith("/gradio"):
        resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers["X-Response-Time-ms"] = f"{(time.perf_counter() - t0) * 1000:.1f}"
    return resp


@app.exception_handler(RequestValidationError)
async def validation_handler(_: Request, exc: RequestValidationError):
    errs = [{"field": ".".join(str(x) for x in e.get("loc", [])[1:]), "message": e.get("msg")} for e in exc.errors()]
    return JSONResponse({"detail": "; ".join(f"{e['field']}: {e['message']}" for e in errs), "errors": errs},
                        status_code=422)


@app.exception_handler(PermissionError)
async def permission_handler(_: Request, exc: PermissionError):
    return JSONResponse({"detail": str(exc)}, status_code=400)


from app.api import access, analytics, auth, catalog, inspect, inspections, models, operations, products, reports, system  # noqa: E402,E501

for r in (system.router, auth.router, access.router, inspect.router, inspections.router, operations.router, analytics.router, models.router,
          catalog.router, reports.router, products.router):
    app.include_router(r)

# ---- optional Gradio UI (the proposal's lightweight interface) at /gradio
try:
    import gradio as gr

    from app.gradio_ui import authenticate, build_ui
    from ml.config import CHECKPOINT_DIR
    app = gr.mount_gradio_app(app, build_ui(), path="/gradio", auth=authenticate,
                              auth_message="Sign in with your QualityGuard account (requires the 'inspect' permission).",
                              allowed_paths=[str(settings.storage_dir), str(CHECKPOINT_DIR)],
                              max_file_size=f"{settings.max_upload_mb}mb", footer_links=[],
                              theme=gr.themes.Soft(primary_hue="blue"))
except ImportError:
    log.info("gradio not installed; /gradio UI disabled (pip install gradio)")

# ---- built React dashboard (frontend/dist) with SPA fallback
DIST = ROOT / "frontend" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"):
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        f = (DIST / path).resolve()
        if path and DIST.resolve() in f.parents and f.is_file():
            return FileResponse(f)
        return FileResponse(DIST / "index.html")
