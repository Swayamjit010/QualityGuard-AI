"""HTTP webhook adapter: POSTs selected events as JSON to QG_WEBHOOK_URL (fire-and-forget, 5 s timeout).

Inspection events carry an extra ``line_signal`` object (PASS / REJECT / HOLD)."""
from __future__ import annotations

import logging
import threading

from app.core.config import get_settings
from app.integrations import line_signal, register
from app.services.events import bus

log = logging.getLogger("qualityguard.integrations.webhook")


def make_sink(url: str, kinds: set[str], post=None):
    stats = {"sent": 0, "failed": 0}

    def _post(body: dict):
        try:
            if post:
                post(url, body)
            else:
                import httpx
                httpx.post(url, json=body, timeout=5).raise_for_status()
            stats["sent"] += 1
        except Exception as e:  # never propagates into the inspection path
            stats["failed"] += 1
            log.warning("webhook delivery failed: %s", e)

    def sink(ev: dict):
        if ev["type"] not in kinds:
            return
        body = {**ev, "line_signal": line_signal(ev["data"])} if ev["type"] == "inspection" else ev
        if post:  # synchronous in tests
            _post(body)
        else:
            threading.Thread(target=_post, args=(body,), daemon=True).start()

    return sink, stats


def start() -> None:
    s = get_settings()
    if not s.webhook_url:
        return
    kinds = {k.strip() for k in s.webhook_events.split(",") if k.strip()}
    sink, stats = make_sink(s.webhook_url, kinds)
    bus.sinks.append(sink)
    register("webhook", {"name": "webhook", "enabled": True, "state": "active", "stats": stats,
                         "detail": f"events {sorted(kinds)} -> {s.webhook_url.split('?')[0]}"})
