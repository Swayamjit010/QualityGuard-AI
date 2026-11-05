"""In-process pub/sub used by the WebSocket endpoint and outbound integration adapters.
Thread-safe publish from worker threads into asyncio subscriber queues."""
from __future__ import annotations

import asyncio
import json
import logging
import threading

log = logging.getLogger("qualityguard.events")


class EventBus:
    def __init__(self):
        self.subs: set[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = set()
        self.lock = threading.Lock()
        self.sinks: list = []  # callables(event) for adapters (webhook, mqtt...)

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=500)
        with self.lock:
            self.subs.add((asyncio.get_running_loop(), q))
        return q

    def unsubscribe(self, q: asyncio.Queue):
        with self.lock:
            self.subs = {s for s in self.subs if s[1] is not q}

    def publish(self, kind: str, data: dict):
        ev = {"type": kind, "data": data}
        payload = json.dumps(ev, default=str)
        with self.lock:
            subs = list(self.subs)
        for loop, q in subs:
            try:
                loop.call_soon_threadsafe(_put, q, payload)
            except RuntimeError:
                pass
        for sink in self.sinks:
            try:
                sink(ev)
            except Exception as exc:  # adapters must never break inspections
                log.warning("event sink failed: %s", exc)


def _put(q: asyncio.Queue, item):
    if q.full():
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            pass
    q.put_nowait(item)


bus = EventBus()
