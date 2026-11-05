"""Application logging: rotating file + in-memory ring buffer exposed at /api/system/logs."""
from __future__ import annotations

import logging
import logging.handlers
from collections import deque
from datetime import datetime, timezone

from app.core.config import get_settings

RING: deque = deque(maxlen=2000)


class RingHandler(logging.Handler):
    def emit(self, record):
        RING.append({"time": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
                     "level": record.levelname, "logger": record.name, "message": self.format(record)})


def setup_logging() -> logging.Logger:
    s = get_settings()
    log_dir = s.storage_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    if getattr(root, "_qg_configured", False):
        return logging.getLogger("qualityguard")
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fh = logging.handlers.RotatingFileHandler(log_dir / "app.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    fh.setFormatter(fmt)
    rh = RingHandler()
    rh.setFormatter(logging.Formatter("%(message)s"))
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    for h in (fh, rh, sh):
        root.addHandler(h)
    root._qg_configured = True  # type: ignore[attr-defined]
    for noisy in ("multipart", "PIL", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger("qualityguard")
