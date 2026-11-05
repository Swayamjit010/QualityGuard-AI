"""Storage abstraction. Local filesystem by default; an S3-compatible backend can implement the same
interface (``put_bytes``, ``put_image``, ``path``, ``open``) without touching callers.

All keys are generated server-side (uuid-based); user-supplied filenames are never used as paths,
which prevents path traversal. ``resolve`` additionally verifies the final path stays inside root.
"""
from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from app.core.config import get_settings

SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(name: str, max_len: int = 120) -> str:
    base = Path(name or "file").name
    base = SAFE.sub("_", base).strip("._") or "file"
    return base[-max_len:]


class LocalStorage:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if self.root not in p.parents and p != self.root:
            raise PermissionError("Path escapes storage root")
        return p

    def put_bytes(self, key: str, data: bytes) -> str:
        p = self.resolve(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return key

    def put_image(self, key: str, rgb: np.ndarray, quality: int = 92) -> str:
        p = self.resolve(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        img = rgb if rgb.ndim == 2 else cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(p.suffix or ".png", img, [cv2.IMWRITE_JPEG_QUALITY, quality] if p.suffix == ".jpg" else [])
        if not ok:
            raise IOError(f"encode failed for {key}")
        p.write_bytes(buf.tobytes())
        return key

    def exists(self, key: str) -> bool:
        return self.resolve(key).exists()

    def delete_tree(self, key: str):
        p = self.resolve(key)
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink()

    def usage(self) -> dict:
        total, n = 0, 0
        for f in self.root.rglob("*"):
            if f.is_file():
                total += f.stat().st_size; n += 1
        du = shutil.disk_usage(self.root)
        return {"files": n, "bytes": total, "disk_free_bytes": du.free, "disk_total_bytes": du.total}


def inspection_dir(inspection_id: str) -> str:
    return f"inspections/{datetime.utcnow():%Y/%m/%d}/{inspection_id}"


storage = LocalStorage(get_settings().storage_dir)
