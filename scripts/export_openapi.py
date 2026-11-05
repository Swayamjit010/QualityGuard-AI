"""Export the API schema and a Markdown endpoint reference generated from it.

    python scripts/export_openapi.py

Writes docs/api/openapi.json and docs/API.md. The interactive version is served at /api/docs.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("QG_STORAGE_DIR", tempfile.mkdtemp(prefix="qg_openapi_"))  # no side effects on storage/
os.environ.setdefault("QG_DATABASE_URL", "sqlite://")
sys.path[:0] = [str(ROOT / "backend"), str(ROOT)]

from app.main import app  # noqa: E402

PERM_HINT = "require("


def main():
    spec = app.openapi()
    out = ROOT / "docs" / "api"
    out.mkdir(parents=True, exist_ok=True)
    (out / "openapi.json").write_text(json.dumps(spec, indent=1))

    groups: dict[str, list] = defaultdict(list)
    for path, ops in spec["paths"].items():
        for method, op in ops.items():
            tag = (op.get("tags") or ["other"])[0]
            summary = (op.get("description") or op.get("summary") or "").strip().split("\n")[0]
            params = [p["name"] for p in op.get("parameters", []) if p.get("in") == "query"]
            body = op.get("requestBody", {}).get("content", {})
            kind = "multipart" if "multipart/form-data" in body else "JSON" if "application/json" in body else ""
            groups[tag].append((path, method.upper(), summary, params, kind))
    md = ["# API reference", "",
          "Generated from the live OpenAPI schema by `python scripts/export_openapi.py`; the machine-readable",
          "schema is [api/openapi.json](api/openapi.json) and the interactive explorer is served at `/api/docs`.", "",
          "**Authentication.** `POST /api/auth/login` returns a JWT; send it as `Authorization: Bearer <token>`.",
          "Machine clients use an API key (Administration → API keys) in the `X-API-Key` header. Browser downloads",
          "and the WebSocket accept `?token=`. Every route except `/api/health`, `/api/auth/login` and",
          "`/api/auth/register` requires authentication, and each checks a permission (see ARCHITECTURE.md).", "",
          "**Errors** are JSON `{\"detail\": ...}` with the usual status codes: 401 (not signed in), 403 (missing",
          "permission), 404, 413 (upload too large), 415 (unsupported file type), 422 (validation), 429 (rate limit),",
          "503 (a required model checkpoint is missing - never replaced by a mock).", ""]
    total = 0
    for tag in sorted(groups):
        rows = sorted(groups[tag], key=lambda r: (r[0], r[1]))
        total += len(rows)
        md += [f"## {tag}", "", "| Method | Path | Description | Query parameters | Body |", "|---|---|---|---|---|"]
        for path, method, summary, params, kind in rows:
            md.append(f"| {method} | `{path}` | {summary.replace('|', '/')} | {', '.join(params)} | {kind} |")
        md.append("")
    md.insert(2, f"{total} operations in {len(groups)} groups.\n")
    (ROOT / "docs" / "API.md").write_text("\n".join(md), encoding="utf-8")
    print(f"{total} operations -> docs/API.md, docs/api/openapi.json")


if __name__ == "__main__":
    main()
