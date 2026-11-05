"""Back up the database and stored inspection files (Part 21: database backup procedures).

    python scripts/backup.py                         # -> backups/<UTC timestamp>/
    python scripts/backup.py --out D:/qg-backups     # another destination
    python scripts/backup.py --no-files              # database only

SQLite: uses SQLite's online backup API, so it is safe while the server is running, then runs
``PRAGMA integrity_check`` on the copy. PostgreSQL: runs ``pg_dump --format=custom`` (must be on PATH).
Stored files (uploads, visualisations, reports, videos) are archived as storage.tar.gz. A manifest.json
records sizes and SHA-256 checksums.

Restore (server stopped):
    SQLite      copy qualityguard.db back to the path in QG_DATABASE_URL
    PostgreSQL  pg_restore --clean --dbname <url> qualityguard.dump
    files       tar -xzf storage.tar.gz -C <QG_STORAGE_DIR parent>
The backup contains confidential inspection data and the initial-credentials file if it still exists:
store it with the same access controls as the live system.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT)]

from app.core.config import get_settings  # noqa: E402


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def backup_sqlite(url: str, dest: Path) -> Path:
    src_path = url.split("sqlite:///", 1)[1]
    out = dest / Path(src_path).name
    src = sqlite3.connect(f"file:{src_path}?mode=ro", uri=True)
    dst = sqlite3.connect(out)
    with dst:
        src.backup(dst)
    src.close()
    ok = dst.execute("PRAGMA integrity_check").fetchone()[0]
    dst.close()
    if ok != "ok":
        raise SystemExit(f"integrity check failed on the backup copy: {ok}")
    return out


def backup_postgres(url: str, dest: Path) -> Path:
    if not shutil.which("pg_dump"):
        raise SystemExit("pg_dump not found on PATH; install the PostgreSQL client tools")
    out = dest / "qualityguard.dump"
    subprocess.run(["pg_dump", "--format=custom", "--file", str(out), url.replace("+psycopg", "")], check=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "backups"))
    ap.add_argument("--no-files", action="store_true", help="skip the storage/ archive")
    a = ap.parse_args()
    s = get_settings()
    dest = Path(a.out) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest.mkdir(parents=True, exist_ok=False)
    url = s.database_url
    if url.startswith("sqlite:///"):
        db_file = backup_sqlite(url, dest)
    elif url.startswith("postgresql"):
        db_file = backup_postgres(url, dest)
    else:
        raise SystemExit(f"unsupported database URL for backup: {url.split(':', 1)[0]}")
    files = [db_file]
    if not a.no_files and s.storage_dir.exists():
        arc = dest / "storage.tar.gz"
        skip = {db_file.name, Path(url.split("sqlite:///", 1)[-1]).name} if url.startswith("sqlite") else set()
        with tarfile.open(arc, "w:gz") as tar:
            tar.add(s.storage_dir, arcname=s.storage_dir.name,
                    filter=lambda ti: None if Path(ti.name).name in skip | {".secret_key"} else ti)
        files.append(arc)
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "database": url.split(":", 1)[0], "storage_dir": str(s.storage_dir),
                "files": [{"name": f.name, "bytes": f.stat().st_size, "sha256": sha256(f)} for f in files],
                "note": "The JWT signing key (storage/.secret_key or QG_SECRET_KEY) is NOT included; keep it separately."}
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for f in manifest["files"]:
        print(f"{f['name']:<24} {f['bytes'] / 2**20:8.1f} MB  sha256 {f['sha256'][:16]}...")
    print(f"backup written to {dest}")


if __name__ == "__main__":
    main()
