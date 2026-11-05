"""Deploy QualityGuard AI to a Hugging Face Space (Docker SDK).

    python scripts/deploy_hf_space.py --space <hf-user>/qualityguard-ai [--private] [--github-url URL] [--dry-run]

Uploads: source code, configs, the frontend source (built inside the Space image), and ONLY the trained checkpoints
the product profiles + component recogniser use (with their metrics/plots). No datasets, no local database, no
uploads, no backups, no credentials.

Secrets (set on the Space, never written to the repo):
  QG_SECRET_KEY              random 64-char JWT signing key
  QG_DEFAULT_ADMIN_PASSWORD  random admin password, also saved locally to storage/HF_SPACE_ADMIN.txt (git-ignored)
Existing secrets are kept unless --rotate-secrets is given.
"""
from __future__ import annotations

import argparse
import json
import secrets
import shutil
import string
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_DIRS = ["backend", "ml", "scripts", "configs"]
FRONTEND_FILES = ["package.json", "package-lock.json", "index.html", "vite.config.js", "tsconfig.json",
                  "tailwind.config.js", "postcss.config.js"]
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "node_modules", "dist")


def deployed_runs() -> list[str]:
    runs = {"component_recognizer-resnet18-v1"}
    for p in (ROOT / "configs" / "profiles").glob("*.json"):
        m = json.loads(p.read_text())["models"]
        runs |= set(m.get("classifiers", [])) | {m.get("anomaly"), m.get("segmentation")}
    runs.discard(None)
    missing = [r for r in runs if not (ROOT / "artifacts" / "checkpoints" / r).exists()]
    if missing:
        sys.exit(f"missing checkpoints (train them first): {missing}")
    return sorted(runs)


def stage(dst: Path, github_url: str | None) -> dict:
    for d in CODE_DIRS:
        shutil.copytree(ROOT / d, dst / d, ignore=SKIP)
    fe = dst / "frontend"
    shutil.copytree(ROOT / "frontend" / "src", fe / "src", ignore=SKIP)
    if (ROOT / "frontend" / "public").exists():
        shutil.copytree(ROOT / "frontend" / "public", fe / "public", ignore=SKIP)
    for f in FRONTEND_FILES:
        if (ROOT / "frontend" / f).exists():
            shutil.copy2(ROOT / "frontend" / f, fe / f)
    runs = deployed_runs()
    for r in runs:
        shutil.copytree(ROOT / "artifacts" / "checkpoints" / r, dst / "artifacts" / "checkpoints" / r,
                        ignore=shutil.ignore_patterns("best_state.pt", "last.pt", "STOP"))
    shutil.copytree(ROOT / "artifacts" / "metrics", dst / "artifacts" / "metrics")
    (dst / "data" / "processed").mkdir(parents=True)
    (dst / "data" / "processed" / ".keep").write_text("")
    for f in ("Dockerfile", "requirements-runtime.txt", "README.md"):
        shutil.copy2(ROOT / "deploy" / "hf_space" / f, dst / f)
    if github_url:
        readme = dst / "README.md"
        readme.write_text(readme.read_text(encoding="utf-8").replace(
            "see the GitHub repository linked in the Space.", f"{github_url}"), encoding="utf-8")
    size = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file())
    return {"runs": runs, "files": sum(1 for p in dst.rglob("*") if p.is_file()), "mb": size / 1e6}


def admin_password() -> str:
    alphabet = string.ascii_letters + string.digits
    while True:
        pw = "".join(secrets.choice(alphabet) for _ in range(20))
        if any(c.islower() for c in pw) and any(c.isupper() for c in pw) and any(c.isdigit() for c in pw):
            return pw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, help="<hf-user>/<space-name>")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--github-url", default=None)
    ap.add_argument("--dry-run", action="store_true", help="stage and report only; upload nothing")
    ap.add_argument("--rotate-secrets", action="store_true")
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="qg_space_"))
    info = stage(tmp / "space", a.github_url)
    print(f"staged {info['files']} files, {info['mb']:.0f} MB, {len(info['runs'])} model runs -> {tmp / 'space'}")
    if a.dry_run:
        return
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(a.space, repo_type="space", space_sdk="docker", private=a.private, exist_ok=True)
    cred = ROOT / "storage" / "HF_SPACE_ADMIN.txt"
    if a.rotate_secrets or not cred.exists():
        pw = admin_password()
        api.add_space_secret(a.space, "QG_SECRET_KEY", secrets.token_urlsafe(48))
        api.add_space_secret(a.space, "QG_DEFAULT_ADMIN_PASSWORD", pw)
        cred.parent.mkdir(parents=True, exist_ok=True)
        cred.write_text(f"Space: https://huggingface.co/spaces/{a.space}\nAdmin login: admin@qualityguard.local\n"
                        f"Admin password: {pw}\n(Change it after first sign-in. Stored only here and as a Space secret.)\n")
        print(f"secrets set; admin credentials saved to {cred}")
    api.upload_folder(repo_id=a.space, repo_type="space", folder_path=str(tmp / "space"),
                      commit_message="Deploy QualityGuard AI",
                      delete_patterns=[f"{d}/**" for d in CODE_DIRS + ["frontend", "artifacts", "data"]])
    shutil.rmtree(tmp, ignore_errors=True)
    host = a.space.replace("/", "-").replace("_", "-").lower()
    print(f"uploaded. Build log: https://huggingface.co/spaces/{a.space}  App: https://{host}.hf.space")


if __name__ == "__main__":
    main()
