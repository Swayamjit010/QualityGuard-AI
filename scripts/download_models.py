"""Download the trained QualityGuard AI checkpoints from the GitHub Release (so the app runs without retraining).

    python scripts/download_models.py [--tag v1.0.0] [--repo Swayamjit010/QualityGuard-AI]

Extracts into artifacts/checkpoints/. The archive's SHA-256 is checked against the .sha256 file published with it.
Models are trained on CC BY-NC-SA 4.0 datasets (MVTec AD, KolektorSDD2, MPDD) and NEU-CLS: non-commercial use only.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
ASSET = "qualityguard_models.zip"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Swayamjit010/QualityGuard-AI")
    ap.add_argument("--tag", default="v1.0.0")
    a = ap.parse_args()
    base = f"https://github.com/{a.repo}/releases/download/{a.tag}"
    dest = ROOT / "artifacts" / "checkpoints"
    dest.mkdir(parents=True, exist_ok=True)
    zp = ROOT / "artifacts" / ASSET
    expected = requests.get(f"{base}/{ASSET}.sha256", timeout=60).text.split()[0]
    print(f"downloading {base}/{ASSET} ...", flush=True)
    h = hashlib.sha256()
    with requests.get(f"{base}/{ASSET}", stream=True, timeout=120) as r, open(zp, "wb") as f:
        r.raise_for_status()
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
            h.update(chunk)
    if h.hexdigest() != expected:
        zp.unlink(missing_ok=True)
        sys.exit("checksum mismatch - download corrupted, try again")
    out = dest.resolve()
    with zipfile.ZipFile(zp) as z:
        for m in z.infolist():
            target = (out / m.filename).resolve()
            if not str(target).startswith(str(out)):
                sys.exit(f"unsafe path in archive: {m.filename}")
        z.extractall(out)
    zp.unlink()
    print(f"models extracted to {out}")


if __name__ == "__main__":
    main()
