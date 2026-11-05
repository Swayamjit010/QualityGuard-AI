"""Download the public industrial-defect datasets used by QualityGuard AI.

Sources (verified reachable 2026-11-05):
  * NEU-CLS (Northeastern University steel surface defects, 1,800 images, 6 classes)
      mirror: huggingface.co/datasets/newguyme/neu_cls_caption  (parquet, includes original filenames)
      original: http://faculty.neu.edu.cn/songkechen  -- research use
  * MVTec AD (bottle, metal_nut by default) -- CC BY-NC-SA 4.0, non-commercial research only
      mirror: huggingface.co/datasets/foersben/mvtec-ad (original folder layout + license.txt)
  * KolektorSDD (optional, --kolektor) -- CC BY-NC-SA 4.0
      mirror: huggingface.co/datasets/hdtech/kolektor_surface_defect_dataset
  * KolektorSDD2 (--ksdd2) -- CC BY-NC-SA 4.0, official ViCoS archive with pixel masks (~853 MB)
      https://data.vicos.si/datasets/KSDD/KolektorSDD2.zip  (via https://go.vicos.si/kolektorsdd2)

Usage:
    python scripts/download_datasets.py --neu --mvtec bottle metal_nut screw grid --ksdd2 --mpdd [--kolektor]

The script is idempotent: existing files are skipped. Images are written under data/raw/<dataset>/.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
HF = "https://huggingface.co"
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "QualityGuardAI-dataset-fetcher/1.0"


def _get(url: str, timeout: int = 120) -> bytes:
    for attempt in range(4):
        try:
            r = SESSION.get(url, timeout=timeout)
            r.raise_for_status()
            return r.content
        except Exception as exc:  # network hiccups
            if attempt == 3:
                raise
            print(f"  retry {attempt + 1} {url}: {exc}", flush=True)
    raise RuntimeError("unreachable")


def download_neu() -> None:
    import pandas as pd
    from PIL import Image

    out = RAW / "neu_cls"
    marker = out / "_COMPLETE.json"
    if marker.exists():
        print("NEU-CLS already present"); return
    out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split in ("train", "test"):
        url = f"{HF}/datasets/newguyme/neu_cls_caption/resolve/main/data/{split}-00000-of-00001.parquet"
        print("Downloading", url, flush=True)
        df = pd.read_parquet(io.BytesIO(_get(url, 300)))
        for _, row in df.iterrows():
            label = str(row["label_str"]).strip()
            fname = Path(str(row["filename"])).name
            d = out / label
            d.mkdir(exist_ok=True)
            img = Image.open(io.BytesIO(row["image"]["bytes"]))
            stem = Path(fname).stem
            img.save(d / f"{stem}.png")  # lossless re-encode of original bitmap
            counts[label] = counts.get(label, 0) + 1
    marker.write_text(json.dumps({"source": "huggingface:newguyme/neu_cls_caption",
                                  "counts": counts}, indent=2))
    print("NEU-CLS counts:", counts)


def _hf_tree(repo: str, path: str) -> list[dict]:
    r = SESSION.get(f"{HF}/api/datasets/{repo}/tree/main/{path}?recursive=true", timeout=60)
    r.raise_for_status()
    return [x for x in r.json() if x["type"] == "file"]


def download_mvtec(categories: list[str]) -> None:
    repo = "foersben/mvtec-ad"
    base = RAW / "mvtec_ad"
    base.mkdir(parents=True, exist_ok=True)
    lic = base / "license.txt"
    if not lic.exists():
        lic.write_bytes(_get(f"{HF}/datasets/{repo}/resolve/main/license.txt"))
    for cat in categories:
        files = _hf_tree(repo, cat)
        todo = [f for f in files if not (base / f["path"]).exists()]
        print(f"MVTec {cat}: {len(files)} files, {len(todo)} to fetch", flush=True)

        def fetch(f: dict) -> None:
            dest = base / f["path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(_get(f"{HF}/datasets/{repo}/resolve/main/{f['path']}"))

        with ThreadPoolExecutor(8) as ex:
            for i, _ in enumerate(ex.map(fetch, todo)):
                if i % 100 == 0:
                    print(f"  {cat}: {i}/{len(todo)}", flush=True)
        print(f"MVTec {cat}: done", flush=True)


def download_kolektor() -> None:
    out = RAW / "kolektor_sdd"
    if (out / "_COMPLETE.json").exists():
        print("KolektorSDD already present"); return
    url = f"{HF}/datasets/hdtech/kolektor_surface_defect_dataset/resolve/main/kolektor_surface_defect_dataset.zip"
    print("Downloading", url, flush=True)
    data = _get(url, 900)
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for m in z.infolist():
            target = (out / m.filename).resolve()
            if not str(target).startswith(str(out.resolve())):  # zip-slip guard
                raise RuntimeError(f"unsafe path in zip: {m.filename}")
            z.extract(m, out)
    (out / "_COMPLETE.json").write_text(json.dumps({"source": url}))
    print("KolektorSDD done")


KSDD2_URL = "https://data.vicos.si/datasets/KSDD/KolektorSDD2.zip"
KSDD2_SIZE = 853126555  # Content-Length of the official archive, checked 2026-11-05


def safe_extract(zip_path: Path, out: Path) -> int:
    """Extract with zip-slip and zip-bomb guards; only image files are written."""
    out = out.resolve()
    n = 0
    with zipfile.ZipFile(zip_path) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError(f"corrupt member in {zip_path.name}: {bad}")
        total = sum(m.file_size for m in z.infolist())
        if total > 5 * 1024 ** 3:
            raise RuntimeError("archive expands to > 5 GB; refusing")
        for m in z.infolist():
            if m.is_dir() or Path(m.filename).suffix.lower() not in {".png", ".jpg", ".bmp"}:
                continue
            target = (out / m.filename).resolve()
            if not str(target).startswith(str(out)):
                raise RuntimeError(f"unsafe path in zip: {m.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(m))
            n += 1
    return n


def download_ksdd2() -> None:
    out = RAW / "ksdd2"
    if (out / "_COMPLETE.json").exists():
        print("KolektorSDD2 already present"); return
    dl = ROOT / "data" / "downloads" / "ksdd2_official"
    dl.mkdir(parents=True, exist_ok=True)
    zp = dl / "KolektorSDD2.zip"
    if not zp.exists() or zp.stat().st_size != KSDD2_SIZE:
        print("Downloading", KSDD2_URL, flush=True)
        with SESSION.get(KSDD2_URL, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(zp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
    if zp.stat().st_size != KSDD2_SIZE:
        raise RuntimeError(f"size mismatch: {zp.stat().st_size} != {KSDD2_SIZE}")
    n = safe_extract(zp, out)
    (out / "_COMPLETE.json").write_text(json.dumps({"source": KSDD2_URL, "files": n,
                                                    "license": "CC BY-NC-SA 4.0 (ViCoS / Kolektor)"}, indent=2))
    print(f"KolektorSDD2: {n} files extracted")


MPDD_URL = "https://huggingface.co/datasets/meksamiao/mpdd/resolve/main/MPDD.zip"
MPDD_SIZE = 1825041283  # bytes of the mirror archive, checked 2026-11-05


def download_mpdd() -> None:
    """MPDD - Metal Parts Defect Detection (Jezek et al., 2021; github.com/stepanje/MPDD), CC BY-NC-SA 4.0.
    Six metal-part categories photographed at varied positions, orientations and backgrounds; MVTec layout."""
    out = RAW / "mpdd"
    if (out / "_COMPLETE.json").exists():
        print("MPDD already present"); return
    dl = ROOT / "data" / "downloads" / "mpdd"
    dl.mkdir(parents=True, exist_ok=True)
    zp = dl / "MPDD.zip"
    if not zp.exists() or zp.stat().st_size != MPDD_SIZE:
        print("Downloading", MPDD_URL, flush=True)
        with SESSION.get(MPDD_URL, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(zp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
    if zp.stat().st_size != MPDD_SIZE:
        raise RuntimeError(f"size mismatch: {zp.stat().st_size} != {MPDD_SIZE}")
    tmp = RAW / "_mpdd_extract"
    n = safe_extract(zp, tmp)
    (tmp / "MPDD").rename(out)
    tmp.rmdir()
    (out / "_COMPLETE.json").write_text(json.dumps({"source": MPDD_URL, "files": n, "license": "CC BY-NC-SA 4.0",
                                                    "original": "https://github.com/stepanje/MPDD"}, indent=2))
    print(f"MPDD: {n} files extracted")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--neu", action="store_true")
    ap.add_argument("--mvtec", nargs="*", default=None)
    ap.add_argument("--kolektor", action="store_true")
    ap.add_argument("--ksdd2", action="store_true")
    ap.add_argument("--mpdd", action="store_true")
    a = ap.parse_args()
    if a.neu:
        download_neu()
    if a.mvtec is not None:
        download_mvtec(a.mvtec or ["bottle", "metal_nut"])
    if a.kolektor:
        download_kolektor()
    if a.ksdd2:
        download_ksdd2()
    if a.mpdd:
        download_mpdd()
    return 0


if __name__ == "__main__":
    sys.exit(main())
