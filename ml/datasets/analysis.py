"""Dataset quality analysis: integrity, dimensions, duplicates, image quality, class balance.

All statistics are computed from the actual files on disk; nothing is estimated.
"""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from ml.config import absp


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def dhash(gray: np.ndarray, size: int = 8) -> int:
    """Difference hash: robust to re-encoding/minor resizing; used for near-duplicate detection."""
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def confirm_near_duplicate(a: np.ndarray, b: np.ndarray, max_mad: float = 4.0, min_corr: float = 0.97) -> bool:
    """Pixel-level confirmation on equal-size grayscale thumbnails."""
    if float(np.abs(a - b).mean()) > max_mad:
        return False
    sa, sb = a.std(), b.std()
    if sa < 1e-6 or sb < 1e-6:
        return True
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) >= min_corr


def read_image(path: Path | str) -> np.ndarray | None:
    """Read image as BGR uint8 (handles unicode paths on Windows). Returns None if unreadable."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def image_quality(gray: np.ndarray) -> dict:
    """Simple, interpretable quality indicators."""
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "sharpness_laplacian_var": lap_var,
        "is_blank": bool(gray.std() < 2.0),
        "is_dark": bool(gray.mean() < 20),
        "is_overexposed": bool(gray.mean() > 235),
    }


def analyze_samples(samples: list[dict], near_dup_threshold: int = 2, max_hist: int = 400) -> dict:
    """Compute a full quality report for a list of sample dicts (``path``, ``label``, ``mask_path``)."""
    n = len(samples)
    corrupted, sizes, hashes_exact = [], Counter(), defaultdict(list)
    dhashes: list[tuple[int, str]] = []
    thumbs: dict[str, np.ndarray] = {}
    quality_rows, flagged = [], []
    channels = Counter()
    hist_acc = np.zeros(256, dtype=np.float64)
    hist_n = 0
    missing_masks, mask_mismatch = [], []

    for i, s in enumerate(samples):
        p = absp(s["path"])
        if not p.exists():
            corrupted.append({"path": s["path"], "reason": "missing"})
            continue
        img = read_image(p)
        if img is None or img.size == 0:
            corrupted.append({"path": s["path"], "reason": "undecodable"})
            continue
        h, w = img.shape[:2]
        sizes[f"{w}x{h}"] += 1
        b, g, r = img[..., 0], img[..., 1], img[..., 2]
        channels["grayscale" if (np.array_equal(b, g) and np.array_equal(g, r)) else "color"] += 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hashes_exact[file_sha256(p)].append(s["path"])
        dhashes.append((dhash(gray), s["path"]))
        thumbs[s["path"]] = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
        q = image_quality(gray)
        q["path"] = s["path"]
        quality_rows.append(q)
        if q["is_blank"] or q["is_dark"] or q["is_overexposed"]:
            flagged.append(q)
        if hist_n < max_hist:
            hist_acc += cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten() / gray.size
            hist_n += 1
        if s.get("label") not in (None, "good") and s.get("mask_path") is None and s.get("expects_mask"):
            missing_masks.append(s["path"])
        if s.get("mask_path"):
            m = read_image(absp(s["mask_path"]))
            if m is None:
                missing_masks.append(s["path"])
            elif m.shape[:2] != (h, w):
                mask_mismatch.append(s["path"])

    exact_dups = [v for v in hashes_exact.values() if len(v) > 1]
    # Near-duplicates: dHash is only a *candidate* filter (homogeneous textures collide often);
    # candidates are confirmed by a pixel-level check on 64x64 thumbnails.
    near, candidates = [], 0
    for i in range(len(dhashes)):
        for j in range(i + 1, len(dhashes)):
            if hamming(dhashes[i][0], dhashes[j][0]) <= near_dup_threshold:
                candidates += 1
                a, b = thumbs[dhashes[i][1]], thumbs[dhashes[j][1]]
                if confirm_near_duplicate(a, b):
                    near.append((dhashes[i][1], dhashes[j][1]))

    labels = Counter(s.get("label") for s in samples)
    counts = list(labels.values())
    imbalance = (max(counts) / max(1, min(counts))) if counts else 0.0
    qa = np.array([[r["brightness"], r["contrast"], r["sharpness_laplacian_var"]] for r in quality_rows]) \
        if quality_rows else np.zeros((0, 3))

    def stats(col):
        if not len(qa):
            return {}
        v = qa[:, col]
        return {"mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()), "max": float(v.max())}

    return {
        "num_samples": n,
        "num_readable": n - len(corrupted),
        "num_corrupted": len(corrupted),
        "corrupted": corrupted[:100],
        "class_distribution": dict(sorted(labels.items())),
        "num_classes": len(labels),
        "imbalance_ratio": round(imbalance, 3),
        "imbalanced": imbalance > 1.5,
        "image_sizes": dict(sizes.most_common()),
        "color_modes": dict(channels),
        "exact_duplicate_groups": len(exact_dups),
        "exact_duplicates": exact_dups[:50],
        "near_duplicate_pairs": len(near),
        "near_duplicate_hash_candidates": candidates,
        "near_duplicates": near[:50],
        "quality": {"brightness": stats(0), "contrast": stats(1), "sharpness": stats(2)},
        "quality_flagged": flagged[:100],
        "num_quality_flagged": len(flagged),
        "num_masks": sum(1 for s in samples if s.get("mask_path")),
        "missing_masks": missing_masks[:50],
        "mask_size_mismatch": mask_mismatch[:50],
        "mean_histogram": (hist_acc / max(1, hist_n)).round(6).tolist(),
        "_dhash": {p: h for h, p in dhashes},
        # complete lists for split grouping (the report fields above are truncated for readability)
        "_exact_duplicates_all": exact_dups,
        "_near_duplicates_all": [list(x) for x in near],
    }
