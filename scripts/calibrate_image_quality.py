"""Calibrate the per-profile blur limit (image.min_sharpness_norm) from each profile's TRAIN + VALIDATION images.

    python scripts/calibrate_image_quality.py

limit = 0.5 x the 1st percentile of normalised sharpness over the train + val splits (all labels: some defect types,
e.g. a flipped metal nut, are legitimately smoother than defect-free training images). The held-out test split is then
used only to report how many genuine test images the limit would wrongly reject, and Gaussian-blurred copies of
those test images are used to report how many blurred images it catches. Results -> artifacts/metrics/image_quality_calibration.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ml.config import CONFIG_DIR, METRICS_DIR, absp  # noqa: E402
from ml.datasets.analysis import read_image  # noqa: E402
from ml.datasets.splitting import load_manifest  # noqa: E402
from ml.inference.validation import normalized_sharpness  # noqa: E402


def sharp(path, blur_sigma=None):
    img = read_image(absp(path))
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if blur_sigma:
        k = int(2 * round(3 * blur_sigma * min(g.shape) / 256) + 1)
        g = cv2.GaussianBlur(g, (k, k), blur_sigma * min(g.shape) / 256)
    return normalized_sharpness(g)


def main():
    out = {}
    rng = np.random.default_rng(0)
    for f in sorted((CONFIG_DIR / "profiles").glob("*.json")):
        prof = json.loads(f.read_text())
        try:
            man = load_manifest(prof["dataset"])
        except (FileNotFoundError, KeyError):
            continue
        tr = [r for r in man["samples"] if r["split"] in ("train", "val")]
        te = [r for r in man["samples"] if r["split"] == "test"]
        tr = [tr[i] for i in rng.choice(len(tr), min(400, len(tr)), replace=False)]
        te = [te[i] for i in rng.choice(len(te), min(150, len(te)), replace=False)]
        s_tr = np.array([sharp(r["path"]) for r in tr])
        limit = float(0.5 * np.quantile(s_tr, 0.01))
        s_te = np.array([sharp(r["path"]) for r in te])
        s_bl = {sig: np.array([sharp(r["path"], sig) for r in te]) for sig in (2.0, 4.0)}
        out[prof["id"]] = {"calibration_images (train+val)": len(tr), "train_p01": float(np.quantile(s_tr, 0.01)),
                           "train_median": float(np.median(s_tr)), "min_sharpness_norm": limit,
                           "test_images": len(te), "test_false_reject_rate": float((s_te < limit).mean()),
                           **{f"blur_sigma{int(k)}_detect_rate": float((v < limit).mean()) for k, v in s_bl.items()}}
        prof.setdefault("image", {})["min_sharpness_norm"] = round(limit, 2)
        f.write_text(json.dumps(prof, indent=2))
        print(prof["id"], json.dumps(out[prof["id"]]))
    (METRICS_DIR / "image_quality_calibration.json").write_text(json.dumps({
        "rule": "min_sharpness_norm = 0.5 x 1st percentile of train+val normalised sharpness (Laplacian variance "
                "at 256 px short side). Blur detection rates are measured on Gaussian-blurred copies of held-out test "
                "images (sigma relative to a 256 px short side).", "profiles": out}, indent=2))


if __name__ == "__main__":
    main()
