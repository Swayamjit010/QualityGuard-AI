"""Reference-based inspection: register a test image onto an approved reference image and highlight differences.

Method: ORB keypoints + RANSAC homography (falls back to identity resize when too few matches), then
Gaussian-smoothed absolute difference in LAB space, Otsu threshold, morphological clean-up and region extraction.
This is a classical CV comparison; it detects *any* visual difference (including benign ones such as lighting
changes), so results are advisory.
"""
from __future__ import annotations

import cv2
import numpy as np

from ml.segmentation.regions import draw_regions, extract_regions


def register(ref: np.ndarray, test: np.ndarray, max_side: int = 768):
    s = min(1.0, max_side / max(ref.shape[:2]))
    ref = cv2.resize(ref, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else ref
    test = cv2.resize(test, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
    g1, g2 = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(3000)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    info = {"keypoints_ref": len(k1), "keypoints_test": len(k2), "method": "identity"}
    if d1 is not None and d2 is not None and len(k1) > 20 and len(k2) > 20:
        m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d2, d1)
        m = sorted(m, key=lambda x: x.distance)[:500]
        info["matches"] = len(m)
        if len(m) >= 15:
            src = np.float32([k2[x.queryIdx].pt for x in m]).reshape(-1, 1, 2)
            dst = np.float32([k1[x.trainIdx].pt for x in m]).reshape(-1, 1, 2)
            H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
            if H is not None and inl is not None and inl.sum() >= 12:
                test = cv2.warpPerspective(test, H, (ref.shape[1], ref.shape[0]), borderMode=cv2.BORDER_REPLICATE)
                info.update(method="orb_homography", inliers=int(inl.sum()))
    return ref, test, info


def compare(ref_bgr: np.ndarray, test_bgr: np.ndarray, min_area_px: int = 40) -> dict:
    ref, test, info = register(ref_bgr, test_bgr)
    a = cv2.GaussianBlur(cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32), (5, 5), 0)
    b = cv2.GaussianBlur(cv2.cvtColor(test, cv2.COLOR_BGR2LAB).astype(np.float32), (5, 5), 0)
    diff = np.linalg.norm(a - b, axis=2)
    d8 = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    t, mask = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = max(t, 40)
    mask = (d8 >= t).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    meas = extract_regions(mask.astype(bool), diff, min_area_px)
    test_rgb = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
    heat = cv2.applyColorMap(d8, cv2.COLORMAP_JET)[..., ::-1]
    return {"registration": info, "difference_threshold": float(t), "mean_difference": float(diff.mean()),
            "measurements": meas,
            "_images": {"reference": cv2.cvtColor(ref, cv2.COLOR_BGR2RGB), "aligned_test": test_rgb,
                        "difference_heatmap": heat, "difference_regions": draw_regions(test_rgb, meas, (255, 60, 60))}}
