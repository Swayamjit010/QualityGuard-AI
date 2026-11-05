"""Defect-region extraction and measurements from a binary mask (from an anomaly map threshold or a
segmentation model).

Pixel measurements are always reported in pixels of the ORIGINAL image. Physical units (mm) are only
reported when a calibration (``mm_per_pixel``) is supplied by the product profile; otherwise the
``*_mm`` fields are None.
"""
from __future__ import annotations

import cv2
import numpy as np


def extract_regions(mask: np.ndarray, score_map: np.ndarray | None = None, min_area_px: int = 20,
                    orig_hw: tuple[int, int] | None = None, mm_per_pixel: float | None = None) -> dict:
    """mask: HxW bool at model resolution. Returns measurements scaled to the original image size."""
    h, w = mask.shape
    oh, ow = orig_hw or (h, w)
    sy, sx = oh / h, ow / w
    area_scale = sy * sx
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    regions = []
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a * area_scale < min_area_px:
            continue
        comp = (lab == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(c)
        perim = float(cv2.arcLength(c, True)) * (sx + sy) / 2
        carea = float(cv2.contourArea(c)) * area_scale
        (_, _), (rw, rh), ang = cv2.minAreaRect(c)
        hull = cv2.convexHull(c)
        hull_area = float(cv2.contourArea(hull)) * area_scale
        long_side, short_side = max(rw, rh), max(min(rw, rh), 1e-6)
        r = {
            "id": len(regions) + 1,
            "area_px": float(a * area_scale),
            "contour_area_px": carea,
            "perimeter_px": perim,
            "bbox_px": [int(x * sx), int(y * sy), int(bw * sx), int(bh * sy)],
            "centroid_px": [float(cent[i][0] * sx), float(cent[i][1] * sy)],
            "aspect_ratio": float(long_side / short_side),
            "orientation_deg": float(ang),
            "solidity": float(carea / hull_area) if hull_area > 0 else None,
            "circularity": float(4 * np.pi * carea / perim ** 2) if perim > 0 else None,
            "elongated": bool(long_side / short_side > 3),
        }
        if score_map is not None:
            r["max_score"] = float(score_map[lab == i].max())
            r["mean_score"] = float(score_map[lab == i].mean())
        if mm_per_pixel:
            r["area_mm2"] = r["area_px"] * mm_per_pixel ** 2
            r["bbox_mm"] = [v * mm_per_pixel for v in r["bbox_px"][2:]]
            r["perimeter_mm"] = perim * mm_per_pixel
        regions.append(r)
    total = float(sum(r["area_px"] for r in regions))
    img_area = float(oh * ow)
    cy = [r["centroid_px"][1] / oh for r in regions]
    cx = [r["centroid_px"][0] / ow for r in regions]
    return {
        "num_regions": len(regions),
        "regions": sorted(regions, key=lambda r: -r["area_px"]),
        "total_defect_area_px": total,
        "defect_area_ratio": total / img_area if img_area else 0.0,
        "largest_region_px": max((r["area_px"] for r in regions), default=0.0),
        "defect_density_per_mpx": len(regions) / (img_area / 1e6) if img_area else 0.0,
        "spatial_spread": float(np.hypot(np.std(cx), np.std(cy))) if len(regions) > 1 else 0.0,
        "calibrated": bool(mm_per_pixel),
        "mm_per_pixel": mm_per_pixel,
        "total_defect_area_mm2": total * mm_per_pixel ** 2 if mm_per_pixel else None,
        "image_size_px": [ow, oh],
    }


def draw_regions(rgb: np.ndarray, measurements: dict, color=(0, 255, 120)) -> np.ndarray:
    out = rgb.copy()
    t = max(2, int(round(max(out.shape[:2]) / 300)))
    for r in measurements["regions"]:
        x, y, w, h = r["bbox_px"]
        cv2.rectangle(out, (x, y), (x + w, y + h), color, t)
        cv2.putText(out, f"#{r['id']}", (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * t, color, max(1, t // 2))
    return out
