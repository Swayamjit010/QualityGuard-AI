"""Input-image validation performed before any model runs."""
from __future__ import annotations

import cv2
import numpy as np

MAX_PIXELS = 40_000_000  # ~40 MP decoded


def decode_image(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, dtype=np.uint8)
    if arr.size == 0:
        return None
    try:
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except cv2.error:
        return None
    return img


def normalized_sharpness(gray: np.ndarray, short_side: int = 256) -> float:
    """Variance of the Laplacian after resizing the short side to ``short_side`` px, so values are comparable
    across input resolutions. Low values indicate blur (or a featureless surface)."""
    h, w = gray.shape[:2]
    s = short_side / min(h, w)
    g = cv2.resize(gray, (max(1, round(w * s)), max(1, round(h * s))),
                   interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def validate_image(img: np.ndarray | None, profile: dict | None = None) -> tuple[bool, list[str], dict]:
    if img is None:
        return False, ["File could not be decoded as an image (corrupted or unsupported format)."], {}
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    info = {"width": w, "height": h, "brightness": float(gray.mean()), "contrast": float(gray.std()),
            "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()), "sharpness_norm": normalized_sharpness(gray)}
    req = (profile or {}).get("image", {})
    issues = []
    min_side = req.get("min_side_px", 64)
    if min(h, w) < min_side:
        issues.append(f"Resolution {w}x{h} is below the profile minimum of {min_side}px on the short side.")
    if h * w > MAX_PIXELS:
        issues.append(f"Image too large ({w}x{h}).")
    if gray.std() < req.get("min_contrast", 3.0):
        issues.append("Image is blank or nearly uniform (contrast too low).")
    if gray.mean() < req.get("min_brightness", 8):
        issues.append("Image is too dark.")
    if gray.mean() > req.get("max_brightness", 250):
        issues.append("Image is over-exposed.")
    if req.get("min_sharpness_norm") is not None and info["sharpness_norm"] < req["min_sharpness_norm"]:
        issues.append(f"Image appears blurred: normalised sharpness {info['sharpness_norm']:.1f} is below this "
                      f"category's minimum of {req['min_sharpness_norm']:.1f} (calibrated on its training and validation images).")
    return not issues, issues, info
