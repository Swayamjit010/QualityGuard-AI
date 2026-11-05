"""Unit tests for the ML building blocks (no trained checkpoints required)."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from ml.datasets.splitting import build_groups, stratified_group_split
from ml.evaluation.metrics import binary_anomaly_report, classification_report
from ml.inference.validation import decode_image, validate_image
from ml.quality.rules import SEVERITY_ORDER, assess_severity, decide
from ml.segmentation.regions import extract_regions
from ml.uncertainty.calibration import fit_temperature, softmax_np

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {p.stem: json.loads(p.read_text()) for p in (ROOT / "configs" / "profiles").glob("*.json")}


# ---------------------------------------------------------------- severity + decision rules
def test_no_defect_is_none_severity_and_pass():
    prof = PROFILES["bottle"]
    sev = assess_severity(prof, False, None, None, None)
    assert sev["level"] == "none"
    d = decide(prof, input_ok=True, input_issues=[], unfamiliar=False, engines_disagree=False, reliability="high",
               defect_present=False, defect_class=None, severity=sev)
    assert d["status"] == "PASS" and d["quality_index"] == 100


@pytest.mark.parametrize("ratio,expected", [(0.001, "low"), (0.01, "medium"), (0.03, "high"), (0.2, "critical")])
def test_area_bands(ratio, expected):
    m = {"num_regions": 1, "defect_area_ratio": ratio, "regions": [{"area_px": 10}], "calibrated": False}
    assert assess_severity(PROFILES["bottle"], True, None, m, 1.1)["level"] == expected


def test_count_and_intensity_escalation():
    m = {"num_regions": 4, "defect_area_ratio": 0.001, "regions": [{"area_px": 10}] * 4, "calibrated": False}
    sev = assess_severity(PROFILES["bottle"], True, None, m, 2.5)
    # low -> +1 (>=3 regions) -> +1 (score >= 2x threshold)
    assert sev["level"] == "high" and len(sev["basis"]) >= 3


def test_class_minimum_severity_for_steel():
    sev = assess_severity(PROFILES["steel_strip"], True, "scratches", None, None)
    assert sev["level"] == "high"
    assert SEVERITY_ORDER.index(sev["level"]) >= SEVERITY_ORDER.index("medium")


def test_defect_fails_at_configured_severity():
    prof = PROFILES["bottle"]  # fail_at_severity = low (zero tolerance)
    sev = {"level": "low", "basis": [], "inputs": {}}
    d = decide(prof, input_ok=True, input_issues=[], unfamiliar=False, engines_disagree=False, reliability="high",
               defect_present=True, defect_class=None, severity=sev)
    assert d["status"] == "FAIL"


def test_unfamiliar_and_disagreement_route_to_human():
    prof = PROFILES["bottle"]
    sev = {"level": "medium", "basis": [], "inputs": {}}
    d = decide(prof, input_ok=True, input_issues=[], unfamiliar=True, engines_disagree=False, reliability="low",
               defect_present=True, defect_class=None, severity=sev)
    assert d["status"] == "INCONCLUSIVE" and d["manual_review_required"]
    d = decide(prof, input_ok=True, input_issues=[], unfamiliar=False, engines_disagree=True, reliability="low",
               defect_present=True, defect_class=None, severity=sev)
    assert d["status"] == "REVIEW REQUIRED" and d["manual_review_required"]


def test_invalid_input_is_inconclusive():
    d = decide(PROFILES["bottle"], input_ok=False, input_issues=["too dark"], unfamiliar=False,
               engines_disagree=False, reliability="low", defect_present=None, defect_class=None,
               severity={"level": "none", "basis": [], "inputs": {}})
    assert d["status"] == "INCONCLUSIVE"


# ---------------------------------------------------------------- input validation
def test_decode_rejects_garbage_and_validates_blank():
    assert decode_image(b"not an image") is None
    ok, issues, _ = validate_image(None, PROFILES["bottle"])
    assert not ok and issues
    blank = np.zeros((300, 300, 3), np.uint8)
    ok, issues, info = validate_image(blank, PROFILES["bottle"])
    assert not ok and info["contrast"] < 1


def test_valid_image_passes_validation():
    rng = np.random.default_rng(0)
    img = (rng.random((256, 256, 3)) * 255).astype(np.uint8)
    buf = cv2.imencode(".png", img)[1].tobytes()
    dec = decode_image(buf)
    assert dec.shape == (256, 256, 3)
    ok, issues, info = validate_image(dec, PROFILES["bottle"])
    assert ok, issues
    assert {"brightness", "contrast", "sharpness"} <= set(info)


# ---------------------------------------------------------------- segmentation
def test_extract_regions_measures_blobs():
    mask = np.zeros((100, 100), bool)
    mask[10:20, 10:20] = True        # 100 px
    mask[60:80, 50:70] = True        # 400 px
    mask[95:96, 95:96] = True        # 1 px (below min area)
    m = extract_regions(mask, mask.astype(float), min_area_px=20, orig_hw=(100, 100))
    assert m["num_regions"] == 2
    assert m["regions"][0]["area_px"] >= m["regions"][1]["area_px"]
    assert m["defect_area_ratio"] == pytest.approx(0.05, abs=0.005)
    assert not m["calibrated"]


# ---------------------------------------------------------------- dataset splitting (leakage safety)
def test_split_keeps_duplicate_groups_together_and_is_stratified():
    samples = [{"path": f"c{c}/{i}.png", "label": f"c{c}"} for c in range(3) for i in range(40)]
    dups = [["c0/0.png", "c0/1.png"], ["c1/5.png", "c1/6.png", "c1/7.png"]]
    groups = build_groups(samples, dups)
    split = stratified_group_split(samples, groups, (0.7, 0.15, 0.15), seed=1)
    by_path = dict(zip((s["path"] for s in samples), split))
    for g in dups:
        assert len({by_path[p] for p in g}) == 1, "duplicates leaked across splits"
    for c in range(3):
        cls = [s for s, sp in zip(samples, split) if s["label"] == f"c{c}"]
        n_test = sum(1 for s in cls if by_path[s["path"]] == "test")
        assert 3 <= n_test <= 10
    assert stratified_group_split(samples, groups, (0.7, 0.15, 0.15), seed=1) == split  # deterministic


# ---------------------------------------------------------------- metrics + calibration
def test_classification_report_perfect():
    y = np.array([0, 1, 2, 1])
    probs = np.eye(3)[y] * 0.9 + 0.1 / 3
    r = classification_report(y, probs, ["a", "b", "c"])
    assert r["accuracy"] == 1.0 and r["f1_macro"] == 1.0


def test_binary_anomaly_report():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    r = binary_anomaly_report(y, s, threshold=0.5)
    assert r["image_auroc"] == 1.0 and r["recall"] == 1.0 and r["false_positive_rate"] == 0.0


def test_temperature_scaling_reduces_overconfidence():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 300)
    logits = rng.normal(0, 1, (300, 3))
    logits[np.arange(300), y] += 1.0
    logits *= 5  # over-confident
    T = fit_temperature(logits, y)
    assert T > 1.0
    p = softmax_np(logits, T)
    assert np.allclose(p.sum(1), 1)
