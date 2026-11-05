"""Verdict mapping, component recognition, supervised segmentation, blur check and the short /api/inspect API."""
import io
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from ml.inference.validation import normalized_sharpness, validate_image
from ml.quality.rules import decide, verdict_for
from ml.recognition.router import ComponentRecognizer
from ml.segmentation.unet import ResNetUNet, image_score, seg_loss
from tests.conftest import ROOT, sample

RECOGNIZER = ROOT / "artifacts/checkpoints/component_recognizer-resnet18-v1/recognizer.pt"
SEGMENTER = ROOT / "artifacts/checkpoints/ksdd2-segmentation-unet_resnet18-v1/segmentation.pt"
PROFILES = {p.stem: json.loads(p.read_text()) for p in (ROOT / "configs" / "profiles").glob("*.json")}


# ---------------------------------------------------------------- verdict
@pytest.mark.parametrize("status,defect,expected", [
    ("PASS", False, "NORMAL"), ("FAIL", True, "DEFECTIVE"), ("PASS", True, "DEFECTIVE"),
    ("REVIEW REQUIRED", True, "MANUAL INSPECTION REQUIRED"), ("INCONCLUSIVE", None, "MANUAL INSPECTION REQUIRED"),
])
def test_verdict_mapping(status, defect, expected):
    assert verdict_for(status, defect) == expected


def test_decide_adds_verdict_and_never_forces_binary_on_unfamiliar():
    sev = {"level": "none", "basis": [], "inputs": {}}
    d = decide(PROFILES["bottle"], input_ok=True, input_issues=[], unfamiliar=True, engines_disagree=False,
               reliability="high", defect_present=False, defect_class=None, severity=sev)
    assert d["verdict"] == "MANUAL INSPECTION REQUIRED" and d["verdict_text"]
    d = decide(PROFILES["bottle"], input_ok=False, input_issues=["blurred"], unfamiliar=False, engines_disagree=False,
               reliability="low", defect_present=None, defect_class=None, severity=sev)
    assert d["verdict"] == "MANUAL INSPECTION REQUIRED"


def test_tolerated_defect_is_still_reported_as_defective():
    prof = PROFILES["metal_nut"]  # fail_at_severity = medium -> low severity is accepted by the rules
    d = decide(prof, input_ok=True, input_issues=[], unfamiliar=False, engines_disagree=False, reliability="high",
               defect_present=True, defect_class=None, severity={"level": "low", "basis": [], "inputs": {}})
    assert d["status"] == "PASS" and d["verdict"] == "DEFECTIVE" and "accept" in d["verdict_text"]


# ---------------------------------------------------------------- recogniser (synthetic embeddings)
def _clusters(rng, n=60, dim=32):
    centers = np.eye(dim, dtype=np.float32)[:3] * 4
    X, y = [], []
    for i, c in enumerate(centers):
        e = c + rng.normal(0, 0.3, (n, dim)).astype(np.float32)
        X.append(e / np.linalg.norm(e, axis=1, keepdims=True)); y += [f"d{i}"] * n
    return np.concatenate(X), y


def test_recognizer_accepts_in_domain_and_rejects_far_inputs():
    rng = np.random.default_rng(0)
    X, y = _clusters(rng)
    rec = ComponentRecognizer(k=3, min_probability=0.8)
    rec.fit(X, y)
    Xc, yc = _clusters(np.random.default_rng(1))
    rec.calibrate(Xc, yc)
    Xt, yt = _clusters(np.random.default_rng(2), n=20)
    out = rec.decide_batch(Xt)
    assert np.mean([o["domain"] == t for o, t in zip(out, yt)]) == 1.0
    assert np.mean([o["supported"] for o in out]) >= 0.9
    far = np.zeros((5, 32), np.float32); far[:, 20] = 1.0           # orthogonal to every training domain
    assert not any(o["supported"] for o in rec.decide_batch(far))
    # persistence round-trip gives identical decisions
    rec2 = ComponentRecognizer.load(rec.state())
    assert [o["domain"] for o in rec2.decide_batch(Xt)] == [o["domain"] for o in out]


# ---------------------------------------------------------------- segmentation model
def test_unet_shapes_and_loss():
    m = ResNetUNet(pretrained=False).eval()
    x = torch.randn(2, 3, 384, 128)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (2, 1, 384, 128)
    t = torch.zeros_like(y); t[0, 0, 100:120, 40:60] = 1
    assert torch.isfinite(seg_loss(y, t))
    assert image_score(torch.sigmoid(y)).shape == (2,)


# ---------------------------------------------------------------- blur check
def test_blur_detection_uses_profile_limit():
    rng = np.random.default_rng(0)
    img = (rng.integers(0, 255, (300, 300)) > 127).astype(np.uint8) * 255
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    blurred = cv2.GaussianBlur(img, (31, 31), 8)
    sharp_v = normalized_sharpness(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    prof = {"image": {"min_sharpness_norm": sharp_v * 0.2}}
    assert validate_image(img, prof)[0]
    ok, issues, _ = validate_image(blurred, prof)
    assert not ok and any("blurred" in i for i in issues)


# ---------------------------------------------------------------- API (real trained models)
need_recognizer = pytest.mark.skipif(not RECOGNIZER.exists(), reason="component recogniser not trained")


def _post(client, headers, path, img_bytes, name, **data):
    return client.post(path, headers=headers, data=data, files={"file": (name, img_bytes, "image/png")})


@need_recognizer
def test_inspect_auto_recognises_and_flags_defect(client, admin):
    p = sample("mvtec_bottle", "broken_large")
    r = _post(client, admin, "/api/inspect", p.read_bytes(), p.name)
    assert r.status_code == 201, r.text
    d = r.json()
    assert d["profile_id"] == "bottle" and d["result"]["recognition"]["domain"] == "bottle"
    assert d["result"]["recognition"]["supported"] is True and d["result"]["recognition"]["mode"] == "auto"
    assert d["verdict"] in ("DEFECTIVE", "MANUAL INSPECTION REQUIRED")
    assert d["verdict"] == d["result"]["decision"]["verdict"]


@need_recognizer
def test_inspect_auto_normal_screw_or_metal_nut(client, admin):
    p = sample("mvtec_metal_nut", "good")
    d = _post(client, admin, "/api/inspect", p.read_bytes(), p.name).json()
    assert d["profile_id"] == "metal_nut"
    assert d["verdict"] in ("NORMAL", "MANUAL INSPECTION REQUIRED")


@need_recognizer
def test_unsupported_image_goes_to_manual_inspection(client, admin):
    probes = sorted((ROOT / "data/raw/foreign_probe/imagenette").glob("*.JPEG"))
    if not probes:
        pytest.skip("foreign probe images not downloaded")
    d = _post(client, admin, "/api/inspect", probes[0].read_bytes(), "photo.jpg").json()
    assert d["verdict"] == "MANUAL INSPECTION REQUIRED"
    assert d["result"]["recognition"]["supported"] is False
    assert any("UNSUPPORTED" in x for x in d["result"]["decision"]["reasons"])


@need_recognizer
def test_category_mismatch_is_not_silently_inspected(client, admin):
    p = sample("mvtec_bottle", "good")
    d = _post(client, admin, "/api/inspect", p.read_bytes(), p.name, profile_id="metal_nut").json()
    assert d["profile_id"] == "metal_nut"
    assert d["verdict"] == "MANUAL INSPECTION REQUIRED"
    assert "MISMATCH" in d["result"]["recognition"]["block"]


@need_recognizer
def test_corrupted_upload_in_auto_mode(client, admin):
    r = _post(client, admin, "/api/inspect", b"\x89PNG\r\n\x1a\n not really an image", "broken.png")
    assert r.status_code == 201, r.text
    assert r.json()["verdict"] == "MANUAL INSPECTION REQUIRED"


def test_blurred_upload_is_manual(client, admin):
    p = sample("mvtec_metal_nut", "good")
    img = cv2.imdecode(np.frombuffer(p.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if PROFILES["metal_nut"].get("image", {}).get("min_sharpness_norm") is None:
        pytest.skip("blur limit not calibrated (scripts/calibrate_image_quality.py)")
    blurred = cv2.GaussianBlur(img, (0, 0), 12)
    ok, buf = cv2.imencode(".png", blurred)
    d = _post(client, admin, "/api/inspections", buf.tobytes(), "blurred.png", profile_id="metal_nut").json()
    assert d["verdict"] == "MANUAL INSPECTION REQUIRED"
    assert any("blurred" in i for i in d["result"]["input_issues"])


def test_low_resolution_upload_is_manual(client, admin):
    p = sample("mvtec_metal_nut", "good")
    img = cv2.imdecode(np.frombuffer(p.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    ok, buf = cv2.imencode(".png", cv2.resize(img, (48, 48)))
    d = _post(client, admin, "/api/inspections", buf.tobytes(), "tiny.png", profile_id="metal_nut").json()
    assert d["verdict"] == "MANUAL INSPECTION REQUIRED" and not d["result"]["input_valid"]


@pytest.mark.skipif(not SEGMENTER.exists(), reason="KSDD2 segmentation model not trained")
def test_commutator_profile_runs_segmentation(client, admin):
    p = sample("ksdd2", "defect")
    d = _post(client, admin, "/api/inspections", p.read_bytes(), p.name, profile_id="commutator_surface").json()
    seg = d["result"]["engines"]["segmentation"]
    assert seg["image_threshold"] > 0 and "segmentation_unet_overlay" in d["visualizations"]
    assert d["verdict"] in ("DEFECTIVE", "MANUAL INSPECTION REQUIRED")


def test_short_api_endpoints(client, admin):
    s = client.get("/api/dashboard/summary", headers=admin)
    assert s.status_code == 200
    j = s.json()
    assert j["total_inspections"] == j["normal"] + j["defective"] + j["manual_inspection_required"]
    for v in ("NORMAL", "DEFECTIVE", "MANUAL INSPECTION REQUIRED"):
        r = client.get("/api/inspections", headers=admin, params={"verdict": v})
        assert r.status_code == 200 and all(i["verdict"] == v for i in r.json()["items"])
    assert client.get("/api/inspections", headers=admin, params={"verdict": "MAYBE"}).status_code == 422
    assert client.get("/api/recognizer", headers=admin).status_code == 200
    h = client.get("/api/health")
    assert h.status_code == 200


def test_inspect_batch_short_endpoint(client, admin):
    p = sample("mvtec_bottle", "good")
    r = client.post("/api/inspect/batch", headers=admin, data={"profile_id": "bottle"},
                    files=[("files", (p.name, p.read_bytes(), "image/png")), ("files", ("notes.txt", b"x", "text/plain"))])
    assert r.status_code == 202, r.text
    assert r.json()["accepted"] == 1 and r.json()["skipped_unsupported"] == ["notes.txt"]


@need_recognizer
@pytest.mark.parametrize("cat", ["mpdd_metal_plate", "mpdd_connector"])
def test_mpdd_parts_are_auto_routed(client, admin, cat):
    if not (ROOT / f"artifacts/checkpoints/{cat}-anomaly-resnet18-v1/anomaly.pt").exists():
        pytest.skip(f"{cat} model not trained")
    p = sample(cat, "good")
    d = _post(client, admin, "/api/inspect", p.read_bytes(), p.name).json()
    assert d["profile_id"] == cat and d["result"]["recognition"]["supported"] is True
    assert d["verdict"] in ("NORMAL", "MANUAL INSPECTION REQUIRED")
