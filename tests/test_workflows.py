"""Background-job workflows with real models: batch and video inspection, golden-reference comparison,
drift monitoring and the batch PDF report."""
import time

import cv2
import numpy as np
import pytest

from conftest import HAS_BOTTLE, sample

pytestmark = pytest.mark.skipif(not HAS_BOTTLE, reason="bottle anomaly model not trained")


def _wait_job(client, headers, job_id, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{job_id}", headers=headers).json()
        if j["status"] in ("completed", "failed", "stopped"):
            return j
        time.sleep(1)
    pytest.fail(f"job {job_id} did not finish in {timeout}s")


def test_batch_inspection_report_and_export(client, admin):
    files = [("files", (p.name, p.read_bytes(), "image/png")) for p in
             (sample("mvtec_bottle", "good"), sample("mvtec_bottle", "broken_large"))]
    files.append(("files", ("notes.txt", b"not an image", "text/plain")))
    r = client.post("/api/inspections/batch", headers=admin, data={"profile_id": "bottle", "name": "test batch"},
                    files=files)
    assert r.status_code == 202, r.text
    b = r.json()
    assert b["accepted"] == 2 and b["skipped_unsupported"] == ["notes.txt"]
    job = _wait_job(client, admin, b["job_id"])
    assert job["status"] == "completed", job
    batch = client.get(f"/api/batches/{b['batch_id']}", headers=admin).json()
    assert len(batch["items"]) == 2
    statuses = {i["filename"].split(".")[0]: i["status"] for i in batch["items"]}
    assert all(s in {"PASS", "FAIL", "REVIEW REQUIRED", "INCONCLUSIVE"} for s in statuses.values())
    csv_text = client.get("/api/inspections/export", headers=admin,
                          params={"fmt": "csv", "batch_id": b["batch_id"]}).text
    assert len(csv_text.strip().splitlines()) == 3  # header + 2 rows
    rep = client.post("/api/reports/generate", headers=admin, json={"kind": "batch", "batch_id": b["batch_id"]})
    assert rep.status_code == 201, rep.text
    pdf = client.get(f"/api/reports/{rep.json()['id']}/download", headers=admin)
    assert pdf.content[:5] == b"%PDF-"


def test_video_inspection_annotated_output(client, admin, tmp_path):
    good, bad = (cv2.imread(str(sample("mvtec_bottle", lbl))) for lbl in ("good", "broken_large"))
    good, bad = (cv2.resize(x, (448, 448)) for x in (good, bad))
    path = tmp_path / "line.avi"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 2.0, (448, 448))
    for frame in [good, good, bad, bad, good, good]:  # 3 s at 2 fps
        vw.write(frame)
    vw.release()
    r = client.post("/api/inspections/video", headers=admin, data={"profile_id": "bottle", "every_seconds": "1",
                                                                   "max_frames": "3"},
                    files={"file": ("line.avi", path.read_bytes(), "video/x-msvideo")})
    assert r.status_code == 202, r.text
    job = _wait_job(client, admin, r.json()["job_id"])
    assert job["status"] == "completed", job
    batch = client.get(f"/api/batches/{r.json()['batch_id']}", headers=admin).json()
    ts = sorted(i["video_timestamp_s"] for i in batch["items"])
    assert len(ts) == 3 and ts == pytest.approx([0.0, 1.0, 2.0], abs=0.6)
    vid = client.get(f"/api/batches/{r.json()['batch_id']}/video", headers=admin)
    assert vid.status_code == 200 and len(vid.content) > 1000  # annotated WebM written by OpenCV


def test_video_rejects_bad_parameters(client, admin):
    r = client.post("/api/inspections/video", headers=admin, data={"profile_id": "bottle", "every_seconds": "0"},
                    files={"file": ("x.avi", b"0" * 10, "video/x-msvideo")})
    assert r.status_code == 422


def test_golden_reference_comparison(client, admin):
    good, good2, bad = (sample("mvtec_bottle", "good", "test"), sample("mvtec_bottle", "good", "train"),
                        sample("mvtec_bottle", "broken_large"))

    def cmp(a, b):
        r = client.post("/api/inspections/compare-reference", headers=admin,
                        files={"reference": (a.name, a.read_bytes(), "image/png"),
                               "test": (b.name, b.read_bytes(), "image/png")})
        assert r.status_code == 200, r.text
        return r.json()

    same = cmp(good, good)
    assert same["mean_difference"] < 1e-3 and same["measurements"]["num_regions"] == 0
    diff = cmp(good2, bad)
    assert diff["measurements"]["num_regions"] >= 1
    assert set(diff["images"]) >= {"reference", "aligned_test", "difference_heatmap"}


def test_drift_check_runs_and_is_listed(client, admin):
    r = client.post("/api/drift/check", headers=admin, params={"profile_id": "bottle", "window": 20})
    assert r.status_code == 200, r.text
    c = r.json()
    assert c["profile_id"] == "bottle" and "details" in c
    assert any(x["id"] == c["id"] for x in client.get("/api/drift/checks", headers=admin).json())
    rep = client.post("/api/reports/generate", headers=admin, json={"kind": "drift", "drift_check_id": c["id"]})
    assert rep.status_code == 201, rep.text
