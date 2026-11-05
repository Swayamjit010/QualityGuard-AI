"""End-to-end API tests: auth/RBAC, real inspection through the ML pipeline, review, alerts, reports, analytics."""
import json
from pathlib import Path

import pytest

from conftest import HAS_BOTTLE, ROOT, sample


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["database"] is True


def test_requires_auth(client):
    assert client.get("/api/inspections").status_code == 401
    assert client.get("/api/inspections", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_login_bad_password(client):
    r = client.post("/api/auth/login", json={"email": "admin@qualityguard.local", "password": "wrong"})
    assert r.status_code == 401


def test_me_and_roles(client, admin):
    me = client.get("/api/auth/me", headers=admin).json()
    assert me["role"] == "admin" and "manage_users" in me["permissions"]
    roles = {r["name"] for r in client.get("/api/roles", headers=admin).json()}
    assert {"admin", "inspector", "viewer", "ml_engineer", "quality_manager"} <= roles


def test_register_creates_viewer_and_weak_password_rejected(client):
    assert client.post("/api/auth/register", json={"email": "w@x.io", "password": "short"}).status_code == 422
    r = client.post("/api/auth/register", json={"email": "new@x.io", "username": "newuser", "password": "GoodPass123",
                                                     "full_name": "N"})
    assert r.status_code == 201 and r.json()["role"] == "viewer"


def test_viewer_cannot_inspect_or_manage(client, make_user):
    viewer = make_user("viewer")
    assert client.get("/api/profiles", headers=viewer).status_code == 200
    assert client.get("/api/users", headers=viewer).status_code == 403
    r = client.post("/api/inspections", headers=viewer, data={"profile_id": "bottle"},
                    files={"file": ("a.png", b"x", "image/png")})
    assert r.status_code == 403


def test_logout_revokes_token(client):
    from conftest import ADMIN, login
    h = login(client, *ADMIN)
    assert client.post("/api/auth/logout", headers=h).status_code == 200
    assert client.get("/api/auth/me", headers=h).status_code == 401


def test_profiles_and_models_registered(client, admin):
    ids = {p["id"] for p in client.get("/api/profiles", headers=admin).json()}
    assert {"bottle", "metal_nut", "steel_strip"} <= ids
    models = client.get("/api/models", headers=admin).json()
    if HAS_BOTTLE:
        assert any(m["run_id"] == "mvtec_bottle-anomaly-resnet18-v1" for m in models)
    archs = {a["name"] for a in client.get("/api/models/architectures", headers=admin).json()}
    assert {"mobilenet_v2", "custom_cnn"} <= archs


def test_upload_validation(client, admin):
    r = client.post("/api/inspections", headers=admin, data={"profile_id": "bottle"},
                    files={"file": ("a.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 415
    r = client.post("/api/inspections", headers=admin, data={"profile_id": "nope"},
                    files={"file": ("a.png", b"x", "image/png")})
    assert r.status_code == 404
    r = client.post("/api/inspections", headers=admin, data={"profile_id": "bottle", "options": "{bad"},
                    files={"file": ("a.png", b"x", "image/png")})
    assert r.status_code == 422


def test_corrupt_image_is_inconclusive_not_crash(client, admin):
    if not HAS_BOTTLE:
        pytest.skip("bottle model not trained")
    r = client.post("/api/inspections", headers=admin, data={"profile_id": "bottle"},
                    files={"file": ("broken.png", b"not really a png", "image/png")})
    assert r.status_code == 201, r.text
    assert r.json()["status"] == "INCONCLUSIVE"


def test_inspection_detects_defect(client, admin, defect_inspection):
    d = defect_inspection
    assert d["defect_present"] is True
    assert d["status"] in ("FAIL", "REVIEW REQUIRED")
    assert d["traceability"]["models"] == ["mvtec_bottle-anomaly-resnet18-v1"]
    assert d["visualizations"], "expected heatmap/segmentation images"
    key = next(iter(d["visualizations"]))
    img = client.get(f"/api/inspections/{d['id']}/images/{key}", headers=admin)
    assert img.status_code == 200 and img.content[:4] == b"\x89PNG"
    assert client.get(f"/api/inspections/{d['id']}/images/../../x", headers=admin).status_code == 404


def test_good_bottle_passes_or_reviews(client, admin):
    if not HAS_BOTTLE:
        pytest.skip("bottle model not trained")
    p = sample("mvtec_bottle", "good")
    r = client.post("/api/inspections", headers=admin, data={"profile_id": "bottle"},
                    files={"file": (p.name, p.read_bytes(), "image/png")})
    assert r.status_code == 201
    assert r.json()["status"] in ("PASS", "REVIEW REQUIRED")


def test_list_filter_and_export(client, admin, defect_inspection):
    r = client.get("/api/inspections", headers=admin, params={"profile_id": "bottle", "q": "SN-1"}).json()
    assert r["total"] >= 1 and r["items"][0]["product_serial"] == "SN-1"
    csv = client.get("/api/inspections/export", headers=admin, params={"fmt": "csv"})
    assert csv.status_code == 200 and b"profile_id" in csv.content.splitlines()[0]


def test_feedback_review_flow(client, admin, make_user, defect_inspection):
    inspector = make_user("inspector")
    iid = defect_inspection["id"]
    bad = client.post(f"/api/inspections/{iid}/feedback", headers=inspector,
                      json={"verdict": "confirm", "final_decision": "reject", "corrected_class": "not_a_class"})
    assert bad.status_code == 422
    r = client.post(f"/api/inspections/{iid}/feedback", headers=inspector,
                    json={"verdict": "confirm", "final_decision": "reject", "corrected_severity": "high",
                          "comment": "crack visible"})
    assert r.status_code == 201, r.text
    d = client.get(f"/api/inspections/{iid}", headers=admin).json()
    assert d["review_status"] == "reviewed" and d["human_decision"] == "reject"
    assert d["feedback"][0]["comment"] == "crack visible"


def test_inspection_pdf_report(client, admin, defect_inspection):
    r = client.get(f"/api/inspections/{defect_inspection['id']}/report", headers=admin)
    assert r.status_code == 200 and r.content[:4] == b"%PDF"
    reps = client.get("/api/reports", headers=admin).json()
    assert any(x["inspection_id"] == defect_inspection["id"] for x in reps)
    dl = client.get(reps[0]["download_url"], headers=admin)
    assert dl.status_code == 200 and dl.content[:4] == b"%PDF"


def test_period_and_review_reports(client, admin, defect_inspection):
    for body in ({"kind": "period", "period": "daily"}, {"kind": "review", "days": 7},
                 {"kind": "dataset", "dataset": "mvtec_bottle"}):
        r = client.post("/api/reports/generate", headers=admin, json=body)
        assert r.status_code == 201, (body, r.text)


def test_analytics(client, admin, defect_inspection):
    d = client.get("/api/analytics/dashboard", headers=admin).json()
    assert d["total"] >= 1 and set(d["by_status"]) == {"PASS", "FAIL", "REVIEW REQUIRED", "INCONCLUSIVE"}
    assert isinstance(d["insights"], list)
    assert client.get("/api/analytics/trend?unit=hour", headers=admin).status_code == 200
    assert client.get("/api/analytics/defects", headers=admin).status_code == 200
    assert client.get("/api/analytics/quality", headers=admin).status_code == 200
    fc = client.get("/api/analytics/forecast", headers=admin).json()
    assert fc["available"] is False  # not enough history in a fresh DB


def test_alerts_lifecycle(client, admin, defect_inspection):
    alerts = client.get("/api/alerts", headers=admin).json()
    if not alerts:
        pytest.skip("no alerts raised for this sample")
    a = alerts[0]
    r = client.post(f"/api/alerts/{a['id']}/acknowledge", headers=admin)
    assert r.status_code == 200 and r.json()["status"] in ("acknowledged", "resolved")
    assert client.post(f"/api/alerts/{a['id']}/resolve", headers=admin).json()["status"] == "resolved"


def test_similar_and_assistant(client, admin, defect_inspection):
    r = client.get(f"/api/inspections/{defect_inspection['id']}/similar", headers=admin)
    assert r.status_code == 200 and "results" in r.json()
    a = client.post("/api/assistant/chat", headers=admin, json={"question": "How many inspections failed?"})
    assert a.status_code == 200 and a.json()["answer"]


def test_datasets_and_samples(client, admin):
    names = {d["name"] for d in client.get("/api/datasets", headers=admin).json()}
    assert "mvtec_bottle" in names
    s = client.get("/api/datasets/mvtec_bottle/samples", headers=admin, params={"split": "test"}).json()
    assert s["total"] == 42 and "good" in s["labels"]
    img = client.get(s["items"][0]["url"], headers=admin)
    downloaded = (ROOT / "data" / "raw").exists() and img.status_code != 404
    assert img.status_code == (200 if downloaded else 404)  # a missing file is a clean 404, never a 500
    # arbitrary paths outside the manifest are refused
    assert client.get("/api/datasets/mvtec_bottle/image", headers=admin,
                      params={"path": "storage/.secret_key"}).status_code == 404


def test_profile_update_versioned(client, admin):
    p = client.get("/api/profiles/bottle", headers=admin).json()
    cfg = p["config"]
    cfg["description"] = "edited in test"
    r = client.put("/api/profiles/bottle", headers=admin, json={"config": cfg, "reason": "test"})
    assert r.status_code == 200 and r.json()["version"] == p["version"] + 1
    bad = dict(cfg, acceptance_rules={**cfg["acceptance_rules"], "fail_at_severity": "bogus"})
    assert client.put("/api/profiles/bottle", headers=admin, json={"config": bad}).status_code == 422
    revs = client.get("/api/profiles/bottle/revisions", headers=admin).json()
    assert revs[0]["reason"] == "test"


def test_settings_and_audit(client, admin):
    s = client.get("/api/settings", headers=admin).json()
    assert "alert_rules" in s
    r = client.put("/api/settings/alert_rules", headers=admin,
                   json={"repeated_defect": {"enabled": True, "count": 3, "window_minutes": 30}})
    assert r.status_code == 200 and r.json()["repeated_defect"]["count"] == 3
    assert client.put("/api/settings/secret", headers=admin, json={}).status_code == 404
    audit = client.get("/api/system/audit", headers=admin).json()
    assert any(e["action"] == "settings.update" for e in audit["items"])


def test_api_key_auth(client, make_user):
    eng = make_user("ml_engineer")
    k = client.post("/api/api-keys", headers=eng, json={"name": "line-1"}).json()["key"]
    r = client.get("/api/profiles", headers={"X-API-Key": k})
    assert r.status_code == 200
    assert client.get("/api/profiles", headers={"X-API-Key": "qg_wrong"}).status_code == 401


def test_websocket_requires_token(client, admin):
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/api/ws/events?token=bad") as ws:
            ws.receive_text()
    token = admin["Authorization"].split()[1]
    with client.websocket_connect(f"/api/ws/events?token={token}") as ws:
        assert json.loads(ws.receive_text())["type"] == "hello"


def test_anomaly_overview_and_system_status(client, admin, make_user):
    r = client.get("/api/analytics/anomaly-overview", headers=admin, params={"profile_id": "bottle"})
    assert r.status_code == 200, r.text
    o = r.json()
    assert o["model"]["engines"]["patchcore"]["deployed_threshold"] > 0  # read from the training record
    assert len(o["score_ratio_histogram"]) == 31 and o["inspections"] >= 0
    assert client.get("/api/analytics/anomaly-overview", headers=admin,
                      params={"profile_id": "nope"}).status_code == 404
    s = client.get("/api/system/status", headers=admin).json()
    assert {m["profile_id"] for m in s["models"]} >= {"bottle", "steel_strip"}
    assert client.get("/api/system/status", headers=make_user("viewer")).status_code == 403


def test_review_priority_scoring():
    from types import SimpleNamespace

    from app.api.operations import review_priority
    base = dict(result={"engines": {"classification": {"members_agree": True}}}, confidence=0.95,
                anomaly_score_ratio=None, unfamiliar=False, defect_class="scratches")
    confident = SimpleNamespace(**base)
    unsure = SimpleNamespace(**{**base, "confidence": 0.4, "unfamiliar": True,
                                "result": {"engines": {"classification": {"members_agree": False}}}})
    counts = {"scratches": 10, "crazing": 0}
    s1, _ = review_priority(confident, counts, 2)
    s2, why = review_priority(unsure, counts, 2)
    assert s2 > s1 and any(w.startswith("disagreement") for w in why) and any(w.startswith("unfamiliar") for w in why)
    rare = SimpleNamespace(**{**base, "defect_class": "crazing"})
    assert review_priority(rare, counts, 2)[0] > s1  # under-represented class is worth labelling first
    border = SimpleNamespace(**{**base, "result": {"engines": {"anomaly": {}}}, "confidence": None,
                                "anomaly_score_ratio": 1.02, "defect_class": None})
    assert review_priority(border, counts, 2)[1][0].startswith("borderline")


def test_review_queue_orders(client, admin):
    for order in ("priority", "oldest", "newest"):
        r = client.get("/api/reviews/queue", headers=admin, params={"order": order})
        assert r.status_code == 200 and r.json()["order"] == order
        pr = [i["priority"] for i in r.json()["items"]]
        if order == "priority":
            assert pr == sorted(pr, reverse=True)
    assert client.get("/api/reviews/queue", headers=admin, params={"order": "x"}).status_code == 422


def test_inspector_annotations(client, admin, defect_inspection, make_user):
    import cv2
    import numpy as np

    iid = defect_inspection["id"]
    d = client.get(f"/api/inspections/{iid}", headers=admin).json()
    w, h = d["result"]["input"]["width"], d["result"]["input"]["height"]
    mask = np.zeros((h, w), np.uint8)
    mask[h // 4: h // 2, w // 4: w // 2] = 255
    png = cv2.imencode(".png", mask)[1].tobytes()
    r = client.post(f"/api/inspections/{iid}/annotations", headers=admin, data={"kind": "mask"},
                    files={"file": ("m.png", png, "image/png")})
    assert r.status_code == 201, r.text
    a = r.json()
    assert abs(a["data"]["defect_fraction"] - 0.0625) < 0.01
    assert client.get(a["mask_url"], headers=admin).content[:4] == b"\x89PNG"
    r = client.post(f"/api/inspections/{iid}/annotations", headers=admin,
                    data={"kind": "bbox", "data": json.dumps({"boxes": [[1, 1, 10, 10]]})})
    assert r.status_code == 201
    bad = [({"kind": "bbox", "data": json.dumps({"boxes": [[0, 0, w + 50, 5]]})}, None),   # outside image
           ({"kind": "mask"}, None),                                                          # no file
           ({"kind": "mask"}, ("m.png", cv2.imencode(".png", np.zeros((10, 50), np.uint8))[1].tobytes(), "image/png")),
           ({"kind": "label", "label": "not-a-class"}, None), ({"kind": "blob"}, None)]
    for form, f in bad:
        rr = client.post(f"/api/inspections/{iid}/annotations", headers=admin, data=form,
                         files={"file": f} if f else None)
        assert rr.status_code == 422, (form, rr.text)
    assert client.post(f"/api/inspections/{iid}/annotations", headers=make_user("viewer"),
                       data={"kind": "label"}).status_code == 403
    anns = client.get(f"/api/inspections/{iid}", headers=admin).json()["annotations"]
    assert [x["kind"] for x in anns][-2:] == ["mask", "bbox"]
    assert client.delete(f"/api/inspections/{iid}/annotations/{anns[-1]['id']}", headers=admin).status_code == 200


def test_feedback_export_carries_inspector_annotations(client, admin, defect_inspection, monkeypatch, tmp_path):
    import csv

    import cv2
    import numpy as np

    import ml.config
    monkeypatch.setattr(ml.config, "DATA_DIR", tmp_path)  # never write into the repository's data/ during tests
    iid = defect_inspection["id"]
    d = client.get(f"/api/inspections/{iid}", headers=admin).json()
    if not any(a["kind"] == "mask" for a in d["annotations"]):
        m = np.zeros((d["result"]["input"]["height"], d["result"]["input"]["width"]), np.uint8)
        m[:20, :20] = 255
        client.post(f"/api/inspections/{iid}/annotations", headers=admin, data={"kind": "mask"},
                    files={"file": ("m.png", cv2.imencode(".png", m)[1].tobytes(), "image/png")})
    client.post(f"/api/inspections/{iid}/feedback", headers=admin,
                json={"verdict": "confirm", "final_decision": "reject", "location_confirmed": True})
    r = client.post("/api/datasets/feedback-export", headers=admin)
    assert r.status_code == 200 and r.json()["exported"] >= 1, r.text
    out = Path(r.json()["path"])
    rows = list(csv.DictReader(open(out / "manifest.csv", encoding="utf-8")))
    row = [x for x in rows if x["inspection_id"] == iid][-1]  # one row per feedback; ours is the latest
    assert row["mask"] and Path(row["mask"]).exists() and row["location_confirmed"] == "True"
