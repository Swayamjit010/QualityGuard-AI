"""Add-a-product lifecycle over the API with real images: upload good photos -> background training ->
approve & activate -> inspect (good item, defective item, different product)."""
import json
import shutil
import time

import pytest

from conftest import ROOT

MAN = ROOT / "data/processed/mvtec_bottle/manifest_latest.json"
pytestmark = pytest.mark.skipif(not MAN.exists(), reason="MVTec bottle dataset not prepared")


def _rows(split, good=True):
    m = json.loads(MAN.read_text())
    rows = [s for s in m["samples"] if s["split"] == split and (s["label"] == "good") == good]
    if not rows or not (ROOT / rows[0]["path"]).exists():
        pytest.skip("dataset images not downloaded")
    return [ROOT / r["path"] for r in rows]


def _files(field, paths):
    return [(field, (p.name, p.read_bytes(), "image/png")) for p in paths]


def test_product_validation(client, admin, make_user):
    r = client.post("/api/products", headers=admin, data={"name": "too few"}, files=_files("good", _rows("train")[:3]))
    assert r.status_code == 422 and "at least" in r.json()["detail"]
    assert client.post("/api/products", headers=make_user("viewer"), data={"name": "x y"},
                       files=_files("good", _rows("train")[:3])).status_code == 403
    assert client.get("/api/products/does_not_exist", headers=admin).status_code == 404
    assert client.get("/api/products/x/image/good/..%2F..%2Fstorage%2F.secret_key", headers=admin).status_code == 404


@pytest.fixture()
def cleanup():
    """The training subprocess writes to the real data/raw and artifacts/checkpoints folders: always remove what
    this test created, pass or fail."""
    slugs = []
    yield slugs
    for slug in slugs:
        shutil.rmtree(ROOT / "data" / "raw" / f"custom_{slug}", ignore_errors=True)
        for d in (ROOT / "artifacts" / "checkpoints").glob(f"custom_{slug}-anomaly-*"):
            shutil.rmtree(d, ignore_errors=True)


def test_add_product_train_activate_inspect(client, admin, cleanup):
    good = _rows("train")[:16]
    defect = _rows("val", good=False)[:3]
    r = client.post("/api/products", headers=admin, data={"name": "pytest bottle", "description": "pytest"},
                    files=_files("good", good) + _files("defect", defect))
    assert r.status_code == 201, r.text
    slug = r.json()["slug"]
    cleanup.append(slug)
    t0 = time.time()
    while True:
        p = client.get(f"/api/products/{slug}", headers=admin).json()
        if p["state"] not in ("running", "not started"):
            break
        assert time.time() - t0 < 900, "training did not finish"
        time.sleep(3)
    assert p["state"] == "trained", p
    assert p["summary"]["good_photos"] == 16 and p["summary"]["defect_photos"] == 3
    assert client.post(f"/api/inspections", headers=admin, data={"profile_id": f"custom_{slug}"},
                       files={"file": ("a.png", good[0].read_bytes(), "image/png")}).status_code in (404, 422)
    a = client.post(f"/api/products/{slug}/activate", headers=admin)
    assert a.status_code == 200 and a.json()["state"] == "ready", a.text
    pid = a.json()["profile_id"]
    assert any(x["id"] == pid for x in client.get("/api/profiles", headers=admin).json())

    def inspect(path):
        r = client.post("/api/inspections", headers=admin, data={"profile_id": pid},
                        files={"file": (path.name, path.read_bytes(), "image/png")})
        assert r.status_code == 201, r.text
        return r.json()

    assert inspect(_rows("test")[0])["status"] in ("PASS", "REVIEW REQUIRED")          # unseen good bottle
    assert inspect(_rows("test", good=False)[-1])["status"] in ("FAIL", "REVIEW REQUIRED")  # unseen broken bottle
    steel = ROOT / json.loads((ROOT / "data/processed/neu_cls/manifest_latest.json").read_text())["samples"][0]["path"]
    other = inspect(steel)
    assert other["unfamiliar"] and other["status"] == "INCONCLUSIVE"                  # a different kind of item
    listed = {x["id"]: x["state"] for x in client.get("/api/products", headers=admin).json()}
    assert listed[pid] == "ready" and listed["bottle"] == "ready"
