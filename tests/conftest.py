"""Test fixtures: an isolated SQLite DB + storage dir per session; real ML checkpoints from artifacts/."""
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_TMP = Path(tempfile.mkdtemp(prefix="qg_test_"))
os.environ.update({
    "QG_DATABASE_URL": f"sqlite:///{(_TMP / 'test.db').as_posix()}",
    "QG_STORAGE_DIR": str(_TMP / "storage"),
    "QG_DEFAULT_ADMIN_PASSWORD": "AdminPass123",
    "QG_SEED_DEMO_USERS": "false",
    "QG_ANTHROPIC_API_KEY": "",
    "ANTHROPIC_API_KEY": "",
    "QG_WORKERS": "1",
})
sys.path[:0] = [str(ROOT / "backend"), str(ROOT)]

ADMIN = ("admin@qualityguard.local", "AdminPass123")


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from app.main import app
    with TestClient(app) as c:
        yield c


def login(client, email, password) -> dict:
    r = client.post("/api/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture(scope="session")
def admin(client):
    return login(client, *ADMIN)


@pytest.fixture(scope="session")
def make_user(client, admin):
    def _make(role: str) -> dict:
        email = f"{role}@test.local"
        pw = "UserPass123"
        r = client.post("/api/users", headers=admin, json={"email": email, "full_name": role, "password": pw,
                                                            "role": role})
        assert r.status_code in (201, 409), r.text
        return login(client, email, pw)
    return _make


def sample(dataset: str, label: str, split: str = "test") -> Path:
    """A real image from a prepared dataset; skips the test when the dataset is not downloaded (e.g. in CI)."""
    mf = ROOT / "data" / "processed" / dataset / "manifest_latest.json"
    if not mf.exists():
        pytest.skip(f"dataset {dataset} not prepared")
    m = json.loads(mf.read_text())
    r = next(s for s in m["samples"] if s["split"] == split and s["label"] == label)
    if not (ROOT / r["path"]).exists():
        pytest.skip(f"dataset {dataset} images not downloaded (scripts/download_datasets.py)")
    return ROOT / r["path"]


HAS_BOTTLE = (ROOT / "artifacts/checkpoints/mvtec_bottle-anomaly-resnet18-v1/anomaly.pt").exists()


@pytest.fixture(scope="module")
def defect_inspection(client, admin):
    """A real inspection of a broken bottle (used by API, assistant and workflow tests)."""
    if not HAS_BOTTLE:
        pytest.skip("bottle model not trained")
    p = sample("mvtec_bottle", "broken_large")
    r = client.post("/api/inspections", headers=admin, data={"profile_id": "bottle", "product_serial": "SN-1"},
                    files={"file": (p.name, p.read_bytes(), "image/png")})
    assert r.status_code == 201, r.text
    return r.json()
