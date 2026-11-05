"""Self-registration (email + username + password) -> request inspect access -> admin approves / denies."""
from tests.conftest import login, sample


def _register(client, email, username, pw="GoodPass123"):
    return client.post("/api/auth/register", json={"email": email, "username": username, "password": pw})


def test_register_with_username_and_login_with_either(client):
    r = _register(client, "maya@plant.io", "maya.k")
    assert r.status_code == 201, r.text
    assert r.json()["username"] == "maya.k" and r.json()["role"] == "viewer"
    assert "inspect" not in r.json()["permissions"]
    assert _register(client, "other@plant.io", "MAYA.K").status_code == 409          # usernames are case-insensitive
    assert _register(client, "maya@plant.io", "someone").status_code == 409         # email taken
    assert _register(client, "x@plant.io", "a b").status_code == 422                # invalid username
    assert login(client, "maya.k", "GoodPass123")
    assert login(client, "Maya@Plant.io", "GoodPass123")
    assert client.post("/api/auth/login", json={"email": "maya.k", "password": "wrong"}).status_code == 401


def test_request_deny_then_approve_grants_inspect(client, admin):
    assert _register(client, "lee@plant.io", "lee").status_code == 201
    u = login(client, "lee", "GoodPass123")
    p = sample("mvtec_bottle", "good")
    files = {"file": (p.name, p.read_bytes(), "image/png")}
    assert client.post("/api/inspect", headers=u, files=files).status_code == 403   # viewer cannot inspect

    # users cannot see or decide other people's requests
    assert client.get("/api/access-requests", headers=u).status_code == 403

    r = client.post("/api/access-requests", headers=u, json={"reason": "Line 3 operator"})
    assert r.status_code == 201 and r.json()["status"] == "pending"
    rid = r.json()["id"]
    assert client.post("/api/access-requests", headers=u, json={}).status_code == 409  # one pending at a time
    assert client.post(f"/api/access-requests/{rid}/decision", headers=u, json={"action": "approve"}).status_code == 403

    pend = client.get("/api/access-requests", headers=admin, params={"status": "pending"}).json()
    assert pend["pending"] >= 1 and any(i["id"] == rid and i["user"]["username"] == "lee" for i in pend["items"])

    d = client.post(f"/api/access-requests/{rid}/decision", headers=admin, json={"action": "deny", "note": "Not on rota"})
    assert d.status_code == 200 and d.json()["status"] == "denied" and d.json()["decision_note"] == "Not on rota"
    assert client.get("/api/auth/me", headers=u).json()["role"] == "viewer"
    assert client.post(f"/api/access-requests/{rid}/decision", headers=admin, json={"action": "approve"}).status_code == 409

    rid2 = client.post("/api/access-requests", headers=u, json={"reason": "Now on rota"}).json()["id"]
    a = client.post(f"/api/access-requests/{rid2}/decision", headers=admin, json={"action": "approve"})
    assert a.status_code == 200 and a.json()["status"] == "approved"
    me = client.get("/api/auth/me", headers=u).json()
    assert me["role"] == "inspector" and "inspect" in me["permissions"]
    mine = client.get("/api/access-requests/mine", headers=u).json()
    assert [m["status"] for m in mine] == ["approved", "denied"]
    assert client.post("/api/inspect", headers=u, files=files).status_code == 201      # can inspect now
    assert client.post("/api/access-requests", headers=u, json={}).status_code == 409  # already has the permission

    audit = client.get("/api/system/audit", headers=admin, params={"action": "access."}).json()
    actions = {e["action"] for e in (audit["items"] if isinstance(audit, dict) else audit)}
    assert {"access.request", "access.denied", "access.approved"} <= actions
