"""Integration adapters: line-signal mapping, MQTT topics/payloads (stub client), webhook payloads, status API."""
import json

from app.integrations import line_signal, status
from app.integrations.mqtt import MqttPublisher
from app.integrations.webhook import make_sink

INSP = {"id": "abc", "status": "FAIL", "line_id": "line-1", "batch_id": "b1", "product_serial": "SN-9",
        "profile_id": "bottle", "defect_class": None, "severity": "high", "manual_review_required": False,
        "is_simulated": True, "created_at": "2026-11-05T10:00:00Z"}


def test_line_signal_mapping():
    assert line_signal({**INSP, "status": "PASS"})["signal"] == "PASS"
    assert line_signal(INSP)["signal"] == "REJECT"
    assert line_signal({**INSP, "status": "REVIEW REQUIRED"})["signal"] == "HOLD"
    assert line_signal({**INSP, "status": "INCONCLUSIVE"})["signal"] == "HOLD"
    s = line_signal(INSP)
    assert s["inspection_id"] == "abc" and s["line_id"] == "line-1" and s["simulated"] is True


class _Info:
    def __init__(self, rc):
        self.rc = rc


class _StubClient:
    def __init__(self, rc=0):
        self.sent, self.rc = [], rc

    def publish(self, topic, payload, qos=0):
        self.sent.append((topic, json.loads(payload), qos))
        return _Info(self.rc)


def test_mqtt_publisher_topics_and_payloads():
    c = _StubClient()
    pub = MqttPublisher(c, "qg/")
    pub({"type": "inspection", "data": INSP})
    pub({"type": "inspection", "data": {**INSP, "line_id": None, "status": "PASS"}})
    pub({"type": "alert", "data": {"id": 1, "title": "Critical defect"}})
    pub({"type": "job", "data": {"id": "j"}})  # not published
    assert [t for t, _, _ in c.sent] == ["qg/line-1/signal", "qg/default/signal", "qg/alerts"]
    assert c.sent[0][1]["signal"] == "REJECT" and c.sent[1][1]["signal"] == "PASS" and c.sent[0][2] == 1
    assert pub.stats == {"published": 3, "failed": 0}
    # MQTT wildcards in a line id must not create a wildcard topic
    assert pub.topic_for({"type": "inspection", "data": {**INSP, "line_id": "a/#"}}) == "qg/a__/signal"


def test_mqtt_publish_failure_is_counted_not_raised():
    pub = MqttPublisher(_StubClient(rc=4), "qg")
    pub({"type": "alert", "data": {"id": 1}})
    assert pub.stats == {"published": 0, "failed": 1}


def test_webhook_sink_filters_and_adds_line_signal():
    posted = []
    sink, stats = make_sink("http://example.invalid/hook", {"inspection"}, post=lambda url, body: posted.append(body))
    sink({"type": "alert", "data": {"id": 1}})  # not subscribed
    sink({"type": "inspection", "data": INSP})
    assert len(posted) == 1 and posted[0]["line_signal"]["signal"] == "REJECT" and stats["sent"] == 1

    def boom(url, body):
        raise ConnectionError("down")
    sink2, stats2 = make_sink("http://example.invalid/hook", {"alert"}, post=boom)
    sink2({"type": "alert", "data": {}})  # must not raise
    assert stats2 == {"sent": 0, "failed": 1}


def test_status_lists_adapters(client, admin):
    names = {a["name"]: a["state"] for a in status()}
    assert names["opc_ua"] == "not implemented" and names["mqtt"] in {"disabled", "active", "unavailable", "error"}
    r = client.get("/api/integrations", headers=admin)
    assert r.status_code == 200 and {a["name"] for a in r.json()} >= {"websocket", "webhook", "mqtt"}
    s = client.get("/api/system/status", headers=admin).json()
    assert s["database"]["ok"] and "resources" in s and "integrations" in s
