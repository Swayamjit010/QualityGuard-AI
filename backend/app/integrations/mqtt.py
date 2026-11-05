"""MQTT adapter: publishes line signals and alerts to a broker (optional; needs ``pip install paho-mqtt``).

Topics (prefix = QG_MQTT_TOPIC_PREFIX, default ``qualityguard``):
    <prefix>/<line_id or "default">/signal   PASS / REJECT / HOLD per inspection (QoS 1, JSON)
    <prefix>/alerts                          alert events (QoS 1, JSON)

Covered by tests with an in-memory client stub (tests/test_integrations.py); not yet validated against a broker
on a physical line.
"""
from __future__ import annotations

import json
import logging

from app.core.config import get_settings
from app.integrations import line_signal, register
from app.services.events import bus

log = logging.getLogger("qualityguard.integrations.mqtt")


class MqttPublisher:
    def __init__(self, client, prefix: str):
        self.client, self.prefix = client, prefix.rstrip("/")
        self.stats = {"published": 0, "failed": 0}

    def topic_for(self, ev: dict) -> str | None:
        if ev["type"] == "inspection":
            line = str(ev["data"].get("line_id") or "default").replace("/", "_").replace("#", "_").replace("+", "_")
            return f"{self.prefix}/{line}/signal"
        if ev["type"] == "alert":
            return f"{self.prefix}/alerts"
        return None

    def __call__(self, ev: dict) -> None:
        topic = self.topic_for(ev)
        if not topic:
            return
        body = line_signal(ev["data"]) if ev["type"] == "inspection" else ev["data"]
        info = self.client.publish(topic, json.dumps(body, default=str), qos=1)
        if getattr(info, "rc", 0) == 0:
            self.stats["published"] += 1
        else:
            self.stats["failed"] += 1
            log.warning("MQTT publish to %s failed (rc=%s)", topic, info.rc)


def start() -> None:
    s = get_settings()
    if not s.mqtt_host:
        return
    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        register("mqtt", {"name": "mqtt", "enabled": True, "state": "unavailable",
                          "detail": "QG_MQTT_HOST is set but paho-mqtt is not installed (pip install paho-mqtt)"})
        log.warning("MQTT configured but paho-mqtt is not installed")
        return
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="qualityguard-ai")
    except AttributeError:  # paho-mqtt < 2
        client = mqtt.Client(client_id="qualityguard-ai")
    if s.mqtt_username:
        client.username_pw_set(s.mqtt_username, s.mqtt_password or None)
    if s.mqtt_tls:
        client.tls_set()
    try:
        client.connect_async(s.mqtt_host, s.mqtt_port, keepalive=30)
        client.loop_start()  # background network thread; reconnects automatically
    except Exception as e:
        register("mqtt", {"name": "mqtt", "enabled": True, "state": "error", "detail": str(e)[:200]})
        log.warning("MQTT connection setup failed: %s", e)
        return
    pub = MqttPublisher(client, s.mqtt_topic_prefix)
    bus.sinks.append(pub)
    register("mqtt", {"name": "mqtt", "enabled": True, "state": "active", "stats": pub.stats,
                      "detail": f"{s.mqtt_host}:{s.mqtt_port} prefix '{pub.prefix}'"})
