"""Optional outbound integration adapters (webhook, MQTT) driven by the in-process event bus.

Each adapter subscribes to ``app.services.events.bus`` and turns events into messages for line equipment or
other systems. Adapters are enabled by configuration and never required for inspection; a failing adapter is
logged without affecting inspections. ``status()`` reports what is enabled and working.

Not implemented (see docs/FEATURE_STATUS.md): OPC UA, Modbus TCP and vendor camera SDKs - they need a server
or hardware to validate against. No PLC integration has been tested against real hardware.
"""
from __future__ import annotations

from datetime import datetime, timezone

_ADAPTERS: dict[str, dict] = {}


def line_signal(inspection: dict) -> dict:
    """Map an inspection summary to the signal a reject gate / PLC would act on.

    PASS -> PASS (let the product through); FAIL -> REJECT; REVIEW REQUIRED / INCONCLUSIVE -> HOLD (divert to a
    manual station). The mapping is fixed and documented so line logic never has to interpret AI scores."""
    status = inspection.get("status")
    sig = "PASS" if status == "PASS" else "REJECT" if status == "FAIL" else "HOLD"
    return {"signal": sig, "status": status, "inspection_id": inspection.get("id"),
            "line_id": inspection.get("line_id"), "batch_id": inspection.get("batch_id"),
            "product_serial": inspection.get("product_serial"), "profile_id": inspection.get("profile_id"),
            "defect_class": inspection.get("defect_class"), "severity": inspection.get("severity"),
            "manual_review_required": inspection.get("manual_review_required"),
            "simulated": bool(inspection.get("is_simulated")), "inspected_at": inspection.get("created_at"),
            "sent_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds")}


def register(name: str, info: dict) -> None:
    _ADAPTERS[name] = info


def status() -> list[dict]:
    out = [{"name": "websocket", "enabled": True, "state": "active", "detail": "/api/ws/events (browser live events)"},
           {"name": "rest_api", "enabled": True, "state": "active",
            "detail": "POST /api/inspections with X-API-Key (camera trigger / MES integration)"}]
    out.append(_ADAPTERS.get("webhook") or {"name": "webhook", "enabled": False, "state": "disabled",
                                            "detail": "set QG_WEBHOOK_URL"})
    out.append(_ADAPTERS.get("mqtt") or {"name": "mqtt", "enabled": False, "state": "disabled",
                                         "detail": "set QG_MQTT_HOST (requires: pip install paho-mqtt)"})
    out += [{"name": "opc_ua", "enabled": False, "state": "not implemented",
             "detail": "needs an OPC UA server to validate against"},
            {"name": "modbus_tcp", "enabled": False, "state": "not implemented",
             "detail": "needs PLC hardware to validate against"}]
    return out


def start_all() -> None:
    from app.integrations import mqtt, webhook

    webhook.start()
    mqtt.start()
