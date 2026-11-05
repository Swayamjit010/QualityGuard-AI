"""Offline assistant: intents are routed correctly, answers are grounded in the database and never fall back to an
unrelated summary."""
import re

import pytest


def ask(client, admin, q, history=None):
    r = client.post("/api/assistant/chat", headers=admin, json={"question": q, "history": history or []})
    assert r.status_code == 200, r.text
    return r.json()


@pytest.mark.parametrize("q,tool", [
    ("How many inspections failed in the last 7 days?", "search_inspections"),
    ("What is the most common defect?", "defect_statistics"),
    ("Which product has the highest rejection rate?", "defect_statistics"),
    ("Explain what Grad-CAM shows", "explain_concept"),
    ("Are there inspections waiting for review?", "search_inspections"),
    ("How accurate is the steel classifier?", "model_comparison"),
    ("Is quality getting worse this week?", "quality_trend"),
    ("Any critical alerts?", "list_alerts"),
    ("How do I check if an item is defective?", "app_help"),
    ("How do I add a new product?", "app_help"),
    ("What products can you inspect?", "list_products"),
    ("What do PASS, FAIL and REVIEW mean?", "explain_concept"),
    ("Show failed bottle inspections", "search_inspections"),
])
def test_intents_route_to_the_right_tool(client, admin, defect_inspection, q, tool):
    a = ask(client, admin, q)
    assert a["mode"] == "offline" and a["answer"]
    assert tool in [t["tool"] for t in a["tool_calls"]], (q, a["tool_calls"])


def test_explain_last_inspection_and_follow_up(client, admin, defect_inspection):
    a = ask(client, admin, "Why was the last bottle rejected?")
    assert "get_inspection" in [t["tool"] for t in a["tool_calls"]]
    ids = re.findall(r"/inspections/([0-9a-f]{32})", a["answer"])
    assert ids, a["answer"]
    hist = [{"role": "user", "content": "Why was the last bottle rejected?"}, {"role": "assistant", "content": a["answer"]}]
    b = ask(client, admin, "find similar defects to it", hist)
    assert "similar_defects" in [t["tool"] for t in b["tool_calls"]]
    assert ids[0][:8] in b["answer"]


def test_counts_are_grounded(client, admin, defect_inspection):
    a = ask(client, admin, "how many inspections failed in total")
    n = int(re.search(r"\*\*(\d+)\*\*", a["answer"]).group(1))
    actual = client.get("/api/inspections", headers=admin, params={"status": "FAIL", "page_size": 1}).json()["total"]
    assert n == actual


def test_unknown_question_is_admitted_not_padded(client, admin):
    a = ask(client, admin, "purple elephants dancing")
    assert "not sure" in a["answer"] and not a["tool_calls"]
    assert ask(client, admin, "hello")["tool_calls"] == []
    assert ask(client, admin, "   ")["answer"]
