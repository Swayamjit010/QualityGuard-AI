"""Gradio UI handlers call the same service layer as the API."""
import pytest

from conftest import ADMIN, HAS_BOTTLE, sample

gr = pytest.importorskip("gradio")


def test_gradio_mounted(client):
    r = client.get("/gradio/")
    assert r.status_code in (200, 401, 302, 307)


def test_gradio_auth(client):
    from app.gradio_ui import authenticate
    assert authenticate(*ADMIN) is True
    assert authenticate(ADMIN[0], "wrong") is False


def test_gradio_inspect_report_history(client):
    if not HAS_BOTTLE:
        pytest.skip("bottle model not trained")
    from app.gradio_ui import history, inspect_image, make_report, model_choices, model_details
    p = sample("mvtec_bottle", "contamination")
    md, gallery, iid, _ = inspect_image(str(p), "bottle", "G-1", True)
    assert iid and any(s in md for s in ("FAIL", "PASS", "REVIEW REQUIRED", "INCONCLUSIVE"))
    assert gallery and all(isinstance(c, str) for _, c in gallery)
    rep = make_report(iid)
    assert rep["value"].endswith(".pdf")
    assert any(row[-1] == iid[:8] for row in history("bottle"))
    mid = next(v for label, v in model_choices() if "bottle" in label)
    text, plots = model_details(mid)
    assert "mvtec_bottle" in text
    with pytest.raises(gr.Error):
        inspect_image(None, "bottle", "", True)
