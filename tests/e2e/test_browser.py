"""Browser end-to-end tests (Playwright) against a real server process with real models.

Runs a uvicorn server on a free port with its own temporary database and storage, then drives the built
React app (frontend/dist) in headless Chrome:

* the public landing page and the sign-in form
* every page in the navigation renders without JavaScript errors
* the complete single-image workflow: upload -> inspection -> result + explanation maps -> full record -> PDF report

Requirements: ``pip install playwright`` plus either an installed Google Chrome (used via channel="chrome")
or ``python -m playwright install chromium``; and ``npm run build`` in frontend/. Skipped otherwise.
Run only these tests with ``pytest -m e2e``; exclude them with ``pytest -m "not e2e"``.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e
playwright = pytest.importorskip("playwright.sync_api")

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "frontend" / "dist" / "index.html"
ADMIN = ("admin@qualityguard.local", "E2eAdminPass123")
ROUTES = ["/", "/inspect", "/inspect/advanced", "/products", "/products/new", "/inspections", "/reviews", "/alerts", "/simulation",
          "/analytics", "/anomaly", "/xai", "/reports", "/assistant", "/models", "/datasets", "/profiles", "/system",
          "/admin"]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server():
    if not DIST.exists():
        pytest.skip("frontend not built (cd frontend && npm run build)")
    tmp = Path(tempfile.mkdtemp(prefix="qg_e2e_"))
    port = _free_port()
    env = {**os.environ, "QG_DATABASE_URL": f"sqlite:///{(tmp / 'e2e.db').as_posix()}",
           "QG_STORAGE_DIR": str(tmp / "storage"), "QG_DEFAULT_ADMIN_PASSWORD": ADMIN[1],
           "QG_SEED_DEMO_USERS": "false", "QG_WORKERS": "1", "QG_ANTHROPIC_API_KEY": "", "ANTHROPIC_API_KEY": ""}
    log = open(tmp / "server.log", "w")
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--app-dir", str(ROOT / "backend"),
                             "--host", "127.0.0.1", "--port", str(port)], cwd=ROOT, env=env, stdout=log,
                            stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    import httpx
    for _ in range(120):
        try:
            if httpx.get(base + "/api/health", timeout=2).status_code == 200:
                break
        except httpx.HTTPError:
            pass
        if proc.poll() is not None:
            pytest.fail("server exited: " + (tmp / "server.log").read_text()[-2000:])
        time.sleep(0.5)
    else:
        proc.kill()
        pytest.fail("server did not start")
    yield base
    proc.terminate()
    try:
        proc.wait(15)
    except subprocess.TimeoutExpired:
        proc.kill()
    log.close()


@pytest.fixture(scope="module")
def browser():
    # fake camera: Chrome supplies a synthetic test-pattern video device and auto-accepts the permission prompt
    args = ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"]
    with playwright.sync_playwright() as p:
        try:
            b = p.chromium.launch(channel="chrome", headless=True, args=args)
        except Exception:
            try:
                b = p.chromium.launch(headless=True, args=args)
            except Exception as e:  # no browser available on this machine
                pytest.skip(f"no Chromium/Chrome available for Playwright: {e}")
        yield b
        b.close()


@pytest.fixture()
def page(browser, server):
    ctx = browser.new_context(base_url=server, viewport={"width": 1440, "height": 900}, accept_downloads=True,
                              permissions=["camera"])
    pg = ctx.new_page()
    pg.errors = []
    pg.on("pageerror", lambda e: pg.errors.append(f"pageerror: {e}"))
    pg.on("console", lambda m: m.type == "error" and pg.errors.append(f"console: {m.text}"))
    yield pg
    ctx.close()


def _login(pg, ident=ADMIN[0], pw=ADMIN[1]):
    pg.goto("/login")
    pg.fill("input[autocomplete=username]", ident)
    pg.fill("input[type=password]", pw)
    pg.click("button.btn-primary")
    pg.wait_for_selector(".sidebar", timeout=20000)


def test_landing_and_login(page):
    page.goto("/")
    assert page.get_by_role("heading", level=1).inner_text().startswith("Find defects")
    page.get_by_role("link", name="Sign in to the platform").click()
    page.wait_for_selector("form.login-card")
    _login(page)
    assert page.locator(".user-name").inner_text() == "Administrator"
    assert not page.errors, page.errors


def test_every_page_renders_without_errors(page):
    _login(page)
    for route in ROUTES:
        page.goto(route)
        page.wait_for_selector(".page-header, .content h1", timeout=20000)
        page.wait_for_load_state("networkidle")
        assert "Page not found" not in page.locator(".content").inner_text(), route
        assert not page.locator(".error-box").count(), f"{route}: {page.locator('.error-box').first.inner_text()}"
    assert not page.errors, page.errors


def _test_image(dataset: str) -> Path:
    m = json.loads((ROOT / "data" / "processed" / dataset / "manifest_latest.json").read_text())
    return ROOT / next(s for s in m["samples"] if s["split"] == "test")["path"]


@pytest.mark.skipif(not (ROOT / "artifacts/checkpoints/neu_cls-mobilenet_v2-v1/model.pt").exists(),
                    reason="steel classifiers not trained")
def test_single_image_inspection_to_pdf(page, tmp_path):
    """Main workflow on the AI Inspection page: upload -> INSPECT (auto category) -> verdict + maps -> PDF -> record."""
    _login(page)
    page.goto("/inspect")
    page.set_input_files("[data-testid=inspect-file]", str(_test_image("neu_cls")))
    page.get_by_test_id("inspect-button").click()
    page.wait_for_selector("[data-testid=verdict]", timeout=120000)
    verdict = page.get_by_test_id("verdict").inner_text().strip()
    assert verdict in {"NORMAL", "DEFECTIVE", "MANUAL INSPECTION REQUIRED"}
    assert "Hot-rolled steel strip surface" in page.get_by_test_id("inspection-result").inner_text()  # auto-recognised
    assert page.locator("[data-testid=inspection-result] figure img").count() >= 2  # input + Grad-CAM maps
    assert page.get_by_text("NOT a defect boundary").count() >= 1  # Grad-CAM is labelled as an explanation
    with page.expect_download(timeout=120000) as dl:
        page.get_by_role("button", name="Download PDF report").click()
    out = tmp_path / "report.pdf"
    dl.value.save_as(out)
    assert out.read_bytes()[:5] == b"%PDF-" and out.stat().st_size > 20_000
    page.get_by_role("link", name="Full record").click()
    page.wait_for_selector("text=Open in XAI lab")
    assert page.get_by_test_id("detail-verdict").inner_text().strip() == verdict
    page.get_by_role("link", name="Open in XAI lab").click()
    page.wait_for_selector("text=Plain-language explanation", timeout=20000)
    assert page.locator(".xai-pane img").count() == 2
    assert not page.errors, page.errors


def test_unsupported_image_in_browser_is_manual(page):
    probes = sorted((ROOT / "data/raw/foreign_probe/imagenette").glob("*.JPEG"))
    if not probes or not (ROOT / "artifacts/checkpoints/component_recognizer-resnet18-v1/recognizer.pt").exists():
        pytest.skip("recogniser or probe images missing")
    _login(page)
    page.goto("/inspect")
    page.set_input_files("[data-testid=inspect-file]", str(probes[0]))
    page.get_by_test_id("inspect-button").click()
    page.wait_for_selector("[data-testid=verdict]", timeout=120000)
    assert page.get_by_test_id("verdict").inner_text().strip() == "MANUAL INSPECTION REQUIRED"
    assert page.get_by_text("UNSUPPORTED OR UNFAMILIAR COMPONENT").count() >= 1
    assert not page.errors, page.errors


@pytest.mark.skipif(not (ROOT / "artifacts/checkpoints/mvtec_bottle-anomaly-resnet18-v1/anomaly.pt").exists(),
                    reason="bottle anomaly model not trained")
def test_live_camera_capture(page):
    """Camera mode end to end with Chrome's fake camera. Its synthetic test pattern is nothing like a bottle,
    so the correct outcome is a non-PASS decision (unfamiliar input / anomaly), never a confident PASS."""
    _login(page)
    page.goto("/inspect/advanced")
    page.select_option(".page-header select", "bottle")
    page.get_by_role("button", name="Camera").click()
    page.get_by_role("button", name="Start camera").click()
    page.wait_for_function("document.querySelector('.camera-wrap video').videoWidth > 0", timeout=20000)
    # devices are listed once permission is granted: "Default camera" + the fake device
    page.wait_for_function("document.querySelectorAll('select[aria-label=Camera] option').length >= 2", timeout=10000)
    page.get_by_role("button", name="Capture & inspect").click()
    page.wait_for_selector(".camera-overlay", timeout=120000)
    status = page.locator(".camera-overlay").inner_text()
    assert status.split()[0] in {"FAIL", "REVIEW", "INCONCLUSIVE"}, status
    assert page.get_by_text("Inspected").count() == 1  # session counters shown
    page.get_by_role("button", name="Stop camera").click()
    assert not [e for e in page.errors if "Camera" not in e], page.errors


def test_every_internal_link_resolves(page):
    """Crawl: collect every in-app link on every page (navigation, cards, tables, KPI tiles) and open each one.
    None may land on the 'Page not found' route or raise a JavaScript error."""
    _login(page)
    seen, queue, broken = set(), list(ROUTES), []
    while queue and len(seen) < 120:
        path = queue.pop(0)
        if path in seen:
            continue
        seen.add(path)
        page.goto(path)
        page.wait_for_load_state("networkidle")
        if "Page not found" in page.locator(".content").inner_text():
            broken.append(path)
            continue
        hrefs = page.eval_on_selector_all("a[href^='/']", "els => els.map(e => e.getAttribute('href'))")
        for h in hrefs:
            h = h.split("#")[0]
            if h and not h.startswith(("/api/", "/gradio")) and h not in seen:
                queue.append(h)
    assert not broken, f"links to missing pages: {broken}"
    assert len(seen) >= len(ROUTES)
    assert not page.errors, page.errors


def test_assistant_answers_with_working_links(page):
    _login(page)
    page.goto("/assistant")
    page.get_by_role("button", name="How do I check if an item is defective?").click()
    page.wait_for_selector(".msg-assistant .md a", timeout=30000)
    assert "Run inspection" in page.locator(".msg-assistant").last.inner_text()
    page.locator(".msg-assistant .md a", has_text="Inspect").first.click()
    page.wait_for_url("**/inspect")
    page.goto("/assistant")  # the conversation is kept for the browser session
    assert page.locator(".msg-user").count() >= 1
    page.fill("input[aria-label=Question]", "purple elephants")
    page.get_by_role("button", name="Send").click()
    page.wait_for_function("document.querySelectorAll('.msg-assistant').length >= 2", timeout=30000)
    assert "not sure" in page.locator(".msg-assistant").last.inner_text()
    assert not page.errors, page.errors


def test_add_product_page_and_plain_verdict(page):
    _login(page)
    page.goto("/products")
    page.get_by_role("link", name="+ Add a product").click()
    page.wait_for_url("**/products/new")
    assert page.get_by_role("button", name="Add 10 more good photo(s)").is_disabled()
    page.goto("/inspect/advanced?profile=bottle")
    page.wait_for_function("document.querySelector('.page-header select').value === 'bottle'", timeout=10000)
    page.goto("/inspect?profile=bottle")  # the main page honours the same parameter
    page.wait_for_function("document.querySelector('#qg-category').value === 'bottle'", timeout=10000)
    assert not page.errors, page.errors


def test_self_registration_access_request_and_admin_approval(browser, server):
    """New user registers (email + username + password) -> sees the access banner -> requests inspection access;
    the admin approves it on Administration -> Access requests; the user's Inspect page then becomes available."""
    def ctx_page():
        c = browser.new_context(base_url=server, viewport={"width": 1440, "height": 900})
        pg = c.new_page()
        pg.errors = []
        pg.on("pageerror", lambda e: pg.errors.append(f"pageerror: {e}"))
        return c, pg

    uc, user = ctx_page()
    user.goto("/login")
    user.get_by_role("button", name="No account? Create one").click()
    user.fill("input[type=email]", "e2e.worker@plant.io")
    user.fill("input[autocomplete=username]", "e2e.worker")
    user.fill("input[type=password]", "WorkerPass123")
    user.get_by_role("button", name="Register").click()
    user.wait_for_selector("text=Account created")
    _login(user, "e2e.worker", "WorkerPass123")                      # sign in with the username
    banner = user.get_by_test_id("access-banner")
    banner.wait_for(timeout=10000)
    assert user.locator("nav a[href='/inspect']").count() == 0       # no Inspect menu for a Viewer
    banner.get_by_role("button", name="Request inspection access").click()
    user.fill("input[aria-label=Reason]", "Line 2 night shift")
    user.get_by_role("button", name="Send request").click()
    user.wait_for_selector("text=is waiting for an administrator", timeout=10000)

    ac, admin = ctx_page()
    _login(admin)
    admin.goto("/admin")
    admin.get_by_role("button", name="Access requests").first.wait_for()
    admin.wait_for_selector("text=e2e.worker@plant.io", timeout=10000)
    assert admin.locator("text=Line 2 night shift").count() == 1
    admin.fill("input[aria-label='Note for e2e.worker@plant.io']", "Welcome")
    admin.get_by_role("button", name="Approve").click()
    admin.wait_for_selector("text=No pending requests.", timeout=10000)

    # the user's session picks up the new role live (WebSocket event) or on reload
    user.reload()
    user.wait_for_selector("nav a[href='/inspect']", timeout=15000)
    assert user.get_by_test_id("access-banner").count() == 0
    user.goto("/inspect")
    user.get_by_test_id("inspect-button").wait_for()
    assert not user.errors and not admin.errors, user.errors + admin.errors
    uc.close()
    ac.close()
