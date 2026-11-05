# QualityGuard AI

> **Live demo:** https://qualityguard.tail945149.ts.net — hosted from the author's PC via Tailscale Funnel, so it is online only
> while that PC is on. Create an account, then use *Request inspection access*; an administrator approves it.
> Trained models: [Release v1.0.0](https://github.com/Swayamjit010/QualityGuard-AI/releases/tag/v1.0.0). Non-commercial use only (dataset licences).


**Deep learning-based industrial product defect detection and quality inspection system**
KIIT School of Computer Engineering · Project 2026–27

QualityGuard AI inspects images of industrial components and surfaces. Upload an image, press **INSPECT**, and the
trained models return one of three results:

| Result | Meaning |
|---|---|
| **NORMAL** | No significant anomaly detected within the models' validated scope |
| **DEFECTIVE** | A supported defect or significant anomaly was detected (with class, location, heat-map, severity) |
| **MANUAL INSPECTION REQUIRED** | The image is unsupported or unfamiliar, unsuitable (blurred, too small, corrupted), ambiguous, or the engines disagree |

The component category is **recognised automatically**. Supported categories (each with its own trained models):

| Category | Training data | Engines |
|---|---|---|
| Hot-rolled steel strip surface | NEU-CLS | defect-type classifier ensemble (6 classes) + Grad-CAM |
| Bottle, metal nut, screw, metal grid | MVTec AD | PatchCore + PaDiM anomaly detection and localisation |
| Electrical commutator surface | KolektorSDD2 | binary classifier ensemble + U-Net segmentation + PatchCore heat-map |
| Metal brackets (black, brown, white), metal connector, metal plate, metal tubes | MPDD (Metal Parts Defect Detection) | PatchCore anomaly detection and localisation |

Anything else (a pump, a gear, a photo of a dog) is reported as **unsupported → MANUAL INSPECTION REQUIRED**, and no
reliability claim is made. Measured results are in [docs/RESULTS.md](docs/RESULTS.md) (§10 for the end-to-end verdicts) and
[reports/end_to_end_evaluation.md](reports/end_to_end_evaluation.md). Read their limitations: for example, 9 % of
defective test commutators were auto-passed, and steel strip has no defect-free training class.

Visual inspection cannot diagnose internal bearing wear, vibration, overheating, lubrication or electrical faults;
the system does not claim to.

For each inspection you also get a line disposition from the configurable product-profile rules
(PASS / FAIL / REVIEW REQUIRED / INCONCLUSIVE), a severity estimate, the explanation maps and a PDF report.

**Inspecting your own kind of item:** open *Products → Add a product*, upload 10+ (ideally 30–100) photos of good
items and wait a few minutes. Once approved, the item can be checked on the *Inspect* page like the built-in
products. See [docs/QUALITY_RULES.md](docs/QUALITY_RULES.md#custom-products-add-a-product).

## Features

- **Inspection modes:** single image, live camera, batch or folder upload, video (frame sampling with an annotated output video), golden-reference comparison, and REST API with API keys.
- **Explainability:** Grad-CAM, Grad-CAM++ and Score-CAM for classifiers; PatchCore and PaDiM anomaly maps, a segmentation mask, region boxes and measurements for anomaly models.
- **Reliability safeguards:** temperature-calibrated confidence, a Mahalanobis/energy out-of-distribution check, ensemble and engine-disagreement detection, and input-quality validation (blur, darkness, contrast, size).
- **Profile-driven decisions:** severity and acceptance rules are configured per product profile in JSON, with a revision history and rollback. See [docs/QUALITY_RULES.md](docs/QUALITY_RULES.md).
- **Human review workflow:** a review queue ordered by active-learning priority (uncertainty, disagreement, borderline scores, unfamiliar inputs, rare classes), confirm/reject with corrected class and severity, inspector annotations (defect masks, boxes), and export of reviewed images with their annotations as a retraining dataset.
- **Operations:**
  - live dashboard with WebSocket updates
  - alerts for critical, repeated and unfamiliar defects, drift and failures
  - production-line simulation
  - drift monitoring (KS test and PSI on input statistics, χ² test on predictions)
- **Analytics and reporting:**
  - trends, defect and severity distributions, false-positive/false-negative rates against ground truth, and a defect-rate forecast
  - PDF reports for inspections, batches, periods, model evaluation and comparison, datasets, drift and human reviews
- **MLOps:**
  - versioned datasets with leakage-safe splits
  - a model registry with an approval gate (validated → approved → deployed) and rollback
  - training and resuming from the UI, with optional MLflow logging
- **Security:**
  - JWT login, five roles with 15 permissions, and API keys
  - rate limiting and an audit trail of every action
  - upload validation and path-traversal-safe storage
- **Interfaces:** a public landing page and a full React dashboard (including an Anomaly center, an Explainable AI lab and a System health page), plus the lightweight **Gradio** UI described in the proposal (`/gradio`).
- **Integration:** REST API with API keys, WebSocket events, webhook and MQTT adapters that emit PASS / REJECT / HOLD line signals (optional; see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#integration-layer-backendappintegrations)).
- **Operations tooling:** Alembic migrations, a backup script, measured benchmarks, ONNX/INT8 edge export, GitHub Actions CI.
- **Add a product:** train an anomaly detector for a new kind of item from photos of good examples (defective photos optional, used only to report detection). Calibration is out-of-fold, and activation needs approval.
- **Quality assistant:** answers questions such as *"why was the last bottle rejected?"*, *"is quality getting worse?"*, *"what needs review?"* or *"how do I add a product?"* from database queries, with links to the records. It works offline (intent engine, no API key needed) and uses Claude (`claude-opus-5-5`) when `QG_ANTHROPIC_API_KEY` is set. It never makes up numbers and doesn't look at images itself.

## Live demo and licence

- **Permanent public link (free, from your own PC):** Tailscale Funnel publishes the local server at a fixed HTTPS address.
  One-time setup: install Tailscale, sign in, then `tailscale funnel --bg 8000` (approve Funnel in the Tailscale admin page when
  asked). Then keep the server running with `scripts\start_server.ps1` (restarts it if it exits). The site works only while the
  PC is on and the server is running. `scripts\start_public.ps1` is an alternative that needs no account but gives a random
  `trycloudflare.com` address that changes on every start. Hugging Face Docker Spaces now require a PRO subscription;
  `scripts/deploy_hf_space.py` is ready for that case.
- **Trained models:** GitHub Release `v1.0.0` (`python scripts/download_models.py`).
- **Licence note:** the models are trained on MVTec AD, KolektorSDD2 and MPDD (CC BY-NC-SA 4.0) and NEU-CLS (research
  use). The models and their outputs are for **non-commercial use only**, with attribution to those datasets.
  The datasets themselves are not redistributed here.

## Repository layout

```
backend/app/        FastAPI application (api/, services/, models/, security/, workers/, integrations/, gradio_ui.py)
backend/migrations/ Alembic migrations (alembic -c backend/alembic.ini upgrade head)
ml/                 ML library: datasets, preprocessing, architectures, training, anomaly_detection,
                    explainability, uncertainty, segmentation, quality rules, evaluation, inference pipeline
frontend/           React + Vite dashboard (built into frontend/dist and served by the backend)
scripts/            download / prepare datasets, train_classifiers.py, train_anomaly.py, compare_models.py,
                    benchmark.py, export_onnx.py, make_demo_set.py, backup.py, export_openapi.py, run_gradio.py
configs/            product profiles (bottle, metal_nut, steel_strip) and the defect knowledge base
artifacts/          checkpoints/<run_id>/ (model, metrics.json, confusion matrix, history, ROC/PR), dataset reports
docs/               proposal, architecture, quality rules, datasets, results
tests/              pytest suite: ML units, API, workflows, migrations, integrations; tests/e2e = Playwright
.github/workflows/  CI (tests + frontend build)
Dockerfile, docker-compose.yml   container build (repository root)
reports/            offline evaluation reports (end-to-end evaluation)
storage/            runtime state: uploads, visualisations, generated PDFs, SQLite DB (created on first start)
```

## Quick start (Windows / Linux / macOS)

Requires Python 3.11 and Node 18+ (Node is only needed to build the dashboard).

```bash
# 1. Python dependencies (CPU is fine; install CUDA torch first if you have a GPU)
pip install -r requirements.txt

# 1b. Trained models (skip if you train them yourself in step 3): ~1.1 GB from the GitHub Release
python scripts/download_models.py

# 2. Data (only needed for training / evaluation): download the public datasets, then analyse them and build leakage-safe splits
python scripts/download_datasets.py --neu --mvtec bottle metal_nut screw grid --ksdd2 --mpdd
python scripts/prepare_datasets.py

# 3. Train (CPU timings: ~15 min per classifier, ~3-8 min per anomaly category)
python scripts/train_classifiers.py --archs custom_cnn mobilenet_v2 efficientnet_b0 resnet50 --epochs 8
python scripts/train_anomaly.py bottle metal_nut screw grid
python -c "from ml.anomaly_detection.train import train_category as t; [t(c, manifest=f'mpdd_{c}') for c in ('bracket_black','bracket_brown','bracket_white','connector','metal_plate','tubes')]"
python -c "from ml.anomaly_detection.train import train_category as t; t('commutator', manifest='ksdd2', max_train=400)"
python scripts/train_classifiers.py --dataset ksdd2 --archs mobilenet_v2 efficientnet_b0 --epochs 10
python scripts/train_segmentation.py              # U-Net on KolektorSDD2 masks (~4.5 min/epoch on CPU)
python scripts/train_recognizer.py                # automatic category recognition (needs all manifests)
python scripts/calibrate_image_quality.py         # per-category blur limits (train+val)
python scripts/evaluate_profiles.py               # end-to-end verdicts on held-out test -> reports/

# optional: the other architectures in the comparison (each 5-35 min on CPU)
python scripts/train_classifiers.py --archs mobilenet_v3_small vit_tiny densenet121 convnext_tiny --epochs 8 --batch-size 16
python scripts/compare_models.py          # refresh docs/RESULTS.md tables + chart

# 4. Build the dashboard
cd frontend && npm install && npm run build && cd ..

# 5. Run
uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Then open:

| URL | What |
|---|---|
| http://127.0.0.1:8000 | React dashboard |
| http://127.0.0.1:8000/gradio | Gradio inspection UI |
| http://127.0.0.1:8000/api/docs | Interactive API documentation (OpenAPI) |

**First login:** on first start the server creates an administrator and four demo accounts
(quality manager, inspector, ML engineer, viewer) with random passwords and writes them to
`storage/INITIAL_CREDENTIALS.txt`. Change those passwords and delete the file. To choose the admin
password yourself, set `QG_DEFAULT_ADMIN_PASSWORD` (see `.env.example`).

**Frontend development** (hot reload; it proxies `/api` to port 8000):

```bash
cd frontend && npm run dev     # http://localhost:5173
```

**Gradio only**, as in the proposal:

```bash
python scripts/run_gradio.py   # http://127.0.0.1:7860
```

**Demo set** (held-out test images, unfamiliar inputs and a short video; see [docs/DEMO_GUIDE.md](docs/DEMO_GUIDE.md)):

```bash
python scripts/make_demo_set.py   # -> data/samples/
```

**Production database:** set `QG_DATABASE_URL=postgresql+psycopg://...` and `QG_SCHEMA_MODE=migrations` (Alembic).
Back up with `python scripts/backup.py`.

**Docker** (files provided but not yet tested; Docker was not available on the development machine):

```bash
docker compose up --build
```

This mounts `artifacts/`, `data/` and `storage/` from the host.

## Accounts and inspection access

1. **Create an account:** on the sign-in page choose *No account? Create one* and enter an email, a username
   (3–32 letters, digits, `.`, `_`, `-`) and a password (≥ 10 characters with upper- and lower-case letters and a digit).
   New accounts get the **Viewer** role: they can see results but cannot run inspections.
2. **Request access:** after signing in (with the email *or* the username), the user clicks
   **Request inspection access** in the banner and can add a reason. One pending request at a time.
3. **Approve or deny:** administrators see a badge on **Administration** and a notification. Under
   *Administration → Access requests* they approve or deny each request, with an optional note. Approving
   gives the **Inspector** role (view, inspect, review, reports, alerts). It never lowers the role of someone who
   already has more rights. A denied user sees the note and can send a new request.
4. The user's screen updates live; the *Inspect* menu appears as soon as the request is approved.

Every request and decision is written to the audit trail. To switch self-registration off, set
`QG_ALLOW_REGISTRATION=false`. API: `POST /api/auth/register`, `POST /api/access-requests`,
`GET /api/access-requests/mine`, `GET /api/access-requests?status=pending` (admin),
`POST /api/access-requests/{id}/decision` with `{"action": "approve"|"deny", "note": "..."}` (admin).

**Schema change:** this added `users.username` and the `access_requests` table (Alembic revision `0002`). In the default
`create_all` mode, missing *nullable* columns are added to existing tables automatically at start-up. Nothing is
dropped or changed, and existing accounts keep signing in with their email.

## Using the API

```bash
# with a JWT from POST /api/auth/login, or an API key created under Administration -> API keys
curl -H "X-API-Key: qg_..." -F file=@part.png http://127.0.0.1:8000/api/inspect          # category auto-detected
curl -H "X-API-Key: qg_..." -F file=@part.png -F profile_id=bottle http://127.0.0.1:8000/api/inspect
curl -H "X-API-Key: qg_..." http://127.0.0.1:8000/api/inspections/<id>/report -o report.pdf
```

Short endpoints: `POST /api/inspect`, `POST /api/inspect/batch`, `GET /api/inspections` (filters incl.
`verdict=NORMAL|DEFECTIVE|MANUAL INSPECTION REQUIRED`), `GET /api/inspections/{id}`, `GET /api/inspections/{id}/report`,
`GET /api/models`, `GET /api/models/{id}/metrics`, `GET /api/dashboard/summary`, `GET /api/recognizer`, `GET /api/health`.
Every inspection response carries `verdict` (the primary result) and `status` (the profile's line disposition).
If a model checkpoint is missing, the API returns **503 Model unavailable** with the checkpoint path and never a
substitute prediction.

The main endpoint groups are `auth`, `users`, `inspections` (single, batch, video, compare-reference,
export), `reviews`, `alerts`, `batches`, `jobs`, `simulation`, `analytics`, `models`, `training`,
`datasets`, `profiles`, `knowledge`, `reports`, `drift`, `assistant`, `settings` and `system`.
Live events stream over the WebSocket at `/api/ws/events?token=<JWT>`.

## Tests

```bash
python -m pytest -m "not e2e" -q      # unit, API, workflow, migration and integration tests
python -m pytest -m e2e -q            # browser end-to-end (incl. register -> request -> admin approval) (Playwright; needs `npm run build` and Chrome)
```

- **API and workflow tests:** these use an isolated temporary database and storage, and run real inspections
  with the trained models. They cover single, batch and video inspection; golden-reference comparison; drift;
  reviews and annotations; feedback export; reports; permissions; the WebSocket; and the Gradio handlers.
- **ML unit tests:** rules, verdict mapping, validation (incl. blur), component recogniser, U-Net shapes, segmentation, splitting, metrics and calibration.
- **Recognition and verdict tests** (`tests/test_recognition_and_verdict.py`): auto-recognition with the real models, unsupported photo → MANUAL, category mismatch blocked, corrupted / blurred / low-resolution uploads → MANUAL, commutator U-Net inspection, short endpoints.
- **Migration tests:** the Alembic revisions build exactly the ORM schema, then downgrade.
- **Integration tests:** line-signal mapping, MQTT topics and payloads, webhook filtering.
- **End-to-end tests:** a real server and headless Chrome. Landing page and login; every page renders without
  JavaScript errors; upload → inspection → PDF → XAI lab; a camera capture with Chrome's fake camera.

Tests that need datasets or trained weights skip themselves when those are absent (as in CI).

## Measuring performance

```bash
python scripts/benchmark.py        # latency, throughput, memory, load time -> artifacts/metrics/benchmark.md
python scripts/export_onnx.py      # ONNX FP32 / INT8 accuracy, size and latency -> artifacts/metrics/edge_export.md
```

## Roles

| Role | Can |
|---|---|
| Administrator | everything |
| Quality Manager | inspect, review, reports, export, alerts, edit profiles, approve and deploy models, view logs |
| Inspector | inspect, review, reports, alerts |
| ML Engineer | inspect, reports, export, train, manage models and datasets, API keys, view logs |
| Viewer | read-only dashboards and inspections |

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): system design, inference pipeline and data model
- [docs/QUALITY_RULES.md](docs/QUALITY_RULES.md): how severity, decisions and the quality index are computed
- [docs/DATASETS.md](docs/DATASETS.md): datasets, licences, preprocessing and split policy
- [docs/RESULTS.md](docs/RESULTS.md): model comparison, evaluation, benchmark and edge-export results
- [docs/FEATURE_STATUS.md](docs/FEATURE_STATUS.md): what is implemented, tested, partial or missing, and why
- [docs/DEMO_GUIDE.md](docs/DEMO_GUIDE.md): a scripted demonstration
- [docs/API.md](docs/API.md): endpoint reference generated from the OpenAPI schema
- [docs/DEVELOPMENT_LOG.md](docs/DEVELOPMENT_LOG.md): completed work, failures and fixes
- [docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md): final project report
- [docs/proposal/Proposal_Report.pdf](docs/proposal/Proposal_Report.pdf): original project proposal

## Limitations

- **Research datasets:** NEU-CLS and MVTec AD are research benchmarks with clean, well-lit images. Performance on a real production line must be re-validated on that line's own images.
- **NEU-CLS has no defect-free class:** the steel profile can name *which* defect is present but cannot certify that a surface is defect-free.
- **Example rules:** the severity bands and acceptance rules in `configs/profiles` are illustrative. Replace them with the manufacturer's validated criteria.
- **Licences:** MVTec AD is licensed CC BY-NC-SA 4.0 (non-commercial use only).
- **Not certified:** the system is not certified for production release decisions, and its industrial
  integrations (MQTT, webhook) have not been tested against physical line equipment.

## Team

Priyanshu Aryan Panda · Swayamjit Sahoo · Sakshi Sahoo · Biswajeet Behera · Lohitakhya Sahu · Srutiman Mohapatra
