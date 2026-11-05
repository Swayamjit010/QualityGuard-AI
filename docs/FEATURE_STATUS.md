# Feature status

Status of every major requirement of the extended specification (the "master prompt") and the original
proposal. A feature is only marked **Tested** when an automated test or a recorded manual run exercised it;
having code for it is not enough.

| Status | Meaning |
|---|---|
| ✅ **Tested** | Implemented and covered by automated tests (`pytest`) or a scripted run whose output is recorded in `docs/` |
| ☑️ **Verified manually** | Implemented and exercised by hand or in the headless-browser smoke test, without a dedicated assertion |
| 🟡 **Partial** | Some of the requirement is implemented; the gap is stated |
| ⛔ **Blocked** | Needs hardware, a service or data that is not available here |
| ❌ **Not implemented** | Deliberately left out; the reason is stated |

Last updated: 2026-11-05 (extension: verdicts, component recognition, KolektorSDD2, screw, grid, U-Net segmentation). Current result: **111 / 111 tests pass** (102 non-browser + 9 Playwright). Test suite: `pytest -m "not e2e"` (unit + API + migrations + integrations) and
`pytest -m e2e` (Playwright, real server and browser).

## Machine learning

| Requirement | Status | Notes |
|---|---|---|
| Custom CNN baseline | ✅ | Trained from scratch; test results in [RESULTS.md](RESULTS.md) |
| MobileNetV2, EfficientNet-B0, ResNet-50 | ✅ | Trained and evaluated on the identical NEU-CLS test split |
| MobileNetV3-Small (edge), ViT-Tiny, DenseNet-121, ConvNeXt-Tiny | ✅ | Trained 2026-11-05 with the same pipeline. All ended by early stopping, none by the time budget. ConvNeXt-Tiny 100 %; the others 99.6 % (RESULTS.md) |
| EfficientNet-B3 | ❌ | In the registry (`ml/architectures/registry.py`) but not trained: 300 px input and ~2.5× B0's cost on a CPU-only host, with no accuracy headroom on NEU-CLS (all pre-trained models already ≥ 99.6 %) |
| ViT-B/16 | ❌ | 330 MB of weights; ViT-Tiny is used as the transformer representative within the CPU budget |
| Transfer learning: frozen-backbone phase, then fine-tuning; AdamW, cosine LR, early stopping, class weights, label smoothing, focal loss option, seeds, resumable checkpoints | ✅ | `ml/training/trainer.py`; resume exercised for real when ResNet-50 was interrupted by low memory |
| GPU detection / CPU fallback | ☑️ | `ml/config.device_info()`; only CPU available here |
| Metrics: accuracy, precision, recall, F1 (macro and weighted), confusion matrix, ROC-AUC, PR-AUC, ECE, per-class, latency, size, training time | ✅ | `ml/evaluation/metrics.py`, unit-tested; per-run plots in `artifacts/checkpoints/<run>/` |
| Memory consumption | ✅ | `scripts/benchmark.py` measures RSS increase on load and peak RSS during inference |
| Model comparison report | ✅ | `scripts/compare_models.py` → RESULTS.md tables and chart; PDF comparison report in the Report Center |
| MLflow experiment tracking | ☑️ | Each training run logs params/metrics/artifacts to `artifacts/experiments/mlflow.db` when MLflow is installed (`mlflow ui --backend-store-uri sqlite:///artifacts/experiments/mlflow.db`) |
| PatchCore + PaDiM anomaly detection | ✅ | Trained on MVTec bottle and metal_nut; image and pixel AUROC in RESULTS.md |
| Unfamiliar-input detection (Mahalanobis, energy, global kNN novelty) | ✅ | Measured on 60 foreign images per category; a bottle image sent to the steel profile → INCONCLUSIVE |
| Temperature-scaling calibration | ✅ | Unit test shows ECE reduction |
| Validated decision fusion (no averaging of unrelated scores) | ✅ | Classifier ensemble = mean of calibrated probabilities over the same label space; anomaly engines fused by per-engine validated thresholds + review margin; disagreement → REVIEW |
| Grad-CAM, Grad-CAM++, Score-CAM | ✅ | `ml/explainability/cam.py`; produced from the live model on every request |
| Defect localisation + measurements (regions, area, ratio, bbox, contour area, perimeter, aspect ratio, centroid, density) | ✅ | `ml/segmentation/regions.py`; mm² only when the profile has a calibrated scale |
| Supervised segmentation | ✅ | U-Net (ResNet-18 encoder) trained on KolektorSDD2 pixel masks: test image AUROC 0.972, Dice 0.640 (RESULTS.md §10.3). Stopped after 7 of 14 planned epochs to save time. DeepLabV3+, YOLO-seg, Mask R-CNN not trained |
| Object detection | ❌ | No dataset here has box annotations for multiple defects per image; regions come from anomaly maps |
| Supervised severity prediction | ❌ | No dataset with severity labels exists; severity is rule-based and configurable (QUALITY_RULES.md) |
| Edge export (ONNX, INT8) | ✅ | `scripts/export_onnx.py`; accuracy/latency/size measured per variant (RESULTS.md §Edge) |

## Extension 2026-11-05 (specification: NORMAL / DEFECTIVE / MANUAL INSPECTION REQUIRED)

| Requirement | Status | Notes |
|---|---|---|
| Primary output NORMAL / DEFECTIVE / MANUAL INSPECTION REQUIRED | ✅ | `ml/quality/rules.verdict_for`; derived from the stored disposition, so it can never drift. Shown on the Inspection page, dashboard, history (filterable), detail page and PDF. Unit + API + browser tests |
| Automatic component recognition; unsupported images -> MANUAL | ✅ | `ml/recognition/router.py`; 100 % correct routing on 1,730 test images across 12 categories, 100 % rejection of 420 unsupported probe images (RESULTS.md §10.5). An explicitly selected category that does not match the image is also blocked. Recognises only the 12 trained categories; studio/web photos of parts are rejected as unsupported |
| Findings of a rejected (unsupported) image are not recorded as a result | ✅ | Kept as `reference_only_findings`; found and fixed during the live check |
| Blur / unsuitable-image detection | ✅ | Per-category limit calibrated on train + val; 0 % false rejects and 100 % detection of σ4 blur on test (RESULTS.md §10.7); weaker for mild blur on steel |
| Cross-engine disagreement -> MANUAL; validation-selected voting engines | ✅ | `pipeline.voting_engines`, `decision_engine`; commutator policy chosen on validation (RESULTS.md §10.4) |
| End-to-end evaluation of the deployed decision path | ✅ | `scripts/evaluate_profiles.py` -> `reports/end_to_end_evaluation.md` (RESULTS.md §10.6) |
| `POST /api/inspect`, `POST /api/inspect/batch`, `GET /api/dashboard/summary`, `GET /api/recognizer` | ✅ | `backend/app/api/inspect.py`; reuse the existing services |
| Self-registration (email + username + password) -> inspection-access request -> admin approve / deny page | ✅ | `backend/app/api/access.py`, Administration → Access requests, banner for Viewers, live updates over WebSocket; API tests + a two-browser Playwright test. Approval = Inspector role |
| Dashboard counts | ✅ | Fixed: manual-review cases were previously counted as "normal" |
| Commutator surface (KSDD2) reliability | 🟡 | 10 of 110 defective test commutators auto-passed as NORMAL; needs human sampling (RESULTS.md §10.6) |
| Screw / grid manual-inspection rate | 🟡 | 0 missed defects on test, but every normal part goes to manual review (review margin); per-category margin tuning on validation not yet done |

## Data

| Requirement | Status | Notes |
|---|---|---|
| NEU-CLS, MVTec AD bottle + metal_nut downloaded, validated, analysed | ✅ | Sizes and licences in [DATASETS.md](DATASETS.md) |
| KolektorSDD2, MVTec AD screw + grid, MPDD (6 metal-part categories) | ✅ | Downloaded (KSDD2 from the official archive, size + CRC verified), analysed, split, trained (DATASETS.md, RESULTS.md §10) |
| MVTec LOCO, DAGM, KolektorSDD (v1), BTAD, VisA | 🟡 | Catalogued with licence and access notes; not downloaded: CPU-only host with ~16 GB free disk. LOCO/VisA target logical anomalies and PCB/food items rather than machinery surfaces |
| Corrupt / duplicate / quality checks, class balance, image sizes | ✅ | `ml/datasets/analysis.py`; dataset quality report per dataset |
| Group-aware stratified splits, versioned manifests, no test-set tuning | ✅ | Unit-tested (no group straddles splits) |
| ZIP import (class folders, MVTec layout, labels.csv / labels.json), zip-bomb and path-traversal protection | ✅ | API test covers upload validation; import tested manually |
| Image–mask pairing | ☑️ | MVTec `ground_truth/` masks paired automatically; inspector masks attached through the review UI |
| Preprocessing: resize, normalise, colour mode, CLAHE, denoise; augmentation (flip, rotate, zoom, translate, brightness/contrast, noise, cutout) | ✅ | `ml/preprocessing/transforms.py`; cutout disabled for NEU-CLS because it can erase the defect that defines the label |
| Dataset export of verified feedback (incl. inspector masks and boxes) | ✅ | `POST /api/datasets/feedback-export`, tested |

## Inspection and quality

| Requirement | Status | Notes |
|---|---|---|
| Single-image inspection with full pipeline | ✅ | API tests + Playwright workflow test with real models |
| Batch and folder inspection, CSV/JSON export, batch PDF | ✅ | `tests/test_workflows.py`: real batch job, unsupported file skipped, CSV export, batch PDF |
| Video inspection: frame sampling, timestamps, annotated video, summary | ✅ | `tests/test_workflows.py`: a 3 s clip sampled every 1 s → 3 inspections with timestamps 0/1/2 s and an annotated WebM |
| Live camera: device selection, preview, start/stop, capture, continuous mode, alerts, counters, throughput, history | ✅ | Playwright test with Chrome's fake camera device: start → capture → decision (the synthetic pattern is correctly *not* passed) → counters → stop. Not yet tried with a physical USB camera; industrial camera SDKs not implemented |
| IP / RTSP cameras | ❌ | Not implemented: no stream to validate against. OpenCV can open RTSP URLs, so a capture worker can be added on the video-job path |
| Production-line simulation (arrival interval, accept/reject/hold, throughput, latency; clearly labelled simulated) | ✅ | Runs real models on held-out images; simulated records flagged everywhere |
| API inspection with API keys | ✅ | Tested |
| Custom pipeline per product (engines, models, CAM methods) | ✅ | Product profiles + per-request options |
| Golden-reference comparison (registration + difference map) | ✅ | `ml/inference/reference.py`: ORB keypoints + RANSAC homography, LAB difference, regions. Tested: identical images → 0 regions; good vs broken bottle → regions found |
| Configurable severity (Low/Medium/High/Critical) and decision rules (PASS/FAIL/REVIEW/INCONCLUSIVE), versioned profiles with rollback | ✅ | Unit- and API-tested; rules are documented placeholders, not industrial standards |
| Quality index | ✅ | Documented, configurable formula; explicitly *not* a validated industrial score |
| Similar-defect retrieval | ✅ | Embedding cosine search, with a caveat that similarity ≠ same cause |
| Knowledge base + corrective-action suggestions (verified / unverified flag) | ☑️ | |

## Human in the loop and MLOps

| Requirement | Status | Notes |
|---|---|---|
| Review: confirm/reject, correct class and severity, accept/reject, comments, difficult flag, location confirmation | ✅ | |
| Upload supporting annotations (mask PNG, boxes, polygon, label) | ✅ | Validated against image size; included in the feedback export |
| Active-learning prioritisation (uncertainty, disagreement, borderline anomaly, unfamiliar, rare class) | ✅ | Transparent additive score with per-item reasons; review queue sorts by it |
| Controlled retraining: feedback → dataset → training job → evaluation → comparison → approval → deployment; rollback | ☑️ | Each step exists and is permission-gated; deployment writes a profile revision. A full loop on real feedback has not been run (no real line data) |
| Model registry and lifecycle (validated → approved → deployed → retired) | ✅ | |
| Drift monitoring (PSI on input statistics, prediction distribution, confidence, anomaly scores) | ✅ | Drift check + drift PDF tested; separates data drift from verified-performance change |

## Application

| Requirement | Status | Notes |
|---|---|---|
| React web app: landing, dashboard, inspect (single/camera/batch/folder/video/reference), inspections, detail, reviews, alerts, simulation, analytics (incl. forecast), anomaly center, XAI lab, reports, assistant, models & training, datasets, profiles, system health, administration (users, API keys, settings, audit, logs) | ✅ | All routes rendered by the Playwright test without JS errors |
| TypeScript, Tailwind CSS | 🟡 | The main **AI Inspection** page (`src/pages/InspectAI.tsx`, `src/lib/qg.ts`) is TypeScript (`strict`) + Tailwind (`tw-` prefix, no preflight, so it coexists with the existing CSS). `npm run build` type-checks first. The other pages are still JavaScript (JSX); Lucide icons not used |
| Dark/light theme, keyboard-reachable controls, labelled inputs | ☑️ | No formal WCAG audit |
| Localisation architecture | 🟡 | `src/i18n.js` with English; navigation and shell use it. Page bodies are not yet extracted into message keys |
| Gradio interface (original proposal) | ✅ | `/gradio`, DB-backed login; handlers unit-tested |
| Quality assistant with tool calls, grounded in DB queries; works without an LLM | ✅ | Rebuilt 2026-11-05. The offline intent engine extracts the period, product, status, defect class and inspection reference (including "the last bottle" and "it" from the previous answer), routes the question to one of 13 tools, and answers in Markdown with in-app links. Unknown questions are admitted, not padded with a summary. Covered by `tests/test_assistant.py` and a browser test. LLM mode (Claude Opus 5.5, manual tool loop, server-side refusal fallback) needs `QG_ANTHROPIC_API_KEY` and was not exercised here (no key) |
| Add a product (inspect new kinds of items from photos of good examples) | ✅ | Upload → out-of-fold-calibrated PatchCore + PaDiM + novelty → approve & activate → inspect. API lifecycle test (`tests/test_products.py`) and a held-out evaluation in RESULTS.md §9 (30 photos per product). Retraining with more photos creates a new version; the active version keeps serving until the new one is approved |
| Every page linked | ✅ | Clickable KPI tiles, cross-links between dashboard, analytics, products, profiles, models, system health, audit trail and records. A Playwright crawler follows every in-app link from every page and fails on any broken one |
| PDF reports: inspection, batch, period (daily/weekly/monthly), model evaluation, model comparison, dataset, drift, review; CSV/JSON exports | ✅ | |

## Platform, security, deployment

| Requirement | Status | Notes |
|---|---|---|
| FastAPI REST API, OpenAPI docs, WebSocket events | ✅ | [API.md](API.md) generated from the schema |
| 32-table schema, FKs, indexes, pagination, filtering | ✅ | |
| Alembic migrations | ✅ | `alembic upgrade head` builds the schema; `alembic check` proves no drift from the ORM (tested) |
| PostgreSQL | 🟡 | Only PostgreSQL-compatible types are used and migrations are dialect-neutral, but no PostgreSQL server was available to run the suite against |
| JWT with revocation, bcrypt, password policy, RBAC (5 roles / 15 permissions), API keys, rate limiting, upload validation, path-safe storage, security headers, audit trail, retention | ✅ | |
| Antivirus scanning of uploads | ❌ | Uploads are decoded and re-encoded rather than stored blindly; no AV engine integrated |
| Backups | ✅ | `scripts/backup.py`: SQLite online backup + integrity check (or `pg_dump`), storage archive, SHA-256 manifest |
| No third-party telemetry | ✅ | Gradio / Hugging Face analytics disabled by default |
| S3-compatible storage | 🟡 | Storage abstraction (`services/storage.py`) with a local backend only |
| Redis for jobs/cache | ❌ | Not needed for a single process; in-process thread pool and event bus. Required before scaling to several workers |
| Webhook and MQTT adapters (PASS / REJECT / HOLD line signals) | ✅ | Tested with stub clients; MQTT not yet validated against a broker on a real line |
| OPC UA, Modbus TCP, PLC | ⛔ | Need a server or PLC to validate against; not claimed |
| Docker / Docker Compose | 🟡 | `Dockerfile` and `docker-compose.yml` at the repository root, **untested**: Docker is not installed on the development machine |
| GitHub Actions CI | 🟡 | Workflow written; its test step was verified by running the suite on a copy containing only git-tracked files (no datasets, no weights). It has not run on GitHub yet |
| Playwright end-to-end tests | ✅ | `pytest -m e2e`: landing + login, every page, upload → inspection → PDF → XAI lab |
| Benchmarks (latency, throughput, memory, load time, API time) | ✅ | `scripts/benchmark.py` → `artifacts/metrics/benchmark.md` |

## Research directions (Part 26)

| Direction | Status | Outcome |
|---|---|---|
| Multi-model ensembles | ✅ | Deployed for steel (MobileNetV2 + EfficientNet-B0) with disagreement → review |
| Open-set / unfamiliar-input recognition | ✅ | Mahalanobis + energy (classifiers), global kNN novelty (anomaly profiles) |
| Explainable anomaly detection | ✅ | Pixel anomaly maps, regions and measurements |
| Similarity-based retrieval | ✅ | |
| Image registration / reference comparison | ✅ | |
| Quality trend forecasting | ☑️ | Holt–Winters exponential smoothing; refuses to forecast with < 14 days of history |
| Lightweight edge AI | ✅ | MobileNetV3-Small trained; ONNX FP32 / INT8 measured |
| Vision Transformers | ✅ | ViT-Tiny trained and compared |
| Few-shot, self-supervised / contrastive, zero-shot anomaly, test-time adaptation, continual learning, federated learning, knowledge distillation, diffusion / synthetic defect generation, domain adaptation, automated rule optimisation | ❌ | Not justified here: NEU-CLS is already saturated by supervised transfer learning, so none of them could show a measurable gain on the available data, and several (diffusion, federated) exceed the CPU-only budget. They are listed as future work with their data requirements in PROJECT_REPORT.md §8 |

## Remaining gaps (precise reasons)

1. **Supervised segmentation (U-Net etc.).** The only pixel-annotated data here are the MVTec defective
   images (bottle 63, metal_nut 93). They are already split into a validation set, used to calibrate the anomaly
   thresholds, and a test set, used once for the reported metrics. Training a segmentation network would need a
   third, disjoint split of a few dozen images. That is too few to train on, and it would shrink the anomaly test
   set. Localisation therefore comes from the anomaly maps, whose pixel AUROC (~0.98) and Dice are measured. A
   U-Net becomes worthwhile once inspector masks accumulate through the review UI: the feedback export already
   includes them.
2. **Docker, PostgreSQL and the GitHub Actions run** are written but unverified on this machine (no Docker, no
   PostgreSQL server, no pushed remote).
3. **Industrial connectivity** beyond webhook and MQTT (OPC UA, Modbus, camera SDKs, RTSP) needs hardware.
4. **Frontend stack deviation** (JavaScript instead of TypeScript, custom CSS instead of Tailwind).
5. **Additional public datasets** were catalogued but not downloaded.
