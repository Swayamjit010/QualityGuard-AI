# Architecture

```
            ┌────────────── Browser ──────────────┐
            │ React dashboard (/)   Gradio UI (/gradio) │
            └──────────┬───────────────┬──────────┘
               REST + WebSocket        │ (same service layer)
┌──────────────────────▼───────────────▼────────────────────────────┐
│ FastAPI (backend/app)                                               │
│  api/        auth · inspections · operations · analytics · models   │
│              catalog (datasets/profiles/knowledge) · reports · system│
│  security/   JWT + API keys · RBAC (5 roles / 15 permissions) · rate limit │
│  services/   inspection_service · alerts · analytics · drift · reports (PDF) │
│              model_registry · datasets · profiles · similarity · assistant │
│              storage (path-safe) · events (pub/sub → WebSocket, webhook) · audit │
│  workers/    thread-pool jobs (batch, video, simulation) · training subprocesses │
│  integrations/ webhook · MQTT line signals (PASS/REJECT/HOLD) · adapter status    │
└───────┬──────────────────────┬───────────────────────────┬────────┘
        │ SQLAlchemy           │ ml.inference.pipeline      │ files
   SQLite / PostgreSQL     ┌───▼─────────────────────┐   storage/ (uploads, visualisations,
   (32 tables)             │ ML library (ml/)         │            reports, videos)
                           │  classification engine   │   artifacts/ (checkpoints, metrics, plots)
                           │  anomaly engine          │   data/ (raw + versioned manifests)
                           │  explainability · rules  │
                           └──────────────────────────┘
```

## Inference pipeline (`ml/inference/pipeline.py`)

1. **Decode and validate** the image (`ml/inference/validation.py`): format, size, brightness, contrast and sharpness.
   If validation fails, the result is `INCONCLUSIVE` and no model runs.
2. **Engine A, classification** (when the profile lists classifiers):
   - Each classifier in the ensemble runs on the checkpoint's own preprocessing.
   - Probabilities are temperature-calibrated, then averaged across the ensemble.
   - Mahalanobis and energy out-of-distribution scores are computed.
   - Grad-CAM, Grad-CAM++ and Score-CAM are run on the predicted class, as requested.
3. **Engine B, anomaly detection** (when the profile has an anomaly model):
   - **PatchCore**: a coreset memory bank of layer-2/3 patch features, scored by nearest-neighbour distance.
   - **PaDiM**: a per-position Gaussian, scored by Mahalanobis distance.
   - Each engine has its own threshold, validated on held-out data.
   - The primary anomaly map is thresholded into a segmentation mask. Connected regions give the area, bounding box,
     centroid and peak score, in mm² when the profile is calibrated.
   - A global novelty check flags inputs that look like nothing in the defect-free references.
4. **Fusion and rules** (`ml/quality/rules.py`): severity, then decision, then quality index. See [QUALITY_RULES.md](QUALITY_RULES.md).
5. **Persistence** (`backend/app/services/inspection_service.py`): the upload, every visualisation PNG, the full JSON
   result, per-model predictions, anomaly and segmentation rows, severity, and an embedding for similarity
   search. Alert rules are then evaluated and an `inspection` event is published.

Models are loaded lazily and cached (`ModelCache`). A missing checkpoint raises `ModelUnavailable` (HTTP 503).
The system **never** substitutes a mock prediction.

## Training (`ml/training/trainer.py`, `ml/anomaly_detection/train.py`)

**Classifiers:**
- Built from ImageNet weights (`ml/architectures/registry.py`).
- The head is trained first with a frozen backbone (feature-extraction phase), then the whole network is fine-tuned.
- AdamW with cosine learning-rate decay, cross-entropy or focal loss, and early stopping on validation loss. Training is time-budgeted (`--max-minutes`) and resumable (`last.pt`).
- After training:
  - temperature scaling and out-of-distribution calibration on the validation split
  - a test-set report (accuracy, macro and weighted precision, recall and F1, ROC-AUC, PR-AUC, ECE, confusion matrix, worst-class recall)
  - a CPU latency, throughput and size benchmark
  - plots

**Anomaly models:**
- Fitted on defect-free training images.
- Thresholds are chosen on the validation split.
- Image and pixel AUROC, recall, FPR and Dice are measured on the test split.
- The two engines are fused into the PASS / FAIL / REVIEW rule.

All run outputs go to `artifacts/checkpoints/<run_id>/`. The registry (`services/model_registry.py`) imports them as
**validated** versions. Deployment requires **approval** by a role with `approve_models`. Deploying writes a new
profile revision, which makes rollback possible.

## Data model (main tables)

| Area | Tables |
|---|---|
| Identity | `users`, `roles`, `permissions`, `api_keys`, `revoked_tokens` |
| Product | `products`, `product_profiles`, `profile_revisions`, `production_lines`, `production_batches` |
| Data | `datasets`, `dataset_versions`, `images`, `image_annotations` |
| Models | `ml_models`, `model_versions`, `evaluation_metrics`, `training_jobs` |
| Inspection | `inspections`, `predictions`, `anomaly_results`, `segmentation_results`, `severity_assessments`, `human_feedback` |
| Operations | `alerts`, `reports`, `audit_events`, `jobs`, `system_settings`, `knowledge_base`, `drift_checks` |

SQLite is the default. Only PostgreSQL-compatible column types are used, so setting `QG_DATABASE_URL` is
enough to switch databases.

**Migrations.**
- **Development** (default): the schema is created with `create_all`.
- **Production:** set `QG_SCHEMA_MODE=migrations`, or run `alembic -c backend/alembic.ini upgrade head`, to apply
  the Alembic revisions in `backend/migrations/`.
- An existing `create_all` database is stamped at the initial revision first.
- `tests/test_migrations.py` proves that the migrations build exactly the ORM schema (`alembic check`) and that
  they downgrade cleanly.

**Backups:** `python scripts/backup.py` (see the script header for restore steps).

## Integration layer (`backend/app/integrations/`)

Outbound adapters subscribe to the in-process event bus and are optional. A failing adapter is logged and
never affects an inspection.

| Adapter | Enabled by | Output |
|---|---|---|
| WebSocket | always | `/api/ws/events`: live events for the dashboard |
| REST API + API keys | always | camera triggers or a manufacturing execution system post images to `POST /api/inspections` |
| Webhook | `QG_WEBHOOK_URL`, `QG_WEBHOOK_EVENTS` | JSON POST per event; inspection events carry a `line_signal` |
| MQTT | `QG_MQTT_HOST` (+ `pip install paho-mqtt`) | `<prefix>/<line>/signal` (PASS / REJECT / HOLD, QoS 1), `<prefix>/alerts` |
| OPC UA, Modbus TCP | — | not implemented (no hardware to validate against) |

The line-signal mapping is fixed: PASS → PASS, FAIL → REJECT, REVIEW REQUIRED or INCONCLUSIVE → HOLD (divert to
manual inspection). Line logic therefore never interprets AI scores. `GET /api/integrations` and the System
health page report each adapter's state and delivery counts.

## Human-in-the-loop data flow

1. A pending review enters the queue, ordered by an **active-learning priority**. The priority is the sum of
   five terms, each between 0 and 1:
   - uncertainty (1 − calibrated confidence);
   - model or engine disagreement;
   - borderline anomaly score;
   - unfamiliar input;
   - rarity of the predicted class among already-verified labels.
2. The inspector's feedback and annotations (mask PNG, boxes, polygon) are stored against the image record.
3. `POST /api/datasets/feedback-export` writes the verified images with labels, masks and boxes to
   `data/feedback/<ts>/`.
4. Training job → evaluation → registry (*validated*) → approval by `approve_models` → deployment as a new profile
   revision. Rollback goes back to any earlier revision.

## Security

- **Passwords:** hashed with bcrypt (cost 12) and subject to a strength policy.
- **Sessions:** JWTs (HS256) carry a `jti`. Logout revokes the token server-side.
- **API keys:** hashed, shown only once, and revocable. Requests made with a key are recorded with source `api`.
- **Access control:**
  - Permission checks run on every route.
  - Self-registration only creates viewer accounts.
  - Admins cannot demote themselves.
- **Rate limiting:** per user and IP; login is limited separately.
- **Uploads:** extension allow-list, size limits and decode validation. Storage keys are generated server-side and
  resolved paths are checked against the storage root. Dataset images are served only if they appear in the manifest.
- **HTTP headers:** `nosniff`, `X-Frame-Options`, `Referrer-Policy`.
- **Audit:** logins, inspections, reviews, model and profile changes, exports and settings changes are all audited.
- **Production:** set `QG_SECRET_KEY`, use PostgreSQL, run behind HTTPS, and replace the in-memory rate limiter
  and event bus with Redis when running more than one process.
