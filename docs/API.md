# API reference

99 operations in 9 groups.

Generated from the live OpenAPI schema by `python scripts/export_openapi.py`; the machine-readable
schema is [api/openapi.json](api/openapi.json) and the interactive explorer is served at `/api/docs`.

**Authentication.** `POST /api/auth/login` returns a JWT; send it as `Authorization: Bearer <token>`.
Machine clients use an API key (Administration → API keys) in the `X-API-Key` header. Browser downloads
and the WebSocket accept `?token=`. Every route except `/api/health`, `/api/auth/login` and
`/api/auth/register` requires authentication, and each checks a permission (see ARCHITECTURE.md).

**Errors** are JSON `{"detail": ...}` with the usual status codes: 401 (not signed in), 403 (missing
permission), 404, 413 (upload too large), 415 (unsupported file type), 422 (validation), 429 (rate limit),
503 (a required model checkpoint is missing - never replaced by a mock).

## analytics

| Method | Path | Description | Query parameters | Body |
|---|---|---|---|---|
| GET | `/api/analytics/anomaly-overview` | Anomaly Detection Center: the deployed thresholds and validated metrics of the profile's anomaly model | profile_id, days |  |
| GET | `/api/analytics/dashboard` | Dashboard | days, start, end, profile_id, defect_class, severity, model, status, include_simulated, source |  |
| GET | `/api/analytics/defects` | Defects | days, start, end, profile_id, defect_class, severity, model, status, include_simulated, source |  |
| GET | `/api/analytics/feedback` | Feedback |  |  |
| GET | `/api/analytics/forecast` | Forecast | horizon, days, start, end, profile_id, defect_class, severity, model, status, include_simulated, source |  |
| GET | `/api/analytics/insights` | Insights | days, start, end, profile_id, defect_class, severity, model, status, include_simulated, source |  |
| GET | `/api/analytics/models` | Model Distribution | days, start, end, profile_id, defect_class, severity, model, status, include_simulated, source |  |
| GET | `/api/analytics/quality` | Quality | days, start, end, profile_id, defect_class, severity, model, status, include_simulated, source |  |
| GET | `/api/analytics/trend` | Trend | unit, days, start, end, profile_id, defect_class, severity, model, status, include_simulated, source |  |

## auth & users

| Method | Path | Description | Query parameters | Body |
|---|---|---|---|---|
| GET | `/api/api-keys` | List Keys |  |  |
| POST | `/api/api-keys` | Create Key |  | JSON |
| DELETE | `/api/api-keys/{kid}` | Revoke Key |  |  |
| POST | `/api/auth/login` | Login |  | JSON |
| POST | `/api/auth/logout` | Logout |  |  |
| GET | `/api/auth/me` | Me |  |  |
| POST | `/api/auth/me/password` | Change Password |  | JSON |
| PUT | `/api/auth/me/preferences` | Prefs |  | JSON |
| POST | `/api/auth/register` | Register |  | JSON |
| GET | `/api/roles` | Roles |  |  |
| GET | `/api/users` | List Users |  |  |
| POST | `/api/users` | Create User |  | JSON |
| DELETE | `/api/users/{uid}` | Deactivate User |  |  |
| PUT | `/api/users/{uid}` | Update User |  | JSON |

## datasets, profiles & knowledge

| Method | Path | Description | Query parameters | Body |
|---|---|---|---|---|
| GET | `/api/datasets` | List Datasets |  |  |
| POST | `/api/datasets/feedback-export` | Feedback Export | only_new |  |
| POST | `/api/datasets/import` | Import Dataset |  | multipart |
| GET | `/api/datasets/{name}` | Get Dataset |  |  |
| POST | `/api/datasets/{name}/analyze` | Analyze |  | JSON |
| GET | `/api/datasets/{name}/image` | Sample Image | path |  |
| GET | `/api/datasets/{name}/samples` | Samples | split, label, page, page_size |  |
| GET | `/api/knowledge` | List Kb | q, defect_class |  |
| POST | `/api/knowledge` | Create Kb |  | JSON |
| PUT | `/api/knowledge/{kid}` | Update Kb |  | JSON |
| GET | `/api/profiles` | List Profiles |  |  |
| GET | `/api/profiles/{pid}` | Get Profile Ep |  |  |
| PUT | `/api/profiles/{pid}` | Put Profile |  | JSON |

## inspections

| Method | Path | Description | Query parameters | Body |
|---|---|---|---|---|
| GET | `/api/inspections` | List Inspections | page, page_size, q, profile_id, status, defect_class, severity, source, batch_id, review_status, simulated, date_from, date_to |  |
| POST | `/api/inspections` | Create |  | multipart |
| GET | `/api/inspections/annotations/{aid}/mask` | Annotation Mask |  |  |
| POST | `/api/inspections/batch` | Batch and folder inspection (the browser sends a selected folder's files with source=folder). |  | multipart |
| POST | `/api/inspections/compare-reference` | Compare Reference |  | multipart |
| GET | `/api/inspections/export` | Export | fmt, profile_id, status, batch_id, days |  |
| POST | `/api/inspections/video` | Video |  | multipart |
| DELETE | `/api/inspections/{iid}` | Delete |  |  |
| GET | `/api/inspections/{iid}` | Get |  |  |
| POST | `/api/inspections/{iid}/annotations` | Attach an inspector annotation to the inspected image: a binary defect ``mask`` (PNG, same aspect as the |  | multipart |
| DELETE | `/api/inspections/{iid}/annotations/{aid}` | Delete Annotation |  |  |
| GET | `/api/inspections/{iid}/images` | Images |  |  |
| GET | `/api/inspections/{iid}/images/{key}` | Image |  |  |
| GET | `/api/inspections/{iid}/report` | Report |  |  |
| GET | `/api/inspections/{iid}/similar` | Similar | k, only_defective |  |
| GET | `/api/inspections/{iid}/visualizations` | Visualizations |  |  |

## models & training

| Method | Path | Description | Query parameters | Body |
|---|---|---|---|---|
| GET | `/api/models` | List Models |  |  |
| GET | `/api/models/architectures` | Architectures |  |  |
| POST | `/api/models/rollback` | Rollback |  | JSON |
| POST | `/api/models/sync` | Sync |  |  |
| GET | `/api/models/{mid}` | Get Model |  |  |
| POST | `/api/models/{mid}/activate` | Deploy an APPROVED model version to a product profile (records a profile revision for rollback). |  | JSON |
| POST | `/api/models/{mid}/approve` | Approve |  | JSON |
| GET | `/api/models/{mid}/metrics` | Metrics |  |  |
| GET | `/api/models/{mid}/plots/{name}` | Plot |  |  |
| POST | `/api/models/{mid}/retire` | Retire |  | JSON |
| GET | `/api/profiles/{pid}/revisions` | Revisions |  |  |
| GET | `/api/training/jobs` | Jobs |  |  |
| GET | `/api/training/jobs/{jid}` | Job |  |  |
| POST | `/api/training/jobs/{jid}/stop` | Stop |  |  |
| POST | `/api/training/start` | Start |  | JSON |

## operations

| Method | Path | Description | Query parameters | Body |
|---|---|---|---|---|
| GET | `/api/alerts` | List Alerts | status, level, kind, limit |  |
| POST | `/api/alerts/{aid}/{action}` | Alert Action |  |  |
| GET | `/api/batches` | List Batches | limit |  |
| GET | `/api/batches/{bid}` | Get Batch |  |  |
| GET | `/api/batches/{bid}/video` | Batch Video |  |  |
| POST | `/api/inspections/{iid}/feedback` | Submit Feedback |  | JSON |
| GET | `/api/jobs` | List Jobs | kind, limit |  |
| GET | `/api/jobs/{jid}` | Get Job |  |  |
| POST | `/api/jobs/{jid}/stop` | Stop Job |  |  |
| GET | `/api/reviews/queue` | Pending reviews. ``order=priority`` ranks by active-learning value (see review_priority); | page, page_size, profile_id, order |  |
| POST | `/api/simulation/start` | Start Simulation |  | JSON |

## products

| Method | Path | Description | Query parameters | Body |
|---|---|---|---|---|
| GET | `/api/products` | List Products |  |  |
| POST | `/api/products` | Create a product from photos of good items (>= 10, ideally 30+) and optional defective photos, then start |  | multipart |
| GET | `/api/products/{slug}` | Get Product |  |  |
| POST | `/api/products/{slug}/activate` | Approve the latest trained model of a custom product and make the product available for inspection |  |  |
| GET | `/api/products/{slug}/image/{kind}/{name}` | Product Image |  |  |
| POST | `/api/products/{slug}/photos` | Add more photos and retrain as a new model version (the active version keeps serving until the new one is |  | multipart |

## reports

| Method | Path | Description | Query parameters | Body |
|---|---|---|---|---|
| GET | `/api/reports` | List Reports | kind, limit |  |
| POST | `/api/reports/generate` | Generate |  | JSON |
| GET | `/api/reports/{rid}/download` | Download |  |  |

## system

| Method | Path | Description | Query parameters | Body |
|---|---|---|---|---|
| POST | `/api/assistant/chat` | Chat |  | JSON |
| GET | `/api/assistant/status` | Assistant Status |  |  |
| POST | `/api/drift/check` | Drift Check | profile_id, window |  |
| GET | `/api/drift/checks` | Drift Checks | profile_id, limit |  |
| GET | `/api/health` | Unauthenticated liveness/readiness probe. |  |  |
| GET | `/api/integrations` | Status of outbound adapters (WebSocket, REST, webhook, MQTT; OPC UA / Modbus reported as not implemented). |  |  |
| GET | `/api/settings` | Get Settings Ep |  |  |
| PUT | `/api/settings/{key}` | Put Setting |  | JSON |
| GET | `/api/system/audit` | Audit Log | action, user_email, page, page_size |  |
| GET | `/api/system/info` | Info |  |  |
| GET | `/api/system/logs` | Logs | level, q, limit |  |
| GET | `/api/system/status` | System health page: API, database, models, compute resources, storage and recent errors. |  |  |
