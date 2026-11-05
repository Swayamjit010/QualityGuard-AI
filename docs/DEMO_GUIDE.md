# Demonstration guide

A 15–20 minute walkthrough that shows every major capability with real models and held-out images.
All inputs come from `data/samples/`, which `python scripts/make_demo_set.py` builds from the **test splits**,
so none of them was seen in training. Ground truth is in each file name and in `data/samples/README.md`.

## 0. Before the demo (once)

```bash
pip install -r requirements.txt
python scripts/download_datasets.py --neu --mvtec bottle metal_nut
python scripts/prepare_datasets.py
python scripts/train_classifiers.py --archs mobilenet_v2 efficientnet_b0 --epochs 8   # the deployed steel ensemble
python scripts/train_anomaly.py bottle metal_nut
python scripts/make_demo_set.py
cd frontend && npm install && npm run build && cd ..
uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Sign-in details for the five demo roles are in `storage/INITIAL_CREDENTIALS.txt` (created on first start).
Do one inspection per profile before presenting: the first inspection loads the models (a few seconds),
and later ones take about 0.3–1 s.

To show non-empty dashboards and analytics, run **Line simulation** for a minute beforehand. Simulated records
are labelled as such everywhere.

## 1. Landing page and roles (1 min)

- Open `http://127.0.0.1:8000`. The public landing page explains the hybrid approach. Click **Sign in**.
- Sign in as the **Inspector** first. The sidebar shows only the pages that role is allowed to use.
- Mention that there are five roles and 15 permissions, enforced on every API route, not only hidden in the UI.

## 2. Known defects on steel: classification and Grad-CAM (3 min)

1. **Inspect** → profile *Hot-rolled steel strip surface* → drop `data/samples/steel_strip/scratches__*.bmp`.
2. Tick Grad-CAM++ as well, then **Run inspection**.
3. Point out:
   - the predicted class and its calibrated confidence;
   - the two ensemble members (MobileNetV2, EfficientNet-B0) agreeing;
   - the Grad-CAM heat-map following the scratch.
4. Make the key distinction: *the heat-map shows where the model looked; it is not a defect boundary.*
5. Click **Open full record**, then **Open in XAI lab**:
   - the side-by-side viewer;
   - the per-model probability chart;
   - the out-of-distribution scores, each against its threshold.

## 3. Unseen defects: anomaly detection and localisation (3 min)

1. Profile *Glass bottle* → `data/samples/bottle/broken_large__*.png`.
2. Result: **FAIL**. Show:
   - the PatchCore and PaDiM anomaly maps;
   - the estimated defect region and the region boxes;
   - the zoom on the largest region, with its pixel area.
3. The anomaly models were trained **only on defect-free bottles**. They flag any deviation, including defect
   types nobody labelled.
4. Inspect `bottle/good__*.png` → **PASS**.
5. Open **Anomaly center**:
   - the deployed thresholds;
   - held-out AUROC;
   - the score distribution of the inspections so far;
   - defect-free reference images next to known defects.

## 4. Knowing what it does not know (2 min)

- Inspect `unfamiliar/bottle_for_steel_profile.png` with the **steel** profile. The result is **INCONCLUSIVE –
  unfamiliar pattern, manual review required**: a bottle is not forced into one of the six steel classes.
- This is the main safety property. High softmax confidence is not trusted on its own; the feature-space
  distance (Mahalanobis), energy score and member agreement are checked as well.

## 5. Human in the loop (3 min)

1. **Review queue**. The default order is *highest learning value first*. Each card shows why it ranks where
   it does: uncertainty, disagreement, borderline score, unfamiliar input, or a rare class.
2. Open an item:
   - confirm or correct the class and severity;
   - set the final decision;
   - add a comment;
   - under **Inspector annotations**, upload a defect mask or enter boxes.
3. The review resolves the related alert. **Datasets → Export reviewed feedback** turns reviews (including masks
   and boxes) into a curated retraining set.
4. Mention the retraining path: export → training job → evaluation → comparison → **approval** → deployment as a
   new profile revision. Nothing is ever replaced silently, and rollback is one click.

## 5b. Add your own product (3 min, plus training time)

1. Open **Products → + Add a product**. Name it, then drop 30 photos of *good* items. Optionally add a few
   defective ones; these are only used to report the detection rate.
2. Training runs in the background with a progress bar, typically 2–3 minutes on a laptop CPU.
3. The results card shows how many good photos would be wrongly flagged, how many supplied defects are caught,
   and whether other products are recognised as unfamiliar.
4. **Approve & activate** (needs a quality manager or administrator), then **Inspect an item**. A photo of a
   different kind of item is reported as *unfamiliar* instead of being judged.

Ask the assistant *"how do I add a new product?"* to show that the help is built in.

## 6. Production modes (2 min)

- **Batch / folder:** select the whole `data/samples/bottle` folder. Show the progress, the per-image results,
  the summary, the CSV export and the batch PDF.
- **Video:** `data/samples/video/bottle_line.avi`, sampling every 1 s. Show the timestamped events and the
  annotated video.
- **Camera:** pick a webcam and start it, then use **Capture & inspect** or continuous mode. Session counters and
  throughput update live, and a FAIL raises a toast.
- **Line simulation:** products arrive at a configurable interval. PASS / REJECT / HOLD signals are shown as a
  reject gate would receive them, together with throughput and latency.
- **Golden reference:** compare `bottle/good__…` against `bottle/broken_large__…`. Registration plus a
  difference map highlights the changed regions.

## 7. Quality intelligence (2 min)

- **Dashboard:** KPIs, trend, defect distribution, recent alerts and generated insights. Every number is a
  database aggregate.
- **Analytics:**
  - filters for profile, period and simulated data;
  - false-positive and false-negative rates where ground truth exists;
  - the forecast, which declines to forecast with fewer than 14 days of history;
  - **Drift monitoring → Run check**.
- **Assistant:** ask *"Which defect is most frequent this week?"* or *"Why was the last bottle rejected?"*.
  Answers come from tool calls over the database, and the tools used are shown.

## 8. Models, MLOps and reports (2 min)

- Sign in as the **ML Engineer** and open **Models & training**:
  - the registry and the four deployed checkpoints;
  - each run's metrics, confusion matrix, ROC/PR curves and history;
  - comparing versions;
  - starting a short training job and watching its log.
- Sign in as the **Quality Manager**: approve a validated model and deploy it to a profile, which creates a
  revision. Then show **Product profiles → history / rollback**.
- **Reports:** generate a daily quality report and a model comparison report, and open the inspection PDF from
  step 3.

## 9. Integration and operations (1 min)

- **System health:** API, database latency, each checkpoint on disk or in memory, CPU and memory, integration
  adapters, recent errors.
- **API for line equipment:** create an API key under **Administration → API keys**, then:

  ```bash
  curl -H "X-API-Key: qg_..." -F file=@data/samples/bottle/good__000.png -F profile_id=bottle \
       http://127.0.0.1:8000/api/inspections
  ```

- `/api/docs` is the interactive API reference. `/gradio` is the proposal's lightweight UI, using the same
  accounts.

## Talking points on limitations (say them)

- **Benchmarks are cleaner than factory images.** NEU-CLS and MVTec are clean benchmarks, so near-perfect
  scores here must be re-validated on a real line's own images before any production use.
- **Severity and acceptance rules are placeholders.** The manufacturer configures the real ones.
- **No certification.** It is an AI-assisted tool. Uncertain cases always go to a person.
- **Hardware integrations are untested.** Webhook and MQTT line signals exist, but none has been tested against
  physical PLCs or cameras.
