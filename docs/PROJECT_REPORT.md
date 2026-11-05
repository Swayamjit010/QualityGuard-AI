# QualityGuard AI — Project Report

*Deep Learning-Based Industrial Product Defect Detection and Quality Inspection System*
School of Computer Engineering, KIIT · Academic Year 2026–2027

## 1. Problem and aim

Manual visual inspection is slow, inconsistent between inspectors, and hard to scale. Small surface defects
are also easy to miss. QualityGuard AI automates image-based inspection: it classifies and localises
defects, reports a confidence score and a severity, explains each decision visually, and produces
inspection reports. Uncertain cases are routed to a human inspector. The system is an **AI-assisted** tool,
not a replacement for established quality-control procedures.

## 2. Proposal objectives → implementation

| # | Proposal objective | Implementation | Where |
|---|---|---|---|
| 1 | Collect and prepare an industrial defect dataset | NEU-CLS (steel, 6 classes) plus MVTec AD bottle and metal_nut (pixel masks); download and analysis scripts | `scripts/download_datasets.py`, `scripts/prepare_datasets.py`, [DATASETS.md](DATASETS.md) |
| 2 | Preprocessing: resizing, normalisation, augmentation, handling corrupted images, train/val/test split | Corrupt, duplicate and quality checks; stratified, duplicate-grouped, versioned splits; dataset-specific augmentation | `ml/datasets/`, `ml/preprocessing/transforms.py` |
| 3 | Baseline classification model | Custom CNN trained from scratch (stride-2 stem, 4 conv-BN-ReLU blocks, global pooling, MLP head) | `ml/architectures/registry.py` (`custom_cnn`) |
| 4 | Transfer learning with MobileNetV2, EfficientNet and ResNet | Feature-extraction phase (frozen backbone), then fine-tuning, AdamW, cosine learning rate, early stopping. Also trained for comparison: MobileNetV3-Small, DenseNet-121, ConvNeXt-Tiny, ViT-Tiny | `ml/training/trainer.py` |
| 5 | Compare models | Identical held-out test split for all models; accuracy, F1, worst-class recall, latency, size | `scripts/compare_models.py`, [RESULTS.md](RESULTS.md), UI **Models** page |
| 6 | Choose a model on performance and compute cost | Selection rationale in [RESULTS.md](RESULTS.md) §3; the deployed ensemble is MobileNetV2 + EfficientNet-B0 | `configs/profiles/steel_strip.json` |
| 7 | Accuracy, precision, recall, F1 and confusion matrix (plus ROC-AUC) | Macro and weighted averages, per-class metrics, ROC/PR curves, calibration (ECE) | `ml/evaluation/metrics.py`, `artifacts/checkpoints/*/` |
| 8 | Grad-CAM explainability | Grad-CAM, Grad-CAM++ and Score-CAM; anomaly heat-maps and segmentation for anomaly models | `ml/explainability/cam.py` |
| 9 | Web interface for upload and prediction | Gradio UI (`/gradio`), as proposed, plus a full React dashboard | `backend/app/gradio_ui.py`, `frontend/` |
| 10 | Structured inspection report | PDF containing the image, prediction, confidence, severity, quality status, explanation images and traceability | `backend/app/services/reports.py` |

## 3. Proposal deliverables

| Deliverable | Status |
|---|---|
| 1. Industrial defect dataset | ✔ 3 datasets with versioned manifests and quality reports (`data/processed`, `artifacts/metrics`) |
| 2. Preprocessing pipeline | ✔ reusable and saved with each model (`preprocess.json`) so it is re-applied identically at inference |
| 3. Trained deep learning models | ✔ 8 evaluated classifiers (custom CNN, MobileNetV2, MobileNetV3-Small, EfficientNet-B0, ResNet-50, DenseNet-121, ConvNeXt-Tiny, ViT-Tiny) and 2 anomaly detectors, plus ONNX/INT8 edge exports |
| 4. Model evaluation | ✔ [RESULTS.md](RESULTS.md); confusion matrices, histories and ROC/PR plots for each run |
| 5. Explainable AI module | ✔ Grad-CAM family and anomaly localisation |
| 6. QualityGuard AI application | ✔ Gradio UI plus React dashboard on a FastAPI backend |
| 7. Inspection report | ✔ PDF per inspection, plus batch, period, model, dataset, drift and review reports |
| 8. Project documentation | ✔ this report, [ARCHITECTURE.md](ARCHITECTURE.md), [QUALITY_RULES.md](QUALITY_RULES.md), [DATASETS.md](DATASETS.md), [RESULTS.md](RESULTS.md), [FEATURE_STATUS.md](FEATURE_STATUS.md), [API.md](API.md), [DEMO_GUIDE.md](DEMO_GUIDE.md), [DEVELOPMENT_LOG.md](DEVELOPMENT_LOG.md), README |
| 9. Final presentation | ✔ `docs/QualityGuard_AI_Presentation.pptx` |

## 4. Methodology

### 4.1 Data

See [DATASETS.md](DATASETS.md).

- **NEU-CLS:** 1,800 grayscale crops, 300 per class, perfectly balanced. One duplicate pair is grouped so it cannot leak between splits. The split is 70/15/15, stratified, with seed 42.
- **MVTec AD:** the official defect-free training images are used to fit the anomaly models. The defective test images are divided into a validation set (for threshold calibration) and a test set (used once, for the final metrics).

### 4.2 Models

The proposal's architecture is *input → pre-trained CNN → feature extraction → global pooling → dense →
classification*. It is implemented by replacing each backbone's head with dropout plus a linear layer.

Training runs in two phases:
1. **Feature extraction:** the backbone is frozen and only the head is trained (1 epoch).
2. **Fine-tuning:** all layers are trained with AdamW and cosine learning-rate decay.

Early stopping uses validation loss (patience 3). The custom CNN baseline trains from scratch with a
higher learning rate and up to 25 epochs.

The proposal is limited to classification. Two problems led us to add a second engine based on
**unsupervised anomaly detection**:

- NEU-CLS has no defect-free class.
- A classifier cannot recognise defect types it has never seen.

The anomaly engine runs PatchCore and PaDiM on ResNet-18 features. It is trained only on defect-free
images, localises defects at pixel level, and measures their area. This covers the proposal's future-scope
item "image segmentation for precise defect boundaries".

### 4.3 Reliability

- **Calibration:** temperature scaling on the validation split.
- **Out-of-distribution detection:** Mahalanobis distance in feature space, with an energy score.
- **Disagreement and margins:** disagreement between ensemble members or anomaly engines, and scores close to a threshold, send the inspection to human review.

An image of a different product type is therefore reported as **INCONCLUSIVE / unfamiliar** instead of
being forced into a known class. This was checked by sending a bottle image to the steel profile.

### 4.4 Decision rules

Severity, PASS/FAIL and the quality index come from transparent, per-product rules in JSON. See
[QUALITY_RULES.md](QUALITY_RULES.md).

## 5. Results

See [RESULTS.md](RESULTS.md) for the full tables. In summary:

- **Transfer learning vs. baseline:** on NEU-CLS, a clean, well-separated benchmark, every classifier is near-perfect. Four of the seven pre-trained models (MobileNetV2, EfficientNet-B0, ResNet-50, ConvNeXt-Tiny) reach 100 % test accuracy, and the other three reach 99.6 % (one image each), all within 5–8 epochs. The from-scratch custom CNN reaches 99.6 % but needs 25 epochs. Transfer learning mainly buys faster, more reliable convergence. On this dataset its accuracy advantage is small, so the scores must be re-validated on harder, real line images.
- **Anomaly detection:** PatchCore reaches an image AUROC of 1.0 on both MVTec categories and a pixel AUROC of about 0.98. With the review-margin rule, every automatically decided item in the test set was correct, and 7–20 % of items were routed to review.
- **Speed (idle CPU):** 13–116 ms per classifier; about 0.24 s for the full anomaly pipeline per image; 0.10 s for the steel ensemble without explanation maps.
- **Edge:** MobileNetV2 exported to ONNX INT8 keeps 100 % accuracy at 2.5 MB and 4.6 ms per image on one CPU thread.

## 6. Application

| Interface | Purpose |
|---|---|
| **Gradio** (`/gradio`, or `scripts/run_gradio.py`) | Upload an image or use the webcam. Shows the decision, confidence, severity, quality index, Grad-CAM or anomaly maps and a PDF report, plus inspection history and model performance. |
| **React dashboard** (`/`) | Public landing page; live KPI dashboard; anomaly center; explainable-AI lab; system health; single, camera, batch, folder, video and golden-reference inspection; inspection history and detail; review queue; alerts; production-line simulation; analytics with drift monitoring and forecast; reports; quality assistant; model registry with approval and deployment; training; datasets; product profiles; administration (users, API keys, settings, audit, logs). |
| **REST API** (`/api/docs`) | All functions, authenticated with JWT or API key, for integration with line equipment or a manufacturing execution system (MES). |

Testing (details in the README):
- **Automated suite:** 64 automated tests (60 non-browser, 4 Playwright browser tests), all passing.
  - Unit tests: rules, splitting, metrics and calibration.
  - API and workflow tests: real inspections through the trained models, covering single, batch and video
    inspection, golden-reference comparison, drift, reviews, annotations and the feedback export.
  - Migration and integration tests.
- **Browser tests (Playwright):** they drive a real server in headless Chrome.
  - Every page renders without JavaScript errors.
  - A full upload → inspection → PDF → XAI-lab run completes.
  - A camera capture works using Chrome's fake camera device.

Feature-by-feature status, including what is missing and why, is in [FEATURE_STATUS.md](FEATURE_STATUS.md).

## 7. Limitations

- **Benchmark datasets:** NEU-CLS and MVTec AD contain controlled, well-lit images. Near-perfect benchmark scores do not transfer automatically to a real line, so performance must be re-validated on that line's own images.
- **NEU-CLS:** the steel profile cannot certify a surface as defect-free because the dataset has no normal class.
- **Severity and acceptance rules:** these are illustrative placeholders, not industrial standards.
- **Hardware:** everything runs on CPU. About 0.25 s per image is enough for inspection at a station, but not for high-speed lines without a GPU, a smaller input size or the INT8 edge models.
- **Licensing:** MVTec AD is CC BY-NC-SA, so it may not be used commercially.

## 8. Future scope

- **Real line integration:** validate the MQTT/webhook PASS / REJECT / HOLD signals against a physical reject gate
  or PLC; add OPC UA when a server is available.
- **Supervised segmentation:** train a U-Net once enough inspector masks have accumulated. The review UI already
  collects them and the feedback export already includes them.
- **Edge deployment:** deploy the INT8 ONNX MobileNetV2 on an edge device (measured in RESULTS.md); avoid INT8
  for MobileNetV3, which loses accuracy when quantised.
- **Active learning:** run the prioritised review → export → retrain → approve loop on real line data.
- **Research directions:** few-shot and self-supervised pre-training, synthetic defect generation and domain
  adaptation. These become worthwhile with harder, real data, where supervised transfer learning no longer
  saturates.
- **MES integration:** connect to a manufacturing execution system through the REST API and API keys.
