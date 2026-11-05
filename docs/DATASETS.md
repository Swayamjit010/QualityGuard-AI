# Datasets

| Dataset | Domain | Images | Labels | Licence | Used for |
|---|---|---|---|---|---|
| **NEU-CLS** (Northeastern University) | hot-rolled steel strip surface, 200×200 grayscale | 1,800 | 6 defect classes × 300: crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches | research use | supervised classification (`steel_strip` profile) |
| **MVTec AD – bottle** | glass bottle, top view, 900×900 colour | 292 | good, broken_large, broken_small, contamination; pixel masks | CC BY-NC-SA 4.0 | anomaly detection and localisation (`bottle` profile) |
| **MVTec AD – metal_nut** | metal nut, 700×700 colour | 335 | good, bent, color, flip, scratch; pixel masks | CC BY-NC-SA 4.0 | anomaly detection and localisation (`metal_nut` profile) |
| **MVTec AD – screw** | machine screw, 1024×1024 grayscale | 480 | good (361), manipulated_front, scratch_head, scratch_neck, thread_side, thread_top (23–25 each); pixel masks | CC BY-NC-SA 4.0 | anomaly detection and localisation (`screw` profile) |
| **MVTec AD – grid** | woven metal grid, 1024×1024 grayscale | 342 | good (285), bent, broken, glue, metal_contamination, thread (11–12 each); pixel masks | CC BY-NC-SA 4.0 | anomaly detection and localisation (`metal_grid` profile) |
| **MPDD** – Metal Parts Defect Detection (Ježek et al., 2021) | metal brackets (black/brown/white), connector, metal plate, tubes; 1024×1024 colour; parts at varied positions, orientations and backgrounds | 1,346 (888 train good; 458 test) | good + hole, scratches, bend_and_parts_mismatch, parts_mismatch, defective_painting, major_rust, total_rust, anomalous; pixel masks | CC BY-NC-SA 4.0 | anomaly detection and localisation (six `mpdd_*` profiles) |
| **KolektorSDD2** (ViCoS / Kolektor) | electrical commutator surface, ≈230×630 colour | 3,335 | good 2,979 / defect 356; pixel masks (`*_GT.png`) | CC BY-NC-SA 4.0 | supervised classification, supervised U-Net segmentation and anomaly detection (`commutator_surface` profile) |

Download with `python scripts/download_datasets.py --neu --mvtec bottle metal_nut screw grid --ksdd2`. The script
lists the sources and is idempotent. Files go to `data/raw/<dataset>/`. KolektorSDD2 comes from the official archive
(`https://go.vicos.si/kolektorsdd2`, 853,126,555 bytes): the size is checked against the server's `Content-Length`,
every member's CRC is verified (`zipfile.testzip`), and extraction is guarded against path traversal and zip bombs.
The archive contains one stray pair (`10301 (copy).png`, `10301_GT (copy).png`) that has no matching mask; it is skipped.
Counts after preparation match the published dataset exactly: official train 2,331 images (2,085 good + 246 defective),
official test 1,004 images (894 good + 110 defective).

MPDD comes from a Hugging Face mirror (`meksamiao/mpdd`, 1,825,041,283 bytes; original: github.com/stepanje/MPDD). The
archive passed the CRC check, and its folder counts match the published dataset. It was extracted with the same guarded
extractor; `python scripts/download_datasets.py --mpdd`.

**Evaluation-only probe sets** (never used for training or threshold selection) live in `data/raw/foreign_probe/`:
300 Imagenette validation photos (natural images: dogs, churches, trucks, …) and 120 test images from six MVTec AD
categories the system does not support (hazelnut, carpet, wood, toothbrush, pill, cable). They measure how often
unsupported images are correctly rejected.

> MVTec AD is **non-commercial**. Do not deploy models trained on it in a commercial product.

## Analysis (`scripts/prepare_datasets.py` → `ml/datasets/analysis.py`)

For every dataset the analysis computes the following and writes it to
`artifacts/metrics/dataset_report_<name>.json`:

- readable and corrupted files (corrupted files are excluded)
- class distribution, imbalance ratio, image sizes and colour modes
- **exact duplicates** (SHA-256) and **near duplicates** (perceptual hash with a pixel-level confirmation)
- quality flags: blank, dark, over-exposed and blurry images (brightness, contrast, Laplacian variance)
- mask presence and size checks (MVTec)

You can view the report on the Datasets page and download it as a PDF (Reports → Dataset quality report).

**Findings on the current data:**

| Dataset | Findings |
|---|---|
| NEU-CLS | Perfectly balanced (imbalance 1.0). One exact-duplicate pair (`patches_101` / `patches_105`). 24 images quality-flagged (very bright or low contrast), kept because they are genuine samples. |
| MVTec | The training folders contain only defect-free images, so the defective test images are split into validation and test to calibrate thresholds without touching the final test set. Screw has 25 confirmed near-duplicate pairs (very similar screw poses); MVTec splits follow the official train/test protocol. |
| KolektorSDD2 | 0 corrupted, 0 exact and 0 near duplicates. Strong imbalance (≈8.4 good : 1 defect), handled by class weighting (classifier) and per-epoch negative sub-sampling (segmentation). The official test split is kept as the held-out test set; the official train split is divided 85/15 into train/val (train 1,772 good + 209 defect, val 313 + 37, test 894 + 110). |

## Split policy (`ml/datasets/splitting.py`)

- **Stratified:** each class keeps its proportion in train, val and test (70 / 15 / 15 by default, seed 42).
- **Duplicate-grouped:** exact and near duplicates are joined into one group, and a group never spans two
  splits. This prevents leakage that would inflate test scores. (Fixed 2026-11-05: grouping previously used the
  report's list, which is truncated to 50 entries; it now uses the complete list. No existing split was affected —
  NEU-CLS has 1 duplicate group and 1 near-duplicate pair.)
- **Versioned:** each split is written to `data/processed/<name>/manifest_<hash>.json`. The hash covers the file list,
  labels and split assignment, and `manifest_latest.json` points to the current version. Every model records
  the `dataset_version` it was trained on.
- **MVTec:** the train split is the official `train/good` folder. The validation split (good plus defective) calibrates the
  anomaly thresholds. The test split is used only once, for final metrics.

## Preprocessing (`ml/preprocessing/transforms.py`)

- Resize to the architecture's input size (224 by default) and normalise with ImageNet mean and std.
  Grayscale images are replicated to three channels.
- **Training augmentation** (classification): horizontal and vertical flips, 90° rotations, scale/translate/rotate
  affine, brightness and contrast jitter, light Gaussian noise and coarse dropout. Each is enabled per
  dataset in the preprocessing config, because flips are valid for steel texture but not for every product.
- The exact preprocessing used for training is saved with each checkpoint (`preprocess.json`) and reapplied at
  inference, so training and inference preprocessing cannot drift apart.

## Importing your own data

**Datasets → Import ZIP** (ML engineer) accepts either:

- one folder per class, for example `scratch/001.png`, or
- images plus a `labels.csv` file with `file,label` columns.

**Re-analyse & split** creates a versioned manifest. Reviewed inspections can be exported as a curated
retraining set (**Export reviewed feedback**), which writes `data/feedback/<timestamp>/<profile>/<label>/…` and a `manifest.csv`.
