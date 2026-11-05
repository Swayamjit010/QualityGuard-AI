# Quality rules: severity, decisions and the quality index

Implementation: `ml/quality/rules.py`. All thresholds come from the **product profile**
(`configs/profiles/<id>.json`, which can be edited in the UI and is versioned). Nothing in the code encodes a
universal industrial tolerance.

Every inspection result keeps three layers separate:

1. **AI assessment:** what the models measured, namely the class, anomaly scores, regions and reliability.
2. **Rule decision:** what the profile's rules conclude from the AI assessment.
3. **Human decision:** added later by an inspector in the review workflow. This module never sets it.

## 1. Severity (`assess_severity`)

| Step | Rule | Profile key |
|---|---|---|
| 0 | No defect → `none` | – |
| 1 | Start at the default level | `severity_rules.default_level` |
| 2 | If regions were localised (anomaly engine), map the area metric to a band; the first band whose `max` is ≥ the value wins | `severity_rules.area_metric`, `area_levels[]` |
| 3 | Many regions → escalate | `count_escalation.regions_ge`, `escalate_by` |
| 4 | Calibrated images: a region larger than the limit → at least `high` | `max_region_area_mm2`, `calibration.mm_per_pixel` |
| 5 | Anomaly score far above its threshold → escalate | `intensity_escalation.score_ratio_ge`, `escalate_by` |
| 6 | Per-class minimum severity (classifier) | `severity_rules.class_min_level` |

The levels are `none < low < medium < high < critical`. Every step that fires adds a
human-readable line to `severity.basis`, which is shown in the UI and in reports.

**Example (bottle):** a defect area ratio of 0.3 % falls in the `low` band. Three or more regions
escalate it to `medium`, and a score at least 2× the threshold escalates it to `high`.

## 2. Decision (`decide`)

The checks are evaluated in this order; the first one that matches decides the status:

1. **Input fails validation** (unreadable, too small, too dark or bright, too blurry, too little contrast) → `INCONCLUSIVE`
2. **Unfamiliar input** (out-of-distribution or novelty check) → `acceptance_rules.unfamiliar_action` (default `INCONCLUSIVE`)
3. **Engines disagree**, or a score lies within the review margin of its threshold → `disagreement_action` (default `REVIEW REQUIRED`)
4. **Low reliability** (low confidence, small top-1/top-2 margin, ensemble disagreement) with `review_on_low_reliability` → `REVIEW REQUIRED`
5. **No defect** → `PASS`
6. **Class listed in `rejected_classes`** → `FAIL`
7. **Severity ≥ `fail_at_severity`** → `FAIL`; otherwise `PASS` (or `REVIEW REQUIRED` if `accept_below_fail_level` is false)

`REVIEW REQUIRED` and `INCONCLUSIVE` always set `manual_review_required`, which puts the inspection
in the review queue and raises an alert (configurable under Administration → Settings → alert rules).

## 3. Quality index

The quality index is a transparent deduction score, **not** a validated quality metric:

```
quality_index = max(0, 100 - deductions[severity])            # quality_index.deductions
REVIEW REQUIRED → min(quality_index, review_cap)              # quality_index.review_cap (default 60)
INCONCLUSIVE    → null
```

The default deductions are none 0, low 10, medium 35, high 65 and critical 100.

## 4. Reliability (inference pipeline)

| Signal | Source | Effect |
|---|---|---|
| Calibrated confidence | temperature scaling fitted on the validation split | below `reliability.min_confidence` → low reliability |
| Top-1/top-2 margin | calibrated probabilities | below `reliability.min_margin` → low reliability |
| Ensemble disagreement | members predict different classes | disagreement → `REVIEW REQUIRED` |
| Mahalanobis distance | penultimate-layer features against training class means (threshold = 99th percentile on validation) | out-of-distribution → unfamiliar |
| Energy score | logits | reported |
| PatchCore and PaDiM | each engine's own threshold, validated on held-out data | a mismatch, or a score within ±`review_margin` of the threshold → `REVIEW REQUIRED` |
| Global novelty | kNN distance to defect-free reference features | above 1.05 × the maximum calibration distance → unfamiliar |

## 5. Shipped example profiles

| Profile | Engines | Policy |
|---|---|---|
| `bottle` | PatchCore + PaDiM + novelty | zero tolerance: any defect region fails (`fail_at_severity = low`) |
| `metal_nut` | PatchCore + PaDiM + novelty | fails at `medium` and above |
| `steel_strip` | classifier ensemble (MobileNetV2 + EfficientNet-B0) | identifies the defect type; per-class minimum severity |

These are **example** configurations. Replace them with your own validated acceptance criteria before
using the system on real production.

## Custom products (Add a product)

A new kind of item can be added from photos of **good** items only (page *Products → Add a product*, or
`POST /api/products`). Training uses the same PatchCore, PaDiM and novelty engines as the built-in products. The
calibration is designed for small photo sets (`ml/anomaly_detection/custom.py`):

| Step | Rule |
|---|---|
| Out-of-fold calibration | Good photos are split into K ≤ 5 folds. Each fold is scored by engines fitted on the other folds, so every good photo is scored by a model that never saw it. The deployed model is then fitted on all good photos. |
| PatchCore threshold (decides) | Maximum out-of-fold good score. Because the final model has seen more data, unseen good items score lower than the out-of-fold models did, so the threshold is already conservative. |
| PaDiM threshold (escalates) | Maximum out-of-fold good score × 1.10. PaDiM is unreliable with few photos, so it can only escalate a PatchCore PASS to REVIEW. |
| Decision | FAIL if the PatchCore score is above its threshold. REVIEW if the score is within ±5 % of the threshold, or if PaDiM flags an item that PatchCore passed. Otherwise PASS. |
| Unfamiliar input | The threshold lies between the good photos and images of other products: max(out-of-fold good distance × 2, ½ × 5th percentile of foreign-image distances). *Unfamiliar* therefore means "a different kind of item", while defective items of this product are still judged as defects. |
| Defective photos | Optional. They are used **only** to report the detection rate, never to set a threshold. |
| Activation | The model is registered as *validated*. A user with `approve_models` must approve and activate it, which creates the product profile with default severity rules (adjust them on the Profiles page). |

These defaults were chosen on the MVTec AD *validation* split. Results on the held-out *test* split are in
[RESULTS.md](RESULTS.md) §9.
