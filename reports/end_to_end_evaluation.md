# End-to-end evaluation (deployed decision path, held-out test images)

Mode: **auto-recognition (as /api/inspect default)**. Generated 2026-11-05T12:27:01.594468+00:00 by `scripts/evaluate_profiles.py`.
Ground truth: `good` → NORMAL, any defect label → DEFECTIVE. Explanations skipped (no effect on decisions).

| Profile | Test images (def/normal) | Routed correctly | Manual rate | Auto-decided accuracy | Missed defects | False alarms | p50 latency |
|---|---|---|---|---|---|---|---|
| bottle | 42 (32/10) | 1.000 | 0.071 | 1.000 | 0 | 0 | 440 ms |
| commutator_surface | 1004 (110/894) | 1.000 | 0.081 | 0.988 | 10 | 1 | 476 ms |
| metal_grid | 41 (30/11) | 1.000 | 0.415 | 1.000 | 0 | 0 | 411 ms |
| metal_nut | 59 (48/11) | 1.000 | 0.220 | 1.000 | 0 | 0 | 387 ms |
| mpdd_bracket_black | 40 (24/16) | 1.000 | 0.700 | 1.000 | 0 | 0 | 282 ms |
| mpdd_bracket_brown | 39 (26/13) | 1.000 | 0.308 | 1.000 | 0 | 0 | 284 ms |
| mpdd_bracket_white | 31 (16/15) | 1.000 | 0.548 | 0.786 | 0 | 3 | 280 ms |
| mpdd_connector | 22 (7/15) | 1.000 | 0.136 | 1.000 | 0 | 0 | 274 ms |
| mpdd_metal_plate | 49 (36/13) | 1.000 | 0.204 | 1.000 | 0 | 0 | 276 ms |
| mpdd_tubes | 51 (35/16) | 1.000 | 0.549 | 1.000 | 0 | 0 | 278 ms |
| screw | 82 (61/21) | 1.000 | 0.427 | 1.000 | 0 | 0 | 297 ms |
| steel_strip | 270 (270/0) | 1.000 | 0.015 | 1.000 | 0 | 0 | 199 ms |

## Unsupported images (should become MANUAL INSPECTION REQUIRED)

| Probe set | Images | MANUAL rate | Verdicts |
|---|---|---|---|
| imagenette_natural_photos | 300 | 1.000 | {'MANUAL INSPECTION REQUIRED': 300} |
| mvtec_unsupported_categories | 120 | 1.000 | {'MANUAL INSPECTION REQUIRED': 120} |
