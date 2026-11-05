---
title: QualityGuard AI
emoji: 🛡️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: cc-by-nc-sa-4.0
short_description: Industrial defect inspection — NORMAL / DEFECTIVE / MANUAL
---

# QualityGuard AI — live demo

Upload an image of a supported industrial component or surface and press **INSPECT**. Trained models return
**NORMAL**, **DEFECTIVE** or **MANUAL INSPECTION REQUIRED**, with the defect location, heat-maps and a PDF report.

**Supported categories** (each with its own trained models): hot-rolled steel strip (NEU-CLS); bottle, metal nut,
screw, metal grid (MVTec AD); electrical commutator surface (KolektorSDD2); metal brackets ×3, connector, metal plate,
tubes (MPDD). Anything else is reported as unsupported → manual inspection.

**Access:** create an account (email + username + password). New accounts can view results; press
*Request inspection access* and an administrator approves or denies it.

**Limitations:** visual inspection of the trained categories only. Photos taken very differently from the training
images (for example studio or web photos) are rejected as unfamiliar. The commutator model auto-passed 9 % of defective
test parts. Data on this free Space is **not persistent**: accounts and inspections are reset when the Space restarts.

**Licence:** the models are trained on MVTec AD, KolektorSDD2 and MPDD (CC BY-NC-SA 4.0) and NEU-CLS (research use).
**Non-commercial use only.** Source code, evaluation and documentation: see the GitHub repository linked in the Space.
