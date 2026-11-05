### Classifiers - NEU-CLS held-out test split (identical for all models)

| Model | Pre-trained | Epochs | Accuracy | Macro P | Macro R | Macro F1 | ROC-AUC | Worst-class recall | ECE | CPU latency (ms) | Size (MB) | Train (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mobilenet_v2 | yes | 8 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.000 | 0.001 | 33.5 | 8.9 | 11.9 |
| efficientnet_b0 | yes | 6 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.000 | 0.000 | 55.7 | 16.1 | 13.1 |
| convnext_tiny | yes | 7 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.000 | 0.000 | 88.1 | 111.3 | 30.3 |
| resnet50 | yes | 8 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.000 | 0.000 | 89.4 | 94.1 | 4.8 |
| mobilenet_v3_small | yes | 8 | 0.9963 | 0.9964 | 0.9963 | 0.9963 | 1.0000 | 0.978 | 0.003 | 15.6 | 6.1 | 4.6 |
| custom_cnn | no (baseline) | 25 | 0.9963 | 0.9964 | 0.9963 | 0.9963 | 1.0000 | 0.978 | 0.006 | 17.3 | 4.9 | 22.4 |
| vit_tiny | yes | 5 | 0.9963 | 0.9964 | 0.9963 | 0.9963 | 1.0000 | 0.978 | 0.003 | 40.3 | 22.1 | 7.1 |
| densenet121 | yes | 5 | 0.9963 | 0.9964 | 0.9963 | 0.9963 | 1.0000 | 0.978 | 0.004 | 130.6 | 27.8 | 23.1 |

### Anomaly detectors - MVTec AD and KolektorSDD2 (commutator) test splits

The fusion rows describe the anomaly engines on their own. The decision actually deployed for each category (including validation-based engine selection and, for the commutator, classifier + segmentation voting) is measured end to end in `reports/end_to_end_evaluation.md`.

| Category | Engine | Image AUROC | Recall | FPR | Pixel AUROC | Dice | Latency (ms) |
|---|---|---|---|---|---|---|---|
| commutator | patchcore | 0.9500 | 0.991 | 0.673 | 0.9812 | 0.393 | 93 |
| commutator | padim | 0.9075 | 0.982 | 0.714 | 0.9706 | 0.423 | 90 |
| commutator | anomaly-engine fusion | auto-decided accuracy 0.398 | missed defects 0 | false rejects 151 | review rate 0.750 | | |
| bracket_black | patchcore | 0.8776 | 0.875 | 0.438 | 0.9761 | 0.131 | 76 |
| bracket_black | padim | 0.6224 | 0.708 | 0.750 | 0.9255 | 0.033 | 89 |
| bracket_black | anomaly-engine fusion | auto-decided accuracy 1.000 | missed defects 0 | false rejects 0 | review rate 0.675 | | |
| bracket_brown | patchcore | 0.9911 | 0.846 | 0.000 | 0.9886 | 0.309 | 74 |
| bracket_brown | padim | 0.7751 | 0.962 | 0.538 | 0.9410 | 0.157 | 92 |
| bracket_brown | anomaly-engine fusion | auto-decided accuracy 1.000 | missed defects 0 | false rejects 0 | review rate 0.282 | | |
| bracket_white | patchcore | 0.8375 | 0.875 | 0.667 | 0.9651 | 0.054 | 65 |
| bracket_white | padim | 0.7208 | 1.000 | 0.733 | 0.9483 | 0.031 | 86 |
| bracket_white | anomaly-engine fusion | auto-decided accuracy 0.786 | missed defects 0 | false rejects 3 | review rate 0.548 | | |
| connector | patchcore | 1.0000 | 1.000 | 0.000 | 0.9967 | 0.643 | 91 |
| connector | padim | 0.8571 | 1.000 | 0.333 | 0.9786 | 0.435 | 96 |
| connector | anomaly-engine fusion | auto-decided accuracy 1.000 | missed defects 0 | false rejects 0 | review rate 0.091 | | |
| metal_plate | patchcore | 1.0000 | 0.944 | 0.000 | 0.9806 | 0.812 | 66 |
| metal_plate | padim | 0.5556 | 0.944 | 1.000 | 0.9310 | 0.677 | 90 |
| metal_plate | anomaly-engine fusion | auto-decided accuracy 1.000 | missed defects 0 | false rejects 0 | review rate 0.204 | | |
| tubes | patchcore | 0.8464 | 0.914 | 0.688 | 0.9834 | 0.503 | 66 |
| tubes | padim | 0.5446 | 0.943 | 1.000 | 0.9345 | 0.194 | 90 |
| tubes | anomaly-engine fusion | auto-decided accuracy 1.000 | missed defects 0 | false rejects 0 | review rate 0.529 | | |
| bottle | patchcore | 1.0000 | 0.969 | 0.000 | 0.9824 | 0.644 | 337 |
| bottle | padim | 0.9938 | 1.000 | 0.100 | 0.9851 | 0.692 | 189 |
| bottle | anomaly-engine fusion | auto-decided accuracy 1.000 | missed defects 0 | false rejects 0 | review rate 0.071 | | |
| grid | patchcore | 0.9636 | 0.967 | 0.273 | 0.9622 | 0.365 | 89 |
| grid | padim | 0.9212 | 0.900 | 0.636 | 0.9095 | 0.322 | 113 |
| grid | anomaly-engine fusion | auto-decided accuracy 0.962 | missed defects 0 | false rejects 1 | review rate 0.366 | | |
| metal_nut | patchcore | 1.0000 | 0.979 | 0.000 | 0.9826 | 0.692 | 84 |
| metal_nut | padim | 0.9867 | 0.979 | 0.000 | 0.9759 | 0.652 | 97 |
| metal_nut | anomaly-engine fusion | auto-decided accuracy 1.000 | missed defects 0 | false rejects 0 | review rate 0.203 | | |
| screw | patchcore | 0.9820 | 1.000 | 0.429 | 0.9891 | 0.288 | 135 |
| screw | padim | 0.7869 | 0.984 | 0.762 | 0.9776 | 0.250 | 140 |
| screw | anomaly-engine fusion (superseded: all engines voting) | auto-decided accuracy 1.000 | missed defects 0 | false rejects 0 | review rate 0.439 | | |
