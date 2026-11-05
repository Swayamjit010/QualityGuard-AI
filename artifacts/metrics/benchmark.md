# Benchmark results

Measured 2026-11-05T06:16:15+00:00 on Intel64 Family 6 Model 154 Stepping 4, GenuineIntel (12 logical CPUs, 15.7 GB RAM, torch 2.8.0+cpu, CUDA: False). CPU load at start: 17.4 %.

## Classifiers (forward pass only)

| Model | Input | Params (M) | Checkpoint (MB) | Load (ms) | Median (ms) | p95 (ms) | img/s @1 | img/s @8 | img/s @16 | RSS on load (MB) |
|---|---|---|---|---|---|---|---|---|---|---|
| convnext_tiny | 224 | 27.82 | 106.2 | 323.2 | 65.19 | 78.15 | 14.7 | 23.0 | 21.6 | 171.1 |
| custom_cnn | 224 | 1.22 | 4.7 | 26.4 | 12.7 | 14.46 | 71.3 | 107.6 | 105.2 | 0.0 |
| densenet121 | 224 | 6.96 | 27.1 | 198.4 | 116.37 | 142.65 | 8.8 | 14.5 | 16.0 | 45.0 |
| efficientnet_b0 | 224 | 4.02 | 15.6 | 115.4 | 41.45 | 49.68 | 22.8 | 41.9 | 41.5 | 17.4 |
| mobilenet_v2 | 224 | 2.23 | 8.7 | 73.5 | 31.48 | 37.96 | 25.9 | 51.4 | 50.3 | 0.0 |
| mobilenet_v3_small | 224 | 1.52 | 5.9 | 85.3 | 15.64 | 17.69 | 57.9 | 190.0 | 217.9 | 0.0 |
| resnet50 | 224 | 23.52 | 90.0 | 343.4 | 95.23 | 122.82 | 10.0 | 14.8 | 15.2 | 91.7 |
| vit_tiny | 224 | 5.53 | 21.1 | 675.2 | 37.19 | 43.03 | 26.7 | 59.3 | 55.8 | 34.4 |

## Anomaly engines (scoring only)

| Run | Engine | Median (ms) | p95 (ms) | Load (ms) | Checkpoint (MB) |
|---|---|---|---|---|---|
| mvtec_bottle-anomaly-resnet18-v1 | patchcore | 33.81 | 35.96 | 162.6 | 80.5 |
| mvtec_bottle-anomaly-resnet18-v1 | padim | 54.25 | 60.95 | 162.6 | 80.5 |
| mvtec_metal_nut-anomaly-resnet18-v1 | patchcore | 43.67 | 49.57 | 241.3 | 80.5 |
| mvtec_metal_nut-anomaly-resnet18-v1 | padim | 52.61 | 55.56 | 241.3 | 80.5 |

## Full pipeline per profile

| Profile | Without explanations, median (ms) | With explanations, median (ms) |
|---|---|---|
| bottle | 242.59 | 238.73 |
| metal_nut | 225.59 | 229.72 |
| steel_strip | 100.53 | 426.81 |

Peak process RSS during the run: 2023.7 MB.
