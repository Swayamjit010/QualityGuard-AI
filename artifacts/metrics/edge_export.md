# Edge export (ONNX / INT8)

Held-out test split; INT8 static quantisation calibrated on validation images; CPU, 1 thread(s).

| Model | Variant | Accuracy | Macro F1 | Agreement with PyTorch | Max abs logit diff | Latency (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|
| mobilenet_v3_small | pytorch_fp32 | 0.9963 | 0.9963 | 1.0000 | 0.0000 | 5.44 | 5.93 |
| mobilenet_v3_small | onnx_fp32 | 0.9963 | 0.9963 | 1.0000 | 0.0000 | 2.35 | 5.82 |
| mobilenet_v3_small | onnx_int8 | 0.3407 | 0.2482 | 0.3444 | 8.8361 | 2.65 | 1.77 |
| mobilenet_v2 | pytorch_fp32 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 16.88 | 8.73 |
| mobilenet_v2 | onnx_fp32 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 7.67 | 8.49 |
| mobilenet_v2 | onnx_int8 | 1.0000 | 1.0000 | 1.0000 | 1.7151 | 4.6 | 2.49 |
