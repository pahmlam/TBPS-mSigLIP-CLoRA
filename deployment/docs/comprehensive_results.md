# Comprehensive Deployment Results

This document aggregates all quantization and deployment results for the mSigLIP model, including QDQ proxies and on-board executions on the Qualcomm RB3 Gen2 (HTP v68).

## 1. Baseline Reference (FP32)

| Direction | R@1 | R@5 | R@10 | mAP | mINP |
|---|---:|---:|---:|---:|---:|
| T2I | 52.40 | 79.38 | 87.80 | 57.38 | 50.67 |
| I2T | 55.30 | 81.45 | 89.70 | 51.38 | 34.50 |


## 2. Vision-Only QAT Evolution (QDQ Proxy)

| Version | Cosine Mean | Cosine Min | T2I R@1 | T2I R@5 | T2I R@10 | T2I mAP | T2I mINP | I2T R@1 | I2T R@5 | I2T R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V2 | 0.8975 | 0.8747 | 45.42 | 73.38 | 83.12 | 49.95 | 42.23 | 49.40 | 75.30 | 84.05 |
| V3 | 0.9353 | 0.9190 | 48.20 | 75.42 | 85.10 | 53.39 | 46.60 | 52.30 | 78.90 | 86.85 |
| V4 | 0.9364 | 0.9091 | 48.50 | 76.25 | 85.38 | 53.73 | 46.91 | 52.95 | 79.75 | 88.40 |
| V5 | 0.9437 | 0.9311 | 49.25 | 77.28 | 85.80 | 54.55 | 47.86 | 53.40 | 80.85 | 88.10 |
| V6 | 0.9491 | 0.9266 | 49.30 | 77.38 | 86.28 | 54.87 | 48.17 | 53.85 | 80.25 | 89.05 |
| V7 | 0.9485 | 0.9083 | 48.38 | 76.53 | 85.68 | 54.36 | 48.10 | 53.05 | 79.80 | 88.05 |
| V8 | 0.9606 | 0.9447 | 50.85 | 77.47 | 86.98 | 55.79 | 49.24 | 52.90 | 80.45 | 88.35 |


## 3. Text-Only Finite Mask (QDQ Proxy)

| Direction | R@1 | R@5 | R@10 | mAP | mINP |
|---|---:|---:|---:|---:|---:|
| T2I | 51.65 | 79.43 | 87.68 | 57.08 | 50.50 |
| I2T | 55.55 | 81.05 | 88.75 | 51.26 | 34.72 |

**Text QDQ Fidelity:** Cosine Mean = N/A, Cosine Min = N/A



## 4. End-to-End Both-INT8 (QDQ Proxy)

| Direction | R@1 | R@5 | R@10 | mAP | mINP | Drop vs FP32 (R@1) |
|---|---:|---:|---:|---:|---:|---:|
| T2I | 50.25 | 77.72 | 86.68 | 55.53 | 48.91 | -2.15 |
| I2T | 52.95 | 79.45 | 87.60 | 49.40 | 33.09 | -2.35 |


## 5. Board Execution Results (RB3 Gen2 HTP v68)

| Variant | Cosine Mean | Cosine Min | T2I R@1 | Runtime (ms/image) | FPS |
|---|---:|---:|---:|---:|---:|
| rotated_w8a8_learned_qat_v8 | 0.9585 | 0.9400 | N/A | 71.80 | 22.77 |
| rotated_w8a8_learned_qat_v8_gallery_2000 | 0.9562 | 0.8840 | 50.20 | N/A | N/A |