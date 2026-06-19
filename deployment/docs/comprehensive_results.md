# Comprehensive Deployment Results

This document aggregates the current quantization and RB3 deployment results for mSigLIP.

Important interpretation:

- **QDQ proxy** means the AI Hub quantized ONNX graph was evaluated off-board with ONNX Runtime. It is a fidelity and retrieval proxy before QNN context-binary execution.
- **Board vision smoke** means the QNN context binary was run on RB3 for a small 10-image input set, mainly for board-vs-PyTorch fidelity and runtime profiling.
- **`rotated_w8a8_learned_qat_v8_gallery_2000` is not a new model variant.** It is the same QAT v8 vision context binary executed on RB3 for the full VN3K test gallery: 2000 image embeddings. The retrieval result in `board_vision_r1.json` uses **board image embeddings + FP32 text embeddings**. Therefore it is a **vision-only board result**, not both-INT8 and not text-on-board.
- Text W8A8 has passed QDQ proxy gates and the finite/f32/link-safe context binary now links, but text board fidelity/retrieval is still the next gate.

---

## 1. Baseline References

| Reference | T2I R@1 | T2I R@5 | T2I R@10 | T2I mAP | T2I mINP | I2T R@1 | I2T R@5 | I2T R@10 | I2T mAP | I2T mINP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Paper/historical baseline | 52.28 | 79.55 | 88.03 | 57.32 | 50.57 | 55.30 | 81.45 | 89.70 | 51.38 | 34.50 |
| Local FP32 sanity | 52.40 | 79.38 | 87.80 | 57.38 | 50.67 | 55.30 | 81.45 | 89.70 | 51.38 | 34.50 |

The reporting baseline for deployment drops is the paper T2I R@1 `52.28`. The local FP32 sanity row is used to verify that the evaluation pipeline is aligned.

---

## 2. Vision-Only QAT Evolution (QDQ Proxy)

These rows are **off-board QDQ proxy** results: image encoder INT8 QDQ, text encoder FP32.

| Version | Main change | Cosine mean | Cosine min | T2I R@1 | T2I R@5 | T2I R@10 | T2I mAP | T2I mINP | I2T R@1 | I2T R@5 | I2T R@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Rotation only | mean-preserving rotation + W8A8 | 0.8975 | 0.8747 | 45.42 | 73.38 | 83.12 | 49.95 | 42.23 | 49.40 | 75.30 | 84.05 |
| V3 | EMA observer QAT | 0.9353 | 0.9190 | 48.20 | 75.42 | 85.10 | 53.39 | 46.60 | 52.30 | 78.90 | 86.85 |
| V4 | + pooling-head fake quant | 0.9364 | 0.9091 | 48.50 | 76.25 | 85.38 | 53.73 | 46.91 | 52.95 | 79.75 | 88.40 |
| V5 | + per-linear fake quant | 0.9437 | 0.9311 | 49.25 | 77.28 | 85.80 | 54.55 | 47.86 | 53.40 | 80.85 | 88.10 |
| V6 | + attention-matmul fake quant | 0.9491 | 0.9266 | 49.30 | 77.38 | 86.28 | 54.87 | 48.17 | 53.85 | 80.25 | 89.05 |
| V7 | v6 coverage + cosine LR | 0.9485 | 0.9083 | 48.38 | 76.53 | 85.68 | 54.36 | 48.10 | 53.05 | 79.80 | 88.05 |
| **V8** | **learned rotation + v6 QAT recipe** | **0.9606** | **0.9447** | **50.85** | **77.47** | **86.98** | **55.79** | **49.24** | **52.90** | **80.45** | **88.35** |

Conclusion: QAT v8 is the first vision-only QDQ proxy to clear the deployment target (`T2I R@1 >= 50.0`).

---

## 3. Text-Only Finite/f32/link-safe Mask (QDQ Proxy)

These rows are **off-board QDQ proxy** results: image encoder FP32, text encoder INT8 QDQ.

| Direction | R@1 | R@5 | R@10 | mAP | mINP |
|---|---:|---:|---:|---:|---:|
| T2I | 51.65 | 79.43 | 87.68 | 57.08 | 50.50 |
| I2T | 55.55 | 81.05 | 88.75 | 51.26 | 34.72 |

Text QDQ fidelity from `text_qdq_fid.json`:

| Metric | Value |
|---|---:|
| cosine mean | 0.9949 |
| cosine min | 0.9912 |
| cosine max | 0.9972 |
| samples | 10 |

Attention-mask QDQ scale gate from `attention_qdq_scales.json`:

| Metric | Value |
|---|---:|
| Softmax paths | 12 |
| QDQ pairs on `scores+mask` | 12 |
| max QDQ scale | 0.3523 |
| min QDQ scale | 0.1828 |

Conclusion: the finite mask avoids the old `-FLT_MAX` dynamic-range collapse. The f32/link-safe mask rewrite is for QNN linkability; it does not change the text retrieval math.

---

## 4. End-to-End Both-INT8 (QDQ Proxy)

This is the current end-to-end deploy proxy: vision INT8 QDQ + text INT8 QDQ, still evaluated off-board.

| Direction | R@1 | R@5 | R@10 | mAP | mINP | Drop vs paper R@1 |
|---|---:|---:|---:|---:|---:|---:|
| T2I | 50.25 | 77.72 | 86.68 | 55.53 | 48.91 | -2.03 |
| I2T | 52.95 | 79.45 | 87.60 | 49.40 | 33.09 | -2.35 |

Conclusion: both-INT8 QDQ passes the deployment target off-board. Board confirmation is pending until text QNN outputs are collected on RB3.

---

## 5. Board Vision Execution (RB3 Gen2 HTP v68)

### 5.1 Smoke Fidelity + Runtime

Artifact directory: `artifacts/deployment/qnn_runs/rotated_w8a8_learned_qat_v8`

This is the 10-image board smoke run. It verifies that the v8 vision context binary executes on HTP and that board outputs match PyTorch closely.

| Metric | Value |
|---|---:|
| samples | 10 |
| board-vs-PyTorch cosine mean | 0.9585 |
| board-vs-PyTorch cosine min | 0.9400 |
| board-vs-PyTorch cosine max | 0.9761 |
| any QNN NaN/Inf | false |
| average inference time | 33.05 ms/image |
| throughput | 22.77 FPS |

Note: `71.80 ms` in the profile is initialization/load time (`NetRun Init`), not per-image inference latency. The per-image runtime is `33.05 ms/image`.

### 5.2 Full-Gallery Board Vision Run

Artifact directory: `artifacts/deployment/qnn_runs/rotated_w8a8_learned_qat_v8_gallery_2000`

This directory contains 2000 RB3/QNN image embeddings from the same v8 vision context binary. It is used for:

- board-vs-PyTorch fidelity over the full gallery (`qnn_vs_pytorch_summary.json`);
- full VN3K retrieval with board image embeddings + FP32 text embeddings (`board_vision_r1.json`).

Full-gallery board-vs-PyTorch fidelity:

| Metric | Value |
|---|---:|
| samples | 2000 |
| embedding dim | 768 |
| cosine mean | 0.9562 |
| cosine min | 0.8840 |
| cosine max | 0.9821 |
| any QNN NaN/Inf | false |

The min cosine is lower than the 10-image smoke min because this run covers all 2000 gallery images, not because it is a different model.

Full-gallery board vision retrieval:

| Direction | R@1 | R@5 | R@10 | mAP | mINP |
|---|---:|---:|---:|---:|---:|
| T2I | 50.20 | 77.62 | 86.73 | 55.84 | 49.51 |
| I2T | 54.50 | 81.65 | 90.00 | 50.22 | 33.25 |

Comparison against the QDQ proxy:

| Metric | QDQ proxy | Board full-gallery | Delta |
|---|---:|---:|---:|
| T2I R@1 | 50.85 | 50.20 | -0.65 |
| I2T R@1 | 52.90 | 54.50 | +1.60 |

Conclusion: the v8 vision binary is board-verified. The full board run drops only `0.65` T2I R@1 from the QDQ proxy and still passes the deployment target (`50.20 >= 50.0`).

---

## 6. Current Deployment Status

| Component | Status | Best current result | Next gate |
|---|---|---|---|
| Vision W8A8 | **Board PASS** | board vision T2I R@1 `50.20`, I2T R@1 `54.50`; runtime `33.05 ms/image` | none for vision-only |
| Text W8A8 | **QDQ PASS, link PASS** | text-isolation QDQ T2I R@1 `51.65`; fidelity `0.9949 / 0.9912` | run text context binary on RB3 and compare fidelity |
| Both W8A8 | **QDQ PASS** | both-INT8 QDQ T2I R@1 `50.25`, I2T R@1 `52.95` | run both-INT8 retrieval from RB3 outputs |

The key remaining result is **both-INT8 board retrieval**, after collecting text QNN embeddings on RB3.
