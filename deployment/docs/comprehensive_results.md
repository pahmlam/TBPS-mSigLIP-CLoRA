# Comprehensive Deployment Results

This document summarizes the final W8A8 deployment result for mSigLIP on Qualcomm RB3 Gen2 / HTP v68, then records the supporting proxy and ablation results.

## 0. Reading Guide

Use this hierarchy when citing results:

| Priority | Result type | Meaning |
|---:|---|---|
| 1 | **Direct board both-INT8** | Official thesis deployment number: both encoders executed as INT8 on RB3. |
| 2 | **Board isolation** | One encoder runs on RB3 INT8 while the other stays FP32; useful to locate drift. |
| 3 | **Off-board QDQ proxy** | AI Hub quantized ONNX evaluated with ONNX Runtime; useful before compile/link/board runs. |
| 4 | **Historical ablations** | Explains why the final v9 recipe was chosen. |

The final deployment path is:

```text
image -> vision v9 QNN context binary on HTP
text  -> RB3 CPU token-embedding lookup -> split-text QNN context binary on HTP
retrieval -> raw dot-product ranking, matching LitTBPS metrics
```

---

## 1. Final Thesis Result

**Official deploy number:** direct board both-INT8, full VN3K test set.

Artifact: `artifacts/deployment/qnn_runs/both_int8_board_r1.json`

| Direction | R@1 | R@5 | R@10 | mAP | mINP | Drop vs paper R@1 |
|---|---:|---:|---:|---:|---:|---:|
| **T2I** | **50.35** | **77.82** | **86.50** | **55.80** | **49.28** | **-1.93** |
| **I2T** | **54.20** | **80.50** | **89.20** | **50.26** | **33.83** | **-1.10** |

The result passes the deployment gate (`T2I R@1 >= 50.0`). The reporting baseline is the paper/historical VN3K T2I R@1 `52.28`; local FP32 sanity is `52.40`.

---

## 2. Final Board Components

### 2.1 Vision v9 Board

Artifact: `artifacts/deployment/qnn_runs/rotated_w8a8_learned_qat_v9_gallery_2000/board_vision_r1.json`

This is **vision-only board isolation**: image embeddings from the v9 QNN context binary, text embeddings from FP32 PyTorch.

| Direction | R@1 | R@5 | R@10 | mAP | mINP |
|---|---:|---:|---:|---:|---:|
| T2I | 50.35 | 77.55 | 86.55 | 55.73 | 49.21 |
| I2T | 54.55 | 82.10 | 89.35 | 50.58 | 33.66 |

Runtime from `artifacts/deployment/qnn_runs/rotated_w8a8_learned_qat_v9_gallery_2000/profile.txt`:

| Metric | Value |
|---|---:|
| Average NetRun latency | 32.54 ms/image |
| Average QNN execute time | 32.49 ms/image |
| Average accelerator time | 30.97 ms/image |
| Min / max NetRun latency | 30.37 / 36.21 ms |
| Throughput | 24.29 FPS |

### 2.2 Text Split-Encoder Board

Artifact: `artifacts/deployment/qnn_runs/text_w8a8_learned_qat_v8_f32mask/board_text_r1.json`

This is **text-only board isolation**: text embeddings from the split-text QNN context binary, image embeddings from FP32 PyTorch.

| Direction | R@1 | R@5 | R@10 | mAP | mINP |
|---|---:|---:|---:|---:|---:|
| T2I | 51.30 | 79.43 | 87.90 | 56.97 | 50.46 |
| I2T | 54.80 | 81.00 | 88.60 | 51.14 | 34.72 |

Runtime from `artifacts/deployment/qnn_runs/onboard_text/profile.txt`:

| Metric | Value |
|---|---:|
| Average NetRun latency | 7.87 ms/query |
| Average QNN execute time | 7.83 ms/query |
| Average accelerator time | 6.76 ms/query |
| Min / max NetRun latency | 7.13 / 12.47 ms |
| Throughput | 74.75 queries/sec |

Board fidelity for the split-text transformer is `0.9951 / 0.9926` cosine mean/min on the 10-query smoke set, matching the text QDQ proxy.

### 2.3 Peak RAM Probe

Peak RAM was measured on RB3 Gen2 with `deployment/scripts/qnn/measure_board_peak_ram.sh`
at `INTERVAL=0.02`. The wrapper records host-process peak RSS/HWM from
`/proc/<pid>/status` and the peak drop in system `MemAvailable` from
`/proc/meminfo`. The system-level drop is the more useful board-memory indicator,
because QNN HTP execution allocates memory outside the host process RSS.

| Branch / step | Input scope | Exit code | Process peak VmRSS | Process peak VmHWM | MemAvailable start | MemAvailable min | MemAvailable end | System peak delta | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Vision v9 HTP context | 1 image | 0 | 101.88 MB | 102.00 MB | 2578.66 MB | 2333.75 MB | 2500.66 MB | 244.91 MB | Valid |
| Text CPU token-embedding lookup | 4000 queries | 0 | 57.88 MB | 58.26 MB | 2584.40 MB | 2458.74 MB | 2539.64 MB | 125.66 MB | Valid |
| Split-text HTP context | 1 query | 0 | 95.25 MB | 95.25 MB | 2590.65 MB | 2265.49 MB | 2458.86 MB | 325.16 MB | Valid |

The valid split-text HTP probe used the board context binary at
`artifacts/deployment/bin/text_encoder_split.bin`. An earlier failed probe pointed
to an empty/missing copy under `artifacts/deployment/runtime/split_text_w8a8/`;
that failed startup measurement is intentionally excluded from the table.

---

## 3. Off-Board QDQ Proxies

QDQ proxy means the AI Hub quantized ONNX graph is evaluated off-board with ONNX Runtime. It is not the official board number, but it is the cheapest way to inspect quantization quality before compile/link.

### 3.1 Vision v9 QDQ

Artifact: `artifacts/deployment/runtime/rotated_w8a8_learned_qat_v9/retrieval_r1.json`

This is **vision-only QDQ isolation**: image encoder v9 QDQ, text encoder FP32.

| Direction | R@1 | R@5 | R@10 | mAP | mINP |
|---|---:|---:|---:|---:|---:|
| T2I | 50.98 | 77.30 | 86.53 | 55.86 | 49.10 |
| I2T | 54.20 | 80.85 | 89.00 | 50.14 | 33.27 |

QDQ-to-board drift for the final vision tower:

| Metric | v9 QDQ | v9 board | Delta |
|---|---:|---:|---:|
| T2I R@1 | 50.98 | 50.35 | -0.63 |
| I2T R@1 | 54.20 | 54.55 | +0.35 |

### 3.2 Text QDQ

Artifact source: `artifacts/deployment/runtime/rotated_w8a8_learned_qat_v9/both_int8_qdq_r1.json`

This is **text-only QDQ isolation** from the same v9 proxy run: text encoder QDQ, image encoder FP32.

| Direction | R@1 | R@5 | R@10 | mAP | mINP |
|---|---:|---:|---:|---:|---:|
| T2I | 51.65 | 79.43 | 87.68 | 57.08 | 50.50 |
| I2T | 55.55 | 81.05 | 88.75 | 51.26 | 34.72 |

Text QDQ smoke fidelity is `0.9949 / 0.9912` cosine mean/min. The finite attention mask keeps the `scores+mask` QDQ scale bounded (`max scale = 0.3523`) and avoids the old `-FLT_MAX` collapse.

### 3.3 Both-INT8 QDQ v9

Artifact: `artifacts/deployment/runtime/rotated_w8a8_learned_qat_v9/both_int8_qdq_r1.json`

This is the updated off-board proxy: vision v9 QDQ + text QDQ.

| Direction | R@1 | R@5 | R@10 | mAP | mINP | Drop vs paper R@1 |
|---|---:|---:|---:|---:|---:|---:|
| T2I | 50.63 | 77.05 | 86.22 | 55.71 | 48.99 | -1.65 |
| I2T | 53.90 | 79.90 | 88.55 | 50.03 | 33.54 | -1.40 |

Comparison with the official board result:

| Metric | QDQ proxy v9 | Direct board | Delta |
|---|---:|---:|---:|
| T2I R@1 | 50.63 | 50.35 | -0.28 |
| I2T R@1 | 53.90 | 54.20 | +0.30 |

The old both-INT8 QDQ proxy (`50.25 / 52.95`) used vision v8 QDQ and is now historical.

---

## 4. Why Text Uses Split-Encoder

The full-graph text W8A8 binary links on HTP v68 but is **not usable on board**: its output ignores `input_ids`. Real `input_ids` and all-zero `input_ids` produce bit-identical board outputs when the mask is held fixed.

Root cause: the dynamic embedding lookup

```text
Gather(token_embedding.weight[250000 x 768], input_ids)
```

is faithful in static ONNX and QDQ, but fails after QNN compile/link on HTP v68. This is a graph/runtime lowering limit, not a QAT or calibration failure.

The deployable split keeps the heavy compute on HTP:

| Stage | Location | Role |
|---|---|---|
| Token embedding lookup | RB3 CPU | table read from INT8 token embedding |
| Transformer + head | HTP v68 | split-text QNN context binary |

Control check: real vs zero `inputs_embeds` changes board outputs by max-abs `3.66–5.28`, proving the split graph uses the embedding input.

---

## 5. Historical Ablations

These results explain the path to v9. They are not the final deployment number.

### 5.1 Vision QDQ Evolution

All rows below are off-board QDQ proxies with text FP32.

| Version | Main change | Cosine mean/min | T2I R@1 | I2T R@1 | Interpretation |
|---|---|---:|---:|---:|---|
| Rotation only | mean-preserving rotation + W8A8 | 0.8975 / 0.8747 | 45.42 | 49.40 | Fail |
| V3 | EMA observer QAT | 0.9353 / 0.9190 | 48.20 | 52.30 | Improved, still below gate |
| V4 | + pooling-head fake quant | 0.9364 / 0.9091 | 48.50 | 52.95 | Board-runnable but below gate |
| V5 | + per-linear fake quant | 0.9437 / 0.9311 | 49.25 | 53.40 | Near gate |
| V6 | + attention-matmul fake quant | 0.9491 / 0.9266 | 49.30 | 53.85 | Random-rotation ceiling |
| V7 | v6 coverage + cosine LR | 0.9485 / 0.9083 | 48.38 | 53.05 | Regressed |
| V8 | learned rotation + v6 QAT recipe | 0.9606 / 0.9447 | 50.85 | 52.90 | First QDQ pass |
| **V9** | larger rotation search + 25-epoch QAT | see v9 artifact | **50.98** | **54.20** | Final vision recipe |

### 5.2 v8 vs v9 Board Vision

| Metric | v8 board | v9 board | Delta |
|---|---:|---:|---:|
| T2I R@1 | 50.20 | 50.35 | +0.15 |
| I2T R@1 | 54.50 | 54.55 | +0.05 |
| Runtime | 33.05 ms/image | 32.54 ms/image | -0.51 ms |

v9 is a recipe refinement rather than a new method: same learned mean-preserving rotation theory, larger rotation search/calibration budget, best-Q selection, and longer QAT.

---

## 6. Artifact Map

| Artifact | Meaning |
|---|---|
| `artifacts/deployment/qnn_runs/both_int8_board_r1.json` | Official final board both-INT8 result |
| `artifacts/deployment/qnn_runs/rotated_w8a8_learned_qat_v9_gallery_2000/board_vision_r1.json` | Vision v9 board isolation |
| `artifacts/deployment/qnn_runs/text_w8a8_learned_qat_v8_f32mask/board_text_r1.json` | Text split-encoder board isolation |
| `artifacts/deployment/runtime/rotated_w8a8_learned_qat_v9/retrieval_r1.json` | Vision v9 QDQ isolation |
| `artifacts/deployment/runtime/rotated_w8a8_learned_qat_v9/both_int8_qdq_r1.json` | Updated both-INT8 QDQ proxy v9 |
| `artifacts/deployment/qnn_runs/rotated_w8a8_learned_qat_v9_gallery_2000/profile.txt` | Vision v9 board runtime |
| `artifacts/deployment/qnn_runs/onboard_text/profile.txt` | Split-text board runtime |
