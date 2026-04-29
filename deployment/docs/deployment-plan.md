# mSigLIP Edge Deployment Plan — Qualcomm RB3 Gen2

> **Status:** In progress | **Target device:** Qualcomm RB3 Gen2 (QCS6490, HTP V68, 4 GB RAM, Ubuntu 24.04 aarch64)
> **Source checkpoint:** `epoch=56-val_score=52.28.ckpt` (VN3K R@1 = 52.28%, LoRA + Curriculum Circle Loss, seed 2400)
> **Last updated:** 2026-04-15

---

## 1. Objective

Deploy the mSigLIP TBPS model on the RB3 Gen2 edge device to run Vietnamese text-based person retrieval **on-device** — no cloud calls. Two independent encoders must run locally:

- **Vision encoder** — input `(1, 3, 256, 256)` image → `(1, 768)` L2-normalized embedding
- **Text encoder** — input `(1, 64)` `input_ids` + `attention_mask` → `(1, 768)` L2-normalized embedding

Cosine similarity between the two embeddings is the retrieval score. Targeted use case: 1 image vs. 1 text query at a time (batch=1).

---

## 2. Hardware Capabilities & Constraints (verified)

### 2.1 Compute units on QCS6490

| Unit | Clock / Cores | Supported I/O types | Verified | Use for |
|------|---------------|---------------------|----------|---------|
| **CPU** — 4×Cortex-A78 @ 2.7 GHz + 4×Cortex-A55 @ 1.9 GHz | aarch64 w/ NEON + `fphp` (FP16) | FP32, FP16, INT8 | ✅ ONNX Runtime works | Fallback, debugging |
| **GPU** — Adreno 643 (OpenCL) | — | FP32, FP16 | ⚠️ validated but not exercised with mSigLIP | Mid-speed fallback if HTP fails |
| **DSP/HTP** — Hexagon 770 / HTP V68 | — | **INT8, INT16 only at I/O boundary**; FP16 allowed for internal compute | ✅ snpe-platform-validator passed | Target runtime (fastest) |

### 2.2 Memory budget

- Total RAM: 5.2 GB, OS reserves ~1.2 GB → **~4 GB available** for inference.
- mSigLIP (FP32): 1.437 GB weights + ~718 MB activations ≈ 2.15 GB → fits, tight.
- mSigLIP (FP16): 720 MB weights + ~360 MB activations ≈ 1.08 GB → comfortable.
- mSigLIP (INT8): 360 MB weights + ~180 MB activations ≈ 540 MB → very comfortable.

### 2.3 What's already installed on the device

| Component | Status |
|-----------|--------|
| SNPE runtime libs (`libsnpe1`, `libqnn1` 2.45.40) | ✅ Installed |
| SNPE CLI tools (`snpe-net-run`, `snpe-throughput-net-run`, etc.) | ✅ Installed |
| HTP V68/V73/V75/V79 libraries | ✅ Installed |
| PyTorch 2.10.0+cpu, ONNX Runtime 1.23.2 | ✅ Installed (in `~/sigm/venv`) |
| `snpe-onnx-to-dlc` conversion tool | ❌ **Not available** — requires full SDK (x86_64 Linux only, not ARM, not macOS) |

### 2.4 Implication — why we use Qualcomm AI Hub

The conversion tool `snpe-onnx-to-dlc` is the missing piece. Options:

- **Full SNPE SDK** → x86_64 Linux only. Mac M2 dev machine & ARM64 RB3 can't run it.
- **Qualcomm AI Hub (cloud)** → accepts ONNX model via `qai-hub` CLI, compiles in the cloud, returns a QNN context binary (`.bin`). Works from any OS. ✅ Chosen approach.

---

## 3. Current Deployment Pipeline (verified working up to step 3)

```
Step 0        Step 1                  Step 2                      Step 3                     Step 4                 Step 5
━━━━━━        ━━━━━━                  ━━━━━━                      ━━━━━━                     ━━━━━━                 ━━━━━━
Train    →   Merge LoRA + FP16   →   Export ONNX (static)    →   Quantize for HTP       →   AI Hub compile    →   Deploy to RB3
             (local, lora_fp16/)     (local, onnx/export.py)     (local, onnx/to_fp16.py    (cloud)                (snpe-net-run)
                                                                  or INT8 calibration)
✅           ✅                       ✅                           🚧 WIP                     🚧 WIP                 ⏭
epoch=56    model_fp16.pt           vision_onnx/                 vision_onnx_fp16/           vision_encoder.bin     DSP/HTP inference
(1.4 GB)    model_fp32.pt           text_onnx/                   OR INT8-quantized ONNX      text_encoder.bin
            config.yaml             (with .onnx.data                                         (+ calibration if INT8)
                                     external weights)
```

### Current status: stuck at step 3 → step 4

**What works:**
- Checkpoint analysis (`deployment/scripts/analyze_checkpoint.py`)
- LoRA merge + FP16 export (`deployment/scripts/lora_fp16/export.py`) → `model_fp32.pt`, `model_fp16.pt`, `config.yaml`
- ONNX export with external weights (`deployment/scripts/onnx/export.py`) → `vision_onnx/`, `text_onnx/`
- ONNX → FP16 conversion with FP16 I/O (`deployment/scripts/onnx/to_fp16.py`) → `vision_onnx_fp16/`

**What doesn't work yet (see §4):**
- Qualcomm AI Hub compile job for `qnn_context_binary` with FP16 I/O — rejected by HTP compiler.

---

## 4. Root Cause: HTP Rejects Floating-Point I/O

After 9 attempts (see `aihub-experiments.md` for detailed log), we've confirmed:

**HTP V68 on QCS6490 requires INT8 or INT16 tensors at the I/O boundary.** Internal compute can use FP16 via fused ops, but the tensors crossing the CPU↔DSP boundary must be integer-quantized. This is a hardware/driver-level constraint, not a bug or flag we can override.

**Why DSPs have this constraint:**
- Tensor transfers between CPU and DSP go through DMA channels optimized for integer blocks.
- HTP's instruction set loads tensors in quantized INT8/INT16 tiles; floats must be dequantized on-chip from INT storage.
- Keeping I/O integer avoids costly FP↔INT conversions at tensor boundaries.

**Options to move forward:**

| Path | Flow | Pros | Cons |
|------|------|------|------|
| **A. INT8 quantization (proper)** | Collect calibration data → `qai-hub` with `--quantize_full_type int8 --calibration_data <id>` | Fastest inference (~18-30x), correct path for production | Needs ~100-500 calibration images + accuracy evaluation |
| **B. INT8 dummy calibration (sanity check)** | `qai-hub ... --calibration_data none` | Quick compile-path verification | Garbage accuracy, only useful to validate the pipeline |
| **C. Target GPU instead of DSP** | `--compute_unit gpu` with FP16 model | No quantization, no calibration | ~3-5x speedup only (vs. 18-30x on HTP), GPU shares RAM with app |
| **D. CPU only** | `--target_runtime onnx` → run ONNX Runtime on device | Simplest, no AI Hub | Slowest (~100 ms/image for MobileNet-class; likely ~1-2 s for mSigLIP) |

---

## 5. Recommended Next Steps

### Phase 1 — Validate the HTP pipeline end-to-end (this week)

1. **Run Option B (INT8 dummy calibration)** on vision_encoder to confirm HTP compilation succeeds with INT I/O.
   - If it works → proceed to Phase 2.
   - If it fails → pivot to Option C (GPU) and deliver a working deployment, then revisit HTP.

2. **In parallel: run Option C (GPU FP16)** to have a fallback that works.

### Phase 2 — Proper INT8 quantization (next week)

1. **Collect calibration data**
   - Sample 200–500 images from VN3K training split, resize to 256×256, normalize with the same mean/std as training (0.5, 0.5, 0.5).
   - Save as a Qualcomm AI Hub calibration dataset via `qai-hub upload-dataset`.
   - Mirror for text: sample 200-500 Vietnamese captions from training, tokenize with `SiglipTokenizer`, save input_ids + attention_mask pairs.

2. **Quantize & compile**
   - Vision: `qai-hub submit-compile-job ... --quantize_full_type int8 --calibration_data <vision_cal_id>`.
   - Text: same but with `--calibration_data <text_cal_id>`.

3. **Accuracy check (critical)**
   - Download quantized ONNX models from AI Hub job results.
   - Run on host with ONNX Runtime against VN3K test set.
   - Compute R@1 and compare against FP32 baseline (52.28%).
   - Acceptance threshold: R@1 ≥ 48% (within 5 pp). If lower, investigate per-layer sensitivity with AIMET or exclude attention softmax from INT8.

### Phase 3 — On-device benchmarking (after Phase 2)

1. Transfer compiled `.bin` files to RB3: `scp vision_encoder.bin text_encoder.bin rb3:~/sigm/`
2. Benchmark latency & throughput:
   ```bash
   snpe-throughput-net-run --container vision_encoder.bin --use_htp --perf_profile high_performance --duration 30
   ```
3. Write an end-to-end retrieval demo: image + text query → embeddings → cosine sim → top-k results.
4. Compare against ONNX Runtime CPU baseline and the expected DSP speedup from `benchmark-rp.md` (18-30x for MobileNet-class).

### Phase 4 — Demo & documentation

1. Build a minimal CLI demo on the device (`retrieve.py`) that takes an image + Vietnamese text and returns top-5 matches.
2. Update `deployment/docs/benchmark-rp.md` with real mSigLIP numbers.
3. Write a teardown/reproducibility guide in `deployment/docs/deploy-to-rb3.md`.

---

## 6. Open Questions / Risks

| # | Question | How to resolve |
|---|----------|---------------|
| 1 | Does INT8 quantization preserve R@1 within acceptable range for SigLIP-style attention? | Phase 2 step 3 — measure empirically. Paper reports only minor degradation for CLIP/SigLIP with PTQ, but we have no VN3K data point yet. |
| 2 | Will the text embedding table (~730 MB in FP32, ~180 MB in INT8) fit in HTP memory? | Check with `qai-hub submit-profile-job` after compile — reports on-chip memory usage. |
| 3 | How much does quantization cost for cross-modal alignment specifically? | Implement A/B on a holdout set — the `logit_scale` and `logit_bias` parameters may need to stay FP16. |
| 4 | LoRA was merged at FP32 — does post-merge quantization lose the LoRA benefit? | Compare quantized (with merged LoRA) vs quantized (without LoRA, base SigLIP only) on VN3K R@1. Expected: merged LoRA retains ~3–5 pp advantage. |

---

## 7. Reference — Related Documents

| Path | Purpose |
|------|---------|
| `deployment/docs/aihub-experiments.md` | **Running log** of every qai-hub compile attempt — always update after each run |
| `deployment/docs/system.md` | RB3 hardware specs (verified on-device) |
| `deployment/docs/experiment.md` | Benchmark methodology for proxy models (MobileNetV2, ResNet18) |
| `deployment/docs/benchmark-rp.md` | Proxy model results (PyTorch CPU vs ONNX Runtime) + SDK status |
| `deployment/docs/end-to-end-system-design.md` | Proposed end-to-end product/system architecture after model deployment is complete |
| `deployment/README.md` | Pipeline quick-reference + AI Hub commands |
| `docs/knowledge.md` §4, §5 | Vietnamese knowledge base entries on Qualcomm SDK & ONNX format |
| `deployment/scripts/onnx/to_fp16.py` | Pre-quantizer for FP16 I/O (step 3) |
