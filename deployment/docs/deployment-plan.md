# mSigLIP Edge Deployment Plan — Qualcomm RB3 Gen2

> **Status:** In progress | **Target device:** Qualcomm RB3 Gen2 (QCS6490, HTP V68, 4 GB RAM, Ubuntu 24.04 aarch64)
> **Source checkpoint:** `artifacts/models/checkpoints/epoch=56-val_score=52.28.ckpt` (VN3K R@1 = 52.28%, LoRA + Curriculum Circle Loss, seed 2400)
> **Last updated:** 2026-05-17

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

## 3. Current Deployment Pipeline (vision encoder through step 4)

```
Step 0        Step 1                  Step 2                      Step 3                     Step 4                 Step 5
━━━━━━        ━━━━━━                  ━━━━━━                      ━━━━━━                     ━━━━━━                 ━━━━━━
Train    →   Merge LoRA + FP16   →   Export ONNX (static)    →   Quantize for HTP       →   AI Hub compile    →   Deploy to RB3
             (local, lora_fp16/)     (local, onnx/export.py)     (local, onnx/to_fp16.py    (cloud)                (qnn-net-run)
                                                                  or INT8 calibration)
✅           ✅                       ✅                           ✅ Done (INT8)              ✅ Vision done         ⏭
epoch=56    model_fp16.pt           vision_onnx/                 INT8-quantized ONNX         vision_encoder.bin     DSP/HTP inference
(1.4 GB)    model_fp32.pt           text_onnx/                   (via AI Hub compile)        text_encoder.bin
            config.yaml             (with .onnx.data                                         (needs same pipeline
                                     external weights)                                        as vision)
```

### Current status: step 4 complete for vision encoder, step 5 next

**What works:**
- Checkpoint analysis (`deployment/scripts/analyze_checkpoint.py`)
- LoRA merge + FP16 export (`deployment/scripts/lora_fp16/export.py`) → `model_fp32.pt`, `model_fp16.pt`, `config.yaml`
- ONNX export with external weights (`deployment/scripts/onnx/export.py`) → `vision_onnx/`, `text_onnx/`
- ONNX → FP16 conversion with FP16 I/O (`deployment/scripts/onnx/to_fp16.py`) → `vision_onnx_fp16/`
- **AI Hub INT8 compile for vision encoder** (job `jgkr7qwn5`) → QNN context binary for HTP V68 ✅

**What still needs to be done:**
- Text encoder: same INT8 compile pipeline on AI Hub
- Download compiled `.bin` to RB3 device and run `qnn-net-run` benchmarks
- Replace dummy calibration with real VN3K calibration data for production accuracy
- Accuracy evaluation: target R@1 ≥ 48% (vs FP32 baseline 52.28%)

**Deprecation warnings:**
- `--quantize_full_type` is deprecated → use `submit_quantize_job` API
- `--target_runtime qnn_context_binary` is deprecated → use `submit_compile_and_link_jobs` API

---

## 4. Root Cause (RESOLVED): HTP Rejects Floating-Point I/O

After 11 attempts (see `aihub-experiments.md` for detailed log), the root cause is confirmed and resolved:

**HTP V68 on QCS6490 requires INT8 or INT16 tensors at the I/O boundary.** Internal compute can use FP16 via fused ops, but the tensors crossing the CPU↔DSP boundary must be integer-quantized. This is a hardware/driver-level constraint, not a bug or flag we can override.

**The key blocker was `--preserve_io_datatype` auto-injection.** When using `--quantize_full_type int8`, AI Hub automatically injects `--preserve_io_datatype` into the qairt-converter and qairt-quantizer commands, keeping I/O tensors in their original FP type. This causes HTP to reject the model at the context-binary stage. Removing this flag (or using the newer `submit_quantize_job` API that doesn't auto-inject it) allows I/O to be quantized to INT8 alongside internal weights/activations, which HTP accepts.

**Resolution:** Job `jgkr7qwn5` compiled successfully with INT8 quantization (dummy calibration). Vision encoder QNN context binary produced for HTP V68.

**Remaining paths for reference:**

| Path | Flow | Status |
|------|------|--------|
| **A. INT8 quantization (proper)** | Collect calibration data → `submit_quantize_job` + `submit_compile_and_link_jobs` | ✅ Pipeline verified (dummy cal). Need real calibration data for production accuracy. |
| **B. INT8 dummy calibration (sanity check)** | `qai-hub ... --calibration_data none` | ✅ Done (job `jgkr7qwn5`). Garbage accuracy, pipeline only. |
| **C. Target GPU instead of DSP** | `--compute_unit gpu` with FP16 model | ⏭ Fallback if INT8 accuracy is unacceptable |
| **D. CPU only** | `--target_runtime onnx` → run ONNX Runtime on device | ⏭ Last resort |

---

## 5. Recommended Next Steps

### Phase 1 — Validate the HTP pipeline end-to-end ✅ DONE

1. ~~**Run Option B (INT8 dummy calibration)** on vision_encoder to confirm HTP compilation succeeds with INT I/O.~~ ✅ Job `jgkr7qwn5` compiled successfully. QNN context binary for HTP V68 produced.
2. ~~**In parallel: run Option C (GPU FP16)** to have a fallback that works.~~ Deferred — HTP path works, no need for GPU fallback yet.

### Phase 2 — On-device benchmarking with dummy-cal model (now)

1. **Download the compiled `.bin` from AI Hub** — job `jgkr7qwn5`, asset `mqyov9dxm`.
2. **Transfer to RB3**: `scp vision_encoder.bin rb3:~/sigm/`
3. **Benchmark latency & throughput**:
   ```bash
   $QNN_BIN/qnn-net-run \
       --backend $QNN_LIB/libQnnHtp.so \
      --retrieve_context artifacts/deployment/qnn_inputs/vision_encoder.bin \
      --config_file deployment/config/qnn/htp_config_245.json \
      --input_list artifacts/deployment/qnn_inputs/vn3k_vision/input_list.txt \
       --output_dir artifacts/deployment/qnn_runs/vision_bench \
       --profiling_level basic \
       --perf_profile high_performance
   ```
   `vision_encoder.bin` is a QNN context binary, not an SNPE DLC. Use `qnn-net-run` with the matching QAIRT/QNN 2.45 runtime and HTP skel libraries; `snpe-net-run` will fail with DLC reader errors for this artifact.
4. **Compile text encoder** — same pipeline as vision:
   ```bash
   qai-hub submit-compile-job \
       --model artifacts/deployment/exports/msiglip_lora/text_onnx/ \
       --device "Dragonwing RB3 Gen 2 Vision Kit" \
       --compile_options " --target_runtime qnn_context_binary --quantize_full_type int8" \
       --input_specs '{"input_ids": ((1, 64), "int64"), "attention_mask": ((1, 64), "int64")}' \
       --calibration_data none \
       --name "mSigLIP-text-int8-dummy" \
       --wait
   ```
5. **Write an RB3-first modular retrieval demo**: `deployment/demo/` provides image/video source adapters, QNN vision encoding on board, local spool/vector-store preflight, and swappable backend/text-service interfaces. Local fake/ONNX checks are preflight only; deployment acceptance must run on RB3 with QNN.

### Phase 3 — Proper INT8 quantization (after Phase 2)

1. **Collect calibration data**
   - Sample 200–500 images from VN3K training split, resize to 256×256, normalize with the same mean/std as training (0.5, 0.5, 0.5).
   - Save as a Qualcomm AI Hub calibration dataset via `qai-hub upload-dataset`.
   - Mirror for text: sample 200-500 Vietnamese captions from training, tokenize with `SiglipTokenizer`, save input_ids + attention_mask pairs.

2. **Quantize & compile** (use new API to avoid `--preserve_io_datatype` auto-injection)
   - Vision: `qai-hub submit-quantize-job` → `qai-hub submit-compile-and-link-jobs`
   - Text: same pipeline.

3. **Accuracy check (critical)**
   - Download quantized ONNX models from AI Hub job results.
   - Run on host with ONNX Runtime against VN3K test set.
   - Compute R@1 and compare against FP32 baseline (52.28%).
   - Acceptance threshold: R@1 ≥ 48% (within 5 pp). If lower, investigate per-layer sensitivity with AIMET or exclude attention softmax from INT8.

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
