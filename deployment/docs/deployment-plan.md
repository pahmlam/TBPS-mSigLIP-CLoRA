# mSigLIP Edge Deployment Plan — Qualcomm RB3 Gen2

> **Status:** In progress | **Target device:** Qualcomm RB3 Gen2 (QCS6490, HTP V68, 4 GB RAM, Ubuntu 24.04 aarch64)
> **Source checkpoint:** `artifacts/models/checkpoints/epoch=56-val_score=52.28.ckpt` (VN3K R@1 = 52.28%, LoRA + Curriculum Circle Loss, seed 2400)
> **Last updated:** 2026-05-27

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

### Current status: vision HTP runtime works; real-cal INT8 fidelity failed

**What works:**
- Checkpoint analysis (`deployment/scripts/analyze_checkpoint.py`)
- LoRA merge + FP16 export (`deployment/scripts/lora_fp16/export.py`) → `model_fp32.pt`, `model_fp16.pt`, `config.yaml`
- ONNX export with external weights (`deployment/scripts/onnx/export.py`) → `vision_onnx/`, `text_onnx/`
- ONNX → FP16 conversion with FP16 I/O (`deployment/scripts/onnx/to_fp16.py`) → `vision_onnx_fp16/`
- **AI Hub INT8 compile for vision encoder** (job `jgkr7qwn5`) → QNN context binary for HTP V68 ✅
- **RB3 QNN HTP runtime** → `vn3k_test_10` runs successfully at ~22.25 ms/image NetRun average, ~20.72 ms accelerator average, 4 HVX threads.
- **Baseline sanity tooling** → `deployment/scripts/qnn/compare_qnn_with_pytorch.py` compares QNN outputs to PyTorch/ONNX on the exact same raw inputs.
- **Real VN3K calibration upload** → dataset `d7x5gzne9`, 500 train samples, accepted by AI Hub.
- **Real-calibration Python API compile/link** → job `jpr9v62vp` produced `vision_encoder_calib500.bin`, and the binary runs on RB3 HTP without NaN/Inf.

**What still needs to be done:**
- **Diagnose INT8 fidelity failure**: `vision_encoder_calib500.bin` runs, but QNN-vs-PyTorch `cosine_l2_mean = 0.1300` on `vn3k_test_10`, lower than dummy-cal `0.1727`. This binary is not retrieval-usable.
- **Isolate where fidelity is lost**: compare the QDQ ONNX quantized model against PyTorch first; only then decide whether to change PTQ/calibration settings or debug QNN I/O/runtime.
- Text encoder: same INT8 compile pipeline on AI Hub
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
| **A. INT8 quantization (proper)** | Collect calibration data → `submit_quantize_job` + `submit_compile_and_link_jobs` | ⚠️ Compile/runtime verified with real calibration, but fidelity failed (`cosine_l2_mean = 0.1300`). |
| **B. INT8 dummy calibration (sanity check)** | `qai-hub ... --calibration_data none` | ✅ Done (job `jgkr7qwn5`). Garbage accuracy, pipeline only. |
| **C. Target GPU instead of DSP** | `--compute_unit gpu` with FP16 model | ⏭ Fallback if INT8 accuracy is unacceptable |
| **D. CPU only** | `--target_runtime onnx` → run ONNX Runtime on device | ⏭ Last resort |

---

## 5. Recommended Next Steps

### Phase 1 — Validate the HTP pipeline end-to-end ✅ DONE

1. ~~**Run Option B (INT8 dummy calibration)** on vision_encoder to confirm HTP compilation succeeds with INT I/O.~~ ✅ Job `jgkr7qwn5` compiled successfully. QNN context binary for HTP V68 produced.
2. ~~**In parallel: run Option C (GPU FP16)** to have a fallback that works.~~ Deferred — HTP path works, no need for GPU fallback yet.

### Phase 2 — On-device benchmark with dummy-cal model ✅ DONE, not accuracy-usable

1. ~~**Download the compiled `.bin` from AI Hub** — job `jgkr7qwn5`, asset `mqyov9dxm`.~~
2. ~~**Transfer to RB3**: `scp vision_encoder.bin rb3:~/sigm/`~~
3. ~~**Benchmark latency & throughput**~~:
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

Observed:
- `vn3k_test_10` produces 10 valid `Result_*/output_0.raw` files.
- Each output is 768 float32 values after qnn-net-run dequantization.
- No NaN/Inf; outputs are not byte-identical.
- Profile from `qnn-profiling-data_1.log`: NetRun average ~22.25 ms/image; accelerator average ~20.72 ms/image.
- Fidelity check fails: QNN vs PyTorch/ONNX cosine mean ~0.1727. Treat this binary as runtime proof only.

### Phase 2b — Real-calibration attempt via deprecated CLI ❌ FAILED

1. Real VN3K train calibration raws were prepared:
   ```bash
   venv/bin/python deployment/scripts/qnn/prepare_vn3k_vision_inputs.py \
     --dataset-root VN3K \
     --split train \
     --selection random \
     --seed 2400 \
     --num-samples 500 \
     --output-dir artifacts/deployment/qnn_inputs/vn3k_train_calib_500 \
     --path-mode relative
   ```
2. Calibration dataset was uploaded using the QAI Hub Python API:
   ```bash
   venv/bin/python deployment/scripts/qnn/upload_qaihub_calibration_dataset.py \
     --input-dir artifacts/deployment/qnn_inputs/vn3k_train_calib_500 \
     --name msiglip-vision-vn3k-train-calib-500
   ```
   Dataset ID: `d7x5gzne9`.
3. Deprecated CLI compile attempt `j5wx6x63p` failed:
   ```bash
   venv/bin/qai-hub submit-compile-job \
     --model artifacts/deployment/exports/exported_model/vision_onnx/ \
     --device "Dragonwing RB3 Gen 2 Vision Kit" \
     --compile_options " --target_runtime qnn_context_binary --quantize_full_type int8" \
     --input_specs '{"image": ((1, 3, 256, 256), "float32")}' \
     --calibration_data <DATASET_ID> \
     --name "mSigLIP-vision-int8-vn3k-calib-500" \
     --wait
   ```
   The log shows:
   - `qairt-converter` received `--preserve_io_datatype image output_0` twice.
   - `qairt-quantizer` also received `--preserve_io_datatype image output_0`.
   - Quantization itself completed successfully.
   - Context-binary creation failed with: `Tensor 'image' has a floating-point type which is not supported by the targeted device.`

Conclusion: dataset `d7x5gzne9` is usable, but this CLI path is not. It preserves FP I/O, so HTP rejects the model before any on-board test can happen.

### Phase 3 — Proper INT8 quantization ✅ compiled, ❌ fidelity failed

1. **Collect calibration data**
   - Done for vision: `artifacts/deployment/qnn_inputs/vn3k_train_calib_500/`.
   - Uploaded dataset: `d7x5gzne9`.
   - Mirror for text: sample 200-500 Vietnamese captions from training, tokenize with `SiglipTokenizer`, save input_ids + attention_mask pairs.

2. **Quantize & compile with the new API**
   - Vision: `qai_hub.submit_quantize_job(...)` → `qai_hub.submit_compile_and_link_jobs(...)`
   - Text: same pipeline.
   - Hard requirement: converter/quantizer/link logs must not preserve FP I/O for `image`.
   - Helper script for the current vision path:
     ```bash
     venv/bin/python deployment/scripts/qnn/submit_qaihub_quantize_compile.py \
       --model artifacts/deployment/exports/exported_model/vision_onnx/ \
       --calibration-data d7x5gzne9 \
       --wait \
       --download artifacts/deployment/qnn_inputs/vision_encoder_calib500.bin
     ```
   - The helper resolves `--calibration-data d7x5gzne9` through `hub.get_dataset(...)`. Passing the raw string directly to `submit_quantize_job` makes `qai_hub 0.48.0` treat it as a local file path.
   - Quantize job `jp13422k5` showed that AI Hub also rejects the original dynamic-batch ONNX at quantize time. The helper now creates `artifacts/deployment/exports/exported_model/vision_onnx_static/` and rewrites input `image` from `['batch_size', 3, 256, 256]` to `[1, 3, 256, 256]` before upload.
   - Compile job `jpr9v62vp` completed with the static QDQ model. The converter command did not include `--preserve_io_datatype`; local output `artifacts/deployment/qnn_inputs/vision_encoder_calib500.bin` exists and is ~90 MB.
   - Board execution on `vn3k_test_10_calib500` completed, but fidelity failed:
     ```text
     cosine_l2_mean = 0.1300
     cosine_l2_min/max = 0.0799 / 0.1774
     l2_l2_mean = 1.3189
     any_qnn_nan/inf = false
     ```

3. **On-board sanity check**
   - Done for `vision_encoder_calib500.bin`.
   - Result: runtime pass, fidelity fail.
   - Do not proceed to `vn3k_test_100`, full VN3K R@1, or text encoder yet.

### Phase 3b — Diagnose calibrated INT8 fidelity (current next step)

1. **Compare QDQ ONNX vs PyTorch**
   - Download/export the quantized QDQ ONNX model from the successful quantize step.
   - Run the same `vn3k_test_10` raw inputs through QDQ ONNX locally and compare against PyTorch.
   - Helper:
     ```bash
     venv/bin/python deployment/scripts/qnn/compare_onnx_with_pytorch.py \
       --onnx-model artifacts/deployment/qnn_inputs/<downloaded_qdq_onnx_or_dir> \
       --model-dir artifacts/deployment/exports/exported_model \
       --input-dir artifacts/deployment/qnn_inputs/vn3k_test_10 \
       --precision fp32 \
       --json artifacts/deployment/qnn_outputs/vn3k_test_10_calib500/qdq_vs_pytorch_summary.json \
       --csv artifacts/deployment/qnn_outputs/vn3k_test_10_calib500/qdq_vs_pytorch.csv
     ```

2. **Branch based on the QDQ result**
   - If QDQ ONNX is already low: fix PTQ/calibration settings first (calibration size/selection, quantization granularity, sensitive op exclusions, or mixed precision).
   - If QDQ ONNX is close to PyTorch: debug QNN compile/runtime/I/O (native input/output encodings, selected output tensor, QNN CPU vs HTP comparison).

3. **Resume pipeline only after vision fidelity improves**
   - Minimum gate: QNN-vs-PyTorch cosine must be dramatically higher than both `0.1727` dummy-cal and `0.1300` calib500.
   - Only then run `vn3k_test_100`, full VN3K R@1, and text encoder compile.

4. **Compile text encoder**
   - Start only after vision INT8 fidelity is usable.
   - Use the same non-preserving I/O quantization/compile path.

5. **Write an RB3-first modular retrieval demo**
   - `deployment/demo/` provides image/video source adapters, QNN vision encoding on board, local spool/vector-store preflight, and swappable backend/text-service interfaces.
   - Local fake/ONNX checks are preflight only; deployment acceptance must run on RB3 with QNN.

### Phase 4 — Demo & documentation

1. Build a minimal CLI demo on the device (`retrieve.py`) that takes an image + Vietnamese text and returns top-5 matches.
2. Update `deployment/docs/benchmark-rp.md` with real mSigLIP numbers.
3. Write a teardown/reproducibility guide in `deployment/docs/deploy-to-rb3.md`.

---

## 6. Open Questions / Risks

| # | Question | How to resolve |
|---|----------|---------------|
| 1 | Does INT8 quantization preserve R@1 within acceptable range for SigLIP-style attention? | Phase 3 sanity/accuracy checks — measure empirically after a real-calibration context binary is produced without preserved FP I/O. |
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
