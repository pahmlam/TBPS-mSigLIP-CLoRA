# Edge Deployment & Model Compression

This folder contains the mSigLIP edge deployment pipeline for the Qualcomm RB3
Gen2 / QCS6490 target. The **vision encoder INT8 path is DEPLOYED and running on
HTP v68** via rotation-based equalization; the text encoder is the next workstream.

## Current Status (2026-06-15)

| Area | Status | Notes |
|---|---|---|
| FP32/FP16 export (LoRA merge) | PASS | `lora_fp16/export.py` merges LoRA → `exported_model` |
| Rotation equalization | PASS | mean-preserving Q + fold γ/β; output-invariant 1.0; residual concentration 252x→5.3x |
| ONNX export (opset 20) | PASS | fused Gelu (no `Pow(x³)`) + fused LayerNorm; static control ≈ 1.0 |
| **Vision W8A8 QDQ fidelity** | **PASS (near)** | `cosine 0.90` — rotation made all-INT8 viable (was 0.14 collapse) |
| **Vision compile/link on v68** | **PASS** | all-INT8 links on HTP v68; `vision_encoder.bin` 89.7 MB |
| **Vision on HTP board** | **PASS** | board fidelity `0.898` = QDQ; **22.5 FPS, 34 ms/img** |
| **Vision R@1 (gate)** | **FAIL** | T2I R@1 `45.42` (vision-only INT8, text FP32) vs `52.28`, gate ≥ 48; proxy `0.90` cosine insufficient → need higher fidelity |
| Text encoder | TODO (blocked) | hold until vision passes gate; full INT8+INT8 ≤ `45.42` |

Key learnings (why this works on v68):

- **v68 rejects 16-bit activation (A16)** for LayerNorm/attention matmul (needs
  v73+). So W8A16 (fidelity `0.9997`) **links-fail** on v68 → must use all-INT8 W8A8.
- Plain W8A8 collapses (`0.14`) due to **residual-stream activation concentration**.
- **tanh-GELU `Pow(x³)`** cubic (exposed by opset-18 decompose) was a separate
  killer → fixed by opset-20 fused `Gelu`.
- **Rotation** (mean-preserving orthogonal Q, fold into weights) spreads the
  concentration so per-tensor INT8 works; **keep fused LayerNorm** (don't convert
  to RMSNorm, which re-exposes `Pow(x²)` to quant).

Latest locked conclusions / full pipeline:

- `deployment/docs/journal/[deploy]-2026-06-15.md` (sections 5-13: rotation method,
  M1-M5 results, and the **full reproducible pipeline from ckpt → .bin in section 13**)

Do not treat `_float` QDQ surgery candidates as deployable (diagnostic only;
link-fail on internal float). Do not use W8A16 on v68 (A16 needs v73).

## Directory Map

```text
deployment/
  scripts/
    analyze_checkpoint.py              # Check checkpoint size/state/RAM hints
    inference_test.py                  # Host inference smoke test
    lora_fp16/export.py                # Merge LoRA and export FP32/FP16 state
    onnx/export.py                     # Export vision/text ONNX directories
    onnx/to_fp16.py                    # Optional ONNX FP16 conversion
    qnn/
      prepare_vn3k_vision_inputs.py    # Build raw image inputs + input_list
      upload_qaihub_calibration_dataset.py
      submit_qaihub_quantize_compile.py
      submit_qaihub_compile_link.py
      compare_onnx_with_pytorch.py
      compare_qnn_with_pytorch.py
      qdq_surgery.py
      analyze_qdq_encodings.py
      tune_qdq_activation_encodings.py
      train_vision_quant_robust.py
      audit_qnn_native_env.py
  config/qnn/                          # HTP/QNN runtime JSON configs
  demo/                                # Modular demo-system scaffold
  docs/
    journal/                           # Canonical deployment logs
    deployment-plan.md                 # Older high-level plan; verify against journal plan
    system.md                          # RB3 hardware notes
    experiment.md                      # Proxy benchmark guide
    benchmark-rp.md                    # Proxy benchmark results
  hardware_profiling/                  # RB3 proxy benchmark scripts
  deploy_utils.py
```

Generated artifacts and logs belong under `artifacts/deployment/`.

## Target Device

| Component | Specification |
|---|---|
| SoC | Qualcomm QCS6490 |
| CPU | 4x Cortex-A78 + 4x Cortex-A55 |
| GPU | Adreno 643 |
| DSP/HTP | Hexagon 770 / HTP V68 |
| RAM | About 4 GB usable for inference |
| Runtime target | QNN context binary on HTP |

Important hardware constraint: HTP requires integer tensors at the model I/O
boundary. Internal floating-point tensors can also break context linking,
depending on the graph pattern.

## Deployment Gates

Do not advance a candidate unless it passes the relevant gate.

| Gate | Threshold | Meaning |
|---|---:|---|
| Static ONNX vs PyTorch | `cosine_l2_mean >= 0.999` | Export/preprocess control |
| QDQ ONNX vs PyTorch mean | `>= 0.95` | Candidate worth compile/link |
| QDQ ONNX vs PyTorch min | `>= 0.90` | No severe sample drift |
| QNN vs PyTorch after link | `>= 0.90` | Candidate worth wider RB3 benchmark |
| Full retrieval | `T2I R@1 >= 48.0` | Minimum deploy target vs `52.28` FP32 baseline |

Diagnostic compile exception: AI Hub-native QDQ from a QAT model may be compiled
if it reaches `mean >= 0.93` and `min >= 0.88`. This exception does **not**
apply to `_float`, ORT QDQ, or INT16 surgery patterns that have already
link-failed.

## Pipeline (current, rotation-based INT8 for v68)

```text
epoch=56-val_score=52.28.ckpt        # LoRA-finetuned Lightning checkpoint
  -> [1] lora_fp16/export.py          # MERGE LoRA (merge_and_unload) -> base weights
         exported_model/{model_fp32.pt, model_fp16.pt, config.yaml}
  -> [2] qnn/rotate_vision_encoder.py # mean-preserving rotation + fold gamma/beta
         exported_model_rotated/{model_fp32.pt, config.yaml}   # output-invariant 1.0
  -> [3] qnn/export_rotated_vision_onnx.py --opset 20  # fused Gelu + fused LayerNorm
         exported_model_rotated/vision_onnx/{vision_encoder.onnx,.data}
  -> [4] qnn/submit_qaihub_quantize_compile.py  # W8A8, calib d7jzjy1m2, device RB3 Gen2
         vision_encoder.bin                      # all-INT8, links on HTP v68 (89.7 MB)
  -> [5] qnn-net-run on RB3 HTP -> compare_qnn_with_pytorch -> retrieval R@1
```

Full step-by-step with commands and verification: see
`deployment/docs/journal/[deploy]-2026-06-15.md` section 13.

Why each step exists: LoRA merge is mandatory (ckpt is LoRA-finetuned); rotation
spreads activation concentration so all-INT8 W8A8 (the only v68-deployable scheme)
keeps fidelity; opset-20 fuses Gelu/LayerNorm to avoid `Pow(x³)`/`Pow(x²)` being
quantized. Swapping to the 53.00 model requires re-running from step [1].

## Quick Commands

### 1. Analyze a checkpoint

```bash
python deployment/scripts/analyze_checkpoint.py \
  --ckpt artifacts/models/checkpoints/epoch=56-val_score=52.28.ckpt
```

### 2. Merge LoRA and export FP32/FP16 state

```bash
python deployment/scripts/lora_fp16/export.py \
  --ckpt artifacts/models/checkpoints/epoch=56-val_score=52.28.ckpt \
  --output-dir artifacts/deployment/exports/exported_model
```

### 3. Export ONNX

```bash
python deployment/scripts/onnx/export.py \
  --model-dir artifacts/deployment/exports/exported_model \
  --precision fp32
```

Use FP32 ONNX for stability unless a specific diagnostic requires FP16.

### 4. Prepare raw VN3K vision inputs

```bash
python deployment/scripts/qnn/prepare_vn3k_vision_inputs.py \
  --dataset-root VN3K \
  --split test \
  --selection first \
  --num-samples 10 \
  --output-dir artifacts/deployment/qnn_inputs/vn3k_test_10 \
  --path-mode relative
```

For calibration:

```bash
python deployment/scripts/qnn/prepare_vn3k_vision_inputs.py \
  --dataset-root VN3K \
  --split train \
  --selection random \
  --seed 2400 \
  --num-samples 2000 \
  --output-dir artifacts/deployment/qnn_inputs/vn3k_train_calib_2000 \
  --path-mode relative
```

### 5. Upload calibration data to Qualcomm AI Hub

```bash
python deployment/scripts/qnn/upload_qaihub_calibration_dataset.py \
  --input-dir artifacts/deployment/qnn_inputs/vn3k_train_calib_2000 \
  --name msiglip-vision-vn3k-train-calib-2000
```

Known calibration dataset:

```text
d7jzjy1m2 / msiglip-vision-vn3k-train-calib-2000
```

### 6. Quantize and compile/link with the newer AI Hub API

Use this helper instead of deprecated `qai-hub submit-compile-job
--quantize_full_type ...`, because the deprecated CLI can preserve FP I/O and
make HTP reject the model.

```bash
python deployment/scripts/qnn/submit_qaihub_quantize_compile.py \
  --model artifacts/deployment/exports/exported_model/vision_onnx \
  --calibration-data d7jzjy1m2 \
  --wait \
  --download artifacts/deployment/qnn_inputs/vision_encoder.bin
```

For QDQ diagnostics only:

```bash
python deployment/scripts/qnn/submit_qaihub_quantize_compile.py \
  --model artifacts/deployment/exports/exported_model/vision_onnx \
  --calibration-data d7jzjy1m2 \
  --quantize-only \
  --wait \
  --download artifacts/deployment/runtime/qaihub_qdq/model.onnx
```

### 7. Compare QDQ ONNX against PyTorch

```bash
python deployment/scripts/qnn/compare_onnx_with_pytorch.py \
  --onnx-model artifacts/deployment/runtime/qaihub_qdq \
  --model-dir artifacts/deployment/exports/exported_model \
  --input-dir artifacts/deployment/qnn_inputs/vn3k_test_10 \
  --precision fp32 \
  --json artifacts/deployment/runtime/qaihub_qdq/qdq_vs_pytorch_summary.json \
  --csv artifacts/deployment/runtime/qaihub_qdq/qdq_vs_pytorch.csv
```

Only compile/link candidates that pass the QDQ gate, except for the documented
diagnostic exception above.

### 8. Run a QNN context binary on RB3

```bash
qnn-net-run \
  --backend "$QNN_LIB/libQnnHtp.so" \
  --retrieve_context artifacts/deployment/qnn_inputs/vision_encoder.bin \
  --config_file deployment/config/qnn/htp_config_245.json \
  --input_list artifacts/deployment/qnn_inputs/vn3k_test_10/input_list.txt \
  --output_dir artifacts/deployment/qnn_outputs/vn3k_test_10 \
  --profiling_level basic \
  --perf_profile high_performance
```

Use `qnn-net-run`, not `snpe-net-run`, for QNN context binaries.

### 9. Compare QNN outputs against PyTorch

```bash
python deployment/scripts/qnn/compare_qnn_with_pytorch.py \
  --qnn-output-dir artifacts/deployment/qnn_outputs/vn3k_test_10 \
  --model-dir artifacts/deployment/exports/exported_model \
  --input-dir artifacts/deployment/qnn_inputs/vn3k_test_10 \
  --precision fp32 \
  --json artifacts/deployment/qnn_outputs/vn3k_test_10/qnn_vs_pytorch_summary.json \
  --csv artifacts/deployment/qnn_outputs/vn3k_test_10/qnn_vs_pytorch.csv
```

### 10. Audit QNN/QAIRT native toolchain

```bash
python deployment/scripts/qnn/audit_qnn_native_env.py \
  --json artifacts/deployment/runtime/qnn_native/env_audit.json
```

Latest local Mac audit found no QNN/QAIRT native tools. Phase H10 needs a server
or machine with the Qualcomm AI Stack / QNN SDK installed.

## Latest Diagnostic Results

| Candidate | Result | Decision |
|---|---:|---|
| Static ONNX vs PyTorch, QAT v5 | `1.000000 / 0.9999998` | Export control PASS |
| AI Hub Lite-MP INT8 QDQ, QAT v5 | `0.244236 / 0.203228` | FAIL, no compile/link |
| AI Hub default INT8 QDQ, QAT v5 | `0.252687 / 0.211948` | FAIL, no compile/link |
| AI Hub native W8A16 QDQ, QAT v5 | `0.155494 / 0.115508` | FAIL, no compile/link |
| `all_weights + blocks 4-9 float` | `0.947507 / 0.913112` | Near-pass diagnostic |
| `all_weights + blocks 4-10 float` | `0.964700 / 0.930359` | PASS diagnostic only |
| `all_weights + blocks 4-11 float` | `0.970312 / 0.939976` | Best diagnostic only |

The best diagnostic proves the error is tied to interaction between weight QDQ
and activation QDQ in encoder blocks 4-11, but it does not provide a deployable
QNN graph.

## Documentation Convention

- Current multi-day deployment checklist:
  `deployment/docs/journal/[deploy-plan]-2026-06-06.md`
- Dated deployment logs:
  `deployment/docs/journal/[deploy]-YYYY-MM-DD.md`
- Demo-system logs:
  `deployment/docs/journal/[demo-system]-YYYY-MM-DD.md`
- Stable concepts such as ONNX, QNN, HTP, PTQ, and QAT:
  `docs/knowledge.md`
- Completed code/config/docs changes:
  `changelog/deployment/changelog.md` after user confirmation.

Do not write AI Hub job logs or QDQ/QNN fidelity results into
`deployment/docs/aihub-experiments.md`; it is legacy.

## Hardware Profiling

Proxy hardware profiling scripts remain under `deployment/hardware_profiling/`.
They are useful for RB3 environment checks, but they are not acceptance tests for
mSigLIP. Deployment acceptance requires mSigLIP QNN fidelity and retrieval
metrics.

```bash
cd deployment/hardware_profiling
./run_all.sh
```
