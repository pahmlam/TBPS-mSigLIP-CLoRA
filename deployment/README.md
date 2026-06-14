# Edge Deployment & Model Compression

This folder contains the mSigLIP edge deployment pipeline for the Qualcomm RB3
Gen2 / QCS6490 target. The current focus is the **vision encoder QNN/HTP path**:
runtime is proven, but quantization fidelity is not deployable yet.

## Current Status

| Area | Status | Notes |
|---|---|---|
| FP32/FP16 export | PASS | LoRA merge, stripped checkpoint export, and ONNX export work |
| Static ONNX control | PASS | Vision ONNX matches PyTorch with cosine near `1.0` |
| QNN HTP runtime | PASS | Vision context binary runs on RB3 HTP; no NaN/Inf |
| INT8/QDQ fidelity | FAIL | AI Hub QDQ remains far below gate |
| Best QDQ diagnostic | PASS local only | `all_weights + blocks 4-11 float` reaches `0.9703/0.9400`, but is not deployable |
| Text encoder | BLOCKED | Do not compile text until vision fidelity passes |
| Active branch | FOLLOW-UP | Phase H10: QNN-native / quantizer-level strategy for blocks 4-11 |

Latest locked conclusions are in:

- `deployment/docs/journal/[deploy-plan]-2026-06-06.md`
- `deployment/docs/journal/[deploy]-2026-06-13.md`
- `deployment/docs/journal/[deploy]-2026-06-14.md`

Do not treat `_float` QDQ surgery candidates as deployable. They are diagnostic
upper bounds only; previous `_float` candidates link-failed on HTP because the
context graph still contained internal floating-point tensors.

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

## Pipeline

```text
checkpoint.ckpt
  -> lora_fp16/export.py
  -> exported_model/{model_fp32.pt, model_fp16.pt, config.yaml}
  -> onnx/export.py
  -> exported_model/{vision_onnx, text_onnx}
  -> QDQ / quantize diagnostics
  -> AI Hub compile/link or QNN-native toolchain
  -> qnn-net-run on RB3 HTP
  -> QNN-vs-PyTorch fidelity
  -> retrieval benchmark
```

Current rule: keep working on the vision encoder until QDQ and QNN fidelity
pass. Text encoder and full retrieval are blocked until then.

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
