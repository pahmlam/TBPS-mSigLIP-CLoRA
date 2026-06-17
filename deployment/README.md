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
| **Vision R@1 (gate)** | **PASS** | Current best is QAT v5: T2I R@1 `49.25`, I2T R@1 `53.40`, QDQ cosine `0.9437 / 0.9311`; QAT v4 remains board-verified at T2I R@1 `48.50` |
| **Vision QAT `.bin` on board** | **PASS** | QAT v4 `.bin` board-verified on v68: fidelity `0.9363` ≈ QDQ, `32.7 ms` / `22.88 FPS` / ~90 MB |
| Text encoder | Pending | start after vision stretch/acceptance; replicate recipe (rotation + QAT + W8A8) for full both-INT8 on-device |

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

- `deployment/docs/journal/[deploy-master].md` is the canonical deployment journal:
  it consolidates all dated deploy logs, the full ckpt-to-bin pipeline, AI Hub/QNN
  job history, board results, v5 best result, and the v6 next plan.

Do not treat `_float` QDQ surgery candidates as deployable (diagnostic only;
link-fail on internal float). Do not use W8A16 on v68 (A16 needs v73).

## Model Footprint

Full `model_fp32.pt` is the whole TBPS (vision + text); the deployed
`vision_encoder.bin` is the vision encoder only. Params dedup'd (TBPS aliases
`vision_model`/`text_model` to `backbone.*`).

| Component | Params | FP32 | FP16 | INT8 |
|---|---|---|---|---|
| vision_model | 92.9 M | 372 MB | 186 MB | **93 MB** (deployed `.bin` 89.7) |
| text_model | 277.7 M | 1111 MB | 555 MB | 278 MB |
| projection + other | 1.2 M | 5 MB | 2 MB | 1 MB |
| **TOTAL** | **371.8 M** | **1487 MB** | 744 MB | 372 MB |

Text is 75% of params — token embedding alone is `250000 × 768` = 192 M params
(768 MB FP32), driven by the 250k multilingual vocab. On 4 GB RAM, text FP32
(1.1 GB) is the real cost; quantizing text to INT8 (~278 MB) is the next lever
(blocked on vision passing the R@1 gate). Details: journal section 17.

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
    journal/                           # Master deploy journal + archived dated logs
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

Full step-by-step with commands, job history, and verification: see
`deployment/docs/journal/[deploy-master].md`.

Why each step exists: LoRA merge is mandatory (ckpt is LoRA-finetuned); rotation
spreads activation concentration so all-INT8 W8A8 (the only v68-deployable scheme)
keeps fidelity; opset-20 fuses Gelu/LayerNorm to avoid `Pow(x³)`/`Pow(x²)` being
quantized. Swapping to the 53.00 model requires re-running from step [1].

## QAT Iteration Workflow (per-round commands)

Rotation alone gives only `T2I R@1 45.42` (gate fail). Quantization-aware fine-tune
(QAT) of the rotated model closes the gap. Each QAT round is one pass of the loop
below. Set `V` to the round name (e.g. `qat_v5`); the AI Hub steps (4) are the only
ones that cost a cloud job — steps 1–3, 5 are local/free.

**One-time prep** (shared by all rounds):

```bash
# [A] merge LoRA -> exported_model/  (see Quick Commands #2)
# [B] mean-preserving rotation -> exported_model_rotated/  (teacher + QAT base)
python deployment/scripts/qnn/rotate_vision_encoder.py \
  --model-dir artifacts/deployment/exports/exported_model \
  --output-dir artifacts/deployment/exports/exported_model_rotated \
  --input-dir artifacts/deployment/qnn_inputs/vn3k_test_10 --seed 2400 --skip-r2
# [C] prepare full-train calib inputs for QAT (4302 images)
python deployment/scripts/qnn/prepare_vn3k_vision_inputs.py \
  --dataset-root VN3K --split train --selection random --seed 2400 \
  --num-samples 4302 --output-dir artifacts/deployment/qnn_inputs/vn3k_train_all_4302 \
  --path-mode relative
```

**Per round** (`V=qat_v5` shown; run [1] on a GPU box, [4] is the cloud job):

```bash
# [1] QAT distillation (teacher=rotated FP32, student=fake-quant). LONG (GPU).
python deployment/scripts/qnn/train_vision_quant_robust.py \
  --model-dir artifacts/deployment/exports/exported_model_rotated \
  --train-input-dir artifacts/deployment/qnn_inputs/vn3k_train_all_4302 \
  --val-input-dir   artifacts/deployment/qnn_inputs/vn3k_test_100 \
  --output-dir      artifacts/deployment/exports/exported_model_rotated_qat_v5 \
  --device cuda --batch-size 48 --epochs 15 --lr 1e-5 \
  --start-layer 0 --end-layer 11 \
  --fake-quant-granularity per_tensor --fake-quant-observer ema --ema-momentum 0.99 \
  --quant-head --quant-linears

# [2] export rotated/QAT ONNX, opset 20 (fused Gelu/LayerNorm). FAST/local.
python deployment/scripts/qnn/export_rotated_vision_onnx.py \
  --model-dir artifacts/deployment/exports/exported_model_rotated_qat_v5

# [3] static control gate (ONNX vs PyTorch, must be ~1.0). FAST/local.
python deployment/scripts/qnn/compare_onnx_with_pytorch.py \
  --onnx-model artifacts/deployment/exports/exported_model_rotated_qat_v5/vision_onnx \
  --model-dir  artifacts/deployment/exports/exported_model_rotated_qat_v5 --rotated \
  --input-dir  artifacts/deployment/qnn_inputs/vn3k_test_10 --precision fp32

# [4] AI Hub W8A8 quantize-only -> QDQ ONNX. CLOUD JOB (log the job id).
python deployment/scripts/qnn/submit_qaihub_quantize_compile.py \
  --model artifacts/deployment/exports/exported_model_rotated_qat_v5/vision_onnx \
  --calibration-data d7jzjy1m2 --weights-dtype int8 --activations-dtype int8 \
  --quantize-only --wait \
  --download-quantized artifacts/deployment/runtime/rotated_w8a8_qat_v5/job_qdq_onnx/model.onnx

# [5] decisive numbers: QDQ cosine + retrieval R@1 (gate >= 48). FAST/local.
python deployment/scripts/qnn/compare_onnx_with_pytorch.py \
  --onnx-model artifacts/deployment/runtime/rotated_w8a8_qat_v5/job_qdq_onnx \
  --model-dir  artifacts/deployment/exports/exported_model \
  --input-dir  artifacts/deployment/qnn_inputs/vn3k_test_10 --precision fp32
python deployment/scripts/qnn/eval_retrieval_quantized_vision.py \
  --qdq-onnx artifacts/deployment/runtime/rotated_w8a8_qat_v5/job_qdq_onnx \
  --json     artifacts/deployment/runtime/rotated_w8a8_qat_v5/retrieval_r1.json
```

If R@1 passes the gate, deploy with the full flow (drop `--quantize-only`, use
`--download .../vision_encoder.bin`) then `qnn-net-run` on the board (Quick Commands #8–9).

**What each round changed, and the result** (the `--fake-quant-*` / `--quant-*`
flags in step [1] are the only differences):

| Round | QAT flags added | QDQ cosine | T2I R@1 |
|---|---|---:|---:|
| rotation only | — (no QAT) | 0.8975 | 45.42 |
| qat_v1 | per-sample fake-quant | 0.9223 | 46.92 |
| qat_v2 | `--fake-quant-granularity per_tensor` | 0.9281 | 47.80 |
| qat_v3 | `--fake-quant-observer ema` | 0.9353 | 48.20 |
| qat_v4 | `--quant-head` | 0.9364 | 48.50 |
| qat_v5 | `--quant-linears` | 0.9437 | 49.25 |

> Dated commands, AI Hub job IDs, and full results per round: journal
> `deployment/docs/journal/[deploy-master].md`. Method + math:
> `deployment/docs/w8a8_qat_rotated.md`.

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

## Current Key Results

| Candidate | Result | Decision |
|---|---:|---|
| Rotation-only W8A8 | QDQ `0.8975 / 0.8747`, T2I R@1 `45.42` | Links/runs, retrieval gate FAIL |
| QAT v3 | QDQ `0.9353 / 0.919`, T2I R@1 `48.20` | First retrieval gate PASS |
| QAT v4 | QDQ `0.9364 / 0.9091`, T2I R@1 `48.50`; board `0.9363 / 0.9068` | Current board-verified deploy binary |
| QAT v5 | QDQ `0.9437 / 0.9311`, T2I R@1 `49.25`, I2T R@1 `53.40` | Current best accuracy candidate |
| QAT v6 planned | `--quant-attention` | Next stretch path toward T2I R@1 >= 50 |

Historical failed diagnostics and rejected branches are consolidated in `deployment/docs/journal/[deploy-master].md`.

## Documentation Convention

- Canonical deployment/model-compression journal: `deployment/docs/journal/[deploy-master].md`
- Demo-system logs: `deployment/docs/journal/[demo-system]-YYYY-MM-DD.md`
- Stable concepts such as ONNX, QNN, HTP, PTQ, and QAT: `docs/knowledge.md`
- Completed code/config/docs changes: `changelog/deployment/changelog.md` after user confirmation.

Write new AI Hub job logs and QDQ/QNN fidelity results into `deployment/docs/journal/[deploy-master].md`. Do not write them into `deployment/docs/aihub-experiments.md`; it is legacy.

## Hardware Profiling

Proxy hardware profiling scripts remain under `deployment/hardware_profiling/`.
They are useful for RB3 environment checks, but they are not acceptance tests for
mSigLIP. Deployment acceptance requires mSigLIP QNN fidelity and retrieval
metrics.

```bash
cd deployment/hardware_profiling
./run_all.sh
```
