# Edge Deployment & Model Compression

Code and documentation for optimizing and deploying the mSigLIP model on edge devices (Qualcomm RB3 Gen2).

## Structure

```
deployment/
├── scripts/                       # mSigLIP deployment pipeline
│   ├── analyze_checkpoint.py      # Shared: Analyze checkpoint (size, RAM, compatibility)
│   ├── inference_test.py          # Shared: Test inference on target device
│   ├── lora_fp16/                 # Step 1: LoRA merge + FP16 export
│   │   └── export.py              #   Merge LoRA → FP16/FP32 state dict
│   └── onnx/                      # Step 2: ONNX conversion
│       └── export.py              #   FP16/FP32 state dict → ONNX
│
├── hardware_profiling/            # RB3 hardware capability testing (proxy models)
│   ├── benchmark.py               # PyTorch CPU vs ONNX Runtime (MobileNetV2, ResNet18)
│   ├── snpe_benchmark.py          # Qualcomm SNPE benchmark (requires DLC models)
│   ├── collect_sysinfo.sh         # Collect system information
│   ├── install_deps.sh            # Install dependencies on device
│   └── run_all.sh                 # Run full hardware profiling suite
│
├── docs/                          # Documentation
│   ├── deployment-plan.md         # ** START HERE ** — full deployment plan, status, next steps
│   ├── end-to-end-system-design.md # Proposed end-to-end product/system architecture
│   ├── aihub-experiments.md       # Legacy redirect to dated deployment journal
│   ├── journal/                   # Dated deployment logs and decisions
│   ├── system.md                  # RB3 Gen2 hardware specifications
│   ├── experiment.md              # Benchmark step-by-step guide
│   └── benchmark-rp.md            # Hardware benchmark results
│
├── config/qnn/                    # QNN/HTP runtime config JSON files
├── deploy_utils.py                # Shared utilities (TeeLogger)
└── README.md
```

All scripts in `scripts/` and `hardware_profiling/` write generated logs under `artifacts/deployment/`.

## Documentation Convention

- `deployment/docs/deployment-plan.md` tracks the current deployment state and next technical steps.
- `deployment/docs/journal/[deploy]-YYYY-MM-DD.md` stores dated deployment results: AI Hub jobs, QNN/QDQ fidelity, RB3 runtime, artifacts, and decisions. This is the canonical place for new `qai-hub` / `qai_hub` job logs.
- Stable concepts such as QNN, ONNX, HTP, and quantization terminology belong in `docs/knowledge.md`.
- Before adding deployment documentation, classify whether it is current state, dated journal, durable knowledge, or changelog, and confirm the target file unless the user explicitly requested the update.
- Use the template in `deployment/docs/journal/README.md` for new dated deployment entries.

## Deployment Pipeline

```
Training          →  Step 1: FP16       →  Step 2: ONNX     →  Step 3: QNN Compile    →  Deploy (RB3 Gen2)
━━━━━━━━━━━━━        ━━━━━━━━━━━━━━━       ━━━━━━━━━━━━━       ━━━━━━━━━━━━━━━━━━        ━━━━━━━━━━━━━━━━━━━
trainer.py           lora_fp16/export.py   onnx/export.py      Qualcomm AI Hub           qnn-net-run
epoch=56.ckpt  →     model_fp16.pt   →    *_onnx/        →    *.bin (QNN context)  →   DSP/HTP inference
(1.4 GB)             (~740 MB)             (dir w/ weights)    (compiled for QCS6490)
```

## Target Device

| Component | Specification |
|-----------|--------------|
| **SoC** | Qualcomm QCS6490 |
| **CPU** | 4x Cortex-A78 @ 2.7GHz + 4x Cortex-A55 @ 1.9GHz |
| **GPU** | Adreno 643 |
| **DSP** | Hexagon 770 |
| **RAM** | 5.2 GB (available ~4 GB) |

## Usage

All scripts auto-log to `artifacts/deployment/logs/` with timestamps.

### 1. Analyze checkpoint
```bash
python deployment/scripts/analyze_checkpoint.py --ckpt path/to/checkpoint.ckpt
```

### 2. Export to FP16 (merge LoRA + strip optimizer)
```bash
python deployment/scripts/lora_fp16/export.py \
    --ckpt artifacts/models/checkpoints/epoch=56-val_score=52.28.ckpt \
    --output-dir artifacts/deployment/exports/msiglip_lora
```

### 3. Convert to ONNX
```bash
python deployment/scripts/onnx/export.py \
    --model-dir artifacts/deployment/exports/msiglip_lora \
    --precision fp32              # fp32 recommended for ONNX stability
```

### 4. Compile for DSP/HTP via Qualcomm AI Hub

Requires a Qualcomm AI Hub account ([aihub.qualcomm.com](https://aihub.qualcomm.com/)).

```bash
pip install qai-hub
qai-hub configure --api_token YOUR_TOKEN
```

```bash
# Vision encoder (pass directory, not .onnx file — includes external weights)
qai-hub submit-compile-job \
    --model artifacts/deployment/exports/msiglip_lora/vision_onnx/ \
    --device "Dragonwing RB3 Gen 2 Vision Kit" \
    --compile_options " --target_runtime qnn_context_binary" \
    --name "mSigLIP-vision" \
    --wait

# Text encoder
qai-hub submit-compile-job \
    --model artifacts/deployment/exports/msiglip_lora/text_onnx/ \
    --device "Dragonwing RB3 Gen 2 Vision Kit" \
    --compile_options " --target_runtime qnn_context_binary" \
    --name "mSigLIP-text" \
    --wait
```

Available `--target_runtime` options: `qnn_context_binary` (DSP/HTP, recommended), `qnn_dlc` (legacy SNPE), `onnx`, `tflite`, `precompiled_qnn_onnx`.

### 5. Test inference (ONNX Runtime on CPU)
```bash
python deployment/scripts/inference_test.py \
    --model-dir artifacts/deployment/exports/msiglip_lora \
    --dtype fp16 \
    --dataset-root /path/to/VN3K
```

### 6. Run on DSP/HTP (on RB3)
```bash
# Transfer compiled QNN context binary to RB3, then:
qnn-net-run \
    --backend "$QNN_LIB/libQnnHtp.so" \
    --retrieve_context artifacts/deployment/qnn_inputs/vision_encoder.bin \
    --config_file deployment/config/qnn/htp_config_245.json \
    --input_list artifacts/deployment/qnn_inputs/vn3k_vision/input_list.txt \
    --output_dir artifacts/deployment/qnn_runs/vision_results \
    --perf_profile high_performance
```

### 7. Hardware profiling on RB3 (proxy models)
```bash
# SSH to RB3, copy hardware_profiling/ to device
cd ~/sigm
./run_all.sh
```
