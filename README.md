# A Hard Negative-Aware Optimization for Multilingual Text-Based Person Search

Official implementation of **"A Hard Negative-Aware Optimization for Multilingual Text-Based Person Search"**: mSigLIP + LoRA + Curriculum Circle Loss for multilingual text-based person search, with ongoing edge deployment on Qualcomm RB3 Gen2.

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="PyTorch Lightning" src="https://img.shields.io/badge/PyTorch%20Lightning-2.x-792EE5?style=flat-square&logo=lightning&logoColor=white">
  <img alt="Hydra" src="https://img.shields.io/badge/Hydra-config-89B8CD?style=flat-square">
  <img alt="LoRA" src="https://img.shields.io/badge/PEFT-LoRA-2F855A?style=flat-square">
  <img alt="RB3 QNN" src="https://img.shields.io/badge/RB3%20Gen2-QNN%20W8A8-111827?style=flat-square">
</p>

## Status at a Glance

| Track | Current state | Next step |
|---|---|---|
| **Paper result** | LoRA + Curriculum Circle Loss reaches **52.28% T2I R@1** on VN3K and **59.35% T2I R@1** on PRW-TPS-CN | Reported headline and deployment source checkpoint |
| **Accuracy extension** | Part-Token Alignment + Attn/FFN LoRA r32 reaches **53.00% T2I R@1** and **53.25% I2T R@1** on VN3K (single seed) | Complete CUHK-PEDES and PRW-TPS-CN ablations |
| **Edge deployment** | Vision encoder runs all-INT8 **W8A8 on HTP v68**; best QDQ/retrieval candidate is **QAT v6: 49.30 T2I R@1 / 53.85 I2T R@1**; QAT v4 is board-verified | Quantize text encoder, then run end-to-end board retrieval |

## Contents

- [Why This Matters](#why-this-matters)
- [Method Overview](#method-overview)
- [Main Results](#main-results)
- [Accuracy Extension](#accuracy-extension)
- [Quick Start](#quick-start)
- [Deployment on RB3 Gen2](#deployment-on-rb3-gen2)
- [Repository Map](#repository-map)
- [Documentation Map](#documentation-map)
- [Contact](#contact)

## Why This Matters

Multilingual Text-Based Person Search (TBPS) retrieves person images from natural-language descriptions. It is difficult in low-resource and multilingual settings because the model must separate visually similar people using subtle cross-modal cues.

This work starts from `siglip-base-patch16-256-multilingual` and adds:

- **Cross-modal Circle Loss** for explicit hard-negative mining.
- **Curriculum hard-mining** so Circle Loss is introduced after early global alignment stabilizes.
- **LoRA adapters** for parameter-efficient optimization: **5.9M trainable parameters, about 1.57% of the 376M-parameter base model**.
- **Optional local alignment and deployment extensions**: Part-Token Alignment for accuracy ablations, and QNN W8A8 deployment for Qualcomm RB3 Gen2.

## Method Overview

The framework keeps the mSigLIP dual encoder and adds a hard-negative branch on top of the baseline image-text objectives.

![Framework Architecture](figures/framework.png)

**Core objective.** The default training objective combines the baseline mSigLIP losses with a curriculum-weighted Cross-modal Circle Loss:

```text
N-ITC/MVS + 0.1 * C-ITC + 0.4 * SimCLR + curriculum(t) * Circle
```

Circle Loss uses margin `m = 0.25`, scale `gamma = 128`, and a curriculum schedule:

| Epoch | Circle weight | Phase |
|---:|---:|---|
| 0-5 | `0` | Warmup, Circle off |
| 6-20 | linear ramp to `0.1` | Hard-mining warmup |
| 21-60 | `0.1` | Stable hard-negative refinement |

<details>
<summary><strong>Mathematical formulation</strong></summary>

The baseline optimizes a multi-task objective over L2-normalized image embeddings `v_i` and text embeddings `u_i`:

```text
L_base = alpha_1 L_N-ITC + alpha_2 L_MVS + alpha_3 L_C-ITC + alpha_4 L_SS
```

N-ITC is the sigmoid-based pairwise alignment loss:

```text
L_N-ITC = -(1/N) sum_i sum_j log sigma(z_ij * (gamma * v_i^T u_j - c))
```

where `z_ij` is `+1` for matched pairs and `-1` otherwise.

The auxiliary Cross-modal Circle Loss mines hard positives and hard negatives:

```text
L_circle =
  log(1 + sum_{j in N} exp(gamma * alpha_n^j * (s_n^j - m))
          * sum_{i in P} exp(-gamma * alpha_p^i * (s_p^i - (1 - m))))
```

with adaptive weights:

```text
alpha_p = [1 + m - s_p]_+
alpha_n = [s_n + m]_+
```

The final objective is:

```text
L = L_base + alpha_5(t) * L_circle
```

</details>

### Analysis Figures

| Gradient behavior | Embedding geometry |
|---|---|
| ![Gradient analysis](figures/gradient_3d_optimized_pub.png) | ![Geometry analysis](figures/distribution_final_v5_pub.png) |

## Main Results

### VN3K / 3000VnPersonSearch

| Method | R@1 | R@5 | R@10 | mAP | mINP |
|---|---:|---:|---:|---:|---:|
| TBPS-mSigLIP (Full FT) | 49.70 | 75.93 | 84.75 | 54.96 | 48.66 |
| Ours (LoRA only) | 49.90 | 78.05 | 86.30 | 55.83 | 49.45 |
| Ours (LoRA + Circle fixed) | 50.53 | 77.78 | 86.43 | 55.94 | 49.37 |
| **Ours (LoRA + Curriculum Circle)** | **52.28** | **79.55** | **88.03** | **57.32** | **50.57** |

Best result uses seed `2400`. Mean over 3 seeds: `R@1 = 51.52 +/- 0.68`.

### 10% CUHK-PEDES

| Method | R@1 | R@5 | R@10 | mAP | mINP |
|---|---:|---:|---:|---:|---:|
| TBPS-mSigLIP (Baseline) | 46.73 | 68.65 | 77.55 | 41.75 | 26.56 |
| Ours (LoRA + Circle fixed) | 56.87 | **77.18** | 84.15 | 50.70 | 34.61 |
| **Ours (LoRA + Curriculum Circle)** | **57.10** | 76.98 | **84.34** | **50.90** | **34.85** |

### PRW-TPS-CN

| Method | R@1 | R@5 | R@10 | mAP | mINP |
|---|---:|---:|---:|---:|---:|
| TPAN | 21.63 | 42.54 | 52.99 | - | - |
| TBPS-mSigLIP (Baseline) | 46.78 | 60.28 | 66.82 | 35.41 | 10.61 |
| **Ours (mSigLIP-CLoRA)** | **59.35** | **70.58** | **75.48** | **46.44** | **15.10** |

### Qualitative Retrieval

The baseline often retrieves visually similar distractors. Curriculum Circle Loss improves fine-grained discrimination over details such as shoes, logos, and clothing attributes.

![Qualitative retrieval comparison](figures/flipped_cases_visualization.png)

## Accuracy Extension

> **Status:** experimental / post-paper ablation. This is not the deployed model. The edge pipeline targets the published `52.28` checkpoint because rotation and quantization artifacts are model-specific.

Two extra levers are explored beyond the paper configuration:

- **Part-Token Alignment** aligns image part regions with text tokens for local supervision.
- **Attention + FFN LoRA r32** extends LoRA targets from attention projections to FFN projections (`fc1`, `fc2`).

| Method | T2I R@1 | T2I R@5 | T2I R@10 | I2T R@1 | Notes |
|---|---:|---:|---:|---:|---|
| LoRA + Curriculum Circle (paper) | 52.28 | 79.55 | 88.03 | - | Reported headline, seed 2400 |
| + Attention/FFN LoRA r32 | 52.83 | 79.03 | 87.58 | 52.30 | Larger adapter capacity |
| **+ Part-Token Alignment** | **53.00** | 78.60 | 87.30 | **53.25** | Best R@1, strongest I2T gain |

Run the current ablation recipe:

```bash
bash run_part_align_lora_attn_ffn_r32.sh \
  dataset.batch_size=64 dataset.test_batch_size=128 trainer.accumulate_grad_batches=2
```

## Quick Start

### 1. Install

```bash
git clone https://github.com/pahmlam/Research_on_CircleLoss_for_TBPS-mSigLIP.git
cd Research_on_CircleLoss_for_TBPS-mSigLIP
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

The package targets Python `>=3.11`. The recommended dependency workflow is `uv sync`.

### 2. Prepare Data and Checkpoints

Download the `siglip-base-patch16-256-multilingual` checkpoint and organize datasets under the project root or the configured paths.

Expected datasets:

| Dataset | Language | Role |
|---|---|---|
| VN3K / 3000VnPersonSearch | Vietnamese | Main low-resource benchmark |
| CUHK-PEDES | English | Natural-noise / multilingual robustness benchmark |
| PRW-TPS-CN | Chinese | Cross-lingual generalization benchmark |

```bash
uv run scripts/prepare_checkpoints.py
```

### 3. Train

Paper recipe:

```bash
bash run_cir_loss.sh
```

Post-paper Part-Align ablation:

```bash
bash run_part_align_lora_attn_ffn_r32.sh \
  dataset.batch_size=64 dataset.test_batch_size=128 trainer.accumulate_grad_batches=2
```

Full fine-tuning baseline:

```bash
bash run_full_finetune.sh
```

Hydra baseline entrypoint:

```bash
uv run trainer.py -cn m_siglip img_size_str="'(256,256)'" \
  dataset=vn3k loss.softlabel_ratio=0.0 trainer.max_epochs=60
```

### 4. Evaluate

`test.py` wraps `src/msiglip/evaluate.py` through Fire:

```bash
uv run test.py --ckpt_path artifacts/models/checkpoints/epoch=56-val_score=52.28.ckpt \
  --dataset_name vn3k
```

## Deployment on RB3 Gen2

The deployment branch targets **Qualcomm RB3 Gen2 / QCS6490 / HTP v68** with QNN context binaries. The current vision encoder path is all-INT8 W8A8.

Canonical references:

- [Rotated W8A8 + QAT method](deployment/docs/w8a8_qat_rotated.md)
- [Deploy master journal](deployment/docs/journal/[deploy-master].md)
- [v8 / both-INT8 runbook](deployment/docs/runbook-w8a8-v8-both-int8.md)

### Current Deployment State

| Area | Status | Notes |
|---|---|---|
| LoRA merge and FP32/FP16 export | PASS | `deployment/scripts/lora_fp16/export.py` |
| Mean-preserving rotation | PASS | Output-invariant, residual concentration 252x -> 5.3x |
| ONNX opset 20 export | PASS | Fused `Gelu`, fused `LayerNormalization`, `Pow=0` |
| Vision W8A8 QDQ fidelity | PASS | QAT v6 cosine `0.9491 / 0.9266` mean/min |
| Vision context binary on v68 | PASS | all-INT8 links on HTP v68 |
| Vision board verification | PASS | QAT v4 board fidelity `0.9363 / 0.9068`, `32.70 ms/img`, `22.88 FPS` |
| Vision retrieval | PASS | QAT v6: T2I R@1 `49.30`, I2T R@1 `53.85` |
| Text encoder | Pending | Next workstream for full both-INT8 retrieval |

Key constraint: HTP v68 rejects broad A16 activation paths for LayerNorm and attention matmul, so W8A16 is not deployable on this board even when QDQ fidelity is high. The successful path is opset-20 fused ops + mean-preserving rotation + W8A8 + QAT.

### Deployment Gates

| Gate | Threshold | Meaning |
|---|---:|---|
| LoRA merge | no `lora` / `adapter` / `base_layer` keys | Export must be a dense deployable model |
| Rotation invariance | cosine min `>= 0.9999` | Rotation must preserve FP32 behavior |
| Static ONNX vs PyTorch | cosine mean `>= 0.999` | Export/preprocess control |
| ONNX op sanity | `Pow=0`, fused `Gelu`, fused `LayerNormalization` | Avoid exposed GELU/RMSNorm internals |
| QDQ ONNX vs PyTorch | mean `>= 0.95`, min `>= 0.90` | Candidate worth compile/link |
| QNN board vs PyTorch | mean `>= 0.90` | Runtime on board is faithful enough |
| Full retrieval | T2I R@1 `>= 48.0` | Minimum deploy target vs FP32 baseline |
| Stretch retrieval | T2I R@1 `>= 50.0` | Current optimization target |

### Model Footprint

| Component | Params | FP32 | FP16 | INT8 |
|---|---:|---:|---:|---:|
| vision_model | 92.9M | 372 MB | 186 MB | 93 MB |
| text_model | 277.7M | 1111 MB | 555 MB | 278 MB |
| projection + other | 1.2M | 5 MB | 2 MB | 1 MB |
| **Total** | **371.8M** | **1487 MB** | **744 MB** | **372 MB** |

Text is 75% of parameters because the multilingual token embedding has `250000 x 768` entries. This is why text INT8 is the next major memory lever for 4 GB RB3 deployment.

<details>
<summary><strong>Deployment pipeline and QAT commands</strong></summary>

```text
epoch=56-val_score=52.28.ckpt
  -> [1] lora_fp16/export.py
         exported_model/{model_fp32.pt, model_fp16.pt, config.yaml}
  -> [2] qnn/rotate_vision_encoder.py
         exported_model_rotated/{model_fp32.pt, config.yaml}
  -> [3] qnn/train_vision_quant_robust.py
         exported_model_rotated_qat_v6/{model_fp32.pt, config.yaml}
  -> [4] qnn/export_rotated_vision_onnx.py --opset 20
         exported_model_rotated_qat_v6/vision_onnx/{vision_encoder.onnx,.data}
  -> [5] qnn/submit_qaihub_quantize_compile.py
         W8A8 QDQ / context binary with quantized I/O
  -> [6] qnn-net-run on RB3 HTP
         compare_qnn_with_pytorch -> retrieval R@1
```

One-time prep:

```bash
python deployment/scripts/lora_fp16/export.py \
  --ckpt artifacts/models/checkpoints/epoch=56-val_score=52.28.ckpt \
  --output-dir artifacts/deployment/exports/exported_model

python deployment/scripts/qnn/rotate_vision_encoder.py \
  --model-dir artifacts/deployment/exports/exported_model \
  --output-dir artifacts/deployment/exports/exported_model_rotated \
  --input-dir artifacts/deployment/qnn_inputs/vn3k_test_10 --seed 2400 --skip-r2

python deployment/scripts/qnn/prepare_vn3k_vision_inputs.py \
  --dataset-root VN3K --split train --selection random --seed 2400 \
  --num-samples 4302 --output-dir artifacts/deployment/qnn_inputs/vn3k_train_all_4302 \
  --path-mode relative
```

QAT v6:

```bash
python deployment/scripts/qnn/train_vision_quant_robust.py \
  --model-dir artifacts/deployment/exports/exported_model_rotated \
  --train-input-dir artifacts/deployment/qnn_inputs/vn3k_train_all_4302 \
  --val-input-dir artifacts/deployment/qnn_inputs/vn3k_test_100 \
  --output-dir artifacts/deployment/exports/exported_model_rotated_qat_v6 \
  --device cuda --batch-size 16 --epochs 15 --lr 1e-5 \
  --start-layer 0 --end-layer 11 --num-workers 4 \
  --fake-quant-observer ema \
  --quant-head --quant-linears --quant-attention

python deployment/scripts/qnn/export_rotated_vision_onnx.py \
  --model-dir artifacts/deployment/exports/exported_model_rotated_qat_v6 \
  --opset 20

python deployment/scripts/qnn/compare_onnx_with_pytorch.py \
  --onnx-model artifacts/deployment/exports/exported_model_rotated_qat_v6/vision_onnx \
  --model-dir artifacts/deployment/exports/exported_model_rotated_qat_v6 \
  --input-dir artifacts/deployment/qnn_inputs/vn3k_test_10 \
  --precision fp32 \
  --json artifacts/deployment/exports/exported_model_rotated_qat_v6/static_vs_pytorch_summary.json \
  --csv artifacts/deployment/exports/exported_model_rotated_qat_v6/static_vs_pytorch.csv
```

AI Hub QDQ diagnostic and local retrieval gate:

```bash
python deployment/scripts/qnn/submit_qaihub_quantize_compile.py \
  --model artifacts/deployment/exports/exported_model_rotated_qat_v6/vision_onnx \
  --calibration-data d7jzjy1m2 --weights-dtype int8 --activations-dtype int8 \
  --quantize-only --wait \
  --download-quantized artifacts/deployment/runtime/rotated_w8a8_qat_v6/job_qdq_onnx

python deployment/scripts/qnn/compare_onnx_with_pytorch.py \
  --onnx-model artifacts/deployment/runtime/rotated_w8a8_qat_v6/job_qdq_onnx \
  --model-dir artifacts/deployment/exports/exported_model \
  --input-dir artifacts/deployment/qnn_inputs/vn3k_test_10 \
  --precision fp32 \
  --json artifacts/deployment/runtime/rotated_w8a8_qat_v6/qdq_vs_pytorch_summary.json \
  --csv artifacts/deployment/runtime/rotated_w8a8_qat_v6/qdq_vs_pytorch.csv

python deployment/scripts/qnn/eval_retrieval_quantized_vision.py \
  --qdq-onnx artifacts/deployment/runtime/rotated_w8a8_qat_v6/job_qdq_onnx \
  --model-dir artifacts/deployment/exports/exported_model \
  --json artifacts/deployment/runtime/rotated_w8a8_qat_v6/retrieval_r1.json
```

</details>

<details>
<summary><strong>RB3/QNN quick commands</strong></summary>

Analyze a checkpoint:

```bash
python deployment/scripts/analyze_checkpoint.py \
  --ckpt artifacts/models/checkpoints/epoch=56-val_score=52.28.ckpt
```

Prepare smoke/calibration inputs:

```bash
python deployment/scripts/qnn/prepare_vn3k_vision_inputs.py \
  --dataset-root VN3K \
  --split test \
  --selection first \
  --num-samples 10 \
  --output-dir artifacts/deployment/qnn_inputs/vn3k_test_10 \
  --path-mode relative

python deployment/scripts/qnn/prepare_vn3k_vision_inputs.py \
  --dataset-root VN3K \
  --split train \
  --selection random \
  --seed 2400 \
  --num-samples 2000 \
  --output-dir artifacts/deployment/qnn_inputs/vn3k_train_calib_2000 \
  --path-mode relative
```

Upload calibration data:

```bash
python deployment/scripts/qnn/upload_qaihub_calibration_dataset.py \
  --input-dir artifacts/deployment/qnn_inputs/vn3k_train_calib_2000 \
  --name msiglip-vision-vn3k-train-calib-2000
```

Known vision calibration dataset:

```text
d7jzjy1m2 / msiglip-vision-vn3k-train-calib-2000
```

Compile/link a full context binary:

```bash
python deployment/scripts/qnn/submit_qaihub_quantize_compile.py \
  --model artifacts/deployment/exports/exported_model_rotated_qat_v6/vision_onnx \
  --calibration-data d7jzjy1m2 \
  --weights-dtype int8 --activations-dtype int8 \
  --wait \
  --download artifacts/deployment/runtime/rotated_w8a8_qat_v6/vision_encoder.bin
```

Run on RB3:

```bash
qnn-net-run \
  --backend "$QNN_LIB/libQnnHtp.so" \
  --retrieve_context artifacts/deployment/runtime/rotated_w8a8_qat_v6/vision_encoder.bin \
  --config_file deployment/config/qnn/htp_config_245.json \
  --input_list artifacts/deployment/qnn_inputs/vn3k_test_10/input_list.txt \
  --output_dir artifacts/deployment/qnn_runs/rotated_w8a8_qat_v6 \
  --profiling_level basic \
  --perf_profile high_performance
```

Compare board outputs against PyTorch:

```bash
python deployment/scripts/qnn/compare_qnn_with_pytorch.py \
  --qnn-output-dir artifacts/deployment/qnn_runs/rotated_w8a8_qat_v6 \
  --model-dir artifacts/deployment/exports/exported_model \
  --input-dir artifacts/deployment/qnn_inputs/vn3k_test_10 \
  --precision fp32 \
  --json artifacts/deployment/qnn_runs/rotated_w8a8_qat_v6/qnn_vs_pytorch_summary.json \
  --csv artifacts/deployment/qnn_runs/rotated_w8a8_qat_v6/qnn_vs_pytorch.csv
```

Audit native QNN/QAIRT tools:

```bash
python deployment/scripts/qnn/audit_qnn_native_env.py \
  --json artifacts/deployment/runtime/qnn_native/env_audit.json
```

</details>

<details>
<summary><strong>QAT iteration history</strong></summary>

| Candidate | Result | Decision |
|---|---:|---|
| Rotation-only W8A8 | QDQ `0.8975 / 0.8747`, T2I R@1 `45.42`, I2T R@1 `49.40` | Links/runs, retrieval gate FAIL |
| QAT v3 | QDQ `0.9353 / 0.919`, T2I R@1 `48.20`, I2T R@1 `52.30` | First retrieval gate PASS |
| QAT v4 | QDQ `0.9364 / 0.9091`, T2I R@1 `48.50`, I2T R@1 `52.95`; board `0.9363 / 0.9068` | Current board-verified deploy binary |
| QAT v5 | QDQ `0.9437 / 0.9311`, T2I R@1 `49.25`, I2T R@1 `53.40` | Strong accuracy candidate |
| **QAT v6** | QDQ `0.9491 / 0.9266`, T2I R@1 `49.30`, I2T R@1 `53.85` | Current best QDQ/retrieval candidate |
| QAT v7 | QDQ `0.9485 / 0.9083`, T2I R@1 `48.38`, I2T R@1 `53.05` | Regressed vs v6 |
| QAT v8 planned | learned rotation + recipe v6 | Next ablation toward stretch target |

</details>

## Repository Map

```text
src/msiglip/                       # Training, data, model, solver, utilities
  lightning_models.py              # LitTBPS Lightning module
  lightning_data.py                # TBPSDataModule
  model/                           # mSigLIP, LoRA, TBPS, losses, evidence bank
  data/                            # VN3K, CUHK-PEDES, PRW-TPS-CN datasets
  solver/                          # Optimizer and scheduler

configs/                           # Hydra configs for model, loss, datasets, LoRA
trainer.py                         # Hydra training entrypoint wrapper
test.py                            # Fire evaluation wrapper
notebooks/workspace.ipynb          # Local research/loss playground
run_*.sh                           # Reproducible training recipes

deployment/                        # RB3/QNN deployment and compression
  scripts/                         # Export, ONNX, QNN, QAT, diagnostics
  config/qnn/                      # HTP/QNN runtime config JSON files
  docs/                            # Deployment docs and runbooks
  hardware_profiling/              # RB3 profiling helpers

artifacts/                         # Generated outputs (ignored)
docs/                              # Architecture, experiment summaries, journals
knowledge/                         # Research notes and paper drafts
reports/                           # Design notes and architecture decisions
changelog/                         # Training/deployment changelogs
figures/                           # README and paper figures
```

## Documentation Map

| Document | Purpose |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Full training/model architecture reference |
| [docs/EXPERIMENT_SUMMARY.md](docs/EXPERIMENT_SUMMARY.md) | Canonical older experiment summary |
| [docs/noise-robust-multilingual-framework.md](docs/noise-robust-multilingual-framework.md) | MNEB-HN research and implementation design |
| [docs/journal/](docs/journal/) | Dated training/model-optimization journals |
| [deployment/docs/w8a8_qat_rotated.md](deployment/docs/w8a8_qat_rotated.md) | Canonical W8A8 rotation/QAT deployment method |
| [deployment/docs/journal/[deploy-master].md](deployment/docs/journal/[deploy-master].md) | Consolidated deployment/model-compression journal |
| [deployment/docs/runbook-w8a8-v8-both-int8.md](deployment/docs/runbook-w8a8-v8-both-int8.md) | Forward runbook for learned rotation, text, and both-INT8 |

## Contact

For questions, please open an issue or contact the authors.
