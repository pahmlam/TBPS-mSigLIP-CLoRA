<div align="center">

# mSigLIP for Multilingual Text-Based Person Search

**Hard-negative-aware training for Vietnamese, English, and Chinese TBPS, with an INT8 deployment path for Qualcomm RB3 Gen2.**

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="PyTorch Lightning" src="https://img.shields.io/badge/Lightning-2.x-792EE5?style=for-the-badge&logo=lightning&logoColor=white">
  <img alt="Hydra" src="https://img.shields.io/badge/Hydra-config-2563EB?style=for-the-badge">
  <img alt="LoRA" src="https://img.shields.io/badge/PEFT-LoRA-15803D?style=for-the-badge">
  <img alt="QNN" src="https://img.shields.io/badge/RB3%20Gen2-QNN%20W8A8-111827?style=for-the-badge">
</p>

<p>
  <a href="#results">Results</a> •
  <a href="#method">Method</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#deployment">Deployment</a> •
  <a href="#project-map">Project Map</a>
</p>

</div>

---

## Snapshot

<table>
  <tr>
    <td width="33%" align="center">
      <strong>VN3K / Vietnamese</strong><br>
      <span style="font-size:24px"><strong>52.28</strong></span><br>
      T2I Rank@1, paper headline
    </td>
    <td width="33%" align="center">
      <strong>PRW-TPS-CN / Chinese</strong><br>
      <span style="font-size:24px"><strong>59.35</strong></span><br>
      T2I Rank@1, multilingual generalization
    </td>
    <td width="33%" align="center">
      <strong>RB3 Gen2 / HTP v68</strong><br>
      <span style="font-size:24px"><strong>49.30</strong></span><br>
      Vision-only INT8 QAT v6 T2I Rank@1
    </td>
  </tr>
</table>

| Track | Status | Current next step |
|---|---|---|
| **Training** | LoRA + Curriculum Circle Loss is the main reported method. It trains **5.9M parameters, about 1.57% of the 376M-parameter base model**. | Keep the paper recipe as the clean deployment baseline. |
| **Multilingual evaluation** | VN3K, 10% CUHK-PEDES, and PRW-TPS-CN results are reported below. | Extend full multilingual ablations only when needed. |
| **Edge deployment** | Vision encoder W8A8 path works on RB3 Gen2 HTP v68. QAT v6 is the best QDQ/retrieval candidate; QAT v4 is board-verified. | Quantize text encoder, then validate both-INT8 retrieval on board. |

## What This Repository Contains

This repository implements the training and deployment stack for paper **A Hard Negative-Aware Optimization for Multilingual Text-Based Person Search** accepted at ICIP 2026 Tampere Finland.

The core idea is deliberately simple:

```text
mSigLIP multilingual dual encoder
  + LoRA adapters
  + N-ITC/MVS, C-ITC, SimCLR
  + curriculum-weighted Cross-modal Circle Loss
  -> multilingual person retrieval embeddings
```

The method keeps the large multilingual mSigLIP backbone mostly frozen and trains a compact LoRA subspace. Circle Loss is introduced through a curriculum so the model first learns stable global image-text alignment, then focuses on hard negatives.

## Results

### Main Results

| Dataset | Language | Main result | Notes |
|---|---|---:|---|
| VN3K / 3000VnPersonSearch | Vietnamese | **52.28 T2I R@1** | Paper headline, seed `2400` |
| 10% CUHK-PEDES | English | **57.10 T2I R@1** | Low-data English setting |
| PRW-TPS-CN | Chinese | **59.35 T2I R@1** | Multilingual generalization |

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

![Qualitative retrieval comparison](figures/flipped_cases_visualization.png)

The model improves retrieval in cases where visually similar distractors differ only by fine-grained clothing, shoe, or logo attributes.

## Method

![Framework Architecture](figures/framework.png)

### Loss Stack

The default training objective is:

$$
\mathcal{L}
= \mathcal{L}_{N\text{-}ITC/MVS}
+ 0.1\,\mathcal{L}_{C\text{-}ITC}
+ 0.4\,\mathcal{L}_{SS}
+ \alpha_5(t)\,\mathcal{L}_{circle}
$$

Circle Loss is the hard-negative component:

| Setting | Value |
|---|---:|
| Margin `m` | `0.25` |
| Scale `gamma` | `128` |
| Epochs 0-5 | Circle off |
| Epochs 6-20 | Linear ramp to `0.1` |
| Epochs 21-60 | Stable at `0.1` |

<details>
<summary><strong>Mathematical details</strong></summary>

The baseline optimizes a multi-task objective over $L_2$-normalized image embeddings $\mathbf{v}_i$ and text embeddings $\mathbf{u}_i$:

$$
\mathcal{L}_{base}
= \alpha_1 \mathcal{L}_{N\text{-}ITC}
+ \alpha_2 \mathcal{L}_{MVS}
+ \alpha_3 \mathcal{L}_{C\text{-}ITC}
+ \alpha_4 \mathcal{L}_{SS}
$$

N-ITC is the sigmoid-based pairwise alignment loss:

$$
\mathcal{L}_{N\text{-}ITC}
= -\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{N}
\log\sigma\left(z_{ij}\left(\gamma\,\mathbf{v}_i^\top\mathbf{u}_j - c\right)\right)
$$

where $z_{ij} \in \{+1, -1\}$ indicates matched and unmatched image-text pairs.

The auxiliary Cross-modal Circle Loss mines hard positives and hard negatives:

$$
\mathcal{L}_{circle}
= \log\left[
1
+ \sum_{j \in \mathcal{N}} e^{\gamma\,\alpha_n^j(s_n^j - m)}
\cdot
\sum_{i \in \mathcal{P}} e^{-\gamma\,\alpha_p^i(s_p^i - (1-m))}
\right]
$$

with adaptive weights:

$$
\alpha_p^i = [1 + m - s_p^i]_+,
\qquad
\alpha_n^j = [s_n^j + m]_+
$$

The final objective is:

$$
\mathcal{L}
= \mathcal{L}_{base}
+ \alpha_5(t)\,\mathcal{L}_{circle}
$$

</details>

### Analysis Figures

| Gradient behavior | Embedding geometry |
|---|---|
| ![Gradient analysis](figures/gradient_3d_optimized_pub.png) | ![Geometry analysis](figures/distribution_final_v5_pub.png) |

## Quick Start

### Install

```bash
git clone https://github.com/pahmlam/Research_on_CircleLoss_for_TBPS-mSigLIP.git
cd Research_on_CircleLoss_for_TBPS-mSigLIP
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

The project targets Python `>=3.11`. `uv sync` is the recommended dependency workflow.

### Prepare Data and Checkpoints

Expected datasets:

| Dataset | Language | Role |
|---|---|---|
| VN3K / 3000VnPersonSearch | Vietnamese | Main low-resource benchmark |
| CUHK-PEDES | English | English retrieval benchmark |
| PRW-TPS-CN | Chinese | Multilingual generalization benchmark |

Prepare local mSigLIP checkpoint assets:

```bash
uv run scripts/prepare_checkpoints.py
```

### Train

Paper recipe:

```bash
bash run_cir_loss.sh
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

### Evaluate

`test.py` wraps `src/msiglip/evaluate.py` through Fire:

```bash
uv run test.py --ckpt_path artifacts/models/checkpoints/epoch=56-val_score=52.28.ckpt \
  --dataset_name vn3k
```

## Deployment

The deployment branch targets **Qualcomm RB3 Gen2 / QCS6490 / HTP v68** with QNN context binaries. The current vision encoder path is all-INT8 W8A8.

| Deployment item | Current state |
|---|---|
| Source checkpoint | `artifacts/models/checkpoints/epoch=56-val_score=52.28.ckpt` |
| Best QDQ/retrieval candidate | QAT v6, `49.30` T2I R@1, `53.85` I2T R@1 |
| Board-verified binary | QAT v4 W8A8 context binary |
| Board runtime | `32.70 ms/img`, `22.88 FPS` |
| Context size | about `90 MB` for vision encoder |
| Text encoder | Pending |

Canonical references:

- [Rotated W8A8 + QAT method](deployment/docs/w8a8_qat_rotated.md)
- [Deploy master journal](deployment/docs/journal/[deploy-master].md)
- [v8 / both-INT8 runbook](deployment/docs/runbook-w8a8-v8-both-int8.md)

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
<summary><strong>Vision W8A8 pipeline and QAT commands</strong></summary>

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

## Project Map

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

## Documentation

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
