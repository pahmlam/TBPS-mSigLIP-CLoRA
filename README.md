
---

# A Hard Negative-Aware Optimization for Multilingual Text-Based Person Search

This repository contains the official implementation for the paper: **"A Hard Negative-Aware Optimization for Multilingual Text-Based Person Search"**, along with ongoing work on parameter-efficient accuracy extensions and edge deployment on the **Qualcomm RB3 Gen2**.

##  Abstract

Multilingual Text-Based Person Search (TBPS) remains challenging in low-resource settings due to ambiguous cross-modal alignment. Although recent methods such as TBPS-mSigLIP employ noise-robust contrastive learning, they suffer from **limited gradient discrimination** between easy and hard negatives.

To address this, we propose an efficient optimization framework that integrates **Cross-modal Circle Loss** with **Low-Rank Adaptation (LoRA)**. Circle Loss enhances fine-grained discrimination via adaptive pair-wise re-weighting, while LoRA stabilizes training by constraining optimization to a low-rank subspace. We further introduce a **Curriculum Hard-Mining Schedule** to balance alignment stability and discrimination. Experiments across three typologically diverse languages — Vietnamese, English, and Chinese — demonstrate consistent improvements, establishing a new state-of-the-art **Rank@1 accuracy of 52.28%** on VnPersonSearch and **59.35%** on PRW-TPS-CN, with only **1.57% trainable parameters**. Additionally, we explore parameter-efficient accuracy extensions (local part-token alignment and attention+FFN LoRA) and deploy the optimized model on edge hardware (Qualcomm RB3 Gen2) for real-time INT8 inference.

---

## Current Status Snapshot

| Track | Current state | Next step |
|---|---|---|
| **Main training result (paper)** | LoRA + Curriculum Circle Loss reaches **52.28% R@1** on VN3K and **59.35% R@1** on PRW-TPS-CN | Reported headline; deployment target |
| **Accuracy extension (ablation)** | Part-Token Alignment + attention/FFN LoRA r32 reaches **53.00% R@1** on VN3K (single seed) | Complete CUHK-PEDES and PRW-TPS-CN runs, then report as ablation |
| **Deployment** | Vision encoder runs **INT8 (W8A8) on HTP v68** via GELU fusion + mean-preserving rotation + QAT; vision-only **T2I R@1 48.20** (gate ≥ 48 passed) | Quantize text encoder, then end-to-end board retrieval |

---

##  Framework Architecture

We propose a unified framework constructed upon the **mSigLIP** foundation model. To bridge the gap in hard-negative mining, we incorporate an **Auxiliary Cross-Modal Circle Loss** for geometric refinement and utilize **LoRA** on the Transformer backbone (Query, Key, Value, Output projections) to ensure optimization stability and memory efficiency (allowing **3x** larger batch sizes). Only **5.9M / 376M parameters (1.57%)** are trainable.

![Framework Architecture](figures/framework.png)

*Figure 1: The overall architecture of the proposed Multilingual TBPS framework. It features a dual-pathway optimization: (1) The baseline noise-robust objectives (N-ITC, etc.) for global alignment, and (2) An auxiliary Circle Loss branch for explicit hard-negative mining, stabilized by LoRA.*

---

##  Key Contributions & Analysis

### 1. Theoretical Gradient Analysis

Why does mSigLIP fail on hard negatives? We analyze the gradient dynamics of the standard Sigmoid loss (N-ITC) versus our Circle Loss.

![Gradients](figures/gradient_3d_optimized_pub.png)

*Figure 2: Theoretical visualization of gradient magnitude. (Left) **N-ITC (Cyan)** exhibits vanishing gradients for semi-hard negatives (), leading to insensitivity. **Circle Loss (Red)** imposes a sharp penalty after the margin, effectively mining hard negatives. (Right) Circle Loss maintains strong signals for positive pairs even as they approach similarity 1.0, preventing premature convergence.*

### 2. Geometric Refinement

Our method transforms the embedding space geometry. By applying a **Curriculum Hard-Mining Schedule** (linearly warming up the Circle Loss weight), we prevent the disruption of early global alignment while enforcing strict spherical constraints in later stages.

![Geometric](figures/distribution_final_v5_pub.png)

*Figure 3: Geometric Analysis of Similarity Distribution ( vs. ). (Left) The Baseline distribution converges linearly to the decision boundary (), causing overlap. (Right) **Ours (LoRA + Circle)** lifts the distribution towards the theoretical margin (), creating a clear spherical boundary that separates correct matches from hard negatives.*

---

##  Mathematical Formulation

### Baseline Objective (TBPS-mSigLIP)

The baseline optimizes a multi-task objective over $L_2$-normalized image embeddings $\mathbf{v}_i$ and text embeddings $\mathbf{u}_i$:

$$\mathcal{L}_{\text{base}} = \alpha_1 \mathcal{L}_{N\text{-}ITC} + \alpha_2 \mathcal{L}_{MVS} + \alpha_3 \mathcal{L}_{C\text{-}ITC} + \alpha_4 \mathcal{L}_{SS}$$

**N-ITC** (Noise-robust Image-Text Contrastive) — sigmoid-based pairwise alignment:

$$\mathcal{L}_{N\text{-}ITC} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{N} \log\sigma\!\left(z_{ij}\left(\gamma\,\mathbf{v}_i^\top\mathbf{u}_j - c\right)\right)$$

where $z_{ij} \in \{+1, -1\}$ indicates matched pairs, and $\gamma, c$ are learned scale and bias.

### Auxiliary Cross-Modal Circle Loss

We introduce Circle Loss to explicitly mine hard negatives via adaptive pair-wise re-weighting:

$$\mathcal{L}_{\text{circle}} = \log\left[1 + \sum_{j \in \mathcal{N}} e^{\gamma\,\alpha_n^j(s_n^j - m)} \cdot \sum_{i \in \mathcal{P}} e^{-\gamma\,\alpha_p^i(s_p^i - (1-m))}\right]$$

where $\mathcal{P}$, $\mathcal{N}$ are positive/negative pair sets, $s$ is cosine similarity, $\gamma=128$ is the scale factor, and $m=0.25$ is the margin. The adaptive weights:

$$\alpha_p^i = [1 + m - s_p^i]_+, \qquad \alpha_n^j = [s_n^j + m]_+$$

dynamically amplify gradients for hard samples (poorly separated pairs) while suppressing well-separated ones.

### Total Objective with Curriculum Schedule

$$\mathcal{L} = \mathcal{L}_{\text{base}} + \alpha_5(t) \cdot \mathcal{L}_{\text{circle}}$$

The curriculum schedule for $\alpha_5(t)$ prevents early disruption of global alignment:

| Epoch $t$ | $\alpha_5(t)$ | Phase |
|---|---|---|
| $t \leq 5$ | $0$ | Warmup (Circle off) |
| $5 < t \leq 20$ | $0.1 \times \frac{t - 5}{15}$ | Linear ramp |
| $t > 20$ | $0.1$ | Stable |

---

## Accuracy Extension: Part-Token Alignment + Attention/FFN LoRA

> Status: **experimental / post-paper ablation**. This extension explores additional Rank@1 gains on top of the reported Circle Loss results. It is reported as an ablation and is **not** the deployed model — the edge pipeline targets the published 52.28% configuration because rotation/quantization artifacts are model-specific.

### Motivation

Two orthogonal levers are explored beyond the published attention-only LoRA + Curriculum Circle setup, both keeping Circle Loss and the curriculum schedule unchanged:

1. **Part-Token Alignment** — a local supervision branch that aligns image part regions with text tokens, sharpening fine-grained discrimination (especially image-to-text).
2. **Attention + FFN LoRA (rank 32)** — extending LoRA from the attention projections to the FFN (`fc1`, `fc2`) projections, increasing adapter capacity while staying parameter-efficient.

### Results (VN3K, single seed)

| Method | t2i R@1 | t2i R@5 | t2i R@10 | i2t R@1 | Notes |
|---|---:|---:|---:|---:|---|
| LoRA + Curriculum Circle (paper) | 52.28 | 79.55 | 88.03 | — | Reported headline (seed 2400; mean 51.52 ± 0.68) |
| + Attention/FFN LoRA r32 | 52.83 | 79.03 | 87.58 | 52.30 | Larger adapter capacity |
| **+ Part-Token Alignment** | **53.00** | 78.60 | 87.30 | **53.25** | Best R@1; clearest gain on i2t |

The Part-Align result is single-seed and trades a small drop in t2i R@5/R@10 for a clear i2t gain; it is reported as "best R@1," not best across all metrics. CUHK-PEDES (English) and PRW-TPS-CN (Chinese) runs are in progress to complete the multilingual ablation.

### How to Run

```bash
bash run_part_align_lora_attn_ffn_r32.sh \
  dataset.batch_size=64 dataset.test_batch_size=128 trainer.accumulate_grad_batches=2
```

---

##  Experimental Results

We evaluate our method on **3000VnPersonSearch** (Low-resource, Vietnamese), **CUHK-PEDES** (High-resource, English), and **PRW-TPS-CN** (Chinese).

### Quantitative Performance (VN3K)

Our method with Curriculum Learning achieves State-of-the-Art performance, significantly outperforming the full fine-tuning baseline despite using only 1.57% trainable parameters.

| Method                       | R@1   | R@5   | R@10  | mAP   | mINP  |
| ---------------------------- | ----- | ------| ----- | ----- | ----- |
| TBPS-mSigLIP (Full FT)       | 49.70 | 75.93 | 84.75 | 54.96 | 48.66 |
| Ours (LoRA Only)             | 49.90 | 78.05 | 86.30 | 55.83 | 49.45 |
| Ours (LoRA + Circle Fixed)   | 50.53 | 77.78 | 86.43 | 55.94 | 49.37 |
| **Ours (LoRA + Curriculum)** | **52.28** | **79.55** | **88.03** | **57.32** | **50.57** |

*Best result with seed 2400. Mean over 3 seeds: R@1 = 51.52 +/- 0.68%.*

### Quantitative Performance (10% CUHK-PEDES, English)

| Method                       | R@1   | R@5   | R@10  | mAP   | mINP  |
| ---------------------------- | ----- | ------| ----- | ----- | ----- |
| TBPS-mSigLIP (Baseline)      | 46.73 | 68.65 | 77.55 | 41.75 | 26.56 |
| Ours (LoRA + Circle Fixed)   | 56.87 | **77.18** | 84.15 | 50.70 | 34.61 |
| **Ours (LoRA + Curriculum)** | **57.10** | 76.98 | **84.34** | **50.90** | **34.85** |

### Quantitative Performance (PRW-TPS-CN, Chinese)

| Method                       | R@1   | R@5   | R@10  | mAP   | mINP  |
| ---------------------------- | ----- | ------| ----- | ----- | ----- |
| TPAN                         | 21.63 | 42.54 | 52.99 | -     | -     |
| TBPS-mSigLIP (Baseline)      | 46.78 | 60.28 | 66.82 | 35.41 | 10.61 |
| **Ours (mSigLIP-CLoRA)**    | **59.35** | **70.58** | **75.48** | **46.44** | **15.10** |

### Qualitative Visualization

The baseline often retrieves visually similar distractors (hard negatives). Our method successfully discriminates fine-grained attributes (e.g., shoe color, logo details).

![Visualize](figures/flipped_cases_visualization.png)

*Figure 4: Qualitative comparison. Green boxes indicate correct matches; Red boxes are incorrect. Note how our method ranks the Ground Truth at #1 even in challenging cases where the baseline fails.*

---

##  Repository Structure

```
├── src/msiglip/                       # Python package: training, data, model, solver, utils
│   ├── train.py                       # Training entry point implementation
│   ├── evaluate.py                    # Evaluation entry point implementation
│   ├── lightning_models.py            # LitTBPS (PyTorch Lightning module)
│   ├── lightning_data.py              # TBPSDataModule (data loading, augmentation)
│   ├── model/                         # TBPS + mSigLIP + losses
│   ├── data/                          # Dataset classes & augmentation
│   ├── solver/                        # Optimizer and LR scheduler
│   └── utils/                         # Metrics, visualization, tokenizer utilities
├── trainer.py                         # Backward-compatible wrapper
├── test.py                            # Backward-compatible wrapper
├── notebooks/workspace.ipynb          # Notebook lab for embedding/loss validation
├── run_cir_loss.sh                    # LoRA + Curriculum Circle Loss training
├── run_part_align_lora_attn_ffn_r32.sh # Part-Token Alignment + Attn/FFN LoRA (ablation)
├── run_full_finetune.sh               # Full fine-tuning baseline
├── configs/                           # Hydra configuration
│   ├── cir_msiglip.yaml               # Main config
│   ├── paths/default.yaml             # Centralized data/artifact paths
│   ├── loss/cir_msiglip.yaml          # Loss flags, Circle, Part-Align config
│   └── ...                            # backbone, trainer, optimizer, dataset, tokenizer, logger, aug
├── artifacts/                         # Ignored generated outputs
│   ├── training/                      # Hydra runs, multirun outputs
│   ├── models/                        # Local checkpoints/pretrained model files
│   └── deployment/                    # Exports, QNN inputs/runs/logs/runtime state
├── scripts/                           # Helper scripts for checkpoints/data preparation
├── experiments/                       # Experiment logs & ablation notes
├── knowledge/                         # Research notes & paper drafts
├── reports/                           # Design notes and implementation plans
├── changelog/                         # Training/deployment changelogs
├── figures/                           # Paper figures
├── docs/                              # Project documentation
│   ├── ARCHITECTURE.md                # Full architecture with diagrams
│   ├── EXPERIMENT_SUMMARY.md          # Canonical experiment record
│   ├── knowledge.md                   # Vietnamese durable concept/definition base
│   └── journal/                       # Dated training/model-optimization logs
│
├── deployment/                        # Edge deployment & compression
│   ├── scripts/
│   │   ├── analyze_checkpoint.py      # Checkpoint size/RAM compatibility
│   │   ├── inference_test.py          # ONNX/PyTorch inference test
│   │   ├── lora_fp16/export.py        # Merge LoRA, export FP32/FP16 state dicts
│   │   └── onnx/
│   │       ├── export.py              # Export vision/text ONNX with external weights
│   │       └── to_fp16.py             # Local ONNX FP16 conversion
│   ├── hardware_profiling/            # RB3 hardware tests with proxy models
│   ├── docs/
│   │   ├── deployment-plan.md         # Current deployment status and next steps
│   │   ├── aihub-experiments.md       # Legacy redirect to deployment journal
│   │   ├── journal/                   # Master deploy journal + archived logs
│   │   ├── system.md                  # RB3 hardware specs
│   │   └── benchmark-rp.md            # Proxy benchmark results
│   └── config/qnn/                    # QNN/HTP runtime config JSON files
│
└── ref/                               # Reference implementations (RDE, etc.)
```

---

##  Installation

### 1. Clone and Setup

```bash
git clone https://github.com/pahmlam/Research_on_CircleLoss_for_TBPS-mSigLIP.git
cd Research_on_CircleLoss_for_TBPS-mSigLIP
./setup.sh

```

### 2. Environment

We recommend using `uv` for fast dependency management.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

```

### 3. Prepare Data & Checkpoints

Download the `siglip-base-patch16-256-multilingual` checkpoints and organize your datasets (VN3K, CUHK-PEDES) in the root directory.

```bash
uv run scripts/prepare_checkpoints.py

```

---

##  Training

Use the provided scripts for normal experiments. The scripts keep the Hydra overrides in one place and avoid long ad-hoc command lines.

### Train with Curriculum Hard-Mining (Recommended)

This runs the proposed method: LoRA + mSigLIP + Auxiliary Circle Loss with a warm-up schedule.

```bash
./run_cir_loss.sh
```

### Train Accuracy Extension (Part-Token Alignment + Attn/FFN LoRA)

This runs the post-paper accuracy ablation: local part-token alignment combined with LoRA extended to the FFN projections at rank 32. It reaches 53.00% R@1 on VN3K (single seed) and is reported as an ablation, not as the deployed model.

```bash
bash run_part_align_lora_attn_ffn_r32.sh \
  dataset.batch_size=64 dataset.test_batch_size=128 trainer.accumulate_grad_batches=2
```

### Full Fine-Tuning Baseline

```bash
./run_full_finetune.sh
```

### Train Baseline (mSigLIP)

```bash
uv run trainer.py -cn m_siglip img_size_str="'(256,256)'" dataset=vn3k loss.softlabel_ratio=0.0 trainer.max_epochs=60

```

---

## Deployment Status

Edge deployment targets the **Qualcomm RB3 Gen2 / QCS6490** with local image/text embedding inference.

> **Canonical deployment document:** [`deployment/docs/w8a8_qat_rotated.md`](deployment/docs/w8a8_qat_rotated.md) — the full method for the best result (mathematical analysis, flow diagram, per-stage commands, and acceptance gates). The summary below is an overview; that document is the source of truth.

Current progress:

| Stage | Status | Notes |
|---|---|---|
| Checkpoint analysis | Done | `deployment/scripts/analyze_checkpoint.py` |
| LoRA merge + FP16/FP32 export | Done | `deployment/scripts/lora_fp16/export.py` |
| ONNX export (opset-20, fused GELU) | Done | Removes decomposed cubic GELU outliers (`Pow=0`) |
| Mean-preserving rotation | Done | QuaRot/SliceGPT-style; residual concentration 252x → 5.3x, output-invariant |
| Vision W8A8 quantize + HTP link | Done | All-INT8 links on HTP **v68** → `vision_encoder.bin`, 89.7 MB |
| Quantization-aware finetune (QAT v5) | Done | Current best vision-only **T2I R@1 is 49.25** with `--quant-linears`; v4 is board-verified at 48.50 |
| On-device RB3 run (vision) | Done | QAT v4 `qnn-net-run` on HTP v68: cosine `0.9363`, **32.7 ms/img, 22.88 FPS** |
| Text encoder quantization | Pending | Text = 75% of params (250k-vocab embedding); needed for the 4GB RAM budget |
| End-to-end board retrieval | Pending | Both encoders INT8 on board |

Key deployment finding: HTP **v68** blocks 16-bit activations broadly (attention act×act and LayerNorm require v73+), so the only deployable path is all-INT8 (W8A8). ViT activation outliers ("massive activations") make naive per-tensor W8A8 collapse retrieval; the working recipe is **opset-20 GELU fusion + mean-preserving rotation (to spread residual outliers) + W8A8 + quantization-aware finetune**, which lifts vision-only T2I R@1 from 45.42 to **49.25** in QAT v5. See [`deployment/docs/w8a8_qat_rotated.md`](deployment/docs/w8a8_qat_rotated.md) for the full method and [`deployment/docs/journal/[deploy-master].md`](deployment/docs/journal/[deploy-master].md) for the consolidated AI Hub/QNN/RB3 journal.

The reproducible vision pipeline is: **(1)** merge LoRA → **(2)** mean-preserving rotation → **(3)** quantization-aware finetune → **(4)** export ONNX (opset-20) → **(5)** W8A8 quantize + HTP link → **(6)** board run. See [`deployment/docs/w8a8_qat_rotated.md`](deployment/docs/w8a8_qat_rotated.md) for the math, gates, and per-stage details.

```bash
# (1) Merge LoRA adapters into the base weights
python deployment/scripts/lora_fp16/export.py \
    --ckpt artifacts/models/checkpoints/epoch=56-val_score=52.28.ckpt \
    --output-dir artifacts/deployment/exports/exported_model

# (2) Mean-preserving rotation (spreads residual outliers; FP32 output-invariant)
python deployment/scripts/qnn/rotate_vision_encoder.py \
    --model-dir artifacts/deployment/exports/exported_model \
    --output-dir artifacts/deployment/exports/exported_model_rotated \
    --input-dir artifacts/deployment/qnn_inputs/vn3k_test_10 --seed 2400 --skip-r2

# (3) Quantization-aware finetune (per-tensor + EMA observer) -> the gate-passing step
python deployment/scripts/qnn/train_vision_quant_robust.py \
    --model-dir artifacts/deployment/exports/exported_model_rotated \
    --train-input-dir artifacts/deployment/qnn_inputs/vn3k_train_4302 \
    --val-input-dir artifacts/deployment/qnn_inputs/vn3k_test_100 \
    --output-dir artifacts/deployment/exports/exported_model_rotated_qat_v3 \
    --start-layer 0 --end-layer 11 \
    --fake-quant-granularity per_tensor --fake-quant-observer ema --ema-momentum 0.99 \
    --batch-size 24 --epochs 8 --lr 1e-5

# (4) Export the rotated/QAT vision encoder to ONNX (opset 20, fused Gelu/LayerNorm)
python deployment/scripts/qnn/export_rotated_vision_onnx.py \
    --model-dir artifacts/deployment/exports/exported_model_rotated_qat_v3 --opset 20

# (5) AI Hub W8A8 quantize + compile + link to a QNN context binary (HTP v68)
python deployment/scripts/qnn/submit_qaihub_quantize_compile.py \
    --model artifacts/deployment/exports/exported_model_rotated_qat_v3/vision_onnx \
    --calibration-data d7jzjy1m2 --weights-dtype int8 --activations-dtype int8 --wait \
    --download artifacts/deployment/runtime/rotated_w8a8_qat_v3/vision_encoder.bin

# (6) Run on RB3 (HTP v68) and evaluate the retrieval gate (see deployment doc §10)
qnn-net-run \
    --backend "$QNN_LIB/libQnnHtp.so" \
    --retrieve_context artifacts/deployment/runtime/rotated_w8a8_qat_v3/vision_encoder.bin \
    --config_file deployment/config/qnn/htp_config_245.json \
    --input_list artifacts/deployment/qnn_inputs/vn3k_test_10/input_list.txt \
    --output_dir artifacts/deployment/qnn_runs/rotated_w8a8_qat_v3 \
    --profiling_level basic --perf_profile high_performance
```

For Qualcomm AI Hub compile commands and RB3 execution details, use [`deployment/README.md`](deployment/README.md).

---

##  Contact

For any questions, please open an issue or contact the authors.
