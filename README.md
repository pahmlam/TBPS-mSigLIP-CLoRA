
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
| **Deployment** | Vision encoder runs **INT8 (W8A8) on HTP v68** via GELU fusion + mean-preserving rotation + QAT; best QDQ/retrieval candidate is **QAT v6: T2I R@1 49.30 / I2T R@1 53.85**; QAT v4 is board-verified | Quantize text encoder, then end-to-end board retrieval |

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

> **Canonical deployment references:** [`deployment/docs/w8a8_qat_rotated.md`](deployment/docs/w8a8_qat_rotated.md) explains the rotation/QAT method and gates; [`deployment/docs/journal/[deploy-master].md`](deployment/docs/journal/[deploy-master].md) is the consolidated AI Hub/QNN/RB3 journal; [`deployment/docs/runbook-w8a8-v8-both-int8.md`](deployment/docs/runbook-w8a8-v8-both-int8.md) is the forward runbook for learned rotation and both-INT8 work.

### Current Status (2026-06-17)

| Area | Status | Notes |
|---|---|---|
| FP32/FP16 export (LoRA merge) | PASS | `deployment/scripts/lora_fp16/export.py` merges LoRA into `exported_model` |
| Rotation equalization | PASS | mean-preserving `Q` + folded gamma/beta; output-invariant; residual concentration 252x -> 5.3x |
| ONNX export (opset 20) | PASS | fused `Gelu` and fused `LayerNormalization`; static control ~= 1.0 |
| Vision W8A8 QDQ fidelity | PASS | QAT v6 QDQ cosine `0.9491 / 0.9266` mean/min |
| Vision compile/link on v68 | PASS | all-INT8 links on HTP v68; deployed context binary is about 90 MB |
| Vision on HTP board | PASS | QAT v4 board fidelity `0.9363 / 0.9068`, `32.70 ms/img`, `22.88 FPS` |
| Vision retrieval gate | PASS | Current best QDQ/retrieval candidate is QAT v6: T2I R@1 `49.30`, I2T R@1 `53.85`; QAT v4 remains board-verified at T2I R@1 `48.50` |
| Text encoder | Pending | Text is 75% of params; replicate rotation + QAT + W8A8 after vision is accepted or stretch work completes |
| End-to-end board retrieval | Pending | Both encoders INT8 on board |

Key deployment findings:

- HTP **v68 rejects 16-bit activations (A16)** for LayerNorm and attention matmul, so W8A16 can reach high QDQ fidelity but still fail context linking on v68.
- Plain W8A8 collapses due to residual-stream activation concentration; the deployable path is all-INT8 W8A8 plus model-side equalization.
- Opset-20 fused `Gelu` avoids exposing tanh-GELU `Pow(x^3)` to quantization. Fused `LayerNormalization` must stay fused; converting to RMSNorm re-exposes `Pow(x^2)`.
- Mean-preserving rotation spreads residual outliers while preserving LayerNorm behavior; QAT then trains the rotated model to tolerate the deploy-faithful W8A8 quantizer.

Do not treat `_float` QDQ surgery candidates as deployable; they are diagnostics and link-fail on internal float. Do not use W8A16 on v68; A16 needs v73-class support.

### Model Footprint

Full `model_fp32.pt` is the whole TBPS model (vision + text). The deployed `vision_encoder.bin` is the vision encoder only. Parameter counts below are deduplicated because TBPS aliases `vision_model`/`text_model` to `backbone.*`.

| Component | Params | FP32 | FP16 | INT8 |
|---|---:|---:|---:|---:|
| vision_model | 92.9 M | 372 MB | 186 MB | **93 MB** (deployed `.bin` ~= 90 MB) |
| text_model | 277.7 M | 1111 MB | 555 MB | 278 MB |
| projection + other | 1.2 M | 5 MB | 2 MB | 1 MB |
| **TOTAL** | **371.8 M** | **1487 MB** | **744 MB** | **372 MB** |

Text is 75% of parameters. The token embedding alone is `250000 x 768` = 192 M params (768 MB FP32), driven by the 250k multilingual vocabulary. On a 4 GB board, text FP32 (1.1 GB) is the real memory cost; text INT8 is the next lever.

### Deployment Directory Map

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
      prepare_vn3k_text_inputs.py      # Build token/input-list assets for text work
      upload_qaihub_calibration_dataset.py
      submit_qaihub_quantize_compile.py
      submit_qaihub_compile_link.py
      compare_onnx_with_pytorch.py
      compare_qnn_with_pytorch.py
      eval_retrieval_quantized_vision.py
      qdq_surgery.py
      analyze_qdq_encodings.py
      tune_qdq_activation_encodings.py
      train_vision_quant_robust.py
      learn_rotation.py
      audit_qnn_native_env.py
  config/qnn/                          # HTP/QNN runtime JSON configs
  demo/                                # Modular demo-system scaffold
  docs/
    journal/                           # Master deploy journal + demo-system logs
    deployment-plan.md                 # Older high-level plan; verify against journal
    system.md                          # RB3 hardware notes
    experiment.md                      # Proxy benchmark guide
    benchmark-rp.md                    # Proxy benchmark results
    w8a8_qat_rotated.md                # Canonical rotation/QAT method
    runbook-w8a8-v8-both-int8.md       # Current forward runbook for v8/text/both-INT8
  hardware_profiling/                  # RB3 proxy benchmark scripts
  deploy_utils.py
```

Generated deployment artifacts and logs belong under `artifacts/deployment/`.

### Target Device

| Component | Specification |
|---|---|
| SoC | Qualcomm QCS6490 |
| CPU | 4x Cortex-A78 + 4x Cortex-A55 |
| GPU | Adreno 643 |
| DSP/HTP | Hexagon 770 / HTP V68 |
| RAM | About 4 GB usable for inference |
| Runtime target | QNN context binary on HTP |

Important hardware constraint: HTP requires integer tensors at the model I/O boundary. Internal floating-point tensors can also break context linking, depending on the graph pattern.

### Deployment Gates

Do not advance a candidate unless it passes the relevant gate.

| Gate | Threshold | Meaning |
|---|---:|---|
| Merge LoRA | no `lora` / `adapter` / `base_layer` keys | Export must be a deployable dense model |
| Rotation FP32 invariance | cosine min `>= 0.9999` | Rotation must preserve model behavior |
| Static ONNX vs PyTorch | `cosine_l2_mean >= 0.999` | Export/preprocess control |
| ONNX op sanity | `Pow=0`, fused `Gelu`, fused `LayerNormalization` | Avoid exposed GELU/RMSNorm internals |
| QDQ ONNX vs PyTorch mean | target `>= 0.95` | Candidate worth compile/link |
| QDQ ONNX vs PyTorch min | `>= 0.90` | No severe sample drift |
| QNN vs PyTorch after link | `>= 0.90` | Candidate worth wider RB3 benchmark |
| Full retrieval | `T2I R@1 >= 48.0` | Minimum deploy target vs `52.28` / `52.40` FP32 baseline |
| Stretch retrieval | `T2I R@1 >= 50.0` | Current optimization target |

Diagnostic compile exception: AI Hub-native QDQ from a QAT model may be compiled if it reaches `mean >= 0.93` and `min >= 0.88`. This exception does **not** apply to `_float`, ORT QDQ, or INT16 surgery patterns that have already link-failed.

Cosine is only a fidelity proxy. Retrieval R@1 is decisive: the rotation-only candidate has QDQ cosine near `0.90` but fails retrieval at T2I R@1 `45.42`.

### Pipeline (current, rotation-based INT8 for v68)

```text
epoch=56-val_score=52.28.ckpt        # LoRA-finetuned Lightning checkpoint
  -> [1] lora_fp16/export.py          # MERGE LoRA -> dense base weights
         exported_model/{model_fp32.pt, model_fp16.pt, config.yaml}
  -> [2] qnn/rotate_vision_encoder.py # mean-preserving rotation + folded gamma/beta
         exported_model_rotated/{model_fp32.pt, config.yaml}
  -> [3] qnn/train_vision_quant_robust.py
         exported_model_rotated_qat_v6/{model_fp32.pt, config.yaml}
  -> [4] qnn/export_rotated_vision_onnx.py --opset 20
         exported_model_rotated_qat_v6/vision_onnx/{vision_encoder.onnx,.data}
  -> [5] qnn/submit_qaihub_quantize_compile.py
         W8A8 QDQ / context binary with quantized I/O on RB3 Gen2
  -> [6] qnn-net-run on RB3 HTP -> compare_qnn_with_pytorch -> retrieval R@1
```

Why each step exists: LoRA merge is mandatory because the checkpoint is LoRA-finetuned; rotation spreads activation concentration so all-INT8 W8A8 becomes viable on v68; opset-20 fuses `Gelu`/`LayerNormalization`; QAT aligns the model with AI Hub's calibrate-once, per-tensor W8A8 quantizer. Swapping to a different checkpoint (for example the 53.00 Part-Align model) requires rerunning from step [1].

### QAT Iteration Workflow (per-round commands)

Rotation alone gives only `T2I R@1 45.42` (gate fail). Quantization-aware fine-tuning of the rotated model closes the gap. Set `V` to the round name, for example `qat_v6`. AI Hub steps are the only cloud-job steps; the other steps are local/free.

**One-time prep** (shared by QAT rounds):

```bash
# [A] Merge LoRA -> exported_model/
python deployment/scripts/lora_fp16/export.py \
  --ckpt artifacts/models/checkpoints/epoch=56-val_score=52.28.ckpt \
  --output-dir artifacts/deployment/exports/exported_model

# [B] Mean-preserving rotation -> exported_model_rotated/
python deployment/scripts/qnn/rotate_vision_encoder.py \
  --model-dir artifacts/deployment/exports/exported_model \
  --output-dir artifacts/deployment/exports/exported_model_rotated \
  --input-dir artifacts/deployment/qnn_inputs/vn3k_test_10 --seed 2400 --skip-r2

# [C] Prepare full-train inputs for QAT (4302 images)
python deployment/scripts/qnn/prepare_vn3k_vision_inputs.py \
  --dataset-root VN3K --split train --selection random --seed 2400 \
  --num-samples 4302 --output-dir artifacts/deployment/qnn_inputs/vn3k_train_all_4302 \
  --path-mode relative
```

**Per round** (`qat_v6` shown):

```bash
# [1] QAT distillation (teacher=rotated FP32, student=fake-quant). LONG (GPU).
python deployment/scripts/qnn/train_vision_quant_robust.py \
  --model-dir artifacts/deployment/exports/exported_model_rotated \
  --train-input-dir artifacts/deployment/qnn_inputs/vn3k_train_all_4302 \
  --val-input-dir artifacts/deployment/qnn_inputs/vn3k_test_100 \
  --output-dir artifacts/deployment/exports/exported_model_rotated_qat_v6 \
  --device cuda --batch-size 16 --epochs 15 --lr 1e-5 \
  --start-layer 0 --end-layer 11 --num-workers 4 \
  --fake-quant-observer ema \
  --quant-head --quant-linears --quant-attention

# [2] Export rotated/QAT ONNX, opset 20 (fused Gelu/LayerNorm).
python deployment/scripts/qnn/export_rotated_vision_onnx.py \
  --model-dir artifacts/deployment/exports/exported_model_rotated_qat_v6 \
  --opset 20

# [3] Static control gate (ONNX vs PyTorch, must be ~1.0).
python deployment/scripts/qnn/compare_onnx_with_pytorch.py \
  --onnx-model artifacts/deployment/exports/exported_model_rotated_qat_v6/vision_onnx \
  --model-dir artifacts/deployment/exports/exported_model_rotated_qat_v6 \
  --input-dir artifacts/deployment/qnn_inputs/vn3k_test_10 \
  --precision fp32 \
  --json artifacts/deployment/exports/exported_model_rotated_qat_v6/static_vs_pytorch_summary.json \
  --csv artifacts/deployment/exports/exported_model_rotated_qat_v6/static_vs_pytorch.csv

# [4] AI Hub W8A8 quantize-only -> QDQ ONNX. CLOUD JOB (log the job id).
python deployment/scripts/qnn/submit_qaihub_quantize_compile.py \
  --model artifacts/deployment/exports/exported_model_rotated_qat_v6/vision_onnx \
  --calibration-data d7jzjy1m2 --weights-dtype int8 --activations-dtype int8 \
  --quantize-only --wait \
  --download-quantized artifacts/deployment/runtime/rotated_w8a8_qat_v6/job_qdq_onnx

# [5] Decisive local numbers: QDQ cosine + retrieval R@1.
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

If R@1 passes the gate, deploy with the full flow (drop `--quantize-only`, use `--download .../vision_encoder.bin`) and then run `qnn-net-run` on the board.

**What each round changed, and the result:**

| Round | QAT flags added | QDQ cosine mean/min | T2I R@1 | I2T R@1 | Decision |
|---|---|---:|---:|---:|---|
| rotation only | no QAT | `0.8975 / 0.8747` | 45.42 | 49.40 | Links/runs, retrieval gate FAIL |
| qat_v1 | per-sample fake-quant | `0.9223 / 0.8917` | 46.92 | 50.45 | Helpful but sim too easy |
| qat_v2 | per-tensor fake-quant | `0.9281 / 0.9093` | 47.80 | 51.65 | Near gate |
| qat_v3 | `--fake-quant-observer ema` | `0.9353 / 0.919` | 48.20 | 52.30 | First retrieval gate PASS |
| qat_v4 | `--quant-head` | `0.9364 / 0.9091` | 48.50 | 52.95 | Board-verified deploy binary |
| qat_v5 | `--quant-linears` | `0.9437 / 0.9311` | 49.25 | 53.40 | Strong accuracy candidate |
| **qat_v6** | `--quant-attention` | **`0.9491 / 0.9266`** | **49.30** | **53.85** | Best QDQ/retrieval candidate |
| qat_v7 | cosine LR + lr `2e-5`, 20 epochs | `0.9485 / 0.9083` | 48.38 | 53.05 | Regressed |
| qat_v8 planned | learned rotation + recipe v6 | pending | pending | pending | Forward ablation |

Dated commands, AI Hub job IDs, and full results per round are consolidated in [`deployment/docs/journal/[deploy-master].md`](deployment/docs/journal/[deploy-master].md). Method and math are in [`deployment/docs/w8a8_qat_rotated.md`](deployment/docs/w8a8_qat_rotated.md).

### Quick Commands

#### 1. Analyze a checkpoint

```bash
python deployment/scripts/analyze_checkpoint.py \
  --ckpt artifacts/models/checkpoints/epoch=56-val_score=52.28.ckpt
```

#### 2. Merge LoRA and export FP32/FP16 state

```bash
python deployment/scripts/lora_fp16/export.py \
  --ckpt artifacts/models/checkpoints/epoch=56-val_score=52.28.ckpt \
  --output-dir artifacts/deployment/exports/exported_model
```

#### 3. Export baseline ONNX

```bash
python deployment/scripts/onnx/export.py \
  --model-dir artifacts/deployment/exports/exported_model \
  --precision fp32
```

Use FP32 ONNX for stability unless a specific diagnostic requires FP16. The current deploy path uses `export_rotated_vision_onnx.py` after rotation/QAT.

#### 4. Prepare raw VN3K vision inputs

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

Known calibration dataset:

```text
d7jzjy1m2 / msiglip-vision-vn3k-train-calib-2000
```

#### 5. Upload calibration data to Qualcomm AI Hub

```bash
python deployment/scripts/qnn/upload_qaihub_calibration_dataset.py \
  --input-dir artifacts/deployment/qnn_inputs/vn3k_train_calib_2000 \
  --name msiglip-vision-vn3k-train-calib-2000
```

#### 6. Quantize and compile/link with the newer AI Hub API

Use this helper instead of deprecated `qai-hub submit-compile-job --quantize_full_type ...`, because the deprecated CLI can preserve FP I/O and make HTP reject the model.

For QDQ diagnostics first:

```bash
python deployment/scripts/qnn/submit_qaihub_quantize_compile.py \
  --model artifacts/deployment/exports/exported_model_rotated_qat_v6/vision_onnx \
  --calibration-data d7jzjy1m2 \
  --weights-dtype int8 --activations-dtype int8 \
  --quantize-only \
  --wait \
  --download-quantized artifacts/deployment/runtime/rotated_w8a8_qat_v6/job_qdq_onnx
```

For full quantize + compile + link:

```bash
python deployment/scripts/qnn/submit_qaihub_quantize_compile.py \
  --model artifacts/deployment/exports/exported_model_rotated_qat_v6/vision_onnx \
  --calibration-data d7jzjy1m2 \
  --weights-dtype int8 --activations-dtype int8 \
  --wait \
  --download artifacts/deployment/runtime/rotated_w8a8_qat_v6/vision_encoder.bin
```

#### 7. Compare QDQ ONNX against PyTorch

```bash
python deployment/scripts/qnn/compare_onnx_with_pytorch.py \
  --onnx-model artifacts/deployment/runtime/rotated_w8a8_qat_v6/job_qdq_onnx \
  --model-dir artifacts/deployment/exports/exported_model \
  --input-dir artifacts/deployment/qnn_inputs/vn3k_test_10 \
  --precision fp32 \
  --json artifacts/deployment/runtime/rotated_w8a8_qat_v6/qdq_vs_pytorch_summary.json \
  --csv artifacts/deployment/runtime/rotated_w8a8_qat_v6/qdq_vs_pytorch.csv
```

Only compile/link candidates that pass the QDQ gate, except for the documented diagnostic exception above.

#### 8. Run a QNN context binary on RB3

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

Use `qnn-net-run`, not `snpe-net-run`, for QNN context binaries.

#### 9. Compare QNN outputs against PyTorch

```bash
python deployment/scripts/qnn/compare_qnn_with_pytorch.py \
  --qnn-output-dir artifacts/deployment/qnn_runs/rotated_w8a8_qat_v6 \
  --model-dir artifacts/deployment/exports/exported_model \
  --input-dir artifacts/deployment/qnn_inputs/vn3k_test_10 \
  --precision fp32 \
  --json artifacts/deployment/qnn_runs/rotated_w8a8_qat_v6/qnn_vs_pytorch_summary.json \
  --csv artifacts/deployment/qnn_runs/rotated_w8a8_qat_v6/qnn_vs_pytorch.csv
```

#### 10. Audit QNN/QAIRT native toolchain

```bash
python deployment/scripts/qnn/audit_qnn_native_env.py \
  --json artifacts/deployment/runtime/qnn_native/env_audit.json
```

Latest local Mac audit found no QNN/QAIRT native tools. Native QNN work needs a server or machine with the Qualcomm AI Stack / QNN SDK installed.

### Current Key Results

| Candidate | Result | Decision |
|---|---:|---|
| Rotation-only W8A8 | QDQ `0.8975 / 0.8747`, T2I R@1 `45.42`, I2T R@1 `49.40` | Links/runs, retrieval gate FAIL |
| QAT v3 | QDQ `0.9353 / 0.919`, T2I R@1 `48.20`, I2T R@1 `52.30` | First retrieval gate PASS |
| QAT v4 | QDQ `0.9364 / 0.9091`, T2I R@1 `48.50`, I2T R@1 `52.95`; board `0.9363 / 0.9068` | Current board-verified deploy binary |
| QAT v5 | QDQ `0.9437 / 0.9311`, T2I R@1 `49.25`, I2T R@1 `53.40` | Strong accuracy candidate |
| **QAT v6** | QDQ `0.9491 / 0.9266`, T2I R@1 `49.30`, I2T R@1 `53.85` | Current best QDQ/retrieval candidate |
| QAT v7 | QDQ `0.9485 / 0.9083`, T2I R@1 `48.38`, I2T R@1 `53.05` | Regressed vs v6 |
| QAT v8 planned | learned rotation + recipe v6 | Next ablation toward stretch target |

Historical failed diagnostics, rejected branches, AI Hub job IDs, and full artifacts are consolidated in [`deployment/docs/journal/[deploy-master].md`](deployment/docs/journal/[deploy-master].md).

### Documentation Convention

- Canonical deployment/model-compression journal: [`deployment/docs/journal/[deploy-master].md`](deployment/docs/journal/[deploy-master].md)
- Demo-system logs: `deployment/docs/journal/[demo-system]-YYYY-MM-DD.md`
- Stable concepts such as ONNX, QNN, HTP, PTQ, and QAT: `docs/knowledge.md`
- Completed code/config/docs changes: `changelog/deployment/changelog.md` after user confirmation.

Write new AI Hub job logs and QDQ/QNN fidelity results into `deployment/docs/journal/[deploy-master].md`. Do not write them into `deployment/docs/aihub-experiments.md`; it is legacy.

### Hardware Profiling

Proxy hardware profiling scripts remain under `deployment/hardware_profiling/`. They are useful for RB3 environment checks, but they are not acceptance tests for mSigLIP. Deployment acceptance requires mSigLIP QNN fidelity and retrieval metrics.

```bash
cd deployment/hardware_profiling
./run_all.sh
```

---

##  Contact

For any questions, please open an issue or contact the authors.
