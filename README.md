
---

# A Hard Negative-Aware Optimization for Multilingual Text-Based Person Search

This repository contains the official implementation for the paper: **"A Hard Negative-Aware Optimization for Multilingual Text-Based Person Search"**, along with ongoing work on noise-robust learning and edge deployment on the **Qualcomm RB3 Gen2**.

##  Abstract

Multilingual Text-Based Person Search (TBPS) remains challenging in low-resource settings due to ambiguous cross-modal alignment. Although recent methods such as TBPS-mSigLIP employ noise-robust contrastive learning, they suffer from **limited gradient discrimination** between easy and hard negatives.

To address this, we propose an efficient optimization framework that integrates **Cross-modal Circle Loss** with **Low-Rank Adaptation (LoRA)**. Circle Loss enhances fine-grained discrimination via adaptive pair-wise re-weighting, while LoRA stabilizes training by constraining optimization to a low-rank subspace. We further introduce a **Curriculum Hard-Mining Schedule** to balance alignment stability and discrimination. Experiments across three typologically diverse languages — Vietnamese, English, and Chinese — demonstrate consistent improvements, establishing a new state-of-the-art **Rank@1 accuracy of 52.28%** on VnPersonSearch and **59.35%** on PRW-TPS-CN, with only **1.57% trainable parameters**. Additionally, we are exploring noise-robust learning strategies and deploying the optimized model on edge hardware (Qualcomm RB3 Gen2) for real-time inference.

---

## Current Status Snapshot

| Track | Current state | Next step |
|---|---|---|
| **Main training result** | LoRA + Curriculum Circle Loss reaches **52.28% R@1** on VN3K and **59.35% R@1** on PRW-TPS-CN | Preserve as the reported baseline |
| **MNEB-HN** | Implemented as an optional modular noise-robust extension; `run_mneb_hn.sh` is available | Validate clean no-op on VN3K, then test natural noise on CUHK-PEDES |
| **Noisy correspondence** | RDE-style caption-shuffle noise is integrated via `dataset.noisy_rate` and `run_noise_experiments.sh` | Use for robustness experiments, mainly FP/noisy-positive validation |
| **Deployment** | LoRA merge, FP16/FP32 export, ONNX export, and **vision INT8 HTP compile** are working | Compile text encoder, benchmark on RB3, then repeat with real calibration data |

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

## Experimental Extension: MNEB-HN

> Status: **experimental / results pending**. This section documents the next noise-robust framework and does not modify or reinterpret the reported Circle Loss results above.

### Motivation

Cross-modal Circle Loss is effective because it keeps strong pressure on hard negatives. A noise-robust extension must therefore be conservative: if a pair may simply be a true hard negative, the framework should leave Circle Loss alone. The current design treats uncertain noise evidence as a no-op and applies FN/FP correction only through auxiliary losses.

The target noise regimes are:

| Noise type | Label says | True relation | MNEB-HN response |
|---|---|---|---|
| False Negative (FN) | Negative | Same person / semantically matching | Add an FNM-style auxiliary correction for high-confidence FN candidates |
| False Positive (FP) | Positive | Different person / wrong caption | Add an RDE-style auxiliary loss from global/local clean-noisy consensus |
| Hard Negative (HN) | Negative | Different but visually/textually similar | Keep the original Circle Loss responsible for separation |

### Framework Design

MNEB-HN stands for **Multilingual Noise Evidence Bank for Hard-Negative TBPS**. It keeps the main objective unchanged:

$$\mathcal{L}_{\text{main}} = \mathcal{L}_{N\text{-}ITC/MVS} + \alpha_5(t)\mathcal{L}_{\text{circle}} + 0.1\mathcal{L}_{C\text{-}ITC} + 0.4\mathcal{L}_{SS}$$

When enabled, MNEB-HN adds a cross-epoch evidence memory bank and optional auxiliary terms:

$$\mathcal{L}_{\text{MNEB-HN}} = \mathcal{L}_{\text{main}} + \lambda_{\text{FN}}\mathcal{L}_{\text{fnm-aux}} + \lambda_{\text{FP}}\mathcal{L}_{\text{rde-aux}} $$

The key constraint is that MNEB-HN **does not mutate Circle Loss weights**. It never directly suppresses Circle's $\alpha_n$ or $\alpha_p$. Instead:

- `EvidenceMemoryBank` stores global/local embeddings, per-sample loss EMA, clean probabilities, seen counts, FIFO sample IDs, FN similarity statistics, and global/local consensus labels.
- `FNMStyleAuxLoss` acts only when high-confidence FN candidates are found; otherwise it returns a grad-safe zero.
- `RDEStyleAuxLoss` acts only for confident clean/noisy consensus; uncertain samples no-op.
- Local evidence reuses the existing part-token path and adds no new trainable projection heads in v1.

### Default Safety

MNEB-HN is disabled by default in `configs/loss/cir_msiglip.yaml`:

```yaml
MNEB: false

mneb_config:
  evidence_bank:
    enabled: true
  fnm_aux:
    enabled: false
  rde_aux:
    enabled: false
```

Behavior guarantees:

| Setting | Training effect |
|---|---|
| `MNEB=false` | Identical baseline path: no evidence bank, no hidden-state request, no auxiliary losses |
| `MNEB=true`, aux disabled | Evidence/diagnostics only; total loss remains the baseline objective |
| `fnm_aux.enabled=true` | FN correction enters only through `fnm_aux_loss` |
| `rde_aux.enabled=true` | FP correction enters only through `rde_aux_loss` |

### How to Run MNEB-HN

Run the dedicated script:

```bash
./run_mneb_hn.sh
```

If the script is not executable on your machine:

```bash
bash run_mneb_hn.sh
```

For evidence-only diagnostics without auxiliary training effects:

```bash
./run_mneb_hn.sh loss.mneb_config.fnm_aux.enabled=false loss.mneb_config.rde_aux.enabled=false
```

Key diagnostics to monitor:

| Metric | Expected behavior |
|---|---|
| `mneb_seen_frac` | Increases as the bank observes training samples |
| `mneb_local_seen_frac` | Tracks local evidence coverage when part-token hidden states are available |
| `mneb_fn_stats_ready` | 1 once in-batch positive/negative similarity statistics are initialized |
| `mneb_consensus_clean_frac` | Confident clean fraction from global/local agreement |
| `mneb_consensus_noisy_frac` | Confident noisy fraction from global/local agreement |
| `mneb_consensus_uncertain_frac` | Should remain high when evidence is weak; uncertainty is a safe no-op |
| `fnm_aux_loss` | Non-zero only when high-confidence FN candidates pass the gates |
| `rde_aux_loss` | Non-zero only when consensus labels provide confident anchors |

### Validation Plan

Clean VN3K is treated as the no-regression benchmark. The expected behavior is that noise modules mostly no-op and true hard negatives remain under Circle Loss.

| Method | Seed | R@1 | R@5 | R@10 | mAP | mINP | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| LoRA + Curriculum Circle | 2400 | 52.28 | 79.55 | 88.03 | 57.32 | 50.57 | Existing result |
| LoRA + Curriculum Circle + MNEB-HN | TBD | TBD | TBD | TBD | TBD | TBD | Pending |

CUHK-PEDES is the main natural-noise validation target:

| Dataset | Baseline target | MNEB-HN target | Notes |
|---|---:|---:|---|
| VN3K | Preserve 52.28 R@1 seed-2400 region | No clean regression | Clean multilingual benchmark |
| CUHK-PEDES | Improve beyond current full-CUHK 71.85 R@1 | Close English SOTA gap | Natural FN/FP noise benchmark |
| Synthetic FP/FN stress tests | Detector precision and no-op rate | Hard-negative preservation | Caption shuffle, PID split, and mixed noise |

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
│   ├── lightning_data.py              # TBPSDataModule, noisy correspondence injection
│   ├── model/                         # TBPS + mSigLIP + losses
│   ├── data/                          # Dataset classes & augmentation
│   ├── solver/                        # Optimizer and LR scheduler
│   └── utils/                         # Metrics, visualization, tokenizer utilities
├── trainer.py                         # Backward-compatible wrapper
├── test.py                            # Backward-compatible wrapper
├── notebooks/workspace.ipynb          # Notebook lab for embedding/loss validation
├── run_cir_loss.sh                    # LoRA + Curriculum Circle Loss training
├── run_mneb_hn.sh                     # MNEB-HN noise-robust training script
├── run_noise_experiments.sh           # RDE-style noisy-correspondence sweep
├── run_full_finetune.sh               # Full fine-tuning baseline
├── configs/                           # Hydra configuration
│   ├── cir_msiglip.yaml               # Main config
│   ├── paths/default.yaml             # Centralized data/artifact paths
│   ├── loss/cir_msiglip.yaml          # Loss flags, Circle, MNEB-HN config
│   └── ...                            # backbone, trainer, optimizer, dataset, tokenizer, logger, aug
├── artifacts/                         # Ignored generated outputs
│   ├── training/                      # Hydra runs, multirun, noisy index files
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
│   │   ├── journal/                   # Dated deployment logs and decisions
│   │   ├── system.md                  # RB3 hardware specs
│   │   └── benchmark-rp.md            # Proxy benchmark results
│   └── config/qnn/                    # QNN/HTP runtime config JSON files
│
└── ref/                               # Reference implementations (RDE, etc.)
```

---

## Documentation Convention

- `docs/knowledge.md` stores durable Vietnamese knowledge only: concepts, definitions, mechanisms, and stable trade-offs.
- `docs/journal/[train]-YYYY-MM-DD.md` stores dated training/model-optimization results: commands, logs, metrics, temporary conclusions, and next experiment decisions.
- `deployment/docs/journal/[deploy]-YYYY-MM-DD.md` stores dated deployment results: AI Hub jobs, QNN/QDQ fidelity, RB3 runtime, artifacts, and next deploy steps.
- Reviewer responses and paper wording belong under `knowledge/response.md` or `knowledge/paper/`, not `docs/knowledge.md`.
- Before adding documentation, classify the content first and confirm the target file unless the user explicitly requested that documentation update.
- Use the templates in `docs/knowledge.md`, `docs/journal/README.md`, and `deployment/docs/journal/README.md` for new entries.

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

### Train MNEB-HN (Experimental)

This runs the modular noise-robust extension on top of the current LoRA + Curriculum Circle baseline. The script enables the evidence bank, FNM-style auxiliary loss, and RDE-style auxiliary loss while leaving Circle Loss unchanged.

```bash
./run_mneb_hn.sh
```

For evidence-only diagnostics without auxiliary loss effects:

```bash
./run_mneb_hn.sh loss.mneb_config.fnm_aux.enabled=false loss.mneb_config.rde_aux.enabled=false
```

### Run Noisy-Correspondence Sweeps

This runs RDE-style caption-shuffle noisy correspondence over `noisy_rate=0.1..0.8` using the current Circle Loss route.

```bash
./run_noise_experiments.sh
```

Noise files are saved under `artifacts/training/noiseindex/` so repeated runs reuse the same index mapping.

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

Current progress:

| Stage | Status | Notes |
|---|---|---|
| Checkpoint analysis | Done | `deployment/scripts/analyze_checkpoint.py` |
| LoRA merge + FP16/FP32 export | Done | `deployment/scripts/lora_fp16/export.py` |
| ONNX export | Done | `deployment/scripts/onnx/export.py`, external-weight directories for vision/text |
| Local ONNX FP16 conversion | Done | `deployment/scripts/onnx/to_fp16.py` |
| AI Hub HTP compile | Vision done | INT8 dummy-calibration compile succeeded for vision encoder, job `jgkr7qwn5` |
| Text encoder compile | Pending | Needs same INT8 pipeline |
| On-device RB3 benchmark | Pending | Download compiled `.bin`, run `qnn-net-run` with QAIRT/QNN |
| Production calibration | Pending | Replace dummy calibration with real VN3K image/text calibration data |
| Quantized accuracy check | Pending | Target: R@1 within acceptable drop from FP32 baseline |

Key deployment finding: QCS6490 HTP rejects floating-point I/O. The working path is INT8 I/O and INT8 quantization for HTP context binaries. See [`deployment/docs/deployment-plan.md`](deployment/docs/deployment-plan.md) for current status and [`deployment/docs/journal/`](deployment/docs/journal/) for dated AI Hub/QNN job logs.

Quick deployment commands:

```bash
python deployment/scripts/lora_fp16/export.py \
    --ckpt artifacts/models/checkpoints/epoch=56-val_score=52.28.ckpt \
    --output-dir artifacts/deployment/exports/msiglip_lora

python deployment/scripts/onnx/export.py \
    --model-dir artifacts/deployment/exports/msiglip_lora \
    --precision fp32
```

For Qualcomm AI Hub compile commands and RB3 execution details, use [`deployment/README.md`](deployment/README.md).

---

##  Contact

For any questions, please open an issue or contact the authors.
