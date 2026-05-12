
---

# A Hard Negative-Aware Optimization for Multilingual Text-Based Person Search

This repository contains the official implementation for the paper: **"A Hard Negative-Aware Optimization for Multilingual Text-Based Person Search"**, along with ongoing work on noise-robust learning and edge deployment on the **Qualcomm RB3 Gen2**.

##  Abstract

Multilingual Text-Based Person Search (TBPS) remains challenging in low-resource settings due to ambiguous cross-modal alignment. Although recent methods such as TBPS-mSigLIP employ noise-robust contrastive learning, they suffer from **limited gradient discrimination** between easy and hard negatives.

To address this, we propose an efficient optimization framework that integrates **Cross-modal Circle Loss** with **Low-Rank Adaptation (LoRA)**. Circle Loss enhances fine-grained discrimination via adaptive pair-wise re-weighting, while LoRA stabilizes training by constraining optimization to a low-rank subspace. We further introduce a **Curriculum Hard-Mining Schedule** to balance alignment stability and discrimination. Experiments across three typologically diverse languages — Vietnamese, English, and Chinese — demonstrate consistent improvements, establishing a new state-of-the-art **Rank@1 accuracy of 52.28%** on VnPersonSearch and **59.35%** on PRW-TPS-CN, with only **1.57% trainable parameters**. Additionally, we are exploring noise-robust learning strategies and deploying the optimized model on edge hardware (Qualcomm RB3 Gen2) for real-time inference.

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

where $\mathcal{P}$, $\mathcal{N}$ are positive/negative pair sets, $s$ is cosine similarity, $\gamma=128$ is the scale factor, and $m=0.35$ is the margin. The adaptive weights:

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

## Experimental Extension: NACIR (Noise-Aware Circle Loss)

> Status: **experimental / results pending**. This section documents ongoing work and does not modify or reinterpret the reported Circle Loss results above.

### Motivation

Cross-modal Circle Loss is effective for hard-negative mining, but its strength can become a weakness under label noise. In TBPS, two noise regimes are especially harmful:

| Noise type | Label says | True relation | Failure mode |
|---|---|---|---|
| False Negative (FN) | Negative | Same person / semantically matching | Circle Loss pushes matching embeddings apart |
| False Positive (FP) | Positive | Different person / wrong caption | Circle Loss pulls mismatched embeddings together |

NACIR is designed as a drop-in replacement for the auxiliary Circle Loss branch. When its detectors are inactive or uncertain, it degenerates exactly to the original Circle Loss; when label noise is detected, it suppresses the corresponding branch-specific gradient.

### Mathematical Formulation

Let $s_{ij}=\mathbf{v}_i^\top\mathbf{u}_j$ be cosine similarity between normalized image and text embeddings. The original Circle Loss uses:

$$\alpha_p^{ij}=[1+m-s_{ij}]_+, \qquad \alpha_n^{ij}=[s_{ij}+m]_+$$

NACIR introduces branch-specific noise-aware weights:

$$\widetilde{\alpha}_n^{ij} = \alpha_n^{ij}\cdot\max(1-P_{\text{FN}}(s_{ij}), \epsilon_n)$$

$$\widetilde{\alpha}_p^{ij} = \alpha_p^{ij}\cdot\max(w_{ij}, \epsilon_p)$$

where:

- $P_{\text{FN}}(s_{ij})$ is the Bayesian posterior that a labeled negative pair is actually a false negative.
- $w_{ij}$ is the clean-pair probability for a positive pair, computed from per-sample clean weights.
- $\epsilon_n$ and $\epsilon_p$ are safety floors that prevent total gradient collapse.

The resulting Noise-Aware Circle Loss is:

$$
\mathcal{L}_{\text{NACIR}} =
\log\left[
1+
\sum_{j\in\mathcal{N}} e^{\gamma\,\widetilde{\alpha}_n^j(s_n^j-m)}
\cdot
\sum_{i\in\mathcal{P}} e^{-\gamma\,\widetilde{\alpha}_p^i(s_p^i-(1-m))}
\right]
$$

#### False-negative detector

NACIR models positive and negative similarity distributions with running Gaussian statistics:

$$f_+(s)=\mathcal{N}(s;\mu_+,\sigma_+), \qquad f_-(s)=\mathcal{N}(s;\mu_-,\sigma_-)$$

For a labeled negative pair with similarity $s$, the false-negative posterior is:

$$
P_{\text{FN}}(s)=
\frac{\pi_{\text{FN}} f_+(s)}
{\pi_{\text{FN}} f_+(s)+(1-\pi_{\text{FN}})f_-(s)}
$$

If a negative pair lies in the positive distribution region, NACIR reduces the negative-branch force by scaling $\alpha_n$ with $1-P_{\text{FN}}(s)$.

#### False-positive detector

NACIR tracks an exponential moving average of per-sample Circle Loss values and periodically fits a two-component 1D Gaussian mixture model:

$$p(\ell)=\pi_c\mathcal{N}(\ell;\mu_c,\sigma_c^2)+\pi_n\mathcal{N}(\ell;\mu_n,\sigma_n^2)$$

The lower-loss component is treated as clean. The clean probability is:

$$
w_i=P(\text{clean}\mid \ell_i)
$$

For a positive pair $(i,j)$, the implementation uses:

$$w_{ij}=\min(w_i,w_j)$$

If the GMM components are not sufficiently separated, NACIR falls back to $w_i=1$ for all samples, making the FP branch a no-op.

### Curriculum and Safety

NACIR reuses the same Circle Loss curriculum weight:

| Epoch | NACIR weight | FN detector | FP detector | Notes |
|---|---:|---|---|---|
| 0-5 | 0 | off | off | Global alignment warmup |
| 6-10 | ramp | off | off | NACIR behaves as vanilla Circle Loss |
| 11-14 | ramp | on | off | EMA similarity statistics have stabilized |
| 15-20 | ramp | on | on | GMM-based FP detection begins |
| 21-60 | 0.1 | on | on | Stable phase |

Default hyperparameters:

| Parameter | Value |
|---|---:|
| `fn_prior` | 0.01 |
| `epsilon_n` | 0.1 |
| `epsilon_p` | 0.2 |
| `ema_beta` | 0.99 |
| `loss_ema_alpha` | 0.9 |
| `gmm_refit_interval` | 5 |
| `gmm_min_separation` | 1.0 |
| `fn_enable_epoch` | 11 |
| `fp_enable_epoch` | 15 |

### Validation Protocol

Before launching a full training run, NACIR should be validated in `workspace.ipynb`.

Recommended notebook checks:

1. **Clean no-op:** `NACIR(detectors off)` must match vanilla Circle Loss with absolute difference `< 1e-4`.
2. **Synthetic FN:** split true PIDs into fake labels and verify that known false negatives receive higher $P_{\text{FN}}$ and lower negative-branch gradient than vanilla Circle Loss.
3. **Synthetic FP:** replace a controlled fraction of text embeddings with different-PID text embeddings and verify that corrupted samples receive lower clean weights.
4. **No collapse:** NACIR should preserve at least 30% of the vanilla negative-branch gradient in controlled tests.

### How to Run NACIR

#### Notebook validation

Open `workspace.ipynb` and run:

1. Sections 0-3 to load the checkpoint and build the aligned loss batch.
2. Section 4.5 for standard NACIR diagnostics.
3. Section 4.6 for controlled clean/FN/FP validation.

The new validation section prints a PASS/FAIL greenlight table. Full training should only be launched after the controlled checks pass.

#### Full training

Enable NACIR via Hydra overrides:

```bash
uv run trainer.py -cn cir_msiglip \
    loss.NACIR=true \
    loss.CIR=true \
    trainer.max_epochs=60 \
    trainer.accumulate_grad_batches=3 \
    +lora=default
```

Optional robustness experiments with synthetic noisy correspondence:

```bash
uv run trainer.py -cn cir_msiglip \
    loss.NACIR=true \
    dataset.noisy_rate=0.2 \
    dataset.noisy_file="./noiseindex/VN3K_VI_0.2.npy" \
    trainer.max_epochs=60 \
    trainer.accumulate_grad_batches=3 \
    +lora=default
```

Key diagnostics to monitor:

| Metric | Expected behavior |
|---|---|
| `nacir_fn_active` | 0 before epoch 11, 1 from epoch 11 onward |
| `nacir_fp_active` | 0 before epoch 15, 1 from epoch 15 onward |
| `nacir_alpha_n_scale_mean` | near 1.0 on clean data; lower when FN-like negatives are detected |
| `nacir_clean_weight_mean` | near 1.0 on clean data; lower under FP/noisy-correspondence settings |
| `gmm_separation` | should exceed `gmm_min_separation` before FP suppression is trusted |
| `gmm_fallback` | 1 means FP detector is inactive and safely falls back to uniform weights |

### Results Template (Pending)

#### Clean VN3K

| Method | Seed | R@1 | R@5 | R@10 | mAP | mINP | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| LoRA + Curriculum Circle | 2400 | 52.28 | 79.55 | 88.03 | 57.32 | 50.57 | Existing result |
| LoRA + NACIR | TBD | TBD | TBD | TBD | TBD | TBD | Pending |

#### Robustness under synthetic noisy correspondence

| Noise rate | Circle R@1 | NACIR R@1 | Delta | `gmm_separation` | `gmm_fallback` | Notes |
|---:|---:|---:|---:|---:|---:|---|
| 0.0 | TBD | TBD | TBD | TBD | TBD | Pending |
| 0.1 | TBD | TBD | TBD | TBD | TBD | Pending |
| 0.2 | TBD | TBD | TBD | TBD | TBD | Pending |
| 0.4 | TBD | TBD | TBD | TBD | TBD | Pending |

### Current Conclusion

Pending. NACIR should only replace the auxiliary Circle Loss branch if it satisfies both conditions:

1. **No clean regression:** clean VN3K performance remains within an acceptable tolerance of the existing Circle Loss baseline.
2. **Noise robustness:** under controlled or real noisy-correspondence settings, NACIR degrades less than vanilla Circle Loss.

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
├── trainer.py                  # Training entry point (Hydra)
├── lightning_models.py         # LitTBPS (PyTorch Lightning module)
├── lightning_data.py           # TBPSDataModule (data loading, augmentation)
├── test.py                     # Evaluation script
├── workspace.ipynb             # Experiment notebook (embedding analysis, loss playground)
├── run_cir_loss.sh             # Training script (LoRA + Curriculum Circle Loss)
├── run_full_finetune.sh        # Training script (full fine-tuning baseline)
│
├── model/                      # Model architecture
│   ├── tbps.py                 # TBPS forward pass & loss routing
│   ├── objectives.py           # Loss functions (N-ITC, Circle, C-ITC, SimCLR)
│   ├── reid_objectives.py      # ReID-specific objectives
│   ├── build.py                # Backbone builder with layer resize
│   ├── lora.py                 # LoRA integration via PEFT
│   └── siglip/                 # mSigLIP model implementation
│
├── data/                       # Dataset classes & augmentation
│   ├── vn3k_vi.py              # VN3K Vietnamese
│   ├── vn3k_en.py              # VN3K English
│   ├── vn3k_mixed.py           # VN3K mixed-language
│   ├── cuhkpedes.py            # CUHK-PEDES
│   ├── prw_tps_cn.py           # PRW-TPS-CN (Chinese)
│   ├── bases.py                # Base dataset classes
│   ├── sampler.py              # RandomIdentitySampler
│   └── augmentation/           # Image & text augmentation
│
├── solver/                     # Optimization
│   ├── build.py                # Optimizer with param groups
│   └── lr_scheduler.py         # Cosine LR with warmup
│
├── config/                     # Hydra configuration
│   ├── cir_msiglip.yaml        # Main config (composes sub-configs)
│   ├── loss/                   # Loss flags & weights
│   ├── backbone/               # Backbone settings
│   ├── trainer/                # Training hyperparams
│   ├── optimizer/              # AdamW param groups
│   ├── scheduler/              # LR schedule
│   ├── lora/                   # LoRA config
│   ├── dataset/                # Dataset paths
│   ├── tokenizer/              # Tokenizer settings
│   ├── logger/                 # W&B logger config
│   └── aug/                    # Augmentation settings
│
├── utils/                      # Utilities (metrics, visualization, tokenizer)
├── scripts/                    # Helper scripts (checkpoint prep, extraction)
├── experiments/                # Experiment logs & ablation notes
├── knowledge/                  # Research notes & paper drafts
├── figures/                    # Paper figures
├── docs/                       # Documentation
│   ├── ARCHITECTURE.md         # Full architecture with diagrams
│   ├── EXPERIMENT_SUMMARY.md   # Results table & training config
│   └── knowledge.md            # Knowledge base (Vietnamese)
│
├── deployment/                 # Edge deployment & compression
│   ├── scripts/                # mSigLIP pipeline (each approach in subfolder)
│   ├── hardware_profiling/     # RB3 hardware testing (proxy models)
│   ├── logs/                   # Auto-generated logs (timestamped)
│   └── docs/                   # Deployment documentation
│
└── ref/                        # Reference implementations (RDE, etc.)
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

To reproduce the results, use the provided scripts. We utilize LoRA to enable large batch sizes (BS=24 on 12GB VRAM).

### Train with Curriculum Hard-Mining (Recommended)

This runs the proposed method: LoRA + mSigLIP + Auxiliary Circle Loss with a warm-up schedule.

```bash
# Run Circle Loss with LoRA fine-tuning and curriculum scheduling
./run_cir_loss.sh

```

### Train Baseline (mSigLIP)

```bash
uv run trainer.py -cn m_siglip img_size_str="'(256,256)'" dataset=vn3k loss.softlabel_ratio=0.0 trainer.max_epochs=60

```

---

## Ongoing Work

- **Noise Handling** — Investigating noise-robust learning strategies to improve training stability and robustness against noisy text-image pairs in low-resource settings.
- **Edge Deployment** — Optimizing and deploying the model on the Qualcomm RB3 Gen2 (QCS6490, 4GB RAM, ARM64) via LoRA merging, FP16/ONNX export, and Qualcomm SNPE acceleration. See [`deployment/`](deployment/) for scripts and documentation.

---

##  Contact

For any questions, please open an issue or contact the authors.
