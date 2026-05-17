# Training Framework Architecture

## Overview

A dual-encoder framework built on **mSigLIP** for multilingual Text-Based Person Search (TBPS). The model learns to align person images and text descriptions in a shared 768-dim embedding space, optimized with four complementary loss functions and a curriculum learning schedule.

```
                        ┌────────────────────────────────────-─-┐
                        │           Input Batch                │
                        │  images, captions, pids, aug_images  │
                        │  ss_images1, ss_images2              │
                        └──────────┬─────────────────────────--─┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼               ▼
             ┌────────────┐ ┌──────────┐  ┌──────────────┐
             │   Vision   │ │   Text   │  │  SimCLR MLP  │
             │  Encoder   │ │ Encoder  │  │  (768→768)   │
             │  (ViT-B)   │ │ (BERT)   │  └──────┬───────┘
             │  +LoRA     │ │ +LoRA    │         │
             └─────┬──────┘ └────┬─────┘         │
                   │             │                │
                   ▼             ▼                ▼
              img_embed     txt_embed      ss_embed_1/2
               (768d)        (768d)          (768d)
                   │             │                │
                   └──────┬──────┘                │
                          │                       │
          ┌───────────────┼───────────────┐       │
          ▼               ▼               ▼       ▼
    ┌──────────┐   ┌────────────┐  ┌─────────┐ ┌────────┐
    │  N-ITC   │   │ Circle Loss│  │  C-ITC  │ │ SimCLR │
    │ (w=1.0)  │   │ (curriculum│  │ (w=0.1) │ │(w=0.4) │
    │ +MVS aug │   │  0→0.1)    │  │         │ │        │
    └────┬─────┘   └─────┬──────┘  └────┬────┘ └───┬────┘
         │               │              │          │
         └───────────────┴──────────────┴──────────┘
                          │
                     total_loss
```

## Module Hierarchy

```
trainer.py                          # Entry point (Hydra)
├── src/msiglip/lightning_data.py               # TBPSDataModule (data loading, augmentation)
│   ├── src/msiglip/data/vn3k_vi.py             # VN3K dataset class
│   ├── src/msiglip/data/bases.py               # ImageTextDataset, ImageDataset, TextDataset
│   ├── src/msiglip/data/sampler.py             # RandomIdentitySampler
│   └── src/msiglip/data/augmentation/          # Image & text augmentation pools
│
└── src/msiglip/lightning_models.py             # LitTBPS (PyTorch Lightning module)
    ├── src/msiglip/model/build.py              # build_backbone_with_proper_layer_resize()
    │   └── src/msiglip/model/siglip/           # mSigLIP model implementation
    ├── src/msiglip/model/lora.py               # get_lora_model() via PEFT
    ├── src/msiglip/model/tbps.py               # TBPS (forward pass, loss computation)
    │   └── src/msiglip/model/objectives.py     # Loss functions
    └── src/msiglip/solver/
        ├── build.py                # Optimizer with param groups
        └── lr_scheduler.py         # Cosine LR with warmup
```

## Backbone: mSigLIP

Multilingual SigLIP (`siglip-base-patch16-256-multilingual`) is a dual-encoder model pre-trained with sigmoid contrastive loss on image-text pairs.

### Vision Encoder (ViT-B/16)

```
Input Image (256 x 256 x 3)
    │
    ▼ Patch Embedding (patch_size=16)
    │ → 16 x 16 = 256 patches, each projected to 768d
    │ + Learnable position embeddings (native 256x256)
    │
    ▼ 12 Transformer Blocks
    │   ┌─────────────────────────────────┐
    │   │  Multi-Head Self-Attention      │ ← LoRA on q_proj, k_proj, v_proj, out_proj
    │   │  (12 heads, 64d per head)       │
    │   ├─────────────────────────────────┤
    │   │  LayerNorm + FFN (768→3072→768) │
    │   │  + LayerNorm                    │
    │   └─────────────────────────────────┘
    │
    ▼ Pooler (mean pooling → 768d)
    │
    ▼ image_embed (768d)
```

### Text Encoder (BERT-like)

```
Input Text (token_ids, max_length=64)
    │
    ▼ Token Embedding (vocab=250,000) + Position Embedding
    │
    ▼ 12 Transformer Blocks
    │   ┌─────────────────────────────────┐
    │   │  Multi-Head Self-Attention      │ ← LoRA on q_proj, k_proj, v_proj, out_proj
    │   │  (12 heads, 64d per head)       │
    │   │  attention_dropout = 0.05       │
    │   ├─────────────────────────────────┤
    │   │  LayerNorm + FFN (768→3072→768) │
    │   │  + LayerNorm                    │
    │   └─────────────────────────────────┘
    │
    ▼ Pooler (mean pooling → 768d)
    │
    ▼ text_embed (768d)
```

### Shared Parameters

- `logit_scale`: Learnable temperature (clamped to exp(scale) <= 100)
- `logit_bias`: Learnable bias for sigmoid similarity

## LoRA Adaptation

[PEFT](https://github.com/huggingface/peft) wraps the frozen backbone, adding low-rank adapters to all attention projections in both encoders.

```
Original:  x → W·x                    (768x768, frozen)
With LoRA: x → W·x + (B·A)·x         (768x32 + 32x768, trainable)
                      ↑
              scaling = alpha/r = 64/32 = 2.0
```

| Parameter | Value |
|---|---|
| Rank (r) | 32 |
| Alpha | 64 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `out_proj` |
| Dropout | 0.05 |
| Trainable params | ~3-5% of total |

## Loss Functions

### 1. N-ITC: Noise-Robust Image-Text Contrastive (weight = 1.0)

Primary alignment loss using SigLIP's sigmoid formulation.

```
sim_targets[i,j] = +1 if pid_i == pid_j, else -1  (sigmoid mode)

logit_i2t = scale * norm(img) @ norm(txt).T + bias
logit_t2i = logit_i2t.T

loss = -mean(logsigmoid(logit_t2i * sim_targets))
```

**MVS (Multi-View Similarity)**: An augmented view of each image is encoded separately. The N-ITC loss is computed for both original and augmented views, then averaged.

### 2. Cross-Modal Circle Loss with Curriculum Schedule (weight = 0.0 → 0.1)

Pairwise loss that adaptively re-weights positive and negative pairs based on their current similarity, mining hard negatives more aggressively than contrastive loss.

```
sim_mat = norm(img) @ norm(txt).T

For positive pairs (s_p):  logit_p = -gamma * alpha_p * (s_p - delta_p)
For negative pairs (s_n):  logit_n =  gamma * alpha_n * (s_n - delta_n)

where:
  alpha_p = clamp(-s_p + 1 + m, min=0)    # harder positives get higher weight
  alpha_n = clamp( s_n + m, min=0)         # harder negatives get higher weight
  delta_p = 1 - m,  delta_n = m            # margin boundaries
  m = 0.35, gamma = 128

loss = softplus(logsumexp(logit_p) + logsumexp(logit_n))
```

**Curriculum Schedule**: Circle loss weight ramps linearly to prevent disrupting early global alignment.

```
Epoch  0-5:   weight = 0.0        (warmup, circle off)
Epoch  6-20:  weight = linear ramp from 0.0 → 0.1
Epoch 21-60:  weight = 0.1        (stable)
```

### 3. C-ITC: Cyclic Image-Text Contrastive (weight = 0.1)

Regularization loss that enforces consistency between intra-modal and inter-modal similarity structures.

```
inmodal_loss  = mean((sim_i2i - sim_t2t)^2) / scale^2
intermodal_loss = mean((sim_i2t - sim_t2i)^2) / scale^2

loss = 0.25 * inmodal_loss + 0.25 * intermodal_loss
```

### 4. SimCLR: Self-Supervised Visual Consistency (weight = 0.4)

Contrastive learning between two augmented views of the same image, projected through a dedicated MLP head (`768 → 768 → 768` with ReLU).

```
embed_1 = simclr_mlp(encode_image(aug_view_1))
embed_2 = simclr_mlp(encode_image(aug_view_2))

sim_ab = norm(embed_1) @ norm(embed_2).T / temperature
loss = cross_entropy(sim_ab, identity_labels)
```

### Total Loss

```
total_loss = 1.0 * nitc_loss
           + curriculum_weight * circle_loss
           + 0.1 * citc_loss
           + 0.4 * simclr_loss
```

## Data Pipeline

```
Raw Sample (image_path, caption, pid)
    │
    ├─► Image Pipeline
    │     Resize(256x256) → ToTensor → ColorJitter → RandomRotation(15)
    │     → RandomResizedCrop(0.9-1.0) → RandomGrayscale(0.1)
    │     → RandomHorizontalFlip(0.5) → RandomErasing(0.10-0.20)
    │     → Normalize(0.5, 0.5)
    │     Output: images (3x256x256)
    │
    ├─► Augmented Image (for MVS)
    │     Same pipeline with different random seeds
    │     Output: aug_images (3x256x256)
    │
    ├─► SimCLR Views (for SS loss)
    │     Two independent augmented views
    │     Output: ss_images1, ss_images2 (3x256x256 each)
    │
    ├─► Text Pipeline
    │     Tokenize(max_len=64) → RandomDeletion(p=0.05)
    │     Output: caption_input_ids, caption_attention_mask
    │
    └─► Labels
          Output: pids (person identity)
```

## Optimizer & Scheduler

### AdamW with Parameter Groups

| Group | LR | Weight Decay | Matches |
|---|---|---|---|
| `lora` | 1e-4 | 0.0 | LoRA adapter weights |
| `backbone` | 1e-4 | 0.01 | Other backbone params (frozen, so empty) |
| `cross` | 1e-4 | 0.05 | Cross-attention modules |
| `simclr` | 1e-4 | 0.05 | SimCLR MLP head |
| `default` | 1e-4 | 0.01 | Everything else trainable |

### Cosine LR Scheduler with Linear Warmup

```
LR
 ▲
 │    ╱‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾╲
 │   ╱                      ╲
 │  ╱                        ╲
 │ ╱                          ╲
 │╱                            ╲
 └──────────────────────────────────► Epoch
 0    12                         60
   warmup        cosine decay
  (linear)      (1e-4 → 1e-5)
```

- Warmup: 20% of total epochs (12 epochs), linear from 1e-6 to 1e-4
- Decay: Cosine from 1e-4 to 1e-5
- Step-level scheduling (updated every optimizer step)

## Training Loop

```python
for epoch in range(60):
    for batch in train_loader:
        # Forward
        ret = model(batch, current_epoch=epoch)
        # ret contains: nitc_loss, circle_loss, citc_loss, ss_loss,
        #               circle_loss_weight, temperature, bias

        # Aggregate
        total_loss = sum(v for k, v in ret.items() if k.endswith("loss"))

        # Backward + step (handled by Lightning)
        # Gradient norm logged every step
```

## Evaluation

Ranking-based retrieval metrics computed on the full test set:

- **R@K** (K=1,5,10): Recall at K
- **mAP**: Mean Average Precision
- **mINP**: Mean Inverse Negative Penalty

Both **text-to-image (t2i)** and **image-to-text (i2t)** directions are evaluated. Primary metric: **t2i R@1**.

## File Map

| File | Role |
|---|---|
| `trainer.py` | Hydra entry point, training orchestration |
| `src/msiglip/lightning_models.py` | LitTBPS: training/val/test loops, metrics, LoRA setup |
| `src/msiglip/lightning_data.py` | TBPSDataModule: data loading, augmentation, samplers |
| `src/msiglip/model/tbps.py` | TBPS: forward pass, loss routing, curriculum schedule |
| `src/msiglip/model/objectives.py` | 4 loss functions: N-ITC, SimCLR, C-ITC, Circle Loss |
| `src/msiglip/model/build.py` | Backbone construction with position embedding interpolation |
| `src/msiglip/model/lora.py` | PEFT LoRA wrapper |
| `src/msiglip/model/siglip/` | mSigLIP model implementation (ViT + BERT) |
| `src/msiglip/solver/build.py` | Optimizer builder with parameter groups |
| `src/msiglip/solver/lr_scheduler.py` | Cosine LR with linear warmup |
| `configs/cir_msiglip.yaml` | Main Hydra config (composes all sub-configs) |
