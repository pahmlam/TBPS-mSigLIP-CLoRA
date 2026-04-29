# AGENTS.md

## Project Overview

**mSigLIP** — Multilingual Text-Based Person Search (TBPS) using Cross-Modal Circle Loss with Curriculum Learning and LoRA.

Workspace consists of 2 parts:
1. **Training & Model Optimization** (root) — Train and optimize model performance
2. **Edge Deployment & Compression** (`deployment/`) — Compress model and deploy on edge devices (Qualcomm RB3 Gen2)

- **Task**: Align person images and Vietnamese text descriptions in a shared 768-dim embedding space
- **Backbone**: `siglip-base-patch16-256-multilingual` with LoRA (r=32, alpha=64, ~3-5% trainable params)
- **Benchmark**: VnPersonSearch (VN3K) — current best **R@1 = 52.28%** (LoRA + Curriculum Circle Loss, seed 2400)
- **Primary metric**: text-to-image R@1
- **Target device**: Qualcomm RB3 Gen2 (QCS6490, 4GB RAM, ARM64)

## Module Hierarchy

### Part 1: Training & Model Optimization (root)

```
trainer.py                          # Entry point (Hydra)
├── lightning_data.py               # TBPSDataModule (data loading, augmentation)
│   ├── data/vn3k_vi.py             # VN3K Vietnamese dataset
│   ├── data/vn3k_en.py             # VN3K English dataset
│   ├── data/vn3k_mixed.py          # VN3K mixed-language dataset
│   ├── data/cuhkpedes.py           # CUHK-PEDES dataset
│   ├── data/prw_tps_cn.py          # PRW-TPS-CN (Chinese) dataset
│   ├── data/cuhk_10_percent_vn3k_mix.py  # 10% CUHK + VN3K mix
│   ├── data/bases.py               # ImageTextDataset, ImageDataset, TextDataset
│   ├── data/sampler.py             # RandomIdentitySampler
│   └── data/augmentation/          # Image & text augmentation pools
│
├── lightning_models.py             # LitTBPS (PyTorch Lightning module)
│   ├── model/build.py              # build_backbone_with_proper_layer_resize()
│   │   └── model/siglip/           # mSigLIP model implementation
│   ├── model/lora.py               # get_lora_model() via PEFT
│   ├── model/tbps.py               # TBPS (forward pass, loss computation)
│   │   ├── model/objectives.py     # Loss functions (N-ITC, Circle, C-ITC, SimCLR)
│   │   └── model/reid_objectives.py # ReID-specific objectives
│   └── solver/
│       ├── build.py                # Optimizer with param groups
│       └── lr_scheduler.py         # Cosine LR with warmup
│
├── test.py                         # Evaluation script
├── workspace.ipynb                 # Experiment notebook (analysis, loss playground)
├── utils/                          # Metrics, visualization, tokenizer utils
├── scripts/                        # Helper scripts (checkpoint prep, extraction)
├── experiments/                    # Experiment logs & ablation notes
└── knowledge/                      # Research notes & paper drafts
```

### Part 2: Edge Deployment & Compression (deployment/)

```
deployment/
├── scripts/                            # mSigLIP deployment pipeline
│   ├── analyze_checkpoint.py           # Shared: Analyze checkpoint (size, RAM, compat)
│   ├── inference_test.py              # Shared: Test inference on target device
│   ├── lora_fp16/                      # Step 1: LoRA merge + FP16 export
│   │   └── export.py                   #   Merge LoRA → FP16/FP32 state dict
│   └── onnx/                           # Step 2: ONNX conversion
│       └── export.py                   #   FP16/FP32 state dict → ONNX
├── hardware_profiling/                 # RB3 hardware capability testing (proxy models)
│   ├── benchmark.py                    # PyTorch CPU vs ONNX Runtime
│   ├── snpe_benchmark.py              # Qualcomm SNPE (DSP/HTP)
│   ├── collect_sysinfo.sh             # Collect system info
│   ├── install_deps.sh                # Install dependencies
│   └── run_all.sh                     # Master script
├── docs/                               # Deployment documentation
│   ├── system.md                       # RB3 hardware specifications
│   ├── experiment.md                   # Benchmark guide
│   └── benchmark-rp.md                # Benchmark results
├── logs/                               # Auto-generated logs (timestamped)
└── deploy_utils.py                     # Shared utilities (TeeLogger)
```

## Loss Functions

Total loss = `1.0*N-ITC + curriculum*Circle + 0.1*C-ITC + 0.4*SimCLR`

| Loss | Weight | Role |
|------|--------|------|
| N-ITC | 1.0 | Primary alignment (sigmoid contrastive, +MVS augmentation) |
| Circle Loss | 0→0.1 (curriculum) | Hard-negative mining (m=0.25, gamma=128) |
| C-ITC | 0.1 | Cyclic consistency regularization |
| SimCLR | 0.4 | Self-supervised visual consistency |

**Curriculum schedule**: epoch 0-5 weight=0, epoch 6-20 linear ramp to 0.1, epoch 21-60 stable at 0.1.

All loss functions live in `model/objectives.py`. Loss routing and curriculum logic in `model/tbps.py`. Config flags in `config/loss/cir_msiglip.yaml`.

## Configuration System (Hydra)

Main config: `config/cir_msiglip.yaml` composes sub-configs:
- `config/loss/cir_msiglip.yaml` — loss flags and weights
- `config/backbone/m_siglip.yaml` — backbone settings
- `config/trainer/best_msiglip.yaml` — training hyperparams (60 epochs, bf16-mixed)
- `config/optimizer/cir_test.yaml` — AdamW with param groups
- `config/scheduler/tbps_clip.yaml` — cosine LR + warmup
- `config/lora/default.yaml` — LoRA config
- `config/dataset/vn3k_vi.yaml` — dataset paths (also: vn3k_en, vn3k_mixed, cuhk_pedes, cuhk_pedes_10_percent, prw_tps_cn)
- `config/tokenizer/m_siglip.yaml` — tokenizer settings
- `config/logger/default.yaml` — W&B logger config
- `config/aug/siglip.yaml` — augmentation settings

## Critical Workflow Rule

**Training costs hours. Always validate ideas in `workspace.ipynb` first.**

Research cycle:
1. **Ideate** — propose loss/architecture change
2. **Implement** — modify code (objectives.py, tbps.py, config)
3. **Validate** — test in workspace.ipynb on frozen embeddings (seconds, not hours)
4. **Train** — only when good signs are confirmed (run_cir_loss.sh)
5. **Analyze** — compare results against `EXPERIMENT_SUMMARY.md`

## workspace.ipynb Conventions

The notebook operates on `W` — a dict of extracted embeddings from a checkpoint:
- `W['image_feats']`, `W['text_feats']` — L2-normalized embeddings (N × 768)
- `W['image_pids']`, `W['text_pids']` — person ID labels
- `W['logit_scale']`, `W['logit_bias']` — learned parameters

**Sections**: 0=Setup, 1=Load & Extract, 2=Similarity Analysis, 3=Loss Playground, 4=Gradient Analysis (most important), 5=Embedding Visualization, 6=Retrieval Metrics, 7=A/B Comparison, 8=Mini Fine-Tune

**Good signs before training**:
- Gradient energy on top-10% hard negatives > N-ITC baseline (Section 4)
- Clear pos/neg separation in similarity histogram (Section 2)
- Loss value finite, similar scale to baselines (Section 3)
- t-SNE clusters tight, >90% points above y=x in scatter (Section 5)
- R@1 stable or improved after mini fine-tune (Section 8)

## Key Documents

- `docs/ARCHITECTURE.md` — Full architecture with diagrams
- `docs/EXPERIMENT_SUMMARY.md` — Results table and training config (canonical record)
- `docs/knowledge.md` — Knowledge base in Vietnamese (see rule below)
- `knowledge/` — Research notes, paper drafts, noise handling analysis
- `experiments/` — Experiment logs and ablation notes
- `ref/rde/` — RDE (CVPR 2024) reference implementation for noise-robust learning
- `deployment/README.md` — Edge deployment overview
- `deployment/docs/deployment-plan.md` — **Current deployment state, pipeline status, next steps** (start here for deployment work)
- `deployment/docs/aihub-experiments.md` — Running log of every Qualcomm AI Hub compile attempt (rule: append a row on every `qai-hub` invocation — see `.claude/rules/aihub-experiments.md`)
- `deployment/docs/system.md` — Qualcomm RB3 Gen2 hardware specifications

## Knowledge Documentation Rule

**Required:** When discussing or implementing any technical/theoretical content task, Claude MUST document related knowledge in `docs/knowledge.md` in Vietnamese.

Each knowledge entry must include:
1. **Definition** — Explain related concepts/terminology
2. **Why (WHY)** — Why this needs to be done
3. **What (WHAT)** — Describe the specific solution/action
4. **How (HOW)** — Technical details, code, commands
5. **Reasoning & approach** — Trade-offs, rationale for the chosen approach

Template is available at the bottom of `docs/knowledge.md`. Update the table of contents when adding new entries.

## Changelog

After completing a task that modifies code, append an entry to the relevant `changelog/{component}/changelog.md`.

Components:
- `changelog/training/changelog.md` — training pipeline, model, losses, data, config
- `changelog/deployment/changelog.md` — edge deployment, ONNX, SNPE, hardware

Entry format:

## [YYYY-MM-DD] Short description

| # | Priority | Type | Action | Status |
|---|----------|------|--------|--------|
| 1 | HIGH/MED/LOW | feature/bugfix/refactor/config | What was done | COMPLETE (reason) |

## Architecture Decisions

When making significant architectural decisions (new dependencies, pattern changes, infrastructure choices), document them in `reports/architecture-decisions.md` with: Decision, Reason, Alternatives considered.

## Coding Conventions

- PyTorch Lightning for training loop
- Hydra for config management
- W&B for experiment tracking
- `ruff` for linting and formatting
- All losses in `model/objectives.py` take L2-normalized features and return scalar tensor
- New loss integration: add to `objectives.py` → add routing in `tbps.py` forward() → add config flag in `config/loss/`
