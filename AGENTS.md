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
├── src/msiglip/lightning_data.py               # TBPSDataModule (data loading, augmentation)
│   ├── src/msiglip/data/vn3k_vi.py             # VN3K Vietnamese dataset
│   ├── src/msiglip/data/vn3k_en.py             # VN3K English dataset
│   ├── src/msiglip/data/vn3k_mixed.py          # VN3K mixed-language dataset
│   ├── src/msiglip/data/cuhkpedes.py           # CUHK-PEDES dataset
│   ├── src/msiglip/data/prw_tps_cn.py          # PRW-TPS-CN (Chinese) dataset
│   ├── src/msiglip/data/cuhk_10_percent_vn3k_mix.py  # 10% CUHK + VN3K mix
│   ├── src/msiglip/data/bases.py               # ImageTextDataset, ImageDataset, TextDataset
│   ├── src/msiglip/data/sampler.py             # RandomIdentitySampler
│   └── src/msiglip/data/augmentation/          # Image & text augmentation pools
│
├── src/msiglip/lightning_models.py             # LitTBPS (PyTorch Lightning module)
│   ├── src/msiglip/model/build.py              # build_backbone_with_proper_layer_resize()
│   │   └── src/msiglip/model/siglip/           # mSigLIP model implementation
│   ├── src/msiglip/model/lora.py               # get_lora_model() via PEFT
│   ├── src/msiglip/model/tbps.py               # TBPS (forward pass, loss computation)
│   │   ├── src/msiglip/model/objectives.py     # Loss functions (N-ITC, Circle, C-ITC, SimCLR)
│   │   └── src/msiglip/model/reid_objectives.py # ReID-specific objectives
│   └── src/msiglip/solver/
│       ├── build.py                # Optimizer with param groups
│       └── lr_scheduler.py         # Cosine LR with warmup
│
├── test.py                         # Evaluation script
├── notebooks/workspace.ipynb                 # Experiment notebook (analysis, loss playground)
├── src/msiglip/utils/                          # Metrics, visualization, tokenizer utils
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
├── config/qnn/                         # QNN/HTP runtime config JSON files
└── deploy_utils.py                     # Shared utilities (TeeLogger)
```

Generated deployment outputs and logs belong under `artifacts/deployment/`.

## Loss Functions

Total loss = `1.0*N-ITC + curriculum*Circle + 0.1*C-ITC + 0.4*SimCLR`

| Loss | Weight | Role |
|------|--------|------|
| N-ITC | 1.0 | Primary alignment (sigmoid contrastive, +MVS augmentation) |
| Circle Loss | 0→0.1 (curriculum) | Hard-negative mining (m=0.25, gamma=128) |
| C-ITC | 0.1 | Cyclic consistency regularization |
| SimCLR | 0.4 | Self-supervised visual consistency |

**Curriculum schedule**: epoch 0-5 weight=0, epoch 6-20 linear ramp to 0.1, epoch 21-60 stable at 0.1.

All loss functions live in `src/msiglip/model/objectives.py`. Loss routing and curriculum logic in `src/msiglip/model/tbps.py`. Config flags in `configs/loss/cir_msiglip.yaml`.

## Configuration System (Hydra)

Main config: `configs/cir_msiglip.yaml` composes sub-configs:
- `configs/loss/cir_msiglip.yaml` — loss flags and weights
- `configs/backbone/m_siglip.yaml` — backbone settings
- `configs/trainer/best_msiglip.yaml` — training hyperparams (60 epochs, bf16-mixed)
- `configs/optimizer/cir_test.yaml` — AdamW with param groups
- `configs/scheduler/tbps_clip.yaml` — cosine LR + warmup
- `configs/lora/default.yaml` — LoRA config
- `configs/dataset/vn3k_vi.yaml` — dataset paths (also: vn3k_en, vn3k_mixed, cuhk_pedes, cuhk_pedes_10_percent, prw_tps_cn)
- `configs/tokenizer/m_siglip.yaml` — tokenizer settings
- `configs/logger/default.yaml` — W&B logger config
- `configs/aug/siglip.yaml` — augmentation settings

## Critical Workflow Rule

**Training costs hours. Always validate ideas in `notebooks/workspace.ipynb` first.**

Research cycle:
1. **Ideate** — propose loss/architecture change
2. **Implement** — modify code (objectives.py, tbps.py, configs)
3. **Validate** — test in notebooks/workspace.ipynb on frozen embeddings (seconds, not hours)
4. **Train** — only when good signs are confirmed (run_cir_loss.sh)
5. **Analyze** — compare results against `EXPERIMENT_SUMMARY.md`

## notebooks/workspace.ipynb Conventions

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
- `docs/knowledge.md` — Vietnamese durable knowledge base for concepts/definitions only
- `docs/journal/` — Dated training/model-optimization research logs (`[train]-YYYY-MM-DD.md`)
- `knowledge/` — Research notes, paper drafts, noise handling analysis
- `experiments/` — Experiment logs and ablation notes
- `ref/rde/` — RDE (CVPR 2024) reference implementation for noise-robust learning
- `deployment/README.md` — Edge deployment overview
- `deployment/docs/deployment-plan.md` — **Current deployment state, pipeline status, next steps** (start here for deployment work)
- `deployment/docs/journal/` — Dated deployment logs (`[deploy]-YYYY-MM-DD.md`)
- `deployment/docs/aihub-experiments.md` — Running log of every Qualcomm AI Hub compile attempt (rule: append a row on every `qai-hub` invocation — see `.claude/rules/aihub-experiments.md`)
- `deployment/docs/system.md` — Qualcomm RB3 Gen2 hardware specifications

## Documentation Policy

Do **not** automatically write to documentation files. Before editing any docs, journal, changelog, README, or paper notes, state the target file(s), what will be recorded, and why, then ask the user to confirm. If the user explicitly asks to create or update documentation in the prompt, that request counts as confirmation for the requested files.

Classify documentation before writing:
- **Knowledge** (`docs/knowledge.md`): durable concepts, definitions, mechanisms, and general trade-offs that should still be true months later.
- **Training journal** (`docs/journal/[train]-YYYY-MM-DD.md`): training/model-optimization results, commands, logs, metrics, temporary conclusions, and next experiment decisions.
- **Deployment journal** (`deployment/docs/journal/[deploy]-YYYY-MM-DD.md`): deploy results, AI Hub jobs, QNN/QDQ fidelity, board runtime, artifacts, and next deploy steps.
- **Changelog** (`changelog/{component}/changelog.md`): completed code/config/docs changes after user-confirmed changelog writing.
- **Paper notes** (`knowledge/response.md`, `knowledge/paper/`, etc.): reviewer responses, paper wording, and presentation-specific phrasing.

Do not put run logs, experiment results, dated progress, reviewer-answer wording, changelog entries, or deployment job results into `docs/knowledge.md`.

Use the templates in `docs/knowledge.md`, `docs/journal/README.md`, and `deployment/docs/journal/README.md` when the user confirms a documentation update.

## Changelog

After completing a task that modifies code/config/docs, ask before appending an entry to the relevant `changelog/{component}/changelog.md`, unless the user already requested changelog updates.

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
- All losses in `src/msiglip/model/objectives.py` take L2-normalized features and return scalar tensor
- New loss integration: add to `src/msiglip/model/objectives.py` → add routing in `src/msiglip/model/tbps.py` forward() → add config flag in `configs/loss/`
