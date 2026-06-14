# AGENTS.md

## Project Overview

**mSigLIP** - Multilingual Text-Based Person Search (TBPS) using mSigLIP, LoRA, Curriculum Circle Loss, optional Part-Token Alignment, and optional noise-robust MNEB-HN modules.

Workspace consists of two main parts:

1. **Training & Model Optimization** at the repository root.
2. **Edge Deployment & Compression** under `deployment/` for Qualcomm RB3 Gen2.

Core task:

- Align person images and multilingual text descriptions in a shared 768-dim embedding space.
- Primary metric: text-to-image Rank@1.
- Backbone: `siglip-base-patch16-256-multilingual`.
- Preferred adapter family: LoRA on attention + FFN targets, currently `+lora=attn_ffn_r32`.
- Target languages/datasets:
  - VN3K / VnPersonSearch: Vietnamese, treated as clean.
  - CUHK-PEDES: English, main natural-noise benchmark.
  - PRW-TPS-CN: Chinese, multilingual generalization benchmark.
- Target deployment device: Qualcomm RB3 Gen2 / QCS6490, 4GB RAM, ARM64.

Current result state:

| Result | Status |
|---|---|
| `52.28` VN3K T2I R@1 | Historical/paper Circle + LoRA baseline, seed 2400 |
| `52.83` VN3K T2I R@1 | Attn+FFN LoRA r32, batch64, accum2 |
| `53.00` VN3K T2I R@1 / `53.25` I2T R@1 | Current best experimental VN3K result: Part Align + Attn+FFN LoRA r32, batch64, epoch 50 |
| PiSSA r32 | Rejected for current VN3K setup; peaked around `47.58` in the observed run |
| MNEB-HN | Implemented, experimental, disabled by default, target is CUHK-PEDES natural noise |

## Module Hierarchy

### Training & Model Optimization

```text
trainer.py                                      # Hydra entry point
test.py                                         # Evaluation entry point
src/msiglip/lightning_data.py                   # TBPSDataModule
src/msiglip/lightning_models.py                 # LitTBPS
src/msiglip/data/
  vn3k_vi.py                                    # VN3K Vietnamese
  vn3k_en.py                                    # VN3K English
  vn3k_mixed.py                                 # VN3K mixed-language
  cuhkpedes.py                                  # CUHK-PEDES
  cuhk_10_percent_vn3k_mix.py                   # 10% CUHK + VN3K mix
  prw_tps_cn.py                                 # PRW-TPS-CN Chinese
  bases.py                                      # ImageTextDataset, ImageDataset, TextDataset
  sampler.py                                    # RandomIdentitySampler
src/msiglip/model/
  build.py                                      # Backbone construction
  lora.py                                       # PEFT LoRA setup
  tbps.py                                       # Forward pass and loss routing
  objectives.py                                 # N-ITC, Circle, C-ITC, SimCLR, Part Align, MNEB aux
  evidence_bank.py                              # MNEB-HN EvidenceMemoryBank
  reid_objectives.py                            # ReID-specific objectives
src/msiglip/solver/
  build.py                                      # Optimizer param groups
  lr_scheduler.py                               # Cosine LR with warmup
configs/
  cir_msiglip.yaml                              # Main Hydra config
  loss/cir_msiglip.yaml                         # Loss flags and MNEB/NACIR config
  lora/*.yaml                                   # LoRA variants
  dataset/*.yaml                                # VN3K, CUHK, PRW dataset configs
notebooks/
  workspace.ipynb                               # Local research/loss playground
  colab_training_experiments.ipynb              # Colab training workflow
scripts/colab/package_training_code.sh          # Colab code packaging
```

### Edge Deployment & Compression

```text
deployment/
  scripts/                                      # Export, ONNX, QNN, diagnostics
  config/qnn/                                   # QNN/HTP runtime config
  docs/                                         # Deployment docs and journals
  hardware_profiling/                           # RB3 profiling helpers
  deploy_utils.py                               # Shared deployment utilities
artifacts/deployment/                           # Generated deployment outputs and logs
```

Do not put generated deployment outputs outside `artifacts/deployment/` unless the user explicitly asks.

## Loss Functions

Default main objective:

```text
1.0 * N-ITC/MVS + curriculum * Circle + 0.1 * C-ITC + 0.4 * SimCLR
```

Important details:

- N-ITC is the primary sigmoid image-text alignment loss.
- MVS is part of the N-ITC path and must not be accidentally dropped from descriptions of the baseline.
- Circle Loss is the hard-negative core with `m=0.25`, `gamma=128`.
- Circle curriculum: epoch 0-5 off, epoch 6-20 linear ramp to `0.1`, epoch 21-60 stable at `0.1`.
- C-ITC weight is `0.1`.
- SimCLR/SS weight is `0.4`.
- Optional `PART_ALIGN` adds part-token local alignment, currently useful with `attn_ffn_r32`.

All training losses live in `src/msiglip/model/objectives.py`; routing and curriculum logic live in `src/msiglip/model/tbps.py`; flags live in `configs/loss/cir_msiglip.yaml`.

## Optional Noise Framework: MNEB-HN

MNEB-HN means **Multilingual Noise Evidence Bank for Hard-Negative TBPS**.

Status:

- Implemented as an optional module.
- Disabled by default: `loss.MNEB=false`.
- MNEB and NACIR are mutually exclusive.
- With MNEB disabled, the baseline path must remain unchanged.
- With MNEB enabled and aux losses disabled, it collects/logs evidence only and must not affect total loss.
- MNEB never mutates Circle Loss `alpha_n` or `alpha_p`.
- FN/FP corrections enter only through `fnm_aux_loss` and `rde_aux_loss`.

Core modules:

- `src/msiglip/model/evidence_bank.py`
  - `EvidenceMemoryBank`
  - global/local embeddings
  - loss EMA
  - clean probabilities
  - seen counts
  - FIFO sample IDs
  - FN stats
  - global/local consensus labels
- `src/msiglip/model/objectives.py`
  - `compute_part_token_score_matrix`
  - `compute_part_token_alignment`
  - `compute_branch_per_sample_contrastive_loss`
  - `compute_fnm_auxiliary_loss`
  - `compute_rde_auxiliary_loss`
- `src/msiglip/lightning_models.py`
  - epoch-end evidence-bank refit and `mneb_*` diagnostics.

Recommended interpretation:

- VN3K is clean, so MNEB should mostly no-op there. A high `mneb_consensus_uncertain_frac` on VN3K can be a good clean-safe signal.
- CUHK-PEDES is the main target for natural FN/FP noise robustness.
- Evidence-only runs are diagnostics, not a new training objective.

Common scripts:

```bash
bash run_mneb_hn.sh
bash run_mneb_hn.sh loss.mneb_config.fnm_aux.enabled=false loss.mneb_config.rde_aux.enabled=false
```

## NACIR Status

NACIR remains in the repo for legacy experiments and ablations, but it is not the current main direction.

Important conclusions:

- NACIR-lite failed as a main clean VN3K framework because the FN branch can mistake true hard negatives for false negatives and weaken Circle Loss.
- NACIR-FP used a single GMM over per-sample Circle loss and is weaker than the current MNEB/RDE-style consensus design.
- Do not present NACIR as the recommended framework unless the user explicitly asks for NACIR-specific work.
- Existing NACIR scripts such as `run_nacir.sh`, `run_nacir_fp_only.sh`, and `run_nacir_detector_off.sh` are legacy/ablation paths.

## LoRA Variants

Configured variants live under `configs/lora/`:

- `default.yaml`: original LoRA target set.
- `attn_ffn_r16.yaml`: attention + FFN, rank 16.
- `attn_ffn_r32.yaml`: attention + FFN, rank 32; current preferred adapter config.
- `attn_ffn_r64.yaml`: attention + FFN, rank 64; candidate capacity ablation, more memory.
- `attn_ffn_r32_pissa.yaml`: PiSSA initialization; rejected for current VN3K setup.
- `attn_ffn_r32_dora.yaml`: DoRA; can cost more VRAM/compute.
- `attn_ffn_r32_rslora.yaml`: rsLoRA; candidate ablation.

Preferred clean VN3K recipe right now:

```bash
bash run_part_align_lora_attn_ffn_r32.sh
```

MNEB recipe builds on that recipe:

```bash
bash run_mneb_hn.sh
```

## Configuration System

Main Hydra config: `configs/cir_msiglip.yaml`.

Important config groups:

- `configs/loss/cir_msiglip.yaml`: N-ITC/MVS, Circle, C-ITC, SimCLR, Part Align, MNEB, NACIR.
- `configs/backbone/m_siglip.yaml`: mSigLIP backbone.
- `configs/trainer/best_msiglip.yaml`: Lightning trainer defaults.
- `configs/optimizer/cir_test.yaml`: AdamW param groups.
- `configs/scheduler/tbps_clip.yaml`: cosine LR with warmup.
- `configs/lora/*.yaml`: LoRA variants.
- `configs/dataset/vn3k_vi.yaml`
- `configs/dataset/vn3k_en.yaml`
- `configs/dataset/vn3k_mixed.yaml`
- `configs/dataset/cuhk_pedes.yaml`
- `configs/dataset/cuhk_pedes_10_percent.yaml`
- `configs/dataset/prw_tps_cn.yaml`
- `configs/tokenizer/m_siglip.yaml`
- `configs/logger/default.yaml`
- `configs/aug/siglip.yaml`

## Experiment Workflow

Training is expensive. Prefer this sequence:

1. Inspect the current journal and relevant config/script.
2. Validate loss/architecture ideas in `notebooks/workspace.ipynb` when feasible.
3. Run focused unit tests.
4. Launch full training only when the change has a clear reason.
5. Record results in the dated training journal when the user asks for documentation.

For Colab:

- Use `notebooks/colab_training_experiments.ipynb`.
- Package the code with `scripts/colab/package_training_code.sh`.
- The package includes `src/`, `configs/`, `scripts/`, `tests/`, `run_*.sh`, and the Colab notebook.
- External Colab assets are expected separately:
  - `VN3K/`
  - `CUHK-PEDES/`
  - `PRW-TPS-CN/`
  - `m_siglip_checkpoints/model.safetensors`

## Tests

Useful focused checks:

```bash
venv/bin/python -m unittest discover -s tests -p 'test_lora_configs.py' -v
venv/bin/python -m unittest discover -s tests -p 'test_part_alignment_loss.py' -v
venv/bin/python -m unittest discover -s tests -p 'test_evidence_bank.py' -v
venv/bin/python -m unittest discover -s tests -p 'test_mneb_objectives.py' -v
venv/bin/python -m unittest discover -s tests -p 'test_mneb_integration.py' -v
venv/bin/python -m unittest discover -s tests -p 'test_noise_injection.py' -v
venv/bin/python -m compileall src/msiglip/model/objectives.py src/msiglip/model/tbps.py src/msiglip/model/evidence_bank.py src/msiglip/lightning_models.py
git diff --check
```

On Colab, prefer `unittest discover -s tests -p ...` instead of module-style paths like `python -m unittest tests/test_evidence_bank.py`, because an installed `tests` package can shadow the local folder.

## Key Documents

- `README.md`: public-facing project overview, current status snapshot, MNEB summary.
- `docs/ARCHITECTURE.md`: architecture reference.
- `docs/EXPERIMENT_SUMMARY.md`: canonical older experiment summary.
- `docs/noise-robust-multilingual-framework.md`: MNEB-HN research and implementation design.
- `docs/knowledge.md`: Vietnamese durable knowledge base for concepts only. This file may be gitignored locally, but it is still the local durable concept note.
- `docs/journal/`: dated training/model-optimization journals.
- `docs/journal/[train]-2026-06-11.md`: NACIR failure analysis and attn+FFN r32 result.
- `docs/journal/[train]-2026-06-13.md`: PiSSA rejection and Part Align `53.00` result.
- `knowledge/`: paper notes, reviewer responses, research notes.
- `changelog/training/changelog.md`: completed training/code/config/docs changes.
- `reports/architecture-decisions.md`: significant architecture decisions.
- `deployment/README.md`: deployment overview.
- `deployment/docs/journal/`: dated deployment logs.

Do not edit `src/person_rlf.egg-info/PKG-INFO` unless packaging metadata regeneration is explicitly requested; it may contain stale README snapshots.

## Documentation Policy

Do **not** automatically write to documentation files. Before editing docs, journal, changelog, README, or paper notes, state the target file(s), what will be recorded, and why, then ask the user to confirm. If the user explicitly asks to create or update documentation in the prompt, that request counts as confirmation for the requested files.

Classify documentation before writing:

- **Knowledge** (`docs/knowledge.md`): durable concepts, definitions, mechanisms, and general trade-offs that should still be true months later.
- **Training journal** (`docs/journal/[train]-YYYY-MM-DD.md`): training/model-optimization results, commands, logs, metrics, temporary conclusions, and next experiment decisions.
- **Deployment journal** (`deployment/docs/journal/[deploy]-YYYY-MM-DD.md`): deploy results, AI Hub jobs, QNN/QDQ fidelity, board runtime, artifacts, and next deploy steps.
- **Demo system journal** (`deployment/docs/journal/[demo-system]-YYYY-MM-DD.md`): modular demo-system work under `deployment/demo/`.
- **Changelog** (`changelog/{component}/changelog.md`): completed code/config/docs changes after user-confirmed changelog writing.
- **Paper notes** (`knowledge/response.md`, `knowledge/paper/`, etc.): reviewer responses, paper wording, and presentation-specific phrasing.

Do not put run logs, experiment results, dated progress, reviewer-answer wording, changelog entries, or deployment job results into `docs/knowledge.md`.

When the user asks for a concept explanation such as "PiSSA là gì" or "Part-Align là gì", answer from the durable knowledge/journal context. If the user asks to "ghi vào", update the appropriate `docs/knowledge.md` concept entry and, if it is run-specific, also the dated journal.

## Changelog

After completing a task that modifies code/config/docs, ask before appending an entry to the relevant `changelog/{component}/changelog.md`, unless the user already requested changelog updates.

Components:

- `changelog/training/changelog.md`: training pipeline, model, losses, data, config, notebooks.
- `changelog/deployment/changelog.md`: edge deployment, ONNX, QNN/SNPE, hardware.

Entry format:

```markdown
## [YYYY-MM-DD] Short description

| # | Priority | Type | Action | Status |
|---|----------|------|--------|--------|
| 1 | HIGH/MED/LOW | feature/bugfix/refactor/config/docs | What was done | COMPLETE (reason) |
```

## Architecture Decisions

When making significant architectural decisions, document them in `reports/architecture-decisions.md` with:

- Decision
- Reason
- Alternatives considered

## Coding Conventions

- PyTorch Lightning for training loop.
- Hydra for config management.
- W&B for experiment tracking.
- `ruff` for linting/formatting where applicable.
- Use `rg`/`rg --files` for search.
- Use `apply_patch` for manual file edits.
- Keep changes modular and disabled by default unless the user explicitly asks to change defaults.
- New loss integration pattern:
  1. Add function in `src/msiglip/model/objectives.py`.
  2. Route it in `src/msiglip/model/tbps.py`.
  3. Add flags/defaults in `configs/loss/cir_msiglip.yaml`.
  4. Add focused tests.
- Preserve current baseline behavior when optional modules are disabled.
