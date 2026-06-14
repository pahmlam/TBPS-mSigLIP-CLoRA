# CLAUDE.md

## Project Snapshot

**mSigLIP** - multilingual Text-Based Person Search with mSigLIP, LoRA,
Curriculum Circle Loss, optional Part-Token Alignment, and optional MNEB-HN noise
modules.

Workspace:

- Root: training and model optimization.
- `deployment/`: edge export/compression/runtime work for Qualcomm RB3 Gen2.

Core task:

- Align person images and multilingual text in a shared 768-dim embedding space.
- Primary metric: text-to-image Rank@1.
- Backbone: `siglip-base-patch16-256-multilingual`.
- Main datasets: VN3K Vietnamese clean, CUHK-PEDES English natural noise,
  PRW-TPS-CN Chinese multilingual sanity.

Current result state:

| Result | Status |
|---|---|
| `52.28` VN3K T2I R@1 | Historical/paper Circle + LoRA baseline, seed 2400 |
| `52.83` VN3K T2I R@1 | Attn+FFN LoRA r32, batch64, accum2 |
| `53.00` T2I / `53.25` I2T | Current best experimental VN3K: Part Align + Attn+FFN LoRA r32 |
| PiSSA r32 | Rejected for current VN3K setup, observed best around `47.58` |
| MNEB-HN | Implemented, experimental, disabled by default, target is CUHK-PEDES |

## Main Files

Training:

- `trainer.py`: Hydra training entry point.
- `test.py`: evaluation entry point.
- `src/msiglip/lightning_data.py`: data module.
- `src/msiglip/lightning_models.py`: Lightning module.
- `src/msiglip/model/tbps.py`: forward pass and loss routing.
- `src/msiglip/model/objectives.py`: N-ITC, Circle, C-ITC, SimCLR, Part Align,
  MNEB aux losses.
- `src/msiglip/model/evidence_bank.py`: MNEB-HN memory bank.
- `src/msiglip/model/lora.py`: PEFT LoRA setup.

Data:

- `src/msiglip/data/vn3k_vi.py`, `vn3k_en.py`, `vn3k_mixed.py`.
- `src/msiglip/data/cuhkpedes.py`.
- `src/msiglip/data/cuhk_10_percent_vn3k_mix.py`.
- `src/msiglip/data/prw_tps_cn.py`.

Config:

- Main config: `configs/cir_msiglip.yaml`.
- Loss config: `configs/loss/cir_msiglip.yaml`.
- Datasets: `configs/dataset/*.yaml`.
- LoRA variants: `configs/lora/*.yaml`.

Deployment outputs and logs belong under `artifacts/deployment/`.

## Loss Design

Default main objective:

```text
1.0 * N-ITC/MVS + curriculum * Circle + 0.1 * C-ITC + 0.4 * SimCLR
```

Important:

- MVS is part of the N-ITC path; do not drop it from baseline descriptions.
- Circle Loss is the hard-negative core: `m=0.25`, `gamma=128`.
- Circle curriculum: epoch 0-5 off, 6-20 ramp to `0.1`, 21-60 stable.
- `PART_ALIGN` is optional and currently useful with `+lora=attn_ffn_r32`.
- Preserve baseline behavior when optional modules are disabled.

## MNEB-HN

MNEB-HN = **Multilingual Noise Evidence Bank for Hard-Negative TBPS**.

Status and rules:

- Disabled by default: `loss.MNEB=false`.
- MNEB and NACIR are mutually exclusive.
- With MNEB disabled, no evidence bank or extra hidden-state request is created.
- With MNEB enabled and aux losses disabled, it is evidence/diagnostics only.
- MNEB never mutates Circle `alpha_n` or `alpha_p`.
- FN/FP correction enters only through `fnm_aux_loss` and `rde_aux_loss`.
- VN3K is clean; high `mneb_consensus_uncertain_frac` there can be clean-safe.
- CUHK-PEDES is the main natural-noise validation target.

Run:

```bash
bash run_mneb_hn.sh
bash run_mneb_hn.sh loss.mneb_config.fnm_aux.enabled=false loss.mneb_config.rde_aux.enabled=false
```

## NACIR Status

NACIR remains for legacy experiments and ablations, not the main direction.

- NACIR-lite failed as a clean VN3K framework because FN detection can suppress
  true hard negatives and weaken Circle Loss.
- NACIR-FP's single-GMM soft weighting is weaker than MNEB/RDE-style consensus.
- Use NACIR scripts only when the user explicitly asks for NACIR-specific work.

## LoRA Recipes

Configured variants:

- `default.yaml`: original LoRA target set.
- `attn_ffn_r16.yaml`.
- `attn_ffn_r32.yaml`: preferred current adapter config.
- `attn_ffn_r64.yaml`: capacity ablation, more memory.
- `attn_ffn_r32_pissa.yaml`: rejected for current VN3K setup.
- `attn_ffn_r32_dora.yaml`: more VRAM/compute.
- `attn_ffn_r32_rslora.yaml`: candidate ablation.

Preferred clean VN3K:

```bash
bash run_part_align_lora_attn_ffn_r32.sh
```

MNEB builds on that recipe:

```bash
bash run_mneb_hn.sh
```

## Colab

- Notebook: `notebooks/colab_training_experiments.ipynb`.
- Package script: `scripts/colab/package_training_code.sh`.
- Package includes `src/`, `configs/`, `scripts/`, `tests/`, `run_*.sh`, notebook.
- External assets expected on Colab:
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
venv/bin/python -m compileall src/msiglip/model/objectives.py src/msiglip/model/tbps.py src/msiglip/model/evidence_bank.py src/msiglip/lightning_models.py
git diff --check
```

On Colab, prefer `unittest discover -s tests -p ...` because installed `tests`
packages can shadow the local folder.

## Key Docs

- `README.md`: public overview and current status.
- `docs/noise-robust-multilingual-framework.md`: MNEB design.
- `docs/knowledge.md`: durable concepts only; no run logs.
- `docs/journal/[train]-2026-06-11.md`: NACIR failure and attn+FFN r32.
- `docs/journal/[train]-2026-06-13.md`: PiSSA rejection and Part Align `53.00`.
- `changelog/training/changelog.md`: completed training/config/docs changes.
- `deployment/docs/journal/`: deployment logs.

Do not edit `src/person_rlf.egg-info/PKG-INFO` unless packaging metadata
regeneration is explicitly requested.

## Documentation Rules

- Do not automatically write docs/journals/changelogs unless the user asks.
- If the prompt explicitly asks to update documentation, that is confirmation.
- Durable mechanisms go to `docs/knowledge.md`.
- Run results and temporary conclusions go to `docs/journal/[train]-YYYY-MM-DD.md`.
- Deployment results go to `deployment/docs/journal/[deploy]-YYYY-MM-DD.md`.
- Completed code/config/docs changes go to `changelog/{component}/changelog.md`
  after user confirmation, unless already requested.
- For concept questions like "PiSSA là gì" or "Part-Align là gì", answer from
  durable knowledge/journal context; if the user says "ghi vào", update the
  appropriate knowledge and/or journal file.

## Coding Rules

- Use `rg`/`rg --files` for search.
- Use `apply_patch` for manual edits.
- Keep optional modules modular and disabled by default unless requested.
- New loss pattern: add objective, route in `tbps.py`, add config, add tests.
- Never revert unrelated user changes.
