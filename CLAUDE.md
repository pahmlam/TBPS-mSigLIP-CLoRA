# CLAUDE.md

## Project Snapshot

**mSigLIP** — multilingual Text-Based Person Search (Vietnamese, English, Chinese)
with mSigLIP + LoRA + Curriculum Circle Loss, plus an INT8 deployment path for
Qualcomm RB3 Gen2. Training method accepted at ICIP 2026.

- Task: align person images and multilingual text in a shared 768-dim space.
- Primary metric: **text-to-image Rank@1** (T2I R@1).
- Backbone: `siglip-base-patch16-256-multilingual` (376M params; LoRA trains ~1.57%).

Two workspaces:

- **Root** — training and model optimization (the paper method).
- **`deployment/`** — edge export/compression/runtime for RB3 Gen2 / QCS6490 / HTP v68.
  This is the current active thesis contribution (Part 2).

## Current State

Training (VN3K T2I R@1):

| Result | Status |
|---|---|
| `52.28` | Paper headline, LoRA + Curriculum Circle, seed 2400 (clean deploy baseline) |
| `52.83` | Attn+FFN LoRA r32, batch64, accum2 |
| `53.00` T2I / `53.25` I2T | Best experimental: Part Align + Attn+FFN LoRA r32 |
| `47.58` | PiSSA r32 — rejected for current VN3K setup |

Deployment (vision encoder, all-INT8 W8A8 on HTP v68; QDQ proxy = board-faithful).
**Deploy target: T2I R@1 ≥ 50; any result < 50 is a FAIL.**

| Result | Status |
|---|---|
| `50.85` T2I R@1 | **Current best — QAT v8 learned rotation** (QDQ proxy, meets the ≥50 deploy target) |
| `49.30` T2I R@1 | QAT v6 random rotation (previous ceiling, below 50 target) |
| `48.50` T2I R@1 | QAT v4 — **board-verified** binary (below 50 target; `~32.7 ms/img`, `22.9 FPS`, `~90 MB`) |
| Text encoder | Pending — next: learned rotation + QAT, then both-INT8 retrieval on board |

The full numeric history (jobs, fidelity, gates) lives in the deploy-master journal
(see Key Docs). README has the public training results and footprint tables.

## Main Files

Training:

- `trainer.py` (Hydra entry), `test.py` (eval entry, Fire wrapper).
- `src/msiglip/lightning_models.py` (LitTBPS), `lightning_data.py` (data module).
- `src/msiglip/model/tbps.py` — forward pass and loss routing.
- `src/msiglip/model/objectives.py` — N-ITC, Circle, C-ITC, SimCLR, Part Align, MNEB aux losses.
- `src/msiglip/model/lora.py` — PEFT LoRA setup; `model/evidence_bank.py` — MNEB-HN bank.
- `src/msiglip/data/` — `vn3k_vi.py`, `vn3k_en.py`, `vn3k_mixed.py`, `cuhkpedes.py`,
  `cuhk_10_percent_vn3k_mix.py`, `prw_tps_cn.py`.

Config (Hydra):

- `configs/cir_msiglip.yaml` (main), `configs/loss/cir_msiglip.yaml` (loss),
  `configs/dataset/*.yaml`, `configs/lora/*.yaml`.

Deployment:

- `deployment/scripts/lora_fp16/export.py` — merge LoRA → dense FP32 export.
- `deployment/scripts/qnn/` — rotation (`rotate_vision_encoder.py`, `learn_rotation.py`),
  QAT (`train_vision_quant_robust.py`), ONNX export, AI Hub submit, eval/compare.
- `deployment/config/qnn/` — HTP runtime config JSON.
- Generated artifacts go under `artifacts/deployment/` (git-ignored).

## Repository Layout

```text
src/msiglip/              # Training stack
  lightning_models.py     #   LitTBPS Lightning module (training/eval loop)
  lightning_data.py       #   TBPSDataModule
  model/                  #   mSigLIP, lora.py, tbps.py (routing), objectives.py, evidence_bank.py
  data/                   #   VN3K / CUHK-PEDES / PRW-TPS-CN dataset adapters
  solver/                 #   optimizer + scheduler
configs/                  # Hydra: cir_msiglip.yaml, loss/, dataset/, lora/
trainer.py / test.py      # Hydra train entry / Fire eval entry
run_*.sh                  # Reproducible training recipes
notebooks/                # workspace.ipynb (loss playground), colab_training_experiments.ipynb

deployment/               # Edge stack (RB3 Gen2 / HTP v68)
  scripts/lora_fp16/      #   LoRA merge → dense FP32 export
  scripts/qnn/            #   rotation, QAT, ONNX export, AI Hub submit, eval/compare
  config/qnn/             #   HTP runtime config JSON
  docs/                   #   w8a8_qat_rotated.md (method), journal/, runbooks

artifacts/                # Generated outputs (git-ignored): checkpoints, deployment/
docs/ knowledge/ changelog/ figures/   # docs, research notes, changelogs, paper figures
```

Data flow (training): Hydra config → `TBPSDataModule` → `LitTBPS` → `tbps.py` forward
routes losses from `objectives.py` → embeddings scored by T2I/I2T R@1.
Data flow (deploy): checkpoint → LoRA merge → rotate → QAT → ONNX → W8A8 → board.

## Loss Design

Default main objective:

```text
1.0 * N-ITC/MVS + curriculum(t) * Circle + 0.1 * C-ITC + 0.4 * SimCLR
```

Non-obvious rules:

- **MVS is part of the N-ITC path** — do not drop it from baseline descriptions.
- Circle Loss is the hard-negative core: `m=0.25`, `gamma=128`.
- Circle curriculum: epoch 0-5 off, 6-20 ramp to `0.1`, 21-60 stable.
- `PART_ALIGN` is optional, currently useful with `+lora=attn_ffn_r32`.
- Preserve baseline behavior when optional modules are disabled.
- New loss pattern (3-file sync): add objective in `objectives.py`, route in `tbps.py`,
  add config flag in `configs/loss/cir_msiglip.yaml`, then add a test.

## Deployment Recipe (vision)

Pipeline: LoRA merge → mean-preserving rotation → opset-20 fused GELU/LayerNorm →
QAT distillation (per-tensor STE + EMA observer) → W8A8 quantize/compile/link → board run.

Non-obvious rules:

- HTP v68 is **all-INT8 only** — no A16 (needs v73+), no internal float, integer I/O.
- Rotation must be **mean-preserving** (`Q·1=1`) to keep fused LayerNorm; never convert
  LN→RMSNorm (it re-exposes `Pow`/`ReduceMean` to the quantizer).
- **Learned rotation (v8)** beats random rotation; objective is `min_Q Σ max|aQᵀ|²`
  (not quant-MSE — STE detaches that gradient). It is the chosen rotation for text too.
- Decisive metric is **retrieval R@1**, not cosine. Always compare QDQ vs the *original*
  merged FP32 model. Keep text FP32 when measuring vision-only (and vice versa).
- Method/math reference: `deployment/docs/w8a8_qat_rotated.md` (math only, no commands).

## Optional Modules (disabled by default)

- **MNEB-HN** (Multilingual Noise Evidence Bank for Hard-Negative TBPS):
  `loss.MNEB=false` by default; target is CUHK-PEDES natural noise, not clean VN3K.
  Never mutates Circle `alpha_n`/`alpha_p`; correction enters only via `fnm_aux_loss`
  and `rde_aux_loss`. Mutually exclusive with NACIR. Design: `docs/noise-robust-multilingual-framework.md`.
- **NACIR**: legacy ablations only — use only when explicitly asked.

## LoRA Recipes

`configs/lora/`: `default`, `attn_ffn_r16`, `attn_ffn_r32` (**preferred**), `attn_ffn_r64`,
`attn_ffn_r32_dora`, `attn_ffn_r32_rslora`, `attn_ffn_r32_pissa` (rejected).
Preferred clean VN3K run: `bash run_part_align_lora_attn_ffn_r32.sh`.

## Tests

```bash
venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
venv/bin/python -m compileall src/msiglip/model/objectives.py src/msiglip/model/tbps.py
git diff --check
```

On Colab, always use `unittest discover -s tests -p ...` (installed `tests` packages can
shadow the local folder). Colab notebook: `notebooks/colab_training_experiments.ipynb`;
package script: `scripts/colab/package_training_code.sh`.

## Documentation Rules

- Do **not** auto-write docs/journals/changelogs. An explicit request in the prompt
  (e.g. "ghi vào", "update docs") counts as confirmation for the named files.
- Durable concepts/mechanisms → `docs/knowledge.md` (Vietnamese, concept-focused; no run logs).
- Training run results / temporary conclusions → `docs/journal/[train]-YYYY-MM-DD.md`.
- Deployment results, AI Hub jobs (`qai-hub`/`qai_hub`), board fidelity → all go to the single
  canonical `deployment/docs/journal/[deploy-master].md`.
- Demo-system changes under `deployment/demo/` → `deployment/docs/journal/[demo-system]-YYYY-MM-DD.md`.
- Completed code/config/docs changes → `changelog/{component}/changelog.md` after confirmation.
- Do not edit `src/person_rlf.egg-info/PKG-INFO` unless packaging regen is requested.

## Coding Rules

- Use `rg` / `rg --files` for search.
- Python: ruff, PEP 8, max line 120, `| None` over `Optional`, comment tensor shapes,
  `F.normalize(..., dim=1, p=2)` for L2, no bare `print()` in training code.
- Keep optional modules modular and disabled by default unless requested.
- Never revert unrelated user changes.

## Key Docs

- `README.md` — public overview, training results, model footprint.
- `docs/knowledge.md` — durable concepts; `docs/ARCHITECTURE.md` — training architecture.
- `docs/journal/[train]-2026-06-11.md` — NACIR failure, attn+FFN r32.
- `docs/journal/[train]-2026-06-13.md` — PiSSA rejection, Part Align `53.00`.
- `deployment/docs/w8a8_qat_rotated.md` — W8A8 rotation/QAT method (math only).
- `deployment/docs/journal/[deploy-master].md` — canonical deployment journal.
- `deployment/docs/runbook-w8a8-v8-both-int8.md` — learned-rotation / text / both-INT8 runbook.
