# AGENTS.md

## Project Snapshot

**mSigLIP** - multilingual Text-Based Person Search (TBPS) with mSigLIP, LoRA,
Curriculum Circle Loss, optional Part Align, and optional MNEB-HN noise robustness.

Workspaces:

- Root: training/model optimization.
- `deployment/`: RB3 Gen2 / QCS6490 / QNN / HTP v68 deployment.

Core facts:

- Task: align person images and multilingual text in a shared 768-d embedding space.
- Primary metric: text-to-image Rank@1.
- Backbone: `siglip-base-patch16-256-multilingual`.
- Preferred adapter: LoRA attention + FFN, `+lora=attn_ffn_r32`.
- Datasets: VN3K/VnPersonSearch (Vietnamese, clean), CUHK-PEDES (English, noisy),
  PRW-TPS-CN (Chinese generalization).
- Target board: Qualcomm RB3 Gen2 / QCS6490, 4 GB RAM, ARM64, Hexagon HTP v68.

## Current Results

Training:

| Result | Status |
|---|---|
| `52.28` VN3K T2I R@1 | Paper/historical Circle + LoRA baseline, seed 2400 |
| `52.83` VN3K T2I R@1 | Attn+FFN LoRA r32, batch64, accum2 |
| `53.00` T2I / `53.25` I2T | Best experimental VN3K: Part Align + Attn+FFN LoRA r32 |
| PiSSA r32 | Rejected for current VN3K setup; peaked around `47.58` |
| MNEB-HN | Implemented, experimental, disabled by default; target CUHK-PEDES noise |

Deployment:

| Result | Status |
|---|---|
| `50.35` T2I / `54.20` I2T | **Official final deploy**: direct board both-INT8 W8A8, vision v9 + split-text, PASS target `>=50` |
| `50.63` T2I / `53.90` I2T | Current off-board both-INT8 QDQ proxy v9 |
| `50.35` T2I / `54.55` I2T | Vision v9 board isolation, `32.54 ms/image`, `24.29 FPS` |
| `51.30` T2I / `54.80` I2T | Split-text board isolation, `7.87 ms/query`, `74.75 q/s` |
| `50.85` T2I / `52.90` I2T | Vision v8 learned-rotation QDQ ablation, first proxy pass |
| Full text graph | Links but unusable on board: HTP v68 dynamic token `Gather` ignores `input_ids` |

Use `52.28` as the reporting baseline for deploy drops. Local FP32 sanity `52.40` only
checks pipeline reproduction.

## Main Files

Training:

- `trainer.py`, `test.py`
- `src/msiglip/lightning_data.py`, `src/msiglip/lightning_models.py`
- `src/msiglip/model/tbps.py` - forward and loss routing
- `src/msiglip/model/objectives.py` - N-ITC, Circle, C-ITC, SimCLR, Part Align, MNEB aux
- `src/msiglip/model/lora.py`, `src/msiglip/model/evidence_bank.py`
- `src/msiglip/data/` - VN3K, CUHK-PEDES, PRW-TPS-CN adapters

Config:

- `configs/cir_msiglip.yaml`
- `configs/loss/cir_msiglip.yaml`
- `configs/lora/*.yaml`
- `configs/dataset/*.yaml`

Deployment:

- `deployment/scripts/lora_fp16/export.py` - merge LoRA to dense export
- `deployment/scripts/qnn/` - rotation, QAT, ONNX, AI Hub, board eval/compare
- `deployment/config/qnn/` - HTP runtime configs
- `artifacts/deployment/` - generated deployment outputs and logs

Do not put generated deployment outputs outside `artifacts/deployment/` unless asked.

## Loss Design

Default objective:

```text
1.0 * N-ITC/MVS + curriculum(t) * Circle + 0.1 * C-ITC + 0.4 * SimCLR
```

Rules:

- MVS is part of N-ITC; do not drop it from baseline descriptions.
- Circle Loss: hard-negative core, `m=0.25`, `gamma=128`.
- Circle curriculum: epochs 0-5 off, 6-20 ramp to `0.1`, 21-60 stable.
- `PART_ALIGN` is optional and currently useful with `attn_ffn_r32`.
- Preserve baseline behavior when optional modules are disabled.
- New loss pattern: add objective in `objectives.py`, route in `tbps.py`, add config
  flag in `configs/loss/cir_msiglip.yaml`, then add focused tests.

## Deployment Rules

Final path:

```text
image -> vision v9 QNN context binary on HTP
text  -> RB3 CPU token-embedding lookup -> split-text QNN context binary on HTP
retrieval -> raw dot-product ranking, matching LitTBPS metrics
```

Key constraints:

- HTP v68 deploy path is all-INT8 W8A8 with integer I/O. A16 needs v73+ for the
  relevant attention/LN patterns; internal float and FP I/O do not link.
- Use opset-20 fused `Gelu` and `LayerNormalization`. Avoid exports that expose
  `Pow` / `ReduceMean` clusters.
- Rotation must be mean-preserving (`Q * 1 = 1`) so fused LayerNorm remains valid.
  Do not convert LN to RMSNorm.
- Learned rotation v8/v9 beats random rotation. Objective: `min_Q sum max|aQ^T|^2`.
- v9 is the final vision board recipe: larger calibration/search budget, best-Q
  folding, same learned mean-preserving theory, longer QAT.
- Retrieval R@1 is decisive; cosine is only a fidelity proxy. Rotation-only had
  near-`0.90` cosine but failed retrieval at `45.42`.
- For isolation tests, keep the other tower FP32. For final reporting, use direct
  board both-INT8.
- Full text graph is rejected as board path even though it links: `input_ids` and
  all-zero IDs produce identical board outputs. Use split-text (`inputs_embeds`).

Acceptance gates:

| Gate | Threshold |
|---|---:|
| LoRA merge | no `lora` / `adapter` / `base_layer` keys |
| Rotation FP32 invariance | cosine min `>=0.9999` |
| Static ONNX vs PyTorch | cosine mean `>=0.999` |
| ONNX op sanity | `Pow=0`, fused `Gelu`, fused `LayerNormalization` |
| QDQ vs PyTorch | target mean `>=0.95`, min `>=0.90` |
| Board vs PyTorch | mean `>=0.90` |
| Deploy retrieval | T2I R@1 `>=50.0` |

Do not repeat failed directions: FP I/O on HTP, dummy calibration as accuracy evidence,
plain PTQ W8A8, more calibration alone, `_float` QDQ surgery for deploy, native W8A16
on v68, clipping outliers, RMSNorm rotation, R2 head-dim Hadamard, or full text graph
with dynamic `input_ids` on board.

## Optional Modules

MNEB-HN:

- `loss.MNEB=false` by default; target is CUHK-PEDES natural FN/FP noise.
- Mutually exclusive with NACIR.
- With aux losses disabled, it collects/logs evidence only and must not affect loss.
- Never mutates Circle `alpha_n` or `alpha_p`; corrections enter only through
  `fnm_aux_loss` and `rde_aux_loss`.

NACIR:

- Legacy ablation path, not current direction.
- Do not recommend unless user explicitly asks for NACIR work.

## LoRA Recipes

`configs/lora/`: `default`, `attn_ffn_r16`, `attn_ffn_r32` (preferred),
`attn_ffn_r64`, `attn_ffn_r32_dora`, `attn_ffn_r32_rslora`,
`attn_ffn_r32_pissa` (rejected).

Preferred clean VN3K run:

```bash
bash run_part_align_lora_attn_ffn_r32.sh
bash run_mneb_hn.sh
```

## Workflow

- Training is expensive: inspect journal/config/script, validate locally or in
  `notebooks/workspace.ipynb` when feasible, run focused tests, then launch full runs.
- Colab: use `notebooks/colab_training_experiments.ipynb` and package with
  `scripts/colab/package_training_code.sh`.
- Use `rg` / `rg --files` for search and `apply_patch` for manual edits.
- Keep optional modules modular and disabled by default unless asked.
- Never revert unrelated user changes.

## Tests

Useful checks: `venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v`,
`venv/bin/python -m compileall src/msiglip/model/objectives.py src/msiglip/model/tbps.py src/msiglip/model/evidence_bank.py src/msiglip/lightning_models.py`,
and `git diff --check`. On Colab, prefer `unittest discover -s tests -p ...`;
an installed `tests` package can shadow the local folder.

## Documentation Policy

- Do not auto-write docs/journals/changelogs. Explicit user requests count as confirmation.
- Concepts -> `docs/knowledge.md`; no run logs there.
- Training runs -> `docs/journal/[train]-YYYY-MM-DD.md`.
- Deployment/model-compression results -> `deployment/docs/journal/[deploy-master].md`.
- Demo-system work -> `deployment/docs/journal/[demo-system]-YYYY-MM-DD.md`.
- Completed code/config/docs changes -> `changelog/{component}/changelog.md` after confirmation.
- Do not edit `src/person_rlf.egg-info/PKG-INFO` unless packaging regen is requested.

## Key Docs

Read `README.md`, `docs/ARCHITECTURE.md`, `docs/knowledge.md`,
`docs/noise-robust-multilingual-framework.md`, the dated training journals, and
`reports/architecture-decisions.md` for training context. For deployment, prefer
`deployment/docs/comprehensive_results.md`, `deployment/docs/w8a8_qat_rotated.md`,
and `deployment/docs/journal/[deploy-master].md`.
