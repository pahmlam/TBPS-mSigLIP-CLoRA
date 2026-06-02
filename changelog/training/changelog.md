# Training Changelog

## [2026-06-02] Add NACIR FP-only script

| # | Priority | Type | Action | Status |
|---|----------|------|--------|--------|
| 1 | MED | config | Added `run_nacir_fp_only.sh` to run NACIR with FN disabled via `fn_enable_epoch=999` while keeping FP detection active at the default epoch | COMPLETE |
| 2 | MED | docs | Updated the NACIR training report and Vietnamese knowledge base to make FP-only the next experiment after detector-off, with clean and noisy run commands | COMPLETE |

## [2026-06-02] Document NACIR detector-off result

| # | Priority | Type | Action | Status |
|---|----------|------|--------|--------|
| 1 | MED | docs | Updated NACIR training report with `output2.log`: detector-off run reached T2I R@1 `51.65`, no `GMM refit`, seed remains `2400`, and the result is near-baseline but not a strict reproduction of historical seed-2400 `52.28` | COMPLETE |
| 2 | MED | docs | Added Vietnamese knowledge entry clarifying that strict same-seed no-op proof needs a current Circle control on the same code/server, while the full NACIR regression still most likely comes from detector policy | COMPLETE |

## [2026-05-29] Add dedicated NACIR detector-off script

| # | Priority | Type | Action | Status |
|---|----------|------|--------|--------|
| 1 | MED | config | Added `run_nacir_detector_off.sh` to run the NACIR loss path with both FN and FP detectors disabled via `fn_enable_epoch=999` and `fp_enable_epoch=999` | COMPLETE |
| 2 | LOW | config | Kept `"$@"` passthrough in the dedicated script so extra Hydra overrides can still be supplied without editing the file | COMPLETE |

## [2026-05-27] Clarify FN branch follow-up after clean NACIR regression

| # | Priority | Type | Action | Status |
|---|----------|------|--------|--------|
| 1 | MED | docs | Documented that FN branch should not be used by default on clean VN3K after clean NACIR dropped to 50.70 R@1; keep FN for controlled/synthetic FN research and prioritize FP-only NACIR for caption-shuffle noise experiments | COMPLETE |
| 2 | MED | docs | Added the distinction between loss-level no-op and training-level clean-safe proof for NACIR, recommending detector-off NACIR as the next clean safety check | COMPLETE |

## [2026-05-26] Update NACIR training script with validated FN override

| # | Priority | Type | Action | Status |
|---|----------|------|--------|--------|
| 1 | MED | config | Added `loss.nacir_config.fn_prior=0.010` and `loss.nacir_config.epsilon_n=0.60` to `run_nacir.sh` so the script uses the latest notebook-selected NACIR candidate by default | COMPLETE |
| 2 | MED | config | Added `"$@"` passthrough to `run_nacir.sh` so follow-up Hydra overrides can be supplied for NACIR ablations without editing the script | COMPLETE |

## [2026-05-20] Fix notebook imports for src package layout

| # | Priority | Type | Action | Status |
|---|----------|------|--------|--------|
| 1 | HIGH | bugfix | Updated `notebooks/workspace.ipynb` imports from root modules (`model`, `data`, `utils`, `lightning_*`) to `msiglip.*` package imports | COMPLETE |
| 2 | MED | bugfix | Added notebook setup logic to insert repo `src/` into `sys.path` when running from either repo root or `notebooks/` | COMPLETE |
| 3 | HIGH | bugfix | Normalized old Hydra config paths in `notebooks/workspace.ipynb` so relative `dataset_root_dir`, tokenizer path, and backbone path resolve against repo root instead of notebook CWD | COMPLETE |

## [2026-05-17] Reorganize training code into package layout

| # | Priority | Type | Action | Status |
|---|----------|------|--------|--------|
| 1 | HIGH | refactor | Moved training/model/data/solver/utils code under `src/msiglip/` and kept root `trainer.py` / `test.py` as backward-compatible wrappers | COMPLETE |
| 2 | HIGH | config | Renamed Hydra config tree to `configs/`, added `configs/paths/default.yaml`, and routed Hydra outputs to `artifacts/training/` | COMPLETE |
| 3 | MED | config | Updated noisy correspondence default storage from root `noiseindex/` to `artifacts/training/noiseindex/` | COMPLETE |
| 4 | MED | docs | Updated README, architecture docs, AGENTS/CLAUDE instructions, and Vietnamese knowledge base for the new layout | COMPLETE |
| 5 | MED | config | Added `scripts/training_paths.sh` and sourced it from training shell wrappers so `run_*.sh` prefers standard `data/raw` / `artifacts/models/pretrained` but still supports server workspaces with root-level `VN3K/` and `m_siglip_checkpoints/` | COMPLETE |

## [2026-04-14] Idea C — Unified Noise-Aware Circle Loss (NACIR)

| # | Priority | Type | Action | Status |
|---|----------|------|--------|--------|
| 1 | HIGH | feature | Added `model/noise_aware.py` with `NoiseAwareCircleState` — manages EMA similarity stats, per-sample loss buffer, and 2-component 1D GMM (pure-PyTorch EM, no sklearn dependency). All state as `register_buffer` for checkpoint safety | COMPLETE (sanity-checked on synthetic bimodal data: separation=3.478) |
| 2 | HIGH | feature | Added `compute_noise_aware_circle()` to `model/objectives.py` — stateless loss accepting optional `fn_stats` dict and `clean_weights` tensor. Degenerates exactly to `compute_cross_modal_circle()` when detectors are off (diff=0.00 verified) | COMPLETE |
| 3 | HIGH | feature | Wired NACIR into `model/tbps.py` with curriculum-gated detectors (fn_enable_epoch=11, fp_enable_epoch=15). NACIR replaces CIR branch when `NACIR: true`; CIR path fully preserved when `NACIR: false` | COMPLETE (supports MVS augmentation via same pattern as CIR) |
| 4 | HIGH | feature | Added `on_train_epoch_end` hook in `lightning_models.py` — refits GMM every `gmm_refit_interval` epochs starting from `fp_enable_epoch`. Logs separation, component stats, and fallback flag to W&B | COMPLETE |
| 5 | MED | config | Added `NACIR: false` flag + `nacir_config` block to `config/loss/cir_msiglip.yaml` with all 9 hyperparameters (ε_n, ε_p, fn_prior, etc.) | COMPLETE (off by default — safe rollout) |
| 6 | MED | refactor | `LitTBPS.__init__` and `TBPS.__init__` accept new `num_train_samples` param; plumbed through from `trainer.py` via `len(dm.train_set)` | COMPLETE |
| 7 | MED | docs | Added Section 6 "Idea C — Unified Noise-Aware Circle Loss" to `docs/knowledge.md` in Vietnamese with Definition / Why / What / How / Thoughts | COMPLETE |
| 8 | MED | docs | Planning doc at `reports/golden-fluttering-sphinx.md` — full implementation plan with curriculum schedule, design decisions, and verification strategy | COMPLETE |

## [2026-05-06] Integrate RDE noise injection into TBPS-mSigLIP pipeline

| # | Priority | Type | Action | Status |
|---|----------|------|--------|--------|
| 1 | HIGH | feature | Added `inject_noisy_correspondence()` to `data/bases.py` — ports RDE's caption shuffling with .npy persistence for reproducibility | COMPLETE |
| 2 | HIGH | config | Added `noisy_rate: 0.0` and `noisy_file: null` to all 6 dataset YAMLs | COMPLETE |
| 3 | HIGH | feature | Wired noise injection into `TBPSDataModule.setup()` with `hydra.utils.get_original_cwd()` for noiseindex path | COMPLETE |
| 4 | MED | feature | Created `run_noise_experiments.sh` — loops over noise rates 0.0-0.8 with Hydra overrides | COMPLETE |
| 5 | LOW | config | Added `noiseindex/*.npy` to `.gitignore`, created `noiseindex/.gitkeep` | COMPLETE |

## [2026-05-12] Fix NACIR notebook paired-batch validation

| # | Priority | Type | Action | Status |
|---|----------|------|--------|--------|
| 1 | HIGH | bugfix | Updated `workspace.ipynb` Section 3 batch construction to build an identity-balanced image-text batch from shared PIDs instead of slicing `image_feats[:B]` and `text_feats[:B]` independently | COMPLETE (cell asserts `image_batch_pids == text_batch_pids`) |
| 2 | MED | docs | Updated `docs/knowledge.md` to record that old NACIR outputs were produced before the paired-batch fix and must be rerun before drawing conclusions | COMPLETE |

## [2026-05-12] Add notebook-only controlled NACIR validation

| # | Priority | Type | Action | Status |
|---|----------|------|--------|--------|
| 1 | HIGH | feature | Added `workspace.ipynb` Section 4.6 with controlled clean no-op, synthetic FN, synthetic FP, and greenlight checks for NACIR | COMPLETE (all new code cells parse with `ast.parse`) |
| 2 | MED | docs | Documented the notebook-only NACIR validation protocol in `docs/knowledge.md` | COMPLETE |

## [2026-05-12] Document NACIR in README as experimental extension

| # | Priority | Type | Action | Status |
|---|----------|------|--------|--------|
| 1 | MED | docs | Added a standalone README section for NACIR with motivation, math, detectors, curriculum, validation protocol, run commands, diagnostics, and pending result tables | COMPLETE (existing reported Circle Loss results left unchanged) |

## [2026-05-12] Document RDE noisy correspondence injection

| # | Priority | Type | Action | Status |
|---|----------|------|--------|--------|
| 1 | LOW | docs | Recorded that the RDE repo injects caption-shuffle noisy correspondence while keeping image PID fixed, creating noisy positive/FP pairs rather than FN pairs | COMPLETE |

## [2026-05-12] Refresh README for NACIR scripts and current project status

| # | Priority | Type | Action | Status |
|---|----------|------|--------|--------|
| 1 | MED | docs | Updated `README.md` to show `run_nacir.sh` as the primary NACIR entrypoint, list current training scripts, refresh repository structure, and keep NACIR results/conclusion pending | COMPLETE |
| 2 | LOW | docs | Added a Vietnamese knowledge entry documenting the README status update and rationale | COMPLETE |

## [2026-05-13] Remove placeholder loss from workspace notebook

| # | Priority | Type | Action | Status |
|---|----------|------|--------|--------|
| 1 | MED | refactor | Removed the `my_new_loss` placeholder cell and its gradient-table wrapper from `workspace.ipynb` to avoid confusing it with NACIR or implemented losses | COMPLETE |
| 2 | MED | refactor | Updated Section 8 mini fine-tune to use `compute_cross_modal_circle()` with the production margin/scale instead of the deleted placeholder loss | COMPLETE |
| 3 | LOW | docs | Updated `docs/knowledge.md` with the rationale and verification commands for the notebook cleanup | COMPLETE |

## [2026-05-16] Assess latest workspace NACIR outputs

| # | Priority | Type | Action | Status |
|---|----------|------|--------|--------|
| 1 | MED | docs | Recorded the latest `workspace.ipynb` assessment: retrieval metrics are strong, clean no-op and synthetic FP pass, but controlled synthetic FN fails no-collapse with total negative gradient ratio 0.075 | COMPLETE |

## [2026-05-16] Tune NACIR FN branch selection in workspace

| # | Priority | Type | Action | Status |
|---|----------|------|--------|--------|
| 1 | HIGH | refactor | Updated `workspace.ipynb` Section 4.6 synthetic FN validation to sweep `fn_prior` and `epsilon_n`, then select a conservative candidate that passes no-collapse instead of maximizing P(FN) gap | COMPLETE |
| 2 | MED | docs | Documented the FN tuning strategy and selection criteria in `docs/knowledge.md` | COMPLETE |

## [2026-05-20] Assess post-tuning NACIR workspace outputs

| # | Priority | Type | Action | Status |
|---|----------|------|--------|--------|
| 1 | MED | docs | Recorded the latest `notebooks/workspace.ipynb` controlled validation: clean no-op and synthetic FP pass, FN suppression works, but no-collapse still fails with total negative gradient ratio 0.258 < 0.30 | COMPLETE |

## [2026-05-20] Extend NACIR FN epsilon sweep in workspace

| # | Priority | Type | Action | Status |
|---|----------|------|--------|--------|
| 1 | MED | config | Extended `notebooks/workspace.ipynb` Section 4.6 `FN_EPS_N_SWEEP` through 0.80 so the notebook can search milder FN suppression candidates that preserve >30% negative gradient | COMPLETE |
| 2 | LOW | docs | Added notebook output guidance for printing the selected NACIR training override only when controlled validation passes | COMPLETE |

## [2026-05-20] Switch workspace mini fine-tune to NACIR

| # | Priority | Type | Action | Status |
|---|----------|------|--------|--------|
| 1 | MED | refactor | Updated `notebooks/workspace.ipynb` Section 8 mini fine-tune to use `compute_noise_aware_circle()` with the Section 4.6 selected `fn_prior` / `epsilon_n` candidate instead of vanilla Cross-Modal Circle | COMPLETE |
| 2 | LOW | config | Reduced mini fine-tune batch size and post-extraction batch size to lower CUDA OOM risk during the notebook sanity check | COMPLETE |
