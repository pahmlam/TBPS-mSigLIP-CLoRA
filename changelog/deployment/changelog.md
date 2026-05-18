# Deployment Changelog

## [2026-05-17] Evaluate VN3K QNN HTP output artifact

| # | Priority | Type | Action | Status |
|---|----------|------|--------|--------|
| 1 | MED | docs | Added `docs/knowledge.md` entry for the first VN3K QNN HTP output evaluation, including output shape checks, NaN/Inf checks, cosine sanity check, and next benchmark/accuracy steps | COMPLETE (records that runtime output passed basic validity but still needs profiling and baseline comparison) |

## [2026-05-17] Track deployment artifacts for board/local sync

| # | Priority | Type | Action | Status |
|---|----------|------|--------|--------|
| 1 | MED | config | Updated `.gitignore` to keep `artifacts/deployment/**` versionable while leaving non-deployment artifacts ignored, with `artifacts/deployment/exports/` excluded as a large generated export directory | COMPLETE (makes QNN inputs/outputs easier to sync between local and RB3 without tracking bulky exported models) |

## [2026-05-17] Move deployment artifacts out of repo root

| # | Priority | Type | Action | Status |
|---|----------|------|--------|--------|
| 1 | HIGH | refactor | Moved QNN runtime configs into `deployment/config/qnn/` and changed deployment script defaults to write exports, logs, QNN inputs, QNN runs, and runtime state under `artifacts/deployment/` | COMPLETE |
| 2 | HIGH | refactor | Relocated existing root deployment clutter (`out_qnn*`, raw input files, exported model, QNN logs/inputs, root checkpoint) under ignored `artifacts/` paths | COMPLETE |
| 3 | MED | bugfix | Updated deployment imports to use the `msiglip.*` package instead of root-level module paths and removed project-root `sys.path` assumptions where practical | COMPLETE |
| 4 | MED | docs | Updated deployment README/docs and knowledge base to prefer `qnn-net-run` and the new artifact/config paths | COMPLETE |

## [2026-05-17] Fix VN3K QNN raw input writer

| # | Priority | Type | Action | Status |
|---|----------|------|--------|--------|
| 1 | HIGH | bugfix | Fixed `prepare_vn3k_vision_inputs.py` to write raw tensors via a binary file handle instead of passing `Path` directly to `array.tofile()` | COMPLETE (resolves `AttributeError: 'PosixPath' object has no attribute 'write'`) |
| 2 | LOW | bugfix | Replaced deprecated Pillow `Image.getdata()` usage with `Image.tobytes()` while preserving RGB-to-NCHW channel-major preprocessing | COMPLETE (avoids Pillow 14 deprecation warning) |

## [2026-05-16] RB3-first modular demo scaffold

| # | Priority | Type | Action | Status |
|---|----------|------|--------|--------|
| 1 | HIGH | feature | Added `deployment/demo/` organized into `core/`, `adapters/`, `cli/`, and `tests/`, with plugin contracts for source, detector, tracker, crop selector, image encoder, text encoder, vector store, spool, and uploader | COMPLETE (creates swappable module boundaries for the end-to-end demo) |
| 2 | HIGH | feature | Added default image/video, full-frame detector, simple tracker, crop selector, fake/ONNX/QNN vision encoder, local JSONL vector store, disk spool, and uploader adapters | COMPLETE (supports local preflight and RB3 QNN path) |
| 3 | MED | feature | Added `run_ingest`, `run_search`, and `health` CLIs plus unit tests for vector collapse and spool behavior | COMPLETE (provides executable local wiring checks) |
| 4 | MED | docs | Updated deployment design, deployment plan, knowledge base, and architecture decisions to mark local tests as preflight only and RB3 QNN as the real acceptance gate | COMPLETE (prevents confusing local smoke tests with board deployment) |

## [2026-05-14] Automate VN3K real-image QNN vision tests

| # | Priority | Type | Action | Status |
|---|----------|------|--------|--------|
| 1 | HIGH | feature | Added `deployment/scripts/qnn/prepare_vn3k_vision_inputs.py` to sample VN3K images, apply mSigLIP vision preprocessing, export NCHW float32 raw inputs, create `input_list.txt`, `manifest.csv`, and a RB3 `run_qnn_vision.sh` helper | COMPLETE (enables repeatable real-image QNN tests) |
| 2 | MED | feature | Added `deployment/scripts/qnn/summarize_qnn_outputs.py` to validate QNN `Result_*/output_0.raw` tensors, report finite/statistical checks, and optionally export normalized embeddings/stat CSVs | COMPLETE (automates output sanity checks after HTP runs) |
| 3 | LOW | docs | Updated `docs/knowledge.md` with the VN3K/QNN automation commands under the QNN HTP runtime note | COMPLETE |

## [2026-05-12] Surface current deployment progress in root README

| # | Priority | Type | Action | Status |
|---|----------|------|--------|--------|
| 1 | MED | docs | Updated root `README.md` deployment status to reflect working LoRA export, ONNX export, local FP16 conversion, and vision INT8 HTP compile success; marked text compile, RB3 benchmark, real calibration, and quantized accuracy as pending | COMPLETE |

## [2026-05-07] Vision encoder INT8 compile success — pipeline documentation updates

| # | Priority | Type | Action | Status |
|---|----------|------|--------|--------|
| 1 | HIGH | docs | Updated `aihub-experiments.md` with experiments #10 (jpyvrrv7p, failed) and #11 (jgkr7qwn5, success). Added Summary of Learnings item #8 about `--preserve_io_datatype` auto-injection. Updated Commands Reference with working command. | COMPLETE (prevents re-running failed configs) |
| 2 | HIGH | docs | Updated `deployment-plan.md` — pipeline status from "stuck at step 3→4" to "step 4 complete for vision encoder". Updated §4 (root cause resolved), §5 (Phase 1 done, Phase 2 = on-device benchmarking). Added deprecation warnings. | COMPLETE (single source of truth reflects current state) |
| 3 | MED | docs | Added §9 to `docs/knowledge.md` — INT8 quantization for HTP, `--preserve_io_datatype` auto-injection, dummy vs real calibration, in Vietnamese. | COMPLETE (preserves HTP compilation knowledge) |

## [2026-04-26] Long-dwell handling for people who stay in frame too long

| # | Priority | Type | Action | Status |
|---|----------|------|--------|--------|
| 1 | HIGH | docs | Added explicit long-dwell handling to `deployment/docs/end-to-end-system-design.md`: freeze new embeddings after the best snapshots are collected, keep only metadata heartbeat, and allow refresh only on major appearance change or long interval | COMPLETE (prevents a stationary person from flooding the vector store) |
| 2 | LOW | docs | Updated `docs/knowledge.md` with the rationale that long-dwell duplication should be handled by metadata continuation rather than repeated vector writes | COMPLETE (keeps the reasoning in the project knowledge base) |

## [2026-04-26] Query-time dedup and result collapsing for top-k retrieval

| # | Priority | Type | Action | Status |
|---|----------|------|--------|--------|
| 1 | HIGH | docs | Updated `deployment/docs/end-to-end-system-design.md` to search on snapshot embeddings but collapse results by `episode_id` before returning top 10 to the Web UI | COMPLETE (prevents top-k from being dominated by near-duplicate rows of the same person) |
| 2 | MED | docs | Added a concrete `search_result_policy` and updated the search API response example with `episode_id`, `snapshot_count`, and `first_seen` | COMPLETE (makes query-time dedup implementable) |
| 3 | LOW | docs | Updated `docs/knowledge.md` with the rationale for separating search unit (snapshot) from display unit (`episode_id`) | COMPLETE (preserves retrieval UX rationale in Vietnamese) |

## [2026-04-26] End-to-end capture suppression policy for duplicate person images

| # | Priority | Type | Action | Status |
|---|----------|------|--------|--------|
| 1 | HIGH | docs | Expanded `deployment/docs/end-to-end-system-design.md` with a concrete anti-duplicate capture policy: frame gating, intra-track rate limiting, track finalization rules, and cross-track suppression via `recent_identity_cache` and `episode_id` | COMPLETE (addresses repeated captures of the same person) |
| 2 | MED | docs | Extended proposed schema with `episode_id`, `suppressed_frame_count`, `last_saved_at`, and `save_reason` to make suppression decisions traceable in the backend | COMPLETE (supports later implementation and debugging) |
| 3 | LOW | docs | Updated `docs/knowledge.md` with the reasoning behind track-level and cross-track suppression for the end-to-end system | COMPLETE (preserves architectural rationale in Vietnamese) |

## [2026-04-15] Deployment plan + AI Hub experiment log

| # | Priority | Type | Action | Status |
|---|----------|------|--------|--------|
| 1 | HIGH | docs | Created `deployment/docs/deployment-plan.md` — synthesizes hardware, pipeline, root cause of HTP FP I/O rejection, and recommended next steps (INT8 path + GPU fallback) | COMPLETE (single source of truth for deployment state) |
| 2 | HIGH | docs | Created `deployment/docs/aihub-experiments.md` — running table of all 9 qai-hub compile attempts + summary of learnings | COMPLETE (prevents repeating failed configs) |
| 3 | HIGH | docs | Added `deployment/scripts/onnx/to_fp16.py` — local onnxconverter-common FP16 conversion with FP16 I/O | COMPLETE (used in experiment #8) |
| 4 | MED | config | Created `.claude/rules/aihub-experiments.md` — forces Claude to log every qai-hub run | COMPLETE (enforcement rule) |
| 5 | LOW | docs | Updated `deployment/README.md` and `.claude/rules/deployment.md` with pointers to new docs | COMPLETE |

## [2026-04-15] Fix ONNX export for Qualcomm AI Hub — use directory format for external weights

| # | Priority | Type | Action | Status |
|---|----------|------|--------|--------|
| 1 | HIGH | bugfix | Save each ONNX encoder in its own subdirectory (`vision_onnx/`, `text_onnx/`) so AI Hub can upload .onnx + .onnx.data together via `--model dir/` | COMPLETE (AI Hub rejects single .onnx file missing external weights) |
| 2 | MED | bugfix | Fix size reporting in export log — now reports graph + external weights total, not just .onnx file | COMPLETE (was showing 1.4 MB instead of actual ~355 MB / ~1.0 GB) |
| 3 | LOW | docs | Updated README.md, knowledge.md qai-hub commands to use directory format | COMPLETE (all references corrected) |

## [2026-04-14] Split ONNX export from lora_fp16/export.py into separate onnx/export.py

| # | Priority | Type | Action | Status |
|---|----------|------|--------|--------|
| 1 | MED | refactor | Removed ONNX export logic and `--format` flag from `lora_fp16/export.py` — now only does LoRA merge + FP16/FP32 export | COMPLETE (folder name `lora_fp16` should match its scope) |
| 2 | MED | feature | Created `deployment/scripts/onnx/export.py` — loads exported .pt + config.yaml, rebuilds TBPS model, exports ONNX | COMPLETE (decouples ONNX step, can run independently) |
| 3 | LOW | docs | Updated README, CLAUDE.md, analyze_checkpoint.py, knowledge.md with new two-step pipeline | COMPLETE (all references updated) |
