# Deployment Evidence Archive

This directory is the canonical evidence archive for the mSigLIP RB3/QNN
deployment work. It keeps small, curated summaries and AI Hub logs under
semantic names, while large generated artifacts remain in their working
locations.

## Layout

| Path | Contents |
|---|---|
| `aihub/` | Curated Qualcomm AI Hub job logs, named by result/failure mode with job ID retained for traceability. |
| `results/board/` | Board retrieval and board fidelity JSON summaries. |
| `results/board/per-sample/` | Small per-sample board fidelity CSV tables. |
| `results/qdq/` | Off-board QDQ proxy retrieval/fidelity JSON summaries. |
| `runtime/board/` | Board `qnn-net-run` execution metadata, generated profiles, and profiling logs. |
| `diagnostics/` | Small diagnostic JSON summaries for environment, mask, and activation-outlier checks. |
| `manifest.json` | Provenance map from original source paths to canonical evidence paths. |

Large model files, QNN context binaries, ONNX/QDQ model directories, raw inputs,
and board `Result_*` outputs are intentionally not stored here.
