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
| `results/qdq/` | Off-board QDQ proxy retrieval/fidelity JSON summaries. |
| `diagnostics/` | Small diagnostic JSON summaries for environment, mask, and activation-outlier checks. |
| `manifest.json` | Provenance map from original source paths to canonical evidence paths. |

Large model files, QNN context binaries, ONNX/QDQ model directories, raw inputs,
board `Result_*` outputs, and per-sample CSV files are intentionally not stored
here.
