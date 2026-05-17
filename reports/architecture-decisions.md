# Architecture Decisions

## [2026-05-17] Standard AI Project Repository Layout

### Source Package and Artifact Separation
- **Decision**: Move core Python code into `src/msiglip/`, rename Hydra configs to `configs/`, keep `deployment/` as a separate subsystem, and route generated outputs to ignored `artifacts/` directories.
- **Reason**: The previous root mixed source code with checkpoints, ONNX exports, QNN input/output files, runtime logs, and deployment configs. Separating source from generated artifacts reduces accidental commits, makes imports explicit, and gives training/deployment scripts stable default locations.
- **Alternatives considered**: Only moving deployment outputs (lower risk but leaves source layout flat), full Kedro-style restructuring (cleaner but too disruptive for the current research workflow).
- **Compatibility**: Root `trainer.py`, `test.py`, and existing `run_*.sh` commands remain wrappers around the new package implementation.

## [2024] Initial Framework Design

### Backbone Choice
- **Decision**: `siglip-base-patch16-256-multilingual` (376M params)
- **Reason**: Native multilingual support (Vietnamese, English, Chinese) with strong vision-language alignment. SigLIP's sigmoid loss is more suitable for noisy data than CLIP's softmax.
- **Alternatives considered**: CLIP (ViT-B/16), ALIGN, multilingual CLIP — all lack native multilingual text encoder or robust noise handling

### LoRA over Full Fine-Tuning
- **Decision**: LoRA (r=32, alpha=64) on Q/K/V/O projections of both vision and text towers
- **Reason**: Enables 3x larger batch sizes (BS=24 on 12GB VRAM vs BS=8 full FT), constrains optimization to low-rank subspace for stability, only 1.57% trainable params
- **Trade-off**: Slightly less expressive than full fine-tuning, but curriculum Circle Loss compensates by providing stronger gradient signal on hard negatives

### Cross-Modal Circle Loss with Curriculum
- **Decision**: Auxiliary Circle Loss (m=0.35, gamma=128) with linear warmup schedule (0 for epochs 0-5, ramp to 0.1 over epochs 6-20, stable at 0.1)
- **Reason**: Standard N-ITC sigmoid loss has vanishing gradients for semi-hard negatives. Circle Loss provides adaptive pair-wise re-weighting. Curriculum prevents early disruption of global alignment.
- **Alternatives considered**: Triplet loss (too slow convergence), Focal loss (not pair-wise), fixed Circle Loss weight (disrupts early training)

### PyTorch Lightning + Hydra
- **Decision**: Lightning for training loop, Hydra for config management
- **Reason**: Lightning handles DDP, mixed precision, gradient accumulation, checkpointing. Hydra enables composable configs with CLI overrides for experiment sweeps.
- **Trade-off**: More boilerplate than raw PyTorch, but reproducibility and experiment management justify it

### Edge Deployment Strategy (RB3 Gen2)
- **Decision**: LoRA merge → FP16 → ONNX → SNPE pipeline
- **Reason**: 4GB RAM constraint requires aggressive compression. LoRA merge eliminates adapter overhead. FP16 halves model size. SNPE leverages Qualcomm Hexagon DSP for inference acceleration.
- **Alternatives considered**: INT8 quantization (too much accuracy loss), TensorRT (not available on Qualcomm), direct PyTorch (too large footprint)

## [2026-04-26] End-to-End System Deployment Design

### Split Image and Text Inference Across Edge and Cloud
- **Decision**: Run person detection, tracking, crop selection, and image embedding on the RB3 Gen2 board; run text embedding on an external service using the exact same mSigLIP text encoder checkpoint.
- **Reason**: The board has limited usable RAM and must prioritize continuous image ingestion. Running both vision and text towers concurrently increases peak memory risk and complicates realtime scheduling. The shared embedding space requirement also rules out replacing the text encoder with an unrelated embedding model.
- **Alternatives considered**: Running both encoders on the board (higher OOM risk), using a third-party text embedding API (embedding space mismatch), moving image embedding to the cloud (breaks on-device objective)

### Keep Vector Search and Persistent Storage Off-Board
- **Decision**: Store embeddings, metadata, and cropped-image artifacts outside the board in cloud services; the board keeps only a local spool/cache for retry and resilience.
- **Reason**: Persistent vector storage on the board competes directly with inference resources, complicates backups, and does not scale with growing track history. Off-board storage allows larger retention windows, richer metadata filtering, and easier recovery if the board fails.
- **Alternatives considered**: Self-hosting vector DB on the board (simple but fragile and resource-constrained), storing only local files without a searchable vector index (poor retrieval UX)

### Use Outbound-Only Edge Connectivity
- **Decision**: The board pushes events to a public backend over HTTPS; Web UI users never call the board directly.
- **Reason**: This reduces attack surface, avoids exposing the board as a public application server, and makes the architecture easier to scale to multiple boards and cameras. Existing tunnel access remains useful for admin/debug but should not be the primary product path.
- **Alternatives considered**: Direct user-to-board access over tunnel/public endpoint (weaker security, poorer scalability), hosting the entire application stack on the board (resource and operability risk)

### Prototype Platform Choice: Supabase + Vercel + External Text Service
- **Decision**: Use Supabase for metadata, `pgvector`, storage, and auth; Vercel for the Web UI; a separate text embedding service for query encoding.
- **Reason**: This minimizes integration overhead in the first end-to-end prototype while preserving a clean separation of concerns. Supabase makes metadata filters and vector search easy to combine, and Vercel is a fast path for frontend deployment.
- **Alternatives considered**: Qdrant + separate Postgres from day one (more moving parts), hosting everything on a single VM (less modular), keeping UI on the board (not aligned with edge resource constraints)

## [2026-05-16] RB3-First Modular Demo Framework

### Adapter-Based Demo Pipeline
- **Decision**: Implement the first end-to-end demo as pluggable adapters under `deployment/demo/`, with RB3/QNN as the deployment acceptance path and local fake/ONNX modes as preflight only.
- **Reason**: The model and backend are still evolving, while the vision QNN path is already partially deployable. Adapter boundaries allow the team to connect deployable pieces now, replace fake/local pieces later, and avoid coupling camera, model runtime, spool, backend, and vector search into one monolithic script.
- **Alternatives considered**: A single `retrieve.py` script (faster initially but hard to extend), implementing Supabase/live camera first (more integration risk before board runtime is stable), local-only demo (misleading because the target is RB3).
