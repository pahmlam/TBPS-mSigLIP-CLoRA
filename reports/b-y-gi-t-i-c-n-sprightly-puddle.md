# Plan: Rewrite Chapter 6 — Conclusions and Future Work

## Context

`Chapter/6_Conclusions.tex` still reflects an earlier state of the thesis and conflicts with the now-finished
work and the updated intro/abstract:
- It calls deployment an "evaluation scaffold" with results "left as Chapter~5 placeholders" — but deployment
  is **done** (board both-INT8 T2I R@1 `50.35`, passes the `≥50` gate).
- Its largest Future-Work subsection is **Noise-Robust Circle Loss** (Ideas D/A/C with FNM/RDE/GMM formulas) —
  the user wants this detail removed (noise handling is now Part-1 paper material, not a thesis thrust).
- It frames every result as a **delta over TBPS-mSigLIP** ("+2.58%", "+12.57%", "+1.09%", etc.). The user
  considers mSigLIP-CLoRA a standalone method; repeatedly anchoring to TBPS-mSigLIP reads as derivative.

Goal: rewrite the chapter so it (1) reports deployment as a completed contribution, (2) drops the Noise-Robust
Circle Loss detail, (3) presents results as standalone absolute numbers without TBPS-mSigLIP comparisons, and
(4) adds a future-work direction about building a **complete system to bring the model into a real-world
application product**.

## User constraints (verbatim intent)
- Remove the Noise-Robust Circle Loss detail.
- Add: future work will build a complete system to deploy the model into a real-world application product.
- Do **not** compare against TBPS-mSigLIP — present the method as its own, with absolute numbers.

## Numbers to use (already verified; no TBPS-mSigLIP deltas)
- Training (absolute R@1): VN3K best-seed `52.28`, multi-seed `51.52 ± 0.68`; PRW-TPS-CN `59.35`;
  10% CUHK-PEDES `57.10`; English CUHK-PEDES `71.85`.
- LoRA: 376M → 5.9M trainable (98.4%), batch 8→24; matched-batch full-FT control `49.18` vs LoRA `49.90`
  (internal control — fine to keep, not a TBPS-mSigLIP comparison).
- Circle instability (internal ablation, keep): `α5=0.2 → 49.83` below LoRA-only `49.90`.
- Deployment: board both-INT8 T2I R@1 `50.35` (I2T `54.20`), `≥50` gate met, FP32 sanity `52.40`,
  drop `-1.93` vs paper; learned mean-preserving rotation + W8A8 QAT + host/accelerator split text encoder;
  vision `32.54 ms/image`, split-text `7.87 ms/query`; on-disk model ≈`370 MB`.

## Edits in `Chapter/6_Conclusions.tex`

### Summary
- Para 1: reframe deployment from "methodology and evaluation scaffold" → a **delivered** all-INT8 on-device
  system; present training (mSigLIP-CLoRA) and deployment as the two primary contributions.
- Para 2 (Circle instability): keep the instability characterization (real Ch3/Ch4 result) but delete the
  clause "motivates the noise-aware future-work directions discussed below".
- Key-findings bullet 3 (SOTA): drop all "over the TBPS-mSigLIP reference" deltas; restate as **absolute**
  accuracies (VN3K `52.28`, multi-seed `51.52±0.68`, PRW `59.35`, 10% CUHK `57.10`, English `71.85`),
  keeping the qualitative claim that the benefit is largest in low-resource settings.
- Key-findings bullet 4 (deployment): rewrite to the finished result — both encoders all-INT8 on RB3/HTP v68,
  T2I R@1 `50.35` above the `≥50` gate, enabled by learned rotation + split-text; reference Chapter~5.

### Limitations
- Remove the **"No explicit noise handling in Circle Loss"** bullet (its only role was to set up the removed
  noise future-work; noise is now paper material).
- Replace the **"Deployment pipeline not yet finalized"** bullet (now false) with a *real* remaining limitation,
  e.g. deployment validated on VN3K only / single board / vision latency dominates the per-query cost.
- Keep the other limitations (aspect ratio, LoRA plasticity, batch size).

### Future Work
- **Delete** `\subsection{Noise-Robust Circle Loss}` entirely (Ideas D/A/C + formulas + the workspace.ipynb
  validation paragraph).
- **Rewrite** `\subsection{Completing the Edge Deployment Pipeline}` → `\subsection{A Complete On-Device
  Application System}`: since the pipeline is done, the next step is a full product — live video pipeline
  (decode → detect → crop → mSigLIP-CLoRA encode → retrieval) on-device, a gallery indexing/search service,
  a user-facing query interface, and field validation for real surveillance/forensic use. Fold the existing
  GStreamer-integration idea into this.
- Keep **Richer Training Dynamics**, **Advanced PEFT**, **Extending to Video-Based Retrieval** (no TBPS-mSigLIP,
  no noise detail). Light touch only.
- Update the intro sentence "five strategic directions" to the new count (4).

### Closing Remarks
- Rewrite to state both contributions delivered: SOTA Vietnamese accuracy (`52.28`) + the all-INT8 on-device
  system (`50.35`, gate met) — drop "leaving deployment with a concrete evaluation template".
- Delete the "noise-aware extensions ... expected to lift accuracy" sentence; replace the closing forward-look
  with the product-system direction (research → deployed product loop closed).

## Style / constraints
- English; match existing `\section/\subsection/itemize` structure. No code, no job IDs / artifact paths,
  no version codenames (v8/v9). Keep cross-refs to Chapter~3/4/5.
- No new bib/glossary entries (removed-glossary keys FNM/RDE/GMM may lose their only conclusion usage — verify
  they still appear elsewhere or accept they drop from the abbreviation list; do NOT re-add unused entries).

## Verification
1. `latexmk -pdf -interaction=nonstopmode -file-line-error main.tex` exits 0; no undefined `\ref`/`\cite`,
   no missing files.
2. Grep `Chapter/6_Conclusions.tex`: no `TBPS-mSigLIP`, no "Noise-Robust"/"FNM"/"GMM"/`P(\text{FN}`, no
   "placeholder"/"not yet finalized"/"evaluation scaffold", no `v8|v9|job|artifacts/`.
3. Re-check glossary: confirm FNM/RDE/GMM usage counts after the edit; if any drop to 0 in the whole thesis,
   note it (may warrant pruning later) but do not silently break the abbreviation list.
4. Proofread: deployment presented as done; results stated as absolute standalone numbers; future work centers
   on the complete application system; no dangling references to the removed noise subsection.
