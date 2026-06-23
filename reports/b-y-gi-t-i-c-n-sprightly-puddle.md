# Plan: Apply reviewer fixes to Chapter 3 (Methodology)

## Context

A reviewer flagged technical and rigor issues in `Chapter/3_Methodology.tex`. My assessment confirmed most are
valid. The user asked to implement them. All edits are in the single file `Chapter/3_Methodology.tex` unless
noted. Two are correctness-critical (#2, #8); the rest are rigor / precision / presentation. Where the
reviewer's suggested *fix* would diverge from the actual implementation (#3), the edit follows the code, not the
suggestion.

## Edits (all in `Chapter/3_Methodology.tex`)

1. **#2 N-ITC gradient (CRITICAL)** — lines ~228–236 + figure caption ~243.
   - Change the equation `∂L_{N-ITC}/∂s ∝ σ(s)` → `∝ γ·σ(γs − c)` (true negative-pair gradient; the bias `c`
     and scale `γ` were dropped).
   - Replace the `σ(0.8)≈0.69` vs `σ(0.1)≈0.52` "≈1.3×" paragraph (mathematically misleading; with `γ=128`
     the logit saturates) with the reframed argument: the gradient depends on the effective logit `γs − c`;
     the *pairwise* sigmoid scores each negative independently and imposes **no softmax-style relative
     competition among negatives**, so in fine-grained TBPS an auxiliary metric-learning term is needed to
     concentrate optimization on the most confusing local neighborhoods.
   - Fix the figure caption clause "Sigmoid allocates comparatively uniform gradients" → an accurate clause
     (pairwise sigmoid lacks the adaptive amplification / inter-negative competition).

2. **#3 Circle Loss P/N (follow the code, not the reviewer's two-branch suggestion)** — line 197.
   - Replace "For each query, let \(\mathcal P\) be the set of positive image--text pairs and \(\mathcal N\)
     … in the mini-batch" with an implementation-faithful global definition: \(\mathcal P\) = all matched
     (same-identity) image--text pairs in the mini-batch; \(\mathcal N\) = all mismatched pairs; the single
     global log-sum-exp couples both retrieval directions. Drop "for each query" (it conflicts with the global
     double sum). **Keep the existing formula unchanged** — it already matches
     `compute_cross_modal_circle` (single `softplus(logsumexp_P + logsumexp_N)`). Do **not** adopt the
     per-anchor `½(L^{T→I}+L^{I→T})` form, which would misrepresent the implementation.

3. **#7 LayerNorm rotation qualifier** — lines ~451–454.
   - Qualify the commute identity to the non-affine core: "the non-affine LayerNorm normalization core
     (mean-subtraction and variance scaling) commutes with the rotation: \(\mathrm{LN}_{\text{core}}(Qx)=
     Q\,\mathrm{LN}_{\text{core}}(x)\)", and note the per-channel affine \(\gamma,\beta\) are handled by the
     folding described immediately below (line ~476). Keeps the substance, closes the math gap.

4. **#8 Weight quantization (CRITICAL gap)** — W8A8 subsection ~604–607 (and/or end of QAT ~591).
   - Add 2–3 sentences, grounded in `deployment/docs/w8a8_qat_rotated.md`: the dominant per-tensor INT8 error
     in this encoder comes from **activations, not weights**, so QAT simulates the deploy-faithful per-tensor
     *activation* quantizer (STE + EMA) and adapts the rotated weight **values** to tolerate it; the weights
     themselves are quantized to INT8 statically at the final W8A8 step with the fixed calibration set. Because
     LoRA is merged before any quantization, the merged dense weights are quantized as ordinary base weights
     (no adapter tensors). Do **not** assert a weight per-channel/per-tensor granularity (not documented) —
     state only what is sourced.

5. **#6a deployment "both encoders on-device"** — line 339.
   - Reword to: run the vision encoder and the compute-intensive Transformer blocks of the text encoder on the
     RB3 Gen2 accelerator, while the multilingual token-embedding lookup stays on the host CPU.

6. **#6b "no memory overhead"** — line ~663.
   - "the split adds no memory overhead" → "adds no additional model-weight footprint, although it passes a
     small boundary tensor (the embedding vectors) from the host to the accelerator."

7. **#5 LoRA threefold-batch claim** — LoRA table/text (~line 68/73).
   - Keep the LoRA numbers and target modules (Q/K/V/O is consistent with the 52.28 headline run). Add a
     cross-reference for the "threefold batch size" claim to the Chapter~4 efficiency table
     (`tab:lora_efficiency`), since the evidence exists there — no softening needed.
   - (Flag to user afterwards, not an edit: confirm the accepted ICIP config is attention-only vs attn+FFN.)

8. **#11 style** —
   - Line 250: "We adopt a \textbf{Hybrid N-ITC + Cross-modal Circle Loss} strategy." → "This thesis adopts a
     \textbf{hybrid N-ITC + Cross-modal Circle Loss} strategy."
   - Line 346: "all-INT8 W8A8 execution pattern" → "W8A8 integer execution pattern".

9. **#10 algorithm boxes** — add two `algorithm2e` boxes (`[ruled,vlined]` already loaded in `main.tex`).
   - **Algorithm 1 — mSigLIP-CLoRA training**, at the end of Part~I (before `\section{Part II}`, ~line 336):
     load mSigLIP → insert LoRA → forward both encoders → compute base loss (N-ITC + MVS + C-ITC + SS) →
     update curriculum weight \(\alpha_5(t)\) → if active, add Circle Loss (primary + MVS branch) → backprop /
     AdamW update.
   - **Algorithm 2 — edge deployment pipeline**, in Part~II after the Deployment Pipeline overview (~line 380):
     merge LoRA → fused opset-20 ONNX export → learn mean-preserving rotation → fold rotation into weights →
     QAT teacher–student distillation → W8A8 quantize → QNN compile/link → board validation gates.
   - Use `\begin{algorithm}[H]`, `\SetAlgoLined`, `\KwIn`/`\KwOut`, `\For`/`\If`; concise, no code.

## Constraints
- English; match existing notation and section style. No new packages (algorithm2e present). No job IDs /
  artifact paths / version codenames. Numbers only from sourced material (Ch3/4, `w8a8_qat_rotated.md`).

## Verification
1. `latexmk -pdf -interaction=nonstopmode -file-line-error main.tex` exits 0; no new undefined `\ref`/`\cite`,
   no algorithm2e errors. (Note: the 5 pre-existing Chapter-2 citation warnings are unrelated and out of scope.)
2. Grep `3_Methodology.tex`: gradient eq now has `\gamma`/`c` and no `1.3` ratio; `non-affine` present near the
   LN identity; weight-quant sentences present; "This thesis adopts a hybrid"; two `\begin{algorithm}` boxes.
3. Confirm the Circle Loss equation body is unchanged (only the surrounding P/N prose changed).
4. Proofread the two algorithm boxes render and read correctly.
