# Plan: Chapter 3 Part II — Edge Deployment Method (final vision + text)

## Context

Chapter 2 (literature review) now carries the foundational deployment theory. The next step is
**Chapter 3, Part II "Edge Deployment Method"** in `Chapter/3_Methodology.tex`. A Part II already
exists (lines ~336–582) but does **not** present the final method:

- it shows only **random** rotation (the √(2 ln d) bound), not the **learned** rotation that is the
  actual final vision method;
- it carries **staging / version / ongoing-work language** ("primary diagnostic branch",
  "text-encoder … remains ongoing work", "Early experiments … Later QAT variants …");
- it has **no text-encoder method** (finite mask, link-safe float32 mask, split-encoder).

Goal: rewrite Part II so it reads as one clean final method — develop it on the **vision encoder
first**, then show the **text encoder adapting and refining** it — with no version labels, no
results/numbers, no job IDs, no code, and ~6 figure placeholders.

Confirmed decisions: **restructure vision + add text**; **reference Ch2 for general theory and keep
Ch3 on the concrete construction**; **rich (~6) figure placeholders**.

Hard constraints (from the user):
- Prose only — no code, no `artifacts/...` paths, no job IDs, no AI-Hub specifics.
- Present the single final method; **never** write "this is v8/v9", no version comparison.
- No results / metrics / latency / footprint (those live in Chapter 5).
- Narrative: vision method developed first → text adapts and is refined further.

## Source of truth

All method content comes from `deployment/docs/w8a8_qat_rotated.md` (already fully read). Use only
the rows classified as **method** in the Part A table of the prior planning round (LoRA merge,
fused export, mean-preserving construction, learned rotation objective + Cayley, QAT distillation,
quantize/link, text deltas: finite mask, link-safe float32 mask, split-encoder).

## New Part II skeleton (in `Chapter/3_Methodology.tex`)

Keep `\section{Part II: Edge Deployment Method}`. Reorganize its subsections into:

**Framing**
1. *Deployment Objective and Hardware Contract* — reframe the existing scope subsection: the final
   method deploys **both** encoders as all-INT8 W8A8 on RB3/HTP v68. Remove all staging/diagnostic/
   ongoing-work wording. State the integer-only contract briefly and refer to Ch2
   (`\ref{sec:npu_int_exec}`, `\ref{sec:edge_deployment_lit}`) instead of re-deriving it.
2. *Deployment Pipeline Overview* — representation-preserving transforms (LoRA merge → fused export
   → mean-preserving learned rotation) + one weight-adapting step (QAT distillation) → W8A8
   quantize/compile/link. **FIG 1: pipeline block diagram.** Drop the `artifacts/` sentence.

**Vision Encoder Deployment (developed first)**
3. *LoRA Merge* — keep `W_merged = W_0 + (α/r)BA`; this is the FP32 reference model.
4. *Hardware-Friendly Fused Export* — keep fused GELU/LayerNorm rationale but shorten and refer to
   Ch2's opset/fusion subsection; keep the static FP32-fidelity gate sentence.
5. *Mean-Preserving Rotation* — keep the construction `Q = U·blockdiag(1,R_c)·Uᵀ`, `Q1=1`,
   `LN(Qx)=Q·LN(x)`, and the fold equations (affine fold; writer `QW`; reader `WQᵀ`). Refer to Ch2
   for incoherence μ and the `s²/12` error rather than repeating the derivation.
   **FIG 2: writer/reader rotation-folding schematic.**
6. *Learned Rotation* (**new — the core upgrade**): present `R_c` as learned. Objective
   `min_Q Σ_sites E_calib[(max|aQᵀ|)²]` s.t. `QQᵀ=I, Q1=1`; explain it targets the quantity that
   sets the per-tensor scale, and why **max-abs² not quant-MSE** (STE detaches the rounding
   gradient — refer to Ch2 `\ref{sec:qat}`). Cayley parametrization
   `Q = U·blockdiag(1,Cayley(S))·Uᵀ`, `Cayley(S)=(I−S)(I+S)⁻¹`, `S` skew-symmetric → orthogonal and
   mean-preserving at every step. Offline calibration-only procedure (collect activations at
   rotation sites, optimize, fold). No budgets/step counts/numbers. Cite `spinquant`, `quarot`,
   `quip`, `slicegpt`. **FIG 3: learned-rotation calibration loop.**
7. *W8A8 QAT with Teacher–Student Distillation* — keep teacher (frozen rotated FP32) / student
   (rotated + fake-quant); keep the distillation loss (cosine+MSE fake path + clean-consistency
   path) and its weights as method config. **De-version**: present the final fake-quant coverage
   (residual + GELU + pooling head + all linear outputs + attention matmuls) as the method, deleting
   "early/later variants". Reword the "In code, the forward pass…" sentence to remove the code
   reference; refer to Ch2 for STE/EMA generalities. **FIG 4: QAT teacher–student + fake-quant
   coverage on an encoder block.**
8. *W8A8 Quantization and QNN Compilation* — rotated+QAT model → static W8A8 quantize → compile +
   link to a context binary with quantized I/O; one sentence on why it links on v68 (no float I/O,
   no internal float, no A16, no decomposed GELU). No job IDs / artifact paths.

**Text Encoder Adaptation (adapt + refine)** — new transition + subsections
9. *Transferring the Recipe to the Text Encoder* — the same four transforms carry over; only the
   residual writer/reader boundary changes (writers: token + position embedding rows, `out_proj`,
   `fc2`; readers: q/k/v, `fc1`, head after `final_layer_norm`).
10. *Finite Attention Mask for Quantized Softmax* — the `-FLT_MAX` padding sentinel breaks per-tensor
    quantization of `scores+mask`; replace it with a finite negative constant (e.g. −32) that keeps
    `exp(−c)≈0`, preserving softmax semantics while bounding the quantized range. Refer to Ch2
    `\ref{sec:transformer_quant_hazards}`. **FIG 5: mask handling (sentinel → finite + link-safe).**
11. *Link-Safe Mask Representation* — export `attention_mask` as float32 0/1 and rewrite the additive
    mask algebraically as `(1−mask)·(−c)` to avoid an internal float-cast island that HTP v68
    rejects. Exact equivalence for binary masks.
12. *Host/Accelerator Split Encoder* (the refinement that makes text deployable) — the large
    runtime-indexed token-embedding gather is a memory read HTP v68 does not honor inside the
    context binary; split the graph at the embedding boundary so the host CPU performs the lookup
    (in the rotated space, since Q is folded into the embedding table) and feeds `inputs_embeds` to
    the on-NPU transformer, which keeps all 12 layers of compute. Shared DRAM → no extra memory.
    **FIG 6: split-encoder topology.**

**Validation (method-level, no numbers)**
13. *Deployment Validation Protocol* — construction-time gates only: FP32 invariance after each
    representation-preserving transform; static FP32↔ONNX fidelity; QDQ proxy before board; board
    fidelity after link; three isolation views (vision-only, text-only, end-to-end) for diagnosis;
    Rank-1 retrieval is the decisive acceptance metric. Defer all numbers, latency, and footprint to
    Chapter 5.

## Figures (6 placeholders, existing project style + `% PLACEHOLDER`)

`deploy_pipeline.png`, `rotation_fold.png`, `learned_rotation.png`, `qat_teacher_student.png`,
`text_mask_linksafe.png`, `text_split_encoder.png` — generated as gray placeholder PNGs (via the
project venv PIL) so the thesis compiles; user overwrites with real diagrams later. Captions
descriptive; `\label{fig:...}`; widths matching existing figures (0.8–1.0\linewidth).

## Supporting files

- `reference.bib`: **no new entries needed** — all method citations (`slicegpt quarot quip spinquant
  jacob2018 ste hinton2015distilling`) already exist. Verify before assuming.
- `glossary.tex`: acronyms already present from the Ch2 pass (W8A8, QAT, STE, QDQ, FFN, NPU, HTP…).
- Match existing Ch3 style: `\subsection/\subsubsection`, `\noindent`, `equation` env,
  `\operatorname{}`, `booktabs` tables, inline-defined acronyms (no `\gls`).

## Execution order

1. Re-verify the method citation keys exist in `reference.bib`.
2. Edit `Chapter/3_Methodology.tex` Part II: reframe subsections 1–2; light-edit 3–4; revise rotation
   (5) and insert learned-rotation (6); de-version QAT (7) and revise compilation (8); insert the
   Text Encoder Adaptation block (9–12); rewrite validation (13). Insert 6 figure placeholders.
3. Generate the 6 placeholder PNGs under `Figure/`.
4. Compile `main.tex` (latexmk + bibtex); fix any undefined `\cite`/`\ref`.

## Verification

- `latexmk -pdf -interaction=nonstopmode main.tex` exits 0; `main.log` has no undefined
  citation/reference and no missing-figure errors.
- Leak scan over `Chapter/3_Methodology.tex`: no `v8|v9|R@1|50.\d|job |ms/image|FPS|artifacts/`,
  no "ongoing"/"diagnostic branch"/"Early experiments"/"Later … variants".
- Proofread: Part II reads vision-first then text-adapt; the final method is stated without versions;
  no results; Ch2 cross-references resolve.
