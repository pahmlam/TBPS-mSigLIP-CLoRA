# Thesis Completion Checklist — mSigLIP-CLoRA

**Project:** Graduation Thesis — Phạm Tùng Lâm (20224321), HUST
**Thesis directory:** `knowledge/PhamTungLam_20224321/`
**Current title:** *Hard Negative-Aware Optimization and Edge Deployment for Multilingual Text-Based Person Search*
**Last updated:** 2026-04-15

---

## Status Legend

| Symbol | Meaning |
|--------|---------|
| 🔴 P0 | Blocker — must fix before submission |
| 🟠 P1 | High priority — ships the thesis reviewer-ready |
| 🔴 P1.5 | Research contribution decision + implementation |
| 🟡 P2 | Depends on deployment work completing |
| 🟡 P2.5 | Optional research extension (schedule variants) |
| 🟢 P3 | Figures and polish |
| 🟢 P4 | Content refinements |
| 🟢 P5 | Documentation hygiene (project rules) |
| 🔵 P6 | Defense preparation |

---

## 🔴 P0 — Critical Blockers

Must be fixed before the thesis can compile/submit cleanly.

- [ ] **Write acknowledgments** (`Chapter/0_2_Acknowledgment.tex`)
  - Currently holds Vietnamese template placeholder (100-150 words target).
  - Uncomment line 141 of `main.tex`: `% \subfile{Chapter/0_2_Acknowledgment.tex}` → `\subfile{Chapter/0_2_Acknowledgment.tex}`
  - Suggested content: thank supervisor (Assoc. Prof. Le Thi Lan), family, lab members, Communication Engineering department.
- [ ] **Remove orphaned `Chapter/7_Reference.tex`**
  - File contains only `\begin{document}\end{document}`.
  - In `main.tex` line 233, delete the `\subfile{Chapter/7_Reference}` line and the `\label{chapter:reference}` above it.
  - Bibliography still renders via `\printbibliography` on line 238.
- [ ] **Fix filename case sensitivity** — pick one:
  - Option A: rename `Chapter/4_Theoretical_Analysis.tex` → `Chapter/4_Theoretical_analysis.tex` (to match `main.tex` reference).
  - Option B: edit `main.tex` line 216 `Chapter/4_Theoretical_analysis` → `Chapter/4_Theoretical_Analysis`.
  - Currently works on macOS (case-insensitive HFS+) but breaks on Linux builds / Overleaf.
- [ ] **Delete orphaned `Chapter/6_Conclusions.tex`**
  - Superseded by `Chapter/7_Conclusions.tex` after chapter restructuring.
  - No longer referenced in `main.tex`. Safe to delete.
- [ ] **Resolve Circle Loss margin discrepancy**
  - Thesis uses `m=0.25` consistently (Ch 3 Table `tab:cir_config`, Ch 4, Ch 5 figure caption, Appendix A code).
  - `CLAUDE.md` line 85 states `m=0.35`.
  - **Action:** verify from `config/loss/cir_msiglip.yaml` which is the actually-trained value, then update either thesis or CLAUDE.md to match. Do **not** fabricate.
- [ ] **End-to-end LaTeX compile test**
  - Install TeX distribution (MacTeX or BasicTeX) — see `reports/curious-booping-ullman.md`.
  - Run:
    ```bash
    cd knowledge/PhamTungLam_20224321
    pdflatex main && bibtex main && pdflatex main && pdflatex main
    ```
  - Verify: no "??" references, no missing figures, no unresolved citations.

---

## 🟠 P1 — Citation & Glossary Audit

Fast to do, high reviewer impact.

- [ ] **Add `\cite{}` commands for new `reference.bib` entries**
  The following were added but may not be cited in the body text:
  - [ ] `rde2024cvpr` (RDE) → Chapter 2 §"Noise-Aware Contrastive Learning", Chapter 7 Future Work
  - [ ] `fnm2026aaai` (FNM) → same locations
  - [ ] `liu2024dora` (DoRA) → Chapter 7 Future Work §"Advanced PEFT Methods"
  - [ ] `pfeiffer2021adapterfusion` (AdapterFusion) → Chapter 7 Future Work
  - [ ] `onnx2019` → Chapter 2 §"Edge Deployment", Chapter 6 §6.5
  - [ ] `onnxruntime` → Chapter 6 §6.6, Appendix B
  - [ ] `qualcomm_qcs6490`, `qualcomm_rb3gen2` → Chapter 6 §6.1
  - [ ] `qualcomm_aihub` → Chapter 6 §6.7, Appendix B
  - [ ] `qualcomm_aied`, `qualcomm_snpe` → Chapter 6 §6.2, §6.7
  - [ ] `sandler2018mobilenetv2`, `he2016resnet`, `tan2019efficientnet` → Chapter 6 §6.6 proxy benchmarks
  - [ ] `gmm_noisy_labels` (DivideMix) → Chapter 7 §"Noise-Robust Circle Loss" (Idea C context)
- [ ] **Audit glossary entries for actual usage**
  - `main.tex` uses `\glsaddall` (line 176) which forces every glossary entry into the abbreviations list regardless of usage.
  - Decision: either wrap first use of each acronym in the text with `\gls{...}` (e.g., `\gls{ONNX}`) to ensure it appears naturally, OR remove `\glsaddall` and add only used entries.
  - New entries to verify: `ONNX`, `SNPE`, `QNN`, `DSP`, `HTP`, `DLC`, `VRAM`, `PEFT`, `MTBPS`, `FNM`, `RDE`, `GMM`, `EMA`, `mINP`, `NPU`, `FP16`, `FP32`, `INT8`, `ARM`, `SoC`, `NEON`, `SIMD`, `DoRA`, `TAL`, `TSE`, `BGE`, `MoC`.

---

## 🔴 P1.5 — Noise Handling Implementation (COMMITTED: Idea C only)

**Decision (2026-04-15):** Noise handling is a **thesis contribution**, and the user has chosen to focus **exclusively on Idea C** (Unified Noise-Aware Circle Loss / NACIR). Ideas D and A are not pursued — they remain in Chapter 7 Future Work as simpler baselines the community could explore.

**Risk:** single-track commitment means no Plan B. If NACIR fails workspace validation, must fall back to Idea D/A; if it fails full training, Chapter 5 §5.7 has no contribution.

**Existing plan:** `reports/golden-fluttering-sphinx.md` — 9 files, 9-step plan, extended curriculum (FN detection at epoch 11, FP detection at epoch 15).

### Execution order (corrected to respect Critical Workflow Rule)

The original plan in `golden-fluttering-sphinx.md` lists workspace validation as Step 7, after production code (Steps 1-6). **This is backwards** per the project's Critical Workflow Rule in `CLAUDE.md`: *"Training costs hours. Always validate ideas in `workspace.ipynb` first."* The corrected order is:

#### Phase 1 — Workspace validation (1-2 days, cheap go/no-go)

- [ ] **Extract or load embeddings** `W` from the best seed 2400 checkpoint per workspace conventions (`W['image_feats']`, `W['text_feats']`, `W['image_pids']`, etc.).
- [ ] **Prototype `compute_noise_aware_circle()` in notebook cells** — no production code yet.
  - Pure function, takes pre-computed `fn_stats` dict + `clean_weights` tensor as args.
  - Include `_bayesian_fn_prob()` helper.
- [ ] **Section 3 Loss Playground** — sanity checks:
  - NACIR with `fn_stats=None, clean_weights=None` must exactly match vanilla Circle Loss (regression test).
  - NACIR with simulated Gaussian `fn_stats` from test-set — verify finite, O(1) scale.
  - Parameter sweep: `fn_prior ∈ {0.001, 0.01, 0.05}`, `epsilon_n ∈ {0.05, 0.1, 0.2}`.
- [ ] **Section 4 Gradient Analysis (CRITICAL)** — the go/no-go criterion:
  - Compare gradient energy on top-10% hard negatives: NACIR vs vanilla Circle Loss.
  - Verify NACIR **suppresses** gradient on overlap-zone negatives (likely FN) — should be < vanilla.
  - Verify NACIR **preserves** gradient on clear hard negatives (true hard) — should be ≈ vanilla.
  - If both conditions hold → GO. Otherwise → abort, fall back to Idea D or A.
- [ ] **Section 5 Embedding Visualization** — plot `P_fn(s)` curve overlaid on pos/neg similarity histograms.
- [ ] **GMM sanity check** — fit 2-component 1D GMM on simulated per-sample losses (Gaussian mixture); verify EM converges and `gmm_min_separation > 1.0` is achievable.
- [ ] **Section 8 Mini Fine-Tune** — apply NACIR for 1 epoch on a VN3K subset; stop-go: R@1 not worse than baseline by more than 1% absolute.

**Phase 1 stop-go (all must pass):**
1. NACIR without noise-aware components matches vanilla Circle Loss exactly (regression).
2. Gradient energy on overlap-zone negatives is reduced > 20% vs vanilla.
3. Gradient energy on clear hard negatives is preserved within 10% of vanilla.
4. GMM EM converges on synthetic data with `gmm_min_separation > 1.0`.
5. Mini fine-tune does not regress R@1 > 1% absolute.

If any condition fails → stop, reassess (consider falling back to Idea D/A).

#### Phase 2 — Production code (1 week, after Phase 1 passes)

Follow `golden-fluttering-sphinx.md` Steps 1-6 in order:

- [ ] **Step 1:** NEW `model/noise_aware.py` — `NoiseAwareCircleState` class with registered buffers.
- [ ] **Step 2:** EDIT `model/objectives.py` — add `compute_noise_aware_circle()` + `_bayesian_fn_prob()`.
- [ ] **Step 3:** EDIT `config/loss/cir_msiglip.yaml` — add NACIR section with curriculum hyperparameters.
- [ ] **Step 4:** EDIT `model/tbps.py` — wire NACIR in `__init__` + `forward()`, add guard so existing CIR skips when NACIR active.
- [ ] **Step 5:** EDIT `lightning_models.py` — pass `num_train_samples`, add `on_train_epoch_end` GMM refit hook.
- [ ] **Step 6:** EDIT `trainer.py` — pass `len(dm.train_set)` to `LitTBPS`.

**Phase 2 integration tests (all must pass before full training):**
1. `NACIR: false` → existing behavior bit-identical (regression check via hash of loss values on fixed seed).
2. `NACIR: true` with `fn_enable_epoch: 999, fp_enable_epoch: 999` → NACIR acts as vanilla Circle Loss.
3. 5-epoch sanity run on VN3K → no crashes, W&B logs show `nacir_loss`, `fn_prob_mean`, `clean_weight_mean`, `gmm_separation`.
4. Checkpoint save/load round-trip → `NoiseAwareCircleState` buffers preserved.

#### Phase 3 — Full training (1-2 weeks, after Phase 2 passes)

- [ ] **Seed 2400 first** — match baseline comparison. Full 60-epoch run with extended curriculum.
- [ ] **Compare to baseline 52.28%** — diagnostic: did FN detection activate at epoch 11? Did GMM fit at epoch 15? Did `gmm_separation > 1.0`?
- [ ] **If R@1 > 52.28% by > std (0.68%)** → proceed to multi-seed (seeds 2307 and 2300) for CI.
- [ ] **If R@1 regression** → diagnose via W&B curves; consider ablating FN-only vs FP-only vs full NACIR.

#### Phase 4 — Thesis integration (3-5 days)

- [ ] **Promote to Chapter 5** — new §5.7 "Unified Noise-Aware Circle Loss (NACIR)".
  - Mathematical formulation (from §4.3 theoretical motivation).
  - Bayesian FN detection mechanism.
  - GMM-based FP detection mechanism.
  - Extended curriculum schedule figure.
  - Single-seed result vs baseline (ablation table).
  - Multi-seed CI (if Phase 3 produced positive result).
  - Ablation: FN-only, FP-only, full NACIR (if time permits).
- [ ] **Update Chapter 4 §4.3 closing paragraph** — connect instability analysis to NACIR solution.
- [ ] **Update Abstract** — add sentence: "To address this instability, we introduce NACIR, a noise-aware variant that scales α_n by Bayesian false-negative suppression and α_p by GMM-based false-positive detection, achieving R@1 = [PLACEHOLDER until trained]."
- [ ] **Update Chapter 1 contributions** — add 5th contribution on NACIR.
- [ ] **Update Chapter 7 Future Work** — remove NACIR from Future Work (it moved to Chapter 5); add "extending NACIR to Idea D / A as simpler variants" as follow-up.
- [ ] **Add NACIR to glossary** — acronym definition.
- [ ] **Update `docs/knowledge.md` in Vietnamese** — per project rule.
- [ ] **Changelog entry** in `changelog/training/changelog.md`.

### Time budget (best case)

| Phase | Days | Cumulative |
|-------|------|------------|
| 1 — Workspace validation | 1-2 | 2 |
| 2 — Production code + tests | 5-7 | 9 |
| 3 — Full training + multi-seed | 10-14 | 23 |
| 4 — Thesis integration | 3-5 | 28 |

**Total: ~4 weeks best case; 6 weeks if multi-seed is added.**

### Minimum ship — Idea D (Distribution Separation)

*Low effort (~10 lines), medium impact. Recommended as the baseline research addition.*

- [ ] **Implement** `compute_distribution_separation(sim_pos, sim_neg)` in `model/objectives.py`
  - Formula: `L_sep = -(μ_p − μ_n) / (σ_p + σ_n + ε)` (minimize negative ratio = maximize separation).
- [ ] **Route** in `model/tbps.py` forward pass.
- [ ] **Add config flag** in `config/loss/cir_msiglip.yaml`:
  ```yaml
  DIST_SEP: true
  dist_sep_weight: 0.05   # to be tuned
  ```
- [ ] **Validate in `workspace.ipynb`**:
  - Section 3 (Loss Playground): compute loss value on extracted embeddings, confirm finite and reasonable scale.
  - Section 4 (Gradient Analysis): verify gradient energy concentrates on mid-distribution samples.
  - Section 6 (Retrieval Metrics): re-rank with regularized embeddings, confirm no R@1 regression.
- [ ] **Train one seed** (seed 2400 for consistency with best baseline), compare to 52.28% R@1.
- [ ] **Update thesis**:
  - Add Chapter 5 §5.7 "Noise-Aware Extensions — Distribution Separation Regularizer".
  - Single table comparing baseline (52.28%) vs +DistSep across 5 metrics.
  - Add one paragraph to Chapter 4 §4.6 connecting the theoretical instability analysis (4.3) to the practical fix.

### Stretch goal — Idea A (FNM-Lite for Circle Loss)

*Medium effort (~30 lines + EMA buffers), high impact. Closes the theoretical loop.*

- [ ] **Implement** Bayesian FN detection in `model/objectives.py`:
  - Maintain EMA mean/std of positive and negative similarity distributions (as training state).
  - Compute `P(FN | s) = p · f_+(s) / (p · f_+(s) + (1-p) · f_−(s))` per pair.
  - Modify `α_n ← α_n · max(1 − P(FN|s), ε_n)` with floor `ε_n = 0.1`.
- [ ] **Add state tracking** in `model/tbps.py` `__init__`:
  - Register buffers for `ema_mu_pos`, `ema_sigma_pos`, `ema_mu_neg`, `ema_sigma_neg`.
  - Add EMA update hook in `training_step`.
- [ ] **Extended curriculum**: activate FN detection at epoch 11 (after baseline EMA stabilizes).
- [ ] **Config flags** in `config/loss/cir_msiglip.yaml`:
  ```yaml
  FNM_LITE: true
  fnm_ema_decay: 0.99
  fnm_floor_eps_n: 0.1
  fnm_activation_epoch: 11
  ```
- [ ] **Validate in `workspace.ipynb`**: confirm `P(FN|s)` distribution matches expectation (skewed toward 0 on clean pairs, >0.3 on visually similar cross-identity pairs).
- [ ] **Train one seed**, compare to both baseline (52.28%) and Idea D result.
- [ ] **Update thesis** with Chapter 5 §5.8 or extend §5.7 with FNM-Lite ablation.

### Ambitious — Idea C (Unified Noise-Aware Circle Loss / NACIR)

*High effort, very high impact. Implementation plan already exists at `reports/golden-fluttering-sphinx.md` (9 files, 9-step plan).*

- [ ] **Follow the existing plan** at `reports/golden-fluttering-sphinx.md`:
  1. Create `model/noise_aware.py` with `NoiseAwareCircleState`.
  2. Add `compute_noise_aware_circle()` to `model/objectives.py`.
  3. Config section in `config/loss/cir_msiglip.yaml`.
  4. Wire in `model/tbps.py` `__init__` + `forward()`.
  5. Pass `num_train_samples` through `lightning_models.py`.
  6. Pass `len(dm.train_set)` in `trainer.py`.
  7. Validation cells in `workspace.ipynb`.
  8. Update `docs/knowledge.md` with the Vietnamese documentation.
  9. Changelog entry.
- [ ] **Curriculum**: FN detection epoch 11, FP detection epoch 15.
- [ ] **GMM** per-sample loss for FP detection (2 components, periodic refit every epoch).
- [ ] **Train 3 seeds** for multi-seed confidence (match methodology of baseline).
- [ ] **Update thesis** — lift from Future Work into a full Chapter 5.5 or new Chapter 5 (could also become the dominant contribution). Add to Abstract contributions list.

### Thesis-side updates (regardless of which idea ships)

- [ ] Move successful idea(s) from Chapter 7 → Chapter 5 (new subsection).
- [ ] Update Abstract: add sentence on noise handling result.
- [ ] Update Chapter 1 contributions list: add 5th contribution if results are positive.
- [ ] Update Chapter 4 §4.3: end with "The proposed noise-aware variant in Chapter 5 §X.Y addresses this root cause empirically..."
- [ ] Update glossary: add `NACIR` if Idea C is chosen.

---

## 🟡 P2 — Deployment Placeholders (15 markers in Chapter 6)

Must be replaced with real measurements as deployment pipeline completes. Every value must come from an actual log — no estimation.

### Compilation

- [ ] **Submit vision encoder to Qualcomm AI Hub**
  ```bash
  qai-hub submit-compile-job \
      --model exported_model/vision_onnx/ \
      --device "Dragonwing RB3 Gen 2 Vision Kit" \
      --compile_options "--target_runtime qnn_context_binary" \
      --name "mSigLIP-vision" --wait
  ```
- [ ] **Log invocation** to `deployment/docs/aihub-experiments.md` per project rule (`.claude/rules/aihub-experiments.md`).
- [ ] **Submit text encoder to Qualcomm AI Hub** (same command, swap `vision_onnx` → `text_onnx`, name `mSigLIP-text`).
- [ ] **Retrieve compiled `.bin` files** and transfer to RB3 Gen2.

### On-device measurement

- [ ] **Vision encoder DSP latency** (batch 1, FP16) → replaces `[PLACEHOLDER ~10 ms]` in `tab:expected_latency`.
- [ ] **Text encoder DSP latency** (batch 1, FP16) → replaces `[PLACEHOLDER ~15 ms]`.
- [ ] **End-to-end query latency** (vision + text + dot product) → replaces `[PLACEHOLDER ~25 ms]`.
- [ ] **ONNX Runtime CPU latency for mSigLIP** → replaces `[PLACEHOLDER ~100 ms vision]`, `[~80 ms text]`.
- [ ] **PyTorch CPU latency for mSigLIP** → replaces `[PLACEHOLDER ~300 ms vision]`, `[~200 ms text]`.

### Quantization

- [ ] **Select 256-sample VN3K calibration subset** (stratified by identity).
- [ ] **Run INT8 quantization** via AI Hub (`--quantize_full_type int8`) or `snpe-dlc-quantize`.
- [ ] **Measure INT8 latency** on device.
- [ ] **Validate accuracy preservation**:
  - PyTorch FP32 R@1 on VN3K test (reference): 52.28%.
  - PyTorch FP16 R@1 (sanity check).
  - ONNX FP32 R@1.
  - QNN FP16 R@1 (on device).
  - QNN INT8 R@1 (on device).
  - Acceptance criterion: ≤ 1% absolute drop at each step.
- [ ] **Update `tab:deployment_status`** — flip `[PLACEHOLDER]` → `Complete` for completed rows.

### Section 6.7 status update

- [ ] Replace "In progress" and `[PLACEHOLDER]` markers in §6.7 with measured values.
- [ ] Update §6.8 pipeline status table and numeric summary.

---

## 🟡 P2.5 — Schedule Improvement Experiments (optional)

Mentioned as "ongoing" in the initial brief. Currently only theoretical in Chapter 4 §4.4.

- [ ] **Implement cosine ramp** in `model/tbps.py` `get_circle_weight()`:
  ```python
  # cosine: α(t) = α_max · (1 − cos(π · (t − T_wu) / T_ramp)) / 2
  ```
- [ ] **Implement exponential ramp**: `α(t) = α_max · (1 − exp(−3 · (t − T_wu) / T_ramp))`.
- [ ] **Config flag** for schedule type: `curriculum_shape: linear | cosine | exponential`.
- [ ] **Sensitivity sweep** over `T_warmup ∈ {0, 3, 5, 8, 10}` (fixed linear, vary warm-up).
- [ ] **Compare all variants** at seed 2400 (baseline consistency).
- [ ] **Update thesis**:
  - Add Chapter 5 §5.9 "Curriculum Schedule Ablation".
  - Update Chapter 4 §4.4 last paragraph — replace "future work" with "the systematic sensitivity analysis in Section 5.9 confirms..."

---

## 🟢 P3 — Figures

- [ ] **Deployment pipeline diagram** (`Figure/deployment_pipeline.png`)
  - Currently ASCII text in Chapter 6 §6.2.
  - Tools: TikZ (native LaTeX, reproducible) or draw.io export to PNG.
  - Must show: Lightning ckpt → LoRA merge → FP16 → ONNX → QNN → DSP.
- [ ] **Benchmark latency bar chart** (`Figure/benchmark_latency.png`)
  - Data from `deployment/docs/benchmark-rp.md` §3 (PyTorch CPU vs ONNX Runtime, MobileNetV2/ResNet18/EfficientNet-B0).
  - Reference in Chapter 6 §6.6.
- [ ] **Memory estimation bar chart** (`Figure/memory_estimation.png`)
  - Data from `tab:memory_budget` (FP32=2155MB / FP16=1077MB / INT8=539MB + headroom on 4GB device).
  - Reference in Chapter 6 §6.3.
- [ ] **Review existing figures**:
  - `framework.png`, `framework-2.png`, `framework-3.png` — confirm updated architecture consistency.
  - `geo.png`, `qua.png` — confirm caption matches the best-seed (52.28%) results.
  - `grad.png` — confirm margin labels match `m=0.25` used elsewhere.
  - `clip.png`, `vit.png`, `transformer.png`, `lora.png` — no changes needed, illustrations only.

---

## 🟢 P4 — Content Refinements

- [ ] **Hardcoded table references** — search for literal "Table 1", "Table 2", "Table 4" etc. and convert to `\ref{...}`:
  ```bash
  grep -rn "Table [0-9]" Chapter/
  ```
- [ ] **Hardcoded chapter references** — similarly search for "Chapter 4", "Chapter 5":
  ```bash
  grep -rn "Chapter [0-9]" Chapter/
  ```
  (Most should already use `Chapter~\ref{...}` but some may have slipped in during expansion.)
- [ ] **Overclaiming audit** — per R4-C1 (`knowledge/response.md`):
  ```bash
  grep -rni "universally\|uniformly best\|outperforms all\|strictly better" Chapter/
  ```
  Any hits: replace with "strongest in multilingual low-resource adaptation" or similar scoped claim.
- [ ] **"mSigLIP-CLoRA" capitalization consistency** — single canonical form:
  ```bash
  grep -rn "[Mm][Ss]ig[Ll][Ii][Pp]" Chapter/ | grep -v "mSigLIP-CLoRA\|mSigLIP-CLoRA\|mSigLIP "
  ```
- [ ] **Chapter 1 §1.5 organization** — verify the prose description of each chapter (2-7) matches what that chapter actually contains after restructuring.
- [ ] **Ensure "\cite{}" appears at first use of every reference in the body** (complements P1).

---

## 🟢 P5 — Documentation Hygiene (per project rules)

Per `CLAUDE.md` and `.claude/rules/*`.

- [ ] **Update `docs/knowledge.md` in Vietnamese**
  Topics requiring Vietnamese documentation:
  - Edge deployment pipeline design (LoRA merge → FP16 → ONNX → QNN)
  - Noise handling analysis (FNM, RDE, proposed Ideas A/C/D)
  - Theoretical instability analysis (α₅=0.2 failure mode)
  - 5.9M vs 4.77M scope reconciliation (LoRA-only vs LoRA+SimCLR trainable counts)
  - Any Idea D/A/C that is actually implemented
- [ ] **Append changelog entries**:
  - `changelog/training/changelog.md` — new theoretical sections, new result tables, (if implemented) noise handling
  - `changelog/deployment/changelog.md` — parameter breakdown correction, pipeline documentation updates
- [ ] **Architecture decisions** — document any non-trivial choices in `reports/architecture-decisions.md`:
  - Keeping `5.9M/1.57%` published figure vs `4.77M/1.27%` PEFT-scope figure
  - Tabularx vs tabular format migration (done)
  - Circle Loss integration strategy selection (Strategy 4 Hybrid, Chapter 3 §3.8)

---

## 🔵 P6 — Defense Preparation

Optional but part of "perfect submission."

### Slide deck (~20-25 slides)

- [ ] Title slide + supervisor + date
- [ ] Problem statement (multilingual TBPS, low-resource, edge deployment)
- [ ] Motivation (gradient saturation → weak hard-negative separation)
- [ ] Background: mSigLIP, Circle Loss, LoRA
- [ ] Method overview: Circle Loss + LoRA + curriculum
- [ ] Theoretical analysis: gradient magnitudes, instability at α₅=0.2
- [ ] Results: Vn3K 52.28% table + multi-seed confidence
- [ ] Results: full FT control + 3 languages
- [ ] Ablation: Circle Loss weighting strategies
- [ ] Geometric analysis (similarity distribution figure)
- [ ] Qualitative example (wrong baseline Rank-1, correct Ours Rank-1)
- [ ] Edge deployment pipeline diagram
- [ ] Hardware: RB3 Gen2 specs + memory budget
- [ ] Deployment status table (complete / in-progress / placeholder)
- [ ] If noise handling implemented: separate results section
- [ ] Limitations (honest)
- [ ] Future work (3-4 bullets)
- [ ] Conclusion + acknowledgments

### Live demo (if feasible)

- [ ] **Option A** (safest): pre-recorded video of inference on RB3 Gen2.
- [ ] **Option B** (impressive): live SSH into RB3 Gen2 during defense, run a query, show top-5 retrieved images.
- [ ] Prepare backup static screenshots in case demo fails.

### Anticipated Q&A

- [ ] "Why Circle Loss over Triplet Loss?" — gradient saturation, adaptive re-weighting, Chapter 4 §4.2.
- [ ] "Why does α₅=0.2 fail despite LoRA enabling larger batches?" — false-negative amplification, Chapter 4 §4.3.
- [ ] "Why is the Chinese gain so much larger than English?" — baseline alignment quality + FN collision rate, Chapter 4 §4.5.
- [ ] "Has the model actually been deployed on the RB3?" — honest: pipeline designed and Steps 1-2 complete; Step 3 QNN compilation in progress; see `tab:deployment_status`.
- [ ] "What's the 5.9M vs 4.77M discrepancy?" — scope difference (Lightning full-model trainable vs PEFT backbone-only), documented in Chapter 6 Table `tab:model_breakdown` footnote.
- [ ] "Why `m=0.25`? Why γ=128?" — margin from DeepFashion conventions, γ from original Circle Loss paper; Chapter 3 §3.6.
- [ ] "What's your contribution vs TBPS-mSigLIP?" — 4 concrete contributions in Chapter 1 §1.4.

---

## Quick Verification Commands

Once P0 items are fixed, use these to verify submission readiness:

```bash
cd knowledge/PhamTungLam_20224321

# Compile cleanly
pdflatex main && bibtex main && pdflatex main && pdflatex main

# Check for undefined references (target: 0)
grep -c "LaTeX Warning: Reference" main.log

# Check for overfull hboxes (target: < 5)
grep -c "Overfull \\\\hbox" main.log

# Count remaining placeholders (intentional in P2 section, 0 elsewhere)
grep -rn "PLACEHOLDER" Chapter/

# Verify all figures are found
grep -c "LaTeX Warning: File" main.log

# Verify citations resolved
grep -c "Citation.*undefined" main.log
```

Expected final state:
- ✅ 0 undefined references
- ✅ 0 undefined citations
- ✅ 0 missing figures
- ✅ < 5 overfull hboxes
- ✅ Exactly 15 `[PLACEHOLDER]` markers (all in Chapter 6) until deployment completes

---

## Recommended Execution Order

| Day(s) | Tasks | Outcome |
|--------|-------|---------|
| 1 | P0 items — blockers fixed | Thesis compiles end-to-end |
| 2 | P1 citation + glossary audit | Reviewer-ready text |
| 3 | P1.5 decision — contribution or extension? | Clarifies scope |
| 4-10 | P1.5 implementation (if contribution chosen) | New Chapter 5 results |
| 11 | P4 content refinements + P5 docs | Polish pass |
| 12 | P3 figures | Visual polish |
| 13-?? | P2 deployment measurements (as pipeline completes) | Fill placeholders |
| Final week | P6 defense prep + final compile pass | Submission-ready |

---

## Source Artifacts (for every number quoted in thesis)

Per the `feedback_no_estimates.md` memory rule — every number must cite one of these:

| Source | Contents |
|--------|----------|
| `deployment/logs/analyze_20260414_164509.log` | Parameter counts (4.77M LoRA, 1.18M SimCLR, 376.57M total), FP32/FP16/INT8 computed sizes |
| `deployment/logs/export_lora_fp16_20260414_185032.log` | On-disk: model_fp32.pt (1418.5 MB), model_fp16.pt (709.3 MB) |
| `deployment/logs/export_onnx_20260415_102552.log` | ONNX export confirmation (vision, text) |
| `deployment/logs/to_fp16_20260415_133611.log` | Vision FP16 ONNX: 178.7 MB; Vision FP32 ONNX: 356.0 MB |
| `exported_model/*/` | On-disk file sizes (verified via `ls -la`) |
| `deployment/docs/benchmark-rp.md` | Proxy model benchmarks (MobileNetV2 92.0/24.7 ms, ResNet18 99.4/84.4 ms, EfficientNet-B0 126.2 ms CPU) |
| `docs/EXPERIMENT_SUMMARY.md` | Multi-seed results (2307/2300/2400), ablation tables |
| `knowledge/response.md` | 5.9M/1.57% LoRA figure (published), PRW-TPS-CN result (59.35%), full FT control (49.18%), multi-seed CI |
| `config/loss/cir_msiglip.yaml` | Actual training hyperparameters (margin, gamma, weights) |

**Any value not traceable to one of these is fabricated and must be removed.**
