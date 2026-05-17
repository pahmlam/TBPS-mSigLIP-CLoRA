# Plan: Comprehensive Thesis Update — mSigLIP-CLoRA

## Context

The graduation thesis (`knowledge/PhamTungLam_20224321/`) currently covers mSigLIP-CLoRA (Circle Loss + LoRA + Curriculum Learning) with R@1=51.30% as the best result. Since then, significant new work has accumulated:

- **Best seed 2400** achieved R@1=52.28% with multi-seed confidence intervals
- **Chinese dataset (PRW-TPS-CN)** evaluation: R@1=59.35% (+12.57% over baseline)
- **Full FT control experiment** ruling out batch size as the source of improvement
- **Corrected Table VI** (10% CUHK-PEDES) — Fixed Weight row had wrong data
- **Edge deployment pipeline** targeting Qualcomm RB3 Gen2 (4GB RAM)
- **Noise handling analysis** (FNM, RDE) with 5 proposed improvement ideas
- **Reviewer responses** with deep analysis on instability, curriculum sensitivity, low-resource gains

All of this needs to be integrated into the thesis, with placeholders for unfinished work (INT8 quantization, DSP benchmarks). The thesis name also needs updating to reflect the expanded scope.

---

## Recommended Thesis Title

**"Hard Negative-Aware Optimization and Edge Deployment for Multilingual Text-Based Person Search"**

Rationale:
- Captures both training optimization AND deployment (the two main contributions)
- "Multilingual" reflects the 3-language validation (Vietnamese, English, Chinese)
- "Hard Negative-Aware" is the core technical contribution
- Concise yet comprehensive — appropriate for a HUST graduation thesis
- Aligns with R4-C1/R4-C4 repositioning (no overclaiming "universally best")

Alternative (longer, more descriptive):
"mSigLIP-CLoRA: From Hard Negative-Aware Optimization to Edge Deployment for Multilingual Low-Resource Text-Based Person Search"

---

## New Chapter Structure

```
Current (6 chapters)              →  Updated (7 chapters)
─────────────────────                ────────────────────
Ch 1: Introduction                   Ch 1: Introduction (updated)
Ch 2: Literature Review              Ch 2: Literature Review (+ noise handling, deployment)
Ch 3: Methodology                    Ch 3: Methodology (+ LoRA details, deployment overview)
Ch 4: Theoretical Analysis           Ch 4: Theoretical Analysis (expanded: 2→5 sections)
Ch 5: Numerical Results              Ch 5: Numerical Results (major update: all new data)
Ch 6: Conclusions                    Ch 6: Edge Deployment and Model Compression (NEW)
                                     Ch 7: Conclusions and Future Work (rewritten)
Appendix A: Code                     Appendix A: Code (keep)
                                     Appendix B: Deployment Code (NEW)
```

---

## File-by-File Changes

### Phase 1: Foundation (structural, no content)

#### 1.1 `main.tex` — Restructure chapters
- After line 220 (`\subfile{Chapter/5_Numerical_results}`), add:
  ```latex
  \newpage
  \chapter{EDGE DEPLOYMENT AND MODEL COMPRESSION}
  \subfile{Chapter/6_Edge_Deployment}
  ```
- Change lines 222-224 from `\chapter{CONCLUSIONS}` / `\subfile{Chapter/6_Conclusions}` to:
  ```latex
  \chapter{CONCLUSIONS AND FUTURE WORK}
  \subfile{Chapter/7_Conclusions}
  ```
- Uncomment Appendix B block (lines 250-252), change to:
  ```latex
  \newpage
  \chapter{DEPLOYMENT PIPELINE DETAILS}
  \subfile{Chapter/Appendix_B}
  ```

#### 1.2 `Cover.tex` — Update title (line 16)
- Old: "Analysis of limitations and proposal of strategies for improving the embedding space in the TBPS-mSigLIP model"
- New: Selected thesis title (see above)
- Update date if defense date changes (line 38: "HANOI, 02/2026")

#### 1.3 `glossary.tex` — Add new abbreviations
New entries: ONNX, SNPE, QNN, DSP, HTP, DLC, VRAM, PEFT, FNM, RDE, FN, FP, ARM, SoC, NEON

#### 1.4 `reference.bib` — Add new bibliography entries
- FNM paper (AAAI 2026)
- RDE paper (CVPR 2024)
- ONNX Runtime
- Qualcomm QCS6490 / RB3 Gen2 documentation
- Qualcomm AI Engine Direct SDK / SNPE
- Qualcomm AI Hub

---

### Phase 2: Fix existing content

#### 2.1 `Chapter/0_3_Abstract.tex` — Rewrite
Changes:
- R@1: 51.30% → **52.28%** (best seed 2400), add mean±std: 51.52±0.68%
- Add Chinese (PRW-TPS-CN) evaluation: R@1=59.35% vs 46.78% baseline
- Add one sentence on edge deployment contribution
- Reposition: "strongest in multilingual low-resource adaptation" (per R4-C1)
- Mention three languages tested (Vietnamese, English, Chinese)

#### 2.2 `Chapter/5_Numerical_results.tex` — Fix Table VI (CRITICAL BUG)
The "Ours (Fixed Weight)" row in the 10% CUHK-PEDES table (lines 236-237) has **wrong data** — it shows VN3K results instead of 10% CUHK-PEDES.

**Current (WRONG):**
| Fixed Weight | 51.30 | 78.20 | 86.68 | 56.46 | 49.89 |

**Corrected:**
| Fixed Weight | 56.87 | 77.18 | 84.15 | 50.70 | 34.61 |

Also rewrite the analysis (lines 244-256): Curriculum now wins 4/5 metrics, no "stability-plasticity trade-off" story needed.

#### 2.3 `Chapter/1_Introduction.tex` — Update contributions & organization
- Add contribution 5: "Multi-seed and multilingual validation (Vietnamese, English, Chinese)"
- Add contribution 6: "Edge deployment pipeline for Qualcomm RB3 Gen2"
- Rewrite Section 1.5 (Organization) to describe 7 chapters
- Add deployment motivation in Section 1.3

---

### Phase 3: Expand existing chapters

#### 3.1 `Chapter/4_Theoretical_Analysis.tex` — Add 3 new sections
Currently has only 2 sections (~55 lines). Expand to 5 sections (~5-6 pages):

**New Section 4.3: "Instability at Higher Circle Loss Weights"** (from R2-Q1)
- Three factors: (1) batch=24 still small, (2) γ=128 amplification, (3) false negative amplification
- Table: α₅=0.2 yields R@1=49.83%, **below** LoRA-only baseline (49.90%)
- Mathematical: show gradient magnitude scales with α₅×γ

**New Section 4.4: "Curriculum Schedule Sensitivity"** (from R2-Q2)
- T_warmup=5 rationale: LoRA init near zero, N-ITC needs convergence first
- Linear ramp: T_ramp=15, per-epoch increment ≈0.0067
- Discussion of non-linear alternatives (cosine, exponential) as future work

**New Section 4.5: "Low-Resource vs. High-Resource Gain Analysis"** (from R2-Q4)
- Baseline alignment quality argument (13K vs 3K identities)
- False negative collision rate in batch=24
- +2.58% on VN3K vs +1.09% on CUHK-PEDES

#### 3.2 `Chapter/5_Numerical_results.tex` — Add new results

**Add 1: Update SOTA table (Table 1) with best seed 2400**
- R@1: 51.30→52.28, R@5: 78.20→79.55, R@10: 86.68→88.03, mAP: 56.46→57.32, mINP: 49.89→50.57
- Improvement: +2.58 R@1 (was +1.60)

**Add 2: Multi-seed confidence interval table** (after Table 1)
| Seed | R@1 | R@5 | R@10 | mAP | mINP |
|------|-----|-----|------|-----|------|
| 2307 | 51.30 | 78.20 | 86.68 | 56.46 | 49.89 |
| 2300 | 50.98 | 78.60 | 86.95 | 57.08 | 51.22 |
| 2400 | **52.28** | **79.55** | **88.03** | **57.32** | 50.57 |
| Mean±std | 51.52±0.68 | 78.78±0.69 | 87.22±0.71 | 56.95±0.44 | 50.56±0.67 |

**Add 3: Full FT control experiment table** (in ablation section, from R3-C2)
| Config | R@1 | R@5 | R@10 | mAP | mINP |
|--------|-----|-----|------|-----|------|
| Full FT (batch=24, eff=72) | 49.18 | 76.30 | 85.58 | 54.49 | 47.87 |
| LoRA (batch=24, eff=72) | 49.90 | 77.45 | 86.20 | 55.23 | 48.65 |
| LoRA + Curriculum Circle | **52.28** | **79.55** | **88.03** | **57.32** | **50.57** |

Analysis: conclusively rules out batch size as improvement source.

**Add 4: New subsection 5.6 "Generalization to Chinese (PRW-TPS-CN)"** (from R1-C1)
| Method | R@1 | R@5 | R@10 | mAP | mINP |
|--------|-----|-----|------|-----|------|
| TPAN | 21.63 | 42.54 | 52.99 | - | - |
| TBPS-mSigLIP | 46.78 | 60.28 | 66.82 | 35.41 | 10.61 |
| **mSigLIP-CLoRA** | **59.35** | **70.58** | **75.48** | **46.44** | **15.10** |

Analysis: +12.57% R@1, largest gain across all datasets, validates multilingual generalization.

**Add 5: LoRA parameter breakdown** (from R3-C3, R4-C3)
Update existing LoRA table in methodology or add to ablation:
- Full FT: 376M params (100%), ~11GB VRAM at BS=8
- LoRA: 5.9M params (1.57%), ~11GB VRAM at BS=24
- 98.4% parameter reduction, 3x batch increase on same 12GB GPU

#### 3.3 `Chapter/2_Literature_review.tex` — Add 2 new subsections

**New subsection: "Noise-Aware Contrastive Learning"** (after current metric learning section)
- Define false negatives (FN) and false positives (FP) in TBPS context
- Summarize FNM: Bayesian posterior P(FN|s), adaptive margin, momentum queue
- Summarize RDE: dual embedding, GMM-based noise detection, triplet alignment loss
- Gap: no noise handling applied to Circle Loss in TBPS

**New subsection: "Edge Deployment of Vision-Language Models"** (~1 page)
- Model compression overview (quantization, ONNX, pruning)
- Qualcomm SNPE/QNN for ARM-based edge devices
- Gap: no prior work on deploying mSigLIP-class models to 4GB RAM devices

Source material: `knowledge/FNM.md`, `knowledge/RDE.md`, `knowledge/noise_handling_analysis.md`

#### 3.4 `Chapter/3_Methodology.tex` — Minor additions
- Update LoRA table: "Less than 2%" → "5.9M / 376M = 1.57%"
- Add brief Section 3.8 "Deployment Pipeline Overview" (~1 page): 4-step pipeline (LoRA Merge → FP16 → ONNX → QNN), pointing to Chapter 6 for details

---

### Phase 4: New content

#### 4.1 `Chapter/6_Edge_Deployment.tex` — NEW FILE (~8-10 pages)

**Section 6.1: Motivation and Target Device**
- Real-world surveillance needs on-device inference
- RB3 Gen2 specs: QCS6490, 4x A78 + 4x A55, Hexagon 770, 4GB usable RAM

**Section 6.2: Deployment Pipeline Architecture**
- Pipeline diagram: Training → LoRA Merge → FP16 → ONNX → QNN/SNPE → DSP
- Rationale for each step

**Section 6.3: Model Analysis and Memory Estimation**
- Checkpoint analysis: 376M params (deduplicated)
- Memory table: FP32=2156 MB (tight), FP16=1077 MB (comfortable), INT8=539 MB (very comfortable)

**Section 6.4: LoRA Merge and FP16 Export**
- merge_and_unload() → strip optimizer → FP16 conversion with tensor deduplication
- Result: 1437 MB FP32 → 740 MB FP16

**Section 6.5: ONNX Conversion**
- Separate vision/text encoders, opset 18, dynamic batch
- Export verified successfully

**Section 6.6: Hardware Benchmarking (Proxy Models)**
- PyTorch CPU vs ONNX Runtime on RB3:
  - MobileNetV2: 92.0ms → 24.7ms (3.72x speedup)
  - ResNet18: 99.4ms → 84.4ms (1.18x)
- All accelerators validated (CPU/GPU/DSP)

**Section 6.7: QNN Compilation Status** [PLACEHOLDER sections]
- Current: ONNX export done, AI Hub compilation in progress
- Expected: QNN context binary for DSP/HTP inference
- [PLACEHOLDER] INT8 quantization results
- [PLACEHOLDER] DSP/HTP inference benchmarks on mSigLIP

**Section 6.8: Deployment Summary**
- Pipeline status table (done/in-progress/planned)
- Expected performance: ~50-100ms ONNX CPU, ~5-15ms DSP (estimated)

Source material: `deployment/docs/system.md`, `deployment/docs/benchmark-rp.md`, `deployment/README.md`, `deployment/scripts/onnx/export.py`, `deployment/logs/export_onnx_20260415_102552.log`

#### 4.2 `Chapter/7_Conclusions.tex` — Rewrite (rename from 6_Conclusions.tex)

**Section 7.1: Summary** — Updated
- Best R@1 = 52.28% (seed 2400), mean 51.52±0.68%
- 3 languages: Vietnamese (+2.58%), English (+1.09%), Chinese (+12.57%)
- Edge deployment pipeline designed and partially validated
- Reposition per R4-C1: strongest in multilingual low-resource adaptation

**Section 7.2: Limitations** — Expanded
- Keep: fixed aspect ratio, limited LoRA plasticity, suboptimal batch size
- Add: no explicit noise handling in Circle Loss (FN amplification at higher weights)
- Add: deployment pipeline not yet validated end-to-end on DSP/HTP

**Section 7.3: Future Work** — Major expansion

*Noise-Robust Circle Loss:*
- Core problem: Circle Loss amplifies false negatives (Section 4.3)
- Idea D: Distribution Separation regularization (low effort, medium impact)
- Idea A: FNM-Lite — Bayesian FN detection for Circle Loss, scale α_n by (1-P_FN) (medium effort, high impact)
- Idea C: Unified Noise-Aware Circle Loss combining FN+FP detection (high effort, very high impact)

*Completing Edge Deployment:*
- INT8 quantization with calibration on VN3K subset
- End-to-end DSP benchmarking
- GStreamer + SNPE plugin for surveillance camera integration

*Existing items (keep):*
- Adaptive resolution, gradient accumulation, advanced PEFT (DoRA), video retrieval

Source material: `knowledge/noise_ideas_concepts.md`, `knowledge/noise_handling_analysis.md`

#### 4.3 `Chapter/Appendix_B.tex` — NEW FILE
- Key deployment commands (LoRA merge, ONNX export, AI Hub submission)
- Critical function signatures from export scripts
- Not full Python files — focused on reproducibility

---

### Phase 5: Figures and polish

#### 5.1 New figures needed
- `Figure/deployment_pipeline.png` — Pipeline diagram (TikZ or draw.io)
- `Figure/benchmark_latency.png` — Bar chart: PyTorch vs ONNX Runtime latencies
- `Figure/memory_estimation.png` — Bar chart: FP32 vs FP16 vs INT8 memory on RB3

#### 5.2 Cross-reference audit
- Search for hardcoded table references ("Table 2", "Table 4") and convert to `\ref{}`
- Update Section 1.5 chapter descriptions for new 7-chapter structure
- Verify all `\label{}` and `\ref{}` still resolve correctly

#### 5.3 Compile and verify
- Run `pdflatex` + `bibtex` + `pdflatex` × 2
- Check for broken references, missing figures, layout issues

---

## Implementation Order (dependency-aware)

| Step | File | Type | Depends On |
|------|------|------|------------|
| 1 | `glossary.tex` | Edit | — |
| 2 | `reference.bib` | Edit | — |
| 3 | `Cover.tex` | Edit | — |
| 4 | `Chapter/0_3_Abstract.tex` | Rewrite | — |
| 5 | `Chapter/5_Numerical_results.tex` | Fix Table VI bug | — |
| 6 | `Chapter/5_Numerical_results.tex` | Add all new results | Step 5 |
| 7 | `Chapter/4_Theoretical_Analysis.tex` | Add 3 new sections | — |
| 8 | `Chapter/1_Introduction.tex` | Update contributions | — |
| 9 | `Chapter/2_Literature_review.tex` | Add 2 subsections | Steps 1,2 |
| 10 | `Chapter/3_Methodology.tex` | Minor additions | — |
| 11 | `Chapter/6_Edge_Deployment.tex` | **NEW** | Steps 1,2 |
| 12 | `Chapter/7_Conclusions.tex` | Rewrite | Steps 7,11 |
| 13 | `Chapter/Appendix_B.tex` | **NEW** | Step 11 |
| 14 | `main.tex` | Restructure | Steps 11,12,13 |
| 15 | Create figures | — | Steps 6,11 |
| 16 | Cross-reference audit | — | All above |

Steps 1-4 and 5,7,8 can run in parallel. Steps 9-13 can partially parallelize.

---

## Verification

1. **LaTeX compilation**: `pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex` — no errors
2. **Table VI correctness**: Verify Fixed Weight row shows 56.87/77.18/84.15/50.70/34.61 (NOT 51.30/78.20/86.68/56.46/49.89)
3. **Best result consistency**: R@1=52.28% appears in Abstract, Chapter 5 Table 1, Chapter 7 Summary
4. **Multi-seed data**: 3-seed table matches response.md exactly
5. **Chinese results**: PRW-TPS-CN table matches response.md R1-C1
6. **Cross-references**: All `\ref{}` resolve, no "??" in output
7. **Glossary**: All new abbreviations appear in the abbreviations list
8. **Placeholder markers**: Search for [PLACEHOLDER] to verify all incomplete sections are clearly marked
9. **Positioning language**: No remaining instances of "universally best" or overclaiming — audit abstract, intro, conclusion
