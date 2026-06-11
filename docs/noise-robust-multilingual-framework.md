# Multilingual Noise-Robust Hard-Negative Framework

> Status: research and implementation design proposal  
> Target baseline: mSigLIP-LoRA + Curriculum Circle Loss  
> Target datasets: VN3K Vietnamese, CUHK-PEDES English, PRW-TPS-CN Chinese  
> Working name: MNEB-HN, Multilingual Noise Evidence Bank for Hard-Negative TBPS

---

## 1. Executive Summary

The current mSigLIP-CLoRA framework already solves a major part of multilingual
Text-Based Person Search (TBPS): it improves hard-negative discrimination while
keeping training parameter-efficient through LoRA. On VN3K, the best reported
run reaches `52.28%` text-to-image R@1 with LoRA + Curriculum Circle Loss. On
PRW-TPS-CN, it reaches `59.35%` R@1. On full CUHK-PEDES, the gain is positive
but smaller, from `70.76%` to `71.85%` R@1.

The next framework should not replace this hard-negative core. It should keep
`mSigLIP-LoRA + N-ITC + Curriculum Circle Loss + C-ITC + SimCLR` as the clean
alignment and hard-negative backbone, then add high-precision noise modules for
two cases:

- False Negative (FN): a pair labeled negative should actually be treated as
  semantically matching or same-person.
- False Positive / Noisy Correspondence (FP/FPP): a pair labeled positive is
  actually mismatched and should not be pulled together.

The central lesson from NACIR-lite is that noise correction must not directly
weaken Circle Loss unless the evidence is extremely strong. Circle Loss lives on
hard negatives. If a true hard negative is incorrectly detected as a false
negative, Circle Loss loses exactly the sample it needs most.

The proposed new direction is a decoupled evidence-driven framework:

```text
mSigLIP-LoRA backbone
  + primary multilingual alignment: N-ITC + MVS
  + hard-negative refinement: Curriculum Circle Loss
  + regularization: C-ITC + SimCLR
  + evidence memory: cross-epoch global/local statistics
  + FN auxiliary branch: FNM-like adaptive margin
  + FP auxiliary branch: RDE-like global-local consensus
```

The important rule is:

```text
High-confidence noise -> auxiliary correction
Uncertain evidence     -> no-op
True hard negative     -> Circle Loss owns it
```

---

## 2. Problem Statement

### 2.1 Current Strength

The current framework is designed for multilingual hard-negative optimization:

- Backbone: `siglip-base-patch16-256-multilingual`.
- Adaptation: LoRA on `q_proj`, `k_proj`, `v_proj`, `out_proj`.
- Primary loss: SigLIP-style N-ITC with `logit_scale` and `logit_bias`.
- Hard-negative loss: Cross-modal Circle Loss with curriculum.
- Current code source of truth for Circle margin: `m=0.25`, `gamma=128`.

This works especially well when the main difficulty is fine-grained retrieval:
different people wear similar colors, uniforms, backpacks, shoes, or other
attributes. In those cases, N-ITC alone gives weak gradient separation between
easy and hard negatives, while Circle Loss amplifies the hard-negative gradient.

### 2.2 Dataset Roles

The design should separate dataset roles clearly:

| Dataset | Language | Main role in new framework |
|---|---|---|
| VN3K / 3000VnPersonSearch | Vietnamese | Clean multilingual low-resource hard-negative benchmark |
| CUHK-PEDES | English | Large natural-noise benchmark with FP/FN ambiguity |
| PRW-TPS-CN | Chinese | Multilingual generalization benchmark |

VN3K should not be used as proof that noise correction improves clean data. It
should be used as a clean fallback acceptance test: the detector must mostly do
nothing and preserve hard-negative mining.

CUHK-PEDES is the better target for the noise-aware contribution because it is a
large English dataset where natural caption ambiguity, annotation noise, and
identity/correspondence noise are more likely to appear.

### 2.3 New Success Criteria

The new framework should satisfy all of the following:

1. Preserve VN3K clean performance around the current Circle baseline:
   - best seed reference: `52.28%` R@1;
   - multi-seed reference: `51.52 +/- 0.68%` R@1.
2. Improve or strongly compete on CUHK-PEDES:
   - current mSigLIP-CLoRA reference: `71.85%` R@1;
   - target direction: close the gap to English SOTA methods by adding real
     FP/FN robustness, not only hard-negative refinement.
3. Preserve multilingual validity:
   - Vietnamese, English, and Chinese must use the same mSigLIP text encoder;
   - local/token logic must be tokenizer-agnostic and language-agnostic.
4. Keep Circle Loss intact as the hard-negative core:
   - no direct default scaling of `alpha_n` or `alpha_p`;
   - use auxiliary losses and evidence-gated sample decisions instead.

---

## 3. Why NACIR-Lite Failed

NACIR attempted to handle FN and FP inside the Circle Loss formula:

```text
alpha_n -> alpha_n * FN_scale
alpha_p -> alpha_p * FP_scale
```

This was attractive because Circle Loss already separates positive and negative
branches. In practice, it was too direct and too under-evidenced.

### 3.1 FN Failure

On clean VN3K, NACIR-FN reduced performance. A clean run with FN detector enabled
reached roughly:

```text
LoRA + Curriculum Circle baseline: 52.28 R@1
Full NACIR clean run:              50.70 R@1
Delta:                             -1.58 R@1
```

The likely reason is that the FN detector confused true hard negatives with
false negatives.

This is the core ambiguity:

| Pair type | Label says | True meaning | Similarity |
|---|---|---|---|
| True hard negative | negative | different person, visually similar | high |
| False negative | negative | should be matching or semantically matching | high |

Similarity alone cannot separate them. If the detector sees high similarity and
concludes "false negative", it suppresses exactly the high-similarity negative
that Circle Loss should push apart.

The failure is not evidence that FNM is wrong. It shows that NACIR-FN was only a
thin approximation of FNM:

- no large momentum queue;
- no strong global-local evidence;
- direct mutation of Circle's negative branch;
- insufficient distinction between hard negatives and false negatives.

### 3.2 FP Failure Risk

NACIR-FP used a single GMM over per-sample Circle loss EMA. The assumption was:
clean positive pairs have lower loss, noisy-positive pairs have higher loss.

This is RDE-inspired, but it is not RDE. A single GMM can separate loss values
without proving that the high-loss component is truly noisy. The high-loss group
may contain:

- false-positive pairs;
- hard-clean positives;
- long or detailed captions;
- rare attributes;
- difficult images;
- samples affected by augmentation.

If these hard-clean positives are downweighted, the model loses useful positive
alignment signal. The correct conclusion is not that RDE has no value. The
correct conclusion is that NACIR-FP-lite lacks the dual-branch consensus and
label recalibration that make RDE robust.

### 3.3 Design Lesson

NACIR failed because it tried to perform noise correction by directly changing
Circle Loss forces with weak evidence.

The next framework must invert this design:

```text
Circle Loss remains the hard-negative optimizer.
Noise modules collect evidence across epochs.
Noise corrections act through auxiliary objectives or very conservative sample
decisions.
Uncertain cases fall back to the original Circle behavior.
```

---

## 4. Backbone Analysis: CLIP vs mSigLIP

FNM and RDE are successful references, but they were built around CLIP-style
English-centric TBPS. The new framework must be adapted to mSigLIP rather than
copied.

### 4.1 CLIP-Based Methods

FNM and RDE use CLIP encoders and often rely on:

- global image/text tokens;
- local token selection from attention maps;
- English caption structure;
- full or heavier fine-tuning;
- local/global branches such as BGE/TSE.

This is natural for CUHK-PEDES, which is an English benchmark.

### 4.2 mSigLIP Constraints

The current framework uses mSigLIP:

- multilingual text encoder;
- SigLIP-style sigmoid contrastive pretraining;
- learned `logit_scale` and `logit_bias`;
- 768-dimensional shared embeddings;
- mean-pooled image/text embeddings in the current training path;
- LoRA adaptation for parameter efficiency.

The local/token branch must therefore be language-agnostic:

- no English word-level assumptions;
- no rule that depends on whitespace tokenization;
- no hand-written Vietnamese or Chinese segmentation;
- use attention masks, token embeddings, pooled token groups, or part-level
  aggregation instead.

### 4.3 Why mSigLIP Should Stay

Switching back to CLIP would weaken the main contribution:

- CLIP is less suitable for Vietnamese and Chinese without extra adaptation;
- mSigLIP already supports multilingual text better;
- LoRA on mSigLIP has proven stable and parameter-efficient;
- existing deployment/export path is based on mSigLIP.

The new framework should be a multilingual extension of mSigLIP-CLoRA, not an
English CLIP reimplementation.

---

## 5. Framework Design: MNEB-HN

MNEB-HN stands for Multilingual Noise Evidence Bank for Hard-Negative TBPS.

The design has four principles:

1. Keep the current mSigLIP-CLoRA training objective as the clean fallback.
2. Preserve Circle Loss as the owner of hard-negative mining.
3. Use cross-epoch evidence before making noise decisions.
4. Use auxiliary correction branches for FN and FP instead of directly mutating
   Circle's `alpha_n` and `alpha_p`.

### 5.1 High-Level Architecture

```text
Batch
  -> mSigLIP vision encoder + LoRA
  -> mSigLIP text encoder + LoRA
  -> global image/text embeddings
  -> optional local/part image/text embeddings

Core losses:
  -> N-ITC + MVS
  -> Curriculum Circle Loss
  -> C-ITC
  -> SimCLR

Evidence branch:
  -> EvidenceMemoryBank update
  -> global/local consistency stats
  -> top-k stability stats
  -> per-sample/per-pair loss EMA
  -> FN posterior candidates
  -> FP clean/noisy consensus

Auxiliary correction:
  -> FNMStyleAuxLoss for high-confidence FN candidates
  -> RDEStyleAuxLoss for high-confidence FP/noisy-positive candidates
  -> no-op for uncertain cases
```

### 5.2 Core Loss Remains Stable

The base objective remains:

```text
L_base = L_NITC + lambda_circle(t) * L_Circle + 0.1 * L_CITC + 0.4 * L_SimCLR
```

The Circle schedule remains:

```text
epoch 0-5:   Circle off
epoch 6-20:  linear ramp to 0.1
epoch 21+:   stable at 0.1
```

The new modules add:

```text
L_total = L_base
        + lambda_fnm(t) * L_FNM_aux
        + lambda_rde(t) * L_RDE_aux
```

Important: `L_Circle` is not rewritten by default. It should use the original
positive and negative pair masks. Noise modules are allowed to add auxiliary
corrections or sample-selection signals, but they should not suppress Circle's
hard-negative force unless a later ablation proves that direct gating is safe.

---

## 6. Evidence Memory Bank

The Evidence Memory Bank is the missing infrastructure that NACIR-lite did not
have. A single mini-batch cannot reliably distinguish hard negatives, false
negatives, hard-clean positives, and false positives. The bank stores evidence
across epochs and turns single-batch signals into temporal decisions.

### 6.1 Stored State

For each training sample:

```text
sample_id
pid
image_id
global_image_embedding_ema
global_text_embedding_ema
optional_local_image_embedding_ema
optional_local_text_embedding_ema
per_sample_loss_ema
clean_probability_history
seen_count
last_seen_epoch
```

For selected pairs or top-k candidates:

```text
image_sample_id
text_sample_id
pid_relation
global_similarity_ema
local_similarity_ema
mutual_topk_count
mutual_topk_streak
pair_loss_ema
fn_posterior_ema
fp_conflict_score_ema
last_updated_epoch
```

Pair state should be sparse. It should not store all `N x N` pairs. It should
store only:

- in-batch positives;
- top-k image-to-text neighbors;
- top-k text-to-image neighbors;
- high-loss positive pairs;
- high-similarity negative pairs.

### 6.2 Evidence Types

The bank should compute and expose:

| Evidence | Use |
|---|---|
| Global similarity stability | Detect persistent high-similarity candidates |
| Local/part consistency | Separate true semantic matches from global-only collisions |
| Mutual top-k stability | Require reciprocal retrieval evidence across time |
| Per-sample loss EMA | Candidate signal for FP/noisy-positive detection |
| Per-pair loss EMA | Better than sample-only loss for correspondence noise |
| Positive/negative distribution stats | Required for FNM-like posterior |
| No-op rate | Acceptance signal on clean VN3K |

### 6.3 Decision Rule

Evidence Memory Bank should not emit hard decisions from a single observation.

```text
single-batch signal       -> candidate
cross-epoch stable signal -> evidence
global-local agreement    -> decision
uncertain                 -> no-op
```

This is the core clean fallback mechanism.

---

## 7. FNM-Like Branch for False Negatives

### 7.1 Goal

The FN branch should prevent the model from strongly penalizing pairs that are
labeled negative but are very likely semantic matches. It must not suppress true
hard negatives on VN3K.

### 7.2 Difference from NACIR-FN

NACIR-FN:

```text
alpha_n <- alpha_n * scale
```

This directly weakened Circle Loss.

The new branch should instead add an auxiliary FNM-style loss:

```text
L_total = L_base + lambda_fnm * L_FNM_aux
```

Circle Loss remains unchanged.

### 7.3 FNM-Style Detection

Use the Evidence Memory Bank to estimate positive and negative similarity
distributions:

```text
f_pos(s) = Normal(mu_pos, sigma_pos)
f_neg(s) = Normal(mu_neg, sigma_neg)
P_FN(s) = p * f_pos(s) / (p * f_pos(s) + (1 - p) * f_neg(s))
```

But a high posterior is not enough. A negative pair becomes a high-confidence FN
candidate only when all gates pass:

```text
P_FN(s) >= theta_fn
global similarity is high and stable
local/part similarity is also high
pair appears in mutual top-k over multiple epochs
positive/negative distribution gap is reliable
candidate count remains below a strict cap
```

### 7.4 FNM-Style Auxiliary Loss

Use an adaptive margin for suspicious negatives rather than removing them from
Circle:

```text
rho(i, j) =
  full_margin                         if pair is normal negative
  margin * (1 - P_FN(i, j))/(1-theta) if pair is high-confidence FN
  0                                   if pair is positive
```

Then compute an InfoNCE-style auxiliary loss with this adaptive margin in the
denominator. This mirrors the FNM idea while keeping the Circle hard-negative
branch intact.

### 7.5 Clean VN3K Requirement

On VN3K, expected behavior:

```text
fn_candidate_rate: near 0
fn_high_confidence_rate: near 0
Circle negative gradient: unchanged
L_FNM_aux: near no-op
```

If a VN3K run shows many high-confidence FN decisions, the detector is too
aggressive and should fail acceptance.

---

## 8. RDE-Like Branch for False Positives

### 8.1 Goal

The FP branch should detect noisy-positive image-text correspondences without
mistaking hard-clean positives for noise.

### 8.2 Difference from NACIR-FP

NACIR-FP used:

```text
one GMM on per-sample Circle loss
alpha_p <- alpha_p * clean_weight
```

This is too weak as evidence and too direct as correction.

The new branch should follow RDE's stronger principle:

```text
global branch decision
local branch decision
consensus -> clean/noisy/uncertain
```

### 8.3 Global-Local Consensus

Use two evidence views:

| Branch | Suggested mSigLIP adaptation |
|---|---|
| Global branch | current pooled 768-dim image/text embeddings |
| Local branch | part-token or token-selection embeddings, language-agnostic |

For the local branch, avoid English-specific token assumptions. Use one of:

- vertical image part pooling plus masked text-token aggregation;
- attention-weighted token pooling using attention masks;
- top-k token selection from transformer attention, but with no language rules;
- existing `PART_ALIGN` path as an experimental starting point.

Each branch fits its own clean/noisy loss model:

```text
GMM_global(loss_global)
GMM_local(loss_local)
```

Then combine decisions:

```text
global clean + local clean -> confident clean
global noisy + local noisy -> confident noisy
otherwise                  -> uncertain
```

### 8.4 Correction Strategy

Do not directly scale Circle `alpha_p` in v1.

Use confident noisy-positive decisions for one of these safer corrections:

1. auxiliary RDE-style TAL loss with recalibrated labels;
2. sample weighting on an auxiliary correspondence loss;
3. exclusion from the auxiliary positive-pair branch only.

The main Circle Loss remains unchanged in v1. This avoids damaging hard-clean
positives before the detector proves its precision.

### 8.5 CUHK-PEDES Requirement

On CUHK-PEDES, expected behavior:

```text
global/local GMM separation should be meaningful
uncertain rate should not be forced low
confident noisy-positive rate should be plausible, not massive
R@1 should improve beyond 71.85 or at least not regress
```

---

## 9. Hard-Negative Preservation

Hard-negative preservation is the main invariant.

The detector must distinguish these cases:

| Case | Desired behavior |
|---|---|
| True hard negative | Circle pushes strongly |
| False negative | FNM auxiliary softens contrastive penalty |
| Hard-clean positive | Keep positive alignment signal |
| False positive | RDE auxiliary reduces noisy-positive influence |
| Uncertain | No-op, preserve base objective |

The default should be conservative:

```text
If the framework is not sure, do not correct.
```

The system should log explicit preservation diagnostics:

```text
hard_negative_topk_overlap
hard_negative_suppression_rate
fn_high_confidence_rate
fp_confident_noisy_rate
uncertain_rate
base_circle_loss
aux_fnm_loss
aux_rde_loss
```

On VN3K, `hard_negative_suppression_rate` should be zero or near zero because
the new v1 design should not directly suppress Circle at all.

---

## 10. Proposed Implementation Blueprint

This section is a future implementation plan, not a statement of current code.

### 10.1 New Conceptual Modules

```text
EvidenceMemoryBank
  Owns cross-epoch sample and sparse-pair evidence.

GlobalLocalNoiseConsensus
  Produces clean/noisy/uncertain decisions from global and local branches.

FNMStyleAuxLoss
  Computes adaptive-margin FN mitigation without changing Circle Loss.

RDEStyleAuxLoss
  Computes an auxiliary robust correspondence loss using consensus labels.
```

### 10.2 Minimal Config Groups

All new modules should be disabled by default until validated:

```yaml
loss:
  evidence_bank:
    enabled: false
    queue_size: 4096
    topk: 20
    ema_beta: 0.99
    min_seen_epochs: 3

  fnm_aux:
    enabled: false
    weight: 0.05
    enable_epoch: 15
    theta_fn: 0.8
    max_candidate_frac: 0.02
    prior: 0.003

  rde_aux:
    enabled: false
    weight: 0.05
    enable_epoch: 15
    gmm_interval: 5
    min_gmm_separation: 1.0
    uncertain_policy: "no_op"
```

The actual values should be validated in `notebooks/workspace.ipynb` before
full training.

### 10.3 Training Schedule

Recommended initial schedule:

| Epoch | Behavior |
|---:|---|
| 0-5 | N-ITC/C-ITC/SimCLR only; collect evidence if enabled |
| 6-14 | Circle ramps in; continue evidence collection only |
| 15-20 | Allow auxiliary FN/FP decisions if evidence is stable |
| 21+ | Circle stable; auxiliary losses stable if diagnostics pass |

This schedule intentionally avoids early noisy decisions while LoRA alignment is
still forming.

### 10.4 Implementation Order

Implement in this order:

1. EvidenceMemoryBank with logging only.
2. Global/local feature extraction for diagnostics only.
3. RDE-style global-local consensus on synthetic FP noise, no training effect.
4. FNM-style posterior and candidate tracking on synthetic FN noise, no training
   effect.
5. Add `L_RDE_aux` only after detector precision is measured.
6. Add `L_FNM_aux` only after FN candidates are stable and rare on clean VN3K.
7. Run full training only after notebook checks pass.

This avoids repeating NACIR's mistake: do not let a detector affect training
before it has proven high precision.

---

## 11. Validation Plan

### 11.1 VN3K Clean Acceptance

Purpose: prove clean fallback and hard-negative preservation.

Required checks:

- `fn_high_confidence_rate` near zero.
- `fp_confident_noisy_rate` near zero.
- `uncertain_rate` can be high; that is acceptable.
- Circle loss and hard-negative statistics remain close to baseline.
- R@1 remains in the current baseline region.

Acceptance reference:

```text
Best seed:          52.28 R@1
Multi-seed mean:    51.52 +/- 0.68 R@1
```

### 11.2 CUHK-PEDES Natural-Noise Validation

Purpose: prove the framework is useful where natural FP/FN noise exists.

Compare:

```text
mSigLIP-LoRA baseline
mSigLIP-CLoRA current Circle framework
MNEB-HN proposed framework
```

Track:

```text
R@1, R@5, R@10, mAP, mINP
confident clean/noisy/uncertain rates
global/local consensus agreement
FN candidate rate
FP candidate rate
evidence-bank stability
```

Target:

```text
Improve beyond current full CUHK-PEDES R@1 = 71.85
or demonstrate no regression plus clear noise-robust behavior.
```

### 11.3 Synthetic Stress Tests

Use synthetic tests to measure detector precision before expensive full runs:

| Stress test | Mechanism | Expected result |
|---|---|---|
| FP noise | caption shuffle via `dataset.noisy_rate` | RDE branch detects noisy positives |
| FN noise | PID split via `dataset.fn_noisy_rate` | FNM branch detects stable FN candidates |
| FP+FN noise | both mechanisms | branches remain decoupled |
| Clean VN3K | no injected noise | detectors mostly no-op |

Important: synthetic success is necessary but not sufficient. The final proof
must come from CUHK-PEDES natural-noise behavior.

---

## 12. Notebook Gate Before Training

Before launching full training, validate the framework in `notebooks/workspace.ipynb`
on frozen embeddings.

Required notebook checks:

1. Evidence bank can reproduce stable top-k and loss EMA from frozen embeddings.
2. VN3K clean no-op: high-confidence FN/FP rates are near zero.
3. Synthetic FP: known corrupted pairs receive high noisy evidence in both
   global and local branches.
4. Synthetic FN: known split-PID pairs receive higher FN posterior than true
   negatives.
5. Hard-negative preservation: top hard negatives remain assigned to Circle,
   not corrected away.
6. Auxiliary loss values are finite and in comparable scale:
   - N-ITC around `3-5`;
   - Circle around `0.5-2.0` near convergence;
   - auxiliary losses should be weighted to avoid dominance.

Full training should not start until these pass.

---

## 13. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Detector suppresses hard negatives | Do not directly modify Circle in v1 |
| One GMM misclassifies hard-clean positives | Require global-local consensus |
| Mini-batch evidence is noisy | Use cross-epoch Evidence Memory Bank |
| Local branch becomes English-specific | Use tokenizer-agnostic masked token pooling |
| Extra modules overfit VN3K | Keep modules no-op on clean data and disabled by default |
| CUHK-PEDES gain is small | Treat framework as natural-noise robustness plus hard-negative refinement |
| Added complexity hurts deployment | Auxiliary branches are training-only unless explicitly needed at inference |

---

## 14. Positioning

The new framework should be positioned carefully:

```text
Not:    NACIR fixes all noise by scaling Circle alpha terms.
Rather: A multilingual hard-negative framework with evidence-gated auxiliary
        noise correction and clean fallback.
```

The strongest claim should be:

```text
MNEB-HN preserves mSigLIP-CLoRA's multilingual hard-negative strength on clean
VN3K while adding FNM/RDE-inspired evidence modules for natural FP/FN noise in
large English TBPS benchmarks such as CUHK-PEDES.
```

This aligns the three pieces:

- mSigLIP-LoRA: multilingual parameter-efficient adaptation;
- Circle Loss: hard-negative discrimination;
- Evidence-gated FNM/RDE branches: natural noise robustness.

---

## 15. Immediate Next Steps

1. Keep current reported mSigLIP-CLoRA results as the clean baseline.
2. Do not continue NACIR-lite as the main path.
3. Prototype `EvidenceMemoryBank` in notebook first, with no training effect.
4. Validate global-local consensus using existing pooled features and the
   current part-token alignment path.
5. Only after detector precision is proven, implement auxiliary losses in code.
6. Run acceptance in this order:
   - VN3K clean no-op;
   - synthetic FP;
   - synthetic FN;
   - CUHK-PEDES full;
   - PRW-TPS-CN multilingual sanity.

The final framework should earn the right to touch training dynamics. Until
then, Circle Loss remains the trusted hard-negative engine.
