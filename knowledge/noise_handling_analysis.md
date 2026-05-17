# Noise Handling Analysis: mSigLIP vs FNM vs RDE

## The Problem Space

In text-based person search (TBPS), label noise manifests in two complementary ways:

| Type | What happens | Why it happens | Effect on training |
|------|-------------|----------------|-------------------|
| **False Negative (FN)** | A true match (same person) is labeled as non-matching | Same person has different IDs across cameras; generic descriptions match multiple people | Model pushes apart embeddings that should be close |
| **False Positive (FP)** | A non-matching pair is labeled as matching | Annotation error; degraded image quality makes annotator misjudge | Model pulls together embeddings that should be apart |

Both corrupt the gradient signal. Hard-negative mining (Circle Loss) **amplifies** this corruption because it specifically focuses on the pairs where the model disagrees with labels — which is exactly where label noise lives.

---

## 1. Current mSigLIP Framework: What We Handle

### Hard Negative Mining (Circle Loss)
- **Mechanism**: `logsumexp` over negative logits emphasizes the hardest negatives (highest similarity among different IDs)
- **Adaptive weighting**: `alpha_n = clamp_min(s_n + m, 0)` — negatives with similarity above margin `m=0.35` get amplified
- **Curriculum**: weight ramps from 0 (epoch 0-5) to 0.1 (epoch 6-20), stabilizes at 0.1 (epoch 21-60)
- **Location**: `model/objectives.py:compute_cross_modal_circle()`, curriculum in `model/tbps.py`

### Soft Label Targets (N-ITC)
- **Mechanism**: `sim_targets` are computed from ID overlap within batch, not hard 0/1
- When multiple samples share the same PID, targets distribute probability mass across all of them
- This provides **implicit** partial FN tolerance — if two different PIDs are the same person, the soft target doesn't zero them out entirely, but it also doesn't give them credit

### Identity-Aware Batch Sampling
- `RandomIdentitySampler`: 8 PIDs x 4 instances per batch
- Guarantees in-batch positives exist, but makes no attempt to filter noisy pairs

### What Is NOT Handled

| Gap | Consequence |
|-----|-------------|
| **No FN detection** | If PID-123 and PID-456 are the same person, Circle Loss actively pushes them apart with gamma=128 amplification |
| **No FP detection** | If image-A is wrongly paired with text-B, N-ITC pulls them together as a positive pair |
| **No per-sample confidence** | All samples weighted equally regardless of annotation quality |
| **No cross-batch statistics** | Noise detection requires distribution modeling, which needs more samples than a single batch provides |

---

## 2. FNM: False Negative Mitigation (AAAI 2026)

**Core idea**: Model the similarity distributions of positives and negatives as Gaussians, use Bayes' theorem to estimate the probability that any given "negative" is actually a false negative, then soften the loss margin for likely false negatives.

### Pipeline

```
Collect S+, S- similarities  -->  Fit Gaussians f+(s), f-(s)  -->  P(FN|s) via Bayes  -->  Adaptive margin rho
        |                                                                                        |
        +---- MoC queues (large-scale statistics) ----+---- Threshold theta ---- margin scaling --+
```

### Key Components

**A. False Negative Detection**

1. Collect similarity scores for positive pairs S+ and negative pairs S-
2. Fit Gaussian: `f+(s) ~ N(mu+, sigma+)`, `f-(s) ~ N(mu-, sigma-)`
3. Posterior probability via Bayes:
   ```
   P(FN|s) = p * f+(s) / [p * f+(s) + (1-p) * f-(s)]
   ```
   where `p` = prior probability of false negatives among negatives
4. Threshold: if `P(FN|s) >= theta` --> false negative, else true negative

**B. Adaptive Margin Loss (L_fnm)**

Standard contrastive loss, but with margin `rho(v_i, t_j)` per negative pair:
- **True negative** (`P(FN|s) < theta`): full margin `gamma` — push hard
- **False negative** (`P(FN|s) >= theta`): reduced margin `gamma * (1 - P(FN|s)) / (1 - theta)` — push gently
- **Positive** (`i = j`): no margin

Effect: the more likely a negative is actually a false negative, the less the model penalizes it.

**C. Momentum Contractive Queue (MoC)**

Problem: fitting Gaussians requires many samples; a mini-batch is too small.

Solution: maintain FIFO queues of features from previous batches (global + local), updated via EMA momentum encoder (`m ~ 0.999`). This gives thousands of similarity scores per step instead of batch_size^2.

### Strengths
- Principled statistical approach (Bayesian posterior)
- Graceful degradation: doesn't discard samples, just adjusts margins
- MoC provides rich statistics without extra forward passes

### Weaknesses
- Gaussian assumption may not hold for complex similarity distributions
- Prior `p` (false negative rate) must be estimated or tuned
- MoC adds memory overhead (4 queues) and a second set of encoders
- Only addresses false negatives, not false positives

---

## 3. RDE: Robust Dual Embedding (CVPR 2024)

**Core idea**: Use the "memorization effect" of neural networks (clean data has lower loss than noisy data) to separate clean vs noisy samples via GMM on per-sample losses, then recalibrate labels and train with a noise-robust triplet loss.

### Pipeline

```
Per-sample loss  -->  2-component GMM  -->  Clean/Noisy split  -->  Consensus (BGE ∩ TSE)  -->  Label recalibration
                                                                                                       |
                                                                                          TAL loss with corrected labels
```

### Key Components

**A. Confident Consensus Division (CCD)**

1. Compute per-sample loss for each (image, text) pair
2. Fit 2-component GMM: component k=0 (low loss = clean), k=1 (high loss = noisy)
3. Posterior: `p(k=0 | loss_i) > 0.5` --> clean, else noisy
4. **Consensus from two branches** (BGE + TSE):
   - Both say clean --> **Confident Clean** (keep label = 1)
   - Both say noisy --> **Confident Noisy** (flip label = 0)
   - Disagreement --> **Uncertain** (random {0, 1})
5. Recalibrated labels used in loss computation

**B. Triplet Alignment Loss (TAL)**

Problem: standard triplet loss uses the single hardest negative, which is catastrophic if that negative is a noisy sample.

Solution: TAL replaces `max negative` with a log-sum-exp over all negatives:
```
L_tal = [m - S+ + tau * log(sum_j q_ij * exp(S(I_i, T_j) / tau))]+
```
where `q_ij = 1 - l_ij` masks out positives.

This distributes gradient across all negatives proportional to their difficulty, instead of concentrating it on one potentially-noisy hardest negative.

**C. Dual Embedding (BGE + TSE)**

- **BGE**: global [CLS] token similarity (fast, coarse)
- **TSE**: attention-weighted top-K local token similarity (detailed, fine-grained)
- Consensus between the two provides robustness: a truly noisy pair is likely identified by both branches

### Strengths
- Directly addresses false positives (noisy correspondences)
- GMM on losses is model-agnostic and doesn't assume distribution shape a priori
- Consensus mechanism reduces false detection of noise
- TAL is a principled replacement for hard-negative triplet loss

### Weaknesses
- GMM fitting adds overhead every epoch
- Random labels for uncertain samples is crude — could be improved with soft labels
- Dual branch architecture (BGE + TSE) increases model complexity
- Does not address false negatives

---

## 4. Comparative Analysis

| Dimension | mSigLIP (Current) | FNM | RDE |
|-----------|-------------------|-----|-----|
| **Noise type addressed** | Neither (only hard negatives) | False Negatives | False Positives |
| **Detection method** | N/A | Bayesian posterior on similarity distributions | GMM on per-sample losses |
| **Correction strategy** | N/A | Adaptive margin (soften penalty for likely FN) | Label recalibration (flip/randomize noisy labels) |
| **Loss function** | Circle + N-ITC + CITC + SimCLR | Modified contrastive with margin | TAL (soft triplet) |
| **Hard-negative handling** | Circle Loss (logsumexp, gamma=128) | Implicit via margin adjustment | TAL (logsumexp over all negatives) |
| **Statistical basis** | None (per-batch only) | MoC queues (cross-batch Gaussian statistics) | GMM on accumulated losses |
| **Feature levels** | Global only (CLS token) | Global + Local (attention-selected) | Global + Local (dual embedding) |
| **Curriculum** | Yes (Circle weight ramp) | No | No |
| **Backbone** | SigLIP (LoRA) | CLIP (full fine-tune) | CLIP (full fine-tune) |
| **Extra memory** | None | 4 FIFO queues + momentum encoder | None (GMM is lightweight) |
| **Architecture change needed** | N/A | Moderate (queues, momentum encoder) | Significant (dual branch) |

### How FNM and RDE Complement Each Other

FNM and RDE are **orthogonal** — they address different noise types with different mechanisms:

- FNM's Bayesian detection operates on **similarity distributions** of the current model to find negatives that look like positives
- RDE's CCD operates on **loss values** to find positives that the model struggles to learn (because they're actually wrong)

A complete noise-robust system would need **both**:
- Detect and soften false negatives (FNM-style) so the model doesn't push apart same-person embeddings
- Detect and correct false positives (RDE-style) so the model doesn't pull together different-person embeddings

---

## 5. Adaptation Ideas for mSigLIP

### Idea A: FNM-Lite — Bayesian False Negative Detection for Circle Loss
**Effort: Medium | Impact: High**

Adapt FNM's core idea specifically to Circle Loss, without the full MoC infrastructure.

**How it works:**
1. Each training step, collect `s_p` (positive similarities) and `s_n` (negative similarities) from Circle Loss computation — these are **already computed** in `compute_cross_modal_circle()`
2. Maintain running statistics (mean, std) of `s_p` and `s_n` with exponential moving average (no queues needed, just 4 scalars)
3. For each negative pair, compute `P(FN|s)` using the Bayesian formula
4. Modify Circle Loss's `alpha_n`: scale down alpha for likely false negatives
   ```python
   # Current
   alpha_n = clamp_min(s_n + m, 0)
   # Modified
   fn_prob = bayesian_fn_prob(s_n, mu_p, sigma_p, mu_n, sigma_n)
   alpha_n = clamp_min(s_n + m, 0) * (1 - fn_prob)  # suppress FN gradient
   ```

**Why this fits mSigLIP:**
- Circle Loss already computes all the similarities needed
- Running EMA statistics are trivial to add (no queues, no momentum encoder)
- Modification is 5-10 lines in `objectives.py`
- Can be controlled via curriculum: disable FN detection in early epochs when statistics are unreliable

**Changes:**
- `model/objectives.py`: add running stats + Bayesian prob to `compute_cross_modal_circle()`
- `config/loss/cir_msiglip.yaml`: add `fn_detection: true`, `fn_prior: 0.01`, `fn_threshold: 0.5`

---

### Idea B: Loss-Based Noise Detection for N-ITC (RDE-Inspired)
**Effort: Medium | Impact: Medium-High**

Adapt RDE's memorization-based noise detection without the dual-branch architecture.

**How it works:**
1. Track per-sample N-ITC loss with EMA: `ema_loss[i] = 0.9 * ema_loss[i] + 0.1 * loss_i`
2. Every N epochs, fit a 2-component GMM to the accumulated losses
3. Compute clean probability `p(clean | loss_i)` for each sample
4. Weight the N-ITC loss: `weighted_loss = p(clean) * nitc_loss_per_sample`
5. Optionally: completely discard samples with `p(clean) < 0.1`

**Simplifications vs RDE:**
- No dual branch — use single mSigLIP encoder (we already have soft targets providing some robustness)
- No label flipping — just down-weight noisy samples (softer, less risky)
- GMM fitting every 5 epochs (not every epoch) to reduce overhead

**Changes:**
- `model/tbps.py`: maintain per-sample loss buffer, periodic GMM fitting
- `model/objectives.py`: add sample-weighted version of `compute_constrative()`
- `config/loss/cir_msiglip.yaml`: add `noise_detection: true`, `gmm_interval: 5`

---

### Idea C: Unified Noise-Aware Circle Loss (Novel)
**Effort: High | Impact: Very High**

Combine FNM's false-negative detection with RDE's false-positive detection into a single modified Circle Loss.

**Key insight**: Circle Loss already has separate treatment for positives (`alpha_p`, `logit_p`) and negatives (`alpha_n`, `logit_n`). We can inject noise-awareness into both branches:

```python
def compute_noise_aware_circle(image_features, text_features, pids,
                                m=0.35, gamma=128,
                                mu_p, sigma_p, mu_n, sigma_n,  # running stats
                                sample_clean_prob=None):        # from GMM
    # ... standard circle loss setup ...

    # FALSE NEGATIVE branch: soften negatives that look like positives
    fn_prob = bayesian_fn_prob(s_n, mu_p, sigma_p, mu_n, sigma_n)
    alpha_n_adjusted = alpha_n * (1 - fn_prob)  # reduce push for likely FN

    # FALSE POSITIVE branch: down-weight positives that are likely mislabeled
    if sample_clean_prob is not None:
        # sample_clean_prob[i] = probability that (image_i, text_i) is clean
        alpha_p_adjusted = alpha_p * sample_clean_prob  # reduce pull for likely FP
    else:
        alpha_p_adjusted = alpha_p

    logit_p = -gamma * alpha_p_adjusted * (s_p - delta_p)
    logit_n = gamma * alpha_n_adjusted * (s_n - delta_n)

    loss = softplus(logsumexp(logit_p, 0) + logsumexp(logit_n, 0))
    return loss
```

**Training schedule:**
- Epoch 0-5: standard N-ITC only (no Circle, no noise detection)
- Epoch 6-10: Circle Loss ramps in, start collecting running stats
- Epoch 11+: enable FN detection in Circle Loss
- Epoch 15+: enable FP detection (GMM needs sufficient history)

**Changes:**
- `model/objectives.py`: new `compute_noise_aware_circle()` function
- `model/tbps.py`: running stat management, periodic GMM, extended curriculum
- `config/loss/cir_msiglip.yaml`: full noise config section

---

### Idea D: Similarity-Distribution Regularization (Novel, Lightweight)
**Effort: Low | Impact: Medium**

Instead of detecting individual noisy samples, **regularize the shape of the similarity distribution** to be naturally resistant to noise.

**How it works:**
Add a regularization term that penalizes overlap between positive and negative similarity distributions:

```python
def distribution_separation_loss(s_p, s_n):
    mu_p, sigma_p = s_p.mean(), s_p.std()
    mu_n, sigma_n = s_n.mean(), s_n.std()

    # Maximize separation between distributions
    separation = (mu_p - mu_n) / (sigma_p + sigma_n + eps)
    return -separation  # negative because we want to maximize
```

**Why it helps with noise:**
- False negatives increase `mu_n` (high-similarity negatives pull the negative mean up)
- False positives decrease `mu_p` (low-similarity positives pull the positive mean down)
- By maximizing distribution separation, the model implicitly resists both types of noise
- No per-sample detection needed — works at the distribution level

**Changes:**
- `model/objectives.py`: add `distribution_separation_loss()` (~10 lines)
- `model/tbps.py`: add to loss aggregation
- `config/loss/cir_msiglip.yaml`: add `dist_sep_weight: 0.05`

---

### Idea E: Momentum Similarity Queue (Infrastructure)
**Effort: Medium | Impact: Enables other ideas**

Implement a simplified version of FNM's MoC that can be used by multiple ideas above.

**Simplification vs FNM:**
- Only 2 queues (global features only — we don't have local features)
- No separate momentum encoder — just detach and store features from the current encoder
- Queue size = 4096 (manageable memory, ~25MB for 768-dim float32)

```python
class SimilarityQueue:
    def __init__(self, dim=768, size=4096):
        self.image_queue = torch.zeros(size, dim)
        self.text_queue = torch.zeros(size, dim)
        self.pid_queue = torch.zeros(size, dtype=torch.long)
        self.ptr = 0

    @torch.no_grad()
    def enqueue(self, img_feats, txt_feats, pids):
        batch_size = img_feats.shape[0]
        self.image_queue[self.ptr:self.ptr+batch_size] = img_feats.detach()
        self.text_queue[self.ptr:self.ptr+batch_size] = txt_feats.detach()
        self.pid_queue[self.ptr:self.ptr+batch_size] = pids.detach()
        self.ptr = (self.ptr + batch_size) % self.size
```

This gives any noise detection method access to ~100x more similarity pairs per step, making Gaussian fitting / GMM / distribution analysis much more accurate.

**Changes:**
- `model/tbps.py`: instantiate queue, enqueue after each forward pass
- Can be used by Ideas A, C, D for richer statistics

---

## 6. Recommended Experiment Order

| Priority | Idea | Rationale |
|----------|------|-----------|
| 1 | **D** (Distribution Separation) | Lowest effort, testable in workspace.ipynb immediately, no infrastructure needed |
| 2 | **A** (FNM-Lite for Circle Loss) | Directly addresses the biggest gap (FN in Circle Loss), moderate effort |
| 3 | **E** (Similarity Queue) | Infrastructure that unlocks better statistics for A and C |
| 4 | **B** (Loss-Based Noise Detection) | Requires per-sample tracking, best done after queue is in place |
| 5 | **C** (Unified Noise-Aware Circle) | Combines A+B, attempt after validating each individually |
