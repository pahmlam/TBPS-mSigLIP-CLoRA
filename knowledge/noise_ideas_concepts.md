# Noise-Handling Ideas: Core Concepts

## The Fundamental Problem

In embedding space, Circle Loss pushes/pulls pairs based on PID labels. But labels can be wrong.

```
Embedding Space (2D projection)
                                    
  Correct world:                     What labels say:
                                    
   P1 ●───── "man, white shirt"      P1 ●───── "man, white shirt"       (correct)
   P1 ●───── "tall man, bag"         P2 ●─ ─ ─ "tall man, bag"          (FN: same person, different PID)
   P3 ●───── "woman, red dress"      P3 ●───── "woman, red dress"       (correct)
   P4 ●───── "girl, blue hat"        P4 ●───── "man, black jacket"      (FP: wrong text paired)
                                    
   ─── positive (pull together)       ─── positive     ─ ─ ─ should be positive but treated as negative
```

**False Negative (FN)**: Same person, different PIDs → model pushes them apart (bad)
**False Positive (FP)**: Wrong pair labeled as match → model pulls them together (bad)

Circle Loss with gamma=128 **amplifies** both errors because it focuses on the hardest-to-classify pairs — which are exactly where noise lives.

---

## Current System: No Noise Handling

```
                        Circle Loss
                       ┌───────────┐
  All negatives ──────►│  alpha_n   │──► Push apart (full force)     ← FN gets max punishment
                       │  gamma=128 │
  All positives ──────►│  alpha_p   │──► Pull together (full force)  ← FP gets max reward
                       └───────────┘
                       
  Every pair treated equally. No questions asked.
```

---

## Idea A: FNM-Lite — "Ask before punishing negatives"

### Core concept
Before pushing a negative pair apart, **ask**: "How likely is this negative actually a same-person pair that was mislabeled?"

If the similarity score of a "negative" pair falls in the **positive distribution's territory**, it's suspicious — reduce the punishment.

### Visual intuition

```
  Similarity score distribution:
  
  Negatives                    Positives
  f-(s)                        f+(s)
   ╱╲                            ╱╲
  ╱  ╲                          ╱  ╲
 ╱    ╲         OVERLAP        ╱    ╲
╱      ╲       ╱──────╲       ╱      ╲
╱        ╲    ╱ ?HERE? ╲    ╱        ╲
───────────╲╱────────────╲╱──────────────►  similarity
         mu_n    ▲       mu_p
                 │
          Negatives in this zone
          have high P(FN) — they
          look like positives
```

### Flow

```
  For each negative pair (i,j):
  
  ┌─────────────┐     ┌──────────────────┐     ┌────────────────┐
  │ Compute s_ij │────►│ Where does s_ij  │────►│ Compute P(FN)  │
  │ (similarity) │     │ fall on the two  │     │ via Bayes rule  │
  └─────────────┘     │ Gaussians?       │     └───────┬────────┘
                       └──────────────────┘             │
                                                        ▼
                                              ┌──────────────────┐
                              P(FN) ≈ 0.0 ───►│ Full alpha_n     │──► Push hard (true negative)
                                              │                  │
                              P(FN) ≈ 0.8 ───►│ 0.2 × alpha_n   │──► Gentle push (suspicious)
                                              │                  │
                              P(FN) ≈ 1.0 ───►│ 0.0 × alpha_n   │──► Don't push (likely FN)
                                              └──────────────────┘
```

### What changes

```
  Before (current Circle Loss):          After (Idea A):
  
  alpha_n = [s + m]+                      alpha_n = [s + m]+ × (1 - P_fn)
           │                                        │
           │  Same for all negatives                │  Scaled per-pair
           ▼                                        ▼
  Hard neg ●●●●● → max gradient           True hard neg ●●●●● → max gradient
  Easy neg ○○○○○ → low gradient            False neg     ○○○○○ → suppressed gradient
```

### Key dependency
Gaussian statistics estimated from running EMA — needs ~10 epochs to stabilize. Aligns with existing curriculum (Circle Loss already off for epoch 0-5).

---

## Idea B: Loss-Based Noise Detection — "Samples the model can't learn are probably wrong"

### Core concept
Neural networks learn **clean patterns first, noisy patterns later** (memorization effect). If a positive pair consistently has high loss across many epochs, it's likely a **false positive** (wrong label).

Track each sample's loss over time, then use a GMM to separate "learnable" (clean) from "unlearnable" (noisy).

### Visual intuition

```
  Per-sample loss distribution after several epochs:
  
  Number of
  samples
    │   Clean samples              Noisy samples
    │   (low loss)                 (high loss)
    │     ╱╲                          ╱╲
    │    ╱  ╲                        ╱  ╲
    │   ╱    ╲                      ╱    ╲
    │  ╱      ╲                    ╱      ╲
    │ ╱        ╲──────────────────╱        ╲
    └──────────────────────────────────────────► loss
              mu_c              mu_n
              
    GMM fits two Gaussians. Samples under the right peak
    are likely mislabeled → reduce their training weight.
```

### Flow

```
  ┌────────────────────┐
  │ Each training step: │
  │ Record loss_i for   │──────┐
  │ each sample i       │      │
  └────────────────────┘      │
                               ▼
                    ┌──────────────────┐
                    │ EMA smoothing:   │
                    │ ℓ_i = 0.9×ℓ_i + │
                    │       0.1×loss_i │
                    └────────┬─────────┘
                             │
                    Every K epochs:
                             │
                             ▼
                    ┌──────────────────┐
                    │ Fit 2-component  │
                    │ GMM to {ℓ_i}    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ w_i = P(clean|ℓ) │
                    └────────┬─────────┘
                             │
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
      w_i ≈ 1.0        w_i ≈ 0.5        w_i ≈ 0.1
      Clean sample      Uncertain        Noisy sample
      Full loss         Half loss        Near-zero loss
```

### What changes

```
  Before:                               After:
  
  L_nitc = mean(loss_i)                 L_nitc = mean(w_i × loss_i)
                                        
  All samples equal weight              Noisy samples down-weighted
  
  Sample 42 (FP): ████████ full         Sample 42 (FP): █░░░░░░░ suppressed
  Sample 73 (ok): ████████ full         Sample 73 (ok): ████████ full
```

### Key assumption
LoRA trains only ~3-5% of parameters, so the memorization effect may be **weaker** than full fine-tuning. The clean/noisy loss gap might be small. Need to validate that the GMM can still separate the two components.

---

## Idea C: Unified Noise-Aware Circle Loss — "Guard both doors"

### Core concept
Circle Loss has two branches: one for positives, one for negatives. Inject noise detection into **both**:
- Negative branch: Idea A's Bayesian FN detection → soften alpha_n
- Positive branch: Idea B's loss-based FP detection → soften alpha_p

### Visual intuition

```
                    ┌─────────────────────────────────┐
                    │       Noise-Aware Circle Loss    │
                    │                                  │
  Positive pairs ──►│  alpha_p × w_i (clean prob)     │──► Pull together
                    │  ▲                               │    (reduced if FP suspected)
                    │  │ From Idea B (GMM)             │
                    │                                  │
  Negative pairs ──►│  alpha_n × (1 - P_fn)           │──► Push apart
                    │  ▲                               │    (reduced if FN suspected)
                    │  │ From Idea A (Bayes)            │
                    └─────────────────────────────────┘
```

### How the two guards interact

```
  The softplus coupling creates a natural balance:
  
  L = softplus(L_pos + L_neg)
                  │       │
                  │       └── If FN detection suppresses many negatives,
                  │           L_neg shrinks, sigmoid(L_pos + L_neg) drops,
                  │           which also moderates positive gradients
                  │
                  └────────── If FP detection suppresses many positives,
                              L_pos shrinks, same moderating effect on negatives
                              
  Built-in self-regulation: when one side is uncertain, both sides become cautious.
```

### Safety mechanism

```
  Without floor:                       With floor (epsilon = 0.1):
  
  If both detectors fire aggressively:  Guaranteed minimum gradient:
  
  alpha_p × 0.01 ≈ dead                alpha_p × max(0.01, 0.2) = 0.2 × alpha_p
  alpha_n × 0.02 ≈ dead                alpha_n × max(0.02, 0.1) = 0.1 × alpha_n
  
  → Loss collapses, no learning         → Always some learning signal
```

---

## Idea D: Distribution Separation — "Don't fix individuals, fix the crowd"

### Core concept
Instead of detecting which specific samples are noisy, **push the entire positive and negative similarity distributions apart**. If the distributions are well-separated, individual noisy samples cause less damage because the decision boundary has a wide buffer zone.

### Visual intuition

```
  Before (overlapping):                 After (separated):
  
     neg    pos                            neg           pos
      ╱╲    ╱╲                              ╱╲            ╱╲
     ╱  ╲  ╱  ╲                            ╱  ╲          ╱  ╲
    ╱   ╱╲╱╲   ╲                          ╱    ╲        ╱    ╲
   ╱   ╱ NOISE ╲   ╲                     ╱      ╲──────╱      ╲
  ────────────────────►              ────────────────────────────►
       ▲ FN and FP live here              ▲ Wide gap = noise tolerance
       
  L_D maximizes:  (mu_p - mu_n) / (sigma_p + sigma_n)
                   ─────────────   ──────────────────
                   push means apart   tighten each cluster
```

### Flow

```
  ┌──────────────┐     ┌──────────────┐     ┌─────────────────────────┐
  │ Collect s_p,  │────►│ Compute      │────►│ L_D = -(mu_p - mu_n)   │
  │ s_n from      │     │ mu, sigma    │     │        / (sig_p + sig_n)│
  │ batch         │     │ for each     │     │      + lambda_v × var   │
  └──────────────┘     └──────────────┘     └─────────────────────────┘
                                                        │
                                             Backprop through mu, sigma
                                             to individual similarities
                                                        │
                                                        ▼
                                             All pairs get small nudge
                                             toward better separation
```

### Why it's noise-resistant (but weak)

```
  A single false negative (high-sim negative):
  
  Circle Loss:     "You! High-sim negative! MAXIMUM penalty!"  → catastrophic if FN
  
  Dist Separation: "The negative mean shifted up slightly.      → gentle, distributed
                    Everyone in the negative set, move down         correction
                    a tiny bit."
                    
  Noise is diluted across the entire distribution instead of concentrated on one pair.
  But also: correction is diluted. Weak standalone, good complement.
```

---

## Idea E: Similarity Queue — "Remember the recent past"

### Core concept
All detection methods (A, B, C, D) need statistics — means, variances, distributions. A single batch of 32 samples gives noisy estimates. A queue of 4096 recent features gives **128x more data** for the same cost.

### Visual intuition

```
  Without queue (batch only):            With queue:
  
  Step t:                                Step t:
  ┌────────────────────┐                 ┌────────────────────┐
  │  Batch (32 samples) │                 │  Batch (32 samples) │ ← current (has gradients)
  │  16 pos pairs       │                 │                     │
  │  1008 neg pairs     │                 │  Queue (4096)       │ ← historical (detached)
  │                     │                 │  ~1M neg pairs      │
  │  Noisy statistics   │                 │  ~10K pos pairs     │
  └────────────────────┘                 │                     │
                                          │  Stable statistics  │
                                          └────────────────────┘
```

### FIFO mechanism

```
  Queue state over time:
  
  Step 1:  [batch_1, _______, _______, _______]
  Step 2:  [batch_1, batch_2, _______, _______]
  Step 3:  [batch_1, batch_2, batch_3, _______]
  ...
  Step 128: [batch_1, batch_2, ...,    batch_128]  ← queue full
  Step 129: [batch_2, batch_3, ...,    batch_129]  ← oldest evicted (FIFO)
  
  Oldest feature age: 128 steps × lr=1e-5 × grad~0.01 ≈ 0.00001 drift
  
  With LoRA's tiny updates, features barely change in 128 steps.
  No momentum encoder needed (unlike FNM's MoC).
```

### What it enables

```
  Idea A (FN detection):
    Without queue: f+(s) from 16 pos pairs   → sigma_+ very noisy
    With queue:    f+(s) from ~10K pos pairs  → sigma_+ reliable
    
  Idea D (distribution separation):
    Without queue: mu, sigma from 1 batch    → high variance between steps
    With queue:    mu, sigma from 128 batches → smooth, stable optimization
```

---

## How They Compose

```
                        ┌─────────────┐
                        │  Idea E     │
                        │  Queue      │
                        │  (infra)    │
                        └──────┬──────┘
                               │ provides rich statistics to
                    ┌──────────┼──────────┐
                    ▼          ▼          ▼
              ┌──────────┐ ┌─────────┐ ┌──────────┐
              │ Idea A   │ │ Idea D  │ │ Idea C   │
              │ FN detect│ │ Dist Sep│ │ Unified  │
              │ (neg br.)│ │ (reg.)  │ │ (both)   │
              └────┬─────┘ └────┬────┘ └──┬───┬───┘
                   │            │         │   │
                   │            │    ┌────┘   └────┐
                   │            │    │              │
                   ▼            ▼    ▼              ▼
              ┌────────────────────────┐    ┌────────────┐
              │     Circle Loss        │    │   N-ITC    │
              │  (negative branch)     │    │  (positive │
              │                        │    │   branch)  │◄── Idea B (FP detect)
              └────────────────────────┘    └────────────┘


  Recommended build order:
  
  1. D alone          (quick win, test in notebook)
  2. A alone          (biggest impact on Circle Loss FN problem)
  3. E + A            (better statistics for A)
  4. E + A + B        (add FP detection on N-ITC)
  5. C = A + B merged (unified, after validating each independently)
```
