# Mathematical Analysis of Noise-Handling Ideas

## Notation

- $v_i, t_j$: L2-normalized image/text features for sample $i, j$
- $s_{ij} = v_i^\top t_j$: cosine similarity (in $[-1, 1]$)
- $\mathcal{P}, \mathcal{N}$: sets of positive and negative pairs (by PID)
- $m$: margin, $\gamma$: scale factor in Circle Loss
- $B$: batch size

### Current Circle Loss (Baseline)

$$\mathcal{L}_{cir} = \text{softplus}\!\Bigl(\underbrace{\text{LSE}_{(i,j)\in\mathcal{P}}\bigl[-\gamma\,\alpha_p^{ij}(s_{ij} - \delta_p)\bigr]}_{\text{positive branch}} + \underbrace{\text{LSE}_{(i,j)\in\mathcal{N}}\bigl[\gamma\,\alpha_n^{ij}(s_{ij} - \delta_n)\bigr]}_{\text{negative branch}}\Bigr)$$

where $\alpha_p^{ij} = [-s_{ij} + 1 + m]_+$, $\alpha_n^{ij} = [s_{ij} + m]_+$, $\delta_p = 1-m$, $\delta_n = m$.

**Gradient w.r.t. a negative pair** $(i,j) \in \mathcal{N}$:

$$\frac{\partial \mathcal{L}_{cir}}{\partial s_{ij}} = \sigma(\cdot) \cdot \frac{\exp(\gamma\,\alpha_n^{ij}(s_{ij}-\delta_n))}{\sum_{(k,l)\in\mathcal{N}} \exp(\gamma\,\alpha_n^{kl}(s_{kl}-\delta_n))} \cdot \gamma\,\alpha_n^{ij}$$

Key observation: a false negative with high $s_{ij}$ gets **maximum gradient** (large $\alpha_n$, large softmax weight). This is exactly the failure mode we need to fix.

---

## Idea A: FNM-Lite (False Negative Softening in Circle Loss)

### Goal
Reduce the gradient magnitude for negative pairs that are likely false negatives.

### Detection

Maintain running EMA statistics of positive and negative similarity distributions:

$$\hat{\mu}_+^{(t)} = \beta\,\hat{\mu}_+^{(t-1)} + (1-\beta)\,\bar{s}_p^{(t)}, \quad \hat{\sigma}_+^{(t)} = \beta\,\hat{\sigma}_+^{(t-1)} + (1-\beta)\,\text{std}(s_p^{(t)})$$

$$\hat{\mu}_-^{(t)} = \beta\,\hat{\mu}_-^{(t-1)} + (1-\beta)\,\bar{s}_n^{(t)}, \quad \hat{\sigma}_-^{(t)} = \beta\,\hat{\sigma}_-^{(t-1)} + (1-\beta)\,\text{std}(s_n^{(t)})$$

where $\beta \approx 0.99$ (momentum), $\bar{s}$ = batch mean, and superscript $(t)$ = training step.

Model each distribution as Gaussian: $f_+(s) = \mathcal{N}(s;\hat{\mu}_+, \hat{\sigma}_+)$, $f_-(s) = \mathcal{N}(s;\hat{\mu}_-, \hat{\sigma}_-)$.

For each negative pair $(i,j) \in \mathcal{N}$, compute false-negative probability:

$$P_{\text{fn}}(s_{ij}) = \frac{p \cdot f_+(s_{ij})}{p \cdot f_+(s_{ij}) + (1-p) \cdot f_-(s_{ij})}$$

where $p$ is the prior probability of false negatives (hyperparameter, e.g. $p = 0.01$).

### Modification

Replace $\alpha_n^{ij}$ with a suppressed version:

$$\tilde{\alpha}_n^{ij} = \alpha_n^{ij} \cdot (1 - P_{\text{fn}}(s_{ij}))$$

The modified Circle Loss becomes:

$$\mathcal{L}_{A} = \text{softplus}\!\Bigl(\text{LSE}_{\mathcal{P}}\bigl[-\gamma\,\alpha_p^{ij}(s_{ij} - \delta_p)\bigr] + \text{LSE}_{\mathcal{N}}\bigl[\gamma\,\tilde{\alpha}_n^{ij}(s_{ij} - \delta_n)\bigr]\Bigr)$$

### Gradient effect

$$\frac{\partial \mathcal{L}_{A}}{\partial s_{ij}}\bigg|_{(i,j)\in\mathcal{N}} \propto \gamma \cdot \alpha_n^{ij} \cdot (1 - P_{\text{fn}}(s_{ij}))$$

- True negative ($P_{\text{fn}} \approx 0$): gradient unchanged
- Likely false negative ($P_{\text{fn}} \approx 0.8$): gradient reduced by 80%
- Certain false negative ($P_{\text{fn}} \approx 1$): gradient nearly zeroed

### Concern

The EMA statistics are estimated from batch-level data. With $B=32$ and 8 PIDs, there are ~$4 \times 4 = 16$ positive pairs and ~$32^2 - 16 = 1008$ negative pairs per batch. The negative statistics are stable, but **positive statistics are noisy** (only 16 samples). This makes $f_+(s)$ unreliable in early training.

**Mitigation**: Enable detection only after epoch $T_{\text{stats}}$ (e.g., epoch 10) when EMA has accumulated enough history, aligning with the existing curriculum schedule.

---

## Idea B: Loss-Based Noise Detection for N-ITC (RDE-Inspired)

### Goal
Down-weight positive pairs that are likely mislabeled (false positives) by tracking per-sample loss history.

### Detection

For each sample $i$ in the dataset, maintain an EMA of its N-ITC loss:

$$\bar{\ell}_i^{(t)} = \alpha\,\bar{\ell}_i^{(t-1)} + (1-\alpha)\,\ell_i^{(t)}$$

where $\ell_i^{(t)}$ is the per-sample N-ITC loss at step $t$, and $\alpha \approx 0.9$.

Every $K$ epochs, fit a 2-component GMM to $\{\bar{\ell}_i\}_{i=1}^N$:

$$p(\bar{\ell}) = \pi_c \cdot \mathcal{N}(\bar{\ell};\mu_c, \sigma_c^2) + \pi_n \cdot \mathcal{N}(\bar{\ell};\mu_n, \sigma_n^2)$$

where component $c$ (lower mean) = clean, component $n$ (higher mean) = noisy.

Clean probability for sample $i$:

$$w_i = p(k=c \mid \bar{\ell}_i) = \frac{\pi_c \cdot \mathcal{N}(\bar{\ell}_i;\mu_c,\sigma_c^2)}{\pi_c \cdot \mathcal{N}(\bar{\ell}_i;\mu_c,\sigma_c^2) + \pi_n \cdot \mathcal{N}(\bar{\ell}_i;\mu_n,\sigma_n^2)}$$

### Modification

Current N-ITC (sigmoid variant):

$$\mathcal{L}_{\text{nitc}} = -\frac{1}{B}\sum_{i=1}^{B} \sum_{j=1}^{B} \log\sigma(s_{ij}^{\text{scaled}} \cdot y_{ij})$$

Modified with per-sample weighting:

$$\mathcal{L}_{B} = -\frac{1}{B}\sum_{i=1}^{B} w_i \sum_{j=1}^{B} \log\sigma(s_{ij}^{\text{scaled}} \cdot y_{ij})$$

where $w_i \in [0, 1]$ is the clean probability for the $i$-th sample.

### Gradient effect

$$\frac{\partial \mathcal{L}_{B}}{\partial v_i} = w_i \cdot \frac{\partial \mathcal{L}_{\text{nitc},i}}{\partial v_i}$$

- Clean sample ($w_i \approx 1$): full gradient — learned normally
- Noisy sample ($w_i \approx 0.2$): 80% gradient reduction — prevents the model from memorizing wrong associations

### Concern

**Memorization effect assumption**: RDE relies on the principle that networks learn clean patterns first, so noisy samples have persistently higher loss. However, with LoRA (only ~3-5% trainable params), the model may not memorize noisy samples at all — the loss separation between clean and noisy may be **too small** for GMM to detect.

**Mitigation**: Monitor the GMM component separation $|\mu_c - \mu_n| / (\sigma_c + \sigma_n)$ during training. If separation < 1.0, the detection is unreliable — fall back to uniform weights.

---

## Idea C: Unified Noise-Aware Circle Loss

### Goal
Handle both false negatives AND false positives in a single Circle Loss formulation.

### Formulation

Combine Idea A's FN detection with a new FP detection on the positive branch:

$$\mathcal{L}_{C} = \text{softplus}\!\Bigl(\text{LSE}_{\mathcal{P}}\bigl[-\gamma\,\tilde{\alpha}_p^{ij}(s_{ij} - \delta_p)\bigr] + \text{LSE}_{\mathcal{N}}\bigl[\gamma\,\tilde{\alpha}_n^{ij}(s_{ij} - \delta_n)\bigr]\Bigr)$$

where:

$$\tilde{\alpha}_n^{ij} = \alpha_n^{ij} \cdot (1 - P_{\text{fn}}(s_{ij})) \quad \text{(from Idea A)}$$

$$\tilde{\alpha}_p^{ij} = \alpha_p^{ij} \cdot w_i \quad \text{(from Idea B, clean probability)}$$

### Gradient analysis — Negative branch (same as Idea A)

$$\frac{\partial \mathcal{L}_{C}}{\partial s_{ij}}\bigg|_{\mathcal{N}} \propto \gamma \cdot \alpha_n^{ij} \cdot (1 - P_{\text{fn}}(s_{ij}))$$

### Gradient analysis — Positive branch

$$\frac{\partial \mathcal{L}_{C}}{\partial s_{ij}}\bigg|_{\mathcal{P}} \propto -\gamma \cdot \alpha_p^{ij} \cdot w_i$$

- Clean positive ($w_i \approx 1$): full pull toward each other
- Noisy positive ($w_i \approx 0.2$): 80% reduced pull — model doesn't force wrong pairs together

### Interaction between the two branches

Define the effective loss as $\mathcal{L}_C = \text{softplus}(L_p + L_n)$ where $L_p, L_n$ are the positive and negative logsumexp terms. The softplus couples them:

$$\frac{\partial \mathcal{L}_C}{\partial L_p} = \frac{\partial \mathcal{L}_C}{\partial L_n} = \sigma(L_p + L_n)$$

This means suppressing one branch (e.g., reducing $L_n$ by detecting FN) also reduces the gradient flowing through the other branch via the shared sigmoid. This is **desirable**: when many negatives are actually false negatives, the model should also be less aggressive on positives (since the positive/negative boundary is uncertain).

### Concern

**Two detection systems running simultaneously** may interfere. If FN detection is too aggressive ($P_{\text{fn}}$ too high), the negative branch contributes little, and the loss collapses toward only optimizing positives. Similarly, if FP detection is too aggressive ($w_i$ too low), the positive branch vanishes.

**Constraint**: enforce a minimum effective weight on both branches:

$$\tilde{\alpha}_n^{ij} = \alpha_n^{ij} \cdot \max(1 - P_{\text{fn}}(s_{ij}),\; \epsilon_n), \quad \epsilon_n = 0.1$$

$$\tilde{\alpha}_p^{ij} = \alpha_p^{ij} \cdot \max(w_i,\; \epsilon_p), \quad \epsilon_p = 0.2$$

---

## Idea D: Similarity-Distribution Separation Regularization

### Goal
Instead of detecting individual noisy samples, regularize the overall shape of pos/neg similarity distributions to maximize their separation — making the model implicitly resistant to both FN and FP.

### Formulation

Given batch similarities $s_p = \{s_{ij} : (i,j) \in \mathcal{P}\}$ and $s_n = \{s_{ij} : (i,j) \in \mathcal{N}\}$:

$$\mathcal{L}_{D} = -\frac{\mu_p - \mu_n}{\sigma_p + \sigma_n + \epsilon} + \lambda_v(\sigma_p^2 + \sigma_n^2)$$

where $\mu_p = \text{mean}(s_p)$, $\sigma_p = \text{std}(s_p)$, and similarly for negatives.

**First term**: maximize the distance between distribution means (Fisher criterion / LDA-like). **Second term**: penalize variance of both distributions (tighter clusters).

### Gradient analysis

$$\frac{\partial \mathcal{L}_D}{\partial s_{ij}}\bigg|_{\mathcal{P}} = -\frac{1}{|\mathcal{P}|(\sigma_p + \sigma_n + \epsilon)} + \frac{2\lambda_v(s_{ij} - \mu_p)}{|\mathcal{P}|}$$

$$\frac{\partial \mathcal{L}_D}{\partial s_{ij}}\bigg|_{\mathcal{N}} = \frac{1}{|\mathcal{N}|(\sigma_p + \sigma_n + \epsilon)} + \frac{2\lambda_v(s_{ij} - \mu_n)}{|\mathcal{N}|}$$

For a **false negative** (true positive labeled as negative) with high $s_{ij}$:
- The second term $2\lambda_v(s_{ij} - \mu_n)$ is large and positive, pushing $s_{ij}$ back toward $\mu_n$
- But since the first term pushes $\mu_n$ down globally, the net effect is weaker than Circle Loss's targeted punishment
- Result: FN still gets some incorrect gradient, but **much less** than Circle Loss because $\mathcal{L}_D$ operates on distribution statistics, not individual pairs

For a **false positive** (wrong pair labeled as positive) with low $s_{ij}$:
- The variance penalty $2\lambda_v(s_{ij} - \mu_p)$ is negative (since $s_{ij} < \mu_p$), which would push it toward $\mu_p$
- But the force is proportional to deviation from mean — an FP that's far from the positive mean gets pulled harder
- Result: still imperfect, but the regularization prevents the FP from dragging the entire positive distribution down

### Concern

This is a **weak** form of noise handling compared to A/B/C. It doesn't detect or correct individual noisy samples — it just makes the model more robust at the distribution level. The Fisher-criterion objective may also conflict with Circle Loss which optimizes individual pair margins rather than distribution statistics.

**Best used as**: a lightweight complement to other methods, not a standalone solution. Add with small weight $\lambda_D \approx 0.05$.

---

## Idea E: Similarity Queue (Infrastructure)

### Goal
Provide richer cross-batch statistics for any detection method (A, C, or D).

### Mechanism

Maintain a FIFO queue $Q$ of size $M$ storing detached features:

$$Q^{(t)} = \{(v_k, t_k, \text{pid}_k)\}_{k=1}^{M}$$

After each forward pass, enqueue current batch features (detached, no grad):

$$Q^{(t)} \leftarrow \text{concat}(Q^{(t-1)}[B:], \; \{(\bar{v}_i, \bar{t}_i, \text{pid}_i)\}_{i=1}^{B})$$

where $\bar{v}_i = v_i.\text{detach}()$.

### Enhanced statistics for Idea A

Without queue (batch only):
- Positive pairs: ~16 per batch
- Negative pairs: ~1008 per batch
- $\hat{\sigma}_+$ very noisy

With queue ($M = 4096$):
- Positive pairs: up to $\sum_{\text{pid}} n_{\text{pid}}^2$ across queue
- Negative pairs: ~$M^2$
- $f_+(s)$ and $f_-(s)$ estimated from thousands of pairs instead of 16/1008

### Staleness analysis

Features in the queue are from previous steps, so they're slightly stale. For a feature stored $k$ steps ago, the staleness error is approximately:

$$\|v_i^{(t-k)} - v_i^{(t)}\| \approx k \cdot \eta \cdot \|\nabla_{v_i}\mathcal{L}\|$$

With LoRA's small learning rate ($\eta \sim 10^{-5}$) and limited trainable parameters (~3-5%), this error is small. For queue size $M = 4096$ and $B = 32$, the oldest features are $M/B = 128$ steps old. With typical gradient norms ~0.01:

$$\text{staleness} \approx 128 \times 10^{-5} \times 0.01 \approx 1.3 \times 10^{-5}$$

This is negligible relative to feature magnitudes (~1.0 after L2 norm). **No momentum encoder needed** for our LoRA setting — direct detached features are sufficiently fresh.

### Concern

Memory: $2 \times 4096 \times 768 \times 4$ bytes $\approx$ 25 MB — trivial.

Computation: computing similarity with queue adds $O(B \times M)$ operations per step, but these are just matrix multiplies on detached tensors (no grad), so overhead is minimal.

---

## Summary Table

| Idea | Noise Type | Mathematical Mechanism | Gradient Modification | Key Hyperparameters | Risk |
|------|-----------|----------------------|----------------------|-------------------|------|
| **A** | False Negative | Bayesian posterior $P_{\text{fn}}(s)$ via EMA Gaussians | $\alpha_n \leftarrow \alpha_n(1-P_{\text{fn}})$ | $p=0.01$, $\beta=0.99$, $T_{\text{stats}}=10$ | Unreliable $f_+(s)$ with few positives per batch |
| **B** | False Positive | GMM on per-sample loss EMA | $\mathcal{L}_i \leftarrow w_i \cdot \mathcal{L}_i$ | $\alpha=0.9$, GMM interval $K=5$ epochs | LoRA may not show clear memorization gap |
| **C** | Both | A + B combined in Circle Loss | $\tilde{\alpha}_p = \alpha_p \cdot w_i$, $\tilde{\alpha}_n = \alpha_n(1-P_{\text{fn}})$ | All of A + B + $\epsilon_n, \epsilon_p$ floors | Two detectors may over-suppress gradients |
| **D** | Both (weak) | Fisher criterion on similarity distributions | Distribution-level, not per-sample | $\lambda_D=0.05$, $\lambda_v$ | Weak correction; may conflict with Circle Loss |
| **E** | N/A (infra) | FIFO queue of detached features | Enables better statistics for A, C, D | Queue size $M=4096$ | Staleness (negligible with LoRA) |
