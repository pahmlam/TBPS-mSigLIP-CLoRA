# Implementation Plan: Idea C — Unified Noise-Aware Circle Loss

## Context

The current Circle Loss treats all PID-based labels as ground truth. In reality, TBPS datasets contain:
- **False Negatives (FN)**: Same person with different PIDs — Circle Loss pushes them apart with gamma=128 amplification
- **False Positives (FP)**: Wrong image-text pair labeled as matching — Circle Loss pulls them together

Idea C injects noise awareness into **both branches** of Circle Loss:
- **Negative branch**: Bayesian FN detection softens `alpha_n` for likely false negatives (from Idea A / FNM)
- **Positive branch**: GMM-based FP detection down-weights `alpha_p` for likely false positives (from Idea B / RDE)

Current best: R@1 = 52.28%. This change targets the ~10-15% of gradient energy wasted on noisy pairs.

---

## Architecture Overview

```
                    ┌─────────────────────────────────┐
                    │   Noise-Aware Circle Loss (NACIR)│
                    │                                  │
  Positive pairs ──►│  alpha_p × max(w_i, ε_p=0.2)   │──► Pull together (reduced if FP)
                    │  ▲  w_i from GMM on loss history │
                    │                                  │
  Negative pairs ──►│  alpha_n × max(1-P_fn, ε_n=0.1) │──► Push apart (reduced if FN)
                    │  ▲  P_fn from Bayesian posterior  │
                    └─────────────────────────────────┘
```

Key design decisions:
1. **New config flag `NACIR`** — separate from existing `CIR` for safe A/B comparison
2. **State in `NoiseAwareCircleState` (nn.Module)** — EMA stats + per-sample buffers as registered buffers (survive checkpointing)
3. **Loss function stays stateless** — `compute_noise_aware_circle()` takes pre-computed stats as arguments
4. **Per-sample tracking via `batch["id"]`** — already returned by `ImageTextDataset.__getitem__` (data/bases.py:193)

---

## File Changes

### 1. NEW: `model/noise_aware.py` — Core state management

```python
class NoiseAwareCircleState(nn.Module):
    def __init__(self, num_train_samples: int, config: dict): ...
    
    # Registered buffers (checkpoint-safe):
    #   mu_pos, sigma_pos, mu_neg, sigma_neg (EMA similarity stats)
    #   stats_initialized (bool flag)
    #   ema_loss (N,), sample_seen (N,), clean_weights (N,) (per-sample FP tracking)
    #   gmm_fitted (bool flag)
    
    @torch.no_grad()
    def update_ema_stats(self, s_p: Tensor, s_n: Tensor) -> None
        # EMA with beta=0.99; first-call initialization from batch stats
    
    @torch.no_grad()
    def update_sample_losses(self, sample_ids: Tensor, per_sample_losses: Tensor) -> None
        # EMA per-sample loss update; initializes unseen samples directly
    
    @torch.no_grad()
    def refit_gmm(self) -> dict
        # 2-component 1D GMM via EM (pure PyTorch, no sklearn dependency)
        # Returns diagnostic dict {separation, mu_clean, mu_noisy, ...}
        # Falls back to uniform weights if separation < gmm_min_separation
    
    def compute_fn_probabilities(self, s_n: Tensor) -> Tensor
        # P_fn(s) = p*f+(s) / [p*f+(s) + (1-p)*f-(s)] with Gaussian likelihoods
    
    def get_clean_weights_for_batch(self, sample_ids: Tensor) -> Tensor
        # Lookup clean_weights[sample_ids]
    
    @staticmethod
    def _fit_1d_gmm_em(data: Tensor, n_iter: int = 50) -> tuple
        # Pure-PyTorch 2-component 1D GMM — avoids sklearn dependency
```

Hyperparameters (from config):
| Param | Default | Role |
|-------|---------|------|
| `ema_beta` | 0.99 | Momentum for similarity EMA |
| `loss_ema_alpha` | 0.9 | Momentum for per-sample loss EMA |
| `fn_prior` | 0.01 | Prior P(FN) in Bayesian formula |
| `epsilon_n` | 0.1 | Floor for FN suppression (prevents gradient collapse) |
| `epsilon_p` | 0.2 | Floor for FP suppression |
| `gmm_refit_interval` | 5 | Epochs between GMM refits |
| `gmm_min_separation` | 1.0 | Min separation for GMM to be trusted |
| `fn_enable_epoch` | 11 | Epoch to activate FN detection |
| `fp_enable_epoch` | 15 | Epoch to activate FP detection |

### 2. MODIFY: `model/objectives.py` — Add `compute_noise_aware_circle()`

Add new function alongside existing `compute_cross_modal_circle` (unchanged):

```python
def compute_noise_aware_circle(
    image_features: Tensor, text_features: Tensor, pids: Tensor,
    m: float = 0.25, gamma: float = 128,
    fn_stats: dict | None = None,       # {mu_pos, sigma_pos, mu_neg, sigma_neg, fn_prior}
    clean_weights: Tensor | None = None, # (B,) from GMM
    epsilon_n: float = 0.1, epsilon_p: float = 0.2,
) -> tuple[Tensor, dict]:
    """Returns (loss, diagnostics_dict)"""
```

Logic:
1. Normalize, compute sim_mat, build pos/neg masks (identical to existing)
2. Compute standard alpha_p, alpha_n (identical to existing)
3. **FN branch**: If `fn_stats` provided, compute `P_fn(s_n)` via `_bayesian_fn_prob()` helper, then `alpha_n *= max(1 - P_fn, epsilon_n)`
4. **FP branch**: If `clean_weights` provided, map to positive pairs via `min(w[i], w[j])` for pair `(i,j)`, then `alpha_p *= max(pair_w, epsilon_p)`
5. Compute logits and softplus loss (identical to existing)
6. Compute per-sample Circle Loss (detached, for GMM tracking) — per-row decomposition using masked logsumexp
7. Return `(loss, {'s_p': ..., 's_n': ..., 'per_sample_loss': ..., 'fn_prob_mean': ..., 'clean_weight_mean': ...})`

Also add a static helper:
```python
def _bayesian_fn_prob(s_n, mu_pos, sigma_pos, mu_neg, sigma_neg, fn_prior):
    # Gaussian likelihood ratio → Bayesian posterior
```

### 3. MODIFY: `model/tbps.py` — Add NACIR block to forward()

Changes to `__init__`:
```python
def __init__(self, config, backbone, num_train_samples=0):
    # ... existing code ...
    if config.loss.get("NACIR", None):
        from model.noise_aware import NoiseAwareCircleState
        self.noise_state = NoiseAwareCircleState(
            num_train_samples, config.loss.get("nacir_config", {})
        )
```

Changes to `forward()` — add new section after existing CIR block (line ~244):
```python
# --- C2. Noise-Aware Circle Loss (NACIR) ---
if self.config.loss.get("NACIR", None) and current_circle_weight > 0:
    # Determine active detections from curriculum
    fn_active = current_epoch >= self.noise_state.fn_enable_epoch
    fp_active = current_epoch >= self.noise_state.fp_enable_epoch
    
    # Build fn_stats dict if FN detection is active
    fn_stats = None
    if fn_active and self.noise_state.stats_initialized:
        fn_stats = {4 scalar values from self.noise_state}
    
    # Get clean weights if FP detection is active
    clean_weights = None
    if fp_active and self.noise_state.gmm_fitted and "id" in batch:
        clean_weights = self.noise_state.get_clean_weights_for_batch(batch["id"])
    
    # Compute loss
    nacir_loss, diag = objectives.compute_noise_aware_circle(...)
    
    # Update state (no_grad)
    self.noise_state.update_ema_stats(diag['s_p'], diag['s_n'])
    if "id" in batch:
        self.noise_state.update_sample_losses(batch["id"], diag['per_sample_loss'])
    
    # MVS augmentation (same pattern as existing CIR MVS)
    if self.config.loss.get("MVS", None):
        aug_nacir_loss, _ = objectives.compute_noise_aware_circle(...)
        nacir_loss = (nacir_loss + aug_nacir_loss) / 2
    
    ret.update({"nacir_loss": nacir_loss * current_circle_weight})
    # Log diagnostics
    ret.update({k: v for k, v in diag.items() if k.endswith('_mean')})

elif self.config.loss.get("NACIR", None):
    ret.update({"nacir_loss": torch.tensor(0.0, ...)})
```

**Important**: When `NACIR: true`, skip the existing `CIR` block. Add guard:
```python
if self.config.loss.get("CIR", None) and not self.config.loss.get("NACIR", None) and current_circle_weight > 0:
    # existing CIR block
```

### 4. MODIFY: `lightning_models.py`

- Add `num_train_samples` param flowing through `__init__` → `_initialize_model` → `TBPS()`
- Add `on_train_epoch_end` hook for periodic GMM refitting:

```python
def on_train_epoch_end(self) -> None:
    if hasattr(self.model, 'noise_state'):
        epoch = self.trainer.current_epoch
        ns = self.model.noise_state
        if (epoch >= ns.fp_enable_epoch and
            epoch % ns.gmm_refit_interval == 0 and
            ns.sample_seen.any()):
            diag = ns.refit_gmm()
            self.log_dict({f"gmm_{k}": v for k, v in diag.items()}, on_epoch=True)
```

### 5. MODIFY: `trainer.py`

Pass dataset size:
```python
model = LitTBPS(
    config,
    num_iters_per_epoch=len(train_loader),
    num_train_samples=len(dm.train_set),  # NEW
)
```

### 6. MODIFY: `config/loss/cir_msiglip.yaml`

Append NACIR config section:
```yaml
# Noise-Aware Circle Loss (Idea C)
NACIR: false

nacir_config:
  ema_beta: 0.99
  fn_prior: 0.01
  epsilon_n: 0.1
  loss_ema_alpha: 0.9
  epsilon_p: 0.2
  gmm_refit_interval: 5
  gmm_min_separation: 1.0
  fn_enable_epoch: 11
  fp_enable_epoch: 15
```

---

## Extended Curriculum

Detectors are **cumulative** — once activated, they stay on for the rest of training. The rows below describe what is *newly active* at each phase boundary, not disjoint time windows.

| Phase | Epochs | NACIR weight | FN detection | FP detection | Net behavior |
|-------|--------|:------------:|:------------:|:------------:|--------------|
| Warmup | 0-5 | 0.0 | off | off | NACIR disabled (inherited warmup) |
| Ramp + stats collection | 6-10 | 0 → ~0.033 | off | off | NACIR behaves as vanilla Circle Loss; EMA stats accumulating silently |
| FN-only | 11-14 | ~0.04 → ~0.07 | **on** | off | Negative branch softened for likely false negatives; positive branch unchanged |
| Full Idea C | 15-20 | ~0.08 → 0.1 | **on** | **on** | Both branches noise-aware — the target configuration |
| Stable | 21-60 | 0.1 | **on** | **on** | Full Idea C at full weight; GMM refit every 5 epochs |

Visualization:

```
Epoch:    0────5────10────15────20────────────60
NACIR:    ────░░░░░▒▒▒▒▒▓▓▓▓▓████████████████   (ramp → stable)
FN det:   ───────────────●════════════════════   (on at 11, stays on)
FP det:   ───────────────────●════════════════   (on at 15, stays on)
```

Reuses the existing `current_circle_weight` curriculum (`tbps.py:169-180`) — no changes to the weight schedule itself. Only the detector activation gates are new.

---

## Implementation Order

| Step | File | Type | Description |
|------|------|------|-------------|
| 1 | `model/noise_aware.py` | NEW | NoiseAwareCircleState class with all state management |
| 2 | `model/objectives.py` | EDIT | Add `compute_noise_aware_circle()` + `_bayesian_fn_prob()` |
| 3 | `config/loss/cir_msiglip.yaml` | EDIT | Add NACIR section |
| 4 | `model/tbps.py` | EDIT | Wire NACIR in `__init__` + `forward()` |
| 5 | `lightning_models.py` | EDIT | Pass num_train_samples + GMM epoch hook |
| 6 | `trainer.py` | EDIT | Pass `len(dm.train_set)` |
| 7 | `workspace.ipynb` | ADD CELLS | Validation cells (see below) |
| 8 | `docs/knowledge.md` | EDIT | Document Idea C implementation |
| 9 | `changelog/training/changelog.md` | EDIT | Log the change |

---

## Verification Plan

### workspace.ipynb Validation (before training)

**Section 3 — Loss Playground:**
- Compute NACIR loss with `fn_stats=None, clean_weights=None` → must match existing Circle Loss exactly
- Compute with simulated `fn_stats` (from test-set Gaussian fit) → verify finite, similar scale
- Parameter sweep: vary `fn_prior` [0.001, 0.01, 0.05], `epsilon_n` [0.05, 0.1, 0.2]

**Section 4 — Gradient Analysis (critical):**
- Compare gradient energy on top-10% hard negatives: NACIR vs vanilla Circle Loss
- Verify NACIR suppresses gradient on overlap-zone negatives (likely FN)
- Verify true hard negatives still receive strong gradient

**Visualizations:**
- Plot P_fn(s) curve overlaid on pos/neg similarity histograms
- Plot per-sample loss distribution + GMM fit (if applicable)

### Integration Tests (after implementation)

1. `NACIR: false` → existing behavior unchanged (regression check)
2. `NACIR: true` with `fn_enable_epoch: 999, fp_enable_epoch: 999` → NACIR acts as vanilla Circle Loss
3. Full 5-epoch sanity run → verify loss finite, diagnostics logged, no crashes
4. Check W&B logs for: `nacir_loss`, `fn_prob_mean`, `clean_weight_mean`, `gmm_separation`

---

## Risk Mitigations

| Risk | Mitigation |
|------|-----------|
| GMM can't separate clean/noisy with LoRA | `gmm_min_separation` threshold → falls back to uniform weights |
| Both detectors over-suppress → gradient collapse | Epsilon floors: ε_n=0.1, ε_p=0.2 guarantee minimum 10%/20% gradient |
| EMA stats unreliable in early training | FN detection only activates at epoch 11 (5 epochs of stats collection) |
| Per-sample loss buffer memory | VN3K ~12K samples × 2 float32 buffers = ~96KB (negligible) |
| `batch["id"]` not flowing through | Already confirmed in `data/bases.py:193` — returned as `"id": index` |
