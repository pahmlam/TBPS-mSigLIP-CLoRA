# Rotated W8A8 + QAT: Mathematics & Method for the mSigLIP Vision + Text Encoders on RB3 Gen2 (HTP v68)

> **Scope:** this is the **theory and method** document for the vision-encoder deployment branch and the matching text-encoder finite/f32/link-safe mask extension. It contains the mathematics of every transform and the reasoning behind every design choice. It contains **no commands, scripts, or code** — for the reproducible command sequence, AI Hub job IDs, and artifact paths, see the consolidated history in [`deployment/docs/journal/[deploy-master].md`](journal/[deploy-master].md).
>
> **Source checkpoint:** LoRA + Curriculum Circle, seed 2400 (FP32 reference ≈ paper 52.28).
>
> **Target device:** Qualcomm RB3 Gen2 / QCS6490 / Hexagon HTP **v68**.
>
> **End-to-end result:** the final direct RB3 board run is **both-INT8 W8A8**: vision v9 context binary + split-text context binary reach **T2I Rank@1 = 50.35** and **I2T Rank@1 = 54.20**, passing the ≥50 deploy target with a `-1.93` T2I drop from the paper baseline `52.28`. The older both-INT8 QDQ proxy remains a reference at **50.25 / 52.95**.
>
> **Vision status:** vision-only QDQ proxy reaches **T2I Rank@1 = 50.85** (learned rotation, QAT v8; `-1.43` vs paper baseline `52.28`). The final RB3 board context binary is **v9**, reaching **50.35** T2I R@1 and **54.55** I2T R@1 with **32.54 ms/image** throughput.
>
> **Text status:** the *full-graph* text binary links but is **unusable on board** — its output ignores `input_ids` because HTP v68 breaks the dynamic 250k-row embedding `Gather` (§12A.8). The **deployable text path is the split-encoder**: RB3-side embedding lookup feeds `inputs_embeds` to an HTP transformer. It is board-verified — board fidelity **0.9951 / 0.9926** (matching the QDQ proxy), text-isolation board **T2I R@1 = 51.30**, and runtime **7.87 ms/query**.

This document explains the hardware constraints that *force* the pipeline, the mathematics of each transform (rotation, learned rotation, quantization-aware training, finite attention masking), why earlier candidates failed, and the acceptance gates that define success. The decisive acceptance metric throughout is **retrieval Rank@1**, not cosine — cosine is only a fidelity proxy (§8).

---

## 0. Result Summary

The journey from naive W8A8 (which collapses retrieval) to the deployable best:

| Stage | Method change | QDQ cosine (mean/min) | Vision-only T2I R@1 | Target (≥50) |
|---|---|---:|---:|:--:|
| FP32 reference | merged baseline | 1.000 | **52.40** | — |
| Rotation only | mean-preserving rotation + W8A8 | 0.8975 / 0.8747 | 45.42 | FAIL |
| QAT v1 | + fake-quant distill (per-sample) | 0.9223 / 0.8917 | 46.92 | FAIL |
| QAT v2 | + per-tensor fake-quant | 0.9281 / 0.9093 | 47.80 | FAIL |
| QAT v3 | + EMA observer (deploy-faithful) | 0.9353 / 0.919 | 48.20 | FAIL |
| QAT v4 | + pooling-head fake-quant | 0.9364 / 0.9091 | 48.50 | FAIL (board-verified) |
| QAT v5 | + per-linear fake-quant | 0.9437 / 0.9311 | 49.25 | FAIL |
| QAT v6 | + attention-matmul fake-quant | 0.9491 / 0.9266 | 49.30 | FAIL (random-rotation ceiling) |
| QAT v7 | v6 coverage + cosine LR + lr 2e-5 | 0.9485 / 0.9083 | 48.38 | FAIL (regress) |
| **QAT v8** | **learned rotation + recipe v6** | **0.9606 / 0.9447** | **50.85** | **PASS (QDQ; board v8 = 50.20)** |

Two facts frame everything below:

- **v6 is the ceiling of *random* rotation.** Widening fake-quant coverage (v3→v6) climbs from 48.20 to 49.30, then saturates — diminishing returns once every activation tensor is covered.
- **v8 breaks that ceiling by changing the rotation itself**, not the QAT. Replacing the random mean-preserving `Q` with a *learned* one (same QAT recipe as v6) lifts T2I R@1 by **+1.55** to 50.85 and improves QDQ fidelity on both mean and min. Because only `Q` differs between v6 and v8, this is a clean ablation isolating "learned vs random rotation." (v7 is recorded as a cautionary regression: changing the LR schedule and doubling LR overshot the quantization minimum.)
- **v9 is a final recipe refinement, not a new theory.** It keeps the same mean-preserving learned-rotation objective and the same QAT coverage, but uses a larger rotation calibration/search budget (512 calibration images, 8000 steps), folds the best observed rotation rather than the last optimizer step, and runs vision QAT longer (25 epochs). Its role is to lift the board deployment margin, not to introduce a separate mathematical method.

Text follows the same v8 recipe but needs two graph-level fixes plus one link-safe graph rewrite:

| Text branch | Mask handling | Link-safe mask graph | Attention QDQ scale | QDQ cosine (mean/min) | Text-isolation T2I R@1 | Target |
|---|---|---|---:|---:|---:|:--:|
| v8 text, original ONNX | `-FLT_MAX` mask in `scores+mask` | `attention_mask int64 -> Cast(FLOAT)` island | ~`1e32` | collapse | collapse | FAIL |
| **v8 text, finite + f32 + linksafe mask** | **replace mask with `-32.0` before AI Hub quantize** | **`input_ids` integer, `attention_mask` float32 0/1, rewrite `(1-mask)*-32`** | **0.3523 max** | **0.9949 / 0.9912** | **51.65** | **Proxy PASS; full-graph board rejected; split-text board PASS** |

The off-board C1 deploy proxy uses both QDQ graphs together and is retained as the reference before direct board execution:

| Combo | T2I R@1 | I2T R@1 | Interpretation |
|---|---:|---:|---|
| Paper baseline | **52.28** | — | main reporting baseline |
| Local FP32 sanity | 52.40 | 55.30 | pipeline reproduction only, not the reporting baseline |
| Vision INT8 only | 50.85 | 52.90 | `-1.43` T2I vs paper baseline |
| Vision INT8 board v8 | 50.20 | 54.50 | full gallery RB3 run; `-0.65` T2I vs QDQ proxy |
| Vision INT8 board v9 | 50.35 | 54.55 | final board-verified vision tower |
| Text INT8 only | 51.65 | 55.55 | `-0.63` T2I vs paper baseline |
| Text INT8 board | 51.30 | 54.80 | split-text RB3 lookup + HTP transformer |
| Both INT8 QDQ proxy | 50.25 | 52.95 | off-board reference, `-2.03` T2I vs paper baseline |
| **Both INT8 board** | **50.35** | **54.20** | **FINAL PASS**, `-1.93` T2I vs paper baseline |

---

## 1. Why This Pipeline Exists (Hardware Constraints)

RB3 Gen2 runs the Hexagon **HTP v68**. The QNN context-binary linker on v68 imposes hard constraints that eliminate most "easy" quantization paths:

1. **Floating-point graph I/O is rejected.** Context binaries require integer (quantized) I/O.
2. **Internal floating-point fallback fails to link.** Leaving sensitive layers in float (the "`_float` surgery" approach) passes local ONNX fidelity but the linker rejects internal float tensors.
3. **16-bit activations (A16) are only partially supported on v68.** Activation×activation matmuls (attention) and LayerNorm in A16 require a newer arch (≥ v73). W8A16 reaches QDQ fidelity ≈ 0.9997 but the **link fails** on v68 at the attention matmul / LayerNorm nodes.
4. **All-INT8 (W8A8) is the path v68 supports broadly.**

The conflict: the model *needs* high precision exactly where v68 *refuses* it (residual stream + LayerNorm). So the only deployable path is all-W8A8 — and we must make the network **tolerate** W8A8 instead of asking the hardware for more bits.

Naive per-tensor W8A8 collapses retrieval (cosine ≈ 0.14–0.17). This is **not** an export or preprocessing bug — static ONNX vs PyTorch is ≈ 1.0. The failure is in **activation quantization** inside the encoder, with two root causes, fixed by two transforms:

- **Decomposed cubic GELU** (an export artifact) → fixed by **opset-20 GELU fusion** (§4).
- **Massive activations / channel outliers** in the residual stream → fixed by **rotation** (§5–§6), then residual per-tensor error is recovered by **QAT** (§7).

Clipping the outliers was tried and failed: the outlier channels carry real signal, so clipping destroys information. The correct operation is to *redistribute* the energy (rotation), not remove it.

---

## 2. Pipeline Overview (conceptual)

The deployable path is a fixed sequence of mathematically motivated transforms. Each block lists *what it does* and *the invariant it preserves*. In command order, opset-20 fusion happens when the rotated/QAT PyTorch model is exported to ONNX; it is not a separate pre-rotation weight transform.

```text
            ┌─────────────────────────────────────────────┐
            │  FP32 reference model  (T2I R@1 = 52.40)   │
            └─────────────────────┬───────────────────────┘
                                  │
            ┌─────────────────────▼───────────────────────┐
   [1]      │  MERGE LoRA                                │
            │  W ← W_base + (α/r)·B·A                    │   exact identity
            └─────────────────────┬───────────────────────┘   (no behavior change)
                                  │
            ┌─────────────────────▼─────────────────────--──┐
   [2]      │  ROTATE residual stream                      │   ‖Qx‖₂ = ‖x‖₂
            │  x → Qx,  Q orthogonal,  Q·1 = 1             │   LN(Qx) = Q·LN(x)
            │  folded offline into weights (no runtime op) │   ⇒ FP32-invariant
            │   |                                          │   μ ↓ to ≈√(2 ln d)
            │   └── learned Q  ◀ v9   (min Σ max|aQᵀ|²)    │   max|a|: 124 → 14.6
            └─────────────────────┬─────────────────────--──┘
                                  │   (steps 1–2 change representation, not function)
            ┌─────────────────────▼───────────────────--────┐
   [3]      │  QAT  (teacher–student distillation)         │   trains weight VALUES
            │  per-tensor STE fake-quant + EMA observer    │   to tolerate INT8;
            │  teacher = rotated FP32 (frozen)             │   INT8 emb ≈ FP32 emb
            └─────────────────────┬───────────────────--────┘
                                  │
            ┌─────────────────────▼───────────────────────┐
   [4]      │  EXPORT OPS 20                              │   cubic x³ & norm
            │  fused GELU / fused LayerNormalization       │   internals never
            └─────────────────────┬───────────────────────┘   exposed to quantizer
                                  │
            ┌─────────────────────▼───────────────────--────┐
   [5]      │  W8A8 QUANTIZE → COMPILE → LINK              │   integer I/O,
            │  all-INT8 context binary for HTP v68         │   links on v68
            └─────────────────────┬────────────────────--───┘
                                  │
            ┌─────────────────────▼───────────────────--────┐
   [6]      │  BOARD RUN + EVALUATE                        │   decisive metric:
            │  vision-isolation + final both-INT8          │   T2I R@1 ≥ 50 target
            └──────────────────────────────────────────--───┘
                  v8 QDQ vision-only:        T2I R@1 = 50.85
                  v9 board vision-only:      T2I R@1 = 50.35
                  final board both-INT8:     T2I R@1 = 50.35
```

In words:

1. **Merge LoRA** into the base weights so the exported model is an ordinary inference network (§3).
2. **Rotate the residual stream** by a mean-preserving orthogonal `Q`, folded offline into the weights, to remove channel outliers while preserving LayerNorm semantics (§5–§6). The rotation is either *random* (v1–v7) or **learned** (v8/v9, §6A).
3. **Quantization-aware finetune** (teacher–student distillation, per-tensor straight-through fake-quant, EMA observer) to train the weights to tolerate the deploy-faithful INT8 quantizer (§7).
4. **Export the rotated/QAT model at opset 20** so GELU and LayerNorm are fused in ONNX and their internals are never exposed as quantized tensors (§4, §8.1).
5. **Quantize to W8A8, compile, and link** a context binary with integer I/O for HTP v68 (§8).
6. **Run on board and evaluate** by retrieval Rank@1 — first as a vision-isolation test with FP32 text, then as the final board both-INT8 path with text and image contexts (§8).

The LoRA merge and rotation in steps 1–2 are **output-invariant in FP32**: they change the *representation*, not the function. Step 3 is the only step that changes weight *values*. Step 4 changes the exported graph representation, and must pass the static ONNX-vs-PyTorch gate before any quantized result is trusted (§9).

The vision diagram above is the current interpretation used for the final report: v8 is kept as the first successful learned-rotation/QDQ reference, while v9 is the final board recipe. The phrase "text kept FP32" applies only to the vision-isolation check, not to the final both-INT8 deployment number.

---

## 2A. Theoretical Foundations & Related Work (the *why* behind each choice)

This pipeline is an engineering composition of three research lines — **outlier-aware quantization**, **rotation / incoherence processing**, and **quantization-aware training** — adapted to a constraint most of that literature never faces: an INT8-only NPU (HTP v68) that rejects 16-bit activations *and* internal float.

### 2A.1 The problem: transformer activations have outliers that defeat per-tensor INT8

Transformers develop a few *outlier* / *massive* activation channels whose magnitude is orders larger than the rest, concentrated in the residual stream; the effect grows with scale and is well documented — **LLM.int8()** [4], **SmoothQuant** [5], **Massive Activations** [6]. Per-tensor symmetric INT8 uses a single scale

$$s=\max_i|x_i|/(2^{b-1}-1),$$

so one outlier inflates $s$ and the remaining channels — which carry the *direction* that L2-normalized retrieval depends on — collapse onto a few levels. The measured residual concentration of $252\times$ and the plain-W8A8 collapse to cosine $0.14$ (§1, §6.1) are the vision-encoder instance of exactly this phenomenon.

The literature offers two cures: **(i)** keep/migrate the outliers in higher precision — mixed-precision (LLM.int8() [4]), activation→weight scale migration (SmoothQuant [5]), activation-aware weight quant (AWQ [7]); or **(ii)** make the representation *incoherent* by an orthogonal transform so no coordinate is special. **HTP v68's broad ban on A16 eliminates family (i)** (16-bit / mixed precision fail to link, §1), which is precisely why we are pushed onto family (ii).

### 2A.2 Rotation as computational invariance folded into weights

- **SliceGPT** [8] established *computational invariance*: inserting $QQ^\top$ ($Q$ orthogonal) on the residual stream and folding $Q$ into adjacent weights leaves the network's function unchanged. Our offline weight folding (§6.4–§6.5) is exactly this.
- **QuaRot** [9] uses randomized Hadamard rotations to remove outliers before low-bit inference, folding some rotations offline and others online via fast Hadamard. Our residual rotation is the **offline, fully-fused** variant (no runtime op — a hard requirement on v68).
- **QuIP / QuIP#** [10, 11] formalize *incoherence processing*: random-orthogonal or Hadamard transforms make weights/Hessians incoherent, with provable error bounds. This supplies the mathematical "why" in §2A.3.
- **SpinQuant** [12] *learns* the rotation instead of using a fixed random/Hadamard one. We **adopt this** for v8 and retain it in v9 (§6A): learning $Q$ against the encoder's own activation statistics is what broke the random-rotation ceiling and met the 50 target.

### 2A.3 Mathematical motivation — why a rotation lowers per-tensor error

Uniform quantization error per coordinate has variance $\approx s^2/12$ with $s \propto \max_i|x_i|$. So **per-tensor error scales with the dynamic range $\max_i|x_i|$, not the energy $\lVert x\rVert_2$.** Define the *incoherence*

$$ \mu(x) \;=\; \frac{\sqrt{d}\,\max_i |x_i|}{\lVert x\rVert_2} \;\in\; [\,1,\ \sqrt{d}\,]. $$

An outlier-dominated vector has $\mu\approx\sqrt{d}$ (worst case, $x=\lVert x\rVert e_k$); a perfectly spread vector has $\mu\approx 1$. Quantization error for fixed energy is monotone in $\mu$, so the goal is to **minimize $\mu$** — the QuIP incoherence objective [10]. An orthogonal $Q$ preserves energy ($\lVert Qx\rVert_2 = \lVert x\rVert_2$) but changes $\mu$: for a random orthogonal $Q$ the coordinates of $Qx$ behave like $\mathcal N(0,\lVert x\rVert_2^2/d)$, so

$$ \max_i |(Qx)_i| \;\approx\; \frac{\lVert x\rVert_2}{\sqrt{d}}\sqrt{2\ln d}
\quad\Longrightarrow\quad \mu(Qx)\approx \sqrt{2\ln d}, $$

versus $\mu(x)$ up to $\sqrt{d}$ for an outlier vector. For $d=768$ that is $\sqrt{2\ln d}\approx 3.6$ against $\sqrt d\approx 27.7$ — a dynamic-range (hence scale, hence error) reduction of up to $\sim 8\times$ from a *random* rotation alone; combined with the specific outlier structure of this encoder it produces the observed $252\times \to 5.3\times$ concentration drop (§6.6).

A **Hadamard** matrix (entries $\pm 1/\sqrt d$) is the deterministic optimum ($\mu=1$ on its worst input). A **learned** rotation (§6A) goes one step further than both random and Hadamard: rather than minimizing a *worst-case* bound, it minimizes the *actual* dynamic range averaged over this network's calibration activations — which is why v8 beats every random-rotation round.

### 2A.4 Our adaptation: the mean-preserving constraint (the non-obvious twist)

QuaRot/SliceGPT target LLaMA-style models whose norm is **RMSNorm** (no mean subtraction), so *any* orthogonal $Q$ commutes through the norm for free. mSigLIP's vision encoder uses **LayerNorm** (with mean subtraction). Converting LN→RMSNorm to admit an arbitrary $Q$ — the textbook move — *backfired on HTP*: RMSNorm decomposes to $\mathrm{Pow}(x^2)/\mathrm{ReduceMean}/\mathrm{Div}$ and re-exposes the normalization internals to the per-tensor quantizer (collapse to cosine $0.16$, §10). Our resolution is to **keep fused LayerNorm and instead constrain the rotation** to fix the mean axis, $Q\mathbf 1=\mathbf 1$, so $\mathrm{LN}(Qx)=Q\,\mathrm{LN}(x)$ holds (§6.3). This confines the rotation to the $(d{-}1)$-dim subspace orthogonal to $\mathbf 1$ — the same subspace where the outliers live — and is, as far as we know, the practical adaptation that makes QuaRot-style equalization (and SpinQuant-style learned rotation) compatible with a **fused-LayerNorm vision encoder on an INT8-only NPU**.

### 2A.5 QAT: training the weights to tolerate the *deploy-faithful* quantizer

Rotation removes concentration but leaves ordinary per-tensor INT8 error accumulating over 12 blocks (§10). Quantization-aware training closes the gap: forward through fake-quant, backward through the **straight-through estimator** [13]; per-tensor fake-quant with **moving-average (EMA) min-max observers** is the standard integer-inference recipe [14, 15]; learnable step sizes (LSQ [16]) are a refinement we did not need. We distill rather than retrain on the task loss: a **teacher–student** [17] setup with the FP32 (rotated) model as frozen teacher and the fake-quant model as student, matching embeddings (cosine + MSE). This is vision-only and label-free, and it directly optimizes the deployment contract — *INT8 embedding ≈ FP32 embedding* — which is what retrieval R@1 depends on. The decisive lesson (§7.3) is **observer faithfulness**: a per-sample/dynamic simulated scale inflates training cosine but does not transfer; matching AI Hub's *calibrate-once, per-tensor* scheme with a per-tensor EMA observer is what crossed the gate.

### 2A.6 Backbone & training context

The backbone is **SigLIP** [2] (sigmoid image–text contrastive pretraining), multilingual variant, fine-tuned with **LoRA** [1] and a **Circle Loss** [3] curriculum (the source checkpoint). Deployment merges LoRA (§3) before any quantization. Related PTQ baselines we did not adopt include **GPTQ** [18] (weight-only PTQ; our bottleneck is activations, not weights).

---

## 3. Stage [1] — Merge LoRA Into the Base Model

The training checkpoint carries PEFT LoRA adapters; Qualcomm tooling has no notion of LoRA. LoRA modifies a dense weight matrix as

$$ W_{\text{merged}} = W_{\text{base}} + \frac{\alpha}{r}\, B A, $$

where $W_{\text{base}}$ is the frozen pretrained weight, $A \in \mathbb{R}^{r\times d}$ and $B \in \mathbb{R}^{d\times r}$ are the low-rank factors, $r$ the rank, $\alpha$ the LoRA scale. Merging produces an ordinary inference model with **no** adapter / base-layer keys. This merged FP32 model is the reference for all subsequent ONNX export and fidelity comparisons.

**Method rule:** rotation/quantization artifacts are *model-specific*. Switching to a different checkpoint requires regenerating every downstream stage from the merge.

---

## 4. Export Contract — Opset-20 Fused GELU (and Fused LayerNorm)

This requirement is applied at the ONNX export step after the model has been merged, rotated, and QAT-finetuned. It is documented here before the rotation mathematics because it explains why every successful branch insists on fused operators, but it is not the second command in the deployment sequence.

The mSigLIP MLP uses the tanh GELU approximation:

$$ \mathrm{GELU}(x) = 0.5\,x\left[1 + \tanh\!\left(\sqrt{\tfrac{2}{\pi}}\,(x + 0.044715\,x^3)\right)\right]. $$

At low opset, ONNX has no `Gelu` op, so this **decomposes** into primitives including $\mathrm{Pow}(x,3)$. The internal $x^3$ term reaches activation magnitudes around $10^5$, which completely dominates any per-tensor quantization scale. This single artifact explains a whole family of historical failures (W8A16 link failures clustered around GELU tensors; "SmoothQuant is neutral" observations).

**Method fix:** export at **opset 20**, where `Gelu` is a *fused* operator and HTP executes the cubic *inside* the runtime op — it is never exposed as a quantized tensor. The same applies to fused `LayerNormalization`. Every successful branch keeps opset-20 fused GELU + fused LayerNorm; any export that re-exposes `Pow`/`ReduceMean` clusters is not the successful pipeline.

Calibration/smoke inputs must use **exactly** the training preprocessing (RGB → resize 256×256 bicubic → normalize to $[-1,1]$, NCHW float32). Using identical input tensors for board runs and local comparisons isolates quantization/runtime error from any image-decoding drift.

---

## 5. Stage [2] — Mean-Preserving Rotation Equalization

### 5.1 The per-tensor quantization problem

Per-tensor symmetric INT8 uses one scale $s$ for the whole tensor:

$$ x_{\text{int}} = \mathrm{round}(x/s), \qquad \hat{x} = s\cdot x_{\text{int}}, \qquad s = \frac{\max_i |x_i|}{2^{b-1}-1}. $$

If one channel is much larger than the rest, $s$ is set by that outlier and ordinary channels collapse to a handful of levels. For retrieval this is fatal: the metric depends on embedding **direction** after L2 normalization, not raw scale. Measured residual-stream concentration before rotation is $\approx 252\times$ (worst-channel abs-max over median).

### 5.2 Rotation idea: redistribute, don't clip

Let $x \in \mathbb{R}^{d}$ ($d=768$) be a residual vector and $Q$ orthogonal. Represent the residual stream as $x_{\text{rot}} = Qx$. Because $Q$ is orthogonal,

$$ \lVert Qx \rVert_2 = \lVert x \rVert_2 . $$

Norms, inner products, and cosine similarity are preserved exactly, but a spiky channel's energy is spread across many coordinates, so per-tensor INT8 is no longer dominated by one coordinate. $Q$ is **folded offline** into existing weights — no new runtime `MatMul`. This is SliceGPT computational invariance [8] combined with the QuIP/QuaRot incoherence rationale [9, 10]; see §2A.3 for the error bound.

### 5.3 Why $Q$ must preserve the mean

An earlier version converted LayerNorm → RMSNorm so an arbitrary orthogonal $Q$ would commute through normalization. It failed: RMSNorm exports as $\mathrm{Pow}(x,2)$ + $\mathrm{ReduceMean}$ + division, re-exposing normalization internals to the quantizer and collapsing W8A8 again (cosine $0.16$, §10).

The successful version keeps **fused LayerNorm**, so $Q$ must commute with it. For identity-affine LayerNorm,

$$ \mathrm{LN}(x) = \frac{x - \mathrm{mean}(x)\,\mathbf{1}}{\mathrm{std}(x)} . $$

For arbitrary orthogonal $Q$, $\mathrm{mean}(Qx) \neq \mathrm{mean}(x)$ in general, so LN does not commute. The fix is to build $Q$ that fixes the all-ones direction:

$$ Q\mathbf{1} = \mathbf{1}, \qquad Q^{\top}Q = I . $$

Then $\mathrm{mean}(Qx) = \tfrac{1}{d}\mathbf{1}^{\top}Qx = \tfrac{1}{d}\mathbf{1}^{\top}x = \mathrm{mean}(x)$ and $\mathrm{std}(Qx) = \mathrm{std}(x)$, giving

$$ \mathrm{LN}(Qx) = Q\,\mathrm{LN}(x). $$

So we can rotate the residual stream and keep `LayerNormalization` fused. The **random** construction is

$$ Q = U\, \mathrm{blockdiag}(1, R_c)\, U^{\top}, \qquad U[:,0] = \mathbf{1}/\sqrt{d}, $$

where $R_c$ is a random orthogonal matrix on the $(d-1)$-dimensional complement of the mean axis. $Q$ is identity on the mean direction and a rotation on the complement — exactly where channel outliers live. The same parametrization, with $R_c$ **learned** instead of random, gives the v8 rotation (§6A).

### 5.4 Folding affine into the reader

LayerNorm has affine params $\mathrm{LN}_{\text{affine}}(x) = \gamma \odot \mathrm{LN}_{\text{id}}(x) + \beta$. For a linear consumer $y = W(\gamma \odot h + \beta) + b$, fold the affine into the reader:

$$ W' = W\,\mathrm{diag}(\gamma), \qquad b' = b + W\beta, $$

then set LN affine to identity ($\gamma=1, \beta=0$). Same FP32 function, now rotation-compatible. Folded readers: $q/k/v$ projections after `layer_norm1`; `mlp.fc1` after `layer_norm2`; K/V slices of the pooling head after `post_layernorm`.

### 5.5 Folding $Q$ into writers and $Q^{\top}$ into readers

A residual **writer** $y = Wx + b$ must now emit $Qy$:

$$ W' = QW, \qquad b' = Qb. $$

Writers: patch-embedding conv output channels, position-embedding rows, every `out_proj`, every `mlp.fc2`.

A residual **reader** sees $x_{\text{rot}} = Qx$ and must recover the original basis:

$$ W' = W Q^{\top}. $$

Readers: every $q/k/v$ projection, every `mlp.fc1`, and the K/V slices of the pooling-head attention. The learned head query/probe is *not* rotated — rotation is localized in the encoder residual stream and undone at the head K/V boundary.

### 5.6 Acceptance gates (rotation is accepted only if FP32 is invariant)

Because rotation is an orthogonal, mean-preserving change of basis, the FP32 embedding must be **exactly** invariant. The rotation is accepted only if:

- Phase A (affine fold) and Phase B (Q fold) invariance cosine mean/min $\approx 1.0$;
- $Q$ orthogonality error $\lVert Q^\top Q - I\rVert_{\max} \approx 3\times10^{-15}$;
- mean-preservation error $\lVert Q\mathbf 1 - \mathbf 1\rVert_{\max} \approx 10^{-14}$;
- reload cosine $\approx 1.0$ (min $\geq 0.9999$);
- residual concentration drops $252\times \to 5.3\times$ (random rotation).

The concentration drop is the quantization payoff: the signal is still present, just no longer concentrated in one dominant channel.

---

## 6A. Stage [2, learned variant] — Learned Rotation (SpinQuant-style, the v8/v9 method)

A random orthogonal $R_c$ achieves the *expected* incoherence $\mu \approx \sqrt{2\ln d}$ (§2A.3), but it is blind to **this** encoder's actual activation distribution. SpinQuant [12] shows that *learning* the rotation against calibration statistics beats random/Hadamard. v8 adapts this to the mean-preserving constraint, and it is what broke the random-rotation ceiling (49.30 → 50.85). v9 keeps the same objective and constraint manifold, but spends more calibration/search budget and selects the best rotation encountered during optimization.

### 6A.1 Objective: directly minimize the quantity that sets the INT8 scale

At each residual **rotation site** (the outputs of `layer_norm1`, `layer_norm2`, `out_proj`, `fc2`, `post_layernorm`), the per-tensor INT8 scale is $s = \max|a| / 127$. The quantity that *sets the error budget* is therefore the post-rotation dynamic range $\max|aQ^\top|$. We minimize, over the calibration set, the sum of squared max-abs across all rotation sites:

$$ \min_{Q}\ \sum_{\text{sites}} \mathbb{E}_{a\sim\text{calib}}\Big[\big(\max_{ij} |(a\,Q^{\top})_{ij}|\big)^2\Big]
\qquad \text{s.t.}\quad QQ^\top = I,\ \ Q\mathbf 1 = \mathbf 1. $$

### 6A.2 Why max-abs², not quantization-MSE

The intuitive objective — minimize $\lVert q(aQ^\top) - aQ^\top\rVert^2$ — has **zero gradient** with respect to $Q$. Under the straight-through estimator (§7.2), $q(x)-x$ is detached on the backward pass, so $\partial/\partial Q$ of the quant-MSE vanishes. By contrast, $\big(\max|aQ^\top|\big)^2$ is differentiable in $Q$ (the $\max$ is subdifferentiable, selecting the argmax coordinate; abs is subdifferentiable) **and is exactly the quantity that determines the per-tensor scale**. Minimizing it directly shrinks $s$, which shrinks the clipping + rounding error budget — the lever that quant-MSE cannot reach through STE.

### 6A.3 Cayley parametrization (orthogonal *and* mean-preserving at every step)

To keep $Q$ on the constraint manifold throughout Adam, we parametrize

$$ Q = U \,\mathrm{blockdiag}\big(1,\ \mathrm{Cayley}(S)\big)\, U^{\top},
\qquad U[:,0] = \mathbf 1/\sqrt d, $$

where $S \in \mathbb{R}^{(d-1)\times(d-1)}$ is a **learnable skew-symmetric** matrix ($S = -S^\top$) and

$$ \mathrm{Cayley}(S) = (I - S)(I + S)^{-1} $$

is orthogonal for *any* skew $S$ (the Cayley transform maps skew-symmetric matrices to special-orthogonal ones). Pinning the first column of $U$ to $\mathbf 1/\sqrt d$ makes $\mathbf 1$ a fixed eigenvector with eigenvalue $1$, so $Q\mathbf 1 = \mathbf 1$ holds **exactly**, and the block structure guarantees $QQ^\top = I$ **exactly** — at every optimization step, not just at convergence. This is the same constraint subspace as the random construction (§5.3); only $R_c = \mathrm{Cayley}(S)$ is now learned.

### 6A.4 Method (offline, calibration-only)

1. **Phase A:** fold LayerNorm affine into the reader and set LN to identity (§5.4), so the rotation sites are clean.
2. **Collect activations** at the rotation sites on a calibration subset of images.
3. **Optimize $S$** with Adam against the objective in §6A.1. Numerical detail: optimize in float32 (float64 matmuls are throttled to $\sim\!1/32$–$1/64$ of float32 on consumer GPUs), track the best objective value over the run, then fold the *best* $Q$ in float64 for precision. This best-$Q$ checkpointing matters because Adam at fixed learning rate can oscillate late in the run; folding the last step can accidentally choose a worse high-dynamic-range point.
4. **Phase B:** fold the learned $Q$ into writers and $Q^\top$ into readers (§5.5). The output is a drop-in replacement for the randomly-rotated model — identical shapes, identical FP32 function — that flows through the same export/QAT/quantize/eval stages.

### 6A.5 Result and ablation

Measured on the vision encoder (256 calib images, 32 tokens/image, 3000 steps):

- **Objective:** $46281 \to 861$ (−98.1%).
- **Dynamic range** $\max|a|$: $123.8 \to 14.56$ (−88.2%) — far tighter than what random rotation achieves on the same sites.
- **FP32 invariance gate:** output cosine min $0.99999988$, orthogonality error $3.1\times10^{-15}$, mean-preservation error $4.0\times10^{-15}$ — PASS.

Under the **identical** QAT recipe v6 (so the only difference from v6 is $Q$):

| Metric | v6 (random) | **v8 (learned)** | Δ |
|---|---:|---:|---:|
| T2I R@1 | 49.30 | **50.85** | **+1.55** |
| I2T R@1 | 53.85 | 52.90 | −0.95 |
| QDQ cosine mean | 0.9491 | **0.9606** | +0.0115 |
| QDQ cosine min | 0.9266 | **0.9447** | +0.0181 |

This is a clean ablation: coverage, learning rate, epochs, and base FP32 model are all held fixed, so the **+1.55** T2I gain is attributable purely to learning $Q$. The QDQ fidelity rising on both mean and min confirms the max-abs² objective genuinely tightened the INT8 scale, as the theory predicts. The small I2T trade-off is acceptable because the primary deployment metric is **T2I R@1**. v8 is the first deployable learned-rotation candidate; v9 keeps the same method and becomes the final board recipe.

### 6A.6 v9 refinement: same theory, larger search budget

v9 should be interpreted as a **recipe refinement** of §6A, not as a new theoretical mechanism. The invariants are unchanged:

- the rotation is still orthogonal and mean-preserving, so the FP32 function is preserved before QAT;
- the objective is still calibration max-abs², i.e. the quantity that controls the per-tensor INT8 activation scale;
- QAT still uses the deploy-faithful per-tensor EMA observer and the same fake-quant coverage (`head`, linears, attention sites).

The practical differences are:

| Item | v8 | v9 |
|---|---:|---:|
| Rotation calibration images | 256 | 512 |
| Tokens per image | 32 | 32 |
| Rotation optimization steps | 3000 | 8000 |
| Rotation checkpoint | last/final $Q$ | best-objective $Q$ |
| Vision QAT epochs | 15 | 25 |
| Vision board T2I R@1 | 50.20 | **50.35** |
| Vision board I2T R@1 | 54.50 | **54.55** |

The mathematical lesson is modest but useful for deployment: when the learned-rotation objective is non-convex and the optimizer can oscillate, the *selection rule* for $Q$ is part of the deployment recipe. The final thesis result uses v9 because it increases the hardware-verified T2I margin while preserving the same theoretical argument as v8.

---

## 7. Stage [3] — Quantization-Aware Finetune (the gate-passing step)

Rotation alone reaches QDQ cosine $0.8975$ but only **R@1 = 45.42** (gate FAIL). The residual error is no longer one outlier; it is ordinary per-tensor INT8 error **accumulated across 12 blocks** (§10). QAT trains the FP32 weights to tolerate that INT8 noise.

### 7.1 Teacher–student distillation

QAT does not export a custom QDQ graph. It finetunes the FP32 encoder under injected fake-quant noise, then saves a normal FP32 export that the existing ONNX + AI Hub quantizer processes. Setup:

- **Teacher:** the rotated FP32 model, frozen.
- **Student:** a copy of the rotated model with fake-quant forward hooks on selected activations.
- **Trainable:** only the selected encoder layers + visual projection; everything else frozen.

Per step the student is run twice — clean (hooks off) and fake-quant — and distilled toward the teacher embedding:

$$ \mathcal{L} = \underbrace{\big(1 - \cos(z_s^{q}, z_t)\big) + \lambda\,\lVert z_s^{q} - z_t\rVert^2}_{\text{fake-quant path}} + \underbrace{w_c\big(1 - \cos(z_s^{c}, z_t)\big) + w_m\lVert z_s^{c} - z_t\rVert^2}_{\text{clean consistency}}, $$

where $z_t$ is the teacher embedding, $z_s^{q}$/$z_s^{c}$ are the fake-quant/clean student embeddings, and $\lambda = w_m = 0.05$, $w_c = 1.0$. The clean term keeps the student from drifting away from the teacher when quant noise is off.

### 7.2 Straight-through fake-quant

Symmetric INT8 fake-quant with a straight-through estimator (STE):

$$ q(x) = s\cdot\mathrm{clamp}\!\Big(\mathrm{round}(x/s),\,-q_{\max},\,q_{\max}\Big), \quad q_{\max} = 2^{b-1}-1 = 127, \quad s = \frac{\max|x|}{q_{\max}}. $$

The forward pass is quantized; the backward pass is identity ($\partial q/\partial x \equiv 1$), so gradients flow to the FP32 weights through the non-differentiable `round`. This is the straight-through estimator [13]. (Note: this is exactly the detachment that makes a quant-MSE rotation objective have zero gradient — see §6A.2.)

### 7.3 Why per-tensor, and why an EMA observer (the two key fixes)

The early R@1 trajectory $45.42 \to 46.92 \to 47.80 \to 48.20$ came from making the *simulated* quantizer match the *real* AI Hub W8A8:

- **v1 → v2: per-sample → per-tensor.** A per-sample scale (one per image) is too easy: simulated cosine looks great ($0.975$) but does not transfer to AI Hub's per-tensor scheme (real $0.92$). Per-tensor fake-quant uses one scale per activation tensor, matching deployment.
- **v2 → v3: dynamic → EMA observer.** A per-batch dynamic max recomputes the scale every forward; AI Hub instead **calibrates once** and freezes the scale. The EMA observer is the standard moving-average min-max observer [14, 15]:

$$ m_t = \mu\, m_{t-1} + (1-\mu)\,\max|x_t|, \quad \mu = 0.99, $$

producing a fixed-ish per-tensor scale. This closed the sim↔real gap: simulated val cosine stayed $\approx 0.98$, but real QDQ cosine rose $0.9281 \to 0.9353$ and R@1 crossed the gate.

### 7.4 Fake-quant coverage (v3 → v6) and the random-rotation ceiling

The third lever — after per-tensor (v2) and EMA (v3) — is *which activations* carry the fake-quant operator during QAT. AI Hub quantizes **every** activation tensor, so any tensor the student never trained against contributes uncorrected INT8 error at deploy. Coverage was widened in steps:

| Round | Fake-quant sites (per block) | QDQ cosine mean / min | T2I R@1 |
|---|---|---:|---:|
| v3 | GELU output + residual output | 0.9353 / 0.919 | 48.20 |
| v4 | + pooling head (post-LN, head attn, head MLP) | 0.9364 / 0.9091 | 48.50 |
| v5 | + every linear output (q/k/v/out_proj, fc1, fc2, head) | 0.9437 / **0.9311** | 49.25 |
| v6 | + attention matmuls ($Q K^\top$, $\mathrm{softmax}\cdot V$) | 0.9491 / 0.9266 | **49.30** |

The mathematics is unchanged across v3–v6 — the same per-tensor straight-through fake-quant (§7.2) and EMA observer (§7.3); **only the *set* of quantized tensors grows.** The decisive early evidence is the **minimum** cosine: widening coverage from the GELU/residual subset to all linear outputs lifted the worst-sample cosine $0.9091 \to 0.9311$, confirming the $\approx 0.936$ plateau was a *coverage gap*, not a fundamental W8A8 limit.

But v6 then **saturates**: adding the attention matmuls lifts R@1 only $49.25 \to 49.30$. Once every activation tensor is covered, QAT alone gives diminishing returns. **This is the random-rotation ceiling** — and the reason the next gain had to come from the rotation itself (learned rotation, §6A), not more coverage.

### 7.5 The v7 cautionary regression (why the recipe was *not* changed for v8)

v7 kept v6 coverage but changed the optimizer schedule: cosine LR + learning rate doubled to $2\times10^{-5}$ over 20 epochs. Result: QDQ min **dropped** $0.9266 \to 0.9083$ and T2I R@1 **regressed** $49.30 \to 48.38$. The likely cause is LR overshoot around the quantization minimum, which the cosine floor could not recover from. The lesson carried into v8: **hold the recipe fixed at v6** (constant lr $1\times10^{-5}$, 15 epochs) so the v8 delta isolates the *rotation* (learned vs random), uncontaminated by a schedule already shown to hurt.

### 7.6 Result (best to date: v8)

| Task | Model | R@1 | R@5 | R@10 | mAP | mINP |
|---|---|---:|---:|---:|---:|---:|
| T2I | FP32 sanity reproduction | 52.40 | 79.38 | 87.80 | 57.38 | 50.67 |
| T2I | **QAT v8 INT8 (learned rot.)** | **50.85** | 77.48 | 86.98 | 55.79 | 49.24 |
| I2T | FP32 sanity reproduction | 55.30 | 81.45 | 89.70 | 51.38 | 34.50 |
| I2T | **QAT v8 INT8 (learned rot.)** | **52.90** | 80.45 | 88.35 | 49.48 | 32.88 |

QDQ cosine $0.9606$ / min $0.9447$. Drop vs the paper baseline $52.28$ is **1.43** on T2I — the smallest of any round, and the first vision-only branch to meet the $\geq 50$ deploy target on the QDQ proxy.

---

## 8. Stages [4–6] — Export, Quantize/Link, Board Run & Evaluation (method)

### 8.1 Export

Because every rotated/QAT model is output-invariant relative to the *rotated* model, the static ONNX-vs-PyTorch control must be $\approx 1.0$; this also confirms the graph is well-formed and op-fused (fused GELU + LayerNorm, no `Pow`/`ReduceMean` clusters). The export is a directory (graph + external-data file) because SigLIP weights are large.

### 8.2 Quantize, compile, link

The deployable graph is produced by: rewriting input shape to static $1\times3\times256\times256$; resolving a fixed calibration dataset; submitting a W8A8 quantize job; then compile (DLC) + link to a context binary **with quantized I/O**. The retrieval gate, not cosine, is decisive, and the QDQ output is always compared against the **original merged FP32 model**, because the deployment contract is "match the original FP32 model," and the rotated/QAT model is behaviorally equivalent on the validation objective.

**Why this links on v68:** no float I/O, no internal float surgery, no A16 (avoids the v73 requirement), no decomposed GELU cubic, no decomposed RMSNorm internals — every main activation is W8A8, which v68 supports broadly.

### 8.3 Board run and fidelity

On board, graph I/O is unsigned-fixed-point-8 and the runtime dequantizes outputs to float. **Board fidelity** (board vs PyTorch cosine) for the verified v8 binary is $0.9585$ (mean) / $0.9399$ (min), closely matching its QDQ ONNX $0.9606 / 0.9447$. Full v8 vision-isolation retrieval on RB3 reaches **T2I R@1 $50.20$** and **I2T R@1 $54.50$**, a small $-0.65$ T2I delta from the $50.85$ QDQ proxy while staying above the deploy target. The final board vision tower is v9: **T2I R@1 $50.35$**, **I2T R@1 $54.55$**, and runtime $\approx 32.54$ ms / $24.29$ FPS. HTP runtime is faithful to the quantized graph; the remaining error is *quantization*, not hardware drift.

### 8.4 Why the retrieval number is trustworthy

- **Vision-isolation:** image embedding = QDQ ONNX (rotated W8A8); text embedding = FP32 PyTorch. Text is kept FP32 for both baseline and quantized so the measurement isolates the vision quantization.
- **Both-INT8 C1:** image embedding = vision QDQ ONNX and text embedding = finite/f32-mask text QDQ ONNX; this is the end-to-end off-board deploy proxy.
- Retrieval mirrors the training evaluator exactly: **raw (un-normalized) pooler features**, dot-product similarity, the same rank metric. (A generic normalize-then-rank evaluator does not reproduce the baseline.)
- **Sanity:** the FP32 pipeline reproduces T2I R@1 $52.40 \approx 52.28$; reported drops use the paper baseline $52.28$.
- **Final deploy result:** direct RB3 board both-INT8 reaches T2I R@1 $50.35$, I2T R@1 $54.20$, passing the $\geq 50$ gate. The off-board QDQ proxy remains a reference at $50.25 / 52.95$.

---

## 9. Why Earlier Candidates Failed

| Attempt | Result | Lesson |
|---|---|---|
| FP32/FP16 ONNX directly on HTP | link fail | v68 context binary requires integer I/O. |
| Deprecated CLI INT8 path | preserves FP I/O | Use API quantize + compile/link with quantized I/O. |
| Plain W8A8 (no rotation) | cosine 0.13–0.17 | Runtime works; per-tensor INT8 destroys embedding direction. |
| More calibration samples | still fails | Not a calibration-coverage problem. |
| Lite-MP / min-max / W8A16 on old graph | fails | Global knobs cannot fix exposed GELU cubic / outliers. |
| `_float` QDQ surgery | local pass, link fail | v68 rejects internal float tensors. |
| ORT W8A16 QDQ | local ≈ 0.999, link fail | Linker rejects internal float / dequantized GELU patterns. |
| Clipping INT8 activations | 0.12–0.40, fails | Outlier channels carry real signal; clipping loses information. |
| Opset-20 + W8A16 | QDQ 0.9997, link fail | A16 attention/LayerNorm need HTP v73+; RB3 is v68. |
| Rotation with RMSNorm | cosine 0.16 | RMSNorm exposes Pow(x²)/ReduceMean to the quantizer. |
| Phase-C R2 (head-dim Hadamard) | R@1 45.25, no gain | Targets the value path; residual error is in MLP activations. |
| Mean-preserving rotation + fused LN + W8A8 | board pass, R@1 45.42 | All-INT8, v68-compatible, but per-tensor error accumulates over 12 blocks. |
| + QAT (per-tensor + EMA observer) | R@1 48.20, below 50 target | Train the weights to tolerate the deploy-faithful INT8 noise. |
| + wider fake-quant coverage (v4–v6) | R@1 49.30, below 50 target | Covering all tensors helps, then hits the random-rotation ceiling. |
| v7 cosine LR + lr 2e-5 | R@1 48.38, regress | LR overshoot near the quantization minimum; hold the v6 recipe. |
| **+ learned rotation (v8)** | **R@1 50.85, meets 50 target** | Optimize $Q$ against the activation max-abs²; beats random rotation. |

---

## 10. Acceptance Gates

| Gate | Threshold | Status |
|---|---:|---|
| Merge LoRA → clean non-adapter weights | no adapter keys | PASS |
| Rotation FP32 invariance | cosine min ≥ 0.9999 | PASS (learned & random) |
| ONNX static control | cosine mean ≥ 0.999 | PASS |
| ONNX op sanity | Pow=0, fused Gelu, fused LayerNorm | PASS |
| QDQ ONNX vs PyTorch (v8) | proxy ≥ 0.95 / 0.90 | **0.9606 / 0.9447** |
| Text finite-mask patch | replace only attention-mask `-FLT_MAX`; 12 Softmax paths found | PASS |
| Text attention QDQ scale | max `< 10.0` on 12 `scores+mask` QDQ pairs | **PASS — 0.3523 max** |
| Text QDQ ONNX vs PyTorch | proxy ≥ 0.95 / 0.90 | **0.9949 / 0.9912** |
| Text-isolation retrieval | T2I R@1 ≥ 50.0 | **PASS — 51.65** |
| Both-INT8 off-board retrieval | T2I R@1 ≥ 50.0 | **PASS — 50.25** |
| QNN board vs PyTorch | mean ≥ 0.90 | **PASS — 0.9585 / 0.9399 (v8 vision)** |
| Board execution | finite outputs, HTP profile | **PASS — v9 vision, 32.54 ms/image, 24.29 FPS; split text, 7.87 ms/query, 74.75 IPS** |
| Vision board retrieval | T2I R@1 ≥ 50.0 | **PASS — 50.35** |
| Text QNN link | context binary created without internal float mask tensors | **PASS — finite/f32/link-safe binary links** |
| **Full retrieval T2I R@1 (deploy target)** | **≥ 50.0** | **PASS — 50.35 (direct board both-INT8)** |

Cosine is a conservative proxy; retrieval R@1 is the decisive metric. The deploy target is **T2I R@1 ≥ 50**; any candidate below 50 (v1–v7) is a FAIL on this target.

---

## 11. Method Pitfalls to Avoid

- Do not skip the LoRA merge — the checkpoint is not directly a deployment model.
- Do not reuse rotation/QDQ artifacts across different checkpoints (they are model-specific).
- Do not export at low opset for the final path — it exposes Pow(x³) from tanh-GELU.
- Do not convert LayerNorm → RMSNorm for the v68 W8A8 path — it exposes normalization internals to the quantizer.
- Do not leave internal float (`_float` surgery) — it passes local diagnostics but fails the HTP link.
- Do not use W8A16 as the final v68 plan — it passes fidelity but fails the link (needs v73+).
- Do not use per-sample / per-batch fake-quant for QAT — it inflates simulated cosine but does not transfer; use per-tensor + EMA observer.
- Do not optimize a learned rotation with a quant-MSE objective — STE detaches its gradient; use max-abs² (§6A.2).
- Do not change the QAT schedule when isolating a rotation change — hold the recipe fixed (the v7 lesson).
- Do not let text attention export a `-FLT_MAX` mask into a quantized `scores+mask` tensor — patch it to a finite negative mask and gate the QDQ scale (§12A.4).
- Do not leave `attention_mask` as int64 in the final QNN-linkable text export if it materializes an internal `Cast(FLOAT)` pre-quant tensor — export the binary mask as float32 0/1 (§12A.5).
- Do not trust PSNR as a retrieval proxy — always compute embedding cosine and, decisively, retrieval R@1.

---

## 12. What Remains

1. ✅ **Legacy Board-verified W8A8 (v4):** established board-to-QDQ fidelity tracking.
2. ✅ **Vision-only v8 QDQ passes:** learned rotation, T2I R@1 $50.85$ (`-1.43` vs paper baseline $52.28$).
3. ✅ **Board-verified v8:** learned-rotation v8 binary executes on HTP v68. Board fidelity $0.9585$ tracks QDQ $0.9606$, full board vision retrieval reaches T2I R@1 $50.20$ / I2T R@1 $54.50$, and runtime is $\approx 33.05$ ms / $22.77$ FPS.
4. ✅ **Text encoder finite/f32/link-safe local path passes:** learned rotation + text QAT + finite attention mask and float32 binary mask I/O give QDQ cosine $0.9949 / 0.9912$ and text-isolation T2I R@1 $51.65$; the link-safe rewrite removes `/text_model/Cast_output_0` and keeps static cosine $0.99999999 / 0.99999976$.
5. ✅ **Text finite/f32/link-safe context binary links:** the algebraic mask rewrite removes the internal float mask island that previously failed HTP v68 link, so the text branch now has a linkable W8A8 QNN context binary.
6. ✅ **End-to-end both-INT8 C1 passes off-board:** vision QDQ + text QDQ gives T2I R@1 $50.25$, I2T R@1 $52.95$, a `-2.03` T2I drop vs paper baseline $52.28$.
7. ✅ **Full-graph text board fails — diagnosed:** the finite/f32/link-safe text binary links but its board output ignores `input_ids` (HTP v68 breaks the dynamic 250k-row embedding `Gather`; §12A.8). Static ONNX and QDQ proxy stay faithful, so it is a graph-lowering limit, not a QAT/calibration failure.
8. ✅ **Split-encoder text is board-verified:** moving the embedding lookup to the host/RB3 CPU and running the transformer on HTP via `inputs_embeds` fixes it — board fidelity $0.9951 / 0.9926$, text-isolation board T2I R@1 $51.30$.
9. ✅ **Board-verified vision v9:** tighter follow-up vision binary reaches T2I R@1 $50.35$ / I2T R@1 $54.55$, improving the deploy-critical T2I metric over v8 by `+0.15`; runtime is $\approx 32.54$ ms / $24.29$ FPS.
10. ✅ **End-to-end both-INT8 board:** vision v9 board + split-text board gives T2I R@1 $50.35$, I2T R@1 $54.20$ — `+0.10` T2I vs the off-board QDQ proxy and `-1.93` vs the paper baseline. This is the final deploy number for the thesis.

---

## 12A. Text Encoder — Deltas vs Vision

The text encoder uses the **same v8 method** as the vision branch: opset-20 fusion (§4) → learned mean-preserving rotation (§5–§6A) → per-tensor STE + EMA QAT distillation (§7) → W8A8 quantize/compile/link (§8). All the residual-stream mathematics carries over unchanged — the rotation objective $\min_Q \sum \mathbb{E}[(\max|aQ^\top|)^2]$, the Cayley parametrization with $Q\mathbf 1 = \mathbf 1$, the straight-through fake-quant, and the teacher–student distillation are identical.

Text adds two requirements that vision does not have. First, **the attention mask must be finite before AI Hub inserts QDQ around the masked logits**. Without this, the text path can collapse even after successful learned rotation and QAT, because the quantizer may see `scores + mask` where padded positions contain `-FLT_MAX`. Second, **the final QNN-linkable export must express the binary mask without redundant `Cast(FLOAT)` / `Cast(BOOL)` / `Where` islands**. The link-safe form feeds `attention_mask` as float32 0/1 and rewrites the additive mask algebraically as `(1-mask)*(-32)`.

### 12A.1 Token indices, binary masks, and the I/O-quantization difference

The text inputs are `input_ids` and `attention_mask`. A token ID is an **index** into the embedding table, not a continuous signal; quantizing an index to INT8 is meaningless (it would destroy the index resolution). Therefore:

- **`input_ids` remain integer indices.** They may be truncated to a QNN-supported integer I/O type during compile, but they are not treated as quantized continuous activations.
- **`attention_mask` is semantically binary, not an index.** For local training and QAT it can be stored as integer 0/1. For the final QNN export it is stored as float32 0/1 and its mask subgraph is simplified, because the downstream formula immediately consumes it as a floating-point additive mask.
- **Calibration data** is *tokenized captions* (parallel `input_ids` plus `attention_mask` 0/1 arrays), not preprocessed image tensors. The final f32-mask AI Hub dataset uses integer `input_ids` and float32 `attention_mask`.
- W8A8 still applies to weights and activations. The embedding *weights* are W8-quantized; the lookup indices stay integer. The graph I/O handling exists to satisfy HTP's context-binary linker, not to reinterpret token IDs as continuous INT8 values.

### 12A.2 Text pipeline overview (conceptual)

The text branch has two related flows. The first is the **full-graph QDQ/proxy flow**, which is useful for off-board accuracy checks but was rejected as the board runtime path because HTP v68 did not correctly consume dynamic `input_ids` through the large token-embedding `Gather`. The second is the **split-encoder deployment flow**, which keeps the token lookup as a deterministic CPU table read and runs the rotated/QAT transformer on HTP.

```text
            ┌─────────────────────────────────────────────┐
            │  FP32 merged text encoder                  │
            │  token ids integer, mask binary 0/1        │
            └─────────────────────┬───────────────────────┘
                                  │
            ┌─────────────────────▼───────────────────────┐
   [1]      │  LEARNED ROTATION                           │
            │  same mean-preserving Q as vision v8        │   FP32-invariant
            │  fold Q into embeddings / writers / readers │
            └─────────────────────┬───────────────────────┘
                                  │
            ┌─────────────────────▼───────────────────────┐
   [2]      │  TEXT QAT DISTILLATION                      │
            │  per-tensor STE + EMA observer              │   trains weights
            │  fake-quant linears, head, attention sites  │   to tolerate W8A8
            └─────────────────────┬───────────────────────┘
                                  │
            ┌─────────────────────▼───────────────────────┐
   [3]      │  EXPORT OPS 20                              │
            │  fused Gelu + LayerNormalization            │   no Pow cubic
            │  full graph OR split transformer export     │
            └─────────────────────┬───────────────────────┘
                                  │
            ┌─────────────────────▼───────────────────────┐
   [4]      │  FINITE + LINK-SAFE ATTENTION MASK          │
            │  -FLT_MAX  →  -32.0 on mask constant path   │   Softmax semantics
            │  Where/Cast mask  →  (1-mask)*(-32)         │   no float island
            └─────────────────────┬───────────────────────┘
                                  │
          ┌───────────────────────┴────────────────────────┐
          │                                                │
          ▼                                                ▼
┌───────────────────────────────┐              ┌───────────────────────────────┐
│ FULL-GRAPH QDQ / PROXY        │              │ SPLIT-ENCODER BOARD PATH      │
│ input_ids + attention_mask    │              │ inputs_embeds + attention_mask│
│ stay in one ONNX graph        │              │ transformer/head only on HTP   │
└───────────────┬───────────────┘              └───────────────┬───────────────┘
                │                                              │
┌───────────────▼───────────────┐              ┌───────────────▼───────────────┐
│ AI HUB W8A8 QDQ               │              │ AI HUB W8A8 QUANTIZE/COMPILE  │
│ QDQ proxy retrieval           │              │ LINK split text context binary│
│ text-only proxy T2I = 51.65   │              └───────────────┬───────────────┘
└───────────────┬───────────────┘                              │
                │                              ┌───────────────▼───────────────┐
┌───────────────▼───────────────┐              │ RB3 CPU token lookup          │
│ FULL-GRAPH CONTEXT REJECTED   │              │ input_ids → INT8 embedding    │
│ links, but board output can   │              │ table → inputs_embeds         │
│ ignore token-id changes       │              └───────────────┬───────────────┘
└───────────────────────────────┘                              │
                                               ┌───────────────▼───────────────┐
                                               │ RB3 HTP text encoder          │
                                               │ board text-only T2I = 51.30   │
                                               └───────────────┬───────────────┘
                                                               │
                                               ┌───────────────▼───────────────┐
                                               │ FINAL BOARD BOTH-INT8         │
                                               │ text board + vision v9 board  │
                                               │ T2I R@1 = 50.35               │
                                               └───────────────────────────────┘
```

The key differences from the vision pipeline are the mask and the token lookup. Vision has no padding mask, so its attention logits never contain a sentinel value. Text does, and the sentinel must be made compatible with per-tensor INT8. In addition, token IDs are indices into a large embedding table. The final board path therefore splits the graph: the CPU performs an exact table lookup from the exported INT8 token table, and HTP runs the quantized transformer/head on `inputs_embeds`. This keeps the mathematical compression recipe intact while avoiding the HTP-v68 dynamic-`Gather` failure observed in the full-graph context.

### 12A.3 Rotation writer/reader mapping

`SiglipTextTransformer` is `token_embedding + position_embedding` → 12 encoder layers → `final_layer_norm` → **last-token pooler** → Linear head. The mean-preserving rotation folds into the text-specific modules:

- **Writers** (emit $Qy$): `token_embedding` and `position_embedding` rows, every `out_proj`, every `fc2`.
- **Readers** (apply $Q^\top$): every $q/k/v$ projection, every `fc1`, and the Linear head after `final_layer_norm`.

This mirrors §5.4–§5.5 exactly; the only change is that the residual *writers* at the input are the two embedding tables rather than a patch-embedding conv, and the *reader* boundary at the output is the last-token pooler + head rather than an attention-pooling head.

### 12A.4 The attention-mask quantization hazard and finite-mask fix

Text self-attention computes

$$ A = \mathrm{softmax}(S + M), \qquad S = \frac{QK^\top}{\sqrt{d_h}}, $$

where $M_{ij}=0$ for valid positions and $M_{ij}=-F$ for padded positions. The original ONNX export uses $F\approx3.402823\times10^{38}$ (`FLT_MAX`). In pure FP32 this is harmless: $\exp(-F)=0$, so padded positions receive zero attention probability.

It becomes harmful only when a per-tensor INT8 quantizer is inserted on the **sum** $S+M$ before Softmax. The real logits $S$ live at ordinary attention scale, but the tensor also contains values near $-3.4\times10^{38}$. The quantizer chooses one scale for the whole tensor, so the mask sentinel dominates the dynamic range and the real logits collapse to almost no resolution. In the bad text QDQ graph, every layer's `self_attn/Add_output_0` QDQ scale was on the order of `1e32`, and the text embedding cosine collapsed.

The fix is to replace the sentinel with a finite negative constant:

$$ M_{ij} \in \{0,\ -32\}. $$

This preserves the attention semantics because

$$ \exp(-32) \approx 1.27\times10^{-14}. $$

Even across a 64-token sequence, the total masked probability leakage is negligible relative to FP32 and INT8 noise. But the quantizer now sees a normal finite range: real attention logits are not crushed by an artificial `FLT_MAX` sentinel. This is a deployment-graph fix, not a training/model-architecture change: it patches only the exported text ONNX constant on the attention-mask path.

The patch is intentionally narrow:

- It replaces only large negative constants on text self-attention mask paths.
- It expects 12 text self-attention Softmax paths.
- It fails if a large negative mask constant remains upstream of those Softmax inputs.
- Static ONNX-vs-PyTorch must still pass before any AI Hub job is trusted.

### 12A.5 Why the QNN-linkable text export rewrites the mask subgraph

The finite-mask patch fixes the *numerical* quantization hazard, but the HTP v68 linker exposed a second, graph-typing hazard. With the original int64 text export, ONNX constructs the additive mask through a path equivalent to

$$
\texttt{attention\_mask}_{\mathrm{int64}}
\rightarrow \mathrm{Expand}
\rightarrow \mathrm{Cast}_{\mathrm{float}}
\rightarrow (1-\cdot)
\rightarrow \mathrm{masked\ fill}.
$$

After AI Hub inserts QDQ, the output of this cast can appear in the QNN graph as an internal pre-quantized floating tensor such as `Cast_output_0_updated_pre_quant`. HTP v68 context binaries reject such internal float tensors, producing a link-time error even though the surrounding model is W8A8. This is not a failure of the learned rotation or QAT; it is a mismatch between the ONNX mask representation and the HTP linker's integer/quantized graph contract.

Changing the exported mask input dtype from int64 to float32 is necessary:

$$
\texttt{attention\_mask}_{\mathrm{float32}}\in\{0.0,1.0\}^{B\times L}.
$$

The mathematical semantics are unchanged because the mask is binary:

$$
0_{\mathrm{int}} \mapsto 0.0_{\mathrm{float}},\qquad
1_{\mathrm{int}} \mapsto 1.0_{\mathrm{float}}.
$$

However, it is not sufficient by itself. HuggingFace's `_expand_mask(...).to(dtype)` can still export a redundant `Cast(FLOAT)` after `Expand`, followed by a boolean cast and `Where`:

$$
\mathrm{Expand}(\texttt{attention\_mask}_{\mathrm{float32}})
\rightarrow \mathrm{Cast}_{\mathrm{float}}
\rightarrow \mathrm{Sub}
\rightarrow \mathrm{Cast}_{\mathrm{bool}}
\rightarrow \mathrm{Where}.
$$

AI Hub then materializes the redundant cast output as `Cast_output_0_updated`, which is still rejected by the HTP linker. The final link-safe graph therefore applies a semantic-preserving rewrite:

$$
\mathrm{Where}(1-\texttt{mask}\ne0,\ -32,\ 0)
\quad\equiv\quad
(1-\texttt{mask})\cdot(-32),
\qquad \texttt{mask}\in\{0,1\}.
$$

The equivalence is exact for binary masks:

$$
\texttt{mask}=1 \Rightarrow M=0,\qquad
\texttt{mask}=0 \Rightarrow M=-32.
$$

The same additive attention is then formed:

$$
M=(1-\texttt{attention\_mask})\cdot(-32),
\qquad A=\mathrm{softmax}(S+M).
$$

Thus the final text deployment path keeps `input_ids` as integer token indices, represents the binary `attention_mask` as float32 for graph compatibility, rewrites the mask construction to a simple multiply, and still quantizes the model weights/activations as W8A8. This is a deployment representation change, not a model-quality change.

### 12A.6 Text gates and current finite/f32/link-safe status

The finite/f32/link-safe branch is accepted only if the mask patch, QDQ scale, embedding fidelity, retrieval, and link gates all pass:

| Gate | Expected | Current text finite/f32/link-safe result |
|---|---:|---:|
| Mask patch | exactly the attention-mask sentinel changed | 1 Constant changed: `-3.402823e38 → -32.0` |
| Final QNN export mask dtype | `attention_mask` float32 0/1 | required for HTP link |
| Link-safe mask rewrite | no `/text_model/Cast_output_0` tensor | PASS in `text_onnx_f32mask_finite_linksafe` |
| Link-safe static ONNX vs PyTorch | cosine mean/min ≥ 0.999 / 0.999 | **0.99999999 / 0.99999976** |
| Text Softmax paths | 12 | 12 |
| Softmax-input QDQ pairs | 12 | 12 |
| Max `scores+mask` QDQ scale | `< 10.0` | **0.3523** |
| QDQ cosine mean / min | ≥ 0.95 / 0.90 | **0.9949 / 0.9912** |
| Text-isolation T2I R@1 | ≥ 50.0 | **51.65** |
| Text-isolation I2T R@1 | monitor | **55.55** |
| AI Hub QNN link | no internal floating-point mask tensor rejected by HTP v68 | **PASS — finite/f32/link-safe context binary links** |

The important diagnostic is the scale gate. A high cosine after local surgery is not enough if the real AI Hub QDQ graph still places an enormous scale on `scores+mask`. The finite-mask QDQ graph has normal scales (`0.1828–0.3523`) across all 12 layers, so the Softmax receives meaningful logits again. Retrieval confirms the proxy: text-only INT8 drops only **0.63** T2I R@1 from the paper baseline (`52.28 → 51.65`), so the text branch now independently clears the deploy target. The f32-mask plus algebraic rewrite is the complementary link gate: it prevents QNN from materializing the binary-mask cast as a rejected internal float tensor.

### 12A.7 Why this is the real memory payoff

The text encoder is $\sim$75\% of the model's parameters because the multilingual token embedding alone is $250{,}000 \times 768 \approx 768$ MB in FP32. Quantizing it to INT8 is the dominant lever for fitting both encoders into the 4 GB RB3 budget — far larger than the vision saving. The end-to-end C1 both-INT8 proxy first cleared the gate at T2I R@1 $50.25$; the final direct board run with vision v9 and split-text confirms the deploy target at T2I R@1 $50.35$.

### 12A.8 The full-graph board failure and the split-encoder method (the deployable text path)

Everything in §12A.1–12A.7 makes the full-graph text **link** on HTP v68. It does not make it **work** on board. The decisive observation: the board context binary produces an output that is a function of `attention_mask` only and is **bit-identical for the real `input_ids` and an all-zero `input_ids`** (cosine $1.0$, max-abs $0.0$ across all smoke samples), while static ONNX ($\approx 1.0$) and the QDQ proxy ($0.9949$) stay faithful. Two independent controls confirm the token IDs are ignored *inside the compiled graph*: (i) zeroing `input_ids` with the mask held fixed leaves the output unchanged; (ii) two different real captions differ on board **iff** their attention masks differ (the diff is perfectly mask-correlated). Because the failure appears only after compile/link — not in ONNX or QDQ — it is a graph-lowering / runtime limit, not a QAT, calibration, or mask-representation failure.

**Root cause.** The text encoder begins with a dynamic gather

$$ \texttt{inputs\_embeds} \;=\; \mathrm{Gather}\big(W_{\text{tok}} \in \mathbb{R}^{250000\times 768},\; \texttt{input\_ids}\big), $$

where `input_ids` is a *runtime* index tensor and $W_{\text{tok}}$ is a $\sim$192 MB INT8 constant. HTP v68 does not honour this runtime lookup in the context binary, so the embedding output decouples from `input_ids`. (Vision has no analogue: its patch-embedding is a small convolution on a continuous input, not an index lookup into a giant table.) Decisively, the lookup is *not compute* — it is a memory read — so the natural fix is to take it off the NPU rather than to repair it.

**The split-encoder.** Cut the graph at the embedding boundary:

$$
\underbrace{\texttt{input\_ids} \mapsto W_{\text{tok}}[\texttt{input\_ids}]}_{\text{host CPU (FP, exact)}}
\;\longrightarrow\;
\underbrace{\texttt{inputs\_embeds} \;(+\,\text{position}) \to \text{encoder} \to \mathrm{LN} \to \text{pooler} \to \text{head}}_{\text{HTP v68, W8A8}} .
$$

The host performs only the table read (64 rows per query); **all 12 transformer layers — the entire compute cost — remain on the NPU**, now consuming `inputs_embeds` as an ordinary quantized activation input (exactly like the vision image input, which HTP handles correctly). This removes both the dynamic `Gather` and the 192 MB constant from the context binary. With the *rotated* model, $Q$ is already folded into $W_{\text{tok}}$ and the position table, so the host lookup lives in the rotated space and the in-graph position add is consistent — no extra math.

**Mask, built link-safe at the boundary.** The split wrapper does not call HuggingFace's `_prepare_4d_attention_mask` (which emits an `Expand → Cast(\text{float}) → Sub → Cast(\text{bool}) → Where` island plus an `-FLT_MAX` sentinel). It builds the additive mask directly,

$$ M \;=\; (1 - \texttt{attention\_mask}) \cdot (-32), \qquad \texttt{attention\_mask}\in\{0,1\}, $$

shaped $[B,1,1,L]$ and broadcast inside the attention $\mathrm{softmax}(S + M)$. Two HTP-v68 rejections were eliminated this way: a redundant `float32→float32` cast on the mask input (materialized as the float island `/Cast_output_0`), and the **materialized** $[B,1,L,L]$ mask from `Expand` (rejected as a floating coefficient `/Expand_coef`). Feeding the un-expanded $[B,1,1,L]$ mask and letting the attention `Add` broadcast keeps the graph free of internal float islands. The split graph reproduces the full `encode_text` to cosine $0.99999999$.

**Result.** The split-text binary is board-faithful (cosine $0.9951 / 0.9926$, matching the QDQ proxy) and **does** depend on `inputs_embeds` (real-vs-zero embedding outputs differ by $3.7$–$5.3$), confirming the bug is fixed. Text-isolation board retrieval is T2I R@1 $51.30$; end-to-end both-INT8 board retrieval with the final vision v9 tower is T2I R@1 $50.35$ (§12 items 7–10). The memory payoff of §12A.7 is unchanged: the $\sim$192 MB INT8 token table still exists, it simply lives in host/RB3 DRAM (read by the CPU) instead of inside the HTP context — and RB3's CPU and NPU share the same DRAM, so there is no extra memory cost.

---

## 13. References

Quantization / outliers:

- [4] T. Dettmers, M. Lewis, Y. Belkada, L. Zettlemoyer. *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.* NeurIPS 2022. arXiv:2208.07339.
- [5] G. Xiao, J. Lin, M. Seznec, H. Wu, J. Demouth, S. Han. *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models.* ICML 2023. arXiv:2211.10438.
- [6] M. Sun, X. Chen, J. Z. Kolter, Z. Liu. *Massive Activations in Large Language Models.* 2024. arXiv:2402.17762.
- [7] J. Lin, J. Tang, H. Tang, S. Yang, X. Dang, S. Han. *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.* MLSys 2024. arXiv:2306.00978.
- [18] E. Frantar, S. Ashkboos, T. Hoefler, D. Alistarh. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* ICLR 2023. arXiv:2210.17323.

Rotation / incoherence:

- [8] S. Ashkboos, M. L. Croci, M. Gennari do Nascimento, T. Hoefler, J. Hensman. *SliceGPT: Compress Large Language Models by Deleting Rows and Columns.* ICLR 2024. arXiv:2401.15024.
- [9] S. Ashkboos, A. Mohtashami, M. L. Croci, B. Li, M. Jaggi, D. Alistarh, T. Hoefler, J. Hensman. *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs.* NeurIPS 2024. arXiv:2404.00456.
- [10] J. Chee, Y. Cai, V. Kuleshov, C. De Sa. *QuIP: 2-Bit Quantization of Large Language Models With Guarantees.* NeurIPS 2023. arXiv:2307.13304.
- [11] A. Tseng, J. Chee, Q. Sun, V. Kuleshov, C. De Sa. *QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks.* ICML 2024. arXiv:2402.04396.
- [12] Z. Liu, C. Zhao, I. Fedorov, B. Soran, D. Choudhary, R. Krishnamoorthi, V. Chandra, Y. Tian, T. Blankevoort. *SpinQuant: LLM Quantization with Learned Rotations.* 2024. arXiv:2405.16406.

Quantization-aware training:

- [13] Y. Bengio, N. Léonard, A. Courville. *Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation.* 2013. arXiv:1308.3432.
- [14] B. Jacob, S. Kligys, B. Chen, M. Zhu, M. Tang, A. Howard, H. Adam, D. Kalenichenko. *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.* CVPR 2018. arXiv:1712.05877.
- [15] R. Krishnamoorthi. *Quantizing deep convolutional networks for efficient inference: A whitepaper.* 2018. arXiv:1806.08342.
- [16] S. K. Esser, J. L. McKinstry, D. Bablani, R. Appuswamy, D. S. Modha. *Learned Step Size Quantization (LSQ).* ICLR 2020. arXiv:1902.08153.
- [17] G. Hinton, O. Vinyals, J. Dean. *Distilling the Knowledge in a Neural Network.* 2015. arXiv:1503.02531.

Model / training context:

- [1] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022. arXiv:2106.09685.
- [2] X. Zhai, B. Mustafa, A. Kolesnikov, L. Beyer. *Sigmoid Loss for Language Image Pre-Training (SigLIP).* ICCV 2023. arXiv:2303.15343.
