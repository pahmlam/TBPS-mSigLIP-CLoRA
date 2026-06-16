# Rotated W8A8 + QAT: Deploying the mSigLIP Vision Encoder on RB3 Gen2 (HTP v68)

> **Scope:** vision encoder deployment branch (best current result).
> **Source checkpoint:** `artifacts/models/checkpoints/epoch=56-val_score=52.28.ckpt` (LoRA + Curriculum Circle, seed 2400).
> **Target device:** Qualcomm RB3 Gen2 / QCS6490 / Hexagon HTP **v68**.
> **Best result:** vision-only **T2I Rank@1 = 48.20** (deploy gate ≥ 48 PASS), all-INT8 W8A8 context binary that links and runs on HTP v68.
> **Canonical recipe:** `LoRA merge → mean-preserving rotation → opset-20 fused GELU/LayerNorm → quantization-aware finetune (EMA observer) → W8A8 quantize + compile/link → board run`.
> **Cross-references:** dated logs in [`deployment/docs/journal/`](journal/); core scripts in `deployment/scripts/qnn/` and `deployment/scripts/lora_fp16/`.

This is the canonical engineering document for the best deployment pipeline. It explains the hardware constraints that shape it, the mathematics of every transform, why earlier candidates failed, how to reproduce each stage, and the acceptance gates.

---

## 0. Result Summary

| Stage of the journey | Configuration | QDQ cosine (mean/min) | Vision-only T2I R@1 | Gate (≥ 48) |
|---|---|---:|---:|:--:|
| FP32 reference | merged baseline | 1.000 | **52.40** (≈ paper 52.28) | — |
| Rotation only | mean-preserving rotation + W8A8 | 0.8975 / 0.8747 | 45.42 | FAIL |
| QAT v1 | + fake-quant distill (per-sample) | 0.9223 / 0.8917 | 46.92 | FAIL |
| QAT v2 | + per-tensor fake-quant | 0.9281 / 0.9093 | 47.80 | FAIL |
| **QAT v3** | **+ EMA observer (deploy-faithful)** | **0.9353 / 0.919** | **48.20** | **PASS** |

On-device verification of the linked W8A8 binary (HTP v68, `qnn-net-run`):

| Metric | Value |
|---|---:|
| QNN(board) vs PyTorch cosine, mean | `0.8982` (matches QDQ ONNX `0.8975`, drift ≈ `0.0007`) |
| NetRun latency | `34.25 ms / image` |
| HTP accelerator execute | `32.5 ms / image` |
| Throughput | `22.5 FPS` |
| Context-binary load (one-time) | `54.7 ms` |
| Context-binary size | `89.7 MB` (INT8) |

> Note on artifacts: the on-board binary that has been physically benchmarked is the rotation-only `rotated_w8a8_v2` (R@1 ≈ 45.42, board fidelity 0.898). The **QAT v3** model passes the retrieval gate (48.20) and uses an identical all-INT8 graph shape, so its `.bin` links on v68 the same way; compiling and benchmarking the QAT v3 binary on the board is the last remaining step. Board fidelity tracks QDQ fidelity to within ~0.001, so the QAT v3 board result is expected to match its QDQ R@1.

The decisive acceptance metric is **retrieval Rank@1**, not cosine. Cosine is only a fidelity proxy; §9 explains why a `0.90` cosine still failed the gate while `0.9353` passes.

---

## 1. Why This Pipeline Exists (Hardware Constraints)

RB3 Gen2 runs the Hexagon **HTP v68**. The QNN context-binary linker on v68 imposes hard constraints that eliminate most "easy" quantization paths:

1. **Floating-point graph I/O is rejected.** Context binaries require integer (quantized) I/O. We therefore compile with `--quantize_io`.
2. **Internal floating-point fallback fails to link.** Leaving sensitive layers in float (the "`_float` surgery" approach) passes local ONNX fidelity but the linker rejects internal float tensors (repeated `exit code 14`).
3. **16-bit activations (A16) are only partially supported on v68.** Activation×activation matmuls (attention) and LayerNorm in A16 require a newer arch (`expected >= 73`). W8A16 reaches QDQ fidelity `0.9997` but the link **fails** on v68 at `node_MatMul_774` / `node_layer_norm`.
4. **All-INT8 (W8A8) is the path v68 supports broadly.**

The conflict: the model *needs* high precision exactly where v68 *refuses* high precision (residual + LayerNorm). So the only deployable path is all-W8A8 — and we must make the network tolerate W8A8 instead of asking the hardware for more bits.

Naive per-tensor W8A8 collapses retrieval (cosine `0.14–0.17`). This is **not** an export or preprocessing bug — static ONNX vs PyTorch is `≈ 1.0`. The failure is in **activation quantization** inside the vision encoder. Two root causes, fixed by two transforms:

- **Decomposed cubic GELU** (export artifact) → fixed by **opset-20 GELU fusion** (§3).
- **Massive activations / channel outliers** in the residual stream → fixed by **mean-preserving rotation** (§4), then the residual per-tensor error is recovered by **QAT** (§7).

Clipping the outliers was tried and failed: the outlier channels carry real signal, so clipping destroys information. The correct operation is to *redistribute* the energy (rotation), not remove it.

---

## 2. Pipeline Overview

```text
[0] LoRA checkpoint  (epoch=56-val_score=52.28.ckpt)
     │
     ▼
[1] MERGE LoRA            deployment/scripts/lora_fp16/export.py
     │                    W_merged = W_base + (α/r)·B·A ; strip optimizer
     ▼                    → exported_model/{model_fp32.pt, model_fp16.pt, config.yaml}
[2] PREP INPUTS           deployment/scripts/qnn/prepare_vn3k_vision_inputs.py
     │                    NCHW float32 .raw in [-1,1]; calib set d7jzjy1m2
     ▼
[3] ROTATE                deployment/scripts/qnn/rotate_vision_encoder.py
     │                    mean-preserving orthogonal Q (Q·1=1); fold into weights
     ▼                    residual concentration 252x → 5.3x ; FP32 output-invariant
     │                    → exported_model_rotated/{model_fp32.pt, config.yaml}
[4] QAT DISTILL           deployment/scripts/qnn/train_vision_quant_robust.py
     │                    teacher=rotated FP32 (frozen); student=rotated + fake-quant
     ▼                    per-tensor STE, EMA observer, layers 0–11, 8 epochs
     │                    → exported_model_rotated_qat_v3/{model_fp32.pt, config.yaml}
[5] EXPORT ONNX           deployment/scripts/qnn/export_rotated_vision_onnx.py
     │                    opset 20: Gelu=13, LayerNormalization=26, Pow=0
     ▼                    → .../vision_onnx/{vision_encoder.onnx, .onnx.data}
[6] QUANTIZE + LINK       deployment/scripts/qnn/submit_qaihub_quantize_compile.py
     │                    AI Hub W8A8 + compile DLC + link context binary
     ▼                    --quantize_io ; device "RB3 Gen 2 Vision Kit"
     │                    → runtime/<run>/vision_encoder.bin (89.7 MB)
[7] BOARD RUN             qnn-net-run on HTP v68
     │                    → qnn_runs/<run>/  (Output_*.raw, dequantized to float)
     ▼
[8] EVALUATE              compare_qnn_with_pytorch.py  (board fidelity, cosine)
                          eval_retrieval_quantized_vision.py  (T2I R@1 — the gate)
```

All generated artifacts live under `artifacts/deployment/`.

---

## 2A. Theoretical Foundations & Related Work (the *why* behind each choice)

This pipeline is an engineering composition of three research lines — **outlier-aware quantization**, **rotation / incoherence processing**, and **quantization-aware training** — adapted to a constraint most of that literature never faces: an INT8-only NPU (HTP v68) that rejects 16-bit activations *and* internal float. This section gives the reasoning and the mathematics that motivate the design, with references.

### 2A.1 The problem: transformer activations have outliers that defeat per-tensor INT8

Transformers develop a few *outlier* / *massive* activation channels whose magnitude is orders larger than the rest, concentrated in the residual stream; the effect grows with scale and is now well documented — **LLM.int8()** [4], **SmoothQuant** [5], **Massive Activations** [6]. Per-tensor symmetric INT8 uses a single scale

$$s=\max_i|x_i|/(2^{b-1}-1)$$,

so one outlier inflates $s$ and the remaining channels — which carry the *direction* that L2-normalized retrieval depends on — collapse onto a few levels. Our measured residual concentration of $252\times$ and the plain-W8A8 collapse to cosine $0.14$ (§6.1, §1) are the vision-encoder instance of exactly this phenomenon.

The literature offers two cures: **(i)** keep/migrate the outliers in higher precision — mixed-precision (LLM.int8() [4]), activation→weight scale migration (SmoothQuant [5]), activation-aware weight quant (AWQ [7]); or **(ii)** make the representation *incoherent* by an orthogonal transform so no coordinate is special. **HTP v68's broad ban on A16 eliminates family (i)** (16-bit activations / mixed precision fail to link,§1), which is precisely why we are pushed onto family (ii).

### 2A.2 Rotation as computational invariance folded into weights

- **SliceGPT** [8] established *computational invariance*: inserting $QQ^\top$ ($Q$ orthogonal) on the residual stream and folding $Q$ into adjacent weights leaves the network's function unchanged. Our offline weight folding (§6.5) is exactly this.
- **QuaRot** [9] uses randomized Hadamard rotations to remove outliers before 4-bit inference, folding some rotations offline and doing others online via fast Hadamard. Our residual rotation is the **offline, fully-fused** variant (no runtime op — a hard requirement on v68); our rejected R2 (§6, §10) was a head-dim Hadamard in the same spirit.
- **QuIP / QuIP#** [10, 11] formalize *incoherence processing*: random-orthogonal or Hadamard transforms make weights/Hessians incoherent, with provable error bounds. This supplies the mathematical "why" below.
- **SpinQuant** [12] *learns* the rotation instead of using a fixed random/Hadamard one — the natural next lever if the stretch gate (R@1 ≥ 50) needs more than QAT.

### 2A.3 Mathematical motivation — why a rotation lowers per-tensor error

Uniform quantization error per coordinate has variance $\approx s^2/12$ with

$$s \propto \max_i|x_i|$$

So **per-tensor error scales with the dynamic range $\max_i|x_i|$, not the energy $\lVert x\rVert_2$.** Define the *incoherence*

$$ \mu(x) \;=\; \frac{\sqrt{d}\,\max_i |x_i|}{\lVert x\rVert_2} \;\in\; [\,1,\ \sqrt{d}\,]. $$

An outlier-dominated vector has $\mu\approx\sqrt{d}$ (worst case, $x=\lVert x\rVert e_k$); a perfectly spread vector has $\mu\approx 1$ (best case). Quantization error for fixed energy is monotone in $\mu$, so the goal is to **minimize $\mu$** — the QuIP incoherence objective [10]. An orthogonal $Q$ preserves energy ($\lVert Qx\rVert_2 = \lVert x\rVert_2$) but changes $\mu$: for a random orthogonal $Q$ the coordinates of $Qx$ behave like $\mathcal N(0,\lVert x\rVert_2^2/d)$, so

$$ \max_i |(Qx)_i| \;\approx\; \frac{\lVert x\rVert_2}{\sqrt{d}}\sqrt{2\ln d}
\quad\Longrightarrow\quad \mu(Qx)\approx \sqrt{2\ln d}, $$

versus $\mu(x)$ up to $\sqrt{d}$ for an outlier vector. 

For $d=768$ that is $\sqrt{2\ln d}\approx 3.6$ against $\sqrt d\approx 27.7$ — a dynamic-range (hence scale, hence error) reduction of up to $\sim 8\times$ from the rotation alone; combined
with the specific outlier structure of this encoder it produces the observed $252\times \to 5.3\times$ concentration drop (§6.6). A **Hadamard** matrix (entries $\pm 1/\sqrt d$) is the deterministic optimum ($\mu=1$ on its worst input), which is why R2 used a Sylvester–Hadamard on the head-dim; the residual uses a random *mean-preserving* orthogonal (next).

### 2A.4 Our adaptation: the mean-preserving constraint (the non-obvious twist)

QuaRot/SliceGPT target LLaMA-style models whose norm is **RMSNorm** (no mean subtraction), so *any* orthogonal $Q$ commutes through the norm for free. mSigLIP's vision encoder uses **LayerNorm** (with mean subtraction). Converting LN→RMSNorm to admit an arbitrary $Q$ — the textbook move — *backfired on HTP*: RMSNorm decomposes to $\mathrm{Pow}(x^2)/\mathrm{ReduceMean}/\mathrm{Div}$ and re-exposes the normalization internals to the per-tensor quantizer (collapse to cosine $0.16$, §10). Our resolution is to **keep fused LayerNorm and instead constrain the rotation** to fix the mean axis, $Q\mathbf 1=\mathbf 1$, so $\mathrm{LN}(Qx)=Q\,\mathrm{LN}(x)$ holds (§6.3). This confines the rotation to the $(d{-}1)$-dim subspace orthogonal to $\mathbf 1$ — the same subspace where the outliers live — and is, as far as we know, the practical adaptation that makes QuaRot-style equalization compatible with a **fused-LayerNorm vision encoder on an INT8-only NPU**.

### 2A.5 QAT: training the weights to tolerate the *deploy-faithful* quantizer

Rotation removes concentration but leaves ordinary per-tensor INT8 error accumulating over 12 blocks (§10). Quantization-aware training closes the gap: forward through fake-quant, backward through the **straight-through estimator** [13]; per-tensor fake-quant with **moving-average (EMA) min-max observers** is the standard integer-inference recipe [14, 15]; learnable step sizes (LSQ [16]) are a refinement we did not need. We distill rather than retrain on the task loss: a **teacher–student** [17] setup with the FP32 (rotated) model as frozen teacher and the fake-quant model as student, matching embeddings (cosine + MSE). This is vision-only and label-free, and it directly optimizes the deployment contract — *INT8 embedding ≈ FP32 embedding* — which is what retrieval R@1 depends on. The decisive lesson (§7.3) is **observer faithfulness**: a per-sample/dynamic simulated scale inflates training cosine but does not transfer; matching AI Hub's *calibrate-once, per-tensor* scheme with a per-tensor EMA observer is what crossed the gate ($0.9281\to0.9353$, R@1 $47.80\to48.20$).

### 2A.6 Backbone & training context

The backbone is **SigLIP** [2] (sigmoid image–text contrastive pretraining), multilingual variant, fine-tuned with **LoRA** [1] and a **Circle Loss** [3] curriculum (the source checkpoint). Deployment merges LoRA (§3) before any quantization. Related PTQ baselines we did not adopt include **GPTQ** [18] (weight-only PTQ; our bottleneck is activations, not weights).

---

## 3. Stage [1] — Merge LoRA Into the Base Model

**Script:** `deployment/scripts/lora_fp16/export.py`

```bash
python deployment/scripts/lora_fp16/export.py \
  --ckpt artifacts/models/checkpoints/epoch=56-val_score=52.28.ckpt \
  --output-dir artifacts/deployment/exports/exported_model
```

The training checkpoint is a PyTorch Lightning checkpoint with PEFT LoRA adapters. Qualcomm tooling only sees exported weights; it has no notion of LoRA. LoRA modifies a dense weight matrix as

$$ W_{\text{merged}} = W_{\text{base}} + \frac{\alpha}{r}\, B A, $$

where $W_{\text{base}}$ is the frozen pretrained weight, $A \in \mathbb{R}^{r\times d}$ and $B \in \mathbb{R}^{d\times r}$ are the low-rank factors, $r$ the rank, and $\alpha$ the LoRA scale. The script rebuilds `LitTBPS`, applies the configured LoRA setup, loads the checkpoint, then calls PEFT `merge_and_unload()` so the backbone becomes an ordinary inference model.

**Verification (must pass):** `model_fp32.pt` contains **0 keys** matching `lora` / `adapter` / `base_layer`. Keys look like `backbone.vision_model...k_proj.weight`.

`model_fp32.pt` is the reference for ONNX export and all fidelity comparisons; `model_fp16.pt` is for size/fallback experiments. **Rule:** when switching to a different checkpoint (e.g. a future 53.00 model with Part-Align + Attn/FFN LoRA), rerun from this stage — rotation/quantization artifacts are model-specific and must be regenerated.

---

## 4. Stage [2] — Calibration & Smoke Inputs

**Script:** `deployment/scripts/qnn/prepare_vn3k_vision_inputs.py`

```bash
# 10-image smoke set (fidelity comparisons)
python deployment/scripts/qnn/prepare_vn3k_vision_inputs.py \
  --dataset-root VN3K --split test --selection first --num-samples 10 \
  --output-dir artifacts/deployment/qnn_inputs/vn3k_test_10 --path-mode relative

# 2000-image calibration set (AI Hub quantizer)
python deployment/scripts/qnn/prepare_vn3k_vision_inputs.py \
  --dataset-root VN3K --split train --selection random --seed 2400 --num-samples 2000 \
  --output-dir artifacts/deployment/qnn_inputs/vn3k_train_calib_2000 --path-mode relative
```

Known AI Hub calibration dataset: `d7jzjy1m2` (`msiglip-vision-vn3k-train-calib-2000`).

**Preprocessing** (must match training exactly): RGB → resize `256×256` bicubic → `ToTensor` (NCHW) → `Normalize(mean=0.5, std=0.5)` → float32 in `[-1, 1]`. Each raw input is `1×3×256×256` float32 = `786432` bytes. Using identical `.raw` tensors for `qnn-net-run` and the local PyTorch/ONNX comparisons isolates quantization/runtime error from any image-decoding drift.

---

## 5. Stage [3] — Opset-20 Fused GELU (and Fused LayerNorm)

The mSigLIP MLP uses the tanh GELU approximation:

$$ \mathrm{GELU}(x) = 0.5\,x\left[1 + \tanh\!\left(\sqrt{\tfrac{2}{\pi}}\,(x + 0.044715\,x^3)\right)\right]. $$

When exported at **opset 18**, ONNX has no `Gelu` op, so this decomposes into primitives including `Pow(x, 3)`. The internal $x^3$ term reaches activation magnitudes around `119k`, which completely dominates any per-tensor quantization scale. This single artifact explains a whole family of historical failures: W8A16 link failures clustered around `gelu_*` tensors, and "SmoothQuant is neutral" observations.

**Fix:** export at **opset 20**, where `Gelu` is a fused operator. The cubic stays *inside* the runtime/HTP operator and is never exposed as a quantized tensor. The successful ONNX op signature is:

```text
Gelu = 13     Pow = 0     Tanh = 0     LayerNormalization = 26     ReduceMean = 0
```

Every successful branch after this breakthrough keeps **opset 20 + fused Gelu + fused LayerNormalization**. (HTP has native GELU and LayerNorm, so these fused ops link safely.)

---

## 6. Stage [3 cont.] — Mean-Preserving Rotation Equalization

**Script:** `deployment/scripts/qnn/rotate_vision_encoder.py`

```bash
python deployment/scripts/qnn/rotate_vision_encoder.py \
  --model-dir artifacts/deployment/exports/exported_model \
  --output-dir artifacts/deployment/exports/exported_model_rotated \
  --input-dir artifacts/deployment/qnn_inputs/vn3k_test_10 \
  --seed 2400 --skip-r2
```

> `--skip-r2` reproduces the canonical pre-R2 rotated model. Phase C (R2, a head-dim Hadamard on the attention value/output path) is implemented but was **rejected** — it aimed at the value path while the residual error is in the MLP activations (§10), so it did not help and slightly hurt.

### 6.1 The per-tensor quantization problem

Per-tensor symmetric INT8 uses one scale $s$ for the whole tensor:

$$ x_{\text{int}} = \mathrm{round}(x/s), \qquad \hat{x} = s\cdot x_{\text{int}}, \qquad s = \frac{\max_i |x_i|}{2^{b-1}-1}. $$

If one channel is much larger than the rest, $s$ is set by that outlier and ordinary channels (e.g. magnitude ~6) collapse to a handful of quantization levels. For retrieval this is fatal: the metric depends on the embedding **direction** after L2 normalization, not on raw scale. Measured residual-stream concentration before rotation is ≈ `252×` (worst-channel abs-max over median), with absolute peaks in the thousands.

### 6.2 Rotation idea: redistribute, don't clip

Let $x \in \mathbb{R}^{d}$ ($d=768$) be a residual vector and $Q$ an orthogonal matrix. Represent the residual stream as $x_{\text{rot}} = Qx$. Because $Q$ is orthogonal,

$$ \lVert Qx \rVert_2 = \lVert x \rVert_2 . $$

The semantic content (norms, inner products, hence cosine similarity) is preserved exactly, but a spiky channel's energy is spread across many coordinates, so per-tensor INT8 is no longer dominated by one coordinate. $Q$ is **folded offline** into existing weights — no new runtime `MatMul`. This is the computational invariance of SliceGPT [8] combined with the incoherence rationale of QuIP [10] / QuaRot [9]; see §2A.3 for the error bound.

### 6.3 Why $Q$ must preserve the mean

An earlier version converted LayerNorm → RMSNorm so an arbitrary orthogonal $Q$ would commute through normalization. It failed: RMSNorm exports as `Pow(x,2)` + `ReduceMean` + division, re-exposing normalization internals to the quantizer and collapsing W8A8 again (cosine `0.16`, §10).

The successful version keeps **fused LayerNorm**, so $Q$ must commute with LayerNorm. For identity-affine LayerNorm,

$$ \mathrm{LN}(x) = \frac{x - \mathrm{mean}(x)\,\mathbf{1}}{\mathrm{std}(x)} . $$

For arbitrary orthogonal $Q$, $\mathrm{mean}(Qx) \neq \mathrm{mean}(x)$ in general, so LN does not commute. The fix is to build $Q$ that fixes the all-ones direction:

$$ Q\mathbf{1} = \mathbf{1}, \qquad Q^{\top}Q = I . $$

Then $\mathrm{mean}(Qx) = \tfrac{1}{d}\mathbf{1}^{\top}Qx = \tfrac{1}{d}\mathbf{1}^{\top}x = \mathrm{mean}(x)$ and $\mathrm{std}(Qx) = \mathrm{std}(x)$, giving

$$ \mathrm{LN}(Qx) = Q\,\mathrm{LN}(x). $$

So we can rotate the residual stream and keep `LayerNormalization` fused. The construction (`_mean_preserving_orthogonal`):

$$ Q = U\, \mathrm{blockdiag}(1, R_c)\, U^{\top}, \qquad U[:,0] = \mathbf{1}/\sqrt{d}, $$

where $R_c$ is a random orthogonal matrix on the $(d-1)$-dimensional complement of the mean axis. $Q$ is identity on the mean direction and a random rotation on the complement — exactly where channel outliers live.

### 6.4 Folding affine into the reader

LayerNorm has affine params $\mathrm{LN}_{\text{affine}}(x) = \gamma \odot \mathrm{LN}_{\text{id}}(x) + \beta$. For a linear consumer $y = W(\gamma \odot h + \beta) + b$, fold the affine into the reader:

$$ W' = W\,\mathrm{diag}(\gamma), \qquad b' = b + W\beta, $$

then set LN affine to identity ($\gamma=1, \beta=0$). Same FP32 function, now rotation-compatible. Folded readers: `q/k/v_proj` after `layer_norm1`; `mlp.fc1` after `layer_norm2`; K/V slices of `head.attention` after `post_layernorm`.

### 6.5 Folding $Q$ into writers and $Q^{\top}$ into readers

A residual **writer** $y = Wx + b$ must now emit $Qy$:

$$ W' = QW, \qquad b' = Qb. $$

Writers: patch-embedding conv output channels, position-embedding rows, every `out_proj`, every `mlp.fc2`.

A residual **reader** sees $x_{\text{rot}} = Qx$ and must recover the original basis:

$$ W' = W Q^{\top}. $$

Readers: every `q/k/v_proj`, every `mlp.fc1`, and the K/V slices of the pooling-head attention. The learned head query/probe is *not* rotated — rotation is localized in the encoder residual stream and undone at the head K/V boundary.

### 6.6 Acceptance gates (rotation is accepted only if FP32 is invariant)

```text
Phase A invariance cosine mean/min ≈ 1.0
Phase B invariance cosine mean/min ≈ 1.0
Q orthogonality max error ≈ 3e-15
Q·1 = 1 max error ≈ 1e-14
reload cosine ≈ 1.0   (min 0.99999988)
residual concentration: 252x → 5.3x
```

The drop `252× → 5.3×` is the quantization payoff: the signal is still present, just no longer concentrated in one dominant channel.

---

## 7. Stage [4] — Quantization-Aware Finetune (the gate-passing step)

Rotation alone reaches QDQ cosine `0.8975` but only **R@1 = 45.42** (gate FAIL). The residual error is no longer one outlier; it is ordinary per-tensor INT8 error **accumulated across 12 blocks** (§10). The fix is to train the FP32 weights to tolerate that INT8 noise.

**Script:** `deployment/scripts/qnn/train_vision_quant_robust.py`

```bash
python deployment/scripts/qnn/train_vision_quant_robust.py \
  --model-dir artifacts/deployment/exports/exported_model_rotated \
  --train-input-dir artifacts/deployment/qnn_inputs/vn3k_train_4302 \
  --val-input-dir   artifacts/deployment/qnn_inputs/vn3k_test_100 \
  --output-dir      artifacts/deployment/exports/exported_model_rotated_qat_v3 \
  --start-layer 0 --end-layer 11 \
  --fake-quant-granularity per_tensor \
  --fake-quant-observer ema --ema-momentum 0.99 \
  --batch-size 24 --epochs 8 --lr 1e-5
```

> v3 trained all 12 layers (0–11) on the full VN3K train split (4302 images), batch 24, 8 epochs (1440 steps). Point `--train-input-dir` at raw inputs prepared the same way as the calibration set but for the full train split.

### 7.1 Teacher–student distillation

The script does **not** export a custom QDQ graph. It finetunes the FP32 vision encoder under injected fake-quant noise, then saves a normal FP32 export directory which the existing ONNX + AI Hub quantizer then process. Setup:

- **Teacher:** the rotated FP32 model, frozen (`requires_grad=False`).
- **Student:** a deep copy of the rotated model, with fake-quant forward hooks on the GELU output and the residual output of each selected block (and optionally the pooling head via `--quant-head`).
- **Trainable:** only the selected encoder layers + `visual_projection`; everything else frozen.

Per step, the student is run twice — clean (hooks disabled) and fake-quant — and distilled toward the teacher embedding:

$$ \mathcal{L} = \underbrace{\big(1 - \cos(z_s^{q}, z_t)\big) + \lambda\,\lVert z_s^{q} - z_t\rVert^2}_{\text{fake-quant path}} + \underbrace{w_c\big(1 - \cos(z_s^{c}, z_t)\big) + w_m\lVert z_s^{c} - z_t\rVert^2}_{\text{clean consistency}}, $$

where $z_t = teacher.encode_{image}(x)$, $z_s^{q}$/ $z_s^{c}$ are the fake-quant / clean student embeddings, $\lambda = w_m = 0.05$, $w_c = 1.0$. The clean term prevents the student from drifting away from the teacher when quant noise is off.

### 7.2 Straight-through fake-quant

Symmetric INT8 fake-quant with a straight-through estimator (STE):

$$ q(x) = s\cdot\mathrm{clamp}\!\Big(\mathrm{round}(x/s),\,-q_{\max},\,q_{\max}\Big), \quad q_{\max} = 2^{b-1}-1 = 127, \quad s = \frac{\max|x|}{q_{\max}}. $$

In code: `x + (quantized - x).detach()` — the forward pass is quantized, the backward pass is identity, so gradients flow to the FP32 weights through the rounding. This is the straight-through estimator [13], the standard way to backprop through the non-differentiable `round`.

### 7.3 Why per-tensor, and why EMA observer (the two key fixes)

The R@1 trajectory `45.42 → 46.92 → 47.80 → 48.20` came from making the *simulated* quantizer match the *real* AI Hub W8A8:

- **v1 → v2: per-sample → per-tensor.** A per-sample scale (one scale per image) is too easy: simulated cosine looks great (`0.975`) but does not transfer to AI Hub's per-tensor scheme (real `0.92`). Per-tensor fake-quant uses one scale per activation tensor, matching deployment.
- **v2 → v3: dynamic → EMA observer.** A per-batch dynamic max recomputes the scale every forward; AI Hub instead **calibrates once** on the calibration set and freezes the scale. The EMA observer is the standard moving-average min-max observer [14, 15]; it mimics calibrate-once with a running max,

$$ m_t = \mu\, m_{t-1} + (1-\mu)\,\max|x_t|, \quad \mu = 0.99, $$

producing a fixed-ish per-tensor scale. This closed the remaining sim↔real gap: simulated val cosine stayed ≈ `0.98`, but real QDQ cosine rose `0.9281 → 0.9353` and R@1 crossed the gate.

### 7.4 Result

| Task | Model | R@1 | R@5 | R@10 | mAP | mINP |
|---|---|---:|---:|---:|---:|---:|
| T2I | FP32 baseline | 52.40 | 79.38 | 87.80 | 57.38 | 50.67 |
| T2I | **QAT v3 INT8** | **48.20** | 75.42 | 85.10 | 53.39 | 46.60 |
| I2T | FP32 baseline | 55.30 | 81.45 | 89.70 | 51.38 | 34.50 |
| I2T | **QAT v3 INT8** | **52.30** | 78.90 | 86.85 | 47.89 | 31.03 |

QDQ cosine `0.9353` / min `0.919`. Drop vs FP32: `4.20` (gate allows ≤ `4.28`). After QAT, re-export ONNX (§8) and re-quantize (§9) on `exported_model_rotated_qat_v3`.

---

## 8. Stage [5] — Export the Rotated/QAT Vision ONNX

**Script:** `deployment/scripts/qnn/export_rotated_vision_onnx.py`

```bash
python deployment/scripts/qnn/export_rotated_vision_onnx.py \
  --model-dir artifacts/deployment/exports/exported_model_rotated_qat_v3 \
  --opset 20
```

Output is a **directory** (`vision_onnx/vision_encoder.onnx` + `.onnx.data`) because SigLIP weights are large and AI Hub expects the graph and its external-data file uploaded together.

Required export properties and gate:

```text
opset = 20 ; Pow = 0 ; Tanh = 0 ; Gelu = 13 ; LayerNormalization = 26 ; ReduceMean = 0
ONNX vs PyTorch (rotated) cosine_l2_mean ≈ 1.0 , min ≈ 1.0
```

Because the rotated/QAT PyTorch model is output-invariant relative to the *rotated* model, this static control also confirms the graph is well-formed. **Any export showing RMSNorm-style `Pow`/`ReduceMean` clusters is not the successful pipeline** — that branch returned QDQ cosine ≈ `0.16`.

---

## 9. Stage [6] — AI Hub W8A8 Quantize, Compile, Link

**Script:** `deployment/scripts/qnn/submit_qaihub_quantize_compile.py`

```bash
# quantize-only fidelity check first (1 job)
python deployment/scripts/qnn/submit_qaihub_quantize_compile.py \
  --model artifacts/deployment/exports/exported_model_rotated_qat_v3/vision_onnx \
  --calibration-data d7jzjy1m2 --weights-dtype int8 --activations-dtype int8 \
  --quantize-only --wait \
  --download-quantized artifacts/deployment/runtime/rotated_w8a8_qat_v3/qaihub_qdq

# full quantize + compile + link + download the context binary
python deployment/scripts/qnn/submit_qaihub_quantize_compile.py \
  --model artifacts/deployment/exports/exported_model_rotated_qat_v3/vision_onnx \
  --calibration-data d7jzjy1m2 --weights-dtype int8 --activations-dtype int8 \
  --wait \
  --download artifacts/deployment/runtime/rotated_w8a8_qat_v3/vision_encoder.bin
```

What the helper does: (1) copies the ONNX dir and rewrites input shape to static `image: 1×3×256×256`; (2) resolves `d7jzjy1m2` to an AI Hub calibration dataset; (3) submits the W8A8 quantize job; (4) submits compile (DLC) + link jobs targeting `Dragonwing RB3 Gen 2 Vision Kit`; (5) compiles with `--quantize_io` so graph I/O is quantized rather than left as float boundary tensors.

The retrieval gate, not cosine, is decisive. Always compare the QDQ ONNX against the **original** merged FP32 model (`--model-dir exported_model`), because the deployment contract is "match the original FP32 model," and the rotated/QAT model is mathematically/behaviorally equivalent on the validation objective.

**Why this finally links on v68:** no float I/O (`--quantize_io`), no internal float surgery, no A16 (avoids the v73 requirement), no decomposed GELU cubic, no decomposed RMSNorm internals — every main activation is W8A8, which v68 supports broadly. Reference job chain for the rotation-only v2 binary: quantize `jpv4j8lkp` → compile `jpxmwq8lg` → link `jp2j211q5`, all SUCCESS → `vision_encoder.bin` 89.7 MB.

---

## 10. Stages [7–8] — Board Run and Evaluation

**Run on RB3** (`qnn-net-run`, HTP v68):

```bash
qnn-net-run \
  --backend "$QNN_LIB/libQnnHtp.so" \
  --retrieve_context artifacts/deployment/runtime/rotated_w8a8_qat_v3/vision_encoder.bin \
  --config_file deployment/config/qnn/htp_config_245.json \
  --input_list artifacts/deployment/qnn_inputs/vn3k_test_10/input_list.txt \
  --output_dir artifacts/deployment/qnn_runs/rotated_w8a8_qat_v3 \
  --profiling_level basic --perf_profile high_performance
```

Graph I/O is `QNN_DATATYPE_UFIXED_POINT_8` (from `--quantize_io`); `qnn-net-run` dequantizes outputs to float `.raw` files. **Board fidelity** (`compare_qnn_with_pytorch.py`) for the v2 binary was `0.8982`, matching the QDQ ONNX `0.8975` to ≈ `0.0007` — i.e. HTP runtime is faithful to the quantized graph; the remaining error is quantization, not hardware drift.

**Retrieval gate** (`eval_retrieval_quantized_vision.py`, on Mac/local, free):

```bash
python deployment/scripts/qnn/eval_retrieval_quantized_vision.py \
  --qdq-onnx artifacts/deployment/runtime/rotated_w8a8_qat_v3/qaihub_qdq \
  --model-dir artifacts/deployment/exports/exported_model \
  --dataset-root VN3K --gate-r1 48.0 --also-cosine \
  --cache artifacts/deployment/runtime/rotated_w8a8_qat_v3/retrieval_embeddings.npz \
  --json artifacts/deployment/runtime/rotated_w8a8_qat_v3/retrieval_r1.json
```

Methodology that makes the number trustworthy:
- **Image embedding** = QDQ ONNX (rotated W8A8) via onnxruntime; **text embedding** = FP32 PyTorch `encode_text`. Text is kept FP32 for both baseline and quantized so the measurement isolates the vision quantization.
- Retrieval mirrors `LitTBPS._compute_metrics` exactly: **raw (un-normalized) pooler features**, dot-product similarity, `utils.metrics.rank`. (The generic `Evaluator` normalizes and does not reproduce 52.28.)
- **Sanity:** FP32 baseline reproduces **T2I R@1 52.40 ≈ 52.28** (±1.5), so the quantized number is comparable.

> Isolation note: this isolates the vision branch (text still FP32). The final both-INT8 system can only be **≤** this number, so a vision-only result of 48.20 is an *upper bound*; the text encoder must also pass before the end-to-end board system is accepted.

---

## 11. Why Earlier Candidates Failed

| Attempt | Result | Lesson |
|---|---|---|
| FP32/FP16 ONNX directly on HTP | link fail | v68 context binary requires integer I/O. |
| Deprecated CLI INT8 path | preserves FP I/O | Use Python-API quantize + compile/link with `--quantize_io`. |
| Plain W8A8 (no rotation) | cosine `0.13–0.17` | Runtime works; per-tensor INT8 destroys embedding direction. |
| More calibration samples | still fails | Not a calibration-coverage problem. |
| Lite-MP / min-max / W8A16 on old graph | fails | Global knobs cannot fix exposed GELU cubic / outliers / concentration. |
| `_float` QDQ surgery | local pass, link fail | v68 rejects internal float tensors. |
| ORT W8A16 QDQ | local ≈ `0.999`, link fail | Linker rejects internal float / dequantized GELU patterns. |
| Clipping INT8 activations (max-abs 4–64) | `0.12–0.40`, fails | Outlier channels carry real signal; clipping loses information. |
| Opset-20 + W8A16 | QDQ `0.9997`, link fail | A16 attention/LayerNorm need HTP v73+; RB3 is v68. |
| Rotation with RMSNorm | cosine `0.16` | RMSNorm exposes `Pow(x²)`/`ReduceMean` to the quantizer. |
| Phase-C R2 (head-dim Hadamard) | R@1 `45.25`, no gain | Targets the value path; residual error is in MLP activations. |
| **Mean-preserving rotation + fused LN + W8A8** | board pass, R@1 `45.42` | All-INT8, v68-compatible, but per-tensor error accumulates over 12 blocks. |
| **+ QAT (per-tensor + EMA observer)** | **R@1 `48.20`, gate PASS** | Train the weights to tolerate the deploy-faithful INT8 noise. |

---

## 12. Acceptance Gates

| Gate | Threshold | Status |
|---|---:|---|
| Merge LoRA → clean non-adapter weights | no `lora`/`adapter` keys | PASS |
| Rotation FP32 invariance | cosine min ≥ `0.9999` | PASS |
| ONNX static control | cosine mean ≥ `0.999` | PASS |
| ONNX op sanity | `Pow=0`, fused `Gelu`, fused `LayerNormalization` | PASS |
| QDQ ONNX vs PyTorch (QAT v3) | proxy ≥ `0.95/0.90` | `0.9353 / 0.919` (near; proxy is conservative) |
| QNN board vs PyTorch | mean ≥ `0.90` | `0.8982` (rotation-only v2 binary) |
| Board execution | finite outputs, HTP profile | PASS (v2 binary) |
| **Full retrieval T2I R@1** | **≥ 48.0** | **PASS — 48.20 (QAT v3, vision-only)** |
| Stretch gate | ≥ 50.0 | not reached |

Cosine is a conservative proxy set before the rotation/QAT W8A8 path existed; retrieval R@1 is the decisive metric and it passes.

---

## 13. Reproducible Command Sequence (canonical best path)

```bash
# [1] merge LoRA
python deployment/scripts/lora_fp16/export.py \
  --ckpt artifacts/models/checkpoints/epoch=56-val_score=52.28.ckpt \
  --output-dir artifacts/deployment/exports/exported_model

# [2] prepare smoke + calibration inputs (see §4)

# [3] mean-preserving rotation (skip R2)
python deployment/scripts/qnn/rotate_vision_encoder.py \
  --model-dir artifacts/deployment/exports/exported_model \
  --output-dir artifacts/deployment/exports/exported_model_rotated \
  --input-dir artifacts/deployment/qnn_inputs/vn3k_test_10 --seed 2400 --skip-r2

# [4] QAT distillation (per-tensor + EMA observer)
python deployment/scripts/qnn/train_vision_quant_robust.py \
  --model-dir artifacts/deployment/exports/exported_model_rotated \
  --train-input-dir artifacts/deployment/qnn_inputs/vn3k_train_4302 \
  --val-input-dir artifacts/deployment/qnn_inputs/vn3k_test_100 \
  --output-dir artifacts/deployment/exports/exported_model_rotated_qat_v3 \
  --start-layer 0 --end-layer 11 \
  --fake-quant-granularity per_tensor --fake-quant-observer ema --ema-momentum 0.99 \
  --batch-size 24 --epochs 8 --lr 1e-5

# [5] export rotated/QAT ONNX (opset 20)
python deployment/scripts/qnn/export_rotated_vision_onnx.py \
  --model-dir artifacts/deployment/exports/exported_model_rotated_qat_v3 --opset 20

# [6] quantize-only fidelity check, then full quantize + compile + link (see §9)

# [7] run on RB3 (see §10)

# [8] board fidelity + retrieval R@1 gate (see §10)
```

---

## 14. Common Mistakes to Avoid

- Do not skip the LoRA merge — the checkpoint is not directly a deployment model.
- Do not reuse rotation/QDQ artifacts across different checkpoints (they are model-specific).
- Do not export at opset 18 for the final path — it exposes `Pow(x³)` from tanh-GELU.
- Do not convert LayerNorm → RMSNorm for the v68 W8A8 path — it exposes normalization internals to the quantizer.
- Do not compile `_float` QDQ-surgery candidates — they pass local diagnostics but fail the HTP link.
- Do not use W8A16 as the final v68 plan — it passes fidelity but fails the link at attention/LayerNorm (needs v73+).
- Do not use per-sample / per-batch fake-quant for QAT — it inflates simulated cosine but does not transfer; use per-tensor + EMA observer.
- Do not trust AI Hub/AIMET PSNR as a retrieval proxy — always compute embedding cosine and, decisively, retrieval R@1.
- Use `qnn-net-run` (not `snpe-net-run`) for QNN context binaries, and `--quantize_io` (no float graph I/O).

---

## 15. What Remains

1. **Compile/link/benchmark the QAT v3 binary on the board** (same all-INT8 shape as v2; expected to link and to match its QDQ R@1 ≈ 48.20).
2. **Quantize the text encoder** with the same recipe (opset-20 fused GELU + mean-preserving rotation + W8A8 + QAT). Text is ~75% of parameters — the 250k-vocab embedding alone is ~768 MB FP32 — so text INT8 is the real 4 GB RAM payoff (both encoders INT8 ≈ 372 MB vs ~1.2 GB today).
3. **End-to-end board retrieval** with both encoders INT8; the both-INT8 R@1 is ≤ the vision-only 48.20, so the text branch must hold the gate.
4. **Stretch gate 50:** if needed, add `--quant-head` QAT (the pooling head's INT8 error is not averaged out by later layers), more data/epochs, or learned rotations (SpinQuant [12]) — never revert to float surgery or A16.

---

## 16. References

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
- [3] Y. Sun, C. Cheng, Y. Zhang, C. Zhang, L. Zheng, Z. Wang, Y. Wei. *Circle Loss: A Unified Perspective of Pair Similarity Optimization.* CVPR 2020. arXiv:2002.10857.


