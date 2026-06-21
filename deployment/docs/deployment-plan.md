# mSigLIP Deployment Plan — Setup and Experiment Design

> **Purpose:** define the deployment setup, experiment pipeline, and comparison goals.
> **Not included here:** concrete metric values, job IDs, dated failures, or final claims.
> **Where concrete results live:** [`comprehensive_results.md`](comprehensive_results.md) and [`journal/[deploy-master].md`](journal/[deploy-master].md).
> **Where runnable commands live:** [`runbook-w8a8-v8-both-int8.md`](runbook-w8a8-v8-both-int8.md).

This file is intentionally general. It should remain useful even when a newer model, quantization recipe, board run, or AI Hub job replaces the current one.

---

## 1. Deployment Goal

The deployment goal is to run mSigLIP text-based person search locally on Qualcomm RB3 Gen2:

| Component | Role |
|---|---|
| Vision encoder | converts a person image into an embedding |
| Text encoder | converts a natural-language description into an embedding |
| Retrieval score | compares image and text embeddings |
| Target runtime | Qualcomm QNN / HTP on RB3 Gen2 |

The final deploy result must be measured with **both encoders running through the target deployment path**, not only with host-side proxies.

---

## 2. Setup Layers

### 2.1 Local Host Setup

The local host is used for:

- merging LoRA weights into a deployable checkpoint;
- preparing calibration, smoke, and retrieval inputs;
- exporting and patching ONNX graphs;
- running local PyTorch / ONNX Runtime comparisons;
- submitting AI Hub quantize/compile/link jobs;
- evaluating retrieval from generated embeddings;
- collecting and documenting results.

Expected local dependencies:

| Dependency | Use |
|---|---|
| PyTorch | FP32 reference model and local embedding generation |
| ONNX / ONNX Runtime | static export checks and QDQ proxy evaluation |
| Qualcomm AI Hub client | cloud quantize, compile, and link |
| Dataset files | image/text input preparation and retrieval evaluation |
| Repo scripts under `deployment/scripts/` | reproducible pipeline steps |

### 2.2 AI Hub Setup

AI Hub is used as the cloud compiler/quantizer for QNN context binaries.

For every AI Hub run, record in the deployment journal:

- source ONNX directory;
- calibration dataset identity;
- input specs and dtypes;
- quantization options;
- compile/link options;
- job IDs;
- downloaded artifacts;
- pass/fail message.

AI Hub experiments should be launched only after local static gates pass. This avoids spending cloud jobs on export or preprocessing bugs.

### 2.3 Board Setup

The board is used for:

- running QNN context binaries with `qnn-net-run`;
- collecting raw output embeddings;
- profiling runtime;
- verifying board-vs-reference fidelity;
- producing final board retrieval inputs.

The board runtime must match the version used by the compiled binary. Keep these versioned together:

| Board item | Why it matters |
|---|---|
| `qnn-net-run` | launches the context binary |
| `libQnnHtp.so` | HTP backend |
| `libQnnHtpNetRunExtensions.so` | QNN NetRun extensions |
| HTP skel libraries in `ADSP_LIBRARY_PATH` | DSP-side runtime compatibility |
| QNN config file | HTP architecture/performance settings |

Version mismatches should be treated as setup failures, not model failures.

---

## 3. Artifact Families

Keep generated artifacts under `artifacts/deployment/`.

| Artifact family | Typical content | Purpose |
|---|---|---|
| `exports/` | merged models, rotated models, QAT models, ONNX directories | model versions before AI Hub |
| `qnn_inputs/` | `.raw` image tensors, token arrays, input lists | identical inputs for host and board |
| `runtime/` | QDQ ONNX downloads, proxy summaries, retrieval JSONs | host-side quantized proxy evaluation |
| `bin/` | QNN context binaries copied for board use | local/manual deployment artifacts |
| `qnn_runs/` | board output directories and profiles | outputs pulled back from RB3 |

Large context binaries should not be committed to Git. They are operational artifacts, not source documentation.

---

## 4. Experiment Principles

Each experiment should be designed around one controlled question.

| Principle | Meaning |
|---|---|
| Change one thing at a time | e.g. rotation type, QAT coverage, mask representation, or runtime target |
| Keep input order fixed | retrieval comparisons are only valid if gallery/query ordering is stable |
| Compare to the nearest reference | static ONNX vs PyTorch, QDQ vs PyTorch, board vs PyTorch/QDQ |
| Stop early on gate failure | do not proceed to board or full retrieval when static/QDQ gates fail |
| Separate proxy from board | QDQ proxy predicts board behavior but is not the board result |
| Separate modality isolation from both-INT8 | image-only and text-only runs diagnose different failure modes |
| Record the reason, not just the number | a failed branch is useful only if the blocker is documented |

---

## 5. Pipeline Overview

The general deployment pipeline is:

```text
Checkpoint
  -> merge LoRA
  -> prepare fixed inputs
  -> apply representation transforms
  -> QAT / quantization-aware adaptation
  -> export static ONNX
  -> local static gate
  -> AI Hub QDQ proxy
  -> local QDQ gate
  -> AI Hub compile/link
  -> board smoke run
  -> board fidelity gate
  -> full retrieval evaluation
```

For a two-encoder retrieval model, run this pipeline in three views:

| View | Image encoder | Text encoder | Purpose |
|---|---|---|---|
| Vision isolation | deployment candidate | FP32 reference | diagnose image branch quality |
| Text isolation | FP32 reference | deployment candidate | diagnose text branch quality |
| End-to-end | deployment candidate | deployment candidate | measure final deploy behavior |

### 5.1 Host/Accelerator Split (when a runtime op is not honored)

A compiled graph that **links** is not guaranteed to **execute correctly**. The target runtime may silently fail to honor a specific operation even though the static graph and its quantized proxy are faithful. The diagnostic is an **input-dependence ablation**: hold all other inputs fixed and check that the output actually changes when the operation's input changes; if it does not, the runtime is ignoring that op.

When a non-compute op (e.g. a large dynamic table lookup / gather) is the culprit, the deployment topology can be **split across host and accelerator**: run the offending op on the host CPU and feed its result as an ordinary tensor input to the accelerator graph, keeping all heavy compute on the accelerator. This is a deployment-representation change, not a model-quality change — the model math is unchanged and the memory footprint is unchanged when host and accelerator share DRAM. Validate the split graph the same way as any other candidate: static gate, QDQ proxy, board fidelity, and an input-dependence control proving the split graph now uses its new input.

---

## 6. Experiment Catalogue

### 6.1 Reference Sanity

Question: does the local evaluation reproduce the expected FP32 retrieval behavior?

| Fixed | Variable | Compare | Output |
|---|---|---|---|
| checkpoint, dataset split, preprocessing | none | local FP32 retrieval vs known reference | baseline sanity table |

Use this before interpreting any quantized result. If the reference sanity is off, fix data loading or preprocessing first.

### 6.2 Export Sanity

Question: did ONNX export preserve the FP32 model?

| Fixed | Variable | Compare | Output |
|---|---|---|---|
| model weights and input tensors | export graph | ONNX Runtime embeddings vs PyTorch embeddings | static fidelity summary |

Failing here means the issue is export/preprocess, not quantization.

### 6.3 Rotation / Representation Experiments

Question: can the model representation be changed to reduce quantization sensitivity while preserving FP32 behavior?

| Fixed | Variable | Compare | Output |
|---|---|---|---|
| checkpoint, input tensors, FP32 objective | rotation or representation transform | transformed FP32 vs original FP32 | invariance summary |

This stage should preserve the model function before quantization. It should not be evaluated as a deploy candidate until static invariance passes.

### 6.4 QAT / Quantization Recipe Experiments

Question: which training or adaptation recipe makes the encoder robust to W8A8?

| Fixed | Variable | Compare | Output |
|---|---|---|---|
| transformed model, data split, evaluation protocol | QAT coverage, observer, schedule, loss, rotation source | QDQ fidelity and modality-isolation retrieval | ablation table |

The comparison goal is not only "higher cosine"; it is whether retrieval survives under the deployable quantization format.

### 6.5 QDQ Proxy Experiments

Question: does the AI Hub quantized graph preserve embeddings before compile/link?

| Fixed | Variable | Compare | Output |
|---|---|---|---|
| exported ONNX, calibration dataset, input order | quantized ONNX graph | QDQ embeddings vs PyTorch embeddings | QDQ fidelity summary |
| same as above | quantized branch only | modality-isolation retrieval vs FP32 reference | proxy retrieval JSON |

If QDQ proxy fails, do not debug board runtime yet. The quantized graph itself is already not good enough.

### 6.6 Graph-Linkability Experiments

Question: can the quantized graph be represented in a form accepted by QNN/HTP?

| Fixed | Variable | Compare | Output |
|---|---|---|---|
| quantization intent and model weights | graph dtype / mask / I/O representation | AI Hub compile/link result | context binary or link error |

This stage is about deployment graph legality. Passing link means the graph can become a context binary; it does not by itself prove runtime quality.

### 6.7 Board Smoke Experiments

Question: does the context binary execute correctly on RB3 and produce plausible embeddings?

| Fixed | Variable | Compare | Output |
|---|---|---|---|
| context binary, small fixed input set | board runtime | board outputs vs PyTorch/QDQ | board fidelity summary |
| same as above | performance settings | QNN profile | latency/FPS profile |

This stage should be small and fast. It catches runtime setup, dtype, output decoding, and version mismatch issues before full retrieval.

### 6.8 Full Board Retrieval Experiments

Question: does the board-executed encoder preserve retrieval behavior at dataset scale?

| Fixed | Variable | Compare | Output |
|---|---|---|---|
| full gallery/query order | board-produced embeddings for one or both encoders | board retrieval vs QDQ proxy and FP32 reference | retrieval JSON |

Run this separately for:

- vision-isolation board retrieval;
- text-isolation board retrieval;
- final both-encoder board retrieval.

---

## 7. Comparison Types

Use these names consistently:

| Name | Meaning | What it can prove |
|---|---|---|
| FP32 reference | original merged PyTorch model | baseline retrieval and sanity |
| Static ONNX | exported ONNX before AI Hub quantization | export correctness |
| QDQ proxy | AI Hub quantized ONNX run on host | quantization fidelity and proxy retrieval |
| Context binary | AI Hub linked QNN `.bin` | graph can be deployed to QNN |
| Board smoke | small board run | runtime setup, output decoding, latency, local fidelity |
| Board full retrieval | dataset-scale board outputs | deployment retrieval quality |

Do not mix these when writing conclusions. For example, a QDQ proxy pass is not a board pass; a vision-only board pass is not a both-INT8 board pass.

---

## 8. Gate Template

Use this template for every stage:

| Field | Fill with |
|---|---|
| Experiment name | short stable name |
| Question | what uncertainty this experiment resolves |
| Fixed variables | checkpoint, data, preprocessing, reference branch |
| Changed variable | the one thing being tested |
| Input artifacts | model/input directories |
| Output artifacts | summaries, QDQ graph, context binary, board run directory |
| Primary comparison | e.g. ONNX vs PyTorch, QDQ vs PyTorch, board vs QDQ |
| Pass criteria | threshold or qualitative requirement |
| Failure interpretation | what a fail means and what not to blame |
| Next step if pass | where the pipeline proceeds |
| Next step if fail | what branch to debug |

This keeps the journal readable and prevents repeated ambiguous runs.

---

## 9. Documentation Roles

| Document | Role |
|---|---|
| `deployment-plan.md` | general setup and experiment design |
| `runbook-w8a8-v8-both-int8.md` | exact current commands |
| `comprehensive_results.md` | current clean metric tables |
| `w8a8_qat_rotated.md` | theory and method |
| `journal/[deploy-master].md` | chronological source of record: job IDs, commands, artifacts, failures, conclusions |

Rule of thumb:

- put **commands and exact paths** in the runbook;
- put **numbers and current status** in the result summary and journal;
- put **general experiment design** here;
- put **mathematical reasoning** in the theory document.

---

## 10. Before Starting a New Deployment Experiment

Before launching an expensive AI Hub or board run, write down:

1. What is the single variable being tested?
2. What is the nearest reference comparison?
3. Which input set will be used?
4. Which artifact directory will receive outputs?
5. What result would make the experiment stop?
6. What result would justify the next stage?

If these are unclear, the experiment is not ready yet.
