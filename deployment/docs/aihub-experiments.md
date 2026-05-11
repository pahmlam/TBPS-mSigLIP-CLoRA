# Qualcomm AI Hub Compile Experiments — Running Log

> **Purpose:** Record every `qai-hub` submit-compile-job attempt (and related on-device compile tests) — the exact command, the error or success, and the lesson learned. Keeps the team from re-running the same failing config.
>
> **Rule:** Append a new row to the table below after **every** `qai-hub` invocation, successful or not. See `.claude/rules/aihub-experiments.md` for the enforcement rule.
>
> **Legend:** ✅ success · ❌ failed · ⚠️ partial (warning, ran with degraded behavior)

---

## Summary of Learnings (read this first)

1. **Upload format** — `qai-hub` does **NOT** auto-upload the `.onnx.data` companion file. You must pass a **directory** (`--model exported_model/vision_onnx/`), not the `.onnx` file alone. CLI zips and uploads the whole directory.
2. **Static shapes required** — `qnn_context_binary` target rejects any dynamic axes. Resolve with `--input_specs '{"name": ((dim1, dim2, ...), "dtype")}'` (Python dict literal, eval'd by the CLI).
3. **No floating-point I/O on HTP** — QCS6490 HTP V68 rejects **any** float type (FP32 **and** FP16) at input/output tensors. Internal FP16 compute is fine, but boundary tensors must be INT8 or INT16.
4. **`--quantize_full_type`** — Works only with non-ONNX runtimes (`qnn_context_binary`, `tflite`). Not supported with `--target_runtime onnx`. Quantizes weights to the requested type but keeps I/O as FP32 by default (via implicit `--preserve_io_datatype`).
5. **`--quantize_io`** — Not a recognized qai-hub compile option. Passing it in `--compile_options` is silently ignored.
6. **`--input_specs` dtype must match the model** — You cannot declare the input as `float16` if the ONNX file's graph declares it as `float32`. AI Hub validates before compiling.
7. **Pre-quantize ONNX locally** via `onnxconverter_common.float16.convert_float_to_float16(..., keep_io_types=False)` → converts both weights and I/O to FP16. Useful for GPU target, but still rejected by HTP (because HTP needs INT, not FP).
8. **`--preserve_io_datatype` auto-injection** — When using `--quantize_full_type int8`, AI Hub **automatically injects** `--preserve_io_datatype` into the qairt-converter and qairt-quantizer commands. This keeps I/O tensors in their original FP type even though internal weights/activations are quantized to INT8 — which HTP then rejects at the context-binary stage. The only way to avoid this is to use the newer `submit_quantize_job` API (which does not auto-inject the flag) or to submit a compile job without `--quantize_full_type` and handle quantization separately.

---

## Experiment Table

| # | Date (UTC+7) | Job ID | Model | Target | Key Flags | Result | Error excerpt / outcome | Lesson |
|---|--------------|--------|-------|--------|-----------|--------|-------------------------|--------|
| 1 | 2026-04-15 10:25 | — | `exported_model/vision_encoder.onnx` (1.4 MB, graph only) | `qnn_context_binary` | (none) | ❌ | `The uploaded ONNX model is missing its external weights. Please use … ONNX model directory format.` | Uploading the bare `.onnx` file loses the `.onnx.data` companion → upload a **directory** instead. |
| 2 | 2026-04-15 10:50 | `jgn9139q5` | `exported_model/vision_onnx/` (dir, 356 MB) | `qnn_context_binary` | (none) | ❌ | `Model input 'image' has dynamic shapes. Please use a static shape.` | Upload succeeded (330 MB zip). `qnn_context_binary` requires static shapes — pass `--input_specs`. |
| 3 | 2026-04-15 11:05 | — | `vision_onnx/` | `qnn_context_binary` | `--input_specs "image:1,3,256,256:float32"` | ❌ | `SyntaxError: invalid syntax` — qai-hub does `eval()` on the string | Format must be a Python dict literal, not colon-separated. |
| 4 | 2026-04-15 11:15 | `j563onvy5` | `vision_onnx/` | `qnn_context_binary` | `--input_specs '{"image": ((1, 3, 256, 256), "float32")}'` | ❌ | `Tensor 'image' has a floating-point type which is not supported by the targeted device. Please quantize the model including its I/O and try again.` | HTP rejects FP32 I/O. Need to quantize or cast I/O. |
| 5 | 2026-04-15 12:12 | `jp2k1l3xg` (approx) | `vision_onnx/` | `qnn_context_binary` | `--input_specs` FP32 + `--compile_options " --target_runtime qnn_context_binary --quantize_full_type float16 --quantize_io"` | ❌ | Same error — converter cmd still included `--preserve_io_datatype image output_0`; `--quantize_io` silently ignored | `--quantize_io` is not a valid qai-hub compile option. Internal FP16 conversion runs but I/O stays FP32. |
| 6 | 2026-04-15 12:18 | `jgdr6o86p` | `vision_onnx/` (still FP32 model) | `qnn_context_binary` | `--input_specs '{"image": ((1, 3, 256, 256), "float16")}' --quantize_full_type float16` | ❌ | `Provided input_shapes={'image': ((1, 3, 256, 256), 'float16')} does not match shapes inferred from the model {'image': ((-1, 3, 256, 256), 'float32')}.` | Cannot override I/O dtype via `--input_specs` — model's declared dtype wins. |
| 7 | 2026-04-15 12:30 | — | `vision_onnx/` | `onnx` (intermediate step) | `--target_runtime onnx --quantize_full_type float16` | ❌ | `qai_hub.client.UserError: The --quantize_full_type option is not supported for target_runtime='ONNX'.` | Two-step (quantize via ONNX target, then compile) is blocked because quantization flags don't apply to ONNX target. Must pre-quantize locally. |
| 8 | 2026-04-15 13:36 | — (local, not AI Hub) | Local `onnxconverter_common.float16.convert_float_to_float16(keep_io_types=False)` on `vision_onnx/` | — | `deployment/scripts/onnx/to_fp16.py` | ✅ | Produced `vision_onnx_fp16/` — 178.7 MB (2× smaller), I/O & weights all FP16. Some truncation warnings for weights < 1e-7 (expected, harmless). | Local tool works. Next: can AI Hub accept an all-FP16 model for HTP? |
| 9 | 2026-04-15 13:45 | `jp27om9r5` | `vision_onnx_fp16/` (178.7 MB, FP16 I/O) | `qnn_context_binary` | `--input_specs '{"image": ((1, 3, 256, 256), "float16")}'` | ❌ | `Tensor 'image' has a floating-point type which is not supported by the targeted device. Please quantize the model including its I/O and try again.` | HTP rejects **any** floating-point I/O, not just FP32. Need INT8 or INT16 at boundaries. |
| 10 | 2026-05-06 14:30 | `jpyvrrv7p` | `vision_onnx/` (356 MB, FP32) | `qnn_context_binary` | `--quantize_full_type int8 --calibration_data none` | ❌ | `Tensor 'image' has a floating-point type which is not supported by the targeted device.` — converter cmd showed `--preserve_io_datatype image output_0` injected **twice** | `--quantize_full_type int8` auto-injects `--preserve_io_datatype`, keeping I/O as FP32 despite INT8 internal quantization. HTP still rejects FP I/O at context-binary stage. |
| 11 | 2026-05-06 15:10 | `jgkr7qwn5` | `vision_onnx/` (356 MB, FP32) | `qnn_context_binary` | `--quantize_full_type int8 --calibration_data none` (without `--preserve_io_datatype`) | ✅ | Full pipeline completed: ONNX → DLC → INT8 quantize (weights+act 8-bit) → QNN context binary for HTP V68. Asset: `mqyov9dxm` (model.bin). 353M MACs, 92M params. | Removing `--preserve_io_datatype` allows I/O to be quantized to INT8 alongside internals — HTP accepts. Dummy calibration only (garbage accuracy); production needs real calibration data. `--quantize_full_type` and `--target_runtime qnn_context_binary` are deprecated — migrate to `submit_quantize_job` + `submit_compile_and_link_jobs`. |
<!-- NEXT ROWS: INT8 quantization experiments (Phase 2) -->

---

## Commands Reference (for quick re-runs)

### ✅ Working: local ONNX export
```bash
python deployment/scripts/lora_fp16/export.py --ckpt epoch=56-val_score=52.28.ckpt --output-dir exported_model
python deployment/scripts/onnx/export.py --model-dir exported_model --precision fp32
python deployment/scripts/onnx/to_fp16.py --input exported_model/vision_onnx --output exported_model/vision_onnx_fp16
```

### ✅ Working: INT8 with dummy calibration (compile-path verified — jgkr7qwn5)
```bash
qai-hub submit-compile-job \
    --model exported_model/vision_onnx/ \
    --device "Dragonwing RB3 Gen 2 Vision Kit" \
    --compile_options " --target_runtime qnn_context_binary --quantize_full_type int8" \
    --input_specs '{"image": ((1, 3, 256, 256), "float32")}' \
    --calibration_data none \
    --name "mSigLIP-vision-int8-dummy" \
    --wait
```
> **Note:** `--quantize_full_type` and `--target_runtime qnn_context_binary` are deprecated. Migrate to `submit_quantize_job` + `submit_compile_and_link_jobs` for future runs. The key insight is that `--preserve_io_datatype` must NOT be present in the quantizer/converter commands.

### 🚧 Fallback: target GPU instead of DSP
```bash
qai-hub submit-compile-job \
    --model exported_model/vision_onnx_fp16/ \
    --device "Dragonwing RB3 Gen 2 Vision Kit" \
    --compile_options " --target_runtime qnn_context_binary --compute_unit gpu" \
    --input_specs '{"image": ((1, 3, 256, 256), "float16")}' \
    --name "mSigLIP-vision-gpu-fp16" \
    --wait
```

### 🎯 Production target: INT8 with real calibration data
```bash
# 1. Prepare calibration dataset from VN3K (200-500 images + text pairs)
# 2. Upload to AI Hub: qai-hub upload-dataset ...
# 3. Compile with calibration:
qai-hub submit-compile-job \
    --model exported_model/vision_onnx/ \
    --device "Dragonwing RB3 Gen 2 Vision Kit" \
    --compile_options " --target_runtime qnn_context_binary --quantize_full_type int8" \
    --input_specs '{"image": ((1, 3, 256, 256), "float32")}' \
    --calibration_data <DATASET_ID> \
    --name "mSigLIP-vision-int8-prod" \
    --wait
```

---

## How to add a new row

When you (or Claude) invoke `qai-hub submit-compile-job`, `qai-hub submit-profile-job`, or `qai-hub submit-inference-job`:

1. Copy the job ID from the CLI output (format: `j…` like `jp27om9r5`).
2. Extract the **key flags that differ from previous attempts** (don't re-list unchanged boilerplate).
3. Copy the **first error line** from the failure, or `success` for a working compile.
4. Write a one-sentence **lesson** — what does this attempt teach us that the table doesn't already say?
5. Append a row to the table above.
6. If a new root-cause pattern emerges (e.g. a new class of error), also update the **Summary of Learnings** section at the top.
