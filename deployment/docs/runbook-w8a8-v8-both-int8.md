# Runbook — W8A8 v8 Both-INT8 (Vision + Text) trên RB3 Gen2 (HTP v68)

> Hướng dẫn **trọn vẹn từ đầu đến cuối**, theo thứ tự: LoRA merge → chuẩn bị data →
> learned rotation → QAT v8 → AI Hub W8A8 → gate fidelity → retrieval R@1 →
> compile/link `.bin` → chạy trên board → **retrieval INT8×INT8 trực tiếp trên board**,
> cho **cả vision và text**, rồi **both-INT8** end-to-end.
>
> Quy ước: mọi bước "local/free" chạy trên máy/cuda box; bước **AI Hub** và **board**
> tốn tài nguyên. Mọi job AI Hub PHẢI log vào `deployment/docs/journal/[deploy-master].md`.
>
> **Toàn bộ runbook chạy path v8 (learned rotation)** — đã thắng v6/v7 (vision T2I R@1 `50.85`). Không còn nhánh random.

---

## 0. Biến quy ước (đặt 1 lần)

| Biến | Ý nghĩa | Ví dụ |
|---|---|---|
| `VISION_CALIB_ID` | AI Hub dataset id calib **vision** (đã có) | `d7jzjy1m2` |
| `TEXT_CALIB_ID` | AI Hub dataset id calib **text** (tạo ở Part B) | `d7oz4gol9` |
| `VISION_QDQ` | thư mục QDQ vision v8 (learned rotation) | `artifacts/deployment/runtime/rotated_w8a8_learned_qat_v8/job_<ID>_qdq_onnx` |
| `TEXT_QDQ` | thư mục QDQ text v8 finite-mask | `artifacts/deployment/runtime/text_w8a8_learned_qat_v8_finite_mask/job_<ID>_qdq_onnx` |
| `BOARD` | host board | `qc-rb3g2` |

## Gates chấp nhận (áp cho cả vision và text)

| Gate | Ngưỡng | Khi nào |
|---|---:|---|
| Rotation FP32 invariance | cosine min ≥ `0.9999` | sau rotate/learn_rotation |
| Static ONNX vs PyTorch | cosine mean ≥ `0.999` | sau export ONNX |
| ONNX op sanity | `Pow=0`, fused `Gelu`/`LayerNormalization` | sau export ONNX |
| Text attention QDQ scale | max `< 10.0` | sau AI Hub quantize-only text |
| QDQ ONNX vs PyTorch | mean ≥ `0.95`, min ≥ `0.90` | sau AI Hub quantize-only |
| QNN board vs PyTorch | mean ≥ `0.90` | sau qnn-net-run |
| Retrieval (deploy target) | T2I R@1 ≥ `50.0` | both-INT8; bất kỳ kết quả `< 50` là FAIL |

> Cosine QDQ chỉ là proxy. **R@1 mới là số quyết định.**

---

# PART P — Prerequisites (LoRA merge + chuẩn bị data, local/free)

> Chạy 1 lần. Tạo ra `exported_model` (FP32 đã merge LoRA) + mọi thư mục input mà
> Part A/B/C cần. Sau bước này không cần đụng lại checkpoint training.

### P1. Merge LoRA → `exported_model`
```bash
python deployment/scripts/lora_fp16/export.py --ckpt artifacts/models/checkpoints/epoch=56-val_score=52.28.ckpt --output-dir artifacts/deployment/exports/exported_model
```
**Check**: `model_fp32.pt` không còn key `lora`/`adapter`/`base_layer`.

### P2. Vision inputs (`.raw` NCHW float32 `[-1,1]`)
```bash
python deployment/scripts/qnn/prepare_vn3k_vision_inputs.py --dataset-root VN3K --split test  --selection first  --num-samples 10   --output-dir artifacts/deployment/qnn_inputs/vn3k_test_10        --path-mode relative   # smoke fidelity
python deployment/scripts/qnn/prepare_vn3k_vision_inputs.py --dataset-root VN3K --split test  --selection first  --num-samples 100  --output-dir artifacts/deployment/qnn_inputs/vn3k_test_100       --path-mode relative   # QAT val
python deployment/scripts/qnn/prepare_vn3k_vision_inputs.py --dataset-root VN3K --split train --selection random --seed 2400 --num-samples 2000 --output-dir artifacts/deployment/qnn_inputs/vn3k_train_calib_2000 --path-mode relative   # learned-rotation calib + AI Hub calib source
python deployment/scripts/qnn/prepare_vn3k_vision_inputs.py --dataset-root VN3K --split train --selection random --seed 2400 --num-samples 4302 --output-dir artifacts/deployment/qnn_inputs/vn3k_train_all_4302 --path-mode relative   # QAT train
```
AI Hub vision calib dataset `d7jzjy1m2` (`msiglip-vision-vn3k-train-calib-2000`) đã upload sẵn — tái dùng. Nếu cần upload lại: `upload_qaihub_calibration_dataset.py --modality vision --input-dir .../vn3k_train_calib_2000`.

### P3. Text inputs (`.raw` dual-int `input_ids` + `attention_mask`)
```bash
python deployment/scripts/qnn/prepare_vn3k_text_inputs.py --split test  --num-samples 10   --selection first  --output-dir artifacts/deployment/qnn_inputs/vn3k_text_10          # smoke fidelity + board
python deployment/scripts/qnn/prepare_vn3k_text_inputs.py --split train --num-samples 500  --selection random --output-dir artifacts/deployment/qnn_inputs/vn3k_text_calib_500    # learned-rotation calib + AI Hub calib
python deployment/scripts/qnn/prepare_vn3k_text_inputs.py --split train --num-samples 4000 --selection random --output-dir artifacts/deployment/qnn_inputs/vn3k_text_train_4000   # QAT train
python deployment/scripts/qnn/prepare_vn3k_text_inputs.py --split test  --num-samples 100  --selection first  --output-dir artifacts/deployment/qnn_inputs/vn3k_text_test_100     # QAT val
```

---

# PART A — VISION v8 (learned rotation + QAT)

> Path v8 cố định: learned rotation → `VISION_BASE=exported_model_rotated_learned`.

### A1. Learned rotation (local/free) — bắt buộc cho v8
```bash
PYTHONUNBUFFERED=1 python deployment/scripts/qnn/learn_rotation.py --model-dir artifacts/deployment/exports/exported_model --output-dir artifacts/deployment/exports/exported_model_rotated_learned --calib-dir artifacts/deployment/qnn_inputs/vn3k_train_calib_2000 --gate-input-dir artifacts/deployment/qnn_inputs/vn3k_test_10 --num-calib 256 --tokens-per-image 32 --steps 3000 --lr 2e-3 --device cuda
```
**GATE**: in `GATE PASS` (cosine min ≥ 0.9999) thì mới lưu `model_fp32.pt`. → `VISION_BASE=exported_model_rotated_learned`.

### A2. QAT v8 (local/free, cuda) — recipe **v6** (const lr 1e-5, 15 ep)
> v7 (cosine + lr 2e-5) đã regress (`48.38` < v6 `49.30`), nên v8 dùng recipe v6 để ablation cô lập đúng phần rotation.
```bash
PYTHONUNBUFFERED=1 python deployment/scripts/qnn/train_vision_quant_robust.py --model-dir artifacts/deployment/exports/exported_model_rotated_learned --train-input-dir artifacts/deployment/qnn_inputs/vn3k_train_all_4302 --val-input-dir artifacts/deployment/qnn_inputs/vn3k_test_100 --output-dir artifacts/deployment/exports/exported_model_rotated_learned_qat_v8 --device cuda --batch-size 16 --epochs 15 --lr 1e-5 --fake-quant-observer ema --quant-head --quant-linears --quant-attention --start-layer 0 --end-layer 11 --num-workers 4
```

### A3. Export ONNX opset-20 (local/free)
```bash
python deployment/scripts/qnn/export_rotated_vision_onnx.py --model-dir artifacts/deployment/exports/exported_model_rotated_learned_qat_v8 --opset 20
```

### A4. GATE static ONNX vs PyTorch (local/free)
```bash
python3 deployment/scripts/qnn/compare_onnx_with_pytorch.py --onnx-model artifacts/deployment/exports/exported_model_rotated_learned_qat_v8/vision_onnx --model-dir artifacts/deployment/exports/exported_model_rotated_learned_qat_v8 --input-dir artifacts/deployment/qnn_inputs/vn3k_test_10 --precision fp32 --json artifacts/deployment/exports/exported_model_rotated_learned_qat_v8/static_vs_pytorch_summary.json --csv artifacts/deployment/exports/exported_model_rotated_learned_qat_v8/static_vs_pytorch.csv
```
Kỳ vọng cosine ~1.0. **Fail thì dừng** — lỗi export, đừng tốn job AI Hub.

### A5. AI Hub quantize-only → QDQ (TỐN JOB — log journal)
```bash
python3 deployment/scripts/qnn/submit_qaihub_quantize_compile.py --model artifacts/deployment/exports/exported_model_rotated_learned_qat_v8/vision_onnx --calibration-data d7jzjy1m2 --weights-dtype int8 --activations-dtype int8 --quantize-only --wait --download-quantized artifacts/deployment/runtime/rotated_w8a8_learned_qat_v8/job_qdq_onnx
```
Ghi lại `<JOB_ID>` từ output → `VISION_QDQ=artifacts/deployment/runtime/rotated_w8a8_learned_qat_v8/job_jp24xxn65_qdq_onnx`.

### A6. GATE QDQ fidelity (local/free)
```bash
python3 deployment/scripts/qnn/compare_onnx_with_pytorch.py --onnx-model $VISION_QDQ --model-dir artifacts/deployment/exports/exported_model --input-dir artifacts/deployment/qnn_inputs/vn3k_test_10 --precision fp32 --json artifacts/deployment/runtime/rotated_w8a8_learned_qat_v8/qdq_vs_pytorch_summary.json --csv artifacts/deployment/runtime/rotated_w8a8_learned_qat_v8/qdq_vs_pytorch.csv
```
Kỳ vọng mean ≥ 0.95, min ≥ 0.90.

### A7. GATE retrieval R@1 — vision-isolation (local/free, FULL set)
```bash
python3 deployment/scripts/qnn/eval_retrieval_quantized_vision.py --qdq-onnx $VISION_QDQ --model-dir artifacts/deployment/exports/exported_model --json artifacts/deployment/runtime/rotated_w8a8_learned_qat_v8/retrieval_r1.json
```
Baseline phải ~52.28; deploy target là `vision_int8` T2I R@1 ≥ 50 (kết quả < 50 là FAIL). Đây là số quyết định v8 thắng/thua v7.

### A8. Compile/link → `.bin` (TỐN JOB — chỉ khi A6+A7 pass — log journal)
```bash
python3 deployment/scripts/qnn/submit_qaihub_quantize_compile.py --model artifacts/deployment/exports/exported_model_rotated_learned_qat_v8/vision_onnx --calibration-data d7jzjy1m2 --weights-dtype int8 --activations-dtype int8 --wait --download artifacts/deployment/runtime/rotated_w8a8_learned_qat_v8/vision_encoder.bin
```

### A9. Board run + fidelity (TRÊN BOARD — xem §Board)
```bash
# trên board (đã push .bin + thư mục input, đã export QNN_LIB)

echo "$QNN_LIB"
find ~/ -name libQnnHtp.so 2>/dev/null | head

export QAIRT=/home/ubuntu/backup_qnn_20260323_095204/dfine_edge/qairt/2.44.0.260225
export QNN_BIN=$QAIRT/bin/aarch64-ubuntu-gcc9.4
export QNN_LIB=$QAIRT/lib/aarch64-ubuntu-gcc9.4
export LD_LIBRARY_PATH="$QNN_LIB:$LD_LIBRARY_PATH"

"$QNN_BIN/qnn-net-run" \
  --backend "$QNN_LIB/libQnnHtp.so" \
  --retrieve_context artifacts/deployment/bin/vision_encoder.bin \
  --config_file deployment/config/qnn/htp_config_245.json \
  --input_list artifacts/deployment/qnn_inputs/vn3k_test_10/input_list.txt \
  --output_dir artifacts/deployment/qnn_runs/rotated_w8a8_learned_qat_v8 \
  --profiling_level basic \
  --perf_profile high_performance

# về máy: board fidelity vs PyTorch
python3 deployment/scripts/qnn/compare_qnn_with_pytorch.py --qnn-output-dir artifacts/deployment/qnn_runs/rotated_w8a8_learned_qat_v8 --model-dir artifacts/deployment/exports/exported_model --input-dir artifacts/deployment/qnn_inputs/vn3k_test_10 --precision fp32 --json artifacts/deployment/qnn_runs/rotated_w8a8_learned_qat_v8/qnn_vs_pytorch_summary.json --csv artifacts/deployment/qnn_runs/rotated_w8a8_learned_qat_v8/qnn_vs_pytorch.csv
```
**GATE board**: mean ≥ 0.90. (Board fidelity ≈ QDQ fidelity là dấu hiệu QDQ proxy đúng.)

---

# PART B — TEXT (mirror v8 — learned rotation)

> Text inputs đã tạo ở **PART P3**. Part B độc lập với Part A (có thể chạy song song).

### B1. Learned rotation text (local/free, cuda) — v8 path
```bash
PYTHONUNBUFFERED=1 python deployment/scripts/qnn/learn_rotation_text.py --model-dir artifacts/deployment/exports/exported_model --output-dir artifacts/deployment/exports/exported_model_text_rotated_learned --calib-dir artifacts/deployment/qnn_inputs/vn3k_text_calib_500 --gate-input-dir artifacts/deployment/qnn_inputs/vn3k_text_10 --num-calib 256 --tokens-per-sample 32 --steps 3000 --lr 2e-3 --device cuda
```
**GATE** cosine min ≥ 0.9999 → `TEXT_BASE=exported_model_text_rotated_learned`.

### B2. Text QAT (local/free, cuda) — `--modality text`, recipe **v6** (const lr 1e-5, 15 ep)
> Dùng recipe v6 (giống vision v8). Base là learned rotation từ B1.
```bash
PYTHONUNBUFFERED=1 python deployment/scripts/qnn/train_vision_quant_robust.py --modality text --model-dir artifacts/deployment/exports/exported_model_text_rotated_learned --train-input-dir artifacts/deployment/qnn_inputs/vn3k_text_train_4000 --val-input-dir artifacts/deployment/qnn_inputs/vn3k_text_test_100 --output-dir artifacts/deployment/exports/exported_model_text_rotated_learned_qat_v8 --device cuda --batch-size 16 --epochs 15 --lr 1e-5 --fake-quant-observer ema --quant-head --quant-linears --quant-attention --start-layer 0 --end-layer 11 --num-workers 4
```

### B3. Export text ONNX opset-20 (local/free)
```bash
python deployment/scripts/qnn/export_text_onnx.py --model-dir artifacts/deployment/exports/exported_model_text_rotated_learned_qat_v8
```

### B4. Patch finite attention mask (local/free)
```bash
python3 deployment/scripts/qnn/patch_text_finite_attention_mask.py --model artifacts/deployment/exports/exported_model_text_rotated_learned_qat_v8/text_onnx --output-dir artifacts/deployment/exports/exported_model_text_rotated_learned_qat_v8/text_onnx_finite_mask --mask-value -32.0 --check --smoke-load
```
Patch này chỉ đổi attention mask constant âm cực lớn (`-3.402823e38`) trên text self-attention path thành `-32.0`, để AI Hub không tạo scale INT8 ~`1e32` tại `scores+mask`. Summary ghi ở `text_onnx_finite_mask/finite_mask_patch_summary.json`; kỳ vọng đổi đúng 1 Constant và thấy 12 Softmax text attention.

### B5. GATE static text ONNX finite-mask vs PyTorch (local/free)
```bash
python3 deployment/scripts/qnn/compare_text_onnx_with_pytorch.py --onnx-model artifacts/deployment/exports/exported_model_text_rotated_learned_qat_v8/text_onnx_finite_mask --model-dir artifacts/deployment/exports/exported_model_text_rotated_learned_qat_v8 --input-dir artifacts/deployment/qnn_inputs/vn3k_text_10 --json artifacts/deployment/exports/exported_model_text_rotated_learned_qat_v8/text_onnx_finite_mask/static_vs_pytorch_summary.json --csv artifacts/deployment/exports/exported_model_text_rotated_learned_qat_v8/text_onnx_finite_mask/static_vs_pytorch.csv
```
Kỳ vọng cosine mean ≥ `0.999`, min ≥ `0.9999`, không NaN, `Pow=0`, fused `Gelu`/`LayerNormalization`. Nếu fail thì dừng trước AI Hub.

### B6. Upload calib text → AI Hub (TỐN JOB nhẹ — log journal; bỏ qua nếu đã có `d7oz4gol9`)
```bash
python deployment/scripts/qnn/upload_qaihub_calibration_dataset.py --modality text --input-dir artifacts/deployment/qnn_inputs/vn3k_text_calib_500 --name msiglip-text-vn3k-calib-500
```
Ghi `Dataset ID` in ra → `TEXT_CALIB_ID`.

### B7. AI Hub quantize-only text finite-mask → QDQ (TỐN JOB — log journal) — `--modality text`
```bash
python3 deployment/scripts/qnn/submit_qaihub_quantize_compile.py --modality text --model artifacts/deployment/exports/exported_model_text_rotated_learned_qat_v8/text_onnx_finite_mask --calibration-data d7oz4gol9 --weights-dtype int8 --activations-dtype int8 --quantize-only --wait --download-quantized artifacts/deployment/runtime/text_w8a8_learned_qat_v8_finite_mask/job_qdq_onnx
```
Ghi `<JOB_ID>` → `TEXT_QDQ=artifacts/deployment/runtime/text_w8a8_learned_qat_v8_finite_mask/job_jp17y648p_qdq_onnx`.
> `--modality text` giữ int I/O và BỎ `--quantize_io` (token id tới ~250k, không int8 hóa được).

### B8. GATE text attention QDQ scale (local/free)
```bash
python3 deployment/scripts/qnn/inspect_text_attention_qdq.py --model $TEXT_QDQ --json artifacts/deployment/runtime/text_w8a8_learned_qat_v8_finite_mask/attention_qdq_scales.json --fail-scale-ge 10.0
```
Kỳ vọng tìm đúng 12 QDQ pair trước Softmax và `max_add_output_scale < 10.0`. Nếu fail, dừng: finite-mask chưa có tác dụng hoặc AI Hub vẫn quantize sai vị trí.

### B9. GATE QDQ fidelity text (local/free)
```bash
python3 deployment/scripts/qnn/compare_text_onnx_with_pytorch.py --onnx-model $TEXT_QDQ --model-dir artifacts/deployment/exports/exported_model_text_rotated_learned_qat_v8 --input-dir artifacts/deployment/qnn_inputs/vn3k_text_10 --json artifacts/deployment/runtime/text_w8a8_learned_qat_v8_finite_mask/text_qdq_fid.json --csv artifacts/deployment/runtime/text_w8a8_learned_qat_v8_finite_mask/text_qdq_fid.csv
```
Kỳ vọng mean ≥ `0.95`, min ≥ `0.90`. Nếu vẫn collapse, dừng trước compile/link và inspect top activation scales.

### B10. GATE retrieval — text-isolation (local/free, FULL set)
```bash
python3 deployment/scripts/qnn/eval_retrieval_quantized_vision.py --skip-vision-qdq --text-qdq-onnx $TEXT_QDQ --model-dir artifacts/deployment/exports/exported_model --json artifacts/deployment/runtime/text_w8a8_learned_qat_v8_finite_mask/text_isolation_r1.json
```
Combo `text_int8` (image FP32 + text QDQ) cho thấy riêng text-quant rớt bao nhiêu. Chỉ compile/link nếu T2I R@1 ≥ `50.0`.

### B11. Compile/link text finite-mask → `.bin` (TỐN JOB — chỉ khi B8+B9+B10 pass — log journal) — `--modality text`
```bash
python3 deployment/scripts/qnn/submit_qaihub_quantize_compile.py \
  --modality text \
  --model artifacts/deployment/exports/exported_model_text_rotated_learned_qat_v8/text_onnx_finite_mask \
  --calibration-data d7oz4gol9 \
  --weights-dtype int8 \
  --activations-dtype int8 \
  --compile-options="--truncate_64bit_io --quantize_io" \
  --wait \
  --download artifacts/deployment/runtime/text_w8a8_learned_qat_v8_finite_mask/text_encoder.bin
```

### B12. Board run + fidelity text (TRÊN BOARD)
```bash
qnn-net-run --backend "$QNN_LIB/libQnnHtp.so" --retrieve_context artifacts/deployment/runtime/text_w8a8_learned_qat_v8_finite_mask/text_encoder.bin --config_file deployment/config/qnn/htp_config_245.json --input_list artifacts/deployment/qnn_inputs/vn3k_text_10/input_list.txt --output_dir artifacts/deployment/qnn_runs/text_w8a8_learned_qat_v8_finite_mask --profiling_level basic --perf_profile high_performance

python3 deployment/scripts/qnn/compare_qnn_with_pytorch.py --qnn-output-dir artifacts/deployment/qnn_runs/text_w8a8_learned_qat_v8_finite_mask --model-dir artifacts/deployment/exports/exported_model --input-dir artifacts/deployment/qnn_inputs/vn3k_text_10 --precision fp32 --json artifacts/deployment/qnn_runs/text_w8a8_learned_qat_v8_finite_mask/qnn_vs_pytorch_summary.json --csv artifacts/deployment/qnn_runs/text_w8a8_learned_qat_v8_finite_mask/qnn_vs_pytorch.csv
```
> Lưu ý: text input là 2 int (`input_ids`, `attention_mask`); `input_list.txt` của text dùng dạng `input_ids:=... attention_mask:=...`.

---

# PART C — BOTH-INT8 (số deploy cuối cùng)

### C1. Off-board both-INT8 R@1 (local/free, FULL set) — **số chính của luận văn**
```bash
python3 deployment/scripts/qnn/eval_retrieval_quantized_vision.py --qdq-onnx $VISION_QDQ --text-qdq-onnx $TEXT_QDQ --model-dir artifacts/deployment/exports/exported_model --json artifacts/deployment/runtime/both_int8/both_int8_r1.json
```
In 4 combo: `baseline_fp32` (~52.28), `vision_int8`, `text_int8`, **`both_int8`** = số deploy thật. Deploy target T2I R@1 ≥ 50 (kết quả < 50 là FAIL).

**Kết quả hiện tại (C1, QDQ/off-board, full VN3K test):**

| Combo | T2I R@1 | I2T R@1 | Ghi chú |
|---|---:|---:|---|
| `baseline_fp32` | `52.40` | `55.30` | local sanity reproduction; baseline báo cáo chính vẫn là paper `52.28` |
| `vision_int8` | `50.85` | `52.90` | vision v8 learned rotation |
| `text_int8` | `51.65` | `55.55` | text v8 learned rotation + finite mask |
| **`both_int8`** | **`50.25`** | **`52.95`** | **PASS** |

Kết luận C1: **both-INT8 T2I R@1 `50.25`**, giảm `-2.03` so với paper baseline `52.28`, vẫn vượt deploy target `50.0` với margin `+0.25`. C2 board retrieval vẫn là bước xác nhận tiếp theo.

### C2. Board both-INT8 — retrieval INT8×INT8 TRỰC TIẾP trên thiết bị

> Đây là số "thuần board": cả gallery image embedding và query text embedding đều sinh
> từ `.bin` chạy trên HTP v68, rồi tính R@1. Yêu cầu A8 (vision `.bin`) + B11 (text `.bin`) đã có.

**C2.1. Chuẩn bị FULL test set `.raw` (local/free)** — gallery 2000 ảnh, query 4000 caption.
```bash
python3 deployment/scripts/qnn/prepare_vn3k_vision_inputs.py --dataset-root VN3K --split test --selection first --num-samples 2000 --output-dir artifacts/deployment/qnn_inputs/vn3k_test_gallery_2000 --path-mode relative
python3 deployment/scripts/qnn/prepare_vn3k_text_inputs.py --split test --num-samples 4000 --selection first --output-dir artifacts/deployment/qnn_inputs/vn3k_test_query_4000
```

**C2.2. Board: vision `.bin` chạy FULL gallery → image embeddings** (trong thư mục input trên board).
```bash
qnn-net-run --backend "$QNN_LIB/libQnnHtp.so" --retrieve_context vision_encoder.bin --config_file deployment/config/qnn/htp_config_245.json --input_list vn3k_test_gallery_2000/input_list.txt --output_dir qnn_runs/both_int8_vision --profiling_level basic --perf_profile high_performance
```

**C2.3. Board: text `.bin` chạy FULL query → text embeddings.**
```bash
qnn-net-run --backend "$QNN_LIB/libQnnHtp.so" --retrieve_context text_encoder.bin --config_file deployment/config/qnn/htp_config_245.json --input_list vn3k_test_query_4000/input_list.txt --output_dir qnn_runs/both_int8_text --profiling_level basic --perf_profile high_performance
```

**C2.4. Kéo embedding board về host** (`adb pull`/`scp` 2 thư mục `qnn_runs/both_int8_{vision,text}` chứa `Output_*.raw` đã dequantize sang float).

**C2.5. Tính R@1 từ embedding board** (local/free) — raw dot product, đúng `LitTBPS._compute_metrics`.
```bash
python3 deployment/scripts/qnn/eval_retrieval_board_embeddings.py --vision-output-dir artifacts/deployment/qnn_runs/both_int8_vision --text-output-dir artifacts/deployment/qnn_runs/both_int8_text --gallery-input-dir artifacts/deployment/qnn_inputs/vn3k_test_gallery_2000 --query-input-dir artifacts/deployment/qnn_inputs/vn3k_test_query_4000 --json artifacts/deployment/qnn_runs/both_int8_board_r1.json
```
> **Trạng thái script:** `eval_retrieval_board_embeddings.py` là helper *cần viết* (đọc `Output_*.raw` board của cả 2 encoder, map pid theo thứ tự `input_list`, tính R@1 raw dot product — port từ `eval_retrieval_quantized_vision.py`, thay phần inference ONNX bằng load `.raw`). Cho tới khi có, **số both-INT8 ở C1 (QDQ) là proxy đã verify board≈QDQ** (A9/B12 cho thấy lệch ≈ `0.0001`).

Deploy target áp cho both-INT8 board: **T2I R@1 ≥ 50** (kết quả < 50 là FAIL).

---

# Board — prerequisites & gotchas

**Chuẩn bị (1 lần mỗi phiên trên board `qc-rb3g2`):**
1. `adb push` (hoặc scp) lên board: file `.bin`, thư mục input kèm `input_list.txt` + `raw/`. Smoke fidelity: `vn3k_test_10` / `vn3k_text_10`. Both-INT8 board (C2): `vn3k_test_gallery_2000` / `vn3k_test_query_4000`.
2. `export QNN_LIB=<đường dẫn QNN libs trên board>` (chứa `libQnnHtp.so`). Nếu chưa set → lỗi `--backend` không tìm thấy.
3. Chạy `qnn-net-run` **từ trong thư mục input** (hoặc dùng path tuyệt đối) vì `raw/` trong `input_list.txt` là path tương đối.

**Gotchas đã gặp (đừng lặp):**
- `--config_file` phải trỏ tới **file** `deployment/config/qnn/htp_config_245.json`, KHÔNG phải thư mục.
- `$QNN_LIB` chưa export → backend fail; export inline trước lệnh.
- Path `raw/` tương đối fail nếu chạy sai cwd → `cd` vào thư mục input rồi `--input_list input_list.txt`.
- Log profiling: run tăng hậu tố; lấy file **mới nhất** (`qnn-profiling-data_1.log`), tránh log của run fail cũ (`_0.log`).

---

# Logging bắt buộc

- **Mỗi job AI Hub** (A5, A8, B6, B7, B11): append vào `deployment/docs/journal/[deploy-master].md` — mục tiêu, job id, input, output/error, fidelity/R@1, quyết định.
- **Số v8 vision** (A7) và **both-INT8** (C1 off-board, C2 board): điền vào bảng `[deploy-master].md` §11 + §6.
- Board fidelity/latency (A9, B12) và board both-INT8 R@1 (C2): vào `[deploy-master].md`.
