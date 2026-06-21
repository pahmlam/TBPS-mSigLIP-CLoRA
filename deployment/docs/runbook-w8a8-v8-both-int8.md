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
| `TEXT_CALIB_ID` | AI Hub dataset id calib **text f32-mask** (tạo ở Part B) | `d7mv1vkv7` |
| `VISION_QDQ` | thư mục QDQ vision v8 (learned rotation) | `artifacts/deployment/runtime/rotated_w8a8_learned_qat_v8/job_<ID>_qdq_onnx` |
| `TEXT_QDQ` | thư mục QDQ text v8 f32-mask finite-mask | `artifacts/deployment/runtime/text_w8a8_learned_qat_v8_f32mask/job_<ID>_qdq_onnx` |
| `BOARD` | host board | `qc-rb3g2` |

new text data ID: `d9vpnzz09` - vn3k_text_calib_500_split_embeds
new text data ID: `d9pg6dpd9` - 2000 TEXT CALIB

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

### P3. Text inputs

Integer `attention_mask` dùng cho learned rotation/QAT local. F32-mask input dùng cho ONNX/QDQ text, vì QNN HTP link reject tensor float nội bộ sinh từ `attention_mask int -> Cast(FLOAT)`. Riêng QNN board input phải dùng `input_ids=int32`, vì compile dùng `--truncate_64bit_io`: graph ONNX/QDQ vẫn thấy `input_ids=int64`, nhưng context binary trên board nhận 64 token x 4 bytes = 256 bytes.

```bash
python deployment/scripts/qnn/prepare_vn3k_text_inputs.py --split test  --num-samples 10   --selection first  --output-dir artifacts/deployment/qnn_inputs/vn3k_text_10          # smoke fidelity + board
python deployment/scripts/qnn/prepare_vn3k_text_inputs.py --split train --num-samples 500  --selection random --output-dir artifacts/deployment/qnn_inputs/vn3k_text_calib_500    # learned-rotation calib
python deployment/scripts/qnn/prepare_vn3k_text_inputs.py --split train --num-samples 4000 --selection random --output-dir artifacts/deployment/qnn_inputs/vn3k_text_train_4000   # QAT train
python deployment/scripts/qnn/prepare_vn3k_text_inputs.py --split test  --num-samples 100  --selection first  --output-dir artifacts/deployment/qnn_inputs/vn3k_text_test_100     # QAT val

python deployment/scripts/qnn/prepare_vn3k_text_inputs.py --split test  --num-samples 10   --selection first  --output-dir artifacts/deployment/qnn_inputs/vn3k_text_10_f32mask       --mask-dtype float32  # ONNX/QDQ text smoke fidelity
python deployment/scripts/qnn/prepare_vn3k_text_inputs.py --split test  --num-samples 10   --selection first  --output-dir artifacts/deployment/qnn_inputs/vn3k_text_10_f32mask_i32   --id-dtype int32 --mask-dtype float32  # QNN board text smoke
python deployment/scripts/qnn/prepare_vn3k_text_inputs.py --split train --num-samples 500  --selection random --output-dir artifacts/deployment/qnn_inputs/vn3k_text_calib_500_f32mask --mask-dtype float32  # AI Hub text calib source
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

### A8. Compile/link → `.bin` (TỐN JOB — chỉ khi A6+A7 pass — log journal)
```bash
python3 deployment/scripts/qnn/submit_qaihub_quantize_compile.py --model artifacts/deployment/exports/exported_model_rotated_learned_qat_v8/vision_onnx --calibration-data d7jzjy1m2 --weights-dtype int8 --activations-dtype int8 --wait --download artifacts/deployment/runtime/rotated_w8a8_learned_qat_v8/vision_encoder.bin
```

### A9. Board run + fidelity (TRÊN BOARD — xem §Board)
```bash
# trên board (đã push .bin + thư mục input, đã export QNN_LIB)

ls /opt/qcom/qairt/2.45.40.260406/lib/aarch64-ubuntu-gcc9.4/libQnnHtp.so
ls /opt/qcom/qairt/2.45.40.260406/lib/aarch64-ubuntu-gcc9.4/libQnnHtpNetRunExtensions.so

export QAIRT=/opt/qcom/qairt/2.45.40.260406
export QNN_BIN=$QAIRT/bin/aarch64-ubuntu-gcc9.4
export QNN_LIB=$QAIRT/lib/aarch64-ubuntu-gcc9.4
export LD_LIBRARY_PATH="$QNN_LIB:$LD_LIBRARY_PATH"
export ADSP_LIBRARY_PATH="$QAIRT/lib/hexagon-v68/unsigned;$QAIRT/lib/hexagon-v68;/usr/lib/rfsa/adsp;/dsp"

cd artifacts/deployment/qnn_inputs/vn3k_test_10

"$QNN_BIN/qnn-net-run" \
  --backend "$QNN_LIB/libQnnHtp.so" \
  --retrieve_context /home/ubuntu/sigm/Lam/artifacts/deployment/bin/vision_encoder.bin \
  --config_file /home/ubuntu/sigm/Lam/deployment/config/qnn/htp_config_245.json \
  --input_list input_list.txt \
  --output_dir /home/ubuntu/sigm/Lam/artifacts/deployment/qnn_runs/rotated_w8a8_learned_qat_v8 \
  --profiling_level basic \
  --perf_profile high_performance

"$QNN_BIN/qnn-profile-viewer" \
  --input_log artifacts/deployment/qnn_runs/rotated_w8a8_learned_qat_v8/qnn-profiling-data_4.log \
  > artifacts/deployment/qnn_runs/rotated_w8a8_learned_qat_v8/profile.txt

# về máy: board fidelity vs PyTorch
python3 deployment/scripts/qnn/compare_qnn_with_pytorch.py --qnn-output-dir artifacts/deployment/qnn_runs/rotated_w8a8_learned_qat_v8 --model-dir artifacts/deployment/exports/exported_model --input-dir artifacts/deployment/qnn_inputs/vn3k_test_10 --precision fp32 --json artifacts/deployment/qnn_runs/rotated_w8a8_learned_qat_v8/qnn_vs_pytorch_summary.json --csv artifacts/deployment/qnn_runs/rotated_w8a8_learned_qat_v8/qnn_vs_pytorch.csv
```

### A10. Board retrieval R@1 — vision-isolation FULL gallery (TRÊN BOARD + local)

Chạy full 2000 ảnh gallery trên board, kéo output về host, rồi tính retrieval với text FP32 để xác nhận proxy `vision_int8`.

```bash
# trên board
cd /home/ubuntu/sigm/Lam/artifacts/deployment/qnn_inputs/vn3k_test_gallery_2000

"$QNN_BIN/qnn-net-run" \
  --backend "$QNN_LIB/libQnnHtp.so" \
  --retrieve_context /home/ubuntu/sigm/Lam/artifacts/deployment/bin/vision_encoder.bin \
  --config_file /home/ubuntu/sigm/Lam/deployment/config/qnn/htp_config_245.json \
  --input_list input_list.txt \
  --output_dir /home/ubuntu/sigm/Lam/artifacts/deployment/qnn_runs/rotated_w8a8_learned_qat_v8_gallery_2000 \
  --profiling_level basic \
  --perf_profile high_performance

# trên host sau khi rsync/scp output về
python3 deployment/scripts/qnn/eval_retrieval_board_vision.py \
  --vision-output-dir artifacts/deployment/qnn_runs/rotated_w8a8_learned_qat_v8_gallery_2000 \
  --gallery-input-dir artifacts/deployment/qnn_inputs/vn3k_test_gallery_2000 \
  --model-dir artifacts/deployment/exports/exported_model \
  --dataset-root . \
  --json artifacts/deployment/qnn_runs/rotated_w8a8_learned_qat_v8_gallery_2000/board_vision_r1.json
```

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
python deployment/scripts/qnn/export_text_onnx.py \
  --model-dir artifacts/deployment/exports/exported_model_text_rotated_learned_qat_v8 \
  --output-subdir text_onnx_f32mask \
  --attention-mask-dtype float32
```

### B4. Patch finite attention mask (local/free)
```bash
python3 deployment/scripts/qnn/patch_text_finite_attention_mask.py \
  --model artifacts/deployment/exports/exported_model_text_rotated_learned_qat_v8/text_onnx_f32mask \
  --output-dir artifacts/deployment/exports/exported_model_text_rotated_learned_qat_v8/text_onnx_f32mask_finite \
  --mask-value -32.0 \
  --check \
  --smoke-load
```
Patch này chỉ đổi attention mask constant âm cực lớn (`-3.402823e38`) trên text self-attention path thành `-32.0`, để AI Hub không tạo scale INT8 ~`1e32` tại `scores+mask`. Summary ghi ở `text_onnx_f32mask_finite/finite_mask_patch_summary.json`; kỳ vọng đổi đúng 1 Constant và thấy 12 Softmax text attention.

### B4b. Patch QNN link-safe mask subgraph (local/free)
```bash
python3 deployment/scripts/qnn/patch_text_qnn_link_safe_mask.py \
  --model artifacts/deployment/exports/exported_model_text_rotated_learned_qat_v8/text_onnx_f32mask_finite \
  --output-dir artifacts/deployment/exports/exported_model_text_rotated_learned_qat_v8/text_onnx_f32mask_finite_linksafe \
  --check \
  --smoke-load
```
Patch này đổi mask subgraph tương đương toán học từ `Where(1-mask != 0, -32, 0)` sang `(1-mask)*(-32)`, đồng thời loại `/text_model/Cast_output_0`. Summary ghi ở `text_onnx_f32mask_finite_linksafe/qnn_link_safe_mask_patch_summary.json`.

### B5. GATE static text ONNX finite/f32/link-safe vs PyTorch (local/free)
```bash
python3 deployment/scripts/qnn/compare_text_onnx_with_pytorch.py \
  --onnx-model artifacts/deployment/exports/exported_model_text_rotated_learned_qat_v8/text_onnx_f32mask_finite_linksafe \
  --model-dir artifacts/deployment/exports/exported_model_text_rotated_learned_qat_v8 \
  --input-dir artifacts/deployment/qnn_inputs/vn3k_text_10_f32mask \
  --json artifacts/deployment/exports/exported_model_text_rotated_learned_qat_v8/text_onnx_f32mask_finite_linksafe/static_vs_pytorch_summary.json \
  --csv artifacts/deployment/exports/exported_model_text_rotated_learned_qat_v8/text_onnx_f32mask_finite_linksafe/static_vs_pytorch.csv
```
Kỳ vọng cosine mean ≥ `0.999`, min ≥ `0.9999`, không NaN, `Pow=0`, fused `Gelu`/`LayerNormalization`. Nếu fail thì dừng trước AI Hub.

### B6. Upload calib text f32-mask → AI Hub (TỐN JOB nhẹ — log journal)
```bash
python deployment/scripts/qnn/upload_qaihub_calibration_dataset.py \
  --modality text \
  --input-dir artifacts/deployment/qnn_inputs/vn3k_text_calib_500_f32mask \
  --id-dtype int64 \
  --mask-dtype float32 \
  --name msiglip-text-vn3k-calib-500-f32mask
```
Ghi `Dataset ID` in ra → `TEXT_CALIB_ID`. Không dùng lại `d7oz4gol9`, vì dataset đó có `attention_mask` integer và không khớp ONNX f32-mask.

### B7. AI Hub quantize-only text finite/f32/link-safe → QDQ (TỐN JOB — log journal) — `--modality text`
```bash
python3 deployment/scripts/qnn/submit_qaihub_quantize_compile.py \
  --modality text \
  --text-attention-mask-dtype float32 \
  --model artifacts/deployment/exports/exported_model_text_rotated_learned_qat_v8/text_onnx_f32mask_finite_linksafe \
  --calibration-data $TEXT_CALIB_ID \
  --weights-dtype int8 \
  --activations-dtype int8 \
  --quantize-only \
  --wait \
  --download-quantized artifacts/deployment/runtime/text_w8a8_learned_qat_v8_f32mask/job_qdq_onnx
```
Ghi `<JOB_ID>` → `TEXT_QDQ=artifacts/deployment/runtime/text_w8a8_learned_qat_v8_f32mask/job_<ID>_qdq_onnx`.
> `input_ids` vẫn là token index integer; riêng `attention_mask` là float32 0/1 và mask subgraph đã rewrite linksafe để tránh node nội bộ `Cast(FLOAT)` làm QNN linker fail. W8A8 vẫn giữ `--weights-dtype int8 --activations-dtype int8`.

### B8. GATE text attention QDQ scale (local/free)
```bash
python3 deployment/scripts/qnn/inspect_text_attention_qdq.py \
  --model $TEXT_QDQ \
  --json artifacts/deployment/runtime/text_w8a8_learned_qat_v8_f32mask/attention_qdq_scales.json \
  --fail-scale-ge 10.0
```
Kỳ vọng tìm đúng 12 QDQ pair trước Softmax và `max_add_output_scale < 10.0`. Nếu fail, dừng: finite-mask chưa có tác dụng hoặc AI Hub vẫn quantize sai vị trí.

### B9. GATE QDQ fidelity text (local/free)
```bash
python3 deployment/scripts/qnn/compare_text_onnx_with_pytorch.py \
  --onnx-model $TEXT_QDQ \
  --model-dir artifacts/deployment/exports/exported_model_text_rotated_learned_qat_v8 \
  --input-dir artifacts/deployment/qnn_inputs/vn3k_text_10_f32mask \
  --json artifacts/deployment/runtime/text_w8a8_learned_qat_v8_f32mask/text_qdq_fid.json \
  --csv artifacts/deployment/runtime/text_w8a8_learned_qat_v8_f32mask/text_qdq_fid.csv
```
Kỳ vọng mean ≥ `0.95`, min ≥ `0.90`. Nếu vẫn collapse, dừng trước compile/link và inspect top activation scales.

### B10. GATE retrieval — text-isolation (local/free, FULL set)
```bash
python3 deployment/scripts/qnn/eval_retrieval_quantized_vision.py \
  --skip-vision-qdq \
  --text-qdq-onnx $TEXT_QDQ \
  --model-dir artifacts/deployment/exports/exported_model \
  --json artifacts/deployment/runtime/text_w8a8_learned_qat_v8_f32mask/text_isolation_r1.json
```
Combo `text_int8` (image FP32 + text QDQ) cho thấy riêng text-quant rớt bao nhiêu. Chỉ compile/link nếu T2I R@1 ≥ `50.0`.

### B11. Compile/link text finite/f32/link-safe → `.bin` (TỐN JOB — chỉ khi B8+B9+B10 pass — log journal) — `--modality text`
```bash
python3 deployment/scripts/qnn/submit_qaihub_quantize_compile.py \
  --modality text \
  --text-attention-mask-dtype float32 \
  --model artifacts/deployment/exports/exported_model_text_rotated_learned_qat_v8/text_onnx_f32mask_finite_linksafe \
  --calibration-data d7ozgzkq9 \
  --weights-dtype int8 \
  --activations-dtype int8 \
  --wait \
  --download artifacts/deployment/runtime/text_w8a8_learned_qat_v8_f32mask/text_encoder.bin
```

### B12. Board run + fidelity text (TRÊN BOARD)
```bash

ls /opt/qcom/qairt/2.45.40.260406/lib/aarch64-ubuntu-gcc9.4/libQnnHtp.so
ls /opt/qcom/qairt/2.45.40.260406/lib/aarch64-ubuntu-gcc9.4/libQnnHtpNetRunExtensions.so

export QAIRT=/opt/qcom/qairt/2.45.40.260406
export QNN_BIN=$QAIRT/bin/aarch64-ubuntu-gcc9.4
export QNN_LIB=$QAIRT/lib/aarch64-ubuntu-gcc9.4
export LD_LIBRARY_PATH="$QNN_LIB:$LD_LIBRARY_PATH"
export ADSP_LIBRARY_PATH="$QAIRT/lib/hexagon-v68/unsigned;$QAIRT/lib/hexagon-v68;/usr/lib/rfsa/adsp;/dsp"

cd /home/ubuntu/sigm/Lam/artifacts/deployment/qnn_inputs/vn3k_text_10_f32mask_i32

wc -c raw/00000_pid2000_02001_1_input_ids.raw raw/00000_pid2000_02001_1_attention_mask.raw
# Expected: input_ids 256 bytes (int32), attention_mask 256 bytes (float32)

"$QNN_BIN/qnn-net-run" \
  --backend "$QNN_LIB/libQnnHtp.so" \
  --retrieve_context /home/ubuntu/sigm/Lam/artifacts/deployment/bin/text_encoder.bin \
  --config_file /home/ubuntu/sigm/Lam/deployment/config/qnn/htp_config_245.json \
  --input_list input_list.txt \
  --output_dir /home/ubuntu/sigm/Lam/artifacts/deployment/qnn_runs/text_w8a8_learned_qat_v8_f32mask \
  --profiling_level basic \
  --perf_profile high_performance

```
> Lưu ý: text input cho QNN f32-mask là `input_ids` **int32** + `attention_mask` float32; `input_list.txt` vẫn dùng dạng `input_ids:=... attention_mask:=...`. Nếu dùng input_ids int64, qnn-net-run báo file size 512 bytes nhưng graph chỉ expect 256 bytes. Đừng dùng `compare_qnn_with_pytorch.py` cho B12 vì script đó hiện chỉ compare vision/image. Nếu cần fidelity text board, viết/ dùng helper text riêng đọc dual-input raw và gọi `encode_text`.

### B13. GATE board text fidelity (host/local sau khi kéo output về)
```bash
python3 deployment/scripts/qnn/compare_text_qnn_with_pytorch.py \
  --qnn-output-dir artifacts/deployment/qnn_runs/text_w8a8_learned_qat_v8_f32mask \
  --input-dir artifacts/deployment/qnn_inputs/vn3k_text_10_f32mask_i32 \
  --model-dir artifacts/deployment/exports/exported_model_text_rotated_learned_qat_v8 \
  --id-dtype int32 \
  --mask-dtype float32 \
  --json artifacts/deployment/qnn_runs/text_w8a8_learned_qat_v8_f32mask/qnn_vs_pytorch_summary.json \
  --csv artifacts/deployment/qnn_runs/text_w8a8_learned_qat_v8_f32mask/qnn_vs_pytorch.csv
```
**GATE**: mean >= `0.90`, không NaN/Inf. Nếu fail ở đây thì debug text `.bin`/input dtype/runtime trước khi chạy full 4000 query.

### B14. Board text retrieval R@1 — text-isolation FULL query (TRÊN BOARD + local)

Chạy full 4000 query text trên board bằng input `input_ids=int32`, kéo output về host, rồi tính retrieval với image FP32 để xác nhận proxy `text_int8`.

```bash
# local: chuẩn bị full query input cho QNN board
python3 deployment/scripts/qnn/prepare_vn3k_text_inputs.py \
  --split test \
  --num-samples 4000 \
  --selection first \
  --output-dir artifacts/deployment/qnn_inputs/vn3k_test_query_4000_f32mask_i32 \
  --id-dtype int32 \
  --mask-dtype float32

# board: cd vào input dir rồi chạy text .bin
cd /home/ubuntu/sigm/Lam/artifacts/deployment/qnn_inputs/vn3k_test_query_4000_f32mask_i32

"$QNN_BIN/qnn-net-run" \
  --backend "$QNN_LIB/libQnnHtp.so" \
  --retrieve_context /home/ubuntu/sigm/Lam/artifacts/deployment/bin/text_encoder.bin \
  --config_file /home/ubuntu/sigm/Lam/deployment/config/qnn/htp_config_245.json \
  --input_list input_list.txt \
  --output_dir /home/ubuntu/sigm/Lam/artifacts/deployment/qnn_runs/text_w8a8_learned_qat_v8_query_4000_f32mask_i32 \
  --profiling_level basic \
  --perf_profile high_performance

# host: sau khi rsync/scp output về
python3 deployment/scripts/qnn/eval_retrieval_board_text.py \
  --text-output-dir artifacts/deployment/qnn_runs/text_w8a8_learned_qat_v8_query_4000_f32mask_i32 \
  --query-input-dir artifacts/deployment/qnn_inputs/vn3k_test_query_4000_f32mask_i32 \
  --model-dir artifacts/deployment/exports/exported_model \
  --dataset-root . \
  --json artifacts/deployment/qnn_runs/text_w8a8_learned_qat_v8_query_4000_f32mask_i32/board_text_r1.json
```
**GATE**: T2I R@1 >= `50.0` cho text-isolation board run. Nếu pass, tiếp tục C2 both-INT8 board retrieval.

---

# PART C — BOTH-INT8 (số deploy cuối cùng)

### C1. Off-board both-INT8 R@1 (local/free, FULL set)
```bash
python3 deployment/scripts/qnn/eval_retrieval_quantized_vision.py --qdq-onnx $VISION_QDQ --text-qdq-onnx $TEXT_QDQ --model-dir artifacts/deployment/exports/exported_model --json artifacts/deployment/runtime/both_int8/both_int8_r1.json
```

### C2. Board both-INT8 — retrieval INT8×INT8 TRỰC TIẾP trên thiết bị

> Đây là số "thuần board": cả gallery image embedding và query text embedding đều sinh
> từ `.bin` chạy trên HTP v68, rồi tính R@1. Yêu cầu A8 (vision `.bin`) + B11 (text `.bin`) đã có.

**C2.1. Chuẩn bị FULL test set `.raw` (local/free)** — gallery 2000 ảnh, query 4000 caption.
```bash
python3 deployment/scripts/qnn/prepare_vn3k_vision_inputs.py --dataset-root VN3K --split test --selection first --num-samples 2000 --output-dir artifacts/deployment/qnn_inputs/vn3k_test_gallery_2000 --path-mode relative
python3 deployment/scripts/qnn/prepare_vn3k_text_inputs.py --split test --num-samples 4000 --selection first --output-dir artifacts/deployment/qnn_inputs/vn3k_test_query_4000_f32mask_i32 --id-dtype int32 --mask-dtype float32
```

**C2.2. Board: vision `.bin` chạy FULL gallery → image embeddings** (trong thư mục input trên board).
```bash
qnn-net-run --backend "$QNN_LIB/libQnnHtp.so" --retrieve_context vision_encoder.bin --config_file deployment/config/qnn/htp_config_245.json --input_list vn3k_test_gallery_2000/input_list.txt --output_dir qnn_runs/both_int8_vision --profiling_level basic --perf_profile high_performance
```

**C2.3. Board: text `.bin` chạy FULL query → text embeddings.**
```bash
qnn-net-run --backend "$QNN_LIB/libQnnHtp.so" --retrieve_context text_encoder.bin --config_file deployment/config/qnn/htp_config_245.json --input_list vn3k_test_query_4000_f32mask_i32/input_list.txt --output_dir qnn_runs/both_int8_text --profiling_level basic --perf_profile high_performance
```

**C2.4. Kéo embedding board về host** (`adb pull`/`scp` 2 thư mục `qnn_runs/both_int8_{vision,text}` chứa `Output_*.raw` đã dequantize sang float).

**C2.5. Tính R@1 từ embedding board** (local/free) — raw dot product, đúng `LitTBPS._compute_metrics`.
```bash
python3 deployment/scripts/qnn/eval_retrieval_board_embeddings.py --vision-output-dir artifacts/deployment/qnn_runs/both_int8_vision --text-output-dir artifacts/deployment/qnn_runs/both_int8_text --gallery-input-dir artifacts/deployment/qnn_inputs/vn3k_test_gallery_2000 --query-input-dir artifacts/deployment/qnn_inputs/vn3k_test_query_4000_f32mask_i32 --json artifacts/deployment/qnn_runs/both_int8_board_r1.json
```
> `eval_retrieval_board_embeddings.py` đọc `Result_*/output_0.raw` board của cả 2 encoder, map pid theo manifest của input dirs, và tính retrieval raw dot product như `eval_retrieval_quantized_vision.py`. Cho tới khi có text board fidelity/retrieval, **số both-INT8 ở C1 (QDQ) vẫn là proxy**; vision board đã verify ở A9/A10.

Deploy target áp cho both-INT8 board: **T2I R@1 ≥ 50** (kết quả < 50 là FAIL).

---

# Board — prerequisites & gotchas

**Chuẩn bị (1 lần mỗi phiên trên board `qc-rb3g2`):**
1. `adb push` (hoặc scp) lên board: file `.bin`, thư mục input kèm `input_list.txt` + `raw/`. Smoke fidelity: `vn3k_test_10` / `vn3k_text_10_f32mask_i32`. Both-INT8 board (C2): `vn3k_test_gallery_2000` / `vn3k_test_query_4000_f32mask_i32`.
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

---

# Appendix — lỗi lệnh đã gặp và cách tránh

Mục này ghi các lỗi thao tác thực tế đã gặp khi chạy AI Hub/QNN/RB3. Khi có lỗi lạ, kiểm các dòng dưới trước khi debug model.

| Triệu chứng/log | Nguyên nhân thường gặp | Cách sửa/kiểm tra nhanh |
|---|---|---|
| `Unable to load backend... /libQnnHtp.so: No such file or directory` | `QNN_LIB` rỗng hoặc chưa export, nên `"$QNN_LIB/libQnnHtp.so"` thành `/libQnnHtp.so` | `echo "$QNN_LIB"` và `ls "$QNN_LIB/libQnnHtp.so"` trước khi chạy |
| `Unable to find a valid interface` | Trộn version `qnn-net-run` và backend `.so` khác nhau, ví dụ runner 2.45 nhưng lib 2.44 | Luôn set cùng một root: `QAIRT=/opt/qcom/qairt/2.45.40.260406`, rồi `QNN_BIN=$QAIRT/bin/...`, `QNN_LIB=$QAIRT/lib/...` |
| `Backend version mismatch` hoặc extension init fail | `htp_config_*.json` trỏ sai `libQnnHtpNetRunExtensions.so`, hoặc dùng config 2.45 với runtime 2.44 | Dùng config đúng version; `shared_library_path` nên là path tuyệt đối, không để literal `$QNN_LIB/...` trong JSON |
| `Skel lib id mismatch: expected 2.45..., detected 2.43...` | FastRPC/ADSP đang load skel cũ trước skel 2.45; lỗi này cũng xảy ra nếu dùng sai dấu phân cách path | `LD_LIBRARY_PATH` dùng dấu `:`, nhưng `ADSP_LIBRARY_PATH` phải dùng dấu `;`: `export ADSP_LIBRARY_PATH="$QAIRT/lib/hexagon-v68/unsigned;$QAIRT/lib/hexagon-v68;/usr/lib/rfsa/adsp;/dsp"` |
| Reboot rồi vẫn `Skel lib id mismatch` | Không phải do tiến trình cũ; thường là `ADSP_LIBRARY_PATH` vẫn sai hoặc có path skel cũ đứng trước | `echo "$ADSP_LIBRARY_PATH"`; đảm bảo path 2.45 đứng đầu và dùng `;`, không dùng `:` |
| `Failed to open input file: raw/...` | `input_list.txt` dùng path tương đối `raw/...` nhưng đang chạy `qnn-net-run` từ cwd khác | `cd` vào đúng thư mục input rồi chạy `--input_list input_list.txt`, hoặc regenerate input list bằng path tuyệt đối |
| `file size 512 bytes... expected 256 bytes` cho `input_ids` | File raw là `int64` (`64 * 8 = 512`) nhưng graph/context nhận `int32` (`64 * 4 = 256`) | Tạo lại text input với `--id-dtype int32`; kiểm `wc -c raw/*_input_ids.raw` phải là `256` cho seq-len 64 |
| `attention_mask` đúng tên nhưng link/run bất thường | Lẫn mask int64/int32 với f32-mask path | Với text f32-mask/i32 path hiện tại: `input_ids=int32`, `attention_mask=float32`; mỗi file đều `256` bytes cho seq-len 64 |
| `argument --compile-options: expected one argument` khi giá trị bắt đầu bằng `--` | `argparse` hiểu `--quantize_io` là option mới thay vì value | Dùng dạng có dấu bằng: `--compile-options=--quantize_io` hoặc `--compile-options='--truncate_64bit_io --quantize_io'` |
| Lệnh nhiều dòng chạy lệch option | Có khoảng trắng sau dấu `\`, hoặc copy/paste làm backslash không còn là ký tự cuối dòng | Dấu `\` phải là ký tự cuối cùng của dòng; nếu nghi ngờ, chạy lại command một dòng |
| Link fail `Tensor ... has a floating-point type... Please quantize the model including its I/O` | HTP v68 không chấp nhận float island nội bộ trong graph context | Với text mask, dùng finite f32-mask + link-safe rewrite; với output float, compile/link phải có `--quantize_io` nếu device yêu cầu I/O quantized |
| Compile fail `Must use --truncate_64bit_io when input tensors have type int64` | Graph input vẫn là `int64` | Hoặc thêm `--compile-options='--truncate_64bit_io --quantize_io'`, hoặc tốt hơn patch/export graph `input_ids=int32` và dùng raw int32 |
| `qnn-net-run` execute thành công nhưng text cosine rất thấp | Không nhất thiết là raw sai; với full text context, HTP path hiện cho output không đổi giữa `input_ids` thật và zero `input_ids` | Chạy zero-token ablation trước khi debug QAT. Nếu real-vs-zero output giống hệt, dừng full-context text path và chuyển sang split-text `inputs_embeds` |
| Board output dir có file nhưng compare script báo thiếu/sai thứ tự | Script kỳ vọng layout `Result_*/output_0.raw`; có thể đang trỏ nhầm run cũ hoặc output hậu tố khác | `find <output_dir> -maxdepth 2 -name 'output_0.raw' | sort | head`; xóa/trỏ đúng run mới trước khi compare |
| `.bin` lớn bị commit/push fail | Context binary là artifact deploy nặng, không nên track Git | `git rm --cached artifacts/deployment/bin/*.bin` nếu đã track nhầm; giữ `.gitignore` chặn `artifacts/deployment/bin/*.bin` và runtime `.bin` |

Checklist trước mọi board run:

```bash
echo "$QAIRT"
echo "$QNN_BIN"
echo "$QNN_LIB"
echo "$LD_LIBRARY_PATH"
echo "$ADSP_LIBRARY_PATH"
ls "$QNN_BIN/qnn-net-run"
ls "$QNN_LIB/libQnnHtp.so"
ls "$QNN_LIB/libQnnHtpNetRunExtensions.so"
```
