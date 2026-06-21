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
| `TEXT_SPLIT_CALIB` | AI Hub dataset id calib **split-text `inputs_embeds`** (tạo ở B4) | `d9vpnzz09` |
| `VISION_QDQ` | thư mục QDQ vision v8 (learned rotation) | `artifacts/deployment/runtime/rotated_w8a8_learned_qat_v8/job_<ID>_qdq_onnx` |
| `TEXT_QDQ` | thư mục QDQ **split-text** v8 | `artifacts/deployment/runtime/split_text_w8a8/job_<ID>_qdq_onnx` |
| `BOARD` | host board | `qc-rb3g2` |

Split-text calib dataset IDs (đã upload): `d9vpnzz09` = `vn3k_text_calib_500_split_embeds`; `d9pg6dpd9` = 2000-sample (calib bão hòa — 500 và 2000 cho QDQ byte-identical, dùng 500 là đủ).

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

# PART B — TEXT (split-encoder, v8 learned rotation)

> Text inputs đã tạo ở **PART P3**. Part B độc lập với Part A (có thể chạy song song).
>
> ⚠️ **Đường full-graph text (input_ids → HTP) ĐÃ BỎ.** Nó link được nhưng **không dùng được trên board**: output bỏ qua `input_ids` vì HTP v68 không xử lý đúng dynamic `Gather` trên bảng token 250k (chi tiết: `[deploy-master].md` §11–§12, `w8a8_qat_rotated.md` §12A.8). **Đường deploy text cuối cùng là split-encoder:** host CPU làm embedding lookup → HTP chạy transformer nhận `inputs_embeds`. B1/B2 (rotation + QAT) giữ nguyên; từ B3 trở đi là split-text.

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

### B3. Export split-text ONNX (`inputs_embeds`) + static gate (local/free)

Script `export_split_text_onnx.py` xuất phần transformer+head nhận `inputs_embeds`, dựng **mask link-safe finite ngay trong graph** (`M=(1-mask)·(-32)` dạng `[B,1,1,L]`, không `_prepare_4d`, không Cast thừa) và **strip Expand** sau export — nên không có float island, link được HTP v68. Có `--check-input-dir` để chạy gate static (split ONNX vs full `encode_text`, reload model tươi) và `--dump-embeds-dir` để xuất `inputs_embeds .raw` smoke cho board.

```bash
python deployment/scripts/qnn/export_split_text_onnx.py \
  --model-dir artifacts/deployment/exports/exported_model_text_rotated_learned_qat_v8 \
  --attention-mask-dtype float32 \
  --check-input-dir artifacts/deployment/qnn_inputs/vn3k_text_10_f32mask_i32 \
  --dump-embeds-dir artifacts/deployment/qnn_inputs/vn3k_text_10_split_embeds \
  --json artifacts/deployment/exports/exported_model_text_rotated_learned_qat_v8/text_onnx_split/static_vs_pytorch_summary.json
```
**GATE**: in `Mask Expand stripped` + `B1 gate ... PASS` (cosine min ≥ `0.999`; thực tế `0.99999999 / 0.99999976`); op sanity `Expand=0`, `attention_mask` chỉ vào `Shape`/`Reshape` (không Cast). Output: `text_onnx_split/text_encoder_split.onnx` + smoke `inputs_embeds .raw`.

### B3b. Dump bảng token-embedding (local/free, 1 lần — prereq cho B4/B7 + PART D)

Bảng `token_embedding.weight` (đã rotate v8) dùng cho mọi bước sinh `inputs_embeds` và cho on-device encoding. Là weight đóng băng — dump 1 lần.
```bash
python deployment/scripts/qnn/dump_text_embedding_table.py \
  --model-dir artifacts/deployment/exports/exported_model_text_rotated_learned_qat_v8 \
  --output-dir artifacts/deployment/bin/token_embedding_v8 --formats int8 fp16
```
Sinh `token_embedding_int8.bin` (~192 MB, deploy mặc định) + `token_embedding_fp16.bin` (~384 MB) + meta. **Không commit** (`.gitignore` chặn `token_embedding_v8/`).

### B4. Calib `inputs_embeds` (host lookup) + upload → AI Hub (local/free + TỐN JOB nhẹ)

`prepare_split_text_embeds.py` tokenize caption VN3K + lookup bảng (B3b) → `inputs_embeds`; rồi upload **raw modality** (2 key float32).
```bash
# (chạy D1 trước nếu chưa có bảng) — calib 500 caption train
python deployment/scripts/qnn/prepare_split_text_embeds.py \
  --split train --num-samples 500 --selection random --seed 2400 \
  --output-dir artifacts/deployment/qnn_inputs/vn3k_text_calib_500_split_embeds

python deployment/scripts/qnn/upload_qaihub_calibration_dataset.py \
  --modality raw --keys inputs_embeds:float32:1,64,768 attention_mask:float32:1,64 \
  --input-dir artifacts/deployment/qnn_inputs/vn3k_text_calib_500_split_embeds \
  --name msiglip-split-text-embeds-calib-500
```
Ghi `Dataset ID` → `TEXT_SPLIT_CALIB` (vd `d9vpnzz09`). **Calib bão hòa**: QDQ byte-identical cho 500 vs 2000 mẫu (md5), nên 500 đủ; per-channel weight AI Hub đã bật sẵn.

### B5. AI Hub quantize-only split-text → QDQ + fidelity (TỐN JOB — log journal)
```bash
python3 deployment/scripts/qnn/submit_qaihub_quantize_compile.py \
  --model artifacts/deployment/exports/exported_model_text_rotated_learned_qat_v8/text_onnx_split/text_encoder_split.onnx \
  --calibration-data $TEXT_SPLIT_CALIB --weights-dtype int8 --activations-dtype int8 \
  --input-specs '{"inputs_embeds": ((1, 64, 768), "float32"), "attention_mask": ((1, 64), "float32")}' \
  --quantize-only --wait \
  --download-quantized artifacts/deployment/runtime/split_text_w8a8/job_qdq_onnx
```
> Input là `inputs_embeds` float + `attention_mask` float (KHÔNG còn `input_ids`). Fidelity QDQ-vs-PyTorch khớp board `0.9951`.

### B6. Compile/link split-text → `.bin` (TỐN JOB — log journal)
```bash
python3 deployment/scripts/qnn/submit_qaihub_quantize_compile.py \
  --model artifacts/deployment/exports/exported_model_text_rotated_learned_qat_v8/text_onnx_split/text_encoder_split.onnx \
  --calibration-data $TEXT_SPLIT_CALIB --weights-dtype int8 --activations-dtype int8 \
  --input-specs '{"inputs_embeds": ((1, 64, 768), "float32"), "attention_mask": ((1, 64), "float32")}' \
  --compile-options "--quantize_io" \
  --wait --download artifacts/deployment/runtime/split_text_w8a8/text_encoder_split.bin
```
> Link PASS vì mask path khớp dạng `(1-mask)·(-32)` đã link được + không Expand/Cast island. Lịch sử 2 lỗi đã sửa (`/Cast_output_0`, `/Expand_coef`): xem `[deploy-master].md` §12.10.

### B7. Prepare `inputs_embeds` cho query + control (local/free)

Board chạy split graph cần `inputs_embeds` (không phải `input_ids`). `prepare_split_text_embeds.py` (tokenize VN3K + lookup bảng B3b) sinh đủ: smoke 10, full **4000 caption** (gate-comparable, mọi caption test pid=`id-1`), và bản **zero-embeds** control.
```bash
# full 4000-caption query (gate-comparable)
python3 deployment/scripts/qnn/prepare_split_text_embeds.py \
  --split test --all-captions \
  --output-dir artifacts/deployment/qnn_inputs/vn3k_test_query_full_split_embeds

# smoke 10 (nếu chưa dump ở B3)
python3 deployment/scripts/qnn/prepare_split_text_embeds.py \
  --split test --num-samples 10 --selection first \
  --output-dir artifacts/deployment/qnn_inputs/vn3k_text_10_split_embeds

# zero-embeds control (cùng mask, embeds = 0) — chứng minh board graph DÙNG embeds
python3 deployment/scripts/qnn/prepare_split_text_embeds.py \
  --split test --num-samples 10 --selection first --zero-embeds \
  --output-dir artifacts/deployment/qnn_inputs/vn3k_text_10_split_embeds_zero
```
> `prepare_vn3k_text_inputs.py` chỉ lấy 1 caption/ảnh (2000); tập gate đầy đủ là **4000 caption** (`--all-captions`). Số 2000-subset KHÔNG dùng làm gate.

### B8. Board run + fidelity + control (TRÊN BOARD + local)
```bash
export QAIRT=/opt/qcom/qairt/2.45.40.260406
export QNN_BIN=$QAIRT/bin/aarch64-ubuntu-gcc9.4
export QNN_LIB=$QAIRT/lib/aarch64-ubuntu-gcc9.4
export LD_LIBRARY_PATH="$QNN_LIB:$LD_LIBRARY_PATH"
export ADSP_LIBRARY_PATH="$QAIRT/lib/hexagon-v68/unsigned;$QAIRT/lib/hexagon-v68;/usr/lib/rfsa/adsp;/dsp"

# smoke real-embeds
cd /home/ubuntu/sigm/Lam/artifacts/deployment/qnn_inputs/vn3k_text_10_split_embeds
"$QNN_BIN/qnn-net-run" --backend "$QNN_LIB/libQnnHtp.so" \
  --retrieve_context /home/ubuntu/sigm/Lam/artifacts/deployment/runtime/split_text_w8a8/text_encoder_split.bin \
  --config_file /home/ubuntu/sigm/Lam/deployment/config/qnn/htp_config_text_i32.json \
  --input_list input_list.txt \
  --output_dir /home/ubuntu/sigm/Lam/artifacts/deployment/qnn_runs/split_text_w8a8 \
  --profiling_level basic --perf_profile high_performance
# control zero-embeds: lặp lại với cwd .../vn3k_text_10_split_embeds_zero, output .../split_text_w8a8_zero
```
**GATE control (local):** real-vs-zero embeds output PHẢI khác (`max_abs > 0`) → split graph DÙNG embeds (ngược full-graph). **GATE fidelity (local):** board-vs-PyTorch cosine mean ≥ `0.90` (thực tế `0.9951 / 0.9926`). Snippet so sánh: xem `[deploy-master].md` §12.6.

### B9. Board text-isolation retrieval R@1 — FULL 4000 query (TRÊN BOARD + local)
```bash
# board: chạy split bin trên 4000 query inputs_embeds
cd /home/ubuntu/sigm/Lam/artifacts/deployment/qnn_inputs/vn3k_test_query_full_split_embeds
"$QNN_BIN/qnn-net-run" --backend "$QNN_LIB/libQnnHtp.so" \
  --retrieve_context /home/ubuntu/sigm/Lam/artifacts/deployment/runtime/split_text_w8a8/text_encoder_split.bin \
  --config_file /home/ubuntu/sigm/Lam/deployment/config/qnn/htp_config_text_i32.json \
  --input_list input_list.txt \
  --output_dir /home/ubuntu/sigm/Lam/artifacts/deployment/qnn_runs/split_text_query_full \
  --profiling_level basic --perf_profile high_performance

# host: text-isolation retrieval (image FP32 + text board)
python3 deployment/scripts/qnn/eval_retrieval_board_text.py \
  --text-output-dir artifacts/deployment/qnn_runs/split_text_query_full \
  --query-input-dir artifacts/deployment/qnn_inputs/vn3k_test_query_full_split_embeds \
  --model-dir artifacts/deployment/exports/exported_model
```
**GATE**: T2I R@1 ≥ `50.0` (thực tế board `51.30` với onboard lookup, I2T `54.80`). Pass → tiếp C2 both-INT8 board.

---

# PART C — BOTH-INT8 (số deploy cuối cùng)

### C1. Off-board both-INT8 R@1 (local/free, FULL set)

> `eval_retrieval_quantized_vision.py --text-qdq-onnx` feed `input_ids`, nên off-board proxy này dùng **full-graph text QDQ** (đường full-graph faithful off-board, chỉ FAIL khi chạy board). Số proxy `50.25` là mốc deploy off-board. Số board thật của đường split-text là **C2**.
```bash
python3 deployment/scripts/qnn/eval_retrieval_quantized_vision.py --qdq-onnx $VISION_QDQ --text-qdq-onnx <FULL_GRAPH_TEXT_QDQ> --model-dir artifacts/deployment/exports/exported_model --json artifacts/deployment/runtime/both_int8/both_int8_r1.json
```

### C2. Board both-INT8 — retrieval INT8×INT8 TRỰC TIẾP trên thiết bị

> Đây là số "thuần board": gallery image embedding sinh từ vision `.bin`, query text embedding sinh từ **split text `.bin`** (nhận `inputs_embeds`), rồi tính R@1. Yêu cầu A8 (vision `.bin`) + B6 (split text `.bin`) đã có.

**C2.1. Chuẩn bị FULL test set (local/free)** — gallery 2000 ảnh (`.raw` image), query **4000 caption** dạng `inputs_embeds` (host lookup, B7).
```bash
python3 deployment/scripts/qnn/prepare_vn3k_vision_inputs.py --dataset-root VN3K --split test --selection first --num-samples 2000 --output-dir artifacts/deployment/qnn_inputs/vn3k_test_gallery_2000 --path-mode relative
# text query: vn3k_test_query_full_split_embeds (inputs_embeds + attention_mask + manifest, 4000 caption) — xem B7
```

**C2.2. Board: vision `.bin` chạy FULL gallery → image embeddings** (tái dùng output gallery của A10 nếu đã chạy).
```bash
cd /home/ubuntu/sigm/Lam/artifacts/deployment/qnn_inputs/vn3k_test_gallery_2000
qnn-net-run --backend "$QNN_LIB/libQnnHtp.so" --retrieve_context /home/ubuntu/sigm/Lam/artifacts/deployment/bin/vision_encoder.bin --config_file /home/ubuntu/sigm/Lam/deployment/config/qnn/htp_config_245.json --input_list input_list.txt --output_dir /home/ubuntu/sigm/Lam/artifacts/deployment/qnn_runs/rotated_w8a8_learned_qat_v8_gallery_2000 --profiling_level basic --perf_profile high_performance
```

**C2.3. Board: split text `.bin` chạy FULL query (`inputs_embeds`) → text embeddings.** (cwd = thư mục query split-embeds; config text-family)
```bash
cd /home/ubuntu/sigm/Lam/artifacts/deployment/qnn_inputs/vn3k_test_query_full_split_embeds
qnn-net-run --backend "$QNN_LIB/libQnnHtp.so" --retrieve_context /home/ubuntu/sigm/Lam/artifacts/deployment/runtime/split_text_w8a8/text_encoder_split.bin --config_file /home/ubuntu/sigm/Lam/deployment/config/qnn/htp_config_text_i32.json --input_list input_list.txt --output_dir /home/ubuntu/sigm/Lam/artifacts/deployment/qnn_runs/split_text_query_full --profiling_level basic --perf_profile high_performance
```

**C2.4. Kéo embedding board về host** (`adb pull`/`scp` thư mục vision (`..._gallery_2000`) và text (`split_text_query_full`) chứa `Result_*/output_0.raw` đã dequantize sang float).

**C2.5. Tính R@1 từ embedding board** (local/free) — raw dot product, đúng `LitTBPS._compute_metrics`. **Không có `--model-dir`** (cả hai embedding đều từ board).
```bash
python3 deployment/scripts/qnn/eval_retrieval_board_embeddings.py --vision-output-dir artifacts/deployment/qnn_runs/rotated_w8a8_learned_qat_v8_gallery_2000 --text-output-dir artifacts/deployment/qnn_runs/split_text_query_full --gallery-input-dir artifacts/deployment/qnn_inputs/vn3k_test_gallery_2000 --query-input-dir artifacts/deployment/qnn_inputs/vn3k_test_query_full_split_embeds --json artifacts/deployment/qnn_runs/both_int8_board_r1.json
```
> `eval_retrieval_board_embeddings.py` đọc `Result_*/output_0.raw` board của cả 2 encoder, map pid theo `manifest.csv` của input dirs, và tính retrieval raw dot product như `eval_retrieval_quantized_vision.py`.

Deploy target áp cho both-INT8: **T2I R@1 ≥ 50**. Kết quả thực tế: off-board QDQ proxy `50.25` (PASS); board both-INT8 `49.95` (T2I) / `53.05` (I2T) — thiếu `0.05` (~2 query, trong nhiễu). Vision là tower sàn; nâng vision (v9) là hướng tùy chọn để đẩy board qua 50.

---

# PART D — On-device text encoding (mọi thứ trên RB3)

> Các bước B/C ở trên sinh `inputs_embeds` **trên host** rồi đẩy lên board — đủ để đo R@1.
> PART D đưa **toàn bộ mã hoá text lên chính RB3**: lookup bảng + transformer chạy trên thiết bị,
> host chỉ còn chạy retrieval. (Lookup chạy host hay board cho `inputs_embeds` gần như bit-identical;
> hơn nữa bin còn quantize input về uint8 ở I/O, nên không đổi R@1.)

### D1. Dump bảng token-embedding (HOST, 1 lần)

Đã làm ở **B3b** (`dump_text_embedding_table.py` → `token_embedding_v8/`). INT8-table vs FP32 cos `0.99997` (gọn RAM 192 MB, chất lượng ~không đổi vì bin còn quantize input về uint8). Nếu chưa chạy, chạy B3b.

### D2. Push lên RB3 (1 lần)
`scp`/`adb push`: `token_embedding_v8/` (bảng), `deployment/scripts/qnn/board_text_encode.py`, `text_encoder_split.bin`. Board cần `python3 + numpy` (lookup); tokenize-on-board (tùy chọn) cần thêm `transformers` + `src/`.

### D3. RB3 CPU: tokenize (tùy chọn) + lookup → `inputs_embeds`
```bash
# Mode A — từ input_ids đã có (numpy-only, chắc chắn chạy):
python3 deployment/scripts/qnn/board_text_encode.py \
  --table artifacts/deployment/bin/token_embedding_v8/token_embedding_int8.bin \
  --input-dir artifacts/deployment/qnn_inputs/vn3k_test_query_full_split_embeds \
  --output-dir artifacts/deployment/qnn_inputs/query_onboard

# Mode B — tokenize luôn trên board (cần transformers + src/):
#   --captions-file caps.txt --pids-file pids.txt   (thay cho --input-dir)
```
Bảng được `np.memmap` → chỉ đọc 64 hàng/query, nhẹ RAM cho board 4GB.

### D4. RB3 NPU: transformer (giống C2.3, nhưng input từ D3)
```bash
cd /home/ubuntu/sigm/Lam/artifacts/deployment/qnn_inputs/query_onboard
qnn-net-run --backend "$QNN_LIB/libQnnHtp.so" \
  --retrieve_context /home/ubuntu/sigm/Lam/artifacts/deployment/runtime/split_text_w8a8/text_encoder_split.bin \
  --config_file /home/ubuntu/sigm/Lam/deployment/config/qnn/htp_config_text_i32.json \
  --input_list input_list.txt --output_dir /home/ubuntu/sigm/Lam/artifacts/deployment/qnn_runs/onboard_text \
  --profiling_level basic --perf_profile high_performance
```

### D5. HOST: chỉ retrieval (đo chất lượng)
`eval_retrieval_board_text.py` / `eval_retrieval_board_embeddings.py` như B9/C2.5, trỏ `--text-output-dir` vào `qnn_runs/onboard_text`. Encode đã 100% trên RB3; host chỉ so embedding.

> **Bảng = weight, không phải data.** Nó là `token_embedding.weight` đã rotate (Q fold sẵn) — sinh 1 lần ở build-time, nạp 1 lần lúc app khởi động, tái dùng cho mọi query. Chỉ regenerate khi đổi model text. Lý thuyết: `w8a8_qat_rotated.md` §12A.8.

---

# Board — prerequisites & gotchas

**Chuẩn bị (1 lần mỗi phiên trên board `qc-rb3g2`):**
1. `adb push` (hoặc scp) lên board: file `.bin`, thư mục input kèm `input_list.txt` + `raw/`. Smoke fidelity: `vn3k_test_10` (vision) / `vn3k_text_10_split_embeds` (text split). Both-INT8 board (C2): `vn3k_test_gallery_2000` (vision) / `vn3k_test_query_full_split_embeds` (text split, 4000 caption).
2. `export QNN_LIB=<đường dẫn QNN libs trên board>` (chứa `libQnnHtp.so`). Nếu chưa set → lỗi `--backend` không tìm thấy.
3. Chạy `qnn-net-run` **từ trong thư mục input** (hoặc dùng path tuyệt đối) vì `raw/` trong `input_list.txt` là path tương đối.

**Gotchas đã gặp (đừng lặp):**
- `--config_file` phải trỏ tới **file** `deployment/config/qnn/htp_config_245.json`, KHÔNG phải thư mục.
- `$QNN_LIB` chưa export → backend fail; export inline trước lệnh.
- Path `raw/` tương đối fail nếu chạy sai cwd → `cd` vào thư mục input rồi `--input_list input_list.txt`.
- Log profiling: run tăng hậu tố; lấy file **mới nhất** (`qnn-profiling-data_1.log`), tránh log của run fail cũ (`_0.log`).

---

# Logging bắt buộc

- **Mỗi job AI Hub** (A5, A8 vision; B4 upload calib, B5 quantize, B6 compile/link split-text): append vào `deployment/docs/journal/[deploy-master].md` — mục tiêu, job id, input, output/error, fidelity/R@1, quyết định.
- **Số v8 vision** (A7) và **both-INT8** (C1 off-board, C2 board): điền vào bảng `[deploy-master].md` §0 + §12.
- Board fidelity/latency (A9, B8) và board both-INT8 R@1 (C2): vào `[deploy-master].md`.

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
