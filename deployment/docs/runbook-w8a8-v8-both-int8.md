# Runbook — W8A8 v8 Both-INT8 (Vision + Text) trên RB3 Gen2 (HTP v68)

> Vòng lặp **chạy + validate** đầy đủ, theo thứ tự, từ rotation → QAT → AI Hub W8A8
> → gate fidelity → retrieval R@1 → compile/link `.bin` → **retrieval thật trên board**,
> cho **cả vision (v8) và text**, rồi **both-INT8** end-to-end.
>
> Quy ước: mọi bước "local/free" chạy trên máy/cuda box; bước **AI Hub** và **board**
> tốn tài nguyên. Mọi job AI Hub PHẢI log vào `deployment/docs/journal/[deploy]-YYYY-MM-DD.md`.

---

## 0. Biến quy ước (đặt 1 lần)

| Biến | Ý nghĩa | Ví dụ |
|---|---|---|
| `VISION_CALIB_ID` | AI Hub dataset id calib **vision** (đã có) | `d7jzjy1m2` |
| `TEXT_CALIB_ID` | AI Hub dataset id calib **text** (tạo ở Part B) | _in ra sau khi upload_ |
| `VISION_QDQ` | thư mục QDQ vision tốt nhất (v8 nếu thắng, else v7/v6) | `artifacts/deployment/runtime/rotated_w8a8_learned_qat_v8/job_<ID>_qdq_onnx` |
| `TEXT_QDQ` | thư mục QDQ text | `artifacts/deployment/runtime/text_w8a8_qat_t1/job_<ID>_qdq_onnx` |
| `BOARD` | host board | `qc-rb3g2` |

## Gates chấp nhận (áp cho cả vision và text)

| Gate | Ngưỡng | Khi nào |
|---|---:|---|
| Rotation FP32 invariance | cosine min ≥ `0.9999` | sau rotate/learn_rotation |
| Static ONNX vs PyTorch | cosine mean ≥ `0.999` | sau export ONNX |
| ONNX op sanity | `Pow=0`, fused `Gelu`/`LayerNormalization` | sau export ONNX |
| QDQ ONNX vs PyTorch | mean ≥ `0.95`, min ≥ `0.90` | sau AI Hub quantize-only |
| QNN board vs PyTorch | mean ≥ `0.90` | sau qnn-net-run |
| Retrieval (deploy) | T2I R@1 ≥ `48.0` | both-INT8 (stretch ≥ `50.0`) |

> Cosine QDQ chỉ là proxy. **R@1 mới là số quyết định.**

---

# PART A — VISION v8 (learned rotation + QAT)

> Nếu v7 (random) ≈ v8 (learned) thì dùng `exported_model_rotated` (random, đã có) và
> bỏ qua A1; còn lại các bước giống nhau. Đặt `VISION_BASE` = dir rotation đã chọn.

### A1. Learned rotation (local/free) — chỉ khi chọn mode learned
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
Ghi lại `<JOB_ID>` từ output → `VISION_QDQ=artifacts/deployment/runtime/rotated_w8a8_learned_qat_v8/job_<JOB_ID>_qdq_onnx`.

### A6. GATE QDQ fidelity (local/free)
```bash
python3 deployment/scripts/qnn/compare_onnx_with_pytorch.py --onnx-model $VISION_QDQ --model-dir artifacts/deployment/exports/exported_model --input-dir artifacts/deployment/qnn_inputs/vn3k_test_10 --precision fp32 --json artifacts/deployment/runtime/rotated_w8a8_learned_qat_v8/qdq_vs_pytorch_summary.json --csv artifacts/deployment/runtime/rotated_w8a8_learned_qat_v8/qdq_vs_pytorch.csv
```
Kỳ vọng mean ≥ 0.95, min ≥ 0.90.

### A7. GATE retrieval R@1 — vision-isolation (local/free, FULL set)
```bash
python3 deployment/scripts/qnn/eval_retrieval_quantized_vision.py --qdq-onnx $VISION_QDQ --model-dir artifacts/deployment/exports/exported_model --json artifacts/deployment/runtime/rotated_w8a8_learned_qat_v8/retrieval_r1.json
```
Baseline phải ~52.28; `vision_int8` T2I R@1 ≥ 48 (mục tiêu ≥ 50). Đây là số quyết định v8 thắng/thua v7.

### A8. Compile/link → `.bin` (TỐN JOB — chỉ khi A6+A7 pass — log journal)
```bash
python3 deployment/scripts/qnn/submit_qaihub_quantize_compile.py --model artifacts/deployment/exports/exported_model_rotated_learned_qat_v8/vision_onnx --calibration-data d7jzjy1m2 --weights-dtype int8 --activations-dtype int8 --wait --download artifacts/deployment/runtime/rotated_w8a8_learned_qat_v8/vision_encoder.bin
```

### A9. Board run + fidelity (TRÊN BOARD — xem §Board)
```bash
# trên board (đã push .bin + thư mục input, đã export QNN_LIB)
qnn-net-run --backend "$QNN_LIB/libQnnHtp.so" --retrieve_context artifacts/deployment/runtime/rotated_w8a8_learned_qat_v8/vision_encoder.bin --config_file deployment/config/qnn/htp_config_245.json --input_list artifacts/deployment/qnn_inputs/vn3k_test_10/input_list.txt --output_dir artifacts/deployment/qnn_runs/rotated_w8a8_learned_qat_v8 --profiling_level basic --perf_profile high_performance
# về máy: board fidelity vs PyTorch
python3 deployment/scripts/qnn/compare_qnn_with_pytorch.py --qnn-output-dir artifacts/deployment/qnn_runs/rotated_w8a8_learned_qat_v8 --model-dir artifacts/deployment/exports/exported_model --input-dir artifacts/deployment/qnn_inputs/vn3k_test_10 --precision fp32 --json artifacts/deployment/qnn_runs/rotated_w8a8_learned_qat_v8/qnn_vs_pytorch_summary.json --csv artifacts/deployment/qnn_runs/rotated_w8a8_learned_qat_v8/qnn_vs_pytorch.csv
```
**GATE board**: mean ≥ 0.90. (Board fidelity ≈ QDQ fidelity là dấu hiệu QDQ proxy đúng.)

---

# PART B — TEXT (mirror v8; chọn mode rotation theo kết quả vision)

### B0. Data prep (local/free, KHÔNG phụ thuộc vision)
```bash
python deployment/scripts/qnn/prepare_vn3k_text_inputs.py --split train --num-samples 500 --selection random --output-dir artifacts/deployment/qnn_inputs/vn3k_text_calib_500
python deployment/scripts/qnn/prepare_vn3k_text_inputs.py --split train --num-samples 4000 --selection random --output-dir artifacts/deployment/qnn_inputs/vn3k_text_train_4000
python deployment/scripts/qnn/prepare_vn3k_text_inputs.py --split test --num-samples 100 --selection first --output-dir artifacts/deployment/qnn_inputs/vn3k_text_test_100
```

### B1. Rotation (chọn mode SAU khi có kết quả vision A7)
- **random** (nếu v7 ≈ v8): đã có `exported_model_text_rotated` → `TEXT_BASE=exported_model_text_rotated`. Bỏ qua lệnh dưới.
- **learned** (nếu v8 thắng):
```bash
PYTHONUNBUFFERED=1 python deployment/scripts/qnn/learn_rotation_text.py --model-dir artifacts/deployment/exports/exported_model --output-dir artifacts/deployment/exports/exported_model_text_rotated_learned --calib-dir artifacts/deployment/qnn_inputs/vn3k_text_calib_500 --gate-input-dir artifacts/deployment/qnn_inputs/vn3k_text_10 --num-calib 256 --tokens-per-sample 32 --steps 3000 --lr 2e-3 --device cuda
```
**GATE** cosine min ≥ 0.9999 → `TEXT_BASE=exported_model_text_rotated_learned`.

### B2. Text QAT (local/free, cuda) — `--modality text`, recipe **v6** (const lr 1e-5, 15 ep)
> Dùng recipe v6 (giống vision sau khi v7 cosine thua). Nếu muốn thử cosine cho text thì chạy thêm 1 biến thể để so.
```bash
PYTHONUNBUFFERED=1 python deployment/scripts/qnn/train_vision_quant_robust.py --modality text --model-dir artifacts/deployment/exports/exported_model_text_rotated --train-input-dir artifacts/deployment/qnn_inputs/vn3k_text_train_4000 --val-input-dir artifacts/deployment/qnn_inputs/vn3k_text_test_100 --output-dir artifacts/deployment/exports/exported_model_text_rotated_qat_t1 --device cuda --batch-size 16 --epochs 15 --lr 1e-5 --fake-quant-observer ema --quant-head --quant-linears --quant-attention --start-layer 0 --end-layer 11 --num-workers 4
```
> Đổi `--model-dir` sang `..._text_rotated_learned` nếu chọn mode learned ở B1.

### B3. Export text ONNX opset-20 (local/free)
```bash
python deployment/scripts/qnn/export_text_onnx.py --model-dir artifacts/deployment/exports/exported_model_text_rotated_qat_t1
```

### B4. GATE static text ONNX vs PyTorch (local/free)
```bash
python3 deployment/scripts/qnn/compare_text_onnx_with_pytorch.py --onnx-model artifacts/deployment/exports/exported_model_text_rotated_qat_t1/text_onnx/text_encoder.onnx --model-dir artifacts/deployment/exports/exported_model_text_rotated_qat_t1 --input-dir artifacts/deployment/qnn_inputs/vn3k_text_10 --json /tmp/text_static.json --csv /tmp/text_static.csv
```
Kỳ vọng cosine ~1.0, `Pow=0`.

### B5. Upload calib text → AI Hub (TỐN JOB nhẹ — log journal)
```bash
python deployment/scripts/qnn/upload_qaihub_calibration_dataset.py --modality text --input-dir artifacts/deployment/qnn_inputs/vn3k_text_calib_500 --name msiglip-text-vn3k-calib-500
```
Ghi `Dataset ID` in ra → `TEXT_CALIB_ID`.

### B6. AI Hub quantize-only text → QDQ (TỐN JOB — log journal) — `--modality text`
```bash
python3 deployment/scripts/qnn/submit_qaihub_quantize_compile.py --modality text --model artifacts/deployment/exports/exported_model_text_rotated_qat_t1/text_onnx --calibration-data $TEXT_CALIB_ID --weights-dtype int8 --activations-dtype int8 --quantize-only --wait --download-quantized artifacts/deployment/runtime/text_w8a8_qat_t1/job_qdq_onnx
```
Ghi `<JOB_ID>` → `TEXT_QDQ=artifacts/deployment/runtime/text_w8a8_qat_t1/job_<JOB_ID>_qdq_onnx`.
> `--modality text` giữ int I/O và BỎ `--quantize_io` (token id tới ~250k, không int8 hóa được).

### B7. GATE QDQ fidelity text (local/free) — **canh rủi ro mask 3.4e38**
```bash
python3 deployment/scripts/qnn/compare_text_onnx_with_pytorch.py --onnx-model $TEXT_QDQ --model-dir artifacts/deployment/exports/exported_model_text_rotated_qat_t1 --input-dir artifacts/deployment/qnn_inputs/vn3k_text_10 --json /tmp/text_qdq_fid.json --csv /tmp/text_qdq_fid.csv
```
Nếu cosine collapse → mask `3.4e38` trong `scores+mask` bị per-tensor INT8 phá; xử lý trước khi compile/link.

### B8. GATE retrieval — text-isolation (local/free, FULL set)
```bash
python3 deployment/scripts/qnn/eval_retrieval_quantized_vision.py --skip-vision-qdq --text-qdq-onnx $TEXT_QDQ --model-dir artifacts/deployment/exports/exported_model --json artifacts/deployment/runtime/text_w8a8_qat_t1/text_isolation_r1.json
```
Combo `text_int8` (image FP32 + text QDQ) cho thấy riêng text-quant rớt bao nhiêu.

### B9. Compile/link text → `.bin` (TỐN JOB — log journal) — `--modality text`
```bash
python3 deployment/scripts/qnn/submit_qaihub_quantize_compile.py --modality text --model artifacts/deployment/exports/exported_model_text_rotated_qat_t1/text_onnx --calibration-data $TEXT_CALIB_ID --weights-dtype int8 --activations-dtype int8 --wait --download artifacts/deployment/runtime/text_w8a8_qat_t1/text_encoder.bin
```

### B10. Board run + fidelity text (TRÊN BOARD)
```bash
qnn-net-run --backend "$QNN_LIB/libQnnHtp.so" --retrieve_context artifacts/deployment/runtime/text_w8a8_qat_t1/text_encoder.bin --config_file deployment/config/qnn/htp_config_245.json --input_list artifacts/deployment/qnn_inputs/vn3k_text_10/input_list.txt --output_dir artifacts/deployment/qnn_runs/text_w8a8_qat_t1 --profiling_level basic --perf_profile high_performance
python3 deployment/scripts/qnn/compare_qnn_with_pytorch.py --qnn-output-dir artifacts/deployment/qnn_runs/text_w8a8_qat_t1 --model-dir artifacts/deployment/exports/exported_model --input-dir artifacts/deployment/qnn_inputs/vn3k_text_10 --precision fp32 --json artifacts/deployment/qnn_runs/text_w8a8_qat_t1/qnn_vs_pytorch_summary.json --csv artifacts/deployment/qnn_runs/text_w8a8_qat_t1/qnn_vs_pytorch.csv
```
> Lưu ý: text input là 2 int (`input_ids`, `attention_mask`); `input_list.txt` của text dùng dạng `input_ids:=... attention_mask:=...`.

---

# PART C — BOTH-INT8 (số deploy cuối cùng)

### C1. Off-board both-INT8 R@1 (local/free, FULL set) — **số chính của luận văn**
```bash
python3 deployment/scripts/qnn/eval_retrieval_quantized_vision.py --qdq-onnx $VISION_QDQ --text-qdq-onnx $TEXT_QDQ --model-dir artifacts/deployment/exports/exported_model --json artifacts/deployment/runtime/both_int8/both_int8_r1.json
```
In 4 combo: `baseline_fp32` (~52.28), `vision_int8`, `text_int8`, **`both_int8`** = số deploy thật. GATE T2I R@1 ≥ 48 (stretch ≥ 50).

### C2. Board both-INT8 (tùy chọn, end-to-end thật trên thiết bị)
- Chạy A9 (vision `.bin`) và B10 (text `.bin`) để có embedding board cho cả 2 encoder.
- Ghép retrieval từ embedding board: hiện `eval_retrieval_quantized_vision.py` đọc QDQ ONNX (proxy đã chứng minh ≈ board). Nếu cần số board thuần, export embedding board của cả vision+text rồi tính R@1 bằng cùng hàm `_metrics` (raw dot product). _(Chưa có script ghép board-embedding; QDQ both-INT8 ở C1 là proxy đã verify board≈QDQ ở A9/B10.)_

---

# Board — prerequisites & gotchas

**Chuẩn bị (1 lần mỗi phiên trên board `qc-rb3g2`):**
1. `adb push` (hoặc scp) lên board: file `.bin`, thư mục input (`vn3k_test_10` / `vn3k_text_10`) kèm `input_list.txt` + `raw/`.
2. `export QNN_LIB=<đường dẫn QNN libs trên board>` (chứa `libQnnHtp.so`). Nếu chưa set → lỗi `--backend` không tìm thấy.
3. Chạy `qnn-net-run` **từ trong thư mục input** (hoặc dùng path tuyệt đối) vì `raw/` trong `input_list.txt` là path tương đối.

**Gotchas đã gặp (đừng lặp):**
- `--config_file` phải trỏ tới **file** `deployment/config/qnn/htp_config_245.json`, KHÔNG phải thư mục.
- `$QNN_LIB` chưa export → backend fail; export inline trước lệnh.
- Path `raw/` tương đối fail nếu chạy sai cwd → `cd` vào thư mục input rồi `--input_list input_list.txt`.
- Log profiling: run tăng hậu tố; lấy file **mới nhất** (`qnn-profiling-data_1.log`), tránh log của run fail cũ (`_0.log`).

---

# Logging bắt buộc

- **Mỗi job AI Hub** (A5, A8, B5, B6, B9): append vào `deployment/docs/journal/[deploy]-YYYY-MM-DD.md` — mục tiêu, job id, input, output/error, fidelity/R@1, quyết định.
- **Số v8 vision** (A7) và **both-INT8** (C1): điền vào bảng `[deploy-master].md` §11 + §6.
- Board fidelity/latency (A9, B10): vào journal ngày tương ứng.
