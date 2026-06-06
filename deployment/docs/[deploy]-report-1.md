# Báo cáo tiến độ nén và deploy mSigLIP lên Qualcomm RB3 Gen2

> **Ngày báo cáo:** 2026-06-05  
> **Thiết bị mục tiêu:** Qualcomm RB3 Gen2 / QCS6490 / HTP V68  
> **Model nguồn:** `epoch=56-val_score=52.28.ckpt`  
> **Baseline training:** VN3K text-to-image R@1 = `52.28%`  
> **Trạng thái hiện tại:** Vision encoder đã chạy được trên RB3 HTP, nhưng các cấu hình PTQ/QDQ đã thử đều fail fidelity embedding. Chưa compile text encoder, chưa chạy retrieval end-to-end trên board.

---

## 1. Tóm tắt 
Pipeline deploy đã đi qua phần khó về runtime: vision encoder đã được export, quantize/compile qua Qualcomm AI Hub, tải QNN context binary về, và chạy thật trên RB3 bằng QNN HTP runtime. Board chạy được graph, sinh output đúng shape, không NaN/Inf, và profiling vision encoder đạt khoảng `22.25 ms/image` theo NetRun, `20.72 ms/image` trên accelerator.

Blocker hiện tại không phải ONNX export, không phải link job, cũng không phải RB3 runtime. Blocker là **fidelity sau post-training quantization (PTQ/QDQ)**. Static ONNX khớp PyTorch gần như tuyệt đối, nhưng QDQ ONNX đã lệch rất mạnh so với PyTorch. Vì vậy mọi candidate QNN mới chỉ được compile/link nếu QDQ ONNX pass gate trước.

Kết luận quan trọng nhất:

- Vision runtime trên HTP: **PASS**.
- Static ONNX control: **PASS**, `cosine_l2_mean = 1.0000`.
- INT8/QDQ fidelity: **FAIL**, các candidate hiện tại chỉ quanh `0.1658 - 0.1906`, rất xa gate `0.95`.
- Không được mở rộng `vn3k_test_100`, full VN3K R@1, compile text encoder, hoặc retrieval end-to-end cho đến khi vision QDQ fidelity pass.

---

## 2. Mục tiêu deploy và tiêu chí gate

Mục tiêu là deploy mSigLIP TBPS lên RB3 Gen2 để chạy tìm kiếm người bằng mô tả tiếng Việt trực tiếp trên edge device, không phụ thuộc cloud runtime. Hai encoder cần chạy cục bộ:

| Encoder | Input | Output | Trạng thái |
|---|---|---|---|
| Vision encoder | `image: 1x3x256x256` | `image_embedding: 1x768` | Runtime HTP pass, fidelity QDQ/QNN fail |
| Text encoder | `input_ids`, `attention_mask` | `text_embedding: 1x768` | Chưa compile, chờ vision pass |

Gate hiện hành cho candidate vision quantization:

| Gate | Ngưỡng | Ý nghĩa |
|---|---:|---|
| Static ONNX vs PyTorch | `cosine_l2_mean >= 0.999` | Control phải pass trước khi debug QDQ |
| QDQ ONNX vs PyTorch | `cosine_l2_mean >= 0.95` | Đủ tốt để thử compile/link QNN |
| QDQ ONNX min | `cosine_l2_min >= 0.90` | Không có sample lệch nặng |
| QNN vs PyTorch sau link | `cosine_l2_mean >= 0.90` | Đủ để mở rộng test trên board |
| Full retrieval | `T2I R@1 >= 48.0` | Mục tiêu deploy so với baseline `52.28%` |

Các gate này dùng để tránh tốn thời gian AI Hub/RB3 khi QDQ ONNX đã sai rõ trước runtime.

---

## 3. Timeline kỹ thuật đã thực hiện

### 3.1 Export model

Đã merge LoRA vào backbone và export artifact inference:

```text
artifacts/deployment/exports/exported_model/model_fp32.pt
artifacts/deployment/exports/exported_model/model_fp16.pt
artifacts/deployment/exports/exported_model/config.yaml
```

Đã export ONNX cho cả hai encoder:

```text
artifacts/deployment/exports/exported_model/vision_onnx/
artifacts/deployment/exports/exported_model/text_onnx/
```

Vision encoder ONNX là đường đang debug:

```text
image 1x3x256x256 -> image_embedding 1x768
```

Text encoder đã có ONNX nhưng chưa compile vì vision encoder chưa pass fidelity.

### 3.2 Compile vision dummy-cal và chạy RB3

AI Hub job `jgkr7qwn5` compile được vision encoder INT8 dummy-cal thành QNN context binary cho HTP V68. Binary này chứng minh compile/runtime path hoạt động, nhưng không dùng cho accuracy vì calibration không thật.

Runtime trên RB3:

```text
qnn-net-run --backend libQnnHtp.so --retrieve_context vision_encoder.bin
```

Kết quả runtime:

```text
10/10 output hợp lệ
mỗi output = 768 float32 = 3072 bytes
không NaN/Inf
NetRun avg ~= 22.25 ms/image
Accelerator avg ~= 20.72 ms/image
```

Fidelity dummy-cal fail:

```text
QNN vs PyTorch cosine_l2_mean = 0.1727
```

### 3.3 Real calibration 500 và lỗi CLI preserve FP I/O

Đã tạo calibration dataset thật 500 ảnh VN3K train:

```text
Local input: artifacts/deployment/qnn_inputs/vn3k_train_calib_500/
AI Hub dataset ID: d7x5gzne9
Dataset name: msiglip-vision-vn3k-train-calib-500
```

Compile job `j5wx6x63p` dùng deprecated CLI path:

```text
qai-hub submit-compile-job --quantize_full_type int8
```

Job fail ở context-binary stage vì CLI tự preserve floating-point I/O:

```text
--preserve_io_datatype image output_0
Tensor 'image' has a floating-point type which is not supported by the targeted device.
```

Kết luận: dataset `d7x5gzne9` dùng được, nhưng CLI flow này không dùng tiếp vì HTP cần INT8/INT16 ở I/O boundary.

### 3.4 Python API flow compile/link real-cal 500

Đã chuyển sang helper dùng Python API:

```text
deployment/scripts/qnn/submit_qaihub_quantize_compile.py
```

Flow:

```text
submit_quantize_job -> submit_compile_and_link_jobs
```

Job `jpr9v62vp` tạo được:

```text
artifacts/deployment/qnn_inputs/vision_encoder_calib500.bin
```

Binary `vision_encoder_calib500.bin` chạy được trên RB3 HTP, không NaN/Inf, nhưng fidelity vẫn fail:

```text
QNN calib500 vs PyTorch cosine_l2_mean = 0.1300
cosine_l2_min/max = 0.0799 / 0.1774
```

### 3.5 QDQ diagnosis: xác định lỗi nằm trước QNN runtime

Đã tải QDQ ONNX sau quantize và so với PyTorch trên cùng `vn3k_test_10`:

```text
Static ONNX vs PyTorch cosine_l2_mean = 1.0000
QDQ ONNX vs PyTorch cosine_l2_mean = 0.1682
QNN calib500 vs PyTorch cosine_l2_mean = 0.1300
```

Kết luận: lỗi fidelity đã xuất hiện ở QDQ ONNX, tức là nằm ở PTQ/QDQ trước khi compile/link và trước runtime QNN.

### 3.6 Calibration 2,000 và quantize-only experiments

Đã tạo calibration set 2,000 ảnh VN3K train:

```text
Local input: artifacts/deployment/qnn_inputs/vn3k_train_calib_2000/
AI Hub dataset ID: d7jzjy1m2
Dataset name: msiglip-vision-vn3k-train-calib-2000
```

Đã xác minh dataset ID trên AI Hub:

```text
d7jzjy1m2 -> msiglip-vision-vn3k-train-calib-2000
d7x5gzne9 -> msiglip-vision-vn3k-train-calib-500
```

Đã chạy các quantize-only candidate với calib2000. Tất cả đều chỉ dừng ở QDQ compare, không compile/link vì chưa đạt gate.

---

## 4. Kết quả fidelity đầy đủ

### 4.1 Baseline và QNN/QDQ calib500

| So sánh / Artifact | Dataset dùng để compare | cosine_l2_mean | cosine_l2_min | cosine_l2_max | Kết luận |
|---|---|---:|---:|---:|---|
| Static ONNX vs PyTorch | `vn3k_test_10` | `1.0000` | - | - | PASS, ONNX export đúng |
| QDQ ONNX calib500 vs PyTorch | `vn3k_test_10` | `0.1682` | `0.1272` | `0.2157` | FAIL, PTQ/QDQ phá embedding |
| QNN calib500 vs PyTorch | `vn3k_test_10_calib500` output | `0.1300` | `0.0799` | `0.1774` | FAIL, runtime thấp hơn nhưng lỗi đã có từ QDQ |

### 4.2 Quantize-only candidates với calib2000

| Candidate | Config | Calibration dataset | AI Hub/AIMET PSNR | cosine_l2_mean | cosine_l2_min | cosine_l2_max | Gate | Quyết định |
|---|---|---|---:|---:|---:|---:|---|---|
| `jgomex415` | W8A8 default | `d7jzjy1m2` | `17.9452` | `0.1692` | `0.1239` | `0.1962` | FAIL | Không compile/link |
| `jp2j31dm5` | W8A16 | `d7jzjy1m2` | `17.3564` | `0.1863` | `0.1366` | `0.2478` | FAIL | Không compile/link |
| `j5m4vjxd5` | W8A8 + `--range_scheme min_max` | `d7jzjy1m2` | `17.8713` | `0.1658` | `0.1218` | `0.2159` | FAIL | Không compile/link |
| `jgl7en9l5` | Lite-MP default | `d7jzjy1m2` | `18.6722` | `0.1906` | `0.1784` | `0.2212` | FAIL | Không compile/link |

Lite-MP default là candidate tốt nhất hiện tại, nhưng vẫn thấp rất xa gate `0.95`.

### 4.3 Artifact của các candidate

```text
artifacts/deployment/runtime/job_jgomex415_qdq_onnx/model.onnx
artifacts/deployment/runtime/job_jp2j31dm5_qdq_onnx/model.onnx
artifacts/deployment/runtime/job_j5m4vjxd5_qdq_onnx/model.onnx
artifacts/deployment/runtime/job_jgl7en9l5_qdq_onnx/model.onnx
```

Compare outputs:

```text
artifacts/deployment/runtime/ptq_experiments/calib2000/qdq_vs_pytorch_summary.json
artifacts/deployment/runtime/ptq_experiments/calib2000_w8a16/qdq_vs_pytorch_summary.json
artifacts/deployment/runtime/ptq_experiments/calib2000_w8a8_minmax/qdq_vs_pytorch_summary.json
artifacts/deployment/runtime/ptq_experiments/calib2000_lite_mp_default/qdq_vs_pytorch_summary.json
```

---

## 5. Raw input và calibration data

Các file `.raw` không phải ảnh RAW camera. Chúng là tensor input đã preprocess xong cho model/QNN:

```text
RGB -> resize 256x256 -> ToTensor [0,1] -> Normalize(mean=0.5, std=0.5)
```

Layout:

```text
NCHW float32 = 1 x 3 x 256 x 256
```

Kích thước kỳ vọng:

```text
1 * 3 * 256 * 256 * 4 = 786432 bytes
```

Raw input audit:

| Set | Listed | Valid | Bytes/file | NaN/Inf | Range | Mean | Std | Kết luận |
|---|---:|---:|---:|---|---|---:|---:|---|
| `vn3k_train_calib_500` | 500 | 500 | `786432` | false | `[-1, 1]` | `-0.2197` | `0.4986` | PASS |
| `vn3k_train_calib_2000` | 2000 | 2000 | `786432` | false | `[-1, 1]` | `-0.2302` | `0.4931` | PASS |
| `vn3k_test_10` | 10 | 10 | `786432` | false | `[-1, 1]` | `-0.3400` | `0.4483` | PASS |

Vai trò của hai tập dữ liệu quan trọng:

- `vn3k_train_calib_2000` / `d7jzjy1m2`: dùng để AI Hub/AIMET calibrate quantization range khi tạo QDQ model.
- `vn3k_test_10`: dùng làm smoke fidelity gate độc lập sau quantize, so QDQ ONNX với PyTorch trên input không dùng để calibrate.

Có thể compare thêm trên `vn3k_train_calib_2000` để hiểu lỗi, nhưng không nên thay gate độc lập bằng calibration set.

---

## 6. Chẩn đoán hiện tại

### 6.1 Những nguyên nhân đã loại trừ

- **ONNX export sai:** loại trừ vì static ONNX vs PyTorch đạt `1.0000`.
- **Raw input hỏng/sai byte size:** loại trừ vì audit pass, tất cả file đúng `786432` bytes và không NaN/Inf.
- **Nhầm dataset calib2000:** loại trừ vì AI Hub metadata xác nhận `d7jzjy1m2 = msiglip-vision-vn3k-train-calib-2000`.
- **QNN runtime không chạy được:** loại trừ vì context binary đã chạy trên RB3, output đúng shape, không NaN/Inf.

### 6.2 Blocker thật sự

PTQ/QDQ đang làm lệch hướng embedding. Đây không chỉ là lệch scale output, vì cosine giữa vector QDQ và vector PyTorch rất thấp sau L2 normalize.

Các thử nghiệm đã cho thấy:

- Tăng calibration từ 500 lên 2,000 không sửa được fidelity.
- Đổi activation dtype sang INT16 chỉ cải thiện nhẹ.
- Đổi range scheme sang `min_max` không cải thiện.
- Lite-MP default cải thiện tốt nhất đến `0.1906`, nhưng vẫn fail nặng.
- PSNR AI Hub/AIMET cao hơn không đồng nghĩa cosine embedding fidelity tốt hơn.

### 6.3 Quyết định kỹ thuật

Không compile/link QNN cho các candidate fail QDQ:

```text
jgomex415
jp2j31dm5
j5m4vjxd5
jgl7en9l5
```

Không chạy `vn3k_test_100`, full VN3K R@1, text encoder, hoặc retrieval end-to-end cho tới khi vision QDQ fidelity pass.

---

## 7. Việc tiếp theo

### 7.1 Candidate tiếp theo

Chạy Lite-MP 30% INT16, quantize-only:

```bash
venv/bin/python deployment/scripts/qnn/submit_qaihub_quantize_compile.py \
  --model artifacts/deployment/exports/exported_model/vision_onnx \
  --calibration-data d7jzjy1m2 \
  --weights-dtype int8 \
  --activations-dtype int8 \
  --quantize-options="--lite_mp percentage=30;override_qtype=int16" \
  --name msiglip-vision-lite-mp-30-int16-calib2000-qonly \
  --wait \
  --quantize-only \
  --download-quantized artifacts/deployment/runtime/ptq_experiments/calib2000_lite_mp_30_int16/qdq_onnx
```

Sau khi có QDQ ONNX, chạy compare:

```bash
venv/bin/python deployment/scripts/qnn/compare_onnx_with_pytorch.py \
  --onnx-model artifacts/deployment/runtime/<QDQ_ONNX_DIR> \
  --model-dir artifacts/deployment/exports/exported_model \
  --input-dir artifacts/deployment/qnn_inputs/vn3k_test_10 \
  --precision fp32 \
  --json artifacts/deployment/runtime/ptq_experiments/<EXP>/qdq_vs_pytorch_summary.json \
  --csv artifacts/deployment/runtime/ptq_experiments/<EXP>/qdq_vs_pytorch.csv
```

### 7.2 Nếu Lite-MP 30% INT16 vẫn fail

Chạy Lite-MP 10% FP16:

```bash
venv/bin/python deployment/scripts/qnn/submit_qaihub_quantize_compile.py \
  --model artifacts/deployment/exports/exported_model/vision_onnx \
  --calibration-data d7jzjy1m2 \
  --weights-dtype int8 \
  --activations-dtype int8 \
  --quantize-options="--lite_mp percentage=10;override_qtype=fp16" \
  --name msiglip-vision-lite-mp-10-fp16-calib2000-qonly \
  --wait \
  --quantize-only \
  --download-quantized artifacts/deployment/runtime/ptq_experiments/calib2000_lite_mp_10_fp16/qdq_onnx
```

### 7.3 Diagnostic phụ

Có thể compare thêm trên calibration set `vn3k_train_calib_2000` để xem fail có xảy ra ngay trên data dùng calibrate không:

```bash
venv/bin/python deployment/scripts/qnn/compare_onnx_with_pytorch.py \
  --onnx-model artifacts/deployment/runtime/<QDQ_ONNX_DIR> \
  --model-dir artifacts/deployment/exports/exported_model \
  --input-dir artifacts/deployment/qnn_inputs/vn3k_train_calib_2000 \
  --precision fp32 \
  --json artifacts/deployment/runtime/ptq_experiments/<EXP>/qdq_vs_pytorch_calib2000_summary.json \
  --csv artifacts/deployment/runtime/ptq_experiments/<EXP>/qdq_vs_pytorch_calib2000.csv
```

Diagnostic này chỉ để hiểu thêm lỗi, không thay thế gate độc lập.

### 7.4 Nếu mọi AI Hub option vẫn fail quanh `0.16 - 0.20`

Cần chuyển sang hướng quantization pipeline có node-level control:

- Exclude hoặc giữ float/mixed precision cho LayerNorm.
- Giữ float/mixed precision cho projection/head cuối.
- Giữ float/mixed precision cho L2-normalization path.
- Kiểm tra attention/softmax score path.
- Ưu tiên QDQ fidelity trước, chưa compile/link QNN cho tới khi pass.

---

## 8. Trạng thái cuối báo cáo

| Hạng mục | Trạng thái |
|---|---|
| Export model | DONE |
| Vision ONNX | DONE |
| Text ONNX | DONE |
| Vision QNN HTP runtime | DONE / runtime pass |
| Vision QNN/QDQ fidelity | FAIL |
| Calibration 500 | DONE |
| Calibration 2,000 | DONE |
| Dataset ID calib2000 | VERIFIED: `d7jzjy1m2` |
| Compile/link candidate mới | BLOCKED until QDQ pass |
| RB3 benchmark mở rộng | BLOCKED until QDQ pass |
| Text encoder compile | BLOCKED until vision pass |
| End-to-end retrieval | BLOCKED until vision + text pass |

Kết luận: deployment hiện đã chứng minh được khả năng chạy vision encoder trên RB3 HTP, nhưng chưa chứng minh được accuracy/retrieval sau nén. Công việc tiếp theo phải tiếp tục nằm ở QDQ/PTQ fidelity gate, không phải mở rộng benchmark hoặc compile text encoder.
