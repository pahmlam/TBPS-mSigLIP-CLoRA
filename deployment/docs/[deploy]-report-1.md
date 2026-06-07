# Báo cáo tiến độ nén và deploy mSigLIP lên Qualcomm RB3 Gen2

> **Ngày báo cáo:** 2026-06-05
> **Cập nhật mới nhất:** 2026-06-06
> **Thiết bị mục tiêu:** Qualcomm RB3 Gen2 / QCS6490 / HTP V68
> **Model nguồn:** `epoch=56-val_score=52.28.ckpt`
> **Baseline training:** VN3K text-to-image R@1 = `52.28%`
> **Trạng thái hiện tại:** Vision encoder đã chạy được trên RB3 HTP. Các cấu hình PTQ/QDQ global của AI Hub vẫn fail fidelity, nhưng local QDQ sensitivity đã tìm được candidate `encoder_blocks_4_11_float` pass gate trên `vn3k_test_10`. Chưa compile/link QNN cho candidate này, chưa compile text encoder, chưa chạy retrieval end-to-end trên board.

---

## 1. Tóm tắt
Pipeline deploy đã đi qua phần khó về runtime: vision encoder đã được export, quantize/compile qua Qualcomm AI Hub, tải QNN context binary về, và chạy thật trên RB3 bằng QNN HTP runtime. Board chạy được graph, sinh output đúng shape, không NaN/Inf, và profiling vision encoder đạt khoảng `22.25 ms/image` theo NetRun, `20.72 ms/image` trên accelerator.

Blocker ban đầu không phải ONNX export, không phải link job, cũng không phải RB3 runtime. Blocker là **fidelity sau post-training quantization (PTQ/QDQ)**. Static ONNX khớp PyTorch gần như tuyệt đối, nhưng QDQ ONNX global đã lệch rất mạnh so với PyTorch. Vì vậy mọi candidate QNN mới chỉ được compile/link nếu QDQ ONNX pass gate trước.

Kết luận quan trọng nhất:

- Vision runtime trên HTP: **PASS**.
- Static ONNX control: **PASS**, `cosine_l2_mean = 1.0000`.
- Global AI Hub PTQ/QDQ fidelity: **FAIL**, candidate tốt nhất đạt `0.3267`, vẫn rất xa gate `0.95`.
- Local QDQ sensitivity: **PASS local gate** với `encoder_blocks_4_11_float`, `cosine_l2_mean = 0.9577`, `cosine_l2_min = 0.9310`.
- Chưa được claim deploy accuracy pass cho tới khi candidate pass QNN compile/link và QNN-vs-PyTorch trên RB3.

---

## 2. Mục tiêu deploy và tiêu chí gate

Mục tiêu là deploy mSigLIP TBPS lên RB3 Gen2 để chạy tìm kiếm người bằng mô tả tiếng Việt trực tiếp trên edge device, không phụ thuộc cloud runtime. Hai encoder cần chạy cục bộ:

| Encoder | Input | Output | Trạng thái |
|---|---|---|---|
| Vision encoder | `image: 1x3x256x256` | `image_embedding: 1x768` | Runtime HTP pass; global QDQ fail; local QDQ candidate pass; QNN pending |
| Text encoder | `input_ids`, `attention_mask` | `text_embedding: 1x768` | Chưa compile, chờ vision QNN fidelity pass |

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

Text encoder đã có ONNX nhưng chưa compile vì vision encoder chưa pass QNN fidelity trên RB3.

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
| `j56vveq6p` | Lite-MP 30% INT16 | `d7jzjy1m2` | `17.2011` | `0.1895` | `0.1509` | `0.2487` | FAIL | Không compile/link |
| `jpe2lnmvp` | Lite-MP 10% FP16 | `d7jzjy1m2` | `20.0953` | `0.3267` | `0.2419` | `0.4402` | FAIL | Không compile/link |

Lite-MP 10% FP16 là AI Hub global quantize-only candidate tốt nhất theo `cosine_l2_mean`, nhưng vẫn thấp rất xa gate `0.95`.

### 4.3 Node-level QDQ sensitivity từ `jpe2lnmvp`

Sau khi global options fail, đã dùng `deployment/scripts/qnn/qdq_surgery.py` để tách lỗi theo loại QDQ và theo block encoder. Tất cả candidate dưới đây chạy gate độc lập `vn3k_test_10`.

| Candidate | QDQ bypass | cosine_l2_mean | cosine_l2_min | cosine_l2_max | Gate | Kết luận |
|---|---|---:|---:|---:|---|---|
| `noop_jpe2lnmvp` | Không bypass, load/save baseline | `0.326745` | `0.241872` | `0.440206` | FAIL | Tooling giữ đúng metric gốc |
| `all_qdq_float` | Toàn bộ QDQ pairs | `0.999687` | `0.999575` | `0.999802` | PASS | Graph mapping đúng |
| `all_weights_float` | QDQ có input là initializer/weight | `0.314436` | `0.217998` | `0.460189` | FAIL | Weight quantization không phải nguyên nhân chính |
| `all_activations_float` | QDQ activation, giữ weight quantized | `0.982455` | `0.958175` | `0.993038` | PASS | Lỗi chính nằm ở activation quantization/range |
| `matmul_gemm_weights_float` | Weight QDQ cho `MatMul`/`Gemm` | `0.320173` | `0.238281` | `0.454809` | FAIL | Không cải thiện |
| `encoder_blocks_0_3_float` | Activation QDQ blocks 0-3 | `0.348658` | `0.271988` | `0.482678` | FAIL | Cải thiện nhẹ |
| `encoder_blocks_4_7_float` | Activation QDQ blocks 4-7 | `0.604363` | `0.534240` | `0.659349` | FAIL | Cải thiện mạnh |
| `encoder_blocks_8_11_float` | Activation QDQ blocks 8-11 | `0.604236` | `0.429726` | `0.773699` | FAIL | Cải thiện mạnh |
| `post_layernorm_head_float` | Activation QDQ post-layernorm/head | `0.306107` | `0.217496` | `0.423946` | FAIL | Không cải thiện |
| `encoder_blocks_4_11_float` | Activation QDQ blocks 4-11 | `0.957671` | `0.931005` | `0.976539` | PASS | Candidate đầu tiên pass QDQ gate local |

Kết luận: lỗi chính không nằm ở weight QDQ. Giữ activation path của encoder blocks 4-11 ở float trong QDQ graph khôi phục fidelity đủ qua gate local, nhưng cần validate lớn hơn và compile/link QNN trước khi xem là deploy candidate thật.

### 4.4 Artifact của các candidate

```text
artifacts/deployment/runtime/job_jgomex415_qdq_onnx/model.onnx
artifacts/deployment/runtime/job_jp2j31dm5_qdq_onnx/model.onnx
artifacts/deployment/runtime/job_j5m4vjxd5_qdq_onnx/model.onnx
artifacts/deployment/runtime/job_jgl7en9l5_qdq_onnx/model.onnx
artifacts/deployment/runtime/job_j56vveq6p_qdq_onnx/model.onnx
artifacts/deployment/runtime/job_jpe2lnmvp_qdq_onnx/model.onnx
artifacts/deployment/runtime/ptq_experiments/node_level/<candidate>/model.onnx
```

Compare outputs:

```text
artifacts/deployment/runtime/ptq_experiments/calib2000/qdq_vs_pytorch_summary.json
artifacts/deployment/runtime/ptq_experiments/calib2000_w8a16/qdq_vs_pytorch_summary.json
artifacts/deployment/runtime/ptq_experiments/calib2000_w8a8_minmax/qdq_vs_pytorch_summary.json
artifacts/deployment/runtime/ptq_experiments/calib2000_lite_mp_default/qdq_vs_pytorch_summary.json
artifacts/deployment/runtime/ptq_experiments/calib2000_lite_mp_30_int16/qdq_vs_pytorch_summary.json
artifacts/deployment/runtime/ptq_experiments/calib2000_lite_mp_10_fp16/qdq_vs_pytorch_summary.json
artifacts/deployment/runtime/ptq_experiments/node_level/<candidate>/qdq_vs_pytorch_summary.json
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
- Lite-MP 30% INT16 đạt `0.1895`, không tốt hơn Lite-MP default.
- Lite-MP 10% FP16 đạt `0.3267`, tốt nhất trong nhóm global quantize-only nhưng vẫn fail xa gate.
- PSNR AI Hub/AIMET cao hơn không đồng nghĩa cosine embedding fidelity tốt hơn.
- QDQ sensitivity decomposition cho thấy `all_activations_float` pass gate còn `all_weights_float` fail, nên activation quantization/range là nguyên nhân chính.
- Block-level isolation xác định encoder blocks 4-11 là vùng nhạy nhất; `encoder_blocks_4_11_float` pass QDQ gate local với `cosine_l2_mean = 0.9577`, `cosine_l2_min = 0.9310`.

### 6.3 Quyết định kỹ thuật

Không compile/link QNN cho các AI Hub global candidate fail QDQ:

```text
jgomex415
jp2j31dm5
j5m4vjxd5
jgl7en9l5
j56vveq6p
jpe2lnmvp
```

Candidate local `encoder_blocks_4_11_float` đã pass QDQ gate trên `vn3k_test_10`, nhưng chưa được compile/link QNN. Không chạy full VN3K R@1, text encoder, hoặc retrieval end-to-end cho tới khi candidate này hoặc candidate tương đương pass QNN-vs-PyTorch trên RB3.

---

## 7. Việc tiếp theo

Plan checklist hiện hành: `deployment/docs/journal/[deploy-plan]-2026-06-06.md`.

### 7.1 Candidate local cần validate: `encoder_blocks_4_11_float`

Candidate local đầu tiên pass QDQ gate:

```text
artifact = artifacts/deployment/runtime/ptq_experiments/node_level/encoder_blocks_4_11_float/
QDQ vs PyTorch cosine_l2_mean = 0.957671
cosine_l2_min/max = 0.931005 / 0.976539
```

Việc nên làm ngay sau report:

- Validate candidate này trên `vn3k_test_100` nếu raw input đã sẵn sàng.
- Nếu vẫn pass, compile/link QNN và chạy RB3 `vn3k_test_10`.
- Nếu QNN compile/link fail do mixed QDQ/float graph, chuyển sang AIMET/local config để encode/exclude activation QDQ cho encoder blocks 4-11 theo cách compile được.

Command compare để validate set lớn hơn:

```bash
venv/bin/python deployment/scripts/qnn/compare_onnx_with_pytorch.py \
  --onnx-model artifacts/deployment/runtime/ptq_experiments/node_level/encoder_blocks_4_11_float \
  --model-dir artifacts/deployment/exports/exported_model \
  --input-dir artifacts/deployment/qnn_inputs/vn3k_test_100 \
  --precision fp32 \
  --json artifacts/deployment/runtime/ptq_experiments/node_level/encoder_blocks_4_11_float/qdq_vs_pytorch_vn3k_test_100_summary.json \
  --csv artifacts/deployment/runtime/ptq_experiments/node_level/encoder_blocks_4_11_float/qdq_vs_pytorch_vn3k_test_100.csv
```

### 7.2 Các candidate global đã đóng

Lite-MP 30% INT16 đã chạy xong với job `j56vveq6p` và fail QDQ gate:

```text
AI Hub/AIMET PSNR = 17.2011
QDQ vs PyTorch cosine_l2_mean = 0.1895
cosine_l2_min/max = 0.1509 / 0.2487
```

Lite-MP 10% FP16 đã chạy xong với job `jpe2lnmvp` và fail QDQ gate nếu giữ global QDQ:

```text
AI Hub/AIMET PSNR = 20.0953
QDQ vs PyTorch cosine_l2_mean = 0.3267
cosine_l2_min/max = 0.2419 / 0.4402
```

### 7.3 Diagnostic phụ

Có thể compare thêm trên calibration set `vn3k_train_calib_2000` để xem candidate `encoder_blocks_4_11_float` có ổn ngay trên data dùng calibrate không:

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

### 7.4 Việc tiếp theo: compile/link có kiểm soát

Các global option đã thử của AI Hub/AIMET vẫn fail QDQ gate, nhưng local sensitivity đã tìm được vùng cần giữ float. Cần chuyển sang hướng quantization pipeline có node-level control:

- Giữ float hoặc override activation encoding cho encoder blocks 4-11.
- Không ưu tiên weight mixed precision trước, vì `all_weights_float` không cải thiện.
- Compile/link chỉ với candidate đã pass QDQ local hoặc config tương đương.
- Sau compile/link, đo QNN-vs-PyTorch trên RB3 `vn3k_test_10` trước khi mở rộng `vn3k_test_100` hoặc retrieval.

---

## 8. Trạng thái cuối báo cáo

| Hạng mục | Trạng thái |
|---|---|
| Export model | DONE |
| Vision ONNX | DONE |
| Text ONNX | DONE |
| Vision QNN HTP runtime | DONE / runtime pass |
| Vision QNN/QDQ fidelity | QDQ local PASS với `encoder_blocks_4_11_float`; QNN pending |
| Calibration 500 | DONE |
| Calibration 2,000 | DONE |
| Dataset ID calib2000 | VERIFIED: `d7jzjy1m2` |
| Compile/link candidate mới | NEXT với `encoder_blocks_4_11_float` hoặc config tương đương |
| RB3 benchmark mở rộng | BLOCKED until QNN fidelity pass |
| Text encoder compile | BLOCKED until vision QNN fidelity pass |
| End-to-end retrieval | BLOCKED until vision QNN + text pass |

Kết luận: deployment hiện đã chứng minh được khả năng chạy vision encoder trên RB3 HTP và đã tìm được một local QDQ mixed-precision candidate pass gate. Công việc tiếp theo là validate/compile-link candidate này và đo QNN fidelity, chưa phải mở rộng benchmark hoặc compile text encoder.
