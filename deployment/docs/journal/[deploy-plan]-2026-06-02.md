# Kế hoạch tiếp theo cho deploy mSigLIP lên RB3 Gen2

> **Ngày:** 2026-06-02
> **Thiết bị mục tiêu:** Qualcomm RB3 Gen2 / QCS6490 / HTP V68
> **Model nguồn:** `epoch=56-val_score=52.28.ckpt`
> **Trạng thái:** Vision encoder chạy được trên HTP, nhưng INT8 PTQ/QDQ làm hỏng embedding.

> **Ghi chú:** File này là kế hoạch/journal lịch sử, đã được supersede bởi `deployment/docs/journal/[deploy-plan]-2026-06-06.md`.
> **Trạng thái:** SUPERSEDED, không còn là plan checklist hiện hành.

---

## 1. Kết luận hiện tại

Pipeline runtime đã pass: QNN context binary `vision_encoder_calib500.bin` load và chạy được trên RB3 HTP. Vấn đề hiện tại là fidelity sau quantization, không phải ONNX export hay link job.

Kết quả đã xác minh trên `vn3k_test_10`:

| So sánh | cosine_l2_mean | Kết luận |
|---|---:|---|
| Static ONNX vs PyTorch | `1.0000` | ONNX export và raw input đúng |
| QDQ ONNX vs PyTorch | `0.1682` | PTQ/QDQ đã làm embedding lệch mạnh |
| QNN calib500 vs PyTorch | `0.1300` | QNN runtime thấp hơn, nhưng lỗi đã xuất hiện trước QNN |

Quyết định kỹ thuật:

- Không compile text encoder lúc này.
- Không chạy `vn3k_test_100` hoặc full VN3K R@1 với binary hiện tại.
- Không chạy lại link job `jgkdevqyp` nếu QDQ ONNX vẫn fail.
- Tập trung sửa PTQ/QDQ fidelity của vision encoder trước.

---

## 2. Gate bắt buộc trước khi đi tiếp

Một candidate quantization mới chỉ được compile/link sang QNN nếu đạt gate QDQ trước:

| Gate | Ngưỡng tạm thời | Ý nghĩa |
|---|---:|---|
| Static ONNX vs PyTorch | `cosine_l2_mean >= 0.999` | Control phải luôn pass |
| QDQ ONNX vs PyTorch smoke | `cosine_l2_mean >= 0.95` | Đủ tốt để thử compile/link |
| QDQ ONNX vs PyTorch min | `cosine_l2_min >= 0.90` | Không có mẫu lệch nặng |
| QNN vs PyTorch sau link | `cosine_l2_mean >= 0.90` | Đủ để mở rộng `vn3k_test_100` |
| Full retrieval | `T2I R@1 >= 48.0` | Mục tiêu deploy, so với FP32 baseline `52.28` |

Các ngưỡng QDQ/QNN là gate thực dụng để tránh tốn thời gian compile/link khi quantized model đã sai rõ. Có thể siết lại sau khi có candidate đầu tiên pass.

---

## 3. Trạng thái lịch sử của kế hoạch 2026-06-02

Phần này giữ lại trạng thái đã ghi nhận của kế hoạch cũ để đối chiếu lịch sử. Plan checklist hiện hành nằm ở `deployment/docs/journal/[deploy-plan]-2026-06-06.md`.

### Phase A — Cố định baseline và dữ liệu kiểm tra

- DONE: Chạy static ONNX vs PyTorch trên `vn3k_test_10`.
- DONE: Chạy QDQ ONNX vs PyTorch trên `vn3k_test_10`.
- DONE: Lưu summary/csv:
  - `artifacts/deployment/qnn_outputs/vn3k_test_10_calib500/static_onnx_vs_pytorch_summary.json`
  - `artifacts/deployment/qnn_outputs/vn3k_test_10_calib500/qdq_vs_pytorch_summary.json`
- DONE: Tạo một folder riêng cho các thử nghiệm quantization tiếp theo:
  - `artifacts/deployment/runtime/ptq_experiments/`
- DONE: Với các thử nghiệm đã chạy, lưu đủ:
  - QDQ ONNX hoặc model ID
  - quantize job ID
  - calibration dataset ID
  - QDQ-vs-PyTorch JSON/CSV
  - quyết định pass/fail
- TODO: Với candidate tiếp theo, tiếp tục lưu đủ log job, QDQ ONNX, JSON/CSV compare, và quyết định pass/fail.

### Phase B — Kiểm tra calibration data trước khi thử quantize mới

- DONE: Xác minh calibration raw có đúng shape và dtype: `1x3x256x256 float32`.
- DONE: Xác minh preprocessing calibration giống inference:
  - RGB
  - resize/crop theo SigLIP 256
  - normalize đúng như export/inference
- DONE: Chạy thống kê calibration set hiện tại `vn3k_train_calib_500`:
  - min/max/mean/std toàn tập
  - phát hiện raw bất thường
  - kiểm tra không có file rỗng hoặc sai byte size
- DONE: So sánh phân phối raw giữa:
  - `vn3k_train_calib_500`
  - `vn3k_test_10`
- DONE: Mở rộng kiểm tra thêm `vn3k_train_calib_2000`.
- DONE: Kết luận chưa thấy lỗi raw byte size, dtype, NaN/Inf, hoặc normalize range.
- DONE: Nếu calibration set hiện tại ổn, dùng nó làm baseline PTQ fail case.

### Phase C — Thử calibration set đại diện hơn

Mục tiêu: kiểm tra liệu lỗi do calibration set 500 mẫu chưa đủ đại diện hay do cơ chế PTQ không phù hợp với SigLIP embedding.

- DONE: Tạo calibration set 2,000 ảnh train VN3K, random seed `2400`.
- DONE: Upload dataset mới lên QAI Hub.
- DONE: Xác minh dataset ID `d7jzjy1m2` là `msiglip-vision-vn3k-train-calib-2000`.
- DONE: Submit quantize-only W8A8 calib2000: job `jgomex415`.
- DONE: Tải QDQ ONNX của `jgomex415` về.
- DONE: Chạy QDQ ONNX vs PyTorch cho `jgomex415`: `cosine_l2_mean = 0.1692`, FAIL.
- DONE: Không compile/link `jgomex415` vì `cosine_l2_mean < 0.95`.
- DONE: Kết luận tăng calibration từ 500 lên 2,000 mẫu không đủ sửa QDQ fidelity.
- TODO: Chỉ cân nhắc calib 5,000 hoặc sampling cân bằng nếu có bằng chứng mới cho thấy calibration coverage vẫn là nghi vấn chính.

Command mẫu tạo calibration set:

```bash
venv/bin/python deployment/scripts/qnn/prepare_vn3k_vision_inputs.py \
  --dataset-root VN3K \
  --split train \
  --selection random \
  --seed 2400 \
  --num-samples 2000 \
  --output-dir artifacts/deployment/qnn_inputs/vn3k_train_calib_2000 \
  --path-mode relative
```

Command upload:

```bash
venv/bin/python deployment/scripts/qnn/upload_qaihub_calibration_dataset.py \
  --input-dir artifacts/deployment/qnn_inputs/vn3k_train_calib_2000 \
  --name msiglip-vision-vn3k-train-calib-2000
```

### Phase D — Thử quantization config khác

Mục tiêu: giảm sai lệch tại QDQ ONNX trước khi quay lại QNN runtime.

- DONE: Kiểm tra hướng option chính thức có thể thử tiếp: `--range_scheme` và `--lite_mp`.
- DONE: Không dùng option chưa xác nhận trong log hoặc docs chính thức.
- DONE: Với mỗi config đã thử, chỉ chạy đến QDQ compare trước.
- DONE: Chỉ compile/link khi QDQ gate pass.
- DONE: Submit quantize-only W8A16 calib2000: job `jp2j31dm5`.
- DONE: Tải QDQ ONNX của `jp2j31dm5` về.
- DONE: Chạy QDQ-vs-PyTorch cho `jp2j31dm5`: `cosine_l2_mean = 0.1863`, FAIL.
- DONE: Không compile/link `jp2j31dm5` vì QDQ gate fail.
- DONE: Chạy W8A8 + `--range_scheme min_max`, quantize-only: job `j5m4vjxd5`.
- DONE: Tải QDQ ONNX của `j5m4vjxd5` về.
- DONE: Chạy QDQ-vs-PyTorch cho `j5m4vjxd5`: `cosine_l2_mean = 0.1658`, FAIL.
- DONE: Không compile/link `j5m4vjxd5` vì QDQ gate fail.
- DONE: Chạy Lite-MP default, quantize-only: job `jgl7en9l5`.
- DONE: Tải QDQ ONNX của `jgl7en9l5` về.
- DONE: Chạy QDQ-vs-PyTorch cho `jgl7en9l5`: `cosine_l2_mean = 0.1906`, FAIL.
- DONE: Không compile/link `jgl7en9l5` vì QDQ gate fail.
- DONE: Chạy Lite-MP 30% INT16, quantize-only: job `j56vveq6p`.
- DONE: Tải QDQ ONNX của `j56vveq6p` về.
- DONE: Chạy QDQ-vs-PyTorch cho `j56vveq6p`: `cosine_l2_mean = 0.1895`, FAIL.
- DONE: Không compile/link `j56vveq6p` vì QDQ gate fail.
- DONE: Chạy Lite-MP 10% FP16, quantize-only: job `jpe2lnmvp`.
- DONE: Tải QDQ ONNX của `jpe2lnmvp` về.
- DONE: Chạy QDQ-vs-PyTorch cho `jpe2lnmvp`: `cosine_l2_mean = 0.3267`, FAIL.
- DONE: Không compile/link `jpe2lnmvp` vì QDQ gate fail.
- TODO: Chuyển sang pipeline quantization khác có khả năng exclude/mixed precision theo node/op.

Ghi chú: log `jpr9v62vp` có nhiều warning kiểu `wrong signed offset` và `Following OPs fallback to float`. Các warning này cần được ghi lại theo từng config vì có thể liên quan trực tiếp đến fidelity.

### Phase E — Compile/link lại khi QDQ pass

Chỉ thực hiện phase này nếu QDQ ONNX candidate đạt gate.

- TODO: Submit compile/link bằng `submit_qaihub_quantize_compile.py` hoặc flow tương đương.
- TODO: Tải QNN context binary về `artifacts/deployment/qnn_inputs/`.
- TODO: Chạy `vn3k_test_10` trên RB3 bằng `qnn-net-run`.
- TODO: Sync output về local.
- TODO: Chạy QNN-vs-PyTorch compare.
- TODO: Nếu QNN mean >= 0.90, mở rộng sang `vn3k_test_100`.
- TODO: Nếu `vn3k_test_100` ổn, chạy full VN3K retrieval/R@1.

### Phase F — Chỉ sau khi vision pass

- TODO: Compile text encoder theo cùng nguyên tắc: QDQ compare trước, QNN runtime sau.
- TODO: Đo end-to-end image/text retrieval trên board.
- TODO: Cập nhật benchmark report.
- TODO: Viết guide deploy tái lập.

---

## 4. Command kiểm tra lại QDQ candidate

Các candidate global option hiện tại của AI Hub/AIMET đều đã fail QDQ gate. Candidate tốt nhất là Lite-MP 10% FP16 `jpe2lnmvp` với `cosine_l2_mean = 0.3267`, vẫn thấp xa ngưỡng `0.95`.

Với mỗi QDQ ONNX mới tải về:

```bash
venv/bin/python deployment/scripts/qnn/compare_onnx_with_pytorch.py \
  --onnx-model artifacts/deployment/runtime/<qdq_onnx_dir> \
  --model-dir artifacts/deployment/exports/exported_model \
  --input-dir artifacts/deployment/qnn_inputs/vn3k_test_10 \
  --precision fp32 \
  --json artifacts/deployment/runtime/ptq_experiments/<experiment_name>/qdq_vs_pytorch_summary.json \
  --csv artifacts/deployment/runtime/ptq_experiments/<experiment_name>/qdq_vs_pytorch.csv
```

Nếu kết quả vẫn dưới gate `cosine_l2_mean >= 0.95` và `cosine_l2_min >= 0.90`, dừng ở QDQ và không chạy compile/link.

---

## 5. Thứ tự ưu tiên

1. Chuyển sang pipeline có node-level mixed precision/exclude op.
2. Ưu tiên giữ float/mixed precision cho LayerNorm, projection/head cuối, L2-normalization path, và attention/softmax score path nếu toolchain cho phép.
3. Có thể chạy diagnostic phụ trên `vn3k_train_calib_2000`, nhưng không thay gate độc lập `vn3k_test_10`.
4. Chỉ khi QDQ pass mới quay lại compile/link QNN.
5. Sau vision pass mới làm text encoder và retrieval end-to-end.

---

## 6. Trạng thái tổng đã ghi nhận

| # | Việc cần làm | Trạng thái |
|---|---|---|
| 1 | Static ONNX control | DONE |
| 2 | QDQ ONNX diagnostic | DONE |
| 3 | Xác định lỗi chính nằm ở PTQ/QDQ | DONE |
| 4 | Kiểm tra calibration raw/preprocess | DONE |
| 5 | Tạo và upload calibration 2,000 mẫu | DONE |
| 6 | W8A8 calib2000 `jgomex415` | DONE / FAIL QDQ |
| 7 | W8A16 calib2000 `jp2j31dm5` | DONE / FAIL QDQ |
| 8 | W8A8 + min_max `j5m4vjxd5` | DONE / FAIL QDQ |
| 9 | Lite-MP default `jgl7en9l5` | DONE / FAIL QDQ |
| 10 | Lite-MP 30% INT16 `j56vveq6p` | DONE / FAIL QDQ |
| 11 | Lite-MP 10% FP16 `jpe2lnmvp` | DONE / FAIL QDQ |
| 12 | Node-level mixed precision/exclude op pipeline | TODO |
| 13 | Compile/link QNN candidate mới | BLOCKED until QDQ pass |
| 14 | RB3 `vn3k_test_10` candidate mới | BLOCKED until QDQ pass |
| 15 | Text encoder compile | BLOCKED until vision pass |
