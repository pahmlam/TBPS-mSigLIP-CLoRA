# [Deploy Plan] 2026-06-14 - Activation-outlier-aware vision QDQ

> **Ngày:** 2026-06-14
> **Thiết bị / Runtime:** RB3 Gen2 / QNN HTP V68 / ONNX Runtime / AI Hub / QNN-QAIRT native
> **Model / Artifact nguồn:** `epoch=56-val_score=52.28.ckpt` → `exported_model` (FP32 baseline) và `exported_model_qat_blocks_6_11_v5`
> **Mục tiêu:** Đổi chiến lược từ "thử nhiều config quantize" sang "chẩn đoán đúng failure mode (activation outlier ở blocks 4-11) rồi áp đúng fix deployable"
> **Plan checklist hiện hành:** file này (thay thế `[deploy-plan]-2026-06-06.md`)
> **Trạng thái:** FOLLOW-UP - P0.1 + P0.2 DONE, verdict = equalization (P1-A); còn P0.3 (server) + P0.4 (HTP capability) trước khi code fix
> **Cập nhật checklist gần nhất:** 2026-06-14

---

## 0. Quan hệ với plan cũ

- `[deploy-plan]-2026-06-06.md` vẫn giữ làm tham chiếu lịch sử cho toàn bộ chuỗi QDQ surgery / ORT / INT16 / QAT v1-v5 / H10 audit. **Không xóa.**
- Từ hôm nay, **file này là plan checklist active duy nhất**. Plan 06-06 không nhận update mới.
- Lý do mở plan mới: các nhánh trong plan cũ (PTQ black-box, `_float` surgery, ORT QDQ, INT16 encoding, QAT proxy) đã hội tụ về cùng một kết luận và cùng một blocker. Cần reframe quanh nguyên nhân gốc thay vì tiếp tục sweep cấu hình.

---

## 1. Tóm tắt quyết định

### Chẩn đoán đã khóa (từ plan 06-06 và journal 06-13/06-14)

- Export/preprocess **đúng**: static ONNX vs PyTorch ≈ `1.000000`. Lỗi không nằm ở export.
- Lỗi nằm ở **quantization của vision encoder**, tập trung ở **activation blocks 4-11**, và là **tương tác weight×activation** (không phải weight-only hay activation-only).
- Bằng chứng định lượng quan trọng: `real_abs_max_mean` blocks 4-11 trung bình `~2075`, lớn nhất `~138717`. Đây là dấu hiệu kinh điển của **activation outlier / massive activations** trong ViT.
- Diagnostic `all_weights + blocks 4-11 float` đạt `0.970312/0.939976` nhưng **không deployable** vì để lại internal float tensor → HTP link fail (`add_1003`, `gelu_10_DequantizeLinear_Output`, `add_103_updated`, exit code 14).
- Fake-quant proxy của QAT **không tương quan** với AI Hub quantizer thật → QAT proxy hiện tại low-value nếu không sửa scheme cho khớp quantizer thật.

### Reframe chiến lược

Failure mode là một bài toán có tên và có công cụ chuẩn. Thay vì sweep tiếp, plan này:

1. **P0 - Chẩn đoán cấu trúc outlier trước** để chọn đúng fix (rẻ, chạy được trên Mac).
2. **P0 - Xác minh năng lực toolchain/HTP thật** (AIMET? QNN-QAIRT native? per-channel activation? all-quantized A16 không internal float?).
3. **P1 - Áp đúng fix deployable**: SmoothQuant/CLE equalization (tạo graph all-INT8) **hoặc** W8A16 calibrate đúng cách — không dùng `_float` surgery.
4. **P4 song song - ORT FP16 trên RB3** để có số latency/RAM thật trên board ngay cả khi HTP còn kẹt.

### Điều chưa được phép claim / mở rộng

- Không compile/link bất kỳ `_float` surgery candidate nào (đã link fail lặp lại).
- Không upload lại ORT W8A16 / INT16 QDQ cùng pattern đã link fail.
- Không chạy thêm QAT proxy nếu chưa chứng minh proxy khớp quantizer thật.
- Không chạy full VN3K R@1 / text encoder / retrieval end-to-end trước khi vision QNN fidelity pass trên RB3.

---

## 2. Gate bắt buộc

| Gate | Ngưỡng | Ý nghĩa |
|---|---:|---|
| Static ONNX vs PyTorch | `cosine_l2_mean >= 0.999` | Control export/preprocess |
| QDQ ONNX vs PyTorch mean | `cosine_l2_mean >= 0.95` | Candidate đáng compile/link |
| QDQ ONNX vs PyTorch min | `cosine_l2_min >= 0.90` | Không sample lệch nặng |
| QNN vs PyTorch sau link | `cosine_l2_mean >= 0.90` | Đáng benchmark rộng trên RB3 |
| Full retrieval | `T2I R@1 >= 48.0` | Mục tiêu deploy so với `52.28` FP32 baseline |

Diagnostic compile exception: chỉ cho QDQ do AI Hub-native/QNN-native quantizer tạo, đạt `mean >= 0.93` và `min >= 0.88`, **và là all-quantized deployable** (không internal float). Không áp dụng cho `_float`, ORT QDQ, INT16 surgery đã fail link.

---

## 3. Checklist tổng của kế hoạch

### Phase P0 - Pre-flight diagnosis (BẮT BUỘC trước mọi fix)

Mục tiêu P0 là **thu hẹp solution space bằng dữ kiện**, không đoán. Tất cả P0 ưu tiên chạy được trên Mac local trừ P0.3/P0.4 cần server.

- [x] **P0.1 - Profile cấu trúc activation outlier blocks 4-11.** DONE (Mac, FP32 ONNX, `vn3k_test_10`).
  - Script: `deployment/scripts/qnn/profile_activation_outliers.py`.
  - Output: `artifacts/deployment/runtime/diag/activation_outliers/blocks_4_11/{summary.json,per_tensor.csv}`.
  - **Verdict: `CONCENTRATED_FIXED`.** Outlier nằm ở vài channel **cố định** trong residual stream; recurring channel set `523/415/7/528` xuyên blocks 9-11 (stability 0.75-1.00). Worst tensor `val_473` (blk5): abs_max `5224`, concentration `876×`, nhưng p99.99 chỉ `34.5` → đúng dạng "massive activations" → equalization-amenable.
  - Phụ: vài late-block residual (`val_947` blk11, `val_868` blk10) là DIFFUSE (concentration ~4-6×, p99.99 ~2200) → vấn đề phụ, không phải nguyên nhân collapse chính.
- [x] **P0.2 - Re-audit kết quả W8A16 = 0.155.** DONE (Mac, dùng `analyze_qdq_encodings.py` trên artifact có sẵn).
  - Output: `artifacts/deployment/runtime/diag/p0_2_w8a16_audit/{w8a16,int8_litemp}_encodings.{json,csv}`.
  - **Kết luận: lỗi là GRANULARITY, không phải bit-width.** Cả Lite-MP INT8 và native W8A16 đều dùng **activation per-tensor (20/20 per-tensor, 0 per-channel)**; `real_abs_max` tới `138718`. Per-tensor scale bị 1 outlier chi phối → giá trị thường (~6) chỉ còn 0 level (INT8) hoặc ~3 level (A16). W8A16 còn TỆ hơn INT8 vì pure-W8A16 quantize cả các tensor mà Lite-MP để FP16.
  - → Bác bỏ "A16 bất lực"; xác nhận "phải hạ per-tensor activation range" = SmoothQuant/equalization.
- [ ] **P0.3 - Audit toolchain trên server (không phải Mac).**
  - Chạy `deployment/scripts/qnn/audit_qnn_native_env.py` trên server.
  - Kiểm tra song song khả năng cài **AIMET** (CLE + bias correction + AdaRound) trên server: Python/glibc/wheel phù hợp.
  - Ghi version QNN/QAIRT/AIMET, OS, Python, command path vào journal ngày chạy.
- [ ] **P0.4 - Xác minh năng lực HTP cho graph deployable.** (đã sắc hơn nhờ P0.2: activation hiện là per-tensor)
  - **Câu hỏi chính cho P1-A:** HTP/AI Hub có nhận **per-channel WEIGHT quant** không? (gần như chắc có; SmoothQuant chỉ cần per-channel weight + per-tensor activation → nếu đúng thì P1-A deployable luôn, không cần internal float.)
  - Bonus: HTP có hỗ trợ **per-channel/per-axis activation** deployable không? (nếu có, là fix thẳng cho fixed-channel outlier mà không cần đụng weight.)
  - HTP có hỗ trợ **all-quantized W8A16** (no internal float) link thành context binary không? (chỉ cần nếu P1-A residual diffuse vẫn fail.)
  - QNN-native quantizer có cho **override encoding/range theo tensor/op** không?
  - Output: bảng "supported / not supported" để chốt branch P1.

### Phase P1 - Áp fix theo kết quả P0 (chọn nhánh, không chạy hết)

- [ ] **P1-A SmoothQuant / equalization (ưu tiên nếu P0.1 cho thấy outlier tập trung ít channel).**
  - Migrate outlier magnitude từ activation sang weight bằng per-channel scale `s`, sao cho graph cuối **all-INT8 deployable**, không có float path.
  - Áp cho blocks 4-11 (mở rộng nếu cần theo P0.1).
  - Không cần retrain; là post-training equalization.
- [ ] **P1-B AIMET CLE + bias correction (+ AdaRound) (nếu P0.3 cho AIMET chạy được trên server).**
  - CLE để cân bằng range cross-layer, bias correction để bù lệch, AdaRound nếu cần weight rounding tốt hơn.
  - Đây là công cụ nhắm trực tiếp vào failure mode; ưu tiên cao nếu môi trường cho phép.
- [ ] **P1-C W8A16 calibrate đúng cách (nếu P0.4 xác nhận A16 deployable trên HTP).**
  - Percentile calibration (99.99) thay vì min-max, per-channel weights, đảm bảo activation blocks 4-11 không bị 1 outlier chi phối.
  - Bắt buộc all-quantized, verify no internal float trước khi nghĩ tới compile.

### Phase P2 - Validate fidelity trước compile

- [ ] Chạy `compare_onnx_with_pytorch.py` trên `vn3k_test_10` cho candidate P1.
- [ ] Chỉ mở `vn3k_test_100` nếu `vn3k_test_10` pass/near-pass gate.
- [ ] Xác minh candidate là **all-quantized deployable** (script kiểm tra không còn float tensor nội bộ) trước khi compile.

### Phase P3 - Compile/link khi candidate deployable pass gate

- [ ] Compile/link chỉ khi candidate all-quantized đạt production gate hoặc diagnostic exception.
- [ ] Tải context binary, chạy `qnn-net-run` trên RB3 với `vn3k_test_10`.
- [ ] `compare_qnn_with_pytorch.py`; chỉ mở `vn3k_test_100` nếu QNN mean `>= 0.90`.

### Phase P4 - Fallback runtime song song (de-risk chapter deploy)

Chạy song song với P0-P3, không chờ HTP.

- [ ] Export/đóng gói ONNX Runtime **FP16** vision encoder cho ARM64.
- [ ] Chạy trên RB3 (CPU và/hoặc GPU), đo latency, FPS, RAM peak thật.
- [ ] Xác minh output 768-d finite, L2-normalized.
- [ ] Ghi số thật vào journal — đây là baseline deploy tối thiểu nếu HTP không kịp.

### Phase P5 - Sau khi vision pass (giữ nguyên từ plan cũ)

- [ ] Compile text encoder theo cùng nguyên tắc: QDQ compare trước, QNN runtime sau.
- [ ] Đo end-to-end image/text retrieval trên board.
- [ ] Cập nhật benchmark report và viết guide deploy tái lập.

---

## 4. Cần check trước tiên (thứ tự thực thi P0)

1. **P0.1 outlier profiling** — rẻ nhất, chạy ngay trên Mac, quyết định P1-A vs P1-C. Làm đầu tiên.
2. **P0.2 W8A16 re-audit** — dùng artifact đã có, không tốn job mới. Làm cùng P0.1.
3. **P0.4 HTP capability** — chốt branch nào deployable; cần đọc doc QNN/AI Hub hoặc thử nhỏ.
4. **P0.3 server toolchain** — cần truy cập server; quyết định P1-B có khả thi không.

Quy tắc: **không viết code fix P1 cho tới khi P0.1 + P0.4 có kết luận.** P0.1 nói "fix gì", P0.4 nói "fix đó có deployable không".

---

## 5. Command / Quy trình đang mở

### P0.1 - profiler outlier (script mới đề xuất)

```bash
# Script mới: deployment/scripts/qnn/profile_activation_outliers.py (chưa tạo)
venv/bin/python deployment/scripts/qnn/profile_activation_outliers.py \
  --onnx-model artifacts/deployment/exports/exported_model/vision_onnx \
  --input-dir artifacts/deployment/qnn_inputs/vn3k_test_10 \
  --blocks 4-11 \
  --percentiles 99.9 99.99 \
  --json artifacts/deployment/runtime/diag/activation_outliers/blocks_4_11/summary.json \
  --csv  artifacts/deployment/runtime/diag/activation_outliers/blocks_4_11/per_channel.csv
```

### P0.3 - audit toolchain trên server

```bash
python deployment/scripts/qnn/audit_qnn_native_env.py \
  --json artifacts/deployment/runtime/qnn_native/env_audit_server.json
```

### P2 - validate candidate (giữ pattern cũ)

```bash
venv/bin/python deployment/scripts/qnn/compare_onnx_with_pytorch.py \
  --onnx-model artifacts/deployment/runtime/<candidate_qdq_dir> \
  --model-dir artifacts/deployment/exports/exported_model \
  --input-dir artifacts/deployment/qnn_inputs/vn3k_test_10 \
  --precision fp32 \
  --json <candidate>/qdq_vs_pytorch_summary.json \
  --csv  <candidate>/qdq_vs_pytorch.csv
```

---

## 6. Thứ tự ưu tiên

1. **P0.1 + P0.2** (Mac, không tốn AI Hub job): biết outlier tập trung hay phân tán, và W8A16 fail vì calibration hay vì bản chất.
2. **P0.4** (capability HTP): chốt nhánh deployable.
3. **P0.3** (server): mở khả năng AIMET/QNN-native.
4. **P1** nhánh được P0 chỉ định (A hoặc B hoặc C) — chỉ một nhánh trước, không sweep song song.
5. **P4 fallback ORT FP16** chạy nền để luôn có số board thật.
6. Nếu cả P1-A/B/C đều fail gate: dừng quantize HTP, chốt deploy bằng ORT FP16 (P4) và ghi rõ giới hạn.

---

## 7. Việc KHÔNG làm trong plan này

- Không compile/link `all_weights_plus_blocks_4_11` hay bất kỳ `_float` surgery nào.
- Không upload lại ORT W8A16 / INT16 QDQ cùng pattern đã link fail.
- Không chạy thêm QAT proxy trừ khi P0 chứng minh proxy khớp quantizer thật.
- Không sweep cấu hình quantize "mù" — mọi job mới phải gắn với một giả thuyết từ P0.
- Không mở full retrieval / text encoder trước khi vision QNN fidelity pass.

---

## 8. Rủi ro / câu hỏi mở

- **R1:** Nếu outlier phân tán nhiều channel (P0.1) và HTP không hỗ trợ A16 deployable (P0.4), solution space INT-only có thể trống → buộc về P4 fallback. Cần biết sớm.
- **R2:** AIMET có thể vẫn không cài được trên server (đã từng fail Python/glibc/Docker ở plan cũ). P1-B phụ thuộc P0.3.
- **R3:** SmoothQuant migrate outlier sang weight có thể đẩy weight ra ngoài range INT8 per-channel; cần kiểm tra weight range sau equalization.
- **R4:** ORT FP16 trên RB3 có thể không đạt latency/RAM mong muốn; cần đo thật, không giả định.
- **R5:** Mọi số trong plan này là kết quả đã cite từ journal trước; candidate mới phải đo lại, không tái dùng số cũ làm bằng chứng pass.
