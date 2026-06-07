# [Deploy Plan] 2026-06-06 - Node-level mixed precision cho vision QDQ

> **Ngày lập kế hoạch:** 2026-06-06
> **Thiết bị mục tiêu:** Qualcomm RB3 Gen2 / QCS6490 / HTP V68
> **Model nguồn:** `epoch=56-val_score=52.28.ckpt`
> **Artifact nguồn hiện tại:** `artifacts/deployment/runtime/job_jpe2lnmvp_qdq_onnx/model.onnx`
> **Calibration dataset:** `d7jzjy1m2` / `msiglip-vision-vn3k-train-calib-2000`
> **Plan checklist hiện hành:** file này
> **Trạng thái:** FOLLOW-UP - local QDQ pass nhưng `_float` candidate không deployable trên HTP, chuyển sang H5 AIMET/QAT
> **Cập nhật checklist gần nhất:** 2026-06-07

---

## 1. Tóm tắt quyết định

Vision encoder đã chứng minh được runtime path trên RB3 HTP, nhưng QDQ fidelity vẫn fail. Các global options của AI Hub/AIMET đã thử đều không đạt gate.

Best global quantize-only candidate trước node surgery:

| Candidate | Config | PSNR | cosine_l2_mean | cosine_l2_min | cosine_l2_max | Gate | Quyết định |
|---|---|---:|---:|---:|---:|---|---|
| `jpe2lnmvp` | Lite-MP 10% FP16 | `20.0953` | `0.3267` | `0.2419` | `0.4402` | FAIL | Không compile/link |

Kết quả node-level surgery đơn giản đã chạy nhanh trên `jpe2lnmvp`: best simple candidate là `all_layernorm_float` với `cosine_l2_mean = 0.3288`, chỉ nhích rất nhẹ so với no-op `0.3267` và vẫn fail xa gate.

Sau đó đã chạy QDQ sensitivity decomposition. Kết luận mới: lỗi chính nằm ở **activation quantization/range**, không phải weight quantization. Candidate `encoder_blocks_4_11_float` đã pass QDQ gate local trên `vn3k_test_10`:

```text
cosine_l2_mean = 0.957671
cosine_l2_min  = 0.931005
cosine_l2_max  = 0.976539
```

Ngày 2026-06-07 đã validate `encoder_blocks_4_11_float` trên `vn3k_test_100`: `cosine_l2_mean = 0.955255`, `cosine_l2_min = 0.900515`, vẫn pass nhưng margin mỏng. Compile job `jgd0zw76p` tạo được DLC, nhưng link job `jgj1wxo1g` fail vì tensor nội bộ `add_1003` còn floating-point và HTP không hỗ trợ. Vì vậy các candidate `_float` chỉ được xem là diagnostic upper-bound, không phải deployable candidate.

Quyết định tiếp theo: không compile/link thêm candidate `_float`; chuyển sang H5 để tạo all-quantized deployable config bằng AIMET/local config hoặc QAT.

---

## 2. Gate bắt buộc

| Gate | Ngưỡng | Ý nghĩa |
|---|---:|---|
| Static ONNX vs PyTorch | `cosine_l2_mean >= 0.999` | Control export/preprocess phải pass |
| QDQ ONNX vs PyTorch mean | `cosine_l2_mean >= 0.95` | Đủ tốt để thử compile/link QNN |
| QDQ ONNX vs PyTorch min | `cosine_l2_min >= 0.90` | Không có sample lệch nặng |
| QNN vs PyTorch sau link | `cosine_l2_mean >= 0.90` | Đủ để mở rộng benchmark trên RB3 |
| Full retrieval | `T2I R@1 >= 48.0` | Mục tiêu deploy so với baseline `52.28%` |

Chỉ chạy `vn3k_test_100` cho candidate đã pass QDQ local. Không chạy full VN3K R@1, text encoder, hoặc retrieval end-to-end cho đến khi vision QNN fidelity pass trên RB3.

---

## 3. Context đã khóa

| Hạng mục | Trạng thái |
|---|---|
| Vision ONNX static vs PyTorch | PASS, `cosine_l2_mean = 1.0000` |
| QDQ calib500 vs PyTorch | FAIL, `cosine_l2_mean = 0.1682` |
| QNN calib500 vs PyTorch | FAIL, `cosine_l2_mean = 0.1300` |
| Raw input audit | PASS, đúng `786432` bytes/file, không NaN/Inf, range `[-1, 1]` |
| Calib2000 dataset ID | VERIFIED: `d7jzjy1m2` |
| Best global option | `jpe2lnmvp`, Lite-MP 10% FP16, FAIL với `0.3267` |
| Best simple node-level local surgery | `all_layernorm_float`, FAIL với `cosine_l2_mean = 0.3288` |
| Best sensitivity candidate | `encoder_blocks_4_11_float`, PASS QDQ local trên `vn3k_test_100` với `cosine_l2_mean = 0.955255`, `cosine_l2_min = 0.900515`; link HTP FAIL vì internal float tensor |
| Compile/link `_float` candidate | `jgd0zw76p` compile DLC PASS, `jgj1wxo1g` link FAIL với `Tensor 'add_1003' has a floating-point type` |

Các kết quả cũ vẫn nằm trong daily journal/report:

```text
deployment/docs/journal/[deploy]-2026-06-02.md
deployment/docs/journal/[deploy]-2026-06-04.md
deployment/docs/journal/[deploy]-2026-06-05.md
deployment/docs/journal/[deploy]-2026-06-06.md
deployment/docs/[deploy]-report-1.md
```

---

## 4. Kết quả node-level local surgery

Các candidate dưới đây được tạo từ `artifacts/deployment/runtime/job_jpe2lnmvp_qdq_onnx/model.onnx` bằng `deployment/scripts/qnn/qdq_surgery.py`, chạy gate độc lập `vn3k_test_10`.

| Candidate | QDQ bypass | cosine_l2_mean | cosine_l2_min | cosine_l2_max | Kết luận |
|---|---|---:|---:|---:|---|
| `noop_jpe2lnmvp` | Không bypass, load/save baseline | `0.326745` | `0.241872` | `0.440206` | Tooling giữ đúng metric gốc |
| `output_float` | Final output QDQ | `0.326745` | `0.241872` | `0.440206` | Không cải thiện |
| `final_head_float` | Head cuối: `node_linear_74`, `node_layer_norm_25`, MLP/GELU, `node_add_1219`, `node_select_2` | `0.314097` | `0.224059` | `0.433362` | Tệ hơn |
| `all_layernorm_float` | Tất cả `LayerNormalization` + output | `0.328831` | `0.258684` | `0.428104` | Tốt nhất nhưng cải thiện rất nhỏ |
| `attention_score_float` | QDQ trước/sau `Softmax` attention score path + output | `0.305198` | `0.243659` | `0.423269` | Tệ hơn |
| `combined_layernorm_final_head` | `all_layernorm_float` + `final_head_float` | `0.316313` | `0.251170` | `0.416648` | Tệ hơn |

Artifact:

```text
artifacts/deployment/runtime/ptq_experiments/node_level/<candidate>/
```

Mỗi thư mục có `model.onnx`, `model.data`, `qdq_surgery_summary.json`, `qdq_vs_pytorch_summary.json`, và `qdq_vs_pytorch.csv`.

Kết luận vòng simple surgery: bypass QDQ ở vài nhóm node nhạy cảm theo graph-level surgery chưa đủ. Vì no-op giữ đúng metric và các candidate đều không có NaN/Inf, lỗi fidelity không đến từ bước load/save ONNX.

### 4.2 QDQ sensitivity decomposition

Các candidate dưới đây dùng cùng nguồn `jpe2lnmvp`, chạy bằng `qdq_surgery.py` sau khi mở rộng preset H1/H4. Gate vẫn là `vn3k_test_10`.

| Candidate | QDQ bypass | cosine_l2_mean | cosine_l2_min | cosine_l2_max | Kết luận |
|---|---|---:|---:|---:|---|
| `all_qdq_float` | Bypass toàn bộ QDQ pairs | `0.999687` | `0.999575` | `0.999802` | PASS, graph mapping đúng |
| `all_weights_float` | Bypass QDQ có input là initializer/weight | `0.314436` | `0.217998` | `0.460189` | FAIL, weight quantization không phải nguyên nhân chính |
| `all_activations_float` | Bypass QDQ activation, giữ weight quantized | `0.982455` | `0.958175` | `0.993038` | PASS, lỗi chính nằm ở activation quantization/range |
| `matmul_gemm_weights_float` | Bypass QDQ weight cho `MatMul`/`Gemm` | `0.320173` | `0.238281` | `0.454809` | FAIL |
| `encoder_blocks_0_3_float` | Bypass activation QDQ trong blocks 0-3 | `0.348658` | `0.271988` | `0.482678` | FAIL, cải thiện nhẹ |
| `encoder_blocks_4_7_float` | Bypass activation QDQ trong blocks 4-7 | `0.604363` | `0.534240` | `0.659349` | FAIL, cải thiện mạnh |
| `encoder_blocks_8_11_float` | Bypass activation QDQ trong blocks 8-11 | `0.604236` | `0.429726` | `0.773699` | FAIL, cải thiện mạnh |
| `post_layernorm_head_float` | Bypass activation QDQ ở post-layernorm/head | `0.306107` | `0.217496` | `0.423946` | FAIL |
| `encoder_blocks_4_11_float` | Bypass activation QDQ trong blocks 4-11 | `0.957671` | `0.931005` | `0.976539` | PASS QDQ gate local |

Kết luận mới: nếu giữ activation path của encoder blocks 4-11 ở float trong QDQ graph, model khôi phục fidelity đủ qua gate local. Đây là candidate đầu tiên sau các run calib2000 đạt QDQ gate, nhưng mới là local ONNX/QDQ result; chưa phải QNN compile/link pass.

### 4.3 Validate 2026-06-07 trên `vn3k_test_100` và compile/link

`vn3k_test_100` được tạo từ VN3K test split với cùng preprocess QNN raw input. Raw audit pass: 100/100 file hợp lệ, đúng `786432` bytes/file, không NaN/Inf, range `[-1, 1]`.

| Candidate / Job | Mục tiêu | Kết quả | Kết luận |
|---|---|---:|---|
| `encoder_blocks_4_11_float` trên `vn3k_test_100` | Validate candidate pass set lớn hơn | mean `0.955255`, min `0.900515`, max `0.978287` | PASS QDQ local, margin mỏng |
| Compile job `jgd0zw76p` | Compile QDQ ONNX đã surgery sang optimized DLC | SUCCESS, tạo `job_jgd0zw76p_optimized_dlc_mqv7g0yjq.dlc` | DLC tạo được, nhưng nhiều op fallback float |
| Link job `jgj1wxo1g` | Link DLC thành QNN context binary HTP | FAIL: `Tensor 'add_1003' has a floating-point type` | Không có deployable `.bin`; không chạy RB3 |

Kết luận: `_float` surgery là diagnostic upper-bound, không phải hướng deploy trực tiếp. HTP context binary cần graph không còn floating-point tensor nội bộ.

### 4.4 Refinement sweep 2026-06-07 trên `vn3k_test_100`

Mục tiêu là thu hẹp vùng activation nhạy trong blocks 4-11 trước khi chuyển sang AIMET/QAT.

| Candidate | Blocks giữ activation float | cosine_l2_mean | cosine_l2_min | cosine_l2_max | Gate | Kết luận |
|---|---|---:|---:|---:|---|---|
| `encoder_blocks_4_9_float` | 4-9 | `0.888407` | `0.797016` | `0.923269` | FAIL | Thiếu blocks 10-11, tụt mạnh |
| `encoder_blocks_4_10_float` | 4-10 | `0.943966` | `0.879753` | `0.971730` | FAIL | Block 11 rất quan trọng để vượt gate |
| `encoder_blocks_4_11_float` | 4-11 | `0.955255` | `0.900515` | `0.978287` | PASS | Cửa sổ liên tục nhỏ nhất đã pass trong các thử nghiệm hiện có |
| `encoder_blocks_5_11_float` | 5-11 | `0.904574` | `0.821596` | `0.960436` | FAIL | Block 4 cũng quan trọng |
| `encoder_blocks_6_11_float` | 6-11 | `0.697282` | `0.464951` | `0.858507` | FAIL | Bỏ blocks 4-5 làm fidelity sụp mạnh |

Kết luận refinement: vùng nhạy hiện vẫn là blocks 4-11. Bỏ đầu hoặc bỏ cuối đều fail, nên AIMET/QAT nên ưu tiên activation encoding/range cho toàn bộ blocks 4-11.

---

## 5. Checklist hiện hành

### Phase G1 - Tooling QDQ surgery local

- [x] Tạo script `deployment/scripts/qnn/qdq_surgery.py`.
- [x] Script nhận tối thiểu:
  - `--model`
  - `--output-dir`
  - `--mode`
  - `--keep-op-float`
  - `--keep-node-float`
  - `--keep-output-float`
- [x] Script save ONNX external-data đúng chuẩn để ONNX Runtime load được.
- [x] Script chạy được với base model `artifacts/deployment/runtime/job_jpe2lnmvp_qdq_onnx/model.onnx`.

### Phase G2 - No-op baseline

- [x] Tạo no-op copy từ `jpe2lnmvp`.
- [x] Chạy `onnx.checker.check_model`.
- [x] Chạy ONNX Runtime load smoke.
- [x] Chạy QDQ-vs-PyTorch trên `vn3k_test_10`.
- [x] Gate no-op: metric giữ gần `cosine_l2_mean = 0.3267`, không có NaN/Inf.

### Phase G3 - Node groups cần thử giữ float

- [x] Candidate `output_float`: gỡ Q/DQ ở final embedding/output path.
- [x] Candidate `final_head_float`: giữ float final head gồm `node_linear_74`, `node_layer_norm_25`, final MLP/GELU path, `node_add_1219`, `node_select_2`.
- [x] Candidate `all_layernorm_float`: giữ float tất cả `LayerNormalization`.
- [x] Candidate `attention_score_float`: giữ float `Softmax` và MatMul score path trong attention.
- [x] Candidate `combined_best`: đã thử `combined_layernorm_final_head`; trong vòng simple surgery không có tổ hợp nào cải thiện đủ để mở compile/link.

### Phase G4 - Fidelity gate cho từng candidate

- [x] Với mỗi candidate, chạy `compare_onnx_with_pytorch.py` trên `vn3k_test_10`.
- [x] Lưu JSON/CSV vào `artifacts/deployment/runtime/ptq_experiments/node_level/<candidate>/`.
- [x] Không compile/link nếu QDQ `cosine_l2_mean < 0.95` hoặc `cosine_l2_min < 0.90`.
- [x] Nếu candidate cải thiện rõ nhưng chưa pass, mở rộng vùng giữ float theo thứ tự: final head, LayerNorm, attention score path, projection/head cuối. Kết quả vòng simple surgery không có cải thiện rõ; best chỉ `0.3288`.

### Phase G5 - Compile/link khi QDQ pass

- [x] Thêm script compile/link QDQ ONNX local: `deployment/scripts/qnn/submit_qaihub_compile_link.py`.
- [x] Submit compile/link candidate pass gate với `--quantize_io`: `jgd0zw76p` compile PASS, `jgj1wxo1g` link FAIL.
- [ ] Tải QNN context binary về `artifacts/deployment/qnn_inputs/`.
- [ ] Chạy `vn3k_test_10` trên RB3 bằng `qnn-net-run`.
- [ ] Sync output về local.
- [ ] Chạy `compare_qnn_with_pytorch.py`.
- [ ] Chỉ mở rộng `vn3k_test_100` nếu QNN mean `>= 0.90`.

### Phase G6 - Nếu simple node-level surgery vẫn fail

- [ ] Đánh giá hướng local AIMET config sâu hơn: exclude/override activation encodings, per-node/per-block mixed precision thay vì chỉ bypass QDQ output.
- [ ] Đánh giá hướng QAT/fine-tune-aware quantization.
- [ ] Không mở rộng benchmark hoặc text encoder khi vision QNN fidelity chưa pass.

### Phase G7 - Sau khi vision pass

- [ ] Compile text encoder theo cùng nguyên tắc: QDQ compare trước, QNN runtime sau.
- [ ] Đo end-to-end image/text retrieval trên board.
- [ ] Cập nhật benchmark report.
- [ ] Viết guide deploy tái lập.

---

## 6. Command kiểm tra QDQ candidate

```bash
venv/bin/python deployment/scripts/qnn/compare_onnx_with_pytorch.py \
  --onnx-model artifacts/deployment/runtime/<candidate_qdq_onnx_dir> \
  --model-dir artifacts/deployment/exports/exported_model \
  --input-dir artifacts/deployment/qnn_inputs/vn3k_test_10 \
  --precision fp32 \
  --json artifacts/deployment/runtime/ptq_experiments/node_level/<candidate>/qdq_vs_pytorch_summary.json \
  --csv artifacts/deployment/runtime/ptq_experiments/node_level/<candidate>/qdq_vs_pytorch.csv
```

Diagnostic phụ trên calibration set được phép dùng để hiểu lỗi, nhưng không thay thế gate độc lập `vn3k_test_10`.

---

## 7. Cập nhật tài liệu khi chạy tiếp

- Daily journal ngày chạy chỉ ghi job/result/chẩn đoán/next step.
- Checklist chính chỉ cập nhật trong file này.
- Nếu cần báo cáo lại tiến độ, cập nhật `deployment/docs/[deploy]-report-1.md`.
- Không cập nhật changelog nếu chưa được yêu cầu riêng.

---

## 8. Kế hoạch tiếp theo - QDQ sensitivity decomposition

Node-level surgery đơn giản đã fail: best global QDQ ban đầu là `jpe2lnmvp` với `cosine_l2_mean = 0.3267`, best simple node-level surgery là `all_layernorm_float` với `cosine_l2_mean = 0.328831`. Cả hai đều fail xa gate QDQ `mean >= 0.95`, `min >= 0.90`.

QDQ sensitivity decomposition sau đó đã tìm được hướng đúng: `all_activations_float` pass gate, `all_weights_float` fail, và block-level combo `encoder_blocks_4_11_float` pass gate local. Vì vậy kế hoạch chuyển từ "tìm nguyên nhân" sang "validate/compile candidate activation mixed precision".

Kiểm tra dependency local hiện tại: `aimet_torch`, `aimet_onnx`, và `aimet_common` chưa có trong venv. Bước nhanh trước đã mở rộng local ONNX/QDQ diagnostics và xác định activation blocks 4-11 là vùng cần giữ float hoặc override encoding; chưa nhảy ngay sang QAT.

### Phase H1 - Upper-bound QDQ surgery candidates

- [x] Mở rộng `deployment/scripts/qnn/qdq_surgery.py` để tạo candidate `all_qdq_float`: bypass toàn bộ QDQ pairs.
- [x] Tạo candidate `all_weights_float`: bypass QDQ pairs có input là initializer/weight.
- [x] Tạo candidate `all_activations_float`: bypass QDQ pairs không phải initializer.
- [x] Tạo candidate `matmul_gemm_weights_float`: bypass weight QDQ cho `MatMul`/`Gemm`.

### Phase H2 - Fidelity gate cho upper-bound candidates

- [x] Với mỗi candidate H1, chạy `onnx.checker.check_model`.
- [x] Với mỗi candidate H1, chạy ONNX Runtime load smoke.
- [x] Với mỗi candidate H1, chạy QDQ-vs-PyTorch trên `vn3k_test_10`.
- [x] Lưu JSON/CSV vào `artifacts/deployment/runtime/ptq_experiments/node_level/<candidate>/`.
- [x] Không compile/link nếu QDQ `cosine_l2_mean < 0.95` hoặc `cosine_l2_min < 0.90`.

### Phase H3 - Diễn giải kết quả upper-bound

- [x] Nếu `all_qdq_float` vẫn fail: nghi model QDQ đã bị transform ngoài QDQ hoặc surgery mapping chưa đủ; dừng compile/link và debug graph mapping trước. Kết quả thực tế: `all_qdq_float` PASS, nên graph mapping hợp lệ.
- [x] Nếu `all_weights_float` cải thiện mạnh: ưu tiên weight mixed precision/per-channel config. Kết quả thực tế: `all_weights_float` FAIL và không cải thiện, nên chưa ưu tiên weight config.
- [x] Nếu `all_activations_float` cải thiện mạnh: ưu tiên activation encoding/range config. Kết quả thực tế: `all_activations_float` PASS, xác nhận activation quantization/range là hướng chính.
- [x] Nếu chỉ `all_qdq_float` pass nhưng weight/activation riêng lẻ fail: lỗi là tương tác weight + activation; ưu tiên block-level isolation hoặc QAT. Kết quả thực tế: `all_activations_float` cũng PASS, nên ưu tiên block-level activation isolation.

### Phase H4 - Block-level isolation nếu upper-bound hữu ích

- [x] Candidate `encoder_blocks_0_3_float`: giữ float QDQ trong encoder blocks 0-3.
- [x] Candidate `encoder_blocks_4_7_float`: giữ float QDQ trong encoder blocks 4-7.
- [x] Candidate `encoder_blocks_8_11_float`: giữ float QDQ trong encoder blocks 8-11.
- [x] Candidate `post_layernorm_head_float`: giữ float post-layernorm và head.
- [x] Candidate `encoder_blocks_4_11_float`: giữ float activation QDQ trong encoder blocks 4-11, đã pass QDQ gate local.
- [x] Chạy refinement sweep `encoder_blocks_4_9_float`, `encoder_blocks_4_10_float`, `encoder_blocks_5_11_float`, `encoder_blocks_6_11_float` trên `vn3k_test_100`.
- [x] Kết luận cửa sổ nhạy hiện là blocks 4-11; bỏ block 4 hoặc 11 đều fail.
- [ ] Candidate `worst_block_float`: chỉ chạy nếu cần isolate từng block trước AIMET/QAT; không dùng để compile/link trực tiếp.

### Phase H5 - AIMET/local config nếu tìm được vùng nhạy

- [x] Đánh giá cài/khả dụng `aimet_onnx` hoặc `aimet_torch`: `aimet_onnx`, `aimet_torch`, `aimet_common` đều missing trong venv.
- [x] Nếu chưa có dependency, ghi rõ đây là việc cần user chuẩn bị hoặc cấp quyền cài đặt.
- [ ] Nếu có AIMET, ưu tiên thử activation encoding override/exclude op theo block trước khi QAT.

### Phase H6 - QAT/fine-tune-aware quantization

- [ ] Nếu AIMET config vẫn không đưa QDQ mean lên gần gate, chuyển sang QAT/fine-tune-aware quantization.
- [ ] QAT chỉ bắt đầu sau khi xác định lỗi chính nằm ở activation/weight/block nào, để tránh fine-tune mù.

### Phase H7 - Quay lại compile/link khi QDQ pass

- [x] Xác định QDQ candidate đạt `cosine_l2_mean >= 0.95` và `cosine_l2_min >= 0.90`: `encoder_blocks_4_11_float`.
- [x] Validate `encoder_blocks_4_11_float` trên `vn3k_test_100`: PASS local QDQ với mean `0.955255`, min `0.900515`.
- [x] Compile/link QNN bằng candidate pass gate: compile DLC PASS, link HTP FAIL do internal float tensor `add_1003`.
- [ ] Chỉ quay lại compile/link khi có all-quantized deployable candidate, không phải `_float` surgery candidate.

Command template cho mỗi candidate sau khi preset H1/H4 được thêm:

```bash
venv/bin/python deployment/scripts/qnn/qdq_surgery.py \
  --model artifacts/deployment/runtime/job_jpe2lnmvp_qdq_onnx \
  --output-dir artifacts/deployment/runtime/ptq_experiments/node_level/<candidate> \
  --mode surgery \
  --preset <preset> \
  --check \
  --smoke-load

venv/bin/python deployment/scripts/qnn/compare_onnx_with_pytorch.py \
  --onnx-model artifacts/deployment/runtime/ptq_experiments/node_level/<candidate> \
  --model-dir artifacts/deployment/exports/exported_model \
  --input-dir artifacts/deployment/qnn_inputs/vn3k_test_10 \
  --precision fp32 \
  --json artifacts/deployment/runtime/ptq_experiments/node_level/<candidate>/qdq_vs_pytorch_summary.json \
  --csv artifacts/deployment/runtime/ptq_experiments/node_level/<candidate>/qdq_vs_pytorch.csv
```
