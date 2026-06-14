# [Deploy Plan] 2026-06-06 - Node-level mixed precision cho vision QDQ

> **Ngày lập kế hoạch:** 2026-06-06
> **Thiết bị mục tiêu:** Qualcomm RB3 Gen2 / QCS6490 / HTP V68
> **Model nguồn:** `epoch=56-val_score=52.28.ckpt`
> **Artifact nguồn hiện tại:** `artifacts/deployment/runtime/qat/blocks_6_11_v5/qaihub_lite_mp_qdq/model.onnx`
> **Calibration dataset:** `d7jzjy1m2` / `msiglip-vision-vn3k-train-calib-2000`
> **Plan checklist hiện hành:** file này
> **Trạng thái:** FOLLOW-UP - QAT blocks 6-11 v5 native QDQ fail, nhưng `all_weights + blocks 4-11` surgery pass QDQ local as diagnostic upper-bound; không compile/link vì `_float` path không deployable, nhánh active là Phase H10 QNN-native controlled quantization
> **Cập nhật checklist gần nhất:** 2026-06-14

---

## 1. Tóm tắt quyết định

Vision encoder đã chứng minh được runtime path trên RB3 HTP, nhưng QDQ fidelity vẫn fail. Các global options của AI Hub/PTQ đã thử đều không đạt gate.

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

Ngày 2026-06-09 đã thử hướng không dùng AIMET bằng ONNX Runtime static quantization. ORT W8A16 QDQ đạt fidelity local rất cao trên `vn3k_test_100`: `cosine_l2_mean = 0.999472`, `cosine_l2_min = 0.998398`. Tuy nhiên nhiều biến thể ORT QDQ đều compile pass rồi link fail trên QNN HTP, chủ yếu do tensor `gelu_10_DequantizeLinear_Output` còn floating-point hoặc context conversion fail exit code 14.

Sau đó đã thử INT16 activation encoding tuning từ `jpe2lnmvp`. Candidate `jpe2lnmvp_blocks_0_11_int16_opset21` đạt local QDQ gần gate (`cosine_l2_mean = 0.947065`, `cosine_l2_min = 0.925417`) nhưng compile/link vẫn fail: compile job `jgd03w76p` SUCCESS, link job `jp0kjyd25` FAIL do tensor nội bộ `add_103_updated` còn floating-point.

Quyết định tiếp theo: không compile/link thêm candidate `_float`, ORT QDQ, hoặc INT16 QDQ surgery cùng pattern. QAT clean-consistency đã giúp tạo được candidate `blocks_6_11_v5` pass clean gate nội bộ và near-pass fake gate (`clean 0.9539/0.9030`, `fake 0.7997/0.7022`). Tuy nhiên AI Hub Lite-MP QDQ từ candidate này fail rất nặng (`cosine_l2_mean = 0.244236`, `cosine_l2_min = 0.203228`) trong khi static ONNX control pass gần tuyệt đối. AI Hub default QDQ không `--lite_mp` cũng fail xa gate (`cosine_l2_mean = 0.252687`, `cosine_l2_min = 0.211948`). QDQ surgery diagnostic cho thấy cần xử lý đồng thời weight QDQ và activation QDQ trong blocks 4-11: `all_weights + blocks 4-11` pass local QDQ gate (`0.970312/0.939976`). Đây vẫn là diagnostic upper-bound không deployable vì có float path, nên không compile/link. Nhánh active chuyển sang tìm QNN-native/quantizer-level strategy tương đương mà vẫn deployable.

---

## 2. Gate bắt buộc

| Gate | Ngưỡng | Ý nghĩa |
|---|---:|---|
| Static ONNX vs PyTorch | `cosine_l2_mean >= 0.999` | Control export/preprocess phải pass |
| QDQ ONNX vs PyTorch mean | `cosine_l2_mean >= 0.95` | Đủ tốt để thử compile/link QNN |
| QDQ ONNX vs PyTorch min | `cosine_l2_min >= 0.90` | Không có sample lệch nặng |
| QNN vs PyTorch sau link | `cosine_l2_mean >= 0.90` | Đủ để mở rộng benchmark trên RB3 |
| Full retrieval | `T2I R@1 >= 48.0` | Mục tiêu deploy so với baseline `52.28%` |

Cho phép một lần diagnostic compile nếu QDQ do AI Hub-native quantizer tạo từ QAT model đạt gần gate: `cosine_l2_mean >= 0.93`, `cosine_l2_min >= 0.88`. Ngoại lệ này không áp dụng cho `_float`, ORT QDQ, hoặc INT16 QDQ surgery vì các pattern đó đã link fail lặp lại.

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
| AIMET branch | Không dùng tiếp trong plan hiện tại: Mac ARM không có wheel phù hợp; Ubuntu 20.04 server thiếu Python/glibc/pip phù hợp; Docker bị chặn quyền daemon |
| ORT W8A16 local fidelity | PASS local, best `minmax_calib100_w8a16s_opset21` / `minmax_calib100_w8a16_opset21`: `vn3k_test_100` mean `0.999472`, min `0.998398` |
| ORT W8A16 compile/link | Compile PASS nhiều lần, link FAIL ổn định quanh `gelu_10_DequantizeLinear_Output` hoặc exit code 14; không có deployable `.bin` |
| INT16 encoding tuning | Local gần gate với `jpe2lnmvp_blocks_0_11_int16_opset21`: mean `0.947065`, min `0.925417`; compile `jgd03w76p` PASS, link `jp0kjyd25` FAIL vì `add_103_updated` còn floating-point |
| QAT tooling | `deployment/scripts/qnn/train_vision_quant_robust.py` được tạo để train vision-only student với fake-quant activation noise ở blocks 4-11 |
| QAT v1 server | FAIL gate nội bộ: `lr=1e-5`, `mse_weight=0.05`, `500` steps, `val_clean` mean/min `0.7929/0.6806`, `val_fake_quant` mean/min `0.7912/0.6832`; không export/submit |
| QAT v1b server | FAIL gate nội bộ: `lr=1e-6`, `mse_weight=0.1`, `500` steps, `val_clean` mean/min `0.6533/0.4248`, `val_fake_quant` mean/min `0.4427/0.3029`; không export/submit |
| QAT clean-consistency tooling | DONE: script đã thêm `clean_weight` / `clean_mse_weight`, smoke CPU 1 step pass với `val_clean` mean `0.999737` |
| QAT v2-v5 full blocks 4-11 | FAIL gate: v4 cân bằng nhất (`clean 0.9505/0.8998`, `fake 0.7539/0.6337`), nhưng chưa đủ để export/submit |
| QAT blocks 6-11 v1 | FAIL gate nhưng promising: `clean 0.9297/0.8606`, `fake 0.8460/0.7515`; fake tốt nhất tại thời điểm chạy |
| QAT blocks 6-11 v2-v5 | v5 là candidate nội bộ tốt nhất: `clean 0.9539/0.9030`, `fake 0.7997/0.7022`; đủ để chạy diagnostic QDQ nhưng chưa phải deploy pass |
| QAT blocks 6-11 v5 static ONNX | PASS control trên `vn3k_test_10`: `cosine_l2_mean = 1.000000`, `cosine_l2_min = 0.9999998` |
| QAT blocks 6-11 v5 AI Hub Lite-MP QDQ | FAIL nặng trên `vn3k_test_10`: `cosine_l2_mean = 0.244236`, `cosine_l2_min = 0.203228`; không compile/link |
| QAT blocks 6-11 v5 AI Hub default QDQ | FAIL nặng trên `vn3k_test_10`: `cosine_l2_mean = 0.252687`, `cosine_l2_min = 0.211948`; không compile/link |
| QAT v5 Lite-MP QDQ no-op surgery | PASS tooling, giữ nguyên metric `0.244236`; `575` QDQ pairs, `0` selected |
| QAT v5 Lite-MP QDQ all-QDQ-float | PASS diagnostic: `cosine_l2_mean = 0.999779`, `cosine_l2_min = 0.999683`; graph mapping/surgery hợp lệ |
| QAT v5 Lite-MP QDQ all-weights-float | FAIL nhưng cải thiện: `cosine_l2_mean = 0.511043`, `cosine_l2_min = 0.378787`; weight QDQ chỉ là một phần vấn đề |
| QAT v5 Lite-MP QDQ all-activations-float | FAIL vì ONNX output NaN: `any_onnx_nan = true`; không deployable |
| QAT v5 Lite-MP QDQ encoder blocks 0-3 float | FAIL: `cosine_l2_mean = 0.169657`, `cosine_l2_min = 0.119832`; tệ hơn baseline QDQ |
| QAT v5 Lite-MP QDQ encoder blocks 4-7 float | FAIL: `cosine_l2_mean = 0.154641`, `cosine_l2_min = 0.094580`; tệ hơn baseline QDQ |
| QAT v5 Lite-MP QDQ encoder blocks 8-11 float | FAIL: `cosine_l2_mean = 0.014968`, `cosine_l2_min = -0.016124`; gần mất hướng embedding |
| QAT v5 Lite-MP QDQ post-layernorm/head float | FAIL vì ONNX output NaN: `any_onnx_nan = true` |
| QAT v5 Lite-MP QDQ all-weights + blocks 0-3 float | FAIL: `cosine_l2_mean = 0.534474`, `cosine_l2_min = 0.383061` |
| QAT v5 Lite-MP QDQ all-weights + blocks 4-7 float | FAIL nhưng tín hiệu tốt nhất: `cosine_l2_mean = 0.858946`, `cosine_l2_min = 0.802091`; dưới diagnostic compile gate |
| QAT v5 Lite-MP QDQ all-weights + blocks 8-11 float | FAIL: `cosine_l2_mean = 0.745280`, `cosine_l2_min = 0.626759` |
| QAT v5 Lite-MP QDQ all-weights + post-layernorm/head float | FAIL: `cosine_l2_mean = 0.493702`, `cosine_l2_min = 0.351181` |
| QAT v5 Lite-MP QDQ all-weights + blocks 4-9 float | PASS diagnostic: `cosine_l2_mean = 0.947507`, `cosine_l2_min = 0.913112`; mean hơi dưới production gate |
| QAT v5 Lite-MP QDQ all-weights + blocks 4-10 float | PASS: `cosine_l2_mean = 0.964700`, `cosine_l2_min = 0.930359` |
| QAT v5 Lite-MP QDQ all-weights + blocks 4-11 float | PASS diagnostic upper-bound: `cosine_l2_mean = 0.970312`, `cosine_l2_min = 0.939976`; không deployable vì `_float` surgery |
| QAT v5 Lite-MP QDQ int8 activation encoding max_abs sweep | FAIL xa gate trên blocks 4-11; best mean là `maxabs256` với `cosine_l2_mean = 0.247837`, `cosine_l2_min = 0.198776`; không compile/link |
| QAT blocks 6-11 v5 AI Hub native W8A16 QDQ | FAIL xa gate trên `vn3k_test_10`: `cosine_l2_mean = 0.155494`, `cosine_l2_min = 0.115508`; tệ hơn Lite-MP INT8, không compile/link |
| Next active branch | Phase H10 QNN-native controlled quantization: không compile/link `_float` surgery, int8 encoding clamp, hoặc native W8A16; tìm cách kiểm soát encoding/range cho weight + activation blocks 4-11 bằng QNN-native toolchain |

Các kết quả cũ vẫn nằm trong daily journal/report:

```text
deployment/docs/journal/[deploy]-2026-06-02.md
deployment/docs/journal/[deploy]-2026-06-04.md
deployment/docs/journal/[deploy]-2026-06-05.md
deployment/docs/journal/[deploy]-2026-06-06.md
deployment/docs/journal/[deploy]-2026-06-07.md
deployment/docs/journal/[deploy]-2026-06-09.md
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

### 4.5 ORT static quantization 2026-06-09 - local fidelity pass, QNN link fail

Sau khi AIMET không khả dụng, đã thử nhánh ONNX Runtime static quantization từ FP32 ONNX. Mục tiêu là tạo candidate all-quantized không dùng `_float` surgery nhưng vẫn giữ được cosine local.

| Candidate | Config chính | Dataset | cosine_l2_mean | cosine_l2_min | cosine_l2_max | Kết luận |
|---|---|---|---:|---:|---:|---|
| `minmax_calib100_w8a8` | ORT QDQ MinMax, W8A8, per-channel | `vn3k_test_10` | `0.177257` | `0.141239` | `0.219684` | FAIL |
| `minmax_calib100_linear_w8a8` | ORT QDQ chỉ `MatMul/Gemm/Conv`, W8A8 | `vn3k_test_10` | `0.873959` | `0.804497` | `0.930232` | FAIL nhưng có tín hiệu diagnostic |
| `qoperator_minmax_calib100_w8a8` | ORT QOperator W8A8 | `vn3k_test_10` | `0.168127` | `0.073877` | `0.219146` | FAIL |
| `minmax_calib100_w8a16_opset21` | ORT QDQ W8A16, `quint16` activation, opset 21 | `vn3k_test_100` | `0.999472` | `0.998398` | `0.999719` | PASS local |
| `minmax_calib100_w8a16s_opset21` | ORT QDQ W8A16, `qint16` activation, opset 21 | `vn3k_test_100` | `0.999472` | `0.998398` | `0.999719` | PASS local |
| `minmax_calib100_w8a16s_gelu_qint8_opset21` | GELU output QDQ retarget `qint8` | `vn3k_test_100` | `0.978237` | `0.954687` | `0.991896` | PASS QDQ gate |
| `minmax_calib100_w8a16s_matmul_act_qint8_opset21` | MatMul/Gemm activation QDQ retarget `qint8` | `vn3k_test_100` | `0.969894` | `0.939456` | `0.986330` | PASS QDQ gate |
| `minmax_calib100_w8a16s_gelu_float_for_requant_opset21` | Bypass GELU output QDQ để QNN thử re-quantize | `vn3k_test_100` | `0.999476` | `0.998436` | `0.999724` | PASS local |

Compile/link AI Hub:

| Candidate | Compile job | Link job | Kết quả |
|---|---|---|---|
| `minmax_calib100_w8a16_opset21` | `jgd03mkrp` | `jgj178xvg` | Compile SUCCESS, link FAIL: `gelu_10_DequantizeLinear_Output` còn floating-point |
| `minmax_calib100_w8a16s_opset21` | `j5wx708jp` | `j576417rg` | Compile SUCCESS, link FAIL: `gelu_10_DequantizeLinear_Output` còn floating-point |
| `minmax_calib100_w8a16s_opset21` + `--quantize_full_type int16` | `jgkd41xnp` | `jgzwm63xg` | Compile SUCCESS, link FAIL: context conversion exit code 14 |
| `minmax_calib100_w8a16s_opset21` + HTP FP16 internal | `jgd03ke6p` | `jpv4d8rzp` | Compile SUCCESS, link FAIL: `gelu_10_DequantizeLinear_Output` còn floating-point |
| `minmax_calib100_w8a16s_gelu_quint8_opset21` | `jgomr74d5` | `jp389l6z5` | Compile SUCCESS, link FAIL: `gelu_10_DequantizeLinear_Output` còn floating-point |
| `minmax_calib100_w8a16s_gelu_qint8_opset21` | `jgj179k7g` | `j5wx7kmzp` | Compile SUCCESS, link FAIL: `gelu_10_DequantizeLinear_Output` còn floating-point |
| `minmax_calib100_w8a16s_matmul_act_qint8_opset21` | `jp88xn48p` | `jgl7xd1l5` | Compile SUCCESS, link FAIL: `gelu_10_DequantizeLinear_Output` còn floating-point |
| `minmax_calib100_w8a16s_gelu_float_for_requant_opset21` + `--quantize_full_type int16` | `jgzwm1kog` | `jpr90k37p` | Compile SUCCESS, link FAIL: context conversion exit code 14 |

Kết luận: ORT W8A16 đã chứng minh fidelity local có thể đạt rất cao, nhưng QNN HTP linker không chấp nhận pattern QDQ/Dequantize quanh GELU. Dừng nhánh upload ORT QDQ tương tự; nếu quay lại QNN compile/link, cần candidate all-quantized do QAT hoặc QNN-native toolchain tạo ra.

### 4.6 INT16 activation encoding tuning 2026-06-09 - gần gate local nhưng link fail

Sau ORT branch, đã thử tune activation encoding/dtype trực tiếp trên `jpe2lnmvp`.

| Candidate | Config chính | Dataset | cosine_l2_mean | cosine_l2_min | cosine_l2_max | Kết luận |
|---|---|---|---:|---:|---:|---|
| `jpe2lnmvp_blocks_4_11_int16_opset21` | Retarget activation QDQ blocks 4-11 sang INT16, opset 21 | `vn3k_test_100` | `0.928473` | `0.913959` | n/a | Near-pass, mean thấp hơn gate |
| `jpe2lnmvp_blocks_0_11_int16_opset21` | Retarget activation QDQ blocks 0-11 sang INT16, opset 21 | `vn3k_test_100` | `0.947065` | `0.925417` | n/a | Gần gate nhất nhưng vẫn chưa đạt production gate |
| `jpe2lnmvp_all_activations_int16_opset21` | Retarget toàn bộ activation QDQ sang INT16, opset 21 | `vn3k_test_100` | `0.945032` | `0.925546` | n/a | Gần gate, nhưng không hơn blocks 0-11 |

Compile/link AI Hub cho candidate gần gate:

| Candidate | Compile job | Link job | Kết quả |
|---|---|---|---|
| `jpe2lnmvp_blocks_0_11_int16_opset21` | `jgd03w76p` | `jp0kjyd25` | Compile SUCCESS, link FAIL: `Tensor 'add_103_updated' has a floating-point type` |

Kết luận: INT16 encoding tuning hữu ích để xác nhận activation quantization là nguyên nhân chính, nhưng vẫn không tạo được graph deployable cho HTP. Dừng compile/link các biến thể INT16 QDQ surgery cùng pattern.

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

- [x] Đánh giá hướng local AIMET config sâu hơn: dependency/môi trường không khả dụng trong plan hiện tại, chi tiết ở Phase H5.
- [x] Đánh giá hướng QAT/fine-tune-aware quantization: chuyển thành nhánh active sau khi ORT W8A16 local pass nhưng QNN link fail.
- [x] Không mở rộng benchmark hoặc text encoder khi vision QNN fidelity chưa pass.

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

Kiểm tra dependency local hiện tại: `aimet_torch`, `aimet_onnx`, và `aimet_common` chưa có trong venv. Bước nhanh trước đã mở rộng local ONNX/QDQ diagnostics và xác định activation blocks 4-11 là vùng cần giữ float hoặc override encoding. Sau khi AIMET bị chặn môi trường và ORT W8A16 local pass nhưng QNN link fail, nhánh active tiếp theo là QAT hoặc QNN-native quantization/export.

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
- [x] Thử chuẩn bị môi trường AIMET ngoài Mac: Ubuntu 20.04 server không phù hợp wheel AIMET 2.31 do Python/glibc/pip mismatch; Docker bị chặn quyền daemon.
- [x] Quyết định không dùng AIMET trong plan hiện tại.

### Phase H6 - QAT/fine-tune-aware quantization

- [x] Đã xác định lỗi chính nằm ở activation quantization/range, nhạy nhất ở encoder blocks 4-11.
- [x] Đã xác định ORT W8A16 có thể pass fidelity local nhưng không link được trên QNN HTP.
- [x] Đã xác định INT16 QDQ surgery có thể gần gate local nhưng vẫn không link được trên QNN HTP.
- [x] Thiết kế QAT/fine-tune-aware quantization tối thiểu cho vision encoder, ưu tiên activation path blocks 4-11.
- [x] Tạo script `deployment/scripts/qnn/train_vision_quant_robust.py`.
- [x] Chạy QAT smoke `--max-steps 1 --batch-size 1`: pass trên CPU, clean student còn gần teacher, fake-quant vẫn lệch mạnh do mới train 1 step.
- [x] Chạy QAT v1 trên Ubuntu RTX 3060 server: FAIL gate nội bộ, `val_clean` mean `0.7929`, `val_fake_quant` mean `0.7912`.
- [x] Chạy QAT v1b với `lr=1e-6`, `mse_weight=0.1`: FAIL gate nội bộ, `val_clean` mean `0.6533`, `val_fake_quant` mean `0.4427`.
- [x] Sửa QAT objective để thêm clean-consistency loss trong cùng batch.
- [x] Chạy smoke CPU 1 step sau khi sửa script: PASS, `val_clean` mean `0.999737`, `val_fake_quant` mean `0.086960`.
- [x] Chạy QAT v2-v5 full blocks 4-11 trên server/local metadata review: tất cả FAIL gate nội bộ; v4 là điểm cân bằng tốt nhất nhưng fake thấp.
- [x] Chạy QAT blocks 6-11 v1 trên server: FAIL clean gate nhưng fake tốt nhất (`0.8460` mean).
- [x] Chạy QAT blocks 6-11 v2-v5 với clean constraint tăng dần; v5 là candidate nội bộ tốt nhất.
- [x] Export QAT blocks 6-11 v5 sang FP32 ONNX; static ONNX control pass trên `vn3k_test_10`.
- [x] Submit AI Hub native Lite-MP quantize-only từ QAT v5 ONNX và chạy `compare_onnx_with_pytorch.py` trên `vn3k_test_10`: FAIL nặng (`0.244236/0.203228`).
- [x] Chạy AI Hub default quantize-only sanity check không `--lite_mp` trên QAT v5: FAIL nặng (`0.252687/0.211948`).
- [x] Chạy no-op QDQ surgery baseline trên `artifacts/deployment/runtime/qat/blocks_6_11_v5/qaihub_lite_mp_qdq`: PASS tooling, metric giữ nguyên `0.244236`.
- [x] Chạy QAT v5 QDQ sensitivity: `all_qdq_float` PASS diagnostic (`0.999779/0.999683`), `all_weights_float` FAIL (`0.511043/0.378787`), `all_activations_float` FAIL do NaN.
- [x] Chạy QAT v5 block-level activation isolation: `encoder_blocks_0_3_float` FAIL (`0.169657/0.119832`), `encoder_blocks_4_7_float` FAIL (`0.154641/0.094580`), `encoder_blocks_8_11_float` FAIL (`0.014968/-0.016124`), `post_layernorm_head_float` FAIL do NaN.
- [x] Chạy tổ hợp `all_weights_float + block_activation_float`: blocks 4-7 là tín hiệu tốt nhất (`0.858946/0.802091`) nhưng vẫn dưới diagnostic compile gate; không compile/link.
- [x] Chạy tổ hợp liên tục `all_weights + blocks 4-9/4-10/4-11`: `4-11` tốt nhất (`0.970312/0.939976`) và pass QDQ local as diagnostic upper-bound.
- [x] Chạy QAT v5 Lite-MP QDQ int8 activation encoding `max_abs` sweep trên blocks 4-11: tất cả FAIL, best `0.247837/0.198776`; không compile/link.
- [x] Chạy QAT v5 AI Hub native W8A16 quantize-only + compare: FAIL xa gate (`0.155494/0.115508`); không compile/link.
- [x] Không compile/link các `_float` surgery candidate hoặc int8 encoding clamp candidate.
- [x] Mở Phase H10 QNN-native controlled quantization cho weight + activation blocks 4-11.
- [ ] Chỉ mở `vn3k_test_100` nếu QAT/native QDQ pass hoặc near-pass gate `vn3k_test_10`.

### Phase H7 - Quay lại compile/link khi QDQ pass

- [x] Xác định QDQ candidate đạt `cosine_l2_mean >= 0.95` và `cosine_l2_min >= 0.90`: `encoder_blocks_4_11_float`.
- [x] Validate `encoder_blocks_4_11_float` trên `vn3k_test_100`: PASS local QDQ với mean `0.955255`, min `0.900515`.
- [x] Compile/link QNN bằng candidate pass gate: compile DLC PASS, link HTP FAIL do internal float tensor `add_1003`.
- [x] Thử ORT W8A16 all-quantized local candidates: fidelity PASS local nhưng QNN link FAIL quanh `gelu_10_DequantizeLinear_Output` / exit code 14.
- [x] Thử INT16 activation encoding tuning gần gate: compile PASS, link FAIL quanh `add_103_updated`.
- [x] Thử AI Hub Lite-MP QDQ từ QAT blocks 6-11 v5: QDQ local FAIL xa gate, nên không compile/link.
- [x] Thử AI Hub default QDQ từ QAT blocks 6-11 v5: QDQ local FAIL xa gate, nên không compile/link.
- [x] Thử int8 activation encoding clamp trên QAT v5 Lite-MP QDQ: local QDQ vẫn FAIL xa gate, nên không compile/link.
- [x] Thử AI Hub native W8A16 từ QAT blocks 6-11 v5: local QDQ FAIL xa gate, nên không compile/link.
- [ ] Chỉ quay lại compile/link khi có all-quantized deployable candidate từ AI Hub/QNN-native quantizer, hoặc khi một diagnostic QDQ local pass/near-pass tạo ra giả thuyết compile có giá trị. Không compile/link lại các pattern `_float`, ORT QDQ, hoặc INT16 surgery đã fail link lặp lại.

### Phase H8 - ORT static quantization branch

- [x] Tạo `deployment/scripts/qnn/quantize_ort_static.py`.
- [x] Tạo `deployment/scripts/qnn/retarget_qdq_dtype.py`.
- [x] Chạy ORT W8A8 full và linear-only diagnostic.
- [x] Chạy ORT W8A16 opset21 signed/unsigned activation; local fidelity pass rất cao.
- [x] Retarget GELU output QDQ sang `quint8` và `qint8`; local vẫn pass gate.
- [x] Retarget MatMul/Gemm activation QDQ sang `qint8`; local vẫn pass gate.
- [x] Thử bypass GELU output QDQ để QNN re-quantize với `--quantize_full_type int16`.
- [x] Compile/link các biến thể chính qua AI Hub: compile pass, link fail.
- [x] Kết luận dừng upload thêm ORT QDQ variants cùng pattern.

### Phase H9 - Vision-only QAT/native quantization branch

- [x] Tạo script `deployment/scripts/qnn/train_vision_quant_robust.py`.
- [x] Script dùng teacher FP32 frozen và student FP32 clone từ `artifacts/deployment/exports/exported_model`.
- [x] Script freeze ngoài vùng nhạy, chỉ train `backbone.vision_model.encoder.layers.4..11` và `backbone.visual_projection`.
- [x] Script inject fake-quant activation noise vào block output và `mlp.activation_fn` cho layers 4-11.
- [x] Script lưu output dạng export FP32 ở `artifacts/deployment/exports/exported_model_qat_v1/` gồm `config.yaml`, `model_fp32.pt`, summary, và danh sách trainable params.
- [x] Chạy `python -m py_compile deployment/scripts/qnn/train_vision_quant_robust.py`.
- [x] Chạy CPU smoke 1 step: `val_clean cosine_l2_mean = 0.999737`, `val_fake_quant cosine_l2_mean = 0.086960`.
- [x] Chạy QAT v1 trên server GPU: FAIL gate nội bộ vì clean drift (`val_clean` mean `0.7929`).
- [x] Chạy QAT v1b trên server GPU với LR thấp hơn/MSE cao hơn: FAIL gate nội bộ và tệ hơn v1 (`val_clean` mean `0.6533`).
- [x] Thêm clean-consistency objective vào script QAT.
- [x] Chạy smoke CPU 1 step: PASS với `clean_weight=1.0`, `clean_mse_weight=0.05`.
- [x] Chạy QAT v2 trên server GPU: FAIL, `clean 0.9396/0.8913`, `fake 0.7826/0.6492`.
- [x] Chạy QAT v3 full blocks 4-11: FAIL, clean pass nhưng fake tụt (`clean 0.9596/0.9201`, `fake 0.7238/0.5700`).
- [x] Chạy QAT v4 full blocks 4-11: FAIL nhưng cân bằng nhất full-window (`clean 0.9505/0.8998`, `fake 0.7539/0.6337`).
- [x] Chạy QAT v5 full blocks 4-11: FAIL; local/server đều không hơn v4.
- [x] Chạy QAT blocks 6-11 v1: FAIL clean gate nhưng fake tốt nhất (`clean 0.9297/0.8606`, `fake 0.8460/0.7515`).
- [x] Chạy QAT blocks 6-11 v2-v5 với clean constraint tăng dần; v5 đạt `clean 0.9539/0.9030`, `fake 0.7997/0.7022`.
- [x] Export ONNX từ QAT blocks 6-11 v5 và xác nhận static ONNX vs PyTorch pass.
- [x] Submit AI Hub native quantize-only với `--quantize-options=--lite_mp` cho QAT blocks 6-11 v5.
- [x] So Lite-MP QDQ native với PyTorch trên `vn3k_test_10`: FAIL, `cosine_l2_mean = 0.244236`, `cosine_l2_min = 0.203228`.
- [x] Submit AI Hub quantize-only default không `--lite_mp` cho QAT blocks 6-11 v5 như sanity check: FAIL, `cosine_l2_mean = 0.252687`, `cosine_l2_min = 0.211948`.
- [x] Chạy QDQ surgery diagnostic trực tiếp trên QAT blocks 6-11 v5 Lite-MP QDQ: no-op, all-QDQ-float, weight-vs-activation.
- [x] Chạy QDQ surgery diagnostic block-level activation isolation trên QAT blocks 6-11 v5 Lite-MP QDQ: tất cả FAIL hoặc NaN.
- [x] Chạy QDQ surgery interaction diagnostic `all_weights_float + block_activation_float`: blocks 4-7 tốt nhất nhưng vẫn FAIL.
- [x] Chạy QDQ surgery continuous-window diagnostic `all_weights + blocks 4-11/4-9/4-10`: blocks 4-11 pass local QDQ gate as diagnostic upper-bound.
- [x] Chạy QAT v5 Lite-MP QDQ int8 activation encoding `max_abs` sweep trên blocks 4-11: best `0.247837/0.198776`, fail xa gate.
- [x] Chạy QAT v5 AI Hub native W8A16 quantize-only + compare: `0.155494/0.115508`, fail xa gate.
- [ ] Nếu diagnostic tạo được QDQ local pass/near-pass bằng một vùng can thiệp deployable hoặc có giả thuyết QNN-native rõ, thử compile/link một lần; nếu chỉ pass bằng `_float` surgery thì không compile/link.

### Phase H10 - QNN-native controlled quantization

Mục tiêu của Phase H10 là thoát khỏi AI Hub quantize như black box. Các kết quả đã khóa cho thấy model/export/preprocess không hỏng; lỗi nằm ở cách quantizer mã hóa weight + activation trong vision encoder, đặc biệt vùng blocks 4-11. Diagnostic `all_weights + blocks 4-11 float` đạt `0.970312/0.939976`, nhưng không deployable vì để lại float path. Vì vậy H10 chỉ có giá trị nếu tạo được candidate **all-quantized hoặc QNN-native deployable**, không phải `_float` surgery.

Điều kiện mở nhánh:

- [x] AI Hub Lite-MP INT8 từ QAT v5 fail xa gate: `0.244236/0.203228`.
- [x] AI Hub default INT8 từ QAT v5 fail xa gate: `0.252687/0.211948`.
- [x] AI Hub native W8A16 từ QAT v5 fail xa gate: `0.155494/0.115508`.
- [x] Int8 activation encoding clamp trên QAT v5 fail xa gate: best `0.247837/0.198776`.
- [x] `_float`, ORT QDQ, và INT16 surgery pattern đã từng compile/link fail trên HTP; không lặp lại các pattern này.

Checklist H10:

- [x] Kiểm tra Mac local bằng `deployment/scripts/qnn/audit_qnn_native_env.py`: không có QNN/QAIRT tools (`0/12` found), không dùng Mac local cho H10.
- [ ] Kiểm tra môi trường server có Qualcomm AI Stack hoặc QNN SDK đầy đủ hay không: converter, quantizer, runtime tools, ví dụ `qairt-converter`, `qnn-onnx-converter`, `qnn-model-lib-generator`, `qnn-context-binary-generator`, `qnn-net-run`.
- [ ] Ghi lại version QNN/QAIRT, target backend HTP, OS, Python, và command path vào journal trước khi chạy job mới.
- [ ] Xác định QNN-native quantizer có hỗ trợ override encoding/range theo tensor/op, calibration method, percentile/cle, per-channel weight, hoặc exclude/mixed precision deployable hay không.
- [ ] Tạo một local/server QNN-native quantize-only artifact từ static QAT v5 ONNX, không compile/link ngay.
- [ ] Chạy QDQ/ONNX-vs-PyTorch trên `vn3k_test_10` cho artifact QNN-native nếu artifact có ONNX/QDQ kiểm được; nếu chỉ sinh DLC/context, chạy QNN-vs-PyTorch trên cùng input raw.
- [ ] Chỉ mở `vn3k_test_100` khi `vn3k_test_10` pass hoặc near-pass gate.
- [ ] Chỉ compile/link HTP khi candidate là all-quantized/deployable và đạt production gate hoặc diagnostic gate.
- [ ] Nếu QNN-native toolchain không cho kiểm soát encoding đủ sâu, đóng H10 và chuyển sang fallback runtime plan: ONNX Runtime CPU/FP16 trên RB3 để đo latency/memory thực tế.

Các việc không làm trong H10:

- Không compile/link `all_weights_plus_blocks_4_11` hoặc bất kỳ `_float` surgery candidate nào.
- Không upload thêm ORT W8A16/INT16 QDQ variants cùng pattern đã fail linker.
- Không chạy thêm QAT proxy nếu chưa có cơ chế mô phỏng đúng QNN/AI Hub quantizer.
- Không chạy full retrieval/R@1 trước khi vision QNN fidelity pass.

Artifact đầu vào ưu tiên:

```text
artifacts/deployment/exports/exported_model_qat_blocks_6_11_v5/vision_onnx
artifacts/deployment/exports/exported_model_qat_blocks_6_11_v5/vision_onnx_static
artifacts/deployment/qnn_inputs/vn3k_train_calib_2000
artifacts/deployment/qnn_inputs/vn3k_test_10
```

Artifact đầu ra dự kiến:

```text
artifacts/deployment/runtime/qnn_native/qat_blocks_6_11_v5/<candidate>/
```

Command QAT smoke:

```bash
python3 deployment/scripts/qnn/train_vision_quant_robust.py \
  --max-steps 1 \
  --batch-size 1 \
  --max-train-samples 1 \
  --max-val-samples 1
```

Command QAT v1 trên server GPU (đã chạy, fail gate nội bộ; giữ lại để tham chiếu, không dùng tiếp nếu chưa sửa objective):

```bash
python3 deployment/scripts/qnn/train_vision_quant_robust.py \
  --batch-size 4 \
  --epochs 1 \
  --lr 1e-5 \
  --mse-weight 0.05 \
  --device auto
```

Command QAT v2 trên server GPU sau khi thêm clean-consistency:

```bash
python3 deployment/scripts/qnn/train_vision_quant_robust.py \
  --batch-size 4 \
  --epochs 1 \
  --lr 1e-5 \
  --mse-weight 0.05 \
  --clean-weight 1.0 \
  --clean-mse-weight 0.05 \
  --device auto \
  --output-dir artifacts/deployment/exports/exported_model_qat_v2
```

Command QAT blocks 6-11 v2:

```bash
python3 deployment/scripts/qnn/train_vision_quant_robust.py \
  --batch-size 4 \
  --epochs 1 \
  --lr 1e-5 \
  --mse-weight 0.05 \
  --clean-weight 1.5 \
  --clean-mse-weight 0.075 \
  --start-layer 6 \
  --end-layer 11 \
  --device auto \
  --output-dir artifacts/deployment/exports/exported_model_qat_blocks_6_11_v2
```

Command export và AI Hub native quantize:

```bash
python3 deployment/scripts/onnx/export.py \
  --model-dir artifacts/deployment/exports/exported_model_qat_v1 \
  --precision fp32

python3 deployment/scripts/qnn/submit_qaihub_quantize_compile.py \
  --model artifacts/deployment/exports/exported_model_qat_v1/vision_onnx \
  --calibration-data d7jzjy1m2 \
  --weights-dtype int8 \
  --activations-dtype int8 \
  --quantize-options=--lite_mp \
  --name msiglip-vision-qat-v1-lite-mp-calib2000-qonly \
  --wait \
  --quantize-only \
  --download-quantized artifacts/deployment/runtime/qat/vision_quant_robust_v1/qaihub_lite_mp_qdq
```

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
