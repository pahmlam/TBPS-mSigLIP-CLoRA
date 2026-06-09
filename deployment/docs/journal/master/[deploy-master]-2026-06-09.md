# [Deploy Master] 2026-06-09 - Lịch sử nén và deploy mSigLIP lên RB3 Gen2

> **Ngày tổng hợp:** 2026-06-09
> **Phạm vi:** Deploy/nén mSigLIP lên Qualcomm RB3 Gen2, không bao gồm demo-system
> **Thiết bị mục tiêu:** Qualcomm RB3 Gen2 / QCS6490 / HTP V68
> **Model nguồn:** `epoch=56-val_score=52.28.ckpt`
> **Baseline training:** VN3K text-to-image R@1 = `52.28%`
> **Plan checklist hiện hành:** `deployment/docs/journal/[deploy-plan]-2026-06-06.md`
> **Trạng thái hiện tại:** Vision encoder runtime HTP đã chứng minh được, nhưng chưa có QNN context binary accuracy-usable; nhánh active là vision-only QAT/fine-tune-aware quantization rồi AI Hub/QNN-native quantizer.

---

## 1. Mục tiêu và nguyên tắc chung

Mục tiêu deploy là đưa mSigLIP TBPS lên RB3 Gen2 để chạy retrieval người bằng mô tả tiếng Việt trực tiếp trên edge device. Hệ thống cuối cần hai encoder chạy cục bộ:

| Encoder | Input | Output | Trạng thái hiện tại |
|---|---|---|---|
| Vision encoder | `image: 1x3x256x256` | `image_embedding: 1x768` | Runtime HTP pass; fidelity QNN chưa pass |
| Text encoder | `input_ids`, `attention_mask` | `text_embedding: 1x768` | Chưa compile, chờ vision QNN fidelity pass |

Nguyên tắc xuyên suốt: không coi một model là deploy thành công chỉ vì compile được hoặc chạy được trên board. Với embedding retrieval, vector phải giữ đúng hướng so với PyTorch/ONNX baseline. Vì vậy cosine sau L2-normalize là metric gate chính, còn PSNR trong log AI Hub/AIMET chỉ là diagnostic phụ.

---

## 2. Decision gates đang dùng

| Gate | Ngưỡng | Ý nghĩa | Khi fail thì làm gì |
|---|---:|---|---|
| Static ONNX vs PyTorch | `cosine_l2_mean >= 0.999` | Export/preprocess control phải đúng | Dừng, sửa export hoặc raw input trước |
| QDQ ONNX vs PyTorch mean | `cosine_l2_mean >= 0.95` | Candidate đủ tốt để thử compile/link QNN | Dừng ở local, không submit compile/link |
| QDQ ONNX vs PyTorch min | `cosine_l2_min >= 0.90` | Không có sample lệch nặng | Dừng hoặc mở diagnostic để tìm worst region |
| QNN vs PyTorch sau link | `cosine_l2_mean >= 0.90` | Runtime HTP đủ tốt để mở rộng board benchmark | Dừng ở `vn3k_test_10`, không chạy retrieval |
| Full retrieval | `T2I R@1 >= 48.0` | Mục tiêu deploy so với baseline `52.28%` | Xem lại text/vision/binarization/runtime |

Ngoại lệ hiện tại: nếu QDQ do AI Hub-native quantizer tạo từ QAT model đạt gần gate (`mean >= 0.93`, `min >= 0.88`) thì cho phép một lần diagnostic compile/link. Ngoại lệ này không áp dụng cho `_float` surgery, ORT QDQ, hoặc INT16 QDQ surgery vì các pattern đó đã link fail lặp lại.

---

## 3. Cách tư duy trước mỗi lần chạy

Trước mỗi run phải xác định run đó đang kiểm tra tầng nào:

| Tầng | Câu hỏi cần trả lời | Bằng chứng cần có |
|---|---|---|
| Export/preprocess | PyTorch và ONNX có khớp không? Raw input có đúng shape/range không? | Static ONNX vs PyTorch, raw audit |
| PTQ/QDQ | Quantized ONNX đã giữ embedding chưa? | QDQ ONNX vs PyTorch trên `vn3k_test_10` hoặc `vn3k_test_100` |
| Compile/link | QNN toolchain có tạo được context binary HTP không? | Compile job + link job đều SUCCESS |
| RB3 runtime | Binary có load/chạy trên HTP không? | `qnn-net-run`, output shape đúng, no NaN/Inf |
| Fidelity runtime | QNN output còn đúng hướng embedding không? | QNN vs PyTorch cosine |
| Retrieval | End-to-end có giữ R@1 không? | Full VN3K retrieval |

Luật chạy:

- Nếu QDQ đã fail, không compile/link. Compile lúc đó chỉ tốn AI Hub và không thêm nhiều thông tin.
- Nếu compile pass nhưng link fail vì internal float tensor, không chạy board; chưa có `.bin` deployable.
- Nếu runtime pass nhưng QNN fidelity fail, không chạy text encoder hoặc retrieval.
- Calibration set (`vn3k_train_calib_*`) dùng để tính range; gate fidelity phải dùng set độc lập như `vn3k_test_10` hoặc `vn3k_test_100`.
- Mỗi run phải ghi rõ input, artifact, job ID, metric, gate, và quyết định pass/fail.

---

## 4. Timeline kỹ thuật

### 4.1 2026-04-15 - Học luật compile AI Hub/QNN HTP

Mục tiêu ngày này là tìm đúng format để đưa vision ONNX lên AI Hub compile cho HTP.

Các kết luận đã khóa:

- ONNX dùng external weights phải upload cả directory, không upload file `.onnx` rời.
- QNN context binary path không chấp nhận dynamic input shape; vision input phải static `1x3x256x256`.
- `--input_specs` phải là Python dict literal.
- HTP V68 reject floating-point I/O boundary, gồm FP32 và FP16.

Các job quan trọng:

| Job / thử nghiệm | Kết quả | Bài học |
|---|---|---|
| `jgn9139q5` | Upload directory OK nhưng compile fail vì dynamic shape | Cần static shape |
| `j563onvy5` | HTP reject floating-point input | FP32 I/O không dùng được |
| `jp2k1l3xg` | Vẫn preserve FP I/O, HTP reject | `--quantize_io` không đủ trong flow CLI cũ |
| `jp27om9r5` | FP16 ONNX vẫn bị reject | FP16 I/O không giải quyết HTP boundary |

Quyết định sau ngày này: dừng thử FP32/FP16 I/O trực tiếp cho HTP, chuyển sang INT8/INT16 I/O.

### 4.2 2026-05-06 - INT8 compile path và dummy calibration

Mục tiêu là xác nhận QNN context binary cho HTP có thể tạo được nếu I/O được quantize đúng.

| Job | Config | Kết quả | Ý nghĩa |
|---|---|---|---|
| `jpyvrrv7p` | INT8 CLI nhưng vẫn preserve I/O | FAIL | CLI path có thể tự giữ FP I/O |
| `jgkr7qwn5` | INT8 dummy calibration, không preserve I/O | PASS | HTP context binary cho vision encoder tạo được |

`jgkr7qwn5` là runtime sanity, không phải accuracy pass. Nó chứng minh đường compile/link/runtime có thể chạy, nhưng dummy calibration không thể dùng để đánh giá retrieval.

### 4.3 2026-05-27 - Vision chạy được trên RB3, nhưng fidelity fail

Vision encoder đã chạy được trên RB3 HTP bằng `qnn-net-run`:

```text
10/10 output hợp lệ
mỗi output = 768 float32 = 3072 bytes
không NaN/Inf
NetRun avg ~= 22.25 ms/image
Accelerator avg ~= 20.72 ms/image
```

Tuy nhiên dummy-cal fidelity fail:

```text
QNN dummy-cal vs PyTorch cosine_l2_mean = 0.1727
```

Sau đó tạo calibration thật 500 ảnh VN3K train:

```text
Local input: artifacts/deployment/qnn_inputs/vn3k_train_calib_500/
AI Hub dataset ID: d7x5gzne9
```

Deprecated CLI job `j5wx6x63p` fail vì AI Hub vẫn inject:

```text
--preserve_io_datatype image output_0
Tensor 'image' has a floating-point type which is not supported by the targeted device.
```

Python API flow `submit_quantize_job -> submit_compile_and_link_jobs` tạo được binary qua job `jpr9v62vp`, nhưng QNN fidelity vẫn fail:

```text
QNN calib500 vs PyTorch cosine_l2_mean = 0.1300
cosine_l2_min/max = 0.0799 / 0.1774
```

Quyết định: runtime HTP đã pass, blocker thật là fidelity sau quantization.

### 4.4 QDQ diagnosis - lỗi nằm trước QNN runtime

Để biết lỗi đến từ QNN runtime hay PTQ/QDQ, đã tải QDQ ONNX về và so trực tiếp với PyTorch.

| So sánh | Dataset | cosine_l2_mean | Kết luận |
|---|---|---:|---|
| Static ONNX vs PyTorch | `vn3k_test_10` | `1.0000` | Export/preprocess đúng |
| QDQ ONNX calib500 vs PyTorch | `vn3k_test_10` | `0.1682` | PTQ/QDQ đã phá embedding |
| QNN calib500 vs PyTorch | board output | `0.1300` | Runtime thấp hơn, nhưng lỗi đã xuất hiện từ QDQ |

Đây là bước bản lề: từ đây mọi candidate phải pass QDQ local trước khi compile/link.

### 4.5 2026-06-04 - Raw audit, calib2000, global PTQ options

Raw input audit loại trừ lỗi data/preprocess:

| Set | Listed | Valid | Bytes/file | NaN/Inf | Range | Kết luận |
|---|---:|---:|---:|---|---|---|
| `vn3k_train_calib_500` | 500 | 500 | `786432` | false | `[-1, 1]` | PASS |
| `vn3k_train_calib_2000` | 2000 | 2000 | `786432` | false | `[-1, 1]` | PASS |
| `vn3k_test_10` | 10 | 10 | `786432` | false | `[-1, 1]` | PASS |

Tạo và upload calibration 2,000 mẫu:

```text
Local input: artifacts/deployment/qnn_inputs/vn3k_train_calib_2000/
AI Hub dataset ID: d7jzjy1m2
Dataset name: msiglip-vision-vn3k-train-calib-2000
```

Các candidate global đều fail QDQ gate:

| Candidate | Config | PSNR | cosine_l2_mean | Quyết định |
|---|---|---:|---:|---|
| `jgomex415` | W8A8 calib2000 | `17.9452` | `0.1692` | Không compile/link |
| `jp2j31dm5` | W8A16 calib2000 | `17.3564` | `0.1863` | Không compile/link |
| `j5m4vjxd5` | W8A8 + `min_max` | `17.8713` | `0.1658` | Không compile/link |
| `jgl7en9l5` | Lite-MP default | `18.6722` | `0.1906` | Không compile/link |

Kết luận: tăng calibration từ 500 lên 2,000, đổi dtype/range scheme, và Lite-MP default đều chưa đủ.

### 4.6 2026-06-06 - Lite-MP sâu hơn và node-level sensitivity

Tiếp tục thử Lite-MP:

| Candidate | Config | PSNR | cosine_l2_mean | Quyết định |
|---|---|---:|---:|---|
| `j56vveq6p` | Lite-MP 30% INT16 | `17.2011` | `0.1895` | Không compile/link |
| `jpe2lnmvp` | Lite-MP 10% FP16 | `20.0953` | `0.3267` | Không compile/link nếu giữ global QDQ |

`jpe2lnmvp` là global quantize-only tốt nhất, nhưng vẫn thấp xa gate. Vì vậy chuyển sang node-level QDQ surgery để tìm lỗi nằm ở đâu.

Simple surgery từ `jpe2lnmvp`:

| Candidate | Ý tưởng | cosine_l2_mean | Kết luận |
|---|---|---:|---|
| `noop_jpe2lnmvp` | Load/save baseline | `0.326745` | Tooling không làm lệch |
| `output_float` | Gỡ QDQ output cuối | `0.326745` | Không phải output QDQ cuối |
| `final_head_float` | Giữ float head cuối | `0.314097` | Tệ hơn |
| `all_layernorm_float` | Giữ float LayerNorm | `0.328831` | Cải thiện rất nhẹ |
| `attention_score_float` | Giữ float attention score path | `0.305198` | Tệ hơn |

QDQ sensitivity decomposition:

| Candidate | Ý tưởng | cosine_l2_mean | cosine_l2_min | Kết luận |
|---|---|---:|---:|---|
| `all_qdq_float` | Bypass toàn bộ QDQ | `0.999687` | `0.999575` | Mapping đúng |
| `all_weights_float` | Bypass weight QDQ | `0.314436` | `0.217998` | Weight không phải nguyên nhân chính |
| `all_activations_float` | Bypass activation QDQ | `0.982455` | `0.958175` | Activation quantization là nguyên nhân chính |
| `matmul_gemm_weights_float` | Bypass MatMul/Gemm weight QDQ | `0.320173` | `0.238281` | Không cải thiện |
| `encoder_blocks_4_11_float` | Bypass activation QDQ blocks 4-11 | `0.957671` | `0.931005` | PASS QDQ local |

Đây là lần đầu tìm được candidate local pass gate, nhưng nó là `_float` diagnostic candidate, chưa chắc deployable.

### 4.7 2026-06-07 - Validate `_float` candidate và link fail

Validate `encoder_blocks_4_11_float` trên `vn3k_test_100`:

```text
cosine_l2_mean = 0.955255
cosine_l2_min  = 0.900515
cosine_l2_max  = 0.978287
```

Candidate vẫn pass gate, nhưng margin mỏng. Sau đó thử compile/link:

| Job | Loại | Kết quả |
|---|---|---|
| `jgd0zw76p` | Compile | SUCCESS, tạo optimized DLC |
| `jgj1wxo1g` | Link | FAIL |

Lỗi link:

```text
Tensor 'add_1003' has a floating-point type which is not supported by the targeted device.
Please quantize the model including its I/O and try again.
```

Refinement sweep xác nhận vùng nhạy là blocks 4-11:

| Candidate | Blocks giữ float | cosine_l2_mean | Kết luận |
|---|---|---:|---|
| `encoder_blocks_4_9_float` | 4-9 | `0.888407` | Thiếu 10-11 |
| `encoder_blocks_4_10_float` | 4-10 | `0.943966` | Thiếu 11 |
| `encoder_blocks_4_11_float` | 4-11 | `0.955255` | Pass |
| `encoder_blocks_5_11_float` | 5-11 | `0.904574` | Thiếu 4 |
| `encoder_blocks_6_11_float` | 6-11 | `0.697282` | Thiếu 4-5 |

Quyết định: `_float` surgery là diagnostic upper-bound, không deploy trực tiếp trên HTP. Cần all-quantized graph hoặc native quantization pipeline.

### 4.8 AIMET environment bị chặn

AIMET là hướng tự nhiên để override/exclude encoding sâu hơn, nhưng môi trường thực tế không thuận:

- Mac ARM không có wheel AIMET phù hợp.
- Ubuntu 20.04 server có Python/glibc/pip mismatch với wheel AIMET 2.31.
- Docker trên server bị chặn quyền daemon.

Quyết định: không dùng AIMET trong plan hiện tại, tìm hướng khác không phụ thuộc AIMET local.

### 4.9 2026-06-09 - ORT W8A16 pass local nhưng QNN linker fail

Thử ONNX Runtime static quantization từ FP32 ONNX để tạo graph all-quantized local.

Kết quả local:

| Candidate | Config | Dataset | cosine_l2_mean | cosine_l2_min | Kết luận |
|---|---|---|---:|---:|---|
| `minmax_calib100_w8a8` | ORT W8A8 full | `vn3k_test_10` | `0.177257` | `0.141239` | FAIL |
| `minmax_calib100_linear_w8a8` | ORT W8A8 linear-only | `vn3k_test_10` | `0.873959` | `0.804497` | Diagnostic, nonlinear ops nhạy |
| `qoperator_minmax_calib100_w8a8` | ORT QOperator W8A8 | `vn3k_test_10` | `0.168127` | `0.073877` | FAIL |
| `minmax_calib100_w8a16_opset21` | ORT W8A16 unsigned | `vn3k_test_100` | `0.999472` | `0.998398` | PASS local |
| `minmax_calib100_w8a16s_opset21` | ORT W8A16 signed | `vn3k_test_100` | `0.999472` | `0.998398` | PASS local |
| `minmax_calib100_w8a16s_gelu_qint8_opset21` | GELU output QDQ retarget INT8 | `vn3k_test_100` | `0.978237` | `0.954687` | PASS local |
| `minmax_calib100_w8a16s_matmul_act_qint8_opset21` | MatMul/Gemm activation retarget INT8 | `vn3k_test_100` | `0.969894` | `0.939456` | PASS local |

Nhưng compile/link đều fail:

| Candidate | Compile job | Link job | Lỗi |
|---|---|---|---|
| `minmax_calib100_w8a16_opset21` | `jgd03mkrp` | `jgj178xvg` | `gelu_10_DequantizeLinear_Output` còn float |
| `minmax_calib100_w8a16s_opset21` | `j5wx708jp` | `j576417rg` | `gelu_10_DequantizeLinear_Output` còn float |
| `minmax_calib100_w8a16s_opset21` + `--quantize_full_type int16` | `jgkd41xnp` | `jgzwm63xg` | context conversion exit code 14 |
| HTP FP16 internal | `jgd03ke6p` | `jpv4d8rzp` | `gelu_10_DequantizeLinear_Output` còn float |
| GELU `quint8` | `jgomr74d5` | `jp389l6z5` | `gelu_10_DequantizeLinear_Output` còn float |
| GELU `qint8` | `jgj179k7g` | `j5wx7kmzp` | `gelu_10_DequantizeLinear_Output` còn float |
| MatMul act `qint8` | `jp88xn48p` | `jgl7xd1l5` | `gelu_10_DequantizeLinear_Output` còn float |
| GELU float for requant + int16 | `jgzwm1kog` | `jpr90k37p` | context conversion exit code 14 |

Kết luận: ORT W8A16 chứng minh fidelity local có thể đạt rất cao, nhưng QNN HTP linker không chấp nhận QDQ/Dequantize pattern quanh GELU.

### 4.10 2026-06-09 - INT16 encoding tuning gần gate nhưng link fail

Thử tune activation encoding/dtype trực tiếp trên `jpe2lnmvp`:

| Candidate | Config | Dataset | cosine_l2_mean | cosine_l2_min | Kết luận |
|---|---|---|---:|---:|---|
| `jpe2lnmvp_blocks_4_11_int16_opset21` | Activation QDQ blocks 4-11 sang INT16 | `vn3k_test_100` | `0.928473` | `0.913959` | Near-pass |
| `jpe2lnmvp_blocks_0_11_int16_opset21` | Activation QDQ blocks 0-11 sang INT16 | `vn3k_test_100` | `0.947065` | `0.925417` | Gần gate nhất |
| `jpe2lnmvp_all_activations_int16_opset21` | Toàn bộ activation QDQ sang INT16 | `vn3k_test_100` | `0.945032` | `0.925546` | Near-pass |

Compile/link candidate tốt nhất:

| Candidate | Compile job | Link job | Kết quả |
|---|---|---|---|
| `jpe2lnmvp_blocks_0_11_int16_opset21` | `jgd03w76p` | `jp0kjyd25` | Compile SUCCESS, link FAIL |

Lỗi:

```text
Tensor 'add_103_updated' has a floating-point type which is not supported by the targeted device.
Please quantize the model including its I/O and try again.
```

Kết luận: INT16 tuning hữu ích cho chẩn đoán, nhưng không deployable theo pattern này.

### 4.11 2026-06-09 - Chuyển sang vision-only QAT/native quantization

Vì các hướng local QDQ surgery/ORT/INT16 đều vướng QNN linker, nhánh active chuyển sang:

```text
vision-only QAT/fine-tune-aware quantization
-> export FP32 tuned model
-> AI Hub/QNN-native quantizer tạo QDQ
-> compare QDQ
-> chỉ compile/link nếu pass hoặc near-pass gate
```

Script đã tạo:

```text
deployment/scripts/qnn/train_vision_quant_robust.py
```

Smoke CPU 1 step đã pass:

```text
Trainable parameters: 56,702,976 / 371,807,234
step=0 epoch=0 loss=0.966445 cos=0.060770 mse=0.544292
val_clean cosine_l2_mean = 0.999737
val_fake_quant cosine_l2_mean = 0.086960
```

Ghi chú: artifact `artifacts/deployment/exports/exported_model_qat_v1/` hiện tại là smoke 1-step, phải được ghi đè bằng run GPU thật trước khi export ONNX và submit AI Hub.

---

## 5. Những nguyên nhân đã loại trừ

| Nghi vấn | Bằng chứng loại trừ |
|---|---|
| ONNX export sai | Static ONNX vs PyTorch đạt `cosine_l2_mean = 1.0000` |
| Raw input sai dtype/shape/range | Raw audit pass, đúng `786432` bytes/file, no NaN/Inf, range `[-1, 1]` |
| Nhầm calibration dataset | `d7jzjy1m2` đã xác minh là `msiglip-vision-vn3k-train-calib-2000` |
| QNN runtime không chạy được | Dummy-cal và calib500 binary chạy được trên RB3, output đúng shape |
| Weight quantization là nguyên nhân chính | `all_weights_float` fail, `all_activations_float` pass |
| Chỉ output QDQ cuối gây lỗi | `output_float` không cải thiện |
| Chỉ cần tăng calibration size | Calib500 -> calib2000 không cải thiện |
| Chỉ cần đổi range scheme | `min_max` không cải thiện |
| ORT QDQ local pass là đủ deploy | ORT W8A16 local pass nhưng QNN link fail |

---

## 6. Những điều chưa được claim

- Chưa có vision QNN binary accuracy-usable.
- Chưa có QNN-vs-PyTorch candidate mới pass gate.
- Chưa compile text encoder trên HTP.
- Chưa chạy end-to-end retrieval trên board.
- Chưa có VN3K R@1 on-device.
- Chưa thể nói “deploy accuracy thành công”; hiện mới có “vision runtime HTP pass” và “nhiều diagnostic đã xác định activation quantization là blocker”.

---

## 7. Runbook trước khi chạy tiếp

### 7.1 Trước khi chạy QAT trên server

Kiểm tra server có đủ code và artifact:

```bash
test -f artifacts/deployment/exports/exported_model/model_fp32.pt
test -f artifacts/deployment/exports/exported_model/config.yaml
find artifacts/deployment/qnn_inputs/vn3k_train_calib_2000/raw -name '*.raw' | wc -l
find artifacts/deployment/qnn_inputs/vn3k_test_100/raw -name '*.raw' | wc -l
```

Kỳ vọng:

```text
vn3k_train_calib_2000 raw count = 2000
vn3k_test_100 raw count = 100
```

Chạy QAT v1:

```bash
python3 deployment/scripts/qnn/train_vision_quant_robust.py \
  --batch-size 4 \
  --epochs 1 \
  --lr 1e-5 \
  --mse-weight 0.05 \
  --device auto
```

Sau run, kiểm tra `vision_quant_robust_summary.json`. Clean student phải còn gần teacher; nếu clean cosine tụt mạnh thì dừng, không export ONNX.

### 7.2 Sau QAT: export ONNX

```bash
python3 deployment/scripts/onnx/export.py \
  --model-dir artifacts/deployment/exports/exported_model_qat_v1 \
  --precision fp32
```

Control cần pass:

```text
Static ONNX tuned vs PyTorch tuned cosine_l2_mean >= 0.999
```

Nếu static ONNX fail, sửa export trước; không submit AI Hub.

### 7.3 Submit AI Hub native quantize-only

```bash
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

Sau khi tải QDQ native về, compare:

```bash
python3 deployment/scripts/qnn/compare_onnx_with_pytorch.py \
  --onnx-model artifacts/deployment/runtime/qat/vision_quant_robust_v1/qaihub_lite_mp_qdq \
  --model-dir artifacts/deployment/exports/exported_model_qat_v1 \
  --input-dir artifacts/deployment/qnn_inputs/vn3k_test_10 \
  --precision fp32 \
  --json artifacts/deployment/runtime/qat/vision_quant_robust_v1/qdq_vs_pytorch_summary.json \
  --csv artifacts/deployment/runtime/qat/vision_quant_robust_v1/qdq_vs_pytorch.csv
```

Chỉ mở `vn3k_test_100` nếu `vn3k_test_10` pass hoặc near-pass theo rule hiện hành.

### 7.4 Chỉ compile/link khi đủ điều kiện

Compile/link chỉ được phép nếu:

- QDQ native từ QAT đạt production gate: mean `>= 0.95`, min `>= 0.90`; hoặc
- QDQ native từ QAT đạt diagnostic exception: mean `>= 0.93`, min `>= 0.88`.

Không compile/link thêm:

```text
_float surgery candidates
ORT W8A16 QDQ variants
INT16 QDQ surgery variants
```

Các pattern đó đã fail linker ở `add_1003`, `gelu_10_DequantizeLinear_Output`, hoặc `add_103_updated`.

### 7.5 Sau compile/link

Nếu link SUCCESS và có `.bin`:

1. Chạy `vn3k_test_10` trên RB3 bằng `qnn-net-run`.
2. Sync output về local.
3. Chạy `compare_qnn_with_pytorch.py`.
4. Chỉ mở `vn3k_test_100` nếu QNN mean `>= 0.90`.
5. Chỉ nghĩ tới text encoder sau khi vision QNN pass.

---

## 8. Cách ghi log cho các run tiếp theo

Mỗi run mới phải có:

| Trường | Ví dụ |
|---|---|
| Ngày | `2026-06-09` |
| Mục tiêu | “Kiểm tra QDQ native từ QAT v1” |
| Input artifact | `exported_model_qat_v1/vision_onnx` |
| Calibration data | `d7jzjy1m2` nếu dùng calib2000 |
| Fidelity gate set | `vn3k_test_10`, `vn3k_test_100` |
| Job ID | AI Hub quantize/compile/link IDs |
| Output artifact | QDQ ONNX dir hoặc QNN `.bin` |
| Metric | cosine mean/min/max, NaN/Inf |
| Quyết định | pass/fail, có compile/link tiếp không |

Daily journal chỉ ghi kết quả trong ngày. Nếu kết quả làm đổi kế hoạch, cập nhật checklist tổng ở:

```text
deployment/docs/journal/[deploy-plan]-2026-06-06.md
```

Không ghi các job AI Hub/QDQ/QNN vào `[demo-system]`.

---

## 9. Trạng thái cuối cùng tính đến 2026-06-09

| Hạng mục | Trạng thái |
|---|---|
| LoRA merge + FP32/FP16 export | DONE |
| Vision ONNX export | DONE |
| Text ONNX export | DONE |
| Vision HTP runtime proof | DONE |
| Vision HTP latency proof | DONE, khoảng `22.25 ms/image` NetRun |
| Static ONNX control | PASS |
| Raw input audit | PASS |
| Calib500 dataset | DONE, `d7x5gzne9` |
| Calib2000 dataset | DONE, `d7jzjy1m2` |
| Global AI Hub PTQ options | FAIL QDQ gate |
| Node-level sensitivity | DONE, activation blocks 4-11 là vùng nhạy |
| `_float` upper-bound | PASS local, FAIL link |
| ORT W8A16 | PASS local, FAIL link |
| INT16 encoding tuning | Near-pass local, FAIL link |
| AIMET local | Blocked by environment; không dùng trong plan hiện tại |
| QAT tooling | Script added, CPU smoke pass |
| Vision deploy accuracy | NOT PASSED |
| Text encoder HTP | NOT STARTED |
| Retrieval on board | NOT STARTED |

Kết luận thực dụng: pipeline đã chứng minh RB3 HTP có thể chạy vision encoder và các diagnostic đã khoanh vùng blocker vào activation quantization/range, đặc biệt quanh encoder blocks 4-11 và QNN linker compatibility. Bước tiếp theo là không vá QDQ ONNX thêm, mà train vision student robust với fake-quant, export FP32, để AI Hub/QNN-native quantizer tạo candidate mới, rồi quay lại gate QDQ trước khi compile/link.

---

## 10. Nguồn tổng hợp

Các file đã dùng để tổng hợp master journal này:

```text
deployment/docs/deployment-plan.md
deployment/docs/[deploy]-report-1.md
deployment/docs/journal/README.md
deployment/docs/journal/[deploy]-2026-04-15.md
deployment/docs/journal/[deploy]-2026-05-06.md
deployment/docs/journal/[deploy]-2026-05-27.md
deployment/docs/journal/[deploy-plan]-2026-06-02.md
deployment/docs/journal/[deploy]-2026-06-04.md
deployment/docs/journal/[deploy]-2026-06-05.md
deployment/docs/journal/[deploy]-2026-06-06.md
deployment/docs/journal/[deploy]-2026-06-07.md
deployment/docs/journal/[deploy]-2026-06-09.md
deployment/docs/journal/[deploy-plan]-2026-06-06.md
```

File `[demo-system]-2026-06-04.md` cố ý không được dùng vì master journal này chỉ tổng hợp deployment/QDQ/QNN/RB3, không tổng hợp demo-system.
