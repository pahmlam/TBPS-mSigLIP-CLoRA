# [Deploy Master] Nhật ký nén vision mSigLIP lên RB3/QNN

> **Cập nhật hợp nhất cuối:** 2026-06-20
> **Phạm vi:** RB3 Gen2 / QNN / AI Hub / ONNX / nén mô hình cho deployment mSigLIP.
> **Nguồn trạng thái chuẩn:** file này.
> **Nhật ký deploy duy nhất:** toàn bộ lịch sử deployment/model-compression và kết quả mới nằm ở đây.
> **Ngoài phạm vi:** ghi chú modular demo-system `[demo-system]-*.md`.

File master này hợp nhất toàn bộ lịch sử deployment/model-compression để không cần các journal deploy rời. Nó giữ lại công việc theo ngày, job ID quan trọng, artifact, metric, nguyên nhân fail, quyết định và kế hoạch tiếp theo trong một file chuẩn duy nhất.

---

## 0. Trạng Thái Hiện Tại

### Trạng Thái Deploy Hiện Tại

| Mục | Trạng thái |
|---|---|
| Checkpoint nguồn | `artifacts/models/checkpoints/epoch=56-val_score=52.28.ckpt` |
| Baseline báo cáo chính | VN3K T2I R@1 `52.28` (paper/historical baseline) |
| FP32 sanity local | VN3K T2I R@1 `52.40` tái lập local; chỉ dùng để kiểm pipeline, không làm mốc drop báo cáo |
| Best end-to-end deploy proxy | **Both-INT8 W8A8 QDQ**: T2I R@1 `50.25`, I2T R@1 `52.95`; drop T2I `-2.03` so với `52.28`, PASS target `50` |
| Artifact both-INT8 C1 | `artifacts/deployment/runtime/both_int8/both_int8_r1.json` |
| Ứng viên QDQ/retrieval tốt nhất hiện tại | **QAT v8 (learned rotation)**: SpinQuant-style Q + recipe v6 `--quant-head --quant-linears --quant-attention` |
| Job AI Hub QAT v8 | `jp24xxn65` quantize-only W8A8 |
| Artifact QDQ QAT v8 | `artifacts/deployment/runtime/rotated_w8a8_learned_qat_v8/job_jp24xxn65_qdq_onnx` |
| Cosine QDQ QAT v8 | **`0.9606 / 0.9447`** mean/min |
| Retrieval QAT v8 vision-isolation | **T2I R@1 `50.85`** (đạt deploy target 50), I2T R@1 `52.90`; drop T2I `-1.43` so với `52.28` |
| Board retrieval QAT v8 vision-isolation | **T2I R@1 `50.20`**, I2T R@1 `54.50`; drop T2I `-0.65` so với QDQ proxy `50.85`, vẫn PASS target `50` |
| Text finite/f32/link-safe path | QDQ proxy `0.9949 / 0.9912`; text-isolation T2I R@1 `51.65`, I2T R@1 `55.55`; link-safe ONNX local gate `0.99999999 / 0.99999976`; AI Hub link PASS, nhưng full text context trên board FAIL fidelity |
| Text board diagnosis | `input_ids` thật và all-zero `input_ids` cho output board giống hệt (`cos=1.0`, `max_abs=0.0` trên 10/10 mẫu); HTP context hiện không dùng dynamic token IDs đúng cách |
| Ứng viên trước đó | QAT v6 (random rotation): T2I R@1 `49.30`, QDQ `0.9491 / 0.9266` — v8 hơn `+1.55` T2I |
| Binary deploy đã verify trên board | **QAT v8** W8A8 context binary (vision-only); text full-context board bị chặn bởi dynamic embedding lookup |
| Fidelity QAT v8 trên board | `0.9585 / 0.9399` mean/min, khớp QDQ `0.9606 / 0.9447` |
| Runtime QAT v8 trên board | `33.05 ms/image`, `22.77 FPS`, context binary khoảng `90 MB` |
| Hướng tiếp theo | Không tiếp tục coi full text `.bin` nhận `input_ids` là đáng tin; thử microbenchmark `Gather`, rồi export split-text: CPU board làm embedding lookup, QNN HTP chạy transformer/head nhận `inputs_embeds` |

Cách hiểu:

- **C1 both-INT8 là số deploy proxy chính hiện tại**: vision QDQ + text QDQ đạt T2I R@1 `50.25`, vượt target `50` và giảm `-2.03` so với paper baseline `52.28`.
- **v8 vision là ablation accuracy quan trọng**: learned rotation nâng vision-isolation T2I R@1 lên `50.85`, cosine QDQ mean/min đều vượt v6. Delta `+1.55` so v6 là ablation sạch "learned vs random" (recipe v6 giữ nguyên).
- **v8 là binary deploy đã verify trên board hiện tại**: link và chạy thành công trên HTP v68, fidelity mean trên board đạt `0.9585` (rất sát với QDQ `0.9606`), full board vision retrieval đạt T2I R@1 `50.20`.
- **Text QDQ đúng nhưng full text HTP context không đáng tin**: static ONNX và QDQ đều giữ fidelity tốt, nhưng board output không đổi khi zero toàn bộ `input_ids`; do đó vấn đề nằm ở compile/link/runtime HTP cho dynamic `Gather(token_embedding.weight, input_ids)`, không phải do QAT hay calibration.

### Pipeline Vision Chuẩn

```text
LoRA checkpoint
  -> [1] Gộp LoRA vào base weights
  -> [2] Rotation residual bảo toàn mean, giữ fused LayerNorm
  -> [3] QAT/distillation trên vision model đã rotate
  -> [4] Export ONNX rotated/QAT ở opset 20
  -> [5] AI Hub W8A8 quantize, compile, link với I/O đã quantize
  -> [6] RB3 qnn-net-run + fidelity PyTorch/QDQ/board + gate retrieval
```

Command bắt buộc cho recipe hiện tại:

```bash
# [1] Gộp LoRA.
python3 deployment/scripts/lora_fp16/export.py \
  --ckpt artifacts/models/checkpoints/epoch=56-val_score=52.28.ckpt \
  --output-dir artifacts/deployment/exports/exported_model

# [2] Rotation bảo toàn mean.
python3 deployment/scripts/qnn/rotate_vision_encoder.py \
  --model-dir artifacts/deployment/exports/exported_model \
  --output-dir artifacts/deployment/exports/exported_model_rotated \
  --input-dir artifacts/deployment/qnn_inputs/vn3k_test_10 \
  --seed 2400 --skip-r2

# [3] QAT v6, ứng viên QDQ/retrieval tốt nhất hiện tại.
PYTHONUNBUFFERED=1 python deployment/scripts/qnn/train_vision_quant_robust.py \
    --model-dir artifacts/deployment/exports/exported_model_rotated \
    --train-input-dir artifacts/deployment/qnn_inputs/vn3k_train_all_4302 \
    --val-input-dir artifacts/deployment/qnn_inputs/vn3k_test_100 \
    --output-dir artifacts/deployment/exports/exported_model_rotated_qat_v6 \
    --device cuda --batch-size 16 --epochs 15 --lr 1e-5 \
    --fake-quant-observer ema --quant-head --quant-linears --quant-attention \
    --start-layer 0 --end-layer 11 --num-workers 4

# [4] Export ONNX.
python deployment/scripts/qnn/export_rotated_vision_onnx.py \
  --model-dir artifacts/deployment/exports/exported_model_rotated_qat_v6 \
  --opset 20

# [4.5] GATE: Static ONNX vs PyTorch (chạy trước khi lên AI Hub)
python3 deployment/scripts/qnn/compare_onnx_with_pytorch.py \
  --onnx-model artifacts/deployment/exports/exported_model_rotated_qat_v6/vision_onnx \
  --model-dir artifacts/deployment/exports/exported_model_rotated_qat_v6 \
  --input-dir artifacts/deployment/qnn_inputs/vn3k_test_10 \
  --precision fp32 \
  --json artifacts/deployment/exports/exported_model_rotated_qat_v6/static_vs_pytorch_summary.json \
  --csv artifacts/deployment/exports/exported_model_rotated_qat_v6/static_vs_pytorch.csv

# [5] Chạy diagnostic QDQ trước.
python3 deployment/scripts/qnn/submit_qaihub_quantize_compile.py \
  --model artifacts/deployment/exports/exported_model_rotated_qat_v6/vision_onnx \
  --calibration-data d7jzjy1m2 --weights-dtype int8 --activations-dtype int8 \
  --quantize-only --wait \
  --download-quantized artifacts/deployment/runtime/rotated_w8a8_qat_v6/job_qdq_onnx

# [5.5] GATE: Kiểm tra Cosine QDQ và đo Retrieval (trước khi link)
python3 deployment/scripts/qnn/compare_onnx_with_pytorch.py \
  --onnx-model artifacts/deployment/runtime/rotated_w8a8_qat_v6/job_j57krdwvp_qdq_onnx \
  --model-dir artifacts/deployment/exports/exported_model \
  --input-dir artifacts/deployment/qnn_inputs/vn3k_test_10 \
  --precision fp32 \
  --json artifacts/deployment/runtime/rotated_w8a8_qat_v6/qdq_vs_pytorch_summary.json \
  --csv artifacts/deployment/runtime/rotated_w8a8_qat_v6/qdq_vs_pytorch.csv

python3 deployment/scripts/qnn/eval_retrieval_quantized_vision.py --qdq-onnx artifacts/deployment/runtime/rotated_w8a8_qat_v6/job_j57krdwvp_qdq_onnx --json artifacts/deployment/runtime/rotated_w8a8_qat_v6/retrieval_r1.json

# [6] Nếu pass toàn bộ gate, compile/link đầy đủ để sinh context binary.
python3 deployment/scripts/qnn/submit_qaihub_quantize_compile.py \
  --model artifacts/deployment/exports/exported_model_rotated_qat_v6/vision_onnx \
  --calibration-data d7jzjy1m2 --weights-dtype int8 --activations-dtype int8 \
  --wait \
  --download artifacts/deployment/runtime/rotated_w8a8_qat_v5/vision_encoder.bin

# [7] Chạy thực tế trên board RB3 (sau khi adb push thư mục input và file .bin lên board).
# Lệnh chạy ngay trên NPU HTP v68 của Qualcomm RB3:
qnn-net-run \
  --backend "$QNN_LIB/libQnnHtp.so" \
  --retrieve_context artifacts/deployment/runtime/rotated_w8a8_qat_v6/vision_encoder.bin \
  --config_file deployment/config/qnn/htp_config_245.json \
  --input_list artifacts/deployment/qnn_inputs/vn3k_test_10/input_list.txt \
  --output_dir artifacts/deployment/qnn_runs/rotated_w8a8_qat_v6 \
  --profiling_level basic --perf_profile high_performance

# [8] GATE cuối: Kiểm tra Board Fidelity (so sánh kết quả từ board với PyTorch gốc).
python3 deployment/scripts/qnn/compare_qnn_with_pytorch.py \
  --qnn-output-dir artifacts/deployment/qnn_runs/rotated_w8a8_qat_v6 \
  --model-dir artifacts/deployment/exports/exported_model \
  --input-dir artifacts/deployment/qnn_inputs/vn3k_test_10 \
  --precision fp32 \
  --json artifacts/deployment/qnn_runs/rotated_w8a8_qat_v6/qnn_vs_pytorch_summary.json \
  --csv artifacts/deployment/qnn_runs/rotated_w8a8_qat_v6/qnn_vs_pytorch.csv
```

### Gate Chấp Nhận

| Gate | Ngưỡng | Mục đích |
|---|---:|---|
| Merge LoRA | không còn key `lora` / `adapter` / `base_layer` | Model export phải là base weights deploy được |
| Rotation FP32 invariance | cosine min >= `0.9999` | Rotation phải giữ nguyên hàm model |
| Static ONNX vs PyTorch | cosine mean >= `0.999` | Control export/preprocess |
| ONNX op sanity | `Pow=0`, fused `Gelu`, fused `LayerNormalization` | Tránh lộ internal GELU/RMSNorm |
| QDQ ONNX vs PyTorch | target mean >= `0.95`, min >= `0.90` | Proxy fidelity trước compile/link |
| QNN board vs PyTorch | mean >= `0.90` | Runtime trên board đủ trung thực |
| Retrieval (deploy target) | T2I R@1 >= `50.0` | Mục tiêu deploy so với paper baseline `52.28`; bất kỳ kết quả `< 50` là FAIL |

Gate cosine QDQ chỉ là proxy. Retrieval R@1 mới là quyết định cuối. Ứng viên rotation-only có cosine QDQ gần `0.90` nhưng fail retrieval ở `45.42`, chứng minh cosine riêng lẻ là chưa đủ.

### Không Lặp Lại

| Hướng | Kết quả | Quyết định |
|---|---|---|
| I/O graph FP32/FP16 trên HTP | HTP reject I/O floating-point | Dùng I/O đã quantize |
| CLI cũ `submit-compile-job --quantize_full_type` | Có thể tự inject `--preserve_io_datatype` | Dùng flow Python API/helper |
| Dummy calibration INT8 | Runtime pass, garbage fidelity | Chỉ chứng minh runtime |
| PTQ W8A8 thường | Cosine QDQ khoảng `0.13-0.17` | Không dùng được |
| Chỉ tăng số mẫu calibration | 500 lên 2000 không sửa được collapse | Không phải lỗi thiếu coverage calibration |
| `_float` QDQ surgery | Pass local nhưng HTP link fail vì internal float | Chỉ dùng diagnostic |
| ORT W8A16/QDQ variants | Fidelity local pass, HTP link fail quanh GELU/internal float | Không upload lại cùng pattern |
| Native W8A16 on v68 | QDQ có thể đạt `0.99969`, nhưng link fail vì A16 cần v73+ cho attention act-act / LayerNorm | Không deploy được trên RB3 v68 |
| Clip activation outliers | Fail; channel outlier mang tín hiệu thật | Không dùng clipping làm hướng chính |
| RMSNorm rotation | Lộ `Pow(x^2)` / `ReduceMean`; QDQ collapse | Dùng rotation bảo toàn mean và fused LayerNorm |
| R2 head-dim Hadamard | T2I `45.25`, worse than no R2 | Đã reject |
| Quantize text trước gate vision | Chỉ làm cộng thêm lỗi | Hoãn đến khi nhánh vision được chấp nhận |

---

## 1. Các Mốc Lịch Sử Đã Bao Phủ

File này hợp nhất công việc deployment/model-compression từ các mốc: 2026-04-15, 2026-05-06, 2026-05-27, 2026-06-02, 2026-06-04, 2026-06-05, 2026-06-06, 2026-06-07, 2026-06-09, 2026-06-10, 2026-06-11, 2026-06-13, 2026-06-14, 2026-06-15, 2026-06-18, và 2026-06-19.

Journal demo-system vẫn tách riêng vì nó theo dõi scaffold demo modular, không phải nén mô hình.

---

## 2. Nhật Ký Master Theo Thời Gian

### 2026-04-15 - Khảo Sát Đường Compile AI Hub

**Mục tiêu:** xác định format artifact, static shape, dtype và yêu cầu I/O biên để compile vision encoder sang QNN context binary cho HTP.

**Đầu vào và artifact:**

- Thư mục ONNX nguồn: `artifacts/deployment/exports/msiglip_lora/vision_onnx/` và `vision_onnx_fp16/`.
- Runtime/thiết bị: Qualcomm AI Hub, QNN HTP, RB3 Gen2 Vision Kit.

**Job và kết quả:**

| Job / lần thử | Kết quả |
|---|---|
| Upload file `.onnx` rời | Fail vì thiếu external weights |
| `jgn9139q5` upload thư mục ONNX | Upload OK, compile fail vì dynamic shape |
| `input_specs` dạng colon-separated | CLI `SyntaxError`; AI Hub cần literal dict Python |
| `j563onvy5` input FP32 static | HTP reject input floating-point |
| `jp2k1l3xg` flag quantize FP16 + `--quantize_io` | Vẫn preserve FP I/O; HTP reject |
| `jgdr6o86p` khai báo input float16 cho graph FP32 | Fail do dtype mismatch |
| Convert ONNX FP16 local | PASS; tạo `vision_onnx_fp16/` khoảng `178.7 MB` |
| `jp27om9r5` compile ONNX FP16 | HTP vẫn reject input floating-point |

**Bài học quan trọng:**

- ONNX có external data phải upload cả thư mục.
- QNN context binary yêu cầu input shape static; dùng `image: 1x3x256x256`.
- `input_specs` phải là literal dict Python, ví dụ:

```bash
qai-hub submit-compile-job \
  --model artifacts/deployment/exports/msiglip_lora/vision_onnx/ \
  --device "Dragonwing RB3 Gen 2 Vision Kit" \
  --compile_options " --target_runtime qnn_context_binary" \
  --input_specs '{"image": ((1, 3, 256, 256), "float32")}' \
  --wait
```

**Quyết định:** dừng thử I/O FP32/FP16 trực tiếp trên HTP. Chuyển sang I/O integer và context binary đã quantize.

### 2026-05-06 - Đường Compile INT8 Và Dummy Calibration

**Mục tiêu:** kiểm tra AI Hub có tạo được QNN context binary INT8 khi graph I/O không bị preserve floating-point hay không.

**Job và kết quả:**

| Job | Kết quả |
|---|---|
| `jpyvrrv7p` đường CLI INT8 nhưng preserve I/O | Fail vì `image` vẫn là floating-point |
| `jgkr7qwn5` INT8 dummy calibration không preserve I/O | PASS; tạo được QNN context binary, asset `mqyov9dxm` |

**Command giữ lại như sanity compile-path lịch sử:**

```bash
qai-hub submit-compile-job \
  --model artifacts/deployment/exports/msiglip_lora/vision_onnx/ \
  --device "Dragonwing RB3 Gen 2 Vision Kit" \
  --compile_options " --target_runtime qnn_context_binary --quantize_full_type int8" \
  --input_specs '{"image": ((1, 3, 256, 256), "float32")}' \
  --calibration_data none \
  --name "mSigLIP-vision-int8-dummy" \
  --wait
```

**Chẩn đoán:** blocker không phải "INT8 bất khả thi" mà là FP I/O bị preserve. Dummy calibration chỉ chứng minh runtime, không chứng minh accuracy.

**Quyết định:** chuẩn bị dữ liệu calibration VN3K thật và dùng flow mới hơn `submit_quantize_job` + `submit_compile_and_link_jobs` flow.

### 2026-05-27 - Runtime RB3 Pass Đầu Tiên, Fidelity Fail

**Mục tiêu:** tổng hợp trạng thái deploy vision sau các lần thử dummy-cal và real-cal.

**Trạng thái pipeline:**

| Mục | Trạng thái |
|---|---|
| LoRA merge + FP32/FP16 export | Done: `model_fp32.pt`, `model_fp16.pt`, `config.yaml` |
| Export ONNX vision | Done: `artifacts/deployment/exports/exported_model/vision_onnx/` |
| Export ONNX text | Done: `artifacts/deployment/exports/exported_model/text_onnx/` |
| Compile INT8 dummy-cal | Done: job `jgkr7qwn5` |
| Runtime RB3 HTP | Done: `qnn-net-run` hoàn tất `vn3k_test_10` |
| Dataset calibration thật | Đã upload: dataset `d7x5gzne9`, 500 train samples |
| Job CLI real-cal cũ | `j5wx6x63p`, fail vì `--preserve_io_datatype image output_0` vẫn còn |
| Compile/link real-cal bằng API mới | `jpr9v62vp`, tạo `vision_encoder_calib500.bin` |
| Compile text encoder | Chưa bắt đầu, bị block bởi fidelity vision |

**Kết quả runtime trên board:**

- `qnn-net-run` load và execute graph vision thành công.
- Output: 10/10 file, mỗi file `3072 bytes = 768 float32`, không NaN/Inf.
- Profile dummy-cal: NetRun avg `22.25 ms/image`, accelerator avg `20.72 ms/image`, 4 HVX threads, NetRun IPS `38.2405`.

**Kết quả fidelity:**

| Ứng viên | Cosine mean QNN vs PyTorch | Min / max | Quyết định |
|---|---:|---:|---|
| Dummy-cal `vision_encoder.bin` | `0.1727` | `0.1440 / 0.2424` | Chỉ chứng minh runtime |
| Real-cal `vision_encoder_calib500.bin` | `0.1300` | `0.0799 / 0.1774` | Fidelity fail |

**Chẩn đoán:** runtime đã ổn nhưng fidelity embedding còn sai nặng. Không chạy retrieval, `vn3k_test_100` hoặc text encoder cho tới khi sửa xong fidelity QDQ/QNN của vision.

**Kế hoạch tiếp theo:** tải QDQ ONNX từ quantization job và so QDQ với PyTorch. Nếu QDQ đã thấp thì sửa PTQ/quantization; nếu QDQ cao nhưng QNN thấp thì debug runtime/I/O.

### 2026-06-02 - Kế Hoạch Gate Fidelity PTQ/QDQ

**Mục tiêu:** định nghĩa kế hoạch có cấu trúc đầu tiên để sửa fidelity INT8 QDQ của vision.

**Bằng chứng đã khóa:**

| So sánh | Kết quả | Ý nghĩa |
|---|---:|---|
| Static ONNX vs PyTorch | `1.0000` | Export và raw input đúng |
| QDQ ONNX vs PyTorch calib500 | `0.1682` | Quantization đã làm hỏng embedding |
| QNN calib500 vs PyTorch | `0.1300` | QNN thấp hơn, nhưng lỗi bắt đầu trước runtime |

**Gate đã định nghĩa:**

- Static ONNX vs PyTorch: `cosine_l2_mean >= 0.999`.
- Smoke QDQ ONNX vs PyTorch: `cosine_l2_mean >= 0.95`, min `>= 0.90`.
- QNN vs PyTorch sau link: `cosine_l2_mean >= 0.90`.
- Full retrieval (deploy target): T2I R@1 `>= 50.0` (kết quả `< 50` là FAIL).

**Phase dự kiến:**

- Audit calibration/raw input.
- Thử calibration set lớn hơn.
- Thử W8A16, min-max, Lite-MP.
- Chỉ compile/link sau khi QDQ pass.
- Text encoder chỉ làm sau khi vision pass.

**Trạng thái sau các run sau đó:** kế hoạch này được thay bằng các kế hoạch 2026-06-06 và 2026-06-14, nhưng các gate vẫn còn hữu ích.

### 2026-06-04 - Thử Nghiệm PTQ/QDQ

**Mục tiêu:** kiểm tra liệu chất lượng raw input, kích thước calibration hoặc option quantization global của AI Hub có sửa được fidelity QDQ hay không.

**Audit raw input:**

| Set | Valid | Bytes/file | NaN/Inf | Range | Mean | Std |
|---|---:|---:|---|---|---:|---:|
| `vn3k_train_calib_500` | 500/500 | `786432` | false | `[-1, 1]` | `-0.2197` | `0.4986` |
| `vn3k_train_calib_2000` | 2000/2000 | `786432` | false | `[-1, 1]` | `-0.2302` | `0.4931` |
| `vn3k_test_10` | 10/10 | `786432` | false | `[-1, 1]` | `-0.3400` | `0.4483` |

**Artifact calibration:** `d7jzjy1m2`, `msiglip-vision-vn3k-train-calib-2000`.

**Ứng viên QDQ:**

| Job | Cấu hình | PSNR | Cosine mean QDQ | Min / max | Quyết định |
|---|---|---:|---:|---:|---|
| `jgomex415` | W8A8 calib2000 | `17.9452` | `0.1692` | `0.1239 / 0.1962` | Fail |
| `jp2j31dm5` | W8A16 calib2000 | `17.3564` | `0.1863` | `0.1366 / 0.2478` | Fail |
| `j5m4vjxd5` | W8A8 + `min_max` | `17.8713` | `0.1658` | `0.1218 / 0.2159` | Fail |
| `jgl7en9l5` | Lite-MP default | `18.6722` | `0.1906` | `0.1784 / 0.2212` | Fail |

**Chẩn đoán:** raw input hợp lệ; tăng calibration từ 500 lên 2000 không sửa được fidelity; W8A16, min-max và Lite-MP default vẫn thấp xa gate.

**Quyết định:** không compile/link các ứng viên này. Tiếp tục với biến thể Lite-MP, sau đó kiểm soát theo node.

### 2026-06-05 - Ngày Chuẩn Bị Mixed Precision

**Mục tiêu:** tiếp tục tìm kiếm Lite-MP/mixed precision.

**Trạng thái:** không ghi nhận run mới. Ứng viên global tốt nhất vẫn là `jgl7en9l5` Lite-MP default với mean `0.1906`.

**Quyết định:** run tiếp theo là Lite-MP 30% INT16, sau đó Lite-MP 10% FP16 nếu cần. Không compile/link khi chưa đạt gate QDQ.

### 2026-06-06 - Lite-MP Và QDQ Surgery Theo Node

**Mục tiêu:** đánh giá các biến thể Lite-MP và dựng tooling QDQ surgery local theo node để định vị vùng nhạy cảm.

**Ứng viên Lite-MP global:**

| Job | Cấu hình | PSNR | Cosine mean QDQ | Min / max | Quyết định |
|---|---|---:|---:|---:|---|
| `j56vveq6p` | Lite-MP 30% INT16 | `17.2011` | `0.1895` | `0.1509 / 0.2487` | Fail |
| `jpe2lnmvp` | Lite-MP 10% FP16 | `20.0953` | `0.3267` | `0.2419 / 0.4402` | Tốt nhất toàn cục nhưng vẫn fail |

**Tooling đã thêm:** `deployment/scripts/qnn/qdq_surgery.py`.

**Surgery node-level đơn giản từ `jpe2lnmvp`:**

| Ứng viên | QDQ bypass | Cosine mean | Min / max | Quyết định |
|---|---|---:|---:|---|
| `noop_jpe2lnmvp` | none | `0.326745` | `0.241872 / 0.440206` | Tooling giữ nguyên metric |
| `output_float` | final output QDQ | `0.326745` | `0.241872 / 0.440206` | Không cải thiện |
| `final_head_float` | final head | `0.314097` | `0.224059 / 0.433362` | Tệ hơn |
| `all_layernorm_float` | all LayerNorm + output | `0.328831` | `0.258684 / 0.428104` | Chỉ cải thiện nhẹ |
| `attention_score_float` | attention score path | `0.305198` | `0.243659 / 0.423269` | Tệ hơn |
| `combined_layernorm_final_head` | LN + final head | `0.316313` | `0.251170 / 0.416648` | Tệ hơn |

**Phân rã sensitivity:**

| Ứng viên | QDQ bypass | Cosine mean | Min / max | Ý nghĩa |
|---|---|---:|---:|---|
| `all_qdq_float` | toàn bộ QDQ pairs | `0.999687` | `0.999575 / 0.999802` | Mapping surgery hợp lệ |
| `all_weights_float` | initializer/weight QDQ | `0.314436` | `0.217998 / 0.460189` | Weight quantization không phải vấn đề chính |
| `all_activations_float` | activation QDQ | `0.982455` | `0.958175 / 0.993038` | Activation quantization là vấn đề chính |
| `matmul_gemm_weights_float` | weight MatMul/Gemm | `0.320173` | `0.238281 / 0.454809` | Chưa đủ |
| `encoder_blocks_0_3_float` | activation blocks 0-3 | `0.348658` | `0.271988 / 0.482678` | Cải thiện nhẹ |
| `encoder_blocks_4_7_float` | activation blocks 4-7 | `0.604363` | `0.534240 / 0.659349` | Tín hiệu mạnh |
| `encoder_blocks_8_11_float` | activation blocks 8-11 | `0.604236` | `0.429726 / 0.773699` | Tín hiệu mạnh |
| `post_layernorm_head_float` | activation post-LN/head | `0.306107` | `0.217496 / 0.423946` | Không |
| `encoder_blocks_4_11_float` | activation blocks 4-11 | `0.957671` | `0.931005 / 0.976539` | Pass QDQ local |

**Chẩn đoán:** lỗi nằm ở activation quantization/range, tập trung chủ yếu ở encoder blocks 4-11.

**Quyết định:** validate `encoder_blocks_4_11_float` trên tập lớn hơn, rồi thử compile/link. Không claim deploy success từ QDQ surgery local.

### 2026-06-07 - Xác Thực Ứng Viên Mixed-Float Và Link Fail

**Mục tiêu:** validate `encoder_blocks_4_11_float` trên `vn3k_test_100`, rồi thử compile/link QNN.

**Data:** audit raw `vn3k_test_100` pass: 100/100 hợp lệ, `786432` bytes/file, không NaN/Inf, range `[-1, 1]`.

**Validation và job:**

| Mục | Kết quả |
|---|---|
| `encoder_blocks_4_11_float` on `vn3k_test_10` | `0.957671 / 0.931005 / 0.976539` |
| `encoder_blocks_4_11_float` on `vn3k_test_100` | `0.955255 / 0.900515 / 0.978287` |
| Compile job | `jgd0zw76p` SUCCESS, optimized DLC `job_jgd0zw76p_optimized_dlc_mqv7g0yjq.dlc` |
| Link job | `jgj1wxo1g` FAILED |

**Lỗi link:**

```text
Tensor 'add_1003' has a floating-point type which is not supported by the targeted device.
Please quantize the model including its I/O and try again.
```

**Refinement sweep trên `vn3k_test_100`:**

| Ứng viên | Blocks giữ float | Mean | Min / max | Quyết định |
|---|---|---:|---:|---|
| `encoder_blocks_4_9_float` | 4-9 | `0.888407` | `0.797016 / 0.923269` | Fail |
| `encoder_blocks_4_10_float` | 4-10 | `0.943966` | `0.879753 / 0.971730` | Fail |
| `encoder_blocks_4_11_float` | 4-11 | `0.955255` | `0.900515 / 0.978287` | Pass, margin mỏng |
| `encoder_blocks_5_11_float` | 5-11 | `0.904574` | `0.821596 / 0.960436` | Fail |
| `encoder_blocks_6_11_float` | 6-11 | `0.697282` | `0.464951 / 0.858507` | Fail |

**Chẩn đoán:** `_float` surgery là upper-bound diagnostic, không deploy được trên HTP. Blocks 4-11 là vùng nhạy cảm, nhưng nghiệm cuối phải all-quantized.

**Quyết định:** dừng compile các ứng viên `_float`. Chuyển sang AIMET, ORT QDQ, INT16 encoding, QAT hoặc QNN-native quantization.

### 2026-06-09 - ORT W8A16 Pass Local, QNN Link Fail

**Mục tiêu:** thử hướng không dùng AIMET bằng ONNX Runtime static quantization.

**Tooling thêm/dùng lại:**

- `deployment/scripts/qnn/quantize_ort_static.py`
- `deployment/scripts/qnn/retarget_qdq_dtype.py`
- `deployment/scripts/qnn/tune_qdq_activation_encodings.py`
- `deployment/scripts/qnn/train_vision_quant_robust.py`
- `deployment/scripts/qnn/submit_qaihub_compile_link.py`

**Kết quả QDQ local:**

| Ứng viên | Cấu hình | Dataset | Mean | Min / max | Quyết định |
|---|---|---|---:|---:|---|
| `minmax_calib100_w8a8` | ORT QDQ W8A8 | test10 | `0.177257` | `0.141239 / 0.219684` | Fail |
| `minmax_calib100_linear_w8a8` | ORT QDQ linear-only | test10 | `0.873959` | `0.804497 / 0.930232` | Diagnostic hữu ích |
| `qoperator_minmax_calib100_w8a8` | ORT QOperator W8A8 | test10 | `0.168127` | `0.073877 / 0.219146` | Fail |
| `minmax_calib100_w8a16_opset21` | ORT QDQ W8A16 `quint16` | test100 | `0.999472` | `0.998398 / 0.999719` | Pass local |
| `minmax_calib100_w8a16s_opset21` | ORT QDQ W8A16 `qint16` | test100 | `0.999472` | `0.998398 / 0.999719` | Pass local |
| GELU output QDQ retarget qint8/quint8 | Mixed | test100 | `0.978237` | `0.954687 / 0.991896` | Pass local |
| MatMul/Gemm activation QDQ retarget qint8 | Mixed | test100 | `0.969894` | `0.939456 / 0.986330` | Pass local |
| GELU output bypass for requant | Mixed | test100 | `0.999476` | `0.998436 / 0.999724` | Pass local |

**Các lần thử compile/link:**

| Ứng viên | Compile job | Link job | Kết quả |
|---|---|---|---|
| `minmax_calib100_w8a16_opset21` | `jgd03mkrp` | `jgj178xvg` | Link fail: `gelu_10_DequantizeLinear_Output` floating-point |
| `minmax_calib100_w8a16s_opset21` | `j5wx708jp` | `j576417rg` | Cùng lỗi link fail |
| W8A16 + `--quantize_full_type int16` | `jgkd41xnp` | `jgzwm63xg` | Context conversion exit code 14 |
| W8A16 + HTP FP16 internal | `jgd03ke6p` | `jpv4d8rzp` | `gelu_10_DequantizeLinear_Output` còn floating |
| GELU qint8/quint8 variants | `jgomr74d5` / `jgj179k7g` | `jp389l6z5` / `j5wx7kmzp` | Cùng nhóm lỗi link fail |
| MatMul act qint8 variant | `jp88xn48p` | `jgl7xd1l5` | Cùng nhóm lỗi link fail |
| GELU float-for-requant + int16 | `jgzwm1kog` | `jpr90k37p` | Exit code 14 |
| `jpe2lnmvp_blocks_0_11_int16_opset21` | `jgd03w76p` | `jp0kjyd25` | Link fail: `add_103_updated` còn floating-point |

**Kết quả tuning INT16 encoding local:**

| Ứng viên | Mean | Min | Quyết định |
|---|---:|---:|---|
| `jpe2lnmvp_blocks_4_11_int16_opset21` | `0.928473` | `0.913959` | Gần pass |
| `jpe2lnmvp_blocks_0_11_int16_opset21` | `0.947065` | `0.925417` | Gần gate, link fail |
| `jpe2lnmvp_all_activations_int16_opset21` | `0.945032` | `0.925546` | Gần gate |

**Chẩn đoán:** ORT W8A16 có thể sửa fidelity local nhưng tạo pattern graph bị QNN HTP linker reject. Không tiếp tục upload các biến thể cùng pattern.

**Quyết định:** chuyển sang QAT vision-only / quantization-aware fine-tune.

### 2026-06-10 - Đánh Giá QAT v1 Vision-Only Trên Server

**Mục tiêu:** đánh giá QAT/quantization-aware fine-tune ban đầu trước khi export ONNX và submit AI Hub.

**Run:**

| Run | Cấu hình | Steps | Val clean mean/min | Val fake mean/min | Quyết định |
|---|---|---:|---:|---:|---|
| `exported_model_qat_v1_server` | batch 4, epoch 1, lr `1e-5`, mse `0.05` | 500 | `0.7929 / 0.6806` | `0.7912 / 0.6832` | Fail do clean drift |
| `exported_model_qat_v1_server_2` | lr `1e-6`, mse `0.1` | 500 | `0.6533 / 0.4248` | `0.4427 / 0.3029` | Tệ hơn |

**Chẩn đoán:** objective QAT ban đầu tối ưu fake-quant path nhưng để clean student drift khỏi teacher, làm vỡ alignment với text encoder chưa đổi.

**Cập nhật implementation lúc 22:38:**

- Đã cập nhật `deployment/scripts/qnn/train_vision_quant_robust.py`.
- Thêm objective clean-consistency:

```text
loss = fake_cosine_loss + fake_mse_weight * fake_mse
     + clean_weight * clean_cosine_loss
     + clean_mse_weight * clean_mse
```

**Kết quả smoke:**

- `python3 -m py_compile deployment/scripts/qnn/train_vision_quant_robust.py`: PASS.
- CPU 1-step smoke: clean_weight `1.0`, clean_mse_weight `0.05`, `val_clean cosine_l2_mean = 0.999737`, `val_fake_quant cosine_l2_mean = 0.086960`.

**Quyết định:** chạy QAT v2 với clean-consistency. Không export v1/v1b.

### 2026-06-11 - Sweep Trade-Off Clean/Fake Cho QAT

**Mục tiêu:** tìm candidate QAT vừa giữ clean embedding vừa cải thiện fake-quant robustness.

**Gate nội bộ trước export:**

```text
val_clean.mean >= 0.95
val_clean.min  >= 0.90
val_fake_quant.mean >= 0.78
```

**Run:**

| Run | Vùng/cấu hình | Val clean mean/min | Val fake mean/min | Quyết định |
|---|---|---:|---:|---|
| v2 full 4-11 | clean `1.0`, clean_mse `0.05` | `0.9396 / 0.8913` | `0.7826 / 0.6492` | Fake gần pass, clean fail |
| v3 full 4-11 | clean `2.0`, clean_mse `0.1` | `0.9596 / 0.9201` | `0.7238 / 0.5700` | Clean pass, fake giảm |
| v4 full 4-11 | clean `1.5`, clean_mse `0.075` | `0.9505 / 0.8998` | `0.7539 / 0.6337` | Cân bằng full-window tốt nhất, vẫn fail |
| v5 local full 4-11 | clean `1.35`, clean_mse `0.075` | `0.9557 / 0.8937` | `0.6551 / 0.5368` | Fake kém |
| v5 server full 4-11 | cùng họ cấu hình | `0.9489 / 0.8936` | `0.7388 / 0.6017` | Không tốt hơn v4 |
| blocks 6-11 v1 | narrower region | `0.9297 / 0.8606` | `0.8460 / 0.7515` | Fake tốt nhất, clean drift |

**Chẩn đoán:** full 4-11 có trade-off clean/fake. Blocks 6-11 cải thiện fake robustness, nên tiếp tục nhánh đó.

**Quyết định:** chạy blocks 6-11 v2 với ràng buộc clean mạnh hơn. Không export candidate nào từ sweep này.

### 2026-06-13 - Kiểm Tra AI Hub QDQ Cho QAT Blocks 6-11 v5

**Mục tiêu:** kiểm tra candidate PyTorch fake-quant QAT tốt nhất có transfer sang AI Hub QDQ thật hay không.

**Kết quả nội bộ QAT:**

| Run | Clean mean/min | Fake mean/min | Quyết định |
|---|---:|---:|---|
| blocks 6-11 v2 | `0.9395 / 0.8767` | `0.8369 / 0.7331` | Clean fail |
| blocks 6-11 v3 | `0.9455 / 0.8866` | `0.8225 / 0.7276` | Clean fail |
| blocks 6-11 v4 | `0.9503 / 0.8901` | `0.8108 / 0.7060` | Clean min fail |
| blocks 6-11 v5 | `0.9539 / 0.9030` | `0.7997 / 0.7022` | Ứng viên nội bộ tốt nhất |

**Kết quả export và QDQ:**

| Mục | Kết quả |
|---|---|
| Static ONNX vs PyTorch QAT v5 | `1.000000`, min `0.9999998` |
| AI Hub Lite-MP QDQ vs PyTorch QAT v5 | `0.244236`, min `0.203228`, max `0.276519` |
| AI Hub default QDQ job | `jgl79ndj5`, mean `0.252687`, min `0.211948` |

**QDQ surgery trên QAT v5 Lite-MP QDQ:**

| Ứng viên | Mean | Min / max | Quyết định |
|---|---:|---:|---|
| no-op | `0.244236` | `0.203228 / 0.276519` | Baseline tooling |
| `all_qdq_float` | `0.999779` | `0.999683 / 0.999836` | Mapping hợp lệ |
| `all_weights_float` | `0.511043` | `0.378787 / 0.671750` | Weight QDQ có góp lỗi |
| `all_activations_float` | NaN | `any_onnx_nan = true` | Không hữu ích |
| blocks 0-3 float | `0.169657` | `0.119832 / 0.237798` | Fail |
| blocks 4-7 float | `0.154641` | `0.094580 / 0.205820` | Fail |
| blocks 8-11 float | `0.014968` | `-0.016124 / 0.043670` | Fail |
| post-LN/head float | NaN | `any_onnx_nan = true` | Fail |
| all weights + blocks 4-7 float | `0.858946` | `0.802091 / 0.896209` | Tín hiệu mạnh nhất, vẫn fail |
| all weights + blocks 8-11 float | `0.745280` | `0.626759 / 0.834999` | Có tín hiệu |

**Cập nhật tooling:** `compare_qnn_with_pytorch.py` đã được patch để `_vector_stats()` không crash trên NaN/Inf; nó ghi lại `any_onnx_nan` / `any_qnn_nan`.

**Chẩn đoán:** proxy PyTorch fake-quant chưa khớp đủ tốt với AI Hub QDQ. Với QAT v5, vấn đề là tương tác giữa weight QDQ và activation QDQ, đặc biệt ở blocks 4-7/4-11.

**Quyết định:** không compile/link các ứng viên QDQ QAT v5 từ nhánh này. Tiếp tục diagnostic bằng continuous windows và encoding tuning.

### 2026-06-14 - Diagnostic QAT v5, Đột Phá GELU, W8A16 Link Fail

#### Buổi Sáng: Chẩn Đoán QDQ QAT v5 Và Tuning Encoding

**Kết quả continuous-window surgery:**

| Candidate | QDQ pairs chọn | Mean | Min / max | Quyết định |
|---|---:|---:|---:|---|
| all weights + blocks 4-9 | 318 | `0.947507` | `0.913112 / 0.965084` | Gần pass diagnostic |
| all weights + blocks 4-10 | 352 | `0.964700` | `0.930359 / 0.980784` | Diagnostic pass |
| all weights + blocks 4-11 | 386 | `0.970312` | `0.939976 / 0.985145` | Diagnostic tốt nhất |

**Phân tích encoding cho `qaihub_lite_mp_qdq`:**

- Tổng QDQ pairs: `575`.
- Activation QDQ pairs: `462`.
- Activation QDQ pairs trong blocks 4-11: `272`.
- `real_abs_max_mean` trung bình trong blocks 4-11: `2075.381531`.
- Lớn nhất `real_abs_max_mean`: `138717.701084`.

**Sweep INT8 max-abs activation trên blocks 4-11:**

| Max abs | Pairs đã đổi | Mean | Min / max | Quyết định |
|---:|---:|---:|---:|---|
| 8 | 94 | `0.184424` | `0.130205 / 0.230198` | Fail |
| 16 | 82 | `0.225542` | `0.147079 / 0.373379` | Fail |
| 24 | 72 | `0.204669` | `0.159287 / 0.294313` | Fail |
| 32 | 62 | `0.206731` | `0.159230 / 0.260376` | Fail |
| 48 | 51 | `0.197173` | `0.138325 / 0.260081` | Fail |
| 64 | 48 | `0.211570` | `0.147779 / 0.311365` | Fail |
| 128 | 42 | `0.216234` | `0.144512 / 0.328789` | Fail |
| 256 | 31 | `0.247837` | `0.198776 / 0.281554` | Tốt nhất nhưng vẫn fail |

**AI Hub native W8A16 QAT v5:**

- Artifact: `artifacts/deployment/runtime/qat/blocks_6_11_v5/qaihub_w8a16_qdq/model.onnx`.
- Kết quả: mean `0.155494`, min `0.115508`, max `0.244340`, tệ hơn Lite-MP INT8.

**Quyết định:** các ứng viên này không deploy được. Mở kế hoạch mới quanh diagnostic activation-outlier và năng lực QNN-native.

#### 15:34 - Chẩn Đoán Pre-Flight P0

**P0.1 profile activation outlier:**

- Script: `deployment/scripts/qnn/profile_activation_outliers.py`.
- Output: `artifacts/deployment/runtime/diag/activation_outliers/blocks_4_11/`.
- Kết luận: `CONCENTRATED_FIXED`.
- Ví dụ nặng nhất: `val_473` block 5, abs max `5224`, concentration `876x`, stability `1.00`, p99.99 `34.5`.
- Channel lặp lại: `523`, `415`, `7`, `528` across blocks 9-11.
- Một số tensor residual late-block là diffuse với concentration `~4-6x` và p99.99 khoảng `2200`, nhưng các channel cố định tập trung vẫn là failure mode chính.

**P0.2 audit W8A16:**

- Lite-MP INT8 và native W8A16 đều dùng activation encoding per-tensor.
- Native W8A16 có `20/20` activation tensor per-tensor, max real_abs_max khoảng `137236`.
- Kết luận: lỗi nằm ở granularity/range, không phải chỉ do bit-width.

**P0.3 audit toolchain:**

- Tìm thấy trên `qc-rb3g2`: `qairt-converter`, `qairt-quantizer`, `qnn-onnx-converter`, `qnn-model-lib-generator`, `qnn-context-binary-generator`, `qnn-net-run`, `qnn-throughput-net-run`, `qnn-profile-viewer`, `qnn-platform-validator`.
- Chỉ thiếu các alias QAIRT đã có bản tương đương `qnn-*`.

**Kết luận nhanh P0.4:**

- `qairt-quantizer` (`libPyIrQuantizer.so`) chỉ chạy x86.
- Board aarch64 có thể convert/run nhưng không chạy được quantizer.
- Quantization cần AI Hub hoặc host x86.

#### 17:03 - Đột Phá Root Cause GELU

**Kết quả SmoothQuant:**

- Script mới: `deployment/scripts/qnn/smoothquant_equalize.py`.
- Fold LN vào projection input; abs max LN-output `121.2 -> 6.4`.
- Cosine bất biến FP32 `1.000000`.
- AI Hub SmoothQuant job `jpv4j737p` tạo mean around `0.15`, trung tính so với baseline full INT8, không cải thiện.

**Postmortem:**

- Các activation tensor top với real_abs_max `119406`, `66048`, `62813`, `56184` đến từ các node ONNX `Pow`.
- Root cause: export opset-18 decompose tanh-GELU thành `Pow(x,3)` và `Tanh`.
- `node_Pow_446 = linear_34 ^ 3.0` nằm trong chuỗi xấp xỉ GELU.
- Term `x^3` làm per-tensor activation quantization collapse.

**Fix:**

- Re-export ở opset 20 để `Gelu` được fused.
- Artifact: `artifacts/deployment/exports/exported_model_gelu20/vision_onnx/`.
- Op counts: `Pow=0`, `Tanh=0`, `Gelu=13`.
- Static control vs PyTorch: mean `1.0000000`, min `0.9999999`.

#### 17:14 - GELU20 INT8 Vẫn Fail Do Concentration

**Job và kết quả:**

- AI Hub gelu20 W8A8 quantize-only job `jgj1jqm8g`.
- QDQ vs PyTorch: mean `0.142`, min `0.090`.
- QDQ giữ fused `Gelu`: `Pow=0`, `Tanh=0`, `Gelu=13`.
- Top activation range giảm từ `119406` to `255.8`.

**Surgery:**

- `all_activations_float`: `0.982 / 0.958`, pass.
- `all_weights_float`: `0.138 / 0.071`, fail.

**Chẩn đoán:** fused GELU đã sửa magnitude cubic, nhưng per-tensor activation quantization vẫn collapse vì một vài channel cố định chi phối. Cần per-channel activation hoặc representation equalization.

#### 17:40 - W8A16 Pass Fidelity

**Job và kết quả:**

- AI Hub gelu20 W8A16 quantize-only job `j5wxjzk4p`.
- Cấu hình W8A16: `--weights-dtype int8 --activations-dtype int16`.
- QDQ vs PyTorch on `vn3k_test_10`: mean `0.99969`, min `0.99950`, max `0.99983`, `l2_l2_mean = 0.0249`, không NaN.

**Text prep hoàn tất in parallel:**

- Export text opset-20: `Pow=0`, `Tanh=0`, `Gelu=12`.
- `prepare_vn3k_text_inputs.py`: tokenize caption VN3K bằng `get_tokenizer` và `bases.tokenize`, tạo hai input int64 `[1,64]`.
- `compare_text_onnx_with_pytorch.py`: static control mean `0.99999993`.

**Quyết định:** fidelity W8A16 gần như không mất mát. Thử compile/link đầy đủ.

#### 18:10 - W8A16 Link Fail Trên HTP v68

**Job:**

| Job | Loại | Kết quả |
|---|---|---|
| `jp13rmzk5` | W8A16 quantize full flow | SUCCESS |
| `j5wxjrwjp` | compile DLC | SUCCESS |
| `j576q80rg` | link context binary | FAILED exit code 14 |

**Lỗi:**

```text
[ERROR] has incorrect Value 68, expected >= 73.
[ERROR] QnnBackend_validateOpConfig failed 3110
[ERROR] Failed to validate op node_MatMul_774 với error 0xc26
```

**Chẩn đoán:**

- `node_MatMul_774` là attention matmul activation x activation.
- A16 cho op này cần HTP v73+, trong khi RB3 Gen2 là v68.
- Đây là lỗi support op phần cứng, không phải lỗi fidelity.

**Quyết định:** W8A16 không deploy được trên v68. Cần all-W8A8 hoặc mixed scheme tránh A16 ở các op không hỗ trợ.

### 2026-06-15 - INT8 Dựa Trên Rotation, Deploy Board, QAT Recovery, v5 Tốt Nhất

#### 1. Mixed A16 Fail Rộng Hơn

**Mục tiêu:** kiểm tra liệu chỉ attention matmul cần INT8 còn residual/MLP có thể giữ A16 hay không.

**Job và kết quả:**

| Job | Loại | Kết quả |
|---|---|---|
| `jp48z6lqg` | compile mixed-int, attention int8 + phần còn lại int16 | SUCCESS |
| `jpxmw8kjg` | link mixed-int | FAILED exit 14 ở `node_layer_norm` |

**Can thiệp local:** mở rộng `retarget_qdq_dtype.py` với `--act-matmul-inputs`, hạ 56 activation-matmul QDQ pairs xuống int8 trong khi giữ residual/MLP int16. Fidelity local là `0.9943`.

**Chẩn đoán:** v68 cũng reject A16 LayerNorm. Giới hạn A16 đủ rộng để W8A16/mixed-A16 không khả thi trên RB3 v68.

#### 2. Clipping W8A8 Fail

**Thử nghiệm:** `tune_qdq_activation_encodings.py --all-activations --max-abs` với các giá trị `{4,8,16,32,64}` trên job gelu20 W8A8 `jgj1jqm8g`.

**Kết quả:** cosine mean `0.118`, `0.155`, `0.128`, `0.405`, `0.190`; tất cả fail.

**Chẩn đoán:** các channel outlier mang tín hiệu thật, không thể chỉ clip đơn giản.

#### 3. Quyết Định Implement Rotation

**Chẩn đoán đã khóa:**

- v68 hỗ trợ W8A8 khá rộng.
- A16 bị block đúng ở nơi model cần: attention act-act và LayerNorm.
- W8A8 thường fail vì activation concentration chi phối scale per-tensor.
- Clipping làm mất thông tin.

**Quyết định:** implement residual rotation kiểu QuaRot/SliceGPT trong PyTorch, fold vào weights, không thêm runtime op.

**Recipe ban đầu:**

- Phase A: fold tham số affine của LayerNorm vào reader.
- Phase B: rotate residual stream với orthogonal Q.
- Writer nhận `Q W` / `Q b`.
- Reader nhận `W Q^T`.
- Head K/V nhận `Q^T` trong khi learned query/probe giữ nguyên.

#### 4. M1 Thử Rotation Lần Đầu Và Bug Reload

**Script:** `deployment/scripts/qnn/rotate_vision_encoder.py`.

**Kết quả ban đầu:**

- Bất biến FP32 in-memory: cosine `1.0`, min `0.99999988`.
- Residual concentration: `220x -> 4.2x`, abs max `267.6 -> 140.2`.

**Bug:** state_dict đã lưu bị reload vào model LayerNorm thường thay vì module RMSNormNoAffine, làm reload cosine còn `0.11`.

**Fix:** thêm `load_rotated_model(model_dir)`, swap module LN sang `RMSNormNoAffine` trước khi load.

**Verify reload:** cosine trở lại `1.0`, min `0.99999988`.

#### 5. M2 RMSNorm Pass Static Control Nhưng M3 Fail

**Script export:** `deployment/scripts/qnn/export_rotated_vision_onnx.py`.

**Kết quả M2:**

- Static ONNX-rotated vs PyTorch-rotated: mean `1.0000000`, min `0.99999988`.
- Op counts: `Gelu=13`, `LayerNormalization=1`, `Pow=25`, `ReduceMean=25`.

**Job M3:** W8A8 rotated RMSNorm quantize job `jp48z6o2g`.

**Kết quả M3:** QDQ vs PyTorch gốc mean `0.162`, gần như không đổi so với W8A8 chưa rotate.

**Surgery:** `all_activations_float = 0.978`, `all_weights_float = 0.162`.

**Root cause:** thay LayerNorm bằng RMSNorm làm lộ `Pow(x^2)`, `ReduceMean` và internal division cho per-tensor activation quantization.

**Quyết định:** bỏ conversion RMSNorm. Giữ fused LayerNorm.

#### 6. Fix Rotation Bảo Toàn Mean

**Fix:** xây Q sao cho `Q * 1 = 1`; rotation chỉ tác động lên không gian con trực giao với hướng mean.

**Vì sao:** với LayerNorm identity-affine, Q trực giao bảo toàn mean cho `LN(Qx) = Q LN(x)`, nên fused LayerNorm có thể ở lại trong graph.

**Triển khai:**

- `_mean_preserving_orthogonal(dim, seed)` xây `Q = U blockdiag(1, Rc) U^T`.
- Fold gamma/beta vào reader.
- Đặt affine của LayerNorm thành identity nhưng vẫn giữ module LayerNorm.
- Dùng Q cho residual writer và `Q^T` cho reader.
- `load_rotated_model` đơn giản hóa lại về load model chuẩn.

**Validation:**

- Phase A/B invariance: `1.0`, min `0.99999988`.
- Q orthogonality error: `3e-15`.
- `Q * 1 = 1` error: `1e-14`.
- Reload invariance: `1.0`.
- Residual concentration: `252x -> 5.3x`.

#### 7. M2 + M3 Rotation Bảo Toàn Mean

**Kết quả export M2:**

- Re-export `exported_model_rotated`.
- Op counts: `LayerNormalization=26`, `Pow=0`, `Gelu=13`, `ReduceMean=0`.
- Static control: khoảng `1.0`.

**Job W8A8 M3:** `jpr9zro9p`.

**Kết quả M3:** QDQ vs PyTorch gốc:

- Mean `0.8975`.
- Min `0.8747`.
- Max `0.9297`.
- Không NaN.

**Chẩn đoán:** Hành trình W8A8 đi từ `0.14` chưa rotate sang `0.16` RMSNorm rotation rồi tới `0.90` với rotation bảo toàn mean. Lỗi còn lại là lỗi INT8 per-tensor diffuse tích lũy qua 12 block.

**Quyết định:** compile/link và chạy board; sau đó đo retrieval, không chỉ cosine.

#### 8. M4 Link Thành Công Và M5 Chạy Board

**Job:**

| Job | Loại | Kết quả |
|---|---|---|
| `jpv4j8lkp` | W8A8 quantize full flow | SUCCESS |
| `jpxmwq8lg` | compile DLC | SUCCESS |
| `jp2j211q5` | link context binary | SUCCESS |

**Đầu ra:** `artifacts/deployment/runtime/rotated_w8a8_v2/vision_encoder.bin`, khoảng `89.7 MB`.

**Chạy board:**

- `qnn-net-run` trên `qc-rb3g2`, HTP v68, profiling basic, high performance.
- 10 inferences hoàn tất.
- Đầu ra sync về `artifacts/deployment/qnn_runs/rotated_w8a8_v2/`.

**Fidelity trên board:**

- QNN vs PyTorch mean `0.8982`.
- Min `0.8606`.
- Max `0.9283`.
- Không NaN.
- Khớp QDQ `0.8975` vớiin khoảng `0.0007`.

**Runtime trên board:**

- NetRun avg mỗi inference `34250 us` = `34.25 ms`.
- Min/max `32958 / 35388 us`.
- Accelerator execute `32478 us` = `32.5 ms`.
- 4 HVX threads.
- Throughput `22.5 inf/sec`.
- Init/load binary `54688 us` = `54.7 ms`.

**Quyết định:** runtime HTP trung thực; vấn đề còn lại là retrieval accuracy, không phải drift phần cứng.

#### 9. Rotation-Only Retrieval Fail

**Phương pháp đánh giá:**

- Script: `deployment/scripts/qnn/eval_retrieval_quantized_vision.py`.
- Embedding ảnh: QDQ ONNX `runtime/rotated_w8a8_v2/job_jpr9zro9p_qdq_onnx`.
- Embedding text: FP32 PyTorch `encode_text`.
- Dataset: VN3K test full, 2000 images + 4000 captions.
- Metric bám theo `LitTBPS._compute_metrics`: raw pooler features và dot product, `utils.metrics.rank`, không dùng `Evaluator` normalized generic.

**Sanity:** baseline FP32 T2I R@1 `52.40`, gần với lịch sử `52.28`. Các drop báo cáo dùng mốc paper baseline `52.28`.

**Kết quả:**

| Metric | FP32 sanity | Rotation W8A8 | Drop vs `52.28`/sanity |
|---|---:|---:|---:|
| T2I R@1 | `52.40` | `45.42` | `-6.86` vs `52.28` |
| I2T R@1 | `55.30` | `49.40` | `-5.90` |
| T2I R@5 / R@10 | `79.38 / 87.80` | `73.38 / 83.12` | - |

**Artifact:**

- `artifacts/deployment/runtime/rotated_w8a8_v2/retrieval_r1.json`
- `artifacts/deployment/runtime/rotated_w8a8_v2/retrieval_embeddings.npz`
- `artifacts/deployment/runtime/rotated_w8a8_v2/retrieval_full.log`

**Quyết định:** chưa quantize text. Chỉ riêng quantization vision đã fail gate, nên both-INT8 sẽ tệ hơn hoặc bằng. Cải thiện vision trước.

#### 10. Reject R2 Head-Dim Hadamard

**Triển khai:** `rotate_vision_encoder.py` phase C, Hadamard block-diagonal trên head_dim=64 fold vào `v_proj` output và `out_proj` input. Không runtime op.

**Gate local:** invariance cosine `1.0`, static ONNX control `1.0`, op counts `Gelu=13`, `LayerNormalization=26`, `Pow=0`.

**Job AI Hub:** `jgomykjq5` quantize-only W8A8.

**Kết quả:**

| Metric | Không R2 | R2 |
|---|---:|---:|
| QDQ cosine mean/min | `0.8975 / 0.8747` | `0.9006 / 0.8564` |
| T2I R@1 | `45.42` | `45.25` |
| I2T R@1 | `49.40` | `47.15` |

**Chẩn đoán:** R2 nhắm vào attention value path, nhưng lỗi còn lại chủ yếu là MLP/activation noise. R2 làm min cosine và retrieval giảm nhẹ.

**Quyết định:** reject R2. Chuyển sang QAT/quant-robust finetune.

#### 11. Diễn Tiến QAT Từ v1 Đến v8

| Vòng | Thay đổi chính | Job AI Hub | QDQ mean/min | T2I R@1 | I2T R@1 | Quyết định |
|---|---|---|---:|---:|---:|---|
| rotation-only | không QAT | `jpr9zro9p` | `0.8975 / 0.8747` | `45.42` | `49.40` | Deploy được trên board, retrieval fail |
| QAT v1 | per-sample fake-quant, local MPS | `jgzwej1kg` | `0.9223 / 0.8917` | `46.92` | `50.45` | Có ích, nhưng sim quá dễ |
| QAT v2 | per-tensor fake-quant, toàn bộ layers, 5 epochs | `jgomym0x5` | `0.9281 / 0.9093` | `47.80` | `51.65` | Dưới target 50 |
| QAT v3 | EMA observer, toàn bộ layers, 8 epochs | `jp383qmn5` | `0.9353 / 0.919` | `48.20` | `52.30` | INT8 ổn định đầu tiên, vẫn < target 50 |
| QAT v4 | + `--quant-head`, 12 epochs | `jgd09l96p` | `0.9364 / 0.9091` | `48.50` | `52.95` | Binary deploy đã verify trên board |
| QAT v5 | + `--quant-linears`, 15 epochs | `jpxm2w0lg` | `0.9437 / 0.9311` | `49.25` | `53.40` | Đã chạm ngưỡng tác dụng của single linear |
| QAT v6 | + `--quant-attention` | `j57krdwvp` | `0.9491 / 0.9266` | `49.30` | `53.85` | Trần coverage của random rotation |
| QAT v7 | v6 coverage + cosine LR + lr `2e-5`, 20 epochs | `jpve62jmg` | `0.9485 / 0.9083` | `48.38` | `53.05` | **Regress** so v6: cosine+lr 2e-5 kém hơn const lr 1e-5 |
| QAT v8 | **Learned rotation (SpinQuant-style)** + recipe **v6** (const lr 1e-5, 15 ep) | `jp24xxn65` | `0.9606 / 0.9447` | **`50.85`** | `52.90` | **WIN**: vượt v6 `+1.55` T2I, đạt deploy target `50`; learned > random |

**Chi tiết QAT v1:**

- Teacher: rotated FP32 frozen.
- Student: rotated + fake-quant trên GELU output và residual.
- Val clean cosine `0.9951`, fake-quant sim `0.9754`.
- Đã xác định transfer gap: per-sample fake-quant quá dễ.

**Chi tiết QAT v2:**

- Per-tensor fake-quant, layers 0-11, 4302 train images, batch 24, 5 epochs, 900 steps, lr `1e-5`, RTX 3060.
- Val sim fake-quant cosine `0.9803`.
- Kết quả `47.80`, chỉ thấp hơn deploy gate `0.20`.

**Chi tiết QAT v3:**

- EMA observer với per-tensor scale, matching AI Hub calibrate-once behavior.
- 4302 images, batch 24, 8 epochs, 1440 steps.
- Retrieval:
  - T2I `48.20 / 75.42 / 85.10`, mAP `53.39`, mINP `46.60`.
  - I2T `52.30 / 78.90 / 86.85`, mAP `47.89`, mINP `31.03`.
- Vẫn dưới deploy target `50` (`48.20`).

**Chi tiết QAT v4:**

- EMA + `--quant-head`, bao phủ post-layernorm/head attention/head MLP.
- 12 epochs, 2160 steps, trainable 92.1M.
- Đã verify trên board:
  - Board fidelity `0.9363 / 0.9068`.
  - QDQ fidelity `0.9364 / 0.9091`.
  - NetRun avg `32.70 ms`, accelerator `31.2 ms`, `22.88 FPS`, init `53.3 ms`.
- Runtime/size giống v2/rotation nhưng retrieval tốt hơn nhiều.

**Chi tiết QAT v5:**

- EMA + `--quant-head --quant-linears`.
- Fake-quant bao phủ mọi `nn.Linear` output: q/k/v/out_proj, fc1, fc2, head linears.
- Layers 0-11, 4302 images, batch 24, 15 epochs, 2700 steps.
- Val sim cosine `0.978`, min `0.9453`.
- QDQ min tăng `0.9091 -> 0.9311`, chứng minh plateau v4 là coverage gap, không phải trần cơ bản của W8A8.
- Retrieval: T2I R@1 `49.25`, R@5 `77.28`, R@10 `85.80`, mAP `54.55`, mINP `47.86`; I2T R@1 `53.40`.
- Khoảng cách còn lại tới deploy target 50: `0.75`.

**Chi tiết QAT v6:**

- Kế thừa v5 và thêm `--quant-attention`.
- Bao phủ thêm 2 functional attention op: `Q.K^T` (scores), softmax probs và context `softmax.V`.
- QDQ Job: `j57krdwvp`.
- QDQ Cosine: mean `0.9491` (tăng nhẹ từ `0.9437`), min `0.9266` (giảm nhẹ từ `0.9311`).
- Retrieval: T2I R@1 `49.30` (tăng từ `49.25`), R@5 `77.38`, R@10 `86.28`, mAP `54.87`, mINP `48.17`; I2T R@1 `53.85` (tăng từ `53.40`).
- Mặc dù R@1 tiếp tục lập đỉnh mới cho W8A8, mức tăng khá nhỏ so với v5, cho thấy `--quant-attention` đem lại hiệu quả marginal.

**Chi tiết QAT v7:** _(REGRESS — cosine + lr 2e-5 kém hơn v6)_

- Kế thừa coverage v6 (`--quant-head --quant-linears --quant-attention`, EMA observer).
- Thay đổi vs v6: cosine LR schedule (`--lr-schedule cosine --warmup-frac 0.05 --min-lr-ratio 0.02`), **lr `2e-5` (gấp đôi v6)**, 20 epochs.
- Base `exported_model_rotated` (random mean-preserving), job QDQ `jpve62jmg`.
- Kết quả: QDQ mean `0.9485` (≈ v6 `0.9491`), min `0.9083` (**giảm** từ v6 `0.9266`); retrieval T2I R@1 `48.38` (giảm từ `49.30`), I2T `53.05` (giảm từ `53.85`). Baseline sanity FP32 `52.40` pass.
- **Kết luận:** thay đổi 3 thứ cùng lúc (cosine + lr×2 + epochs) → regress `-0.93` R@1. Nghi can chính là **lr 2e-5 overshoot** quanh điểm minimum lượng tử hóa; cosine floor (`4e-7`) không cứu lại được hỏng ở epoch đầu.
- **Hệ quả cho v8:** recipe tốt nhất vẫn là **v6** (const lr `1e-5`, 15 epochs). v8 (learned rotation) nên dùng recipe v6 để delta R@1 cô lập đúng phần *rotation* (learned vs random), không lẫn schedule đã thua.

**Chi tiết QAT v8 — Learned Rotation (SpinQuant-style):** _(WIN — learned > random, đạt deploy target 50)_

- **Kết quả thực tế (đã chạy 2026-06-18):**
  - Learned rotation gate PASS: objective `46281 → 861` (−98.1%), max|a| `123.8 → 14.56` (−88.2%), cosine min `0.99999988`, orth_err `3.1e-15`, mean_err `4.0e-15` (256 calib, 32 tok/ảnh, 3000 step, lr `2e-3`).
  - Export static-vs-pytorch sanity: cosine mean `1.0000`, L2 `1.4e-6` (rotation output-invariant, xác nhận fold đúng).
  - QDQ job AI Hub `jp24xxn65`, QDQ-vs-pytorch fidelity: cosine mean **`0.9606`**, min **`0.9447`** (cả hai **tốt hơn** v6 `0.9491 / 0.9266`).
  - Retrieval (full 2000 gallery / 4000 query): baseline FP32 sanity T2I `52.40` PASS; **vision-INT8 T2I R@1 `50.85`** (R@5 `77.48`, R@10 `86.98`, mAP `55.79`, mINP `49.24`), I2T R@1 `52.90`. Drop T2I báo cáo là `−1.43` so với paper baseline `52.28`.
  - Vision ONNX latency CPU `81.9 ms/ảnh`.
- **Kết luận ablation (learned vs random, recipe v6 giữ nguyên):**
  - T2I R@1: **`50.85` (learned) vs `49.30` (random v6) → +1.55**. Đây là delta sạch chỉ do `Q`, vì mọi thứ khác (coverage, lr, epochs, base FP32) giống hệt.
  - QDQ fidelity cũng nhích lên cả mean lẫn min ⇒ tối ưu max-abs² thật sự thắt được scale INT8 như lý thuyết.
  - **Đạt deploy target `50`** lần đầu cho vision-INT8 W8A8 trên QDQ proxy.
  - Đánh đổi nhỏ: I2T R@1 `52.90 < 53.85` (v6). Vì metric chính của đề tài là **T2I R@1**, learned rotation là lựa chọn deploy.
- **Hệ quả cho text:** chốt dùng **learned rotation** cho text (`learn_rotation_text.py`), không dùng random nữa.

- **Điểm mới về phương pháp:** thay vì dùng ma trận quay ngẫu nhiên bảo toàn mean (v1–v7), v8 *học* ma trận quay `Q` bằng cách tối ưu trực tiếp đại lượng quyết định scale INT8.
- **Công thức.** Gọi `a` là activation tại các "rotation site" của residual stream (output của `layer_norm1/2`, `out_proj`, `fc2`, `post_layernorm`). Per-tensor INT8 scale là `s = max|a| / 127`. Ta tối thiểu hóa tổng bình phương biên độ cực đại sau khi quay:

  ```
  min_Q  Σ_sites ( max_ij | (a · Qᵀ)_ij | )²
  s.t.   Q Qᵀ = I            (trực giao, bảo toàn norm)
         Q · 1 = 1           (bảo toàn mean → giữ được LayerNorm đã fuse)
  ```

- **Tại sao là max-abs² chứ không phải quant-MSE.** Với straight-through estimator (STE), gradient của `q(x) − x` bị detach nên mục tiêu quant-MSE có gradient bằng 0 theo `Q`. Ngược lại, `max|a·Qᵀ|²` khả vi theo `Q` và *chính là* đại lượng đặt ra bước lượng tử hóa per-tensor. Giảm nó ⇒ giảm scale ⇒ giảm clipping/rounding error ⇒ tăng R@1.
- **Tham số hóa Cayley.** Để giữ `Q` trực giao chính xác suốt quá trình Adam: `Q = U · blockdiag(1, Cayley(skew)) · Uᵀ`, với `U[:,0] = 1/√d` ghim ràng buộc bảo toàn mean (vector 1 là eigenvector). `Cayley(S) = (I − S)(I + S)⁻¹` với `S` phản đối xứng luôn cho ma trận trực giao.
- **Quy trình.** Phase A (fold affine của LayerNorm vào reader, đặt LN identity) → cache activation tại rotation site trên calib → tối ưu `Q` → Phase B (fold `Q` vào weight). Output là model drop-in tương đương `exported_model_rotated` về shape, nạp thẳng vào pipeline export/QAT/quantize/eval.
- **Gate output-invariance.** Vì rotation chỉ là đổi cơ sở trực giao bảo toàn mean, embedding FP32 phải bất biến: gate yêu cầu cosine(ref, rotated) min ≥ `0.9999` trước khi lưu. Smoke test (8 ảnh, 80 step): objective `29268 → 1209`, max|a| `122 → 24`, cosine min `0.99999982`, orth_err `4.2e-15`, mean_err `4.8e-14`.
- **Ablation.** Dùng recipe **v6** (const lr `1e-5`, 15 ep) cho v8 (vì v7 cosine+lr2e-5 đã regress). v8 (learned + recipe v6) vs **v6** (random + recipe v6) giống nhau mọi thứ trừ `Q` ⇒ delta R@1 là ablation sạch "learned vs random rotation under identical QAT".
- Script: `deployment/scripts/qnn/learn_rotation.py`.
- Artifact dự kiến: `exported_model_rotated_learned` (Q đã fold) → `exported_model_rotated_learned_qat_v8`.

#### 12. Hiểu Về Model Size / Memory

Breakdown đã khử trùng lặp của `exported_model/model_fp32.pt` theo parameter:

| Thành phần | Params | FP32 | FP16 | INT8 |
|---|---:|---:|---:|---:|
| vision_model | 92.9M | 372 MB | 186 MB | 93 MB |
| text_model | 277.7M | 1111 MB | 555 MB | 278 MB |
| projection + other | 1.2M | 5 MB | 2 MB | 1 MB |
| total | 371.8M | 1487 MB | 744 MB | 372 MB |

Text chiếm 75% parameter của model. Riêng token embedding là `250000 x 768 = 192M params = 768 MB FP32`.

**Quyết định:** text INT8 quan trọng cho memory cuối cùng, nhưng nên làm sau khi nhánh vision đạt target xong.

#### 13. Kế Hoạch Tiếp Theo Sau v6

**Phân tích sau v6:**
- `v6` cho thấy fake-quant attention có tăng nhẹ R@1 (`49.30`), tiến thêm 1 bước nhỏ tới mốc `50.0`.
- Hiệu quả marginal của `v6` so với `v5` cho thấy việc QAT từng module lẻ đã chạm ngưỡng (diminishing returns).

**Thứ tự dự kiến:**
1. Cân nhắc Train QAT với số lượng mẫu lớn hơn hoặc dài epoch hơn để ép mô hình học tốt hơn bù lại loss lượng tử hóa.
2. Nếu R@1 vẫn chưa vượt qua mốc `50.0`, cân nhắc thuật toán Learned Rotation (như SpinQuant) tối ưu hóa góc quay rotation dựa trên activation thay vì chỉ bảo toàn mean.
3. Nếu R@1 `49.30` được xem là "đủ tốt" để đánh đổi, tiến hành compile/link `v6` và đo đạc Board Fidelity (thông qua `qnn-net-run`).
4. Bắt đầu quy trình nén Text Encoder (`T0` - đo concentration) để hoàn thiện hệ thống W8A8 end-to-end.

---

## 3. Chỉ Mục Artifact Hiện Tại

| Artifact | Ý nghĩa |
|---|---|
| `artifacts/deployment/exports/exported_model` | Baseline TBPS FP32/FP16 đã merge LoRA |
| `artifacts/deployment/exports/exported_model_rotated` | Model đã rotate bảo toàn mean |
| `artifacts/deployment/exports/exported_model_rotated_qat_v3` | Model QAT đầu tiên pass gate |
| `artifacts/deployment/exports/exported_model_rotated_qat_v4` | Model deploy đã verify trên board |
| `artifacts/deployment/exports/exported_model_rotated_qat_v5` | Ứng viên accuracy trước đó |
| `artifacts/deployment/exports/exported_model_rotated_qat_v6` | Ứng viên accuracy tốt nhất hiện tại |
| `artifacts/deployment/runtime/rotated_w8a8_v2/vision_encoder.bin` | Binary all-INT8 đầu tiên link được v68, retrieval fail |
| `artifacts/deployment/runtime/rotated_w8a8_qat_v4/vision_encoder.bin` | Binary QAT v4 đã verify trên board khi có sẵn |
| `artifacts/deployment/runtime/rotated_w8a8_qat_v6/job_j57krdwvp_qdq_onnx` | Artifact QDQ tốt nhất hiện tại |
| `artifacts/deployment/qnn_inputs/vn3k_train_calib_2000` | Raw calibration AI Hub |
| `d7jzjy1m2` | Dataset AI Hub `msiglip-vision-vn3k-train-calib-2000` |

---

## 4. KẾ HOẠCH: quantize Text Encoder (full both-INT8 on-device)

> Trạng thái: **PLANNING** (chưa chạy).
> Mục tiêu: Đưa text encoder về INT8 W8A8 chạy trên v68 → hệ cả-hai-INT8 (~372MB weights, gọn cho RAM 4GB).

Vision đã xong (v5 `49.25`, v4 board-verified). Text là 75% tham số (token-embedding 250k×768 = 768MB FP32) nên là phần ăn RAM chính.

### 4.1 Cấu trúc text encoder (đã rà code)

`SiglipTextTransformer`: `embeddings` (token 250k×768 + position) → `encoder` (12× **SiglipEncoderLayer y hệt vision**: LN1, attn q/k/v/out_proj, LN2, mlp fc1/GELU/fc2) → `final_layer_norm` → **last-token** `[:, -1, :]` → `head` (nn.Linear 768→768).

* **Giống vision** (tái dùng được rotation + QAT): encoder layers, LayerNorm, GELU, attention.
* **Khác vision:**
* Input = **input_ids (int) + attention_mask** (vision: ảnh float, không mask). Mask cộng vào scores qua `_prepare_4d_attention_mask`.
* Writer residual đầu vào = **token_embedding + position_embedding** (vision: patch-conv).
* Pooler = **last-token + Linear head** (vision: MultiheadAttentionPoolingHead) → đơn giản hơn.

### 4.2 Kế hoạch theo giai đoạn (diagnostic-first)

* **T0:** Baseline plain W8A8 (AI Hub quantize-only) + **text-isolation R@1** (ảnh FP32 + text QDQ).
* **Quyết định:** Concentration nhẹ → có thể bỏ qua rotation (đơn giản); nặng (như vision 252×) → cần rotation.

**T1 — Áp method (theo T0):**

* (Nếu cần) `rotate_text_encoder.py` MỚI: writers = token_embedding + position_embedding + out_proj + fc2; readers = q/k/v + fc1 + head (đọc final_layer_norm); mean-preserving Q (text cũng LayerNorm). KHÔNG có patch-conv / MHA-pooling-head.
* QAT: tổng quát hóa `train_vision_quant_robust.py` sang text (`encode_text`, `text_model`, distill text embedding). Thang coverage v1→v6 dùng lại; `--quant-attention` đã có sẵn nhánh `+attention_mask` nên chạy cho text.

**T2 — Quantize + board:** AI Hub W8A8 text → compile/link → `text_encoder.bin` → board (latency, fidelity). Xác nhận token-embedding table quant INT8 (768→192MB).

**T3 — Số quyết định cuối:** mở rộng `eval_retrieval_quantized_vision.py` cho text qua QDQ ONNX. Đo:

1. text-isolation (ảnh FP32 + text QDQ)
2. **end-to-end both-INT8** (ảnh QDQ + text QDQ) = số deploy thật.

### 4.3 Script: tái dùng / build / generalize

| Script | Trạng thái |
| --- | --- |
| `prepare_vn3k_text_inputs.py`, `compare_text_onnx_with_pytorch.py` | ✅ tái dùng |
| `profile_activation_outliers.py` | 🔧 adapt cho text |
| `export_text_onnx.py` (opset-20, int inputs, last-token) | ⬜ build |
| `rotate_text_encoder.py` (nếu T0 cần) | ⬜ build |
| `train_vision_quant_robust.py` → text | 🔧 generalize (`--modality text`) |
| `eval_retrieval_quantized_vision.py` → text QDQ + both-INT8 | 🔧 mở rộng |

### 4.4 Rủi ro cần canh

* **Input int trên AI Hub:** input_ids/attention_mask là chỉ số/mask, KHÔNG quantize. `submit_qaihub_quantize_compile.py` đang hardcode input ảnh float `[1,3,256,256]` → phải sửa `--input-specs` cho 2 int input của text.
* **Pooler last-token** (`[:,-1,:]` gather) + **attention_mask** export ONNX đúng không.
* **Token-embedding INT8** (per-row weight quant, vocab 250k): nguồn size win chính; kiểm cosine không sụt.

### 4.5 Bước đầu tiên

* **T0:** Viết `export_text_onnx.py` (opset-20) + adapt profiler → đo concentration text → quyết định rotation. Local/free, chưa tốn job AI Hub tới bước baseline W8A8.

---

## 5. T0 — Text concentration profiling: **text CẦN rotation** (giống vision)

> Local/free. Scripts mới: `export_text_onnx.py` (opset-20, 2 int input, last-token pooler), `profile_text_activation_outliers.py` (import lại lõi của `profile_activation_outliers.py`, chỉ khác phần feed 2 int input).

### Export + static control (T0a) — PASS
- `export_text_onnx.py --model-dir exported_model` → `exported_model/text_onnx/text_encoder.onnx`. Op counts: **Gelu=12, LayerNormalization=25, Pow=0** (fused, sạch; 12 vì text head là Linear, không có gelu pooling-head).
- Static control `compare_text_onnx_with_pytorch.py`: cosine **1.0000** / min `0.99999988`, no NaN → export trung thực.

### Concentration (T0b) — residual stream tập trung nặng
profile blocks 0-11 trên `vn3k_text_10` (10 mẫu), 206 target tensor.

| Nhóm | abs_max | concentration | Kết luận |
|---|---|---|---|
| **Residual Adds** (`layers.N/Add`, `Add_1`, `fc2/Add`) | ~810 | **200–404×** | tập trung nặng (≈ vision 252×) |
| Median toàn cục (194 tensor sạch) | — | 3.1 | đa số ổn; chỉ residual stream nặng |

→ **Quyết định: text CẦN mean-preserving rotation.** Plain per-tensor W8A8 sẽ collapse như vision.

### Artifact mask (text-specific, đã loại + ghi nhận rủi ro)
- 12 tensor `…/self_attn/Add_output_0` có abs_max **3.4e38** (= FLT_MAX) = hằng số mask `_prepare_4d_attention_mask` cộng vào scores trước softmax. KHÔNG phải outlier (sau softmax → 0); đã loại khỏi phân tích concentration.
- **Rủi ro lúc quantize text trên AI Hub:** per-tensor INT8 cho tensor scores+mask (dải −3.4e38) bất khả → cần AI Hub xử lý mask trong softmax fused, hoặc dùng giá trị mask ôn hòa hơn. Sẽ lộ ở baseline W8A8. (Vision không có — không mask.)

### Artifacts
- `exported_model/text_onnx/{text_encoder.onnx,.onnx.data}`
- `runtime/diag/text_outliers/{summary.json, per_tensor.csv}`

### Bước kế: T1 — `rotate_text_encoder.py`
Mean-preserving rotation cho text: writers (fold `Q`) = token_embedding + position_embedding + out_proj + fc2; readers (fold `Qᵀ`) = q/k/v_proj + fc1 + head (đọc `final_layer_norm`); giữ fused LayerNorm; gate output-invariant cosine ~1.0. Không patch-conv / không MHA-pooling-head. Sau đó QAT (generalize `train_vision_quant_robust.py` sang text).

---

## 6. T1 — Text rotation: **PASS**, residual concentration `404× → 5.3×`

> Local/free. Script mới: `rotate_text_encoder.py` (tái dùng toàn bộ low-level folds + `_mean_preserving_orthogonal` của `rotate_vision_encoder.py`; chỉ khác bản đồ writer/reader của text và gate dùng `encode_text` với 2 int input).

### Bản đồ rotation (SiglipTextTransformer)
- **Writers** (fold `W ← QW, b ← Qb`; embedding: `P ← P Qᵀ`): `embeddings.token_embedding`, `embeddings.position_embedding`, mỗi layer `self_attn.out_proj`, mỗi layer `mlp.fc2`.
- **Readers** (fold `W ← W Qᵀ`): mỗi layer `q/k/v_proj` (đọc `layer_norm1`), `mlp.fc1` (đọc `layer_norm2`), `head` (đọc `final_layer_norm`, sau last-token pooler).
- 3 LayerNorm (`layer_norm1/2`, `final_layer_norm`): fold affine vào readers → set identity, **giữ fused LayerNorm**. `Q` bảo toàn mean (`Q·1=1`) nên identity-affine LayerNorm giao hoán với `Q`.
- Khác vision: không có patch-conv (token_embedding là Embedding writer thay thế) và không có MHA-pooling-head (head text là `nn.Linear` reader đơn giản → không cần slicing K/V của packed in_proj). `backbone.text_projection` nằm ở không gian head-output chưa quay → không đụng.

### Gate output-invariance (T1) — PASS
- `encode_text` FP32 cosine vs model gốc trên `vn3k_text_10`: mean `1.0`, **min `0.99999982`**.
- `Q` orthogonality err `3.1e-15`; `Q·1=1` err `1.0e-14`.
- Static control ONNX (rotated) vs PyTorch (rotated): cosine_raw `1.0`, cosine_l2 `0.99999988`, no NaN.
- Export ONNX opset-20 rotated: **Gelu=12, LayerNormalization=25, Pow=0** (sạch, fused).

### Concentration sau rotation (bằng chứng rotation có tác dụng)
profile lại `text_outliers_rotated` trên cùng 10 mẫu, lọc bỏ tensor mask `self_attn/Add` (hằng `3.4e38`).

| Nhóm | BEFORE (T0) | AFTER (T1) | Ghi chú |
|---|---:|---:|---|
| **Residual Adds** (`layers.N/Add`, `Add_1`, `fc2/Add`) max conc | `404.3×` | **`5.3×`** | residual stream gần như phẳng |
| Residual Adds max abs_max | `828` | **`113`** | per-tensor scale giảm mạnh |
| `mlp/fc1/Add` (pre-GELU intermediate, 3072-d) | — | `66.7×` | hotspot còn lại, **không** trên trục residual 768-d → QAT fake-quant GELU xử lý (giống vision v3+) |

→ Mean-preserving `Q` đã trải đều outlier của residual stream như trên vision. Hotspot pre-GELU còn lại là phần QAT đảm nhiệm, đúng pattern vision.

> Lưu ý verdict script báo "DIFFUSE" là **báo động giả** do hằng mask `3.4e38` (T0 đã phân loại benign — về 0 sau softmax). Tín hiệu residual stream thực tế đã phẳng rõ ràng.

### Commands (local/free)
```bash
# [T1-1] Rotate text (gate cosine ~1.0 mới lưu)
venv/bin/python deployment/scripts/qnn/rotate_text_encoder.py --model-dir artifacts/deployment/exports/exported_model --output-dir artifacts/deployment/exports/exported_model_text_rotated --input-dir artifacts/deployment/qnn_inputs/vn3k_text_10

# [T1-2] Export rotated text ONNX opset-20
venv/bin/python deployment/scripts/qnn/export_text_onnx.py --model-dir artifacts/deployment/exports/exported_model_text_rotated

# [T1-3] Static control (rotated ONNX vs rotated PyTorch)
venv/bin/python deployment/scripts/qnn/compare_text_onnx_with_pytorch.py --onnx-model artifacts/deployment/exports/exported_model_text_rotated/text_onnx/text_encoder.onnx --model-dir artifacts/deployment/exports/exported_model_text_rotated --input-dir artifacts/deployment/qnn_inputs/vn3k_text_10 --json /tmp/text_rotated_static.json --csv /tmp/text_rotated_static.csv

# [T1-4] Re-profile concentration để xác nhận rotation phẳng residual
venv/bin/python deployment/scripts/qnn/profile_text_activation_outliers.py --onnx-model artifacts/deployment/exports/exported_model_text_rotated/text_onnx --input-dir artifacts/deployment/qnn_inputs/vn3k_text_10 --json artifacts/deployment/runtime/diag/text_outliers_rotated/summary.json --csv artifacts/deployment/runtime/diag/text_outliers_rotated/per_tensor.csv
```

### Artifacts
- `exported_model_text_rotated/{config.yaml, model_fp32.pt, rotation_summary.json}`
- `exported_model_text_rotated/text_onnx/{text_encoder.onnx, .onnx.data}`
- `runtime/diag/text_outliers_rotated/{summary.json, per_tensor.csv}`

### Bước kế: T1-QAT — generalize `train_vision_quant_robust.py` sang text (`--modality text`)
Drive `encode_text`, distill sentence embedding, xử lý 2 int input + attention_mask, dùng lại thang coverage EMA v3→v6. Base = `exported_model_text_rotated`.

### T1-QAT + T2 + T3: scripts đã generalize sang text (đã smoke-verify, chưa có số thật)

**T1-QAT — `train_vision_quant_robust.py --modality text`** (✅ generalized, smoke PASS):
- Thêm `--modality {vision,text}` + `--seq-len`. Text path: `RawTextDataset` (2 int .raw → dict, default-collate), controller resolve tower (`text_model`, head = `final_layer_norm` + Linear `head`), freeze prefix text (`text_projection`/`final_layer_norm`), encode qua `encode_text`.
- Dùng lại nguyên: EMA observer, thang coverage GELU+residual→head→linears→attention, fake-quant attention matmul (cùng class `SiglipAttention`; text truyền 4D mask, quantize scores **trước** khi cộng mask).
- Vision path regression: clean cosine `1.0`, không đổi.
- Lệnh QAT: xem README/commands; base `exported_model_text_rotated` → `exported_model_text_rotated_qat_t1`.

**T3 — `eval_retrieval_quantized_vision.py` mở rộng** (✅ extended, verified):
- Thêm `--text-qdq-onnx` + `--skip-vision-qdq`. Ma trận 4 combo (raw dot product, khớp `_compute_metrics`):
  - `baseline_fp32` (text FP32 + image FP32, phải ~52.28)
  - `vision_int8` (text FP32 + image QDQ)
  - `text_int8` (text QDQ + image FP32)
  - `both_int8` (text QDQ + image QDQ) = **số deploy thật**, là gate.
- Text QDQ chạy trên cùng token tensor như FP32 → drop tách bạch đúng phần text-quant. Deploy combo tự chọn = most-quantized available.
- Verify: feed text ONNX FP32 (rotated, output-invariant) → `text_int8 == baseline`, `both_int8 == vision_int8` (đúng như kỳ vọng vì text ONNX = FP32).

**T2 — `submit_qaihub_quantize_compile.py --modality text`** (✅ added, verified prepare-only):
- `--modality text` đặt default an toàn: `--input-specs` = 2 int input `((1,S),"int64")`, `--compile-options` = `""` (BỎ `--quantize_io` vì token id tới ~250k, không thể int8 hóa graph I/O). Vision default không đổi.
- Verify: staticize `input_ids`/`attention_mask` → `(1,64)`, giữ INT64 (elem_type 7).
- **Rủi ro còn lại (canh ở T2 QDQ-fidelity gate):** hằng mask `3.4e38` trong `scores+mask`; nếu per-tensor INT8 collapse sẽ lộ ở cosine gate (local, rẻ).
- **Prerequisite:** cần upload **text** calibration dataset (int input_ids+attention_mask) lên AI Hub; dataset vision `d7jzjy1m2` KHÔNG dùng được.

## 7. 2026-06-19 - C1 Off-board both-INT8 retrieval PASS

**Mục tiêu:** đo số deploy proxy end-to-end khi cả vision encoder và text encoder đều dùng W8A8 QDQ.

**Command:**

```bash
python3 deployment/scripts/qnn/eval_retrieval_quantized_vision.py \
  --qdq-onnx artifacts/deployment/runtime/rotated_w8a8_learned_qat_v8/job_jp24xxn65_qdq_onnx \
  --text-qdq-onnx artifacts/deployment/runtime/text_w8a8_learned_qat_v8_finite_mask/job_jp17y648p_qdq_onnx \
  --model-dir artifacts/deployment/exports/exported_model \
  --json artifacts/deployment/runtime/both_int8/both_int8_r1.json
```

**Artifacts:**

| Artifact | Ý nghĩa |
|---|---|
| `artifacts/deployment/runtime/rotated_w8a8_learned_qat_v8/job_jp24xxn65_qdq_onnx` | Vision QDQ v8 learned rotation |
| `artifacts/deployment/runtime/text_w8a8_learned_qat_v8_finite_mask/job_jp17y648p_qdq_onnx` | Text QDQ v8 learned rotation + finite mask |
| `artifacts/deployment/runtime/both_int8/both_int8_r1.json` | Kết quả C1 both-INT8 full VN3K test |

**Kết quả retrieval full VN3K test:**

| Combo | T2I R@1 | I2T R@1 | Ghi chú |
|---|---:|---:|---|
| `baseline_fp32` | `52.40` | `55.30` | local sanity reproduction; baseline báo cáo chính vẫn là `52.28` |
| `vision_int8` | `50.85` | `52.90` | vision v8 learned rotation |
| `text_int8` | `51.65` | `55.55` | text v8 learned rotation + finite mask |
| **`both_int8`** | **`50.25`** | **`52.95`** | **PASS** |

**Kết luận:** C1 both-INT8 đạt T2I R@1 **`50.25`**, vượt deploy target `50.0` với margin `+0.25`. Khi báo cáo drop, dùng paper baseline `52.28`: both-INT8 giảm **`-2.03`** T2I R@1 (`52.28 → 50.25`). Số `52.40` chỉ là sanity reproduction của pipeline local, không dùng làm mốc drop chính.

**Bước tiếp theo:** compile/link v8 vision và text finite-mask thành `.bin`, chạy board fidelity/runtime, rồi chạy C2 both-INT8 trực tiếp trên RB3.

## 8. 2026-06-19 - QAT v8 Vision Board Retrieval PASS

**Mục tiêu:** xác nhận binary vision v8 learned-rotation trên RB3 bằng retrieval full VN3K gallery, không chỉ smoke cosine trên 10 ảnh.

**Board run:** chạy `vision_encoder.bin` trên `vn3k_test_gallery_2000` bằng QAIRT `2.45.40.260406`, HTP v68, graph I/O `UFIXED_POINT_8`.

**Local retrieval command:**

```bash
python3 deployment/scripts/qnn/eval_retrieval_board_vision.py \
  --vision-output-dir artifacts/deployment/qnn_runs/rotated_w8a8_learned_qat_v8_gallery_2000 \
  --gallery-input-dir artifacts/deployment/qnn_inputs/vn3k_test_gallery_2000 \
  --model-dir artifacts/deployment/exports/exported_model \
  --dataset-root . \
  --json artifacts/deployment/qnn_runs/rotated_w8a8_learned_qat_v8_gallery_2000/board_vision_r1.json
```

**Artifacts:**

| Artifact | Ý nghĩa |
|---|---|
| `artifacts/deployment/qnn_runs/rotated_w8a8_learned_qat_v8/qnn_vs_pytorch_summary.json` | Smoke board fidelity 10 ảnh |
| `artifacts/deployment/qnn_runs/rotated_w8a8_learned_qat_v8/profile.txt` | QNN profile: `33.05 ms/image`, `22.77 FPS` |
| `artifacts/deployment/qnn_runs/rotated_w8a8_learned_qat_v8_gallery_2000/board_vision_r1.json` | Full board vision-isolation retrieval |

**Kết quả:**

| Metric | QDQ proxy | Board | Delta |
|---|---:|---:|---:|
| T2I R@1 | `50.85` | `50.20` | `-0.65` |
| I2T R@1 | `52.90` | `54.50` | `+1.60` |

**Chi tiết board retrieval:**

| Task | R@1 | R@5 | R@10 | mAP | mINP |
|---|---:|---:|---:|---:|---:|
| T2I | `50.20` | `77.62` | `86.73` | `55.84` | `49.51` |
| I2T | `54.50` | `81.65` | `90.00` | `50.22` | `33.25` |

**Kết luận:** v8 vision binary **PASS** board retrieval gate (`50.20 >= 50.0`). QDQ proxy `50.85` dự đoán tốt, board giảm nhẹ `-0.65` T2I nhưng vẫn đạt mục tiêu deploy. Bước còn lại cho số deploy cuối là board-verify text finite/f32/linksafe và C2 both-INT8 trực tiếp trên RB3.

## 9. 2026-06-19 - Text f32-mask link fail và link-safe mask rewrite

**Mục tiêu:** compile/link text encoder W8A8 trên AI Hub sau khi chuyển `attention_mask` sang float32 0/1 để tránh lỗi `Cast_output_0_updated_pre_quant`.

**AI Hub run:**

```bash
python3 deployment/scripts/qnn/submit_qaihub_quantize_compile.py \
  --modality text \
  --text-attention-mask-dtype float32 \
  --model artifacts/deployment/exports/exported_model_text_rotated_learned_qat_v8/text_onnx_f32mask_finite \
  --calibration-data d7ozgzkq9 \
  --weights-dtype int8 \
  --activations-dtype int8 \
  --wait \
  --download artifacts/deployment/runtime/text_w8a8_learned_qat_v8_f32mask/text_encoder.bin
```

**Jobs:**

| Job | ID | Status | Note |
|---|---|---|---|
| Quantize | `jglo6q3jg` | SUCCESS | f32-mask finite ONNX uploaded |
| Compile | `jp17ymq7p` | SUCCESS | input specs: `input_ids int64`, `attention_mask float32`; options `--truncate_64bit_io --quantize_io` |
| Link | `j56re0vy5` | FAILED | `Tensor '/text_model/Cast_output_0_updated' has a floating-point type...` |

**Diagnosis:** f32 input dtype was necessary but not sufficient. The exported ONNX still contained a redundant mask path:

```text
Expand(attention_mask float32)
  -> Cast(FLOAT)
  -> Sub
  -> Cast(BOOL)
  -> Where
```

AI Hub QDQ materialized the redundant cast output as `/text_model/Cast_output_0_updated`, and HTP v68 rejected it as an internal floating-point tensor during link. This is a graph representation issue, not a model/QAT accuracy issue.

**Mathematical rewrite:** because `attention_mask` is binary, the ONNX mask subgraph can be simplified without changing semantics:

```text
Where(1 - mask != 0, -32, 0)  ==  (1 - mask) * (-32),  mask in {0, 1}
```

**Implemented local fix:** added `deployment/scripts/qnn/patch_text_qnn_link_safe_mask.py`, which:

1. removes `/text_model/Cast -> /text_model/Cast_output_0`;
2. removes the redundant `Cast(BOOL)` / `Where` path;
3. inserts `Mul((1-attention_mask), -32)` as the shared 4D additive mask.

**Artifact:**

| Artifact | Ý nghĩa |
|---|---|
| `artifacts/deployment/exports/exported_model_text_rotated_learned_qat_v8/text_onnx_f32mask_finite_linksafe` | Text ONNX f32-mask finite + QNN-link-safe mask rewrite |
| `.../qnn_link_safe_mask_patch_summary.json` | Patch summary; confirms removed `Cast/Where` nodes |
| `.../static_vs_pytorch_summary.json` | Static ONNX-vs-PyTorch gate after rewrite |

**Local gates:**

| Gate | Result |
|---|---:|
| `/text_model/Cast_output_0` remaining in graph | `False` |
| ONNX checker | PASS |
| ONNX Runtime smoke load | PASS |
| Static cosine mean/min vs PyTorch | `0.99999999 / 0.99999976` |
| NaN | none |

**Kết luận lúc đó:** dùng `text_onnx_f32mask_finite_linksafe` cho lần submit AI Hub kế tiếp. Lần submit sau đã pass link, xem section 10; bước tiếp theo vẫn là board-run text smoke/fidelity rồi C2 both-INT8 board retrieval.

## 10. 2026-06-19 - Text finite/f32/link-safe AI Hub Link PASS

**Mục tiêu:** xác nhận graph text finite/f32/link-safe có thể đi qua đủ chuỗi AI Hub W8A8 quantize → compile → link để sinh QNN context binary cho HTP v68.

**Input model:** `artifacts/deployment/exports/exported_model_text_rotated_learned_qat_v8/text_onnx_f32mask_finite_linksafe`

**Calibration data:** `d7ozgzkq9`

**Kết quả:** AI Hub link đã PASS sau link-safe mask rewrite. Job IDs của lần pass không được ghi trong prompt, nên journal này chỉ ghi artifact đã có trong workspace.

**Artifact:**

| Artifact | Size | Ý nghĩa |
|---|---:|---|
| `artifacts/deployment/bin/text_encoder.bin` | `266M` | Text W8A8 QNN context binary đã link được |

**Ý nghĩa kỹ thuật:** lỗi trước đó không chứng minh text QAT/W8A8 sai về chất lượng. Nó là lỗi biểu diễn graph: finite mask và f32 mask đã sửa dynamic-range/QDQ scale, còn link-safe rewrite loại các float mask islands (`Cast`/`Where`) mà HTP v68 không chấp nhận trong context binary.

**Kết luận:** text encoder hiện đã có binary linkable. Các gate còn lại là chạy `qnn-net-run` text trên RB3, so fidelity board-vs-PyTorch/QDQ, sau đó chạy C2 both-INT8 board retrieval.

## 11. 2026-06-20 - Text board FAIL do dynamic token embedding lookup trên HTP

**Mục tiêu:** xác định vì sao text W8A8 context binary đã link được nhưng board fidelity rất thấp, và liệu còn hướng cứu để chạy text trên RB3 hay không.

**Bối cảnh:** sau link-safe mask rewrite, text graph có thể đi qua AI Hub quantize → compile → link. Tuy nhiên board smoke cho thấy output text trên HTP không khớp PyTorch/QDQ, dù input files và dtype đã được sửa từ `int64` sang `int32`.

### Kết quả đã xác nhận

| Gate | Artifact | Kết quả |
|---|---|---:|
| Static ONNX i32/f32-mask/link-safe vs PyTorch | `artifacts/deployment/exports/exported_model_text_rotated_learned_qat_v8/text_onnx_i32_f32mask_finite_linksafe/static_vs_pytorch_i32_summary.json` | cosine mean/min `0.99999999 / 0.99999976` |
| QDQ i32/f32-mask/link-safe vs PyTorch | `artifacts/deployment/runtime/text_w8a8_learned_qat_v8_i32_f32mask/text_qdq_fid.json` | cosine mean/min/max `0.99494732 / 0.99116683 / 0.99719608` |
| Board QNN i32/f32-mask vs PyTorch | `artifacts/deployment/qnn_runs/text_w8a8_learned_qat_v8_i32_f32mask/qnn_vs_pytorch_summary.json` | cosine mean/min/max `0.12666028 / 0.05224004 / 0.23556966` |
| Board execution metadata | `artifacts/deployment/qnn_runs/text_w8a8_learned_qat_v8_i32_f32mask/execution_metadata.yaml` | 10 inferences completed; `input_ids` = `QNN_DATATYPE_INT_32`, `attention_mask`/`output_0` = `QNN_DATATYPE_UFIXED_POINT_8` |

**Input file sanity:** board `input_ids` `.raw` đã được kiểm bằng `od` và chứa token thật, ví dụ sample đầu bắt đầu bằng:

```text
259 272 2342 2214 266 326 1842 12996 ...
```

Như vậy lỗi không phải do file `.raw` bị rỗng, sai endian, sai dtype, hoặc copy nhầm input.

### Zero-token ablation

Để kiểm tra graph HTP có thật sự dùng `input_ids` hay không, đã tạo bản input mới bằng cách copy `vn3k_text_10_f32mask_i32` rồi zero toàn bộ file `*_input_ids.raw`, giữ nguyên `attention_mask`.

**Board runs được so sánh:**

| Run | Ý nghĩa |
|---|---|
| `artifacts/deployment/qnn_runs/text_w8a8_learned_qat_v8_i32_f32mask` | token IDs thật |
| `artifacts/deployment/qnn_runs/text_w8a8_learned_qat_v8_i32_zero_ids` | cùng mask, nhưng `input_ids` toàn 0 |

**Kết quả so real-vs-zero board output:**

```text
0 cos(real,zero)= 1.0 max_abs= 0.0
1 cos(real,zero)= 1.0 max_abs= 0.0
2 cos(real,zero)= 1.0 max_abs= 0.0
3 cos(real,zero)= 1.0 max_abs= 0.0
4 cos(real,zero)= 1.0 max_abs= 0.0
5 cos(real,zero)= 1.0 max_abs= 0.0
6 cos(real,zero)= 1.0 max_abs= 0.0
7 cos(real,zero)= 1.0 max_abs= 0.0
8 cos(real,zero)= 1.0 max_abs= 0.0
9 cos(real,zero)= 1.0 max_abs= 0.0
```

**Kết luận thực nghiệm:** text HTP context binary hiện cho output bit-identical khi `input_ids` thật và khi `input_ids` toàn zero. Vì vậy full-context text binary nhận `input_ids` không đáng tin, dù graph đã link và `qnn-net-run` báo execute thành công.

### Diễn giải kỹ thuật

Text encoder bắt đầu bằng embedding lookup:

```text
input_ids -> Gather(token_embedding.weight, input_ids)
          -> + position_embedding
          -> transformer encoder
          -> final layer norm/head
```

ONNX graph hợp lệ: `Gather` cho phép indices `int32`/`int64`, và local ONNX Runtime chạy đúng với cosine gần `1.0`. QDQ ONNX cũng đúng với cosine mean `0.9949`. Điểm fail chỉ xuất hiện sau khi graph được compile/link thành HTP context binary và chạy bằng `qnn-net-run` trên RB3.

Do đó nguyên nhân phù hợp nhất hiện tại là **giới hạn/bug ở QNN HTP path cho dynamic `Gather(token_embedding.weight, input_ids)`**, hoặc ở lowering/runtime của subgraph embedding lookup khi input indices là runtime tensor. Các thay đổi trước đó như finite mask, f32 mask, link-safe mask, `input_ids int32`, `--quantize_io`, hay calibration chỉ giải quyết linkability và dtype; chúng không sửa được việc HTP context không phụ thuộc vào token IDs.

Đây không phải là thất bại của QAT/text model:

- static ONNX vẫn trung thực với PyTorch;
- QDQ proxy vẫn trung thực với PyTorch;
- text-isolation retrieval proxy vẫn tốt (`51.65` T2I R@1, `55.55` I2T R@1);
- board output sai vì dynamic token lookup không được runtime tôn trọng.

### Trạng thái deploy sau phát hiện này

| Hạng mục | Trạng thái |
|---|---|
| Vision QAT v8 board | PASS: full VN3K gallery board retrieval T2I `50.20`, I2T `54.50` |
| Text QDQ proxy | PASS: fidelity `0.9949 / 0.9912`, text-isolation T2I `51.65` |
| Both-INT8 QDQ proxy | PASS: T2I `50.25`, I2T `52.95` |
| Text full-context HTP board | FAIL: output không đổi giữa real `input_ids` và zero `input_ids` |
| Both-INT8 board trực tiếp | BLOCKED cho tới khi có text board path đáng tin |

### Hướng cứu không bỏ all-board

Không nên hiểu kết luận này là "không thể chạy text trên board". Kết luận chính xác hơn là: **không nên chạy toàn bộ text encoder thành một HTP context duy nhất nhận `input_ids`** trên stack hiện tại.

Hướng khả thi nhất là split text encoder:

1. CPU/host trên RB3 làm token embedding lookup:

```text
input_ids -> token_embedding[input_ids] + position_embedding -> inputs_embeds [1,64,768]
```

2. QNN HTP chạy phần nặng còn lại:

```text
inputs_embeds + attention_mask -> transformer encoder -> final layer norm/head -> text embedding
```

Ưu điểm:

- vẫn là pipeline chạy local trên RB3;
- loại bỏ dynamic integer `Gather` khỏi HTP;
- giữ transformer/head, phần compute nặng nhất, trên QNN HTP;
- không cần huấn luyện lại ngay vì code `SiglipTextEmbeddings` đã hỗ trợ `inputs_embeds`, và `SiglipEncoder` nhận trực tiếp `inputs_embeds`.

Rủi ro/chi phí:

- cần export wrapper ONNX mới nhận `inputs_embeds`;
- cần tạo input `.raw` cho embedding tensor thay vì `input_ids`;
- cần giữ token embedding table ở CPU-side runtime;
- nếu muốn all-INT8 nghiêm ngặt, cần quyết định cách lưu/lookup embedding table: FP16/FP32 CPU đơn giản trước, sau đó mới xét int8 table + dequant selected rows.

### Bước tiếp theo đề xuất

1. Làm microbenchmark QNN `Gather`: model rất nhỏ `input_ids -> Gather(embedding_table) -> output`. Nếu real/zero vẫn giống nhau trên board, có bằng chứng tối giản để chốt bug/limit HTP.
2. Export split-text ONNX nhận `inputs_embeds` và `attention_mask`.
3. Static compare split-text ONNX vs PyTorch.
4. AI Hub W8A8 quantize/compile/link split-text graph.
5. Board smoke split-text vs PyTorch/QDQ.
6. Nếu pass, chạy full VN3K board text-isolation rồi C2 both-INT8 board retrieval.

**Quyết định tạm thời:** dừng đầu tư vào full text HTP context nhận `input_ids` như đường deploy chính. Tiếp tục theo hướng split-text để vẫn đạt mục tiêu all-board, nhưng không phụ thuộc vào dynamic embedding lookup trên HTP.

## 12. 2026-06-20 - Kế hoạch thử nghiệm bóc tách nguyên nhân text board + tìm giải pháp

**Mục tiêu:** trước khi đổ công vào split-text, thiết kế bộ thử nghiệm có kiểm soát để (A) đóng đinh nguyên nhân gốc của việc text board bỏ qua `input_ids`, và (B) tìm/đo một đường deploy chạy được. Hai mục tiêu chạy song song vì một số thử nghiệm của B đồng thời là bằng chứng cho A.

### 12.1 Đánh giá chẩn đoán hiện tại (section 11)

- **Phần thực nghiệm — chắc chắn đúng.** Zero-token ablation (giữ mask, zero toàn bộ `input_ids`) cho board output bit-identical (`cos=1.0, max_abs=0.0` trên 10/10), cộng với `.raw` đã verify chứa token thật bằng `od`. Kết luận "board output không phụ thuộc `input_ids`" là không thể bác bỏ. Vì test **giữ nguyên mask**, nó cũng gián tiếp loại mask path khỏi nghi can — lỗi bị khóa vào nửa embedding/ids.
- **Phần quy kết nguyên nhân — hợp lý nhưng CHƯA chứng minh.** Zero-ids chỉ chứng minh "ids bị bỏ qua", không chứng minh thủ phạm là op `Gather`. Hạ kết luận "HTP không làm được dynamic Gather" từ **kết luận** xuống **giả thuyết hàng đầu**.
- **Lưu ý confound:** zero-ids identical KHÔNG chứng minh transformer còn sống — output hằng số có thể là transformer nhả rác cố định. Mọi test "feed X vào" phải kèm control "đổi X → output phải đổi".

### 12.2 Bốn giả thuyết cạnh tranh

| GT | Cơ chế | Đã loại trừ? |
|---|---|---|
| **H1** | HTP v68 không lower được dynamic `Gather(weight, input_ids)` với runtime indices | Chưa — mới suy đoán |
| **H2** | Bảng token-embedding INT8 ~192 MB (250k×768) vượt giới hạn constant buffer của DLC/HTP → compiler degrade graph | **Chưa xét** — giải thích sạch bất đối xứng vision (conv vài MB, PASS) vs text (bảng khổng lồ, FAIL) |
| **H3** | Pass quantize/compile của AI Hub làm hỏng Gather (constant-fold/clamp indices về 0) | Chưa — chưa thử toolchain khác |
| **H4** | Lỗi binding `qnn-net-run` (sai thứ tự `input_list`, input_ids map nhầm) | Đã verify nội dung file, **chưa** verify binding |

### 12.3 Ý tưởng cốt lõi: phân hoạch logic Gather ∘ Transformer

```text
Full text graph (FAIL)  =  E_gather (embedding lookup)  ∘  E_trans (transformer+head)
```

Graph đầy đủ fail ⇒ lỗi ở ít nhất một nửa. Test riêng từng nửa trên board ⇒ định vị thủ phạm. Nửa "transformer nhận `inputs_embeds`" đồng thời là giải pháp split-text. **Hai mục tiêu trùng nhau tại đây.**

Cây quyết định (ghép A2 = test E_gather riêng, B2 = test E_trans riêng):

| A2 (Gather/embed riêng) | B2 (Transformer nhận embeds riêng) | Kết luận → Hành động |
|---|---|---|
| FAIL | PASS | Lỗi ở embedding lookup (H1/H2) → **split-text GIẢI QUYẾT**; chốt H1 vs H2 bằng A1 (bảng nhỏ vs to) |
| PASS | FAIL | Lỗi ở transformer lowering (bất ngờ lớn) → split-text KHÔNG đủ; phải debug op transformer/mask |
| PASS | PASS | Mỗi nửa OK, full-graph fail khi ghép → lỗi interaction/size lúc compile full → vẫn split để né |
| FAIL | FAIL | Cả hai nửa hỏng → nghi toolchain (chạy A3) hoặc fallback CPU (B3) |

Kỳ vọng theo bằng chứng hiện có: nhánh **A2 FAIL + B2 PASS** (split-text thắng). Giá trị của kế hoạch: mọi nhánh đều có lối ra, kể cả nhánh xấu B2 FAIL mà trước đó chưa lường.

### 12.4 Nhóm A — đóng đinh nguyên nhân gốc

| # | Thử nghiệm | Đổi biến | Chi phí | Outcome → kết luận |
|---|---|---|---|---|
| **A0** | Hai bộ `input_ids` thật khác nhau (X vs Y), không chỉ real-vs-zero | nội dung input | Board, ~free (dùng lại `.bin`) | X≡Y → khẳng định chắc "ids bị bỏ qua". X≠Y → **H4 binding** |
| **A1** | Microbench Gather **bảng nhỏ** (vocab~1000): `ids → Gather(W) → out` | bỏ transformer + bảng nhỏ | 1 job AI Hub + board | Chạy đúng → **H1 sai** (HTP làm được Gather động). Fail → H1/H3 |
| **A2** | Microbench Gather **bảng thật 250k**, output = `inputs_embeds` | chỉ phóng to bảng | 1 job + board | A1 PASS & A2 FAIL → **H2 confirmed** (giới hạn buffer). Cả hai PASS → Gather không phải thủ phạm |
| **A3** | Compile graph fail cũ qua host x86 `qairt-converter/quantizer` thay vì AI Hub | toolchain | Host + board | Direct-QNN đúng & AI Hub sai → **H3 confirmed** (pass AI Hub) |

### 12.5 Nhóm B — tìm/đo giải pháp (song song)

| # | Thử nghiệm | Vai trò kép | Chi phí |
|---|---|---|---|
| **B1** | Export split-text ONNX: `inputs_embeds[1,64,768] + attention_mask → transformer → head`; static-compare vs PyTorch | Gate local bắt buộc cho mọi đường split | Local/free |
| **B2** | Board split-text: quantize/compile/link graph B1, chạy board, **feed 2 bộ embeds khác nhau** | Vừa là lời giải (nếu PASS) vừa test nửa E_trans | 1 job + board |
| **B3** | Chạy text QDQ ONNX trên **CPU (ORT) ngay trên RB3 ARM**: đo latency/query, RAM, both-INT8 R@1 | Lưới an toàn — đường both-INT8 chắc chắn chạy với memory win; text không nằm trên hot path | Board + local |
| **B4** | (chỉ khi B2 PASS) Quyết định lưu bảng: FP16 384 MB vs INT8 per-row 192 MB; đo RAM headroom 4 GB | Tối ưu bộ nhớ cuối | Board |

### 12.6 Thứ tự lệnh đề xuất (local làm trước, board/AI Hub để sau)

Quy ước: `[LOCAL]` chạy nhanh trên máy này; `[AIHUB]` tốn job; `[BOARD]` chạy trên RB3 (để user chạy).

```bash
# ─────────────────────────────────────────────────────────────
# A0 — chuẩn bị bộ input_ids thật KHÁC (Y), để board test real-vs-real
# [LOCAL] prep input (caption khác: start-index 100), i32 + f32 mask khớp graph hiện tại
venv/bin/python deployment/scripts/qnn/prepare_vn3k_text_inputs.py \
  --split test --num-samples 10 --selection first --start-index 100 \
  --id-dtype int32 --mask-dtype float32 \
  --output-dir artifacts/deployment/qnn_inputs/vn3k_text_10_altreal_i32
# [BOARD] chạy ĐÚNG binary i32 đã FAIL ở §11 (text_encoder_i32.bin + htp_config_text_i32.json) trên Y.
#   X đã có sẵn ở qnn_runs/text_w8a8_learned_qat_v8_i32_f32mask (cùng binary, input vn3k_text_10_f32mask_i32).
#   Input Y (vn3k_text_10_altreal_i32) đã là int32 ids (256 bytes/sample) + f32 mask — khớp graph i32.
qnn-net-run --backend "$QNN_LIB/libQnnHtp.so" \
  --retrieve_context artifacts/deployment/bin/text_encoder_i32.bin \
  --config_file deployment/config/qnn/htp_config_text_i32.json \
  --input_list artifacts/deployment/qnn_inputs/vn3k_text_10_altreal_i32/input_list.txt \
  --output_dir artifacts/deployment/qnn_runs/text_altreal \
  --profiling_level basic --perf_profile high_performance
# [LOCAL] so output Y vs X: nếu max_abs==0 mọi mẫu => ids bị bỏ qua (loại H4); nếu khác => H4 binding
venv/bin/python - <<'PY'
import numpy as np, glob
ys=sorted(glob.glob("artifacts/deployment/qnn_runs/text_altreal/Result_*/*.raw"))
xs=sorted(glob.glob("artifacts/deployment/qnn_runs/text_w8a8_learned_qat_v8_i32_f32mask/Result_*/*.raw"))
d=[float(np.abs(np.fromfile(y,np.float32)-np.fromfile(x,np.float32)).max()) for y,x in zip(ys,xs)]
print("max_abs(Y-X):",[round(v,4) for v in d], "=> IGNORES ids (loại H4)" if max(d)==0 else "=> H4 binding")
PY

# ─────────────────────────────────────────────────────────────
# A1/A2 — microbench Gather (script build_gather_microbench.py)
# [LOCAL] build 2 ONNX (vocab 1000 + 250k thật) + ORT sanity (đã chạy, depends_on_ids=True)
venv/bin/python deployment/scripts/qnn/build_gather_microbench.py \
  --model-dir artifacts/deployment/exports/exported_model_text_rotated_learned_qat_v8 \
  --out-dir artifacts/deployment/exports/gather_microbench

# [AIHUB] upload calib input_ids cho từng bảng (raw modality, 1 key int32) — ghi lại dataset ID in ra
venv/bin/python deployment/scripts/qnn/upload_qaihub_calibration_dataset.py \
  --modality raw --keys input_ids:int32:1,64 \
  --input-dir artifacts/deployment/exports/gather_microbench/small \
  --input-list input_list_real.txt --name msiglip-gather-small-calib
venv/bin/python deployment/scripts/qnn/upload_qaihub_calibration_dataset.py \
  --modality raw --keys input_ids:int32:1,64 \
  --input-dir artifacts/deployment/exports/gather_microbench/big \
  --input-list input_list_real.txt --name msiglip-gather-big-calib

# [AIHUB] quantize/compile/link mỗi bảng. input_ids là INDEX int32 → KHÔNG quantize_io
#   (thay <DS_SMALL>/<DS_BIG> bằng dataset ID vừa upload)
venv/bin/python deployment/scripts/qnn/submit_qaihub_quantize_compile.py \
  --model artifacts/deployment/exports/gather_microbench/small/gather_small.onnx \
  --calibration-data <DS_SMALL> --weights-dtype int8 --activations-dtype int8 \
  --no-staticize \
  --input-specs '{"input_ids": ((1, 64), "int32")}' \
  --compile-options "--truncate_64bit_io" \
  --wait --download artifacts/deployment/runtime/gather_microbench/small/gather_small.bin
venv/bin/python deployment/scripts/qnn/submit_qaihub_quantize_compile.py \
  --model artifacts/deployment/exports/gather_microbench/big/gather_big.onnx \
  --calibration-data <DS_BIG> --weights-dtype int8 --activations-dtype int8 \
  --no-staticize \
  --input-specs '{"input_ids": ((1, 64), "int32")}' \
  --compile-options "--truncate_64bit_io" \
  --wait --download artifacts/deployment/runtime/gather_microbench/big/gather_big.bin

# [BOARD] với mỗi bảng, chạy real và zero rồi so output (real_vs_zero). Ví dụ small
#   (graph text-family int32-in → dùng htp_config_text_i32.json như binary i32):
qnn-net-run --backend "$QNN_LIB/libQnnHtp.so" \
  --retrieve_context artifacts/deployment/runtime/gather_microbench/small/gather_small.bin \
  --config_file deployment/config/qnn/htp_config_text_i32.json \
  --input_list artifacts/deployment/exports/gather_microbench/small/input_list_real.txt \
  --output_dir artifacts/deployment/qnn_runs/gather_small_real --perf_profile high_performance
qnn-net-run --backend "$QNN_LIB/libQnnHtp.so" \
  --retrieve_context artifacts/deployment/runtime/gather_microbench/small/gather_small.bin \
  --config_file deployment/config/qnn/htp_config_text_i32.json \
  --input_list artifacts/deployment/exports/gather_microbench/small/input_list_zero.txt \
  --output_dir artifacts/deployment/qnn_runs/gather_small_zero --perf_profile high_performance
# [LOCAL] so real vs zero (np): nếu max_abs==0 trên mọi mẫu => bảng đó BỎ QUA ids
venv/bin/python - <<'PY'
import numpy as np, glob, os
for tag in ["small","big"]:
    rs=sorted(glob.glob(f"artifacts/deployment/qnn_runs/gather_{tag}_real/Result_*/*.raw"))
    zs=sorted(glob.glob(f"artifacts/deployment/qnn_runs/gather_{tag}_zero/Result_*/*.raw"))
    if not rs: print(tag,"(chưa có output)"); continue
    diffs=[float(np.abs(np.fromfile(r,np.float32)-np.fromfile(z,np.float32)).max()) for r,z in zip(rs,zs)]
    print(tag,"max_abs(real-zero) per sample:",[round(d,4) for d in diffs],
          "=> RESPECTS ids" if max(diffs)>0 else "=> IGNORES ids")
PY
# Kết luận: small RESPECTS & big IGNORES => H2 (size). small IGNORES => H1/H3. cả hai RESPECTS => Gather không phải thủ phạm.

# ─────────────────────────────────────────────────────────────
# B1 — split-text export nhận inputs_embeds (script export_split_text_onnx.py)
# [LOCAL] export transformer+head nhận inputs_embeds + attention_mask (opset-20),
#         CHẠY LUÔN static gate inline (--check-input-dir) + dump inputs_embeds .raw cho board (--dump-embeds-dir)
venv/bin/python deployment/scripts/qnn/export_split_text_onnx.py \
  --model-dir artifacts/deployment/exports/exported_model_text_rotated_learned_qat_v8 \
  --attention-mask-dtype float32 \
  --check-input-dir artifacts/deployment/qnn_inputs/vn3k_text_10_f32mask_i32 \
  --dump-embeds-dir artifacts/deployment/qnn_inputs/vn3k_text_10_split_embeds \
  --json artifacts/deployment/exports/exported_model_text_rotated_learned_qat_v8/text_onnx_split/static_vs_pytorch_summary.json

# ─────────────────────────────────────────────────────────────
# B2 — board split-text (sau khi B1 PASS)
# [LOCAL] inputs_embeds .raw đã dump sẵn: vn3k_text_10_split_embeds (smoke), vn3k_text_calib_500_split_embeds (calib)
# [AIHUB] upload calib inputs_embeds + mask (raw modality 2 key) — ghi dataset ID
venv/bin/python deployment/scripts/qnn/upload_qaihub_calibration_dataset.py \
  --modality raw \
  --keys inputs_embeds:float32:1,64,768 attention_mask:float32:1,64 \
  --input-dir artifacts/deployment/qnn_inputs/vn3k_text_calib_500_split_embeds \
  --name msiglip-split-text-embeds-calib-500

# [AIHUB] W8A8 quantize/compile/link split-text (inputs_embeds là activation float → quantize; mask f32)
#   (thay <DS_SPLIT> bằng dataset ID vừa upload)
venv/bin/python deployment/scripts/qnn/submit_qaihub_quantize_compile.py \
  --model artifacts/deployment/exports/exported_model_text_rotated_learned_qat_v8/text_onnx_split/text_encoder_split.onnx \
  --calibration-data <DS_SPLIT> --weights-dtype int8 --activations-dtype int8 \
  --input-specs '{"inputs_embeds": ((1, 64, 768), "float32"), "attention_mask": ((1, 64), "float32")}' \
  --compile-options "--quantize_io" \
  --wait --download artifacts/deployment/runtime/split_text_w8a8/text_encoder_split.bin

# [AIHUB] (tuỳ chọn) QDQ-only gate trước khi link — tải QDQ ONNX rồi so vs PyTorch split
#   thêm --quantize-only --download-quantized artifacts/deployment/runtime/split_text_w8a8/job_qdq_onnx
#   rồi: export_split_text_onnx.py --check-input-dir ... trên QDQ ONNX (so split-vs-full)

# [BOARD] chạy split-text trên board (input = inputs_embeds + attention_mask)
#   NOTE: inputs_embeds là activation float đã quantize (UFIXED_8) — KHÁC graph i32 (input_ids INT_32).
#   Dùng config text-family; nếu I/O khác thì chỉnh config tương ứng lúc compile.
qnn-net-run --backend "$QNN_LIB/libQnnHtp.so" \
  --retrieve_context artifacts/deployment/runtime/split_text_w8a8/text_encoder_split.bin \
  --config_file deployment/config/qnn/htp_config_text_i32.json \
  --input_list artifacts/deployment/qnn_inputs/vn3k_text_10_split_embeds/input_list.txt \
  --output_dir artifacts/deployment/qnn_runs/split_text_w8a8 \
  --profiling_level basic --perf_profile high_performance
# [BOARD] control: feed embeds đã zero để kiểm output có phụ thuộc embeds (phải KHÁC, không như full-graph)
#   (tạo bản zero embeds: copy vn3k_text_10_split_embeds, zero các *_inputs_embeds.raw, giữ mask)
# [LOCAL] fidelity board vs PyTorch (split wrapper): so output_0.raw vs encode_text(input_ids gốc)

# ─────────────────────────────────────────────────────────────
# B3 — fallback CPU-INT8 text (lưới an toàn, song song)
# [BOARD] chạy text QDQ ONNX bằng onnxruntime trên RB3 ARM (full-graph QDQ đã có), đo latency/RAM:
#   onnxruntime.InferenceSession(text_qdq.onnx) trên input_ids+attention_mask, providers=CPUExecutionProvider
# [LOCAL] both-INT8 R@1 đã có off-board = 50.25 (§7) làm tham chiếu; B3 chỉ cần xác nhận chạy được + RAM trên board
```

### 12.7 Trạng thái checklist

| Hạng mục | Trạng thái |
|---|---|
| A0 prep input Y | ✅ `vn3k_text_10_altreal_i32` (caption khác, X≠Y verified) |
| A0 board real-vs-real | ⬜ (board, user) |
| A1/A2 build microbench | ✅ `gather_microbench/{small,big}`, ORT depends_on_ids=True |
| A1/A2 board | ⬜ (AI Hub + board, user) |
| B1 export split-text + static gate | ✅ **PASS** cosine mean/min `0.99977 / 0.99972`; torch-split==full exact `1.0/0.0` |
| B2 board split-text | ⬜ (AI Hub + board, user; inputs_embeds .raw đã dump) |
| B3 CPU-INT8 text fallback | ⬜ (board, user) |

### 12.8 Kết quả local đã chạy (2026-06-20)

| Thử nghiệm | Lệnh/script | Kết quả | Artifact |
|---|---|---|---|
| A0 prep | `prepare_vn3k_text_inputs.py --start-index 100 --id-dtype int32 --mask-dtype float32` | 10 caption khác; token X≠Y (`X[:4]=[259,272,2342,2214]` vs `Y[:4]=[2135,326,1335,297]`) | `qnn_inputs/vn3k_text_10_altreal_i32` |
| B1 export + gate | `export_split_text_onnx.py ... --check-input-dir ... --dump-embeds-dir ...` | split ONNX vs full `encode_text`: cosine mean **0.99977**, min **0.99972** → **PASS** (≥0.999). 29 Gather còn lại = position-embed tĩnh, KHÔNG phải token table động | `exported_model_text_rotated_learned_qat_v8/text_onnx_split/`, `qnn_inputs/vn3k_text_10_split_embeds` |
| B1 sanity toán | torch-split vs torch-full (inline check) | cosine **1.0**, max abs diff **0.0** → wrapper đúng tuyệt đối; gap 0.9997 chỉ là noise fused-op ONNX, lành tính | — |
| A1/A2 build | `build_gather_microbench.py` | small (vocab1000, ~1MB) + big (250k, ~192MB INT8) ONNX; ORT real-vs-zero depends_on_ids=True cả hai | `exports/gather_microbench/{small,big}/` (+ input_list_real/zero.txt) |
| B2 calib prep | host token-lookup trên 500 calib caption | 500 `inputs_embeds` [1,64,768] f32 + mask f32 | `qnn_inputs/vn3k_text_calib_500_split_embeds` |
| Uploader mở rộng | `upload_qaihub_calibration_dataset.py` + `--modality raw --keys name:dtype:shape` | hỗ trợ calib generic (microbench input_ids, split-text inputs_embeds) | — |

> Lệnh đầy đủ (upload calib → submit AI Hub → board → so sánh) cho A0/A1/A2/B2/B3 nằm ở §12.6; chỉ cần thay `<DS_*>` bằng dataset ID in ra khi upload.

**Phát hiện phụ:** model `exported_model_text_rotated_learned_qat_v8` **không có `text_projection`** → `encode_text` trả thẳng `pooler_output` (head). Split wrapper đã guard `hasattr` nên khớp; lưu ý nếu port sang model có projection.

**Sẵn sàng cho user (board/AI Hub):**
1. **A0**: board-run **`text_encoder_i32.bin`** (+ `htp_config_text_i32.json`, đúng binary đã FAIL ở §11) trên `vn3k_text_10_altreal_i32`, so output Y vs X (`vn3k_text_10_f32mask_i32`). X≡Y ⇒ ids bị bỏ qua (loại H4); X≠Y ⇒ H4 binding.
2. **A1/A2**: AI Hub quantize/compile/link `gather_microbench/{small,big}` (input_ids INT32 index, KHÔNG quantize); board-run `input_list_real.txt` vs `input_list_zero.txt`. small PASS & big FAIL ⇒ H2; small FAIL ⇒ H1/H3; both PASS ⇒ Gather không phải thủ phạm.
3. **B2**: AI Hub W8A8 split-text (`text_onnx_split`, input `inputs_embeds` f32 + `attention_mask` f32), board-run với `vn3k_text_10_split_embeds`.
