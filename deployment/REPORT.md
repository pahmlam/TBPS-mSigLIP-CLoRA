# BÁO CÁO - Pipeline Triển Khai Thành Công mSigLIP Vision Trên RB3 Gen2 HTP v68

> Ngày: 15-06-2026
> Phạm vi: chỉ nhánh triển khai vision encoder
> Checkpoint nguồn: `artifacts/models/checkpoints/epoch=56-val_score=52.28.ckpt`
> Thiết bị đích: Qualcomm RB3 Gen2 / QCS6490 / HTP v68
> Artifact thành công: `artifacts/deployment/runtime/rotated_w8a8_v2/vision_encoder.bin`

Báo cáo này ghi lại deployment pipeline end-to-end thành công đầu tiên cho vision encoder mSigLIP trên RB3 Gen2 HTP v68. Quá trình bắt đầu từ Lightning checkpoint đã được finetune bằng LoRA, merge các LoRA adapter, áp dụng mean-preserving rotation để biến all-INT8 quantization thành khả thi, xuất đồ thị ONNX opset-20, quantize/compile/link thông qua AI Hub, và xác minh nhị phân trên bo mạch RB3 với QNN HTP runtime.

Quy trình cốt lõi thành công là:

```text
LoRA checkpoint
  -> merge LoRA vào base weights
  -> mean-preserving residual rotation
  -> ONNX opset 20 với fused Gelu và fused LayerNormalization
  -> W8A8 AI Hub quantize + compile/link với quantized I/O
  -> qnn-net-run trên RB3 HTP v68
```

Kết quả cuối cùng trên bo mạch:

| Mục | Kết quả |
|---|---:|
| QDQ ONNX vs PyTorch cosine, mean | `0.8975` |
| QNN board vs PyTorch cosine, mean | `0.8982` |
| QNN board vs PyTorch cosine, min/max | `0.8606 / 0.9283` |
| HTP NetRun latency | `34.25 ms/image` |
| HTP accelerator execute | `32.5 ms/image` |
| Throughput | `22.5 FPS` |
| Context binary size | `89.7 MB` |

Cosine là một proxy fidelity metric. Acceptance metric tiếp theo là full VN3K T2I Rank@1 của nhánh vision đã quantize, với deploy gate hiện tại `>= 48.0` so với FP32 baseline là `52.28`.

---

## 1. Tại Sao Có Pipeline Này

Phần cứng đích không phải là một accelerator thông thường. RB3 Gen2 sử dụng HTP v68 và deployment pipeline bị ràng buộc bởi sự hỗ trợ QNN context-binary:

- Các floating-point tensor tại đồ thị I/O bị từ chối đối với HTP context binary.
- Internal floating-point fallback cũng gây ra nhiều lỗi link liên tiếp.
- Hỗ trợ A16 activation bị hạn chế trên v68; activation-by-activation matmul của attention và LayerNorm với A16 yêu cầu HTP mới hơn, theo quan sát là `expected >= 73`.
- All-INT8 W8A8 là con đường mà v68 hỗ trợ rộng rãi.

Tuy nhiên, mô hình ban đầu không thân thiện với W8A8. Plain per-tensor INT8 quantization làm sụp đổ embedding direction, với cosine quanh mức `0.14-0.17`. Lỗi này không phải do preprocessing hay do xuất ONNX: static ONNX vs PyTorch xấp xỉ `1.0`. Nó bị gây ra bởi activation quantization trong vision encoder.

Cách sửa lỗi cuối cùng không phải là clip các outlier hoặc giữ các layer ở dạng float. Việc clip thất bại vì các outlier channel mang tín hiệu thực; float surgery vượt qua bài kiểm tra độ trung thực ONNX cục bộ nhưng thất bại khi link QNN. Cách sửa lỗi có thể deploy là rotate residual channel basis offline để cùng một tín hiệu đó được phân tán ra khắp các channel. Việc này làm giảm áp lực per-tensor INT8 trong khi vẫn bảo toàn chức năng FP32.

---

## 2. Tổng Quan Pipeline

```text
[0] Input checkpoint
    epoch=56-val_score=52.28.ckpt
    Lightning checkpoint với mô hình TBPS mSigLIP finetuned bằng LoRA

[1] Merge LoRA
    deployment/scripts/lora_fp16/export.py
    -> artifacts/deployment/exports/exported_model/
       model_fp32.pt, model_fp16.pt, config.yaml

[2] Rotate vision encoder
    deployment/scripts/qnn/rotate_vision_encoder.py
    -> artifacts/deployment/exports/exported_model_rotated/
       model_fp32.pt, config.yaml, rotation_summary.json

[3] Export rotated vision ONNX
    deployment/scripts/qnn/export_rotated_vision_onnx.py
    -> artifacts/deployment/exports/exported_model_rotated/vision_onnx/
       vision_encoder.onnx, vision_encoder.onnx.data

[4] Quantize + compile/link
    deployment/scripts/qnn/submit_qaihub_quantize_compile.py
    -> artifacts/deployment/runtime/rotated_w8a8_v2/vision_encoder.bin

[5] Run trên RB3
    qnn-net-run với libQnnHtp.so và htp_config_245.json
    -> artifacts/deployment/qnn_runs/rotated_w8a8_v2/

[6] So sánh board output
    deployment/scripts/qnn/compare_qnn_with_pytorch.py
    -> So sánh cosine QNN(board) vs PyTorch và các kiểm tra finite-output
```

Tất cả các artifact được tạo ra được giữ trong thư mục `artifacts/deployment/`.

---

## 3. Khối [1] - Merge LoRA Vào Base Model

Script:

```text
deployment/scripts/lora_fp16/export.py
```

Lệnh:

```bash
python deployment/scripts/lora_fp16/export.py \
  --ckpt artifacts/models/checkpoints/epoch=56-val_score=52.28.ckpt \
  --output-dir artifacts/deployment/exports/exported_model
```

Output:

```text
artifacts/deployment/exports/exported_model/
  model_fp32.pt
  model_fp16.pt
  config.yaml
```

Tại sao khối này là bắt buộc:

Training checkpoint là một PyTorch Lightning checkpoint với các LoRA adapter. Để deployment, chúng ta cần một ordinary inference state dict, không phải là một base model cộng với các adapter module. Các công cụ của Qualcomm chỉ nhìn thấy các trọng số đã được export; chúng không hiểu PEFT LoRA semantics.

LoRA sửa đổi một dense weight matrix theo dạng:


$$W_{merged} = W_{base} + (\alpha / r) * B A$$


trong đó:

- $W_{base}$ là frozen model weight ban đầu.
- $A$ và $B$ là các low-rank LoRA matrix.
- $r$ là LoRA rank.
- $\alpha$ là LoRA scaling factor.

Script tái cấu trúc lại `LitTBPS`, áp dụng setup LoRA đã cấu hình, load checkpoint state dict, sau đó gọi PEFT `merge_and_unload()` khi backbone là một `PeftModel`. Sau bước này, `model_fp32.pt` được lưu chỉ nên chứa normal weights.

Verification được sử dụng trong lần chạy thành công:

```text
model_fp32.pt has 0 keys containing lora / adapter / base_layer.
```

Tại sao cả FP32 và FP16 đều được lưu:

- `model_fp32.pt` là accuracy/reference source dùng để export ONNX và các fidelity comparison.
- `model_fp16.pt` hữu ích cho các fallback runtime experiment và size check.
- Path HTP thành công cuối cùng sử dụng W8A8, nhưng FP32 vẫn là reference model cho tất cả các script compare.

Quy tắc quan trọng:

Khi chuyển từ checkpoint `52.28` sang checkpoint `53.00` sau này, hãy chạy lại khối này từ đầu. Các artifact của rotation và quantization là đặc thù cho từng bộ trọng số và không được sử dụng lại cho các checkpoint khác nhau.

---

## 4. Khối [2] - Chuẩn Bị Real VN3K Inputs và Calibration Data

Script:

```text
deployment/scripts/qnn/prepare_vn3k_vision_inputs.py
```

Smoke input command:

```bash
python deployment/scripts/qnn/prepare_vn3k_vision_inputs.py \
  --dataset-root VN3K \
  --split test \
  --selection first \
  --num-samples 10 \
  --output-dir artifacts/deployment/qnn_inputs/vn3k_test_10 \
  --path-mode relative
```

Calibration command:

```bash
python deployment/scripts/qnn/prepare_vn3k_vision_inputs.py \
  --dataset-root VN3K \
  --split train \
  --selection random \
  --seed 2400 \
  --num-samples 2000 \
  --output-dir artifacts/deployment/qnn_inputs/vn3k_train_calib_2000 \
  --path-mode relative
```

Dataset calibration đã biết trên AI Hub:

```text
d7jzjy1m2 / msiglip-vision-vn3k-train-calib-2000
```

Preprocessing:

```text
RGB image
  -> resize 256 x 256, bicubic
  -> ToTensor, channel-first NCHW
  -> Normalize(mean=0.5, std=0.5)
  -> float32 raw tensor in range [-1, 1]
```

Mỗi raw input là:

```text
1 x 3 x 256 x 256 float32 = 786432 bytes
```

Tại sao khối này quan trọng:

QNN comparison phải isolate quantization/runtime error. Các raw tensor được dùng bởi `qnn-net-run` giống với các tensor được dùng bởi các script so sánh PyTorch/ONNX local. Việc này loại bỏ JPEG decoding, resize, color conversion, và normalization drift khỏi quá trình đo lường.

---

## 5. Khối [3] - Tại Sao Cần Phải Có Opset 20 Fused Gelu

Trước final rotation pipeline, một root cause quan trọng đã được tìm ra: việc export opset-18 đã decompose tanh-GELU thành các primitive ops bao gồm cả `Pow(x, 3)`.

Tanh GELU approximation là:


$$GELU(x) = 0.5 x [1 + \tanh (\sqrt{\frac2\pi} * (x + 0.044715 x^3))]$$

Nếu biểu thức này được export dưới dạng primitive ONNX ops, internal tensor `x^3` có thể trở nên rất lớn. Trong đồ thị được quan sát, điều này đã tạo ra các activation range quanh mức `119k`, dominate hoàn toàn per-tensor quantization scale. Thậm chí W8A16 cũng không thể deploy thành công trên v68 sau con đường này, và một số link failures tập trung xung quanh các `gelu_*` tensor.

Cách sửa lỗi là export với ONNX opset 20, trong đó `Gelu` có sẵn dưới dạng một fused operator. Việc này giữ cho cubic implementation nằm bên trong runtime operator thay vì expose `Pow(x, 3)` thành một quantized tensor.

Expected op signature trong ONNX thành công:

```text
Gelu = 13
Pow = 0
Tanh = 0
LayerNormalization = 26
ReduceMean = 0
```

Đây là lý do tại sao mọi successful branch sau breakthrough này phải giữ nguyên:

```text
opset 20 + fused Gelu + fused LayerNormalization
```

---

## 6. Khối [4] - Mean-Preserving Rotation Equalization

Script:

```text
deployment/scripts/qnn/rotate_vision_encoder.py
```

Lệnh:

```bash
python deployment/scripts/qnn/rotate_vision_encoder.py \
  --model-dir artifacts/deployment/exports/exported_model \
  --output-dir artifacts/deployment/exports/exported_model_rotated \
  --input-dir artifacts/deployment/qnn_inputs/vn3k_test_10 \
  --seed 2400
```

Output:

```text
artifacts/deployment/exports/exported_model_rotated/
  model_fp32.pt
  config.yaml
  rotation_summary.json
```

### 6.1 Vấn Đề Quantization

Plain W8A8 per-tensor quantization thất bại vì một vài channel ở residual-stream có magnitude lớn hơn nhiều so với các channel còn lại. Per-tensor quantization sử dụng chung một scale cho toàn bộ tensor:

```text
x_int = round(x / s) + z
x_hat = s * (x_int - z)
```

với một scale `s` đại khái được gắn với tensor maximum. Nếu một channel rất lớn, `s` sẽ trở nên lớn và các ordinary channel bị mất resolution. Điều này đặc biệt tệ đối với embedding retrieval vì metric phụ thuộc vào direction sau L2 normalization, chứ không chỉ ở raw scale.

Việc clip đã được thử nghiệm và thất bại. Điều đó có nghĩa là các large channel không phải là harmless noise; chúng mang useful information. Phép toán đúng là bảo toàn năng lượng trong khi phân tán nó ra các dimension khác nhau.

### 6.2 Ý Tưởng Rotation

Giả sử `x` là một 768-dimensional residual vector. Chọn một orthogonal matrix `Q` và biểu diễn internal residual stream thành:

```text
x_rot = Q x
```

Vì `Q` là orthogonal:

```text
||Qx||_2 = ||x||_2
```

Semantic information được bảo toàn, nhưng năng lượng của một spiky channel có thể được phân tán sang nhiều channel. Điều này làm cho per-tensor INT8 quantization ít bị chi phối bởi một coordinate.

Rotation được fold vào existing weights offline. Không thêm một runtime `MatMul` mới nào.

### 6.3 Tại Sao Q Phải Preserve the Mean

Một phiên bản cũ đã chuyển LayerNorm thành RMSNorm để một arbitrary orthogonal rotation có thể commute qua normalization. Phiên bản đó đã thất bại vì RMSNorm được export thành `Pow(x, 2)`, `ReduceMean`, và các phép chia. Những normalization internals đó bị quantizer nhìn thấy và lại làm sụp đổ W8A8 một lần nữa.

Phiên bản thành công giữ lại fused `LayerNormalization`, do đó rotation phải commute với LayerNorm. Một standard LayerNorm với identity affine là:

```text
LN(x) = (x - mean(x) * 1) / std(x)
```

Với một arbitrary orthogonal `Q`, `mean(Qx)` nói chung không bằng `mean(x)`, do đó LayerNorm không commute.

Cách sửa lỗi là xây dựng `Q` sao cho nó preserve all-ones direction:

```text
Q 1 = 1
Q^T Q = I
```

Sau đó:

```text
mean(Qx) = mean(x)
Q(x - mean(x)1) = Qx - mean(x)1
std(Qx) = std(x)
```

Do đó:

```text
LN(Qx) = Q LN(x)
```

Điều này cho phép chúng ta rotate residual stream trong khi vẫn giữ `LayerNormalization` làm một fused ONNX/QNN operator.

Script tạo ra:

```text
Q = U blockdiag(1, R_c) U^T
```

trong đó:

- column đầu tiên của `U` là `1 / sqrt(d)`;
- `R_c` là một random orthogonal matrix trên `(d - 1)` dimensional subspace orthogonal với mean direction;
- `d = 768`.

### 6.4 Fold LayerNorm Affine Vào Reader

LayerNorm chứa các affine parameter:

```text
LN_affine(x) = gamma * LN_identity(x) + beta
```

Nếu một linear reader consume nó:

```text
y = W (gamma * h + beta) + b
```

thì affine có thể được fold vào reader:

```text
W' = W diag(gamma)
b' = b + W beta
```

Sau lần fold này, LayerNorm module có thể được set về identity affine:

```text
gamma = 1
beta = 0
```

Việc này giữ cùng một FP32 function trong khi làm cho normalization compatible với rotation.

Các reader được fold trong vision encoder:

- `q_proj`, `k_proj`, `v_proj` sau `layer_norm1`;
- `mlp.fc1` sau `layer_norm2`;
- K/V slices của `head.attention` sau `post_layernorm`.

### 6.5 Fold Q Vào Writer Và Q^T Vào Reader

Đối với một residual writer:

```text
y = W x + b
```

chúng ta muốn residual trở thành `Qy`, do đó:

```text
W_writer' = Q W
b_writer' = Q b
```

Các writer được fold:

- patch embedding output channels;
- position embedding rows;
- mỗi `out_proj` của self-attention;
- mỗi `fc2` của MLP.

Đối với một residual reader:

```text
y = W x_rot + b
```

nơi `x_rot = Qx`, chúng ta cần reader nhìn thấy original basis:

```text
W_reader' = W Q^T
```

Các reader được fold:

- mỗi `q_proj`, `k_proj`, `v_proj` của self-attention;
- mỗi `fc1` của MLP;
- K/V slices của final pooling head attention.

Learned query/probe ở head không bị rotate. Rotation được localize trong encoder residual stream và undo ở head K/V boundary.

### 6.6 Rotation Gates

Rotation block chỉ được accepted nếu FP32 output là invariant:

```text
cosine(original encode_image, rotated encode_image) ~= 1.0
```

Các successful observed check:

```text
Phase A invariance cosine mean/min ~= 1.0
Phase B invariance cosine mean/min ~= 1.0
Q orthogonality max error ~= 3e-15
Q@1=1 max error ~= 1e-14
reload cosine ~= 1.0
residual concentration 252x -> 5.3x
```

Reduction từ `252x` xuống `5.3x` là quantization effect quan trọng: signal vẫn hiện diện, nhưng nó không còn tập trung ở một dominant channel nào nữa.

---

## 7. Khối [5] - Export Rotated Vision ONNX

Script:

```text
deployment/scripts/qnn/export_rotated_vision_onnx.py
```

Lệnh:

```bash
python deployment/scripts/qnn/export_rotated_vision_onnx.py \
  --model-dir artifacts/deployment/exports/exported_model_rotated \
  --opset 20
```

Output:

```text
artifacts/deployment/exports/exported_model_rotated/vision_onnx/
  vision_encoder.onnx
  vision_encoder.onnx.data
```

Tại sao ONNX được lưu dưới dạng một thư mục:

Mô hình SigLIP có các external weights lớn. AI Hub kỳ vọng `.onnx` graph và file external data của nó sẽ được upload cùng nhau dưới dạng một directory, chứ không phải một `.onnx` file đơn lẻ.

Required export properties:

```text
opset = 20
Pow = 0
Tanh = 0
Gelu = 13
LayerNormalization = 26
ReduceMean = 0
```

Static-control gate là:

```text
ONNX rotated vs PyTorch rotated cosine_l2_mean ~= 1.0
ONNX rotated vs PyTorch rotated cosine_l2_min  ~= 1.0
```

Bởi vì rotated PyTorch model là output-invariant, điều này cũng có nghĩa là export vẫn khớp với original merged PyTorch model.

Important implementation detail:

Successful rotation giữ regular LayerNorm module. Nhánh RMSNorm cũ hơn đã bị rejected sau khi nó expose `Pow(x, 2)` cho quantizer và trả về QDQ cosine khoảng `0.16`. Bất kỳ export nào show ra RMSNorm-style `Pow` hay `ReduceMean` cluster đều không phải là successful pipeline.

---

## 8. Khối [6] - AI Hub Quantize, Compile, Và Link

Script:

```text
deployment/scripts/qnn/submit_qaihub_quantize_compile.py
```

Successful command shape:

```bash
python deployment/scripts/qnn/submit_qaihub_quantize_compile.py \
  --model artifacts/deployment/exports/exported_model_rotated/vision_onnx \
  --calibration-data d7jzjy1m2 \
  --weights-dtype int8 \
  --activations-dtype int8 \
  --wait \
  --download artifacts/deployment/runtime/rotated_w8a8_v2/vision_encoder.bin
```

Helper này làm gì:

1. Copy ONNX directory và rewrite input shapes thành static `image: 1x3x256x256`.
2. Resolve `d7jzjy1m2` thành một AI Hub calibration dataset.
3. Chạy `submit_quantize_job()` với W8A8.
4. Chạy `submit_compile_and_link_jobs()` target `Dragonwing RB3 Gen 2 Vision Kit`.
5. Sử dụng compile option `--quantize_io` để QNN graph I/O được quantized thay vì preserve floating-point boundary tensors.

Successful job chain:

| Job | Stage | Result |
|---|---|---|
| `jpv4j8lkp` | quantize W8A8 | SUCCESS |
| `jpxmwq8lg` | compile DLC | SUCCESS |
| `jp2j211q5` | link QNN context binary | SUCCESS |

Output:

```text
artifacts/deployment/runtime/rotated_w8a8_v2/vision_encoder.bin
size: 89.7 MB
```

QDQ gate result trước khi chạy board:

```text
cosine_l2_mean = 0.8975
cosine_l2_min  = 0.8747
cosine_l2_max  = 0.9297
```

Mức này thấp hơn conservative QDQ gate ban đầu là `0.95/0.90`, nhưng nó là một bước nhảy vọt so với all-INT8 range trước đây là `0.14-0.17`, và graph này có thể deploy trên v68. Tại thời điểm này, decision là chạy board và sau đó đo real retrieval R@1 thay vì over-optimize proxy cosine.

Tại sao lần này cuối cùng cũng link thành công:

- Không có floating-point I/O boundary do `--quantize_io`.
- Không có internal float surgery.
- Không có A16 activation yêu cầu v73.
- Không có decomposed GELU cubic.
- Không có decomposed RMSNorm normalization internals.
- Tất cả main activation là W8A8, mức v68 hỗ trợ rộng rãi.

---

## 9. Khối [7] - Run QNN Context Binary Trên RB3

Runtime command pattern:

```bash
qnn-net-run \
  --backend "$QNN_LIB/libQnnHtp.so" \
  --retrieve_context artifacts/deployment/runtime/rotated_w8a8_v2/vision_encoder.bin \
  --config_file deployment/config/qnn/htp_config_245.json \
  --input_list artifacts/deployment/qnn_inputs/vn3k_test_10/input_list.txt \
  --output_dir artifacts/deployment/qnn_runs/rotated_w8a8_v2 \
  --profiling_level basic \
  --perf_profile high_performance
```

HTP config:

```text
deployment/config/qnn/htp_config_245.json
```

Graph I/O types là quantized fixed-point:

```text
image    -> QNN_DATATYPE_UFIXED_POINT_8
output_0 -> QNN_DATATYPE_UFIXED_POINT_8
```

Mặc định, `qnn-net-run` ghi các `.raw` file dequantized float output, do đó `compare_qnn_with_pytorch.py` đọc chúng thành các float32 embeddings.

Board outcome:

```text
10 / 10 inferences completed
no NaN / Inf in outputs
QNN(board) vs PyTorch cosine_l2_mean = 0.8982
QNN(board) vs PyTorch cosine_l2_min  = 0.8606
QNN(board) vs PyTorch cosine_l2_max  = 0.9283
```

Board result khớp với QDQ ONNX:

```text
QDQ ONNX mean: 0.8975
QNN board mean: 0.8982
difference: khoảng 0.0007
```

Điều này có nghĩa là HTP runtime faithful với quantized graph. Remaining error là quantization error, không phải QNN runtime drift hay I/O parsing error.

---

## 10. Khối [8] - Compare QNN Output Vs PyTorch

Script:

```text
deployment/scripts/qnn/compare_qnn_with_pytorch.py
```

Command pattern:

```bash
python deployment/scripts/qnn/compare_qnn_with_pytorch.py \
  --qnn-output-dir artifacts/deployment/qnn_runs/rotated_w8a8_v2 \
  --model-dir artifacts/deployment/exports/exported_model \
  --input-dir artifacts/deployment/qnn_inputs/vn3k_test_10 \
  --precision fp32 \
  --json artifacts/deployment/qnn_runs/rotated_w8a8_v2/qnn_vs_pytorch_summary.json \
  --csv artifacts/deployment/qnn_runs/rotated_w8a8_v2/qnn_vs_pytorch.csv
```

Tại sao compare với `exported_model`, chứ không phải `exported_model_rotated`:

Rotated model là mathematically output-invariant. Deployment contract không phải là "match rotated implementation"; mà là "match original merged FP32 model." Compare với original `exported_model` sẽ đo được full deployment error mà retrieval thấy.

Metrics:

- `cosine_raw`: cosine giữa các raw unnormalized vectors.
- `cosine_l2`: cosine sau khi L2 normalization, relevant proxy nhất cho retrieval embedding direction.
- `l2_l2`: L2 distance giữa các normalized embeddings.
- NaN/Inf flags cho QNN và PyTorch outputs.

Successful board run có finite 768-d outputs và QNN cosine align với QDQ proxy.

---

## 11. Performance

Đo bằng `qnn-profile-viewer` trên RB3/aarch64 toolchain:

| Metric | Result |
|---|---:|
| NetRun average per inference | `34250 us` |
| NetRun min/max | `32958 / 35388 us` |
| Accelerator execute average | `32478 us` |
| Init/load binary | `54688 us` |
| HVX threads | `4` |
| Throughput | `22.5 FPS` |

Interpretation:

- Context binary khoảng `89.7 MB`, đủ nhỏ đối với 4 GB board.
- HTP runtime đủ nhanh cho một practical camera/image ingestion path.
- Text encoder deployment vẫn được tách biệt và có thể có những memory và latency characteristics khác nhau vì large multilingual embedding table.

---

## 12. Tại Sao Các Candidate Trước Đó Thất Bại

Bảng này explain tại sao final pipeline lại có chính xác shape này.

| Attempt | Result | Lesson |
|---|---|---|
| FP32/FP16 ONNX chạy direct lên HTP | link fail | v68 context binary yêu cầu integer I/O. |
| Deprecated CLI INT8 path | thường preserve FP I/O | Use Python API quantize + compile/link và `--quantize_io`. |
| Dummy/real W8A8 without rotation | cosine `~0.13-0.17` | Runtime hoạt động, nhưng quantization đã phá hỏng embedding direction. |
| Thêm calibration samples | still failed | Failure không nằm ở calibration coverage. |
| Lite-MP / min-max / W8A16 trên old graph | failed | Global knobs không resolve được exposed GELU/outliers/concentration issues. |
| `_float` QDQ surgery | local pass, link fail | HTP v68 reject internal float tensors. |
| ORT W8A16 QDQ | local `~0.999`, link fail | QNN linker reject internal float/dequantized GELU patterns. |
| QAT fake-quant proxy | weak correlation | AI Hub QDQ không follow PyTorch fake-quant proxy. |
| Opset 20 + W8A16 | QDQ `0.9997`, link fail | A16 attention/LayerNorm cần HTP v73+, nhưng RB3 lại là v68. |
| Clipping INT8 activations | failed | Outlier channels mang useful signal; clipping sẽ làm mất information. |
| Rotation với RMSNorm | cosine `~0.16` | RMSNorm expose `Pow(x^2)` và normalization internals cho quantization. |
| Mean-preserving rotation + fused LN + W8A8 | board pass | All-INT8, không exposed cubic/norm internals, v68-compatible. |

---

## 13. Reproducible Command Sequence

### 13.1 Merge LoRA

```bash
python deployment/scripts/lora_fp16/export.py \
  --ckpt artifacts/models/checkpoints/epoch=56-val_score=52.28.ckpt \
  --output-dir artifacts/deployment/exports/exported_model
```

### 13.2 Prepare calibration và smoke inputs

```bash
python deployment/scripts/qnn/prepare_vn3k_vision_inputs.py \
  --dataset-root VN3K \
  --split test \
  --selection first \
  --num-samples 10 \
  --output-dir artifacts/deployment/qnn_inputs/vn3k_test_10 \
  --path-mode relative
```

```bash
python deployment/scripts/qnn/prepare_vn3k_vision_inputs.py \
  --dataset-root VN3K \
  --split train \
  --selection random \
  --seed 2400 \
  --num-samples 2000 \
  --output-dir artifacts/deployment/qnn_inputs/vn3k_train_calib_2000 \
  --path-mode relative
```

Upload calibration nếu cần:

```bash
python deployment/scripts/qnn/upload_qaihub_calibration_dataset.py \
  --input-dir artifacts/deployment/qnn_inputs/vn3k_train_calib_2000 \
  --name msiglip-vision-vn3k-train-calib-2000
```

Known dataset ID:

```text
d7jzjy1m2
```

### 13.3 Rotate vision encoder

```bash
python deployment/scripts/qnn/rotate_vision_encoder.py \
  --model-dir artifacts/deployment/exports/exported_model \
  --output-dir artifacts/deployment/exports/exported_model_rotated \
  --input-dir artifacts/deployment/qnn_inputs/vn3k_test_10 \
  --seed 2400
```

Gate:

```text
FP32 encode_image cosine vs original ~= 1.0
```

### 13.4 Export rotated ONNX opset 20

```bash
python deployment/scripts/qnn/export_rotated_vision_onnx.py \
  --model-dir artifacts/deployment/exports/exported_model_rotated \
  --opset 20
```

Gate:

```text
Pow = 0
Tanh = 0
Gelu = 13
LayerNormalization = 26
ONNX vs PyTorch cosine ~= 1.0
```

### 13.5 Quantize-only check

```bash
python deployment/scripts/qnn/submit_qaihub_quantize_compile.py \
  --model artifacts/deployment/exports/exported_model_rotated/vision_onnx \
  --calibration-data d7jzjy1m2 \
  --weights-dtype int8 \
  --activations-dtype int8 \
  --quantize-only \
  --wait \
  --download-quantized artifacts/deployment/runtime/rotated_w8a8_v2/qaihub_qdq
```

Compare:

```bash
python deployment/scripts/qnn/compare_onnx_with_pytorch.py \
  --onnx-model artifacts/deployment/runtime/rotated_w8a8_v2/qaihub_qdq \
  --model-dir artifacts/deployment/exports/exported_model \
  --input-dir artifacts/deployment/qnn_inputs/vn3k_test_10 \
  --precision fp32 \
  --json artifacts/deployment/runtime/rotated_w8a8_v2/qdq_vs_pytorch_summary.json \
  --csv artifacts/deployment/runtime/rotated_w8a8_v2/qdq_vs_pytorch.csv
```

Observed:

```text
cosine_l2_mean = 0.8975
```

### 13.6 Quantize, compile, link, và download binary

```bash
python deployment/scripts/qnn/submit_qaihub_quantize_compile.py \
  --model artifacts/deployment/exports/exported_model_rotated/vision_onnx \
  --calibration-data d7jzjy1m2 \
  --weights-dtype int8 \
  --activations-dtype int8 \
  --wait \
  --download artifacts/deployment/runtime/rotated_w8a8_v2/vision_encoder.bin
```

Expected:

```text
quantize SUCCESS
compile SUCCESS
link SUCCESS
vision_encoder.bin ~= 89.7 MB
```

### 13.7 Run trên RB3

```bash
qnn-net-run \
  --backend "$QNN_LIB/libQnnHtp.so" \
  --retrieve_context artifacts/deployment/runtime/rotated_w8a8_v2/vision_encoder.bin \
  --config_file deployment/config/qnn/htp_config_245.json \
  --input_list artifacts/deployment/qnn_inputs/vn3k_test_10/input_list.txt \
  --output_dir artifacts/deployment/qnn_runs/rotated_w8a8_v2 \
  --profiling_level basic \
  --perf_profile high_performance
```

### 13.8 Compare board output

```bash
python deployment/scripts/qnn/compare_qnn_with_pytorch.py \
  --qnn-output-dir artifacts/deployment/qnn_runs/rotated_w8a8_v2 \
  --model-dir artifacts/deployment/exports/exported_model \
  --input-dir artifacts/deployment/qnn_inputs/vn3k_test_10 \
  --precision fp32 \
  --json artifacts/deployment/qnn_runs/rotated_w8a8_v2/qnn_vs_pytorch_summary.json \
  --csv artifacts/deployment/qnn_runs/rotated_w8a8_v2/qnn_vs_pytorch.csv
```

Observed:

```text
cosine_l2_mean = 0.8982
cosine_l2_min  = 0.8606
cosine_l2_max  = 0.9283
```

---

## 14. Acceptance Gates

| Gate | Threshold | Current status |
|---|---:|---|
| Merge LoRA tạo ra clean non-adapter weights | no `lora`/`adapter` keys | PASS |
| Rotation FP32 invariance | cosine min `>= 0.9999` | PASS |
| ONNX static control | cosine mean `>= 0.999` | PASS |
| ONNX op sanity | `Pow=0`, fused `Gelu`, fused `LayerNormalization` | PASS |
| QDQ ONNX vs PyTorch | original proxy `>= 0.95/0.90` | NEAR, `0.8975/0.8747` |
| QNN board vs PyTorch | practical gate `>= 0.90` mean | NEAR, `0.8982` |
| Board execution | finite outputs, HTP profile | PASS |
| Full retrieval | T2I R@1 `>= 48.0` | TODO |

QDQ/QNN cosine proxy khá conservative và được set trước khi W8A8 path dựa vào rotation này tồn tại. Do binary hiện đã link thành công và board output đã match QDQ, bài kiểm tra decisive tiếp theo sẽ là retrieval Rank@1.

---

## 15. Common Mistakes Cần Tránh

- Không bỏ qua LoRA merge. Checkpoint không phải trực tiếp là deployment model.
- Không reuse artifact về rotation hoặc QDQ trên các checkpoint khác nhau.
- Không export với opset 18 cho final path; nó có thể expose `Pow(x, 3)` từ tanh-GELU.
- Không convert LayerNorm thành RMSNorm cho v68 W8A8 path này; nó expose normalization internals cho QDQ.
- Không compile các candidate `_float` QDQ surgery; chúng pass local diagnostics nhưng fail HTP link.
- Không sử dụng W8A16 làm final v68 plan; nó pass fidelity nhưng fail link tại attention/LayerNorm vì v68 thiếu sự hỗ trợ A16 cần thiết.
- Không trust PSNR của AI Hub/AIMET làm retrieval embedding proxy. Luôn luôn chạy compare cosine embedding.
- Không sử dụng `snpe-net-run` cho QNN context binaries; use `qnn-net-run`.
- Không preserve floating-point graph I/O cho HTP context binaries; use `--quantize_io`.

---

## 16. Những Việc Còn Lại (What Remains)

1. Đo full VN3K T2I Rank@1 bằng cách sử dụng quantized/rotated vision embeddings. Deploy gate là `>= 48.0` so với `52.28` FP32 baseline.
2. Replicate successful pattern cho text encoder:
   opset-20 fused Gelu, inspect/handle concentration, quantize về một v68-safe graph, sau đó compare trước khi compile/link.
3. Nếu retrieval R@1 dưới gate, hãy improve rotation path thay vì back về float surgery hay A16. Candidate improvements bao gồm deeper QuaRot style rotations xung quanh attention/MLP internal projections.
4. Update `deployment/README.md` và deployment changelog sau explicit changelog/docs confirmation nếu report này trở thành canonical guide.
