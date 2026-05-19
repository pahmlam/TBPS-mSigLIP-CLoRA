# Knowledge Base

> Tài liệu kiến thức tích lũy trong quá trình nghiên cứu và deploy mSigLIP.
> Mỗi mục ghi lại: **định nghĩa** các khái niệm liên quan, **vì sao** cần làm, **làm gì**, **làm như thế nào**, và **suy nghĩ/cách tiếp cận**  khi giải quyết vấn đề.

<!-- TEMPLATE CHO MỤC MỚI

## N. [Tiêu đề]

> **Ngày:** YYYY-MM  
> **Liên quan:** `file/path`, `another/path`

### Định nghĩa
- **Thuật ngữ 1:** Giải thích ngắn gọn
- **Thuật ngữ 2:** Giải thích ngắn gọn

### Vì sao (WHY)
Giải thích vấn đề / động lực.

### Làm gì (WHAT)
Mô tả giải pháp / hành động cụ thể.

### Làm như thế nào (HOW)
Code, commands, hoặc chi tiết kỹ thuật.

### Suy nghĩ & cách tiếp cận
Phân tích, trade-offs, lý do chọn cách này thay vì cách khác.
-->
---

## Mục lục

1. [Export model trước khi deploy lên RB3](#1-export-model-trước-khi-deploy-lên-rb3)
2. [File .pt vs .ckpt — Checkpoint formats trong PyTorch](#2-file-pt-vs-ckpt--checkpoint-formats-trong-pytorch)
3. [Activations — Bộ nhớ trung gian khi inference](#3-activations--bộ-nhớ-trung-gian-khi-inference)
4. [Qualcomm AI Engine SDK — ONNX → DLC → DSP/HTP](#4-qualcomm-ai-engine-sdk--onnx--dlc--dsphtp)
5. [ONNX — Định dạng trung gian phổ quát cho model AI](#5-onnx--định-dạng-trung-gian-phổ-quát-cho-model-ai)
6. [Idea C — Unified Noise-Aware Circle Loss (NACIR)](#6-idea-c--unified-noise-aware-circle-loss-nacir)
7. [Thiết kế hệ thống end-to-end cho mSigLIP trên RB3 Gen2](#7-thiết-kế-hệ-thống-end-to-end-cho-msiglip-trên-rb3-gen2)
8. [Noisy Correspondence Injection — Tiêm nhiễu cặp ảnh-văn bản](#8-noisy-correspondence-injection--tiêm-nhiễu-cặp-ảnh-văn-bản)
9. [INT8 Quantization cho HTP — `--preserve_io_datatype` và pipeline compile](#9-int8-quantization-cho-htp----preserve_io_datatype-và-pipeline-compile)
10. [Đánh giá output `notebooks/workspace.ipynb` cho NACIR](#10-đánh-giá-output-workspaceipynb-cho-nacir)
11. [Dataset có thể không có FN/FP không?](#11-dataset-có-thể-không-có-fnfp-không)
12. [Chủ động tạo nhiễu có giúp model cải thiện không?](#12-chủ-động-tạo-nhiễu-có-giúp-model-cải-thiện-không)
13. [Notebook-only controlled validation cho NACIR](#13-notebook-only-controlled-validation-cho-nacir)
14. [Nếu VN3K sạch thì tăng R@1 bằng hướng nào?](#14-nếu-vn3k-sạch-thì-tăng-r1-bằng-hướng-nào)
15. [Nhiễu được inject trong repo RDE-mSigLIP-3000VnPersonsearch](#15-nhiễu-được-inject-trong-repo-rde-msiglip-3000vnpersonsearch)
16. [Cập nhật README theo tiến độ NACIR và deployment](#16-cập-nhật-readme-theo-tiến-độ-nacir-và-deployment)
17. [Dọn placeholder `my_new_loss` khỏi workspace notebook](#17-dọn-placeholder-my_new_loss-khỏi-workspace-notebook)
18. [Chạy `vision_encoder.bin` trên RB3 bằng QNN HTP runtime](#18-chạy-vision_encoderbin-trên-rb3-bằng-qnn-htp-runtime)
19. [Bối cảnh nghiên cứu trước mSigLIP-CLoRA](#19-bối-cảnh-nghiên-cứu-trước-msiglip-clora)
20. [Slide hàm loss của mSigLIP-CLoRA](#20-slide-hàm-loss-của-msiglip-clora)
21. [Đánh giá output mới nhất của `notebooks/workspace.ipynb`](#21-đánh-giá-output-mới-nhất-của-workspaceipynb)
22. [Tune FN branch của NACIR trong notebook](#22-tune-fn-branch-của-nacir-trong-notebook)
23. [RB3-first modular demo cho hệ thống end-to-end](#23-rb3-first-modular-demo-cho-hệ-thống-end-to-end)
24. [Tái cấu trúc repo theo layout AI project chuẩn](#24-tái-cấu-trúc-repo-theo-layout-ai-project-chuẩn)
25. [Đánh giá QNN HTP output đầu tiên trên VN3K test 10](#25-đánh-giá-qnn-htp-output-đầu-tiên-trên-vn3k-test-10)
26. [Đánh giá output NACIR sau tuning FN sweep](#26-đánh-giá-output-nacir-sau-tuning-fn-sweep)

---

## 1. Export model trước khi deploy lên RB3

> **Ngày:** 2026-04  
> **Liên quan:** `deployment/scripts/lora_fp16/export.py`, `deployment/docs/system.md`

### Định nghĩa

- **Lightning Checkpoint (.ckpt):** File lưu trạng thái đầy đủ của quá trình training — bao gồm model weights (state_dict), optimizer states (Adam momentum + variance), learning rate scheduler, epoch/step counter, và Hydra config. Mục đích: resume training.
- **LoRA (Low-Rank Adaptation):** Kỹ thuật fine-tuning hiệu quả, thêm 2 ma trận nhỏ A (d×r) và B (r×d) vào mỗi attention layer. Output = W·x + B·A·x. Chỉ train A, B (r=32 → ~1.5% tổng params).
- **LoRA Merge:** Cộng sẵn W_merged = W + B·A, loại bỏ adapter. Kết quả toán học giống hệt nhưng chỉ cần 1 matmul thay vì 2 matmul + 1 add mỗi layer.
- **FP16 (Half-precision):** Biểu diễn số thực 16-bit thay vì FP32 (32-bit). Giảm một nửa dung lượng và RAM, tốc độ nhanh hơn trên phần cứng hỗ trợ (ARM NEON có `fphp` flag).

### Vì sao (WHY)

Qualcomm RB3 Gen2 chỉ có ~4 GB RAM khả dụng. Lightning checkpoint 1.4 GB chứa nhiều dữ liệu không cần cho inference:
- Optimizer states: ~160 MB (Adam momentum + variance cho trainable params)
- Training metadata: epoch, lr scheduler state, Hydra config
- LoRA adapters chưa merge: tăng computation (192 phép tính thừa mỗi forward pass)

`torch.load()` đọc **toàn bộ file** vào RAM trước khi lọc → peak RAM ~3.5–3.8 GB, sát giới hạn RB3 → **nguy cơ OOM**.

### Làm gì (WHAT)

Pipeline export 4 bước:
1. Load full checkpoint trên máy dev (RAM thoải mái)
2. Bỏ optimizer states — chỉ giữ state_dict
3. Merge LoRA vào base model — W_merged = W + B·A
4. Chuyển FP32 → FP16

### Làm như thế nào (HOW)

Pipeline gồm 3 bước: **Phân tích → Export → Test inference**.

#### Bước 1: Phân tích checkpoint (`analyze_checkpoint.py`)

1. Load toàn bộ Lightning checkpoint lên CPU
2. Extract `state_dict` (model weights, có thể chứa LoRA adapters)
3. **Đếm params với deduplication:** duyệt state_dict, track `data_ptr()` của mỗi tensor (địa chỉ bộ nhớ thực). Nếu 2 key trỏ đến cùng tensor → chỉ đếm 1 lần (SigLIP share weights giữa vision/text encoder)
4. **Phân loại params** thành các nhóm: `vision_encoder`, `text_encoder`, `lora_adapters`, `simclr_mlp`, `logit_scale_bias`, `other`
5. **Ước lượng RAM** cho FP32/FP16/INT8: `RAM ≈ model_size × 1.5` (hệ số 0.5 cho activation overhead)
6. **Kiểm tra tương thích RB3:** so sánh RAM ước lượng với ~4GB available → trả về OK / TIGHT / OOM

#### Bước 2: Export model (`lora_fp16/export.py`)

1. **Load & rebuild model:**
   - Load checkpoint → extract Hydra config từ `hyper_parameters`
   - Khởi tạo `LitTBPS(config)` → setup LoRA nếu có → `load_state_dict(strict=False)`

2. **Merge LoRA (nếu có):**
   - Kiểm tra `isinstance(backbone, PeftModel)`
   - Gọi `merge_and_unload()`: tính `W_merged = W_base + α × (A @ B)` cho mỗi LoRA layer
   - Loại bỏ adapter → model chỉ còn 1 bộ weights, không cần PEFT dependency

3. **Export FP32:** save `model.state_dict()` trực tiếp

4. **Export FP16 với tensor deduplication:**
   ```
   seen = {}  // data_ptr → converted tensor
   for (key, tensor) in state_dict:
       ptr = tensor.data_ptr()
       if ptr in seen:
           fp16_state[key] = seen[ptr]    // reuse, không convert lại
       else:
           converted = tensor.half() if floating_point else tensor  // giữ nguyên int
           seen[ptr] = converted
           fp16_state[key] = converted
   torch.save(fp16_state)  // nhận diện object identity → deduplicate trong file
   ```
   - `data_ptr()` trả về địa chỉ bộ nhớ thực của data buffer → phát hiện shared weights
   - Khi `torch.save()` gặp 2 entry trỏ cùng object → chỉ lưu data 1 lần → giảm ~40% file size

5. **Save config.yaml:** resolve tất cả Hydra `${...}` interpolations, custom representer cho tuple → list

#### Bước 2b: Export ONNX (`onnx/export.py`)

> Tách riêng khỏi bước 2 vì folder `lora_fp16/` chỉ nên chứa logic LoRA merge + FP16. ONNX là format khác, dependency khác, có thể chạy độc lập.

1. Load `config.yaml` + `model_fp32.pt` (hoặc `model_fp16.pt`) từ thư mục đã export
2. Rebuild model architecture: `build_backbone_with_proper_layer_resize()` → `TBPS()` → `load_state_dict()`
3. Wrap `model.encode_image()` và `model.encode_text()` trong 2 `nn.Module` riêng biệt
4. `torch.onnx.export()` trace qua module với dummy input → sinh ONNX graph
5. Dynamic axes: batch dimension có thể thay đổi lúc inference, opset version 17
6. Mặc định dùng FP32 state dict cho ONNX (ổn định hơn FP16 khi tracing)

#### Bước 3: Test inference (`inference_test.py`)

1. Load exported state_dict với `weights_only=True` (bảo mật — không execute arbitrary code)
2. Rebuild model từ `config.yaml`: `build_backbone_with_proper_layer_resize()` → `TBPS()` → `load_state_dict()`
3. **Benchmark pattern:** 1 warmup iteration (bỏ qua — JIT compilation, cache miss) + N iterations đo thực
4. Đo riêng image encoding và text encoding → mean latency, stddev, throughput (samples/sec)
5. Tính cosine similarity: `L2_normalize(img_feat) @ L2_normalize(txt_feat).T`
6. Track peak memory via `resource.getrusage(RUSAGE_SELF).ru_maxrss`

**Quick reference:**

```bash
# Bước 1: Phân tích
python deployment/scripts/analyze_checkpoint.py --ckpt path/to/checkpoint.ckpt

# Bước 2a: Export FP16
python deployment/scripts/lora_fp16/export.py \
    --ckpt epoch=56-val_score=52.28.ckpt \
    --output-dir artifacts/deployment/exports/msiglip_lora

# Bước 2b: Export ONNX
python deployment/scripts/onnx/export.py \
    --model-dir artifacts/deployment/exports/msiglip_lora \
    --precision fp32

# Bước 3: Test
python deployment/scripts/inference_test.py \
    --model-dir artifacts/deployment/exports/msiglip_lora \
    --dtype fp16
```

**Output sau export:**

```
artifacts/deployment/exports/msiglip_lora/
├── model_fp32.pt          # Full precision        (từ lora_fp16/export.py)
├── model_fp16.pt          # Half precision         (từ lora_fp16/export.py)
├── config.yaml            # Resolved Hydra config  (từ lora_fp16/export.py)
├── vision_onnx/           # Vision encoder ONNX    (từ onnx/export.py)
│   ├── vision_encoder.onnx       # graph (~1.4 MB)
│   └── vision_encoder.onnx.data  # weights (~355 MB)
└── text_onnx/             # Text encoder ONNX      (từ onnx/export.py)
    ├── text_encoder.onnx         # graph (~1.4 MB)
    └── text_encoder.onnx.data    # weights (~1.0 GB)
```

**So sánh kết quả:**


### Suy nghĩ & cách tiếp cận

- **Tại sao merge LoRA thay vì giữ nguyên?** Trên server GPU, overhead LoRA không đáng kể. Nhưng trên ARM CPU (RB3), mỗi matmul thừa đều tốn thời gian. 24 layers × 4 projections = 96 lần compute thừa mỗi forward. Merge LoRA là "free optimization" — không mất accuracy, chỉ cần 1 dòng code (`merge_and_unload()`).
- **Tại sao FP16 mà không INT8?** FP16 là bước an toàn nhất — không cần calibration data, không mất accuracy. INT8 quantization cho tốc độ tốt hơn nhưng cần calibration set và có thể giảm accuracy. Nên làm FP16 trước, đánh giá, rồi mới thử INT8.
- **Tensor deduplication khi save FP16:** SigLIP share weights giữa vision/text encoder (`model.siglip.xxx` = `model.backbone.xxx`). Khi convert FP16, cần track `data_ptr()` để tránh lưu trùng → giảm file size thêm ~40%.

---

## 2. File .pt vs .ckpt — Checkpoint formats trong PyTorch

> **Ngày:** 2026-04  
> **Liên quan:** `deployment/scripts/lora_fp16/export.py`, `lightning_models.py`

### Định nghĩa

- **`.ckpt` (Lightning Checkpoint):** Format lưu trạng thái **đầy đủ** do PyTorch Lightning tạo (`Trainer.save_checkpoint()`). Chứa: model `state_dict`, optimizer states (Adam momentum + variance), `hyper_parameters` (Hydra config gốc), epoch/global_step, lr_scheduler state, callback states.
- **`.pt` (PyTorch native):** Format lưu bất kỳ Python object nào qua `torch.save()`. Trong project, dùng để lưu **chỉ state_dict** (weights only) sau khi đã strip optimizer và merge LoRA.

### Vì sao (WHY)

Lightning checkpoint (~1.4 GB) chứa nhiều thành phần không cần cho inference:
- Optimizer states: ~160 MB (Adam lưu 2 buffer/param: momentum + variance)
- Training metadata: epoch, lr_scheduler, callbacks
- LoRA adapters chưa merge (thừa compute khi inference)

Khi deploy lên RB3 (4 GB RAM), `torch.load()` đọc **toàn bộ** `.ckpt` vào RAM → peak ~3.5 GB → sát giới hạn OOM. File `.pt` chỉ chứa weights → nhẹ hơn ~50%, load nhanh hơn, an toàn hơn trên edge.

### Làm gì (WHAT)

Pipeline export: `.ckpt` → load trên dev machine → strip optimizer → merge LoRA → convert FP16 → save `.pt`

### Làm như thế nào (HOW)

| | `.ckpt` | `.pt` |
|---|---|---|
| Tạo bởi | `Trainer.save_checkpoint()` | `torch.save(state_dict)` |
| Chứa optimizer | Có | Không |
| Chứa config | Có (`hyper_parameters`) | Không (save config.yaml riêng) |
| Size (project) | ~1.4 GB | ~740 MB (FP16) |
| Mục đích | Resume training | Inference only |
| Load | `torch.load(weights_only=False)` | `torch.load(weights_only=True)` |

**Load `.ckpt` (training):**
```python
ckpt = torch.load("epoch=53.ckpt", map_location="cpu", weights_only=False)
config = ckpt["hyper_parameters"]["config"]  # Hydra config
state = ckpt["state_dict"]                   # model weights
# ckpt còn có: optimizer_states, epoch, global_step, lr_schedulers, callbacks
```

**Load `.pt` (inference):**
```python
state = torch.load("model_fp16.pt", map_location="cpu", weights_only=True)
# Chỉ có weights, không có optimizer hay config
model.load_state_dict(state)
```

`weights_only=True` quan trọng cho `.pt`: ngăn `torch.load()` execute arbitrary pickle code → bảo mật hơn khi load model từ nguồn khác.

### Suy nghĩ & cách tiếp cận

- **Tại sao không dùng `.safetensors`?** Safetensors (HuggingFace) an toàn hơn `.pt` (không dùng pickle), nhưng project hiện dùng `torch.save/load` xuyên suốt và cần tensor deduplication qua `data_ptr()` — safetensors chưa hỗ trợ shared tensors tốt. Có thể migrate sau.
- **`weights_only` flag:** PyTorch >= 2.0 mặc định warn nếu `weights_only=False`. Với `.ckpt` phải dùng `False` vì chứa custom objects (OmegaConf config). Với `.pt` chỉ chứa tensors → dùng `True` an toàn.

---

## 3. Activations — Bộ nhớ trung gian khi inference

> **Ngày:** 2026-04  
> **Liên quan:** `deployment/scripts/analyze_checkpoint.py`, `artifacts/deployment/logs/analyze_20260414_164509.log`

### Định nghĩa

- **Activations (kích hoạt):** Các tensor trung gian được tạo ra tại output của mỗi layer trong quá trình forward pass. Ví dụ: input image đi qua patch embedding → tensor (256, 768), qua transformer layer 1 → tensor (256, 768), v.v. Tất cả các tensor này phải tồn tại đồng thời trong RAM vì layer tiếp theo cần output của layer trước.
- **Model weights:** Bộ trọng số cố định của model, được load 1 lần từ file vào RAM.
- **Peak RAM:** Lượng RAM tối đa cần tại bất kỳ thời điểm nào = model weights + activations + overhead nhỏ (Python objects, framework buffers).

### Vì sao (WHY)

Khi ước lượng RAM cho inference trên RB3 (4 GB), chỉ tính model weights là **không đủ**. Ví dụ model FP32 nặng 1437 MB nhưng thực tế cần ~2155 MB vì activations chiếm thêm ~718 MB. Nếu chỉ tính weights → tưởng còn dư 2.5 GB → thực tế chỉ còn 1.8 GB → có thể gây OOM nếu batch size > 1 hoặc OS dùng nhiều RAM.

### Làm gì (WHAT)

Công thức ước lượng trong `analyze_checkpoint.py`:

```
Total RAM ≈ model_size × 1.5
           = model_weights + activations
           = model_weights + model_weights × 0.5
```

Hệ số 0.5 là ước lượng bảo thủ (conservative) cho single-sample inference. Thực tế phụ thuộc vào:
- **Batch size:** N samples → activations ×N
- **Sequence length / số patches:** ảnh 256×256, patch 16 → 256 patches → 256 tokens
- **Hidden dimension:** 768 (SigLIP base)
- **Số layers:** mỗi layer giữ output riêng

### Làm như thế nào (HOW)

Kết quả phân tích checkpoint `epoch=56-val_score=52.28.ckpt`:

| Precision | Weights | Activations (~0.5×) | Tổng | Dư RAM (4 GB) | Đánh giá |
|-----------|---------|---------------------|------|---------------|----------|
| FP32 | 1437 MB | ~718 MB | ~2155 MB | 1845 MB | OK nhưng chật |
| FP16 | 718 MB | ~359 MB | ~1077 MB | 2923 MB | Thoải mái |
| INT8 | 359 MB | ~180 MB | ~539 MB | 3461 MB | Rất thoải mái |

**Lưu ý:** Activations luôn cùng precision với model weights — nếu weights FP16 thì activations cũng FP16, giảm một nửa so với FP32.

### Suy nghĩ & cách tiếp cận

- **Hệ số 0.5 có chính xác không?** Đây là heuristic. Với transformer model, activations thực tế phụ thuộc vào kiến trúc (attention cần lưu QKV + attention scores). Hệ số thực có thể từ 0.3–0.8 tùy model. 0.5 là mức an toàn cho single-sample.
- **Training vs inference:** Khi training, activations lớn hơn nhiều vì cần lưu **tất cả** intermediate tensors cho backward pass (backpropagation). Inference chỉ cần forward → có thể giải phóng activations sớm hơn. Đây là lý do inference luôn cần ít RAM hơn training.
- **FP16 là sweet spot cho RB3:** Giảm cả weights lẫn activations xuống một nửa, không cần calibration data (khác INT8), không mất accuracy. Kết luận: FP16 export là bước đầu tiên an toàn nhất.

---

## 4. Qualcomm AI Engine SDK — ONNX → DLC → DSP/HTP

> **Ngày:** 2026-04  
> **Liên quan:** `deployment/scripts/onnx/export.py`, `deployment/hardware_profiling/snpe_benchmark.py`, `deployment/docs/benchmark-rp.md`

### Định nghĩa

- **SNPE (Snapdragon Neural Processing Engine):** SDK cũ của Qualcomm cho inference trên Snapdragon SoC. Hỗ trợ CPU, GPU (Adreno), DSP (Hexagon). Format model: DLC (Deep Learning Container).
- **QNN (Qualcomm Neural Network):** SDK thế hệ mới thay thế SNPE. Hỗ trợ thêm HTP (Hexagon Tensor Processor) và NPU. Từ v2.40+ Qualcomm gộp SNPE/QNN thành **QAIRT (Qualcomm AI Runtime)**.
- **DLC (Deep Learning Container):** Format model đã được compile tối ưu cho hardware Qualcomm. ONNX graph → DLC = "compile time" cho phần cứng cụ thể.
- **HTP (Hexagon Tensor Processor):** Co-processor chuyên cho tensor operations trên Hexagon DSP. Hỗ trợ INT8/FP16, throughput cao hơn CPU 18-30x.
- **QNN Context Binary:** Format model mới thay thế DLC — pre-compiled QNN graph chạy trực tiếp trên DSP/HTP. Tạo qua Qualcomm AI Hub.
- **Qualcomm AI Hub:** Dịch vụ cloud của Qualcomm, cho phép upload ONNX model và nhận lại model đã compile cho device cụ thể (DLC hoặc QNN context binary). Không cần cài SDK local.

### Vì sao (WHY)

Pipeline hiện tại dừng ở ONNX Runtime trên CPU → latency ~300-500ms/image (ước lượng cho mSigLIP). Benchmark trên proxy models cho thấy:

| Runtime | MobileNetV2 | Speedup vs PyTorch CPU |
|---------|------------|----------------------|
| PyTorch CPU | 92.0 ms | 1x (baseline) |
| ONNX Runtime CPU | 24.7 ms | 3.72x |
| SNPE DSP (ước lượng) | ~3-5 ms | **18-30x** |
| SNPE HTP INT8 (ước lượng) | ~2 ms | **~45x** |

Để đạt real-time trên RB3, **bắt buộc** phải chuyển từ ONNX CPU → DLC/QNN trên DSP/HTP.

### Làm gì (WHAT)

Pipeline đầy đủ 4 bước:
1. **LoRA merge + FP16** (`lora_fp16/export.py`) → `model_fp16.pt` + `config.yaml`
2. **ONNX conversion** (`onnx/export.py`) → `vision_encoder.onnx` + `text_encoder.onnx`
3. **DLC/QNN compile** (Qualcomm AI Hub hoặc SNPE SDK) → `.dlc` hoặc `.bin`
4. **DSP/HTP inference** (`snpe-net-run` hoặc QNN API trên RB3)

Bước 3 cần Qualcomm AI Engine Direct SDK hoặc AI Hub vì conversion tool (`snpe-onnx-to-dlc`) không có sẵn trên RB3 và không hỗ trợ macOS (x86_64 Linux only).

### Làm như thế nào (HOW)

#### Trạng thái RB3 (verified 2026-04-15)

```bash
$ snpe-platform-validator --runtime all --testRuntime
# CPU:  ✅ Passed
# GPU:  ✅ Passed (warning libOpenCL.so nhưng unit test passed)
# DSP:  ✅ Passed (HTP V68 backend initialized, calculator test OK)
# AIP:  ⏭ Skipped (not available on QCS6490)
```

QAIRT version: **2.45.40** (nâng cấp từ 2.40.0). Runtime + CLI tools (`snpe-net-run`, `snpe-throughput-net-run`) đã cài. **Chỉ thiếu conversion tool.**

#### Conversion qua Qualcomm AI Hub (từ Mac M2)

```bash
pip install qai-hub
qai-hub configure --api_token YOUR_TOKEN

# Vision encoder (pass directory, not .onnx file — includes external weights)
qai-hub submit-compile-job \
  --model artifacts/deployment/exports/msiglip_lora/vision_onnx/ \
  --device "Dragonwing RB3 Gen 2 Vision Kit" \
  --compile_options " --target_runtime qnn_context_binary" \
  --name "mSigLIP-vision" \
  --wait

# Text encoder
qai-hub submit-compile-job \
  --model artifacts/deployment/exports/msiglip_lora/text_onnx/ \
  --device "Dragonwing RB3 Gen 2 Vision Kit" \
  --compile_options " --target_runtime qnn_context_binary" \
  --name "mSigLIP-text" \
  --wait
```

**Target runtime options:** `qnn_context_binary` (DSP/HTP, recommended), `qnn_dlc` (legacy DLC), `onnx`, `tflite`, `precompiled_qnn_onnx`.

**Lưu ý:** Cần upload 2 file ONNX riêng (vision + text) vì chúng là 2 model khác nhau với kiến trúc và input shape khác nhau.

#### Conversion qua Full SDK (chỉ trên x86_64 Linux)

```bash
export SNPE_ROOT=/opt/snpe-2.x.x
snpe-onnx-to-dlc --input_network vision_encoder.onnx --output_path vision_encoder.dlc

# Quantize INT8 (cần calibration data)
snpe-dlc-quantize \
  --input_dlc vision_encoder.dlc \
  --input_list calibration_list.txt \
  --output_dlc vision_encoder_int8.dlc \
  --enable_htp
```

#### Inference trên RB3

```bash
# Chạy trên DSP
snpe-net-run --container model.dlc --input_list input_list.txt --use_dsp --perf_profile high_performance

# Hoặc HTP (INT8)
snpe-net-run --container model_int8.dlc --input_list input_list.txt --use_htp
```

### Suy nghĩ & cách tiếp cận

- **Tại sao AI Hub thay vì SDK local?** SNPE SDK chỉ chạy trên x86_64 Linux. Mac M2 (ARM macOS) không support. AI Hub cho phép compile trên cloud từ bất kỳ máy nào — là lựa chọn duy nhất khi dev machine là Mac.
- **`qnn_context_binary` vs `qnn_dlc`?** Context binary là format mới, pre-compiled cho device cụ thể → khởi tạo nhanh hơn, không cần compile on-device. DLC là format legacy cần compile lần đầu chạy. Ưu tiên context binary cho deployment.
- **Vision vs Text encoder riêng biệt:** Tách 2 encoder cho phép: (a) profile riêng từng encoder, (b) quantize khác nhau (vision có thể chịu INT8 tốt hơn text), (c) chạy song song nếu hardware cho phép.
- **FP16 vs INT8 cho ONNX export:** Dùng FP32 state dict khi export ONNX (ổn định hơn khi tracing). Quantization INT8 nên làm ở bước DLC conversion (SNPE/QNN có quantizer chuyên dụng với calibration data).
- **`sys.modules` poisoning bug:** Khi deployment scripts import `utils` (TeeLogger), nó shadow project root's `utils/` package. Fix: đổi tên `deployment/utils.py` → `deployment/deploy_utils.py`. Bài học: tránh đặt tên module deployment trùng với package trong project root.

---

## 5. ONNX — Định dạng trung gian phổ quát cho model AI

> **Ngày:** 2026-04  
> **Liên quan:** `deployment/scripts/onnx/export.py`, `artifacts/deployment/exports/msiglip_lora/`

### Định nghĩa

- **ONNX (Open Neural Network Exchange):** Định dạng file mở, tiêu chuẩn cho biểu diễn model machine learning. Hoạt động như "lingua franca" giữa các framework training (PyTorch, TensorFlow, JAX) và các runtime inference (ONNX Runtime, Qualcomm QNN, TensorRT). Được serialize dưới dạng Protocol Buffer (protobuf).
- **ONNX Graph:** Đồ thị tính toán DAG (Directed Acyclic Graph) gồm các node operator (MatMul, Conv, Softmax, LayerNorm...), input/output shapes, và weight tensors (initializers). File `.onnx` chứa graph này.
- **ONNX External Data Format:** Khi model lớn (>2 GB), protobuf có hard limit về file size. ONNX tách weights ra file `.onnx.data` riêng, file `.onnx` chỉ chứa graph structure + con trỏ (relative path) đến `.data` file. Tools đọc `.onnx` sẽ tự động load `.data` file từ cùng thư mục.
- **Opset Version:** Phiên bản của tập operator ONNX hỗ trợ. Project dùng opset 18. Opset càng mới → hỗ trợ nhiều op hơn, nhưng cần runtime mới hơn.
- **Dynamic Axes:** Cho phép 1 hoặc nhiều dimension của input/output thay đổi lúc runtime (ví dụ batch size). Khai báo khi export để model không bị fix cứng batch=1.

### Vì sao (WHY)

Pipeline deploy cần chuyển model từ PyTorch (dùng để training) sang format mà Qualcomm DSP/HTP hiểu được. PyTorch → Qualcomm trực tiếp **không thể** — Qualcomm không hỗ trợ đọc `.pt` file. ONNX là format trung gian duy nhất được cả 2 bên hỗ trợ:

```
PyTorch (training)  →  ONNX (interchange)  →  QNN context binary (device)
    trainer.py           onnx/export.py          qai-hub compile
```

Ngoài ra, ONNX cho phép dùng ONNX Runtime để test inference trên host machine trước khi deploy lên device — verify kết quả đúng mà không cần hardware thật.

### Làm gì (WHAT)

Export 2 ONNX graph riêng biệt từ model PyTorch đã merge LoRA:
1. `vision_encoder.onnx` — encode image → 768-dim embedding
2. `text_encoder.onnx` — encode text → 768-dim embedding

Tách riêng vì: (a) chạy độc lập khi inference, (b) optimize/quantize riêng, (c) profile riêng.

### Làm như thế nào (HOW)

#### Export process (`onnx/export.py`)

```python
# Wrap từng encoder thành nn.Module riêng
class VisionWrapper(nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(image)

# torch.onnx.export trace qua module với dummy input
torch.onnx.export(
    VisionWrapper(model),
    dummy_image,                          # (1, 3, 256, 256)
    "vision_encoder.onnx",
    input_names=["image"],
    output_names=["image_embedding"],
    dynamic_axes={"image": {0: "batch_size"}},  # batch dimension linh hoạt
    opset_version=18,
)
```

PyTorch 2.x dynamo exporter tự động dùng external data format khi model lớn → sinh 2 file:
- `.onnx` — graph structure (~1.4 MB)
- `.onnx.data` — weight tensors (355 MB vision, 1.0 GB text)

#### File sizes sau export

```
artifacts/deployment/exports/msiglip_lora/
├── vision_encoder.onnx         1.4 MB  (graph)
├── vision_encoder.onnx.data    355 MB  (weights)
├── text_encoder.onnx           1.4 MB  (graph)
├── text_encoder.onnx.data      1.0 GB  (weights)
└── Tổng ONNX:                 ~1.36 GB ≈ model_fp32.pt (1.4 GB)
```

**Tại sao text encoder (1.0 GB) lớn hơn vision (355 MB)?** SigLIP multilingual tokenizer có vocabulary ~250K tokens. Embedding table = 250K × 768 × 4 bytes ≈ 730 MB (FP32). Vision encoder không có lookup table lớn như vậy — chỉ convolutions và attention layers.

#### Lưu ý quan trọng

- `.onnx` và `.onnx.data` **phải nằm cùng thư mục**. Di chuyển `.onnx` mà không mang theo `.data` → lỗi missing data.
- File `.onnx` chứa tên `.data` file dưới dạng relative path — nếu đổi tên `.data` file → lỗi.
- **Qualcomm AI Hub KHÔNG tự upload `.data` file** khi trỏ `--model` vào file `.onnx`. Phải dùng **directory format**: đặt `.onnx` + `.onnx.data` vào 1 thư mục riêng, rồi truyền thư mục: `qai-hub submit-compile-job --model vision_onnx/`. Nếu chỉ truyền file `.onnx` → lỗi "missing external weights".
- Export script (`onnx/export.py`) đã được cập nhật: mỗi encoder lưu vào thư mục riêng (`vision_onnx/`, `text_onnx/`) và log báo tổng size (graph + weights).

### Suy nghĩ & cách tiếp cận

- **Tại sao dùng FP32 cho ONNX export?** FP32 ổn định hơn khi torch.onnx.export trace graph. FP16 có thể gây numerical issue trong một số op (LayerNorm, Softmax). Quantization nên làm ở bước sau (SNPE/QNN quantizer có calibration data → kết quả tốt hơn).
- **External data format là hành vi mặc định**, không phải lỗi. Protobuf giới hạn file size ~2 GB — model FP32 1.4 GB vượt giới hạn nếu nhét tất cả vào 1 file. ONNX tự tách ra để tránh lỗi.
- **Opset 18 vs thấp hơn?** Opset 18 hỗ trợ đầy đủ các op trong SigLIP (attention, LayerNorm, GELU). Opset cũ hơn có thể thiếu một số op → export fail hoặc cần workaround.

---

## 6. Idea C — Unified Noise-Aware Circle Loss (NACIR)

> **Ngày:** 2026-04-14
> **Liên quan:** `model/noise_aware.py`, `model/objectives.py:compute_noise_aware_circle`, `model/tbps.py`, `configs/loss/cir_msiglip.yaml`, `knowledge/noise_ideas_math.md`

### Định nghĩa

- **False Negative (FN):** cặp cùng người nhưng nhãn PID khác nhau → Circle Loss đẩy embedding của chúng xa ra (sai).
- **False Positive (FP):** cặp image–text sai nhãn (ví dụ annotator gán sai caption) nhưng nhãn cho biết đây là cặp dương → Circle Loss kéo chúng lại gần (sai).
- **Bayesian FN detection (từ Idea A / FNM):** với mỗi negative pair có similarity `s`, tính xác suất hậu nghiệm `P(FN|s) = p·f+(s) / [p·f+(s) + (1-p)·f-(s)]` dựa trên 2 Gaussian mô hình hóa phân phối s_pos và s_neg.
- **GMM-based FP detection (từ Idea B / RDE):** fit 2-component Gaussian Mixture lên lịch sử loss của từng sample. Component có mean thấp = "clean", component mean cao = "noisy". Trọng số `w_i = P(clean | loss_i)` dùng để giảm ảnh hưởng của sample nghi bị noisy.
- **EMA (Exponential Moving Average):** trung bình trượt có trọng số mũ, `stat_t = β·stat_{t-1} + (1-β)·batch_stat_t`. Dùng để tracking phân phối similarity và loss của từng sample mà không cần lưu toàn bộ lịch sử.
- **ε floor:** sàn an toàn cho hệ số suppression — `α_n *= max(1-P_fn, ε_n)` đảm bảo gradient không collapse hoàn toàn kể cả khi detector firing mạnh.

### Vì sao (WHY)

Circle Loss hiện tại (`compute_cross_modal_circle`) với `gamma=128` là bộ khuếch đại hard-negative rất mạnh. Vấn đề: hard negatives — những cặp mà model đang "phân vân" — **chính là nơi label noise tập trung**. Trong datasets TBPS:
- Cùng một người xuất hiện trong nhiều camera → nhiều PID khác nhau → model bị ép đẩy embedding của cùng một người ra xa (FN).
- Caption tả sai người / ảnh mờ khiến annotator nhầm → model bị ép kéo embedding của hai người khác nhau lại gần (FP).

Với `gamma=128` nhân vào alpha, mỗi cặp noisy này đóng góp lượng gradient cực lớn, kéo tụt chất lượng retrieval. Soft-label N-ITC hiện tại chỉ "làm mềm" một phần, không phân biệt được pair nào là noisy.

Idea C giải quyết **đồng thời cả hai loại noise trong một hàm loss duy nhất**, tận dụng cấu trúc 2 nhánh (positive/negative) có sẵn của Circle Loss.

### Làm gì (WHAT)

Thay `compute_cross_modal_circle` bằng biến thể noise-aware, inject detection vào cả hai nhánh:

```
L_C = softplus(
    LSE_P[-γ·α̃_p(s − δ_p)]  +  LSE_N[γ·α̃_n(s − δ_n)]
)

α̃_n = α_n · max(1 − P_fn(s),  ε_n=0.1)     ← FN detection (negative branch)
α̃_p = α_p · max(min(w_i,w_j), ε_p=0.2)     ← FP detection (positive branch)
```

Khi cả hai detector đều tắt (P_fn≡0, w≡1), công thức thoái hóa **chính xác** về Circle Loss gốc. Điều này quan trọng cho curriculum: cùng một hàm được gọi xuyên suốt quá trình training; detector chỉ "bật" thông qua argument.

Curriculum 4 giai đoạn (chồng lấn, không disjoint):

| Epoch | NACIR weight | FN detection | FP detection | Ghi chú |
|-------|:-:|:-:|:-:|---|
| 0–5 | 0 | off | off | Warmup; N-ITC + SimCLR driving alignment |
| 6–10 | ramp 0→~0.033 | off | off | NACIR = vanilla Circle; EMA bắt đầu tích lũy |
| 11–14 | ramp ~0.04→0.07 | **on** | off | FN detection kích hoạt (EMA đã ổn định) |
| 15–20 | ramp ~0.08→0.1 | **on** | **on** | Full Idea C, GMM đã có đủ lịch sử loss |
| 21–60 | 0.1 | **on** | **on** | Stable — refit GMM mỗi 5 epoch |

### Làm như thế nào (HOW)

#### Cấu trúc file

- **`model/noise_aware.py` (mới):** class `NoiseAwareCircleState(nn.Module)` quản lý state (EMA stats, per-sample loss buffer, clean weights). Tất cả state là `register_buffer` để checkpoint-safe.
- **`model/objectives.py`:** thêm `compute_noise_aware_circle(image_features, text_features, pids, m, gamma, fn_stats, clean_weights, ε_n, ε_p)` trả về `(loss, diagnostics_dict)`. Helper `_bayesian_fn_prob` tính xác suất hậu nghiệm trong log-space.
- **`model/tbps.py`:** `__init__` nhận thêm `num_train_samples`, tạo `self.noise_state` khi `NACIR=true`. `forward()` thêm block C2 gating detector theo epoch, bảo vệ block CIR cũ bằng `and not nacir_enabled`.
- **`lightning_models.py`:** `__init__` nhận thêm `num_train_samples`. Thêm hook `on_train_epoch_end` để refit GMM định kỳ (mỗi `gmm_refit_interval` epoch kể từ `fp_enable_epoch`).
- **`trainer.py`:** truyền `len(dm.train_set)` khi tạo `LitTBPS`.
- **`configs/loss/cir_msiglip.yaml`:** flag `NACIR: false` + block `nacir_config` chứa tất cả hyperparameter.

#### Bật NACIR trong cấu hình

```yaml
# configs/loss/cir_msiglip.yaml
CIR: true                # giữ nguyên — NACIR tự động tắt CIR branch khi true
NACIR: true              # ← bật Idea C
circle_loss_weight: 0.1
circle_margin: 0.25
circle_gamma: 128

nacir_config:
  ema_beta: 0.99         # momentum EMA cho similarity stats
  fn_prior: 0.01         # prior P(FN) trong Bayesian formula
  epsilon_n: 0.1
  loss_ema_alpha: 0.9    # momentum EMA cho per-sample loss
  epsilon_p: 0.2
  gmm_refit_interval: 5
  gmm_min_separation: 1.0
  fn_enable_epoch: 11
  fp_enable_epoch: 15
```

#### Diagnostics để theo dõi trên W&B

- `nacir_loss` — giá trị loss sau khi scale bởi curriculum weight.
- `nacir_fn_prob_mean` / `nacir_alpha_n_scale_mean` — mức độ suppression nhánh negative. Kỳ vọng: tăng dần từ epoch 11.
- `nacir_clean_weight_mean` / `nacir_alpha_p_scale_mean` — mức độ suppression nhánh positive. Kỳ vọng: giữ gần 1.0 trừ khi GMM phát hiện bimodality rõ.
- `nacir_fn_active`, `nacir_fp_active` — cờ 0/1 cho biết detector nào đang chạy.
- `gmm_separation` — `|μ_clean − μ_noisy| / (σ_clean + σ_noisy)`. Nếu < `gmm_min_separation` thì fallback về uniform weights (sanity check: LoRA có đủ memorization để phân biệt clean/noisy không?).
- `gmm_fallback` — 1.0 khi GMM fail, 0.0 khi OK.

### Suy nghĩ & cách tiếp cận

- **Tại sao dùng `min(w_i, w_j)` cho positive pair thay vì `w_i · w_j`?** Phép nhân làm giảm trọng số quá nhanh (0.7 × 0.7 = 0.49), trong khi min giữ được tín hiệu từ sample ít nghi ngờ hơn (min(0.7, 0.7) = 0.7). Idea Math spec đề xuất chỉ dùng `w_i` nhưng không đối xứng — `min` là lựa chọn trung gian, conservative hơn `mean` và ít aggressive hơn `product`.

- **Tại sao GMM dùng pure PyTorch EM thay vì `sklearn.mixture.GaussianMixture`?** Tránh thêm dependency, đồng thời GMM chạy trên GPU cùng với các buffer khác (không phải chuyển CPU↔GPU mỗi 5 epoch). Với dataset ~12K sample, EM hội tụ trong < 50 iter, không phải bottleneck.

- **Tại sao `stats_initialized` là float buffer thay vì Python bool?** Buffer phải là tensor để `register_buffer` hoạt động. Float 0.0/1.0 rẻ và tương thích với state_dict, trong khi bool tensor có thể gây vấn đề khi cast dtype trong mixed-precision training.

- **Tại sao tách `NACIR` flag thay vì extend `CIR`?** Để A/B test an toàn: `NACIR: false` giữ nguyên hoàn toàn đường đi cũ (đã verified bằng unit test cho diff = 0.00). Khi `NACIR: true`, code tự động bypass block CIR vanilla.

- **Rủi ro với LoRA (~3-5% trainable params):** memorization effect yếu hơn full fine-tuning → GMM có thể không phân biệt được clean/noisy. `gmm_min_separation=1.0` đóng vai trò van an toàn: nếu separation thấp, tất cả `clean_weights` = 1.0 (không suppression). Trường hợp xấu nhất: FP detection vô hiệu hóa, nhưng FN detection vẫn hoạt động và training không bị hỏng.

- **Staleness của EMA stats:** với `ema_beta=0.99` effective window ≈ 100 steps. Tại epoch 11 (sau 5 epoch ramp), EMA đã reflect trạng thái model khá mới. Nếu bật detector sớm hơn, EMA lag sẽ ref phân phối similarity từ khi Circle Loss chưa ramp lên → sai lệch.

- **Kết hợp với MVS (multi-view similarity) augmentation:** NACIR block gọi loss 2 lần (original + aug images), cả hai đều dùng cùng `fn_stats` và `clean_weights` (stats là global, không phụ thuộc view). Cách này mirror pattern MVS của CIR hiện tại và tránh double-update EMA (chỉ update từ primary forward).

- **Curriculum gate bằng `>=` không phải disjoint window:** một khi detector bật thì nó on suốt phần còn lại của training. Statistics chỉ càng ngày càng đáng tin cậy, không có lý do tắt lại.

- **Verification trước training:** notebooks/workspace.ipynb Section 3 + 4 nên test với embeddings đã extract từ checkpoint hiện tại (R@1=52.28%) trước khi chạy full training. Nếu gradient analysis cho thấy NACIR suppress gradient của top-10% hard negatives so với vanilla Circle Loss → dấu hiệu tốt (đang bỏ qua một số FN đáng ngờ thay vì amplify chúng).

---

## 7. Thiết kế hệ thống end-to-end cho mSigLIP trên RB3 Gen2

> **Ngày:** 2026-04  
> **Liên quan:** `deployment/docs/end-to-end-system-design.md`, `deployment/docs/system.md`, `deployment/docs/deployment-plan.md`

### Định nghĩa

- **End-to-end TBPS system:** hệ thống hoàn chỉnh từ camera -> detect người -> crop -> image embedding -> vector search -> Web UI.
- **Edge node:** board RB3 Gen2 đặt gần camera, chịu trách nhiệm ingest video và inference ảnh.
- **Vector database:** nơi lưu embedding để thực hiện truy hồi theo cosine similarity hoặc ANN search.
- **Track-level event:** đơn vị dữ liệu đại diện cho một người được tracker theo dõi trong một khoảng thời gian, thay vì lưu mọi frame như những mẫu độc lập.
- **Outbound-only topology:** board chỉ chủ động gửi dữ liệu ra backend qua HTTPS, không mở public API trực tiếp cho người dùng cuối.

### Vì sao (WHY)

Board RB3 Gen2 chỉ có khoảng 4 GB RAM khả dụng nên không phù hợp để đồng thời:
- chạy detector/tracker
- chạy image encoder liên tục
- chạy thêm text encoder lớn của mSigLIP
- lưu vector DB chính
- host luôn web service public

Nếu cố gom tất cả lên board, hệ thống sẽ dễ OOM, khó scale, khó backup và khó bảo mật. Ngoài ra, text encoder phải nằm trong cùng embedding space với image encoder, nên không thể thay bằng dịch vụ embedding ngoài hệ khác.

### Làm gì (WHAT)

Thiết kế được chốt cho prototype là:
- **Board** chỉ chạy camera ingest, person detect/track, crop selection, image embedding và upload.
- **Text embedding** chạy ngoài board nhưng phải dùng đúng text encoder của cùng checkpoint mSigLIP.
- **Vector DB + metadata DB + storage** đặt ngoài board.
- **Web UI** cũng đặt ngoài board, nói chuyện với backend public thay vì gọi trực tiếp vào board.
- Dữ liệu được lưu theo **track-level snapshots**, không lưu mọi frame.

### Làm như thế nào (HOW)

Kiến trúc khuyến nghị:

```text
Camera
  -> RB3 Gen2
     -> detect + track + crop + image embedding
     -> local spool
     -> ingest API
  -> backend cloud
     -> metadata DB + vector DB + storage
     -> text embedding service
     -> search API
  -> Web UI
```

Nguyên tắc triển khai:
- Dùng queue hữu hạn trên board để chống OOM.
- Nếu upload lỗi, ghi event xuống local disk spool rồi retry.
- Với mỗi `track_id`, chỉ lưu 1-3 snapshot tốt nhất thay vì mọi frame.
- Query text từ Web UI sẽ đi qua backend, backend gọi text service để lấy embedding 768 chiều, rồi search trên vector DB.
- Để tránh một người bị capture quá nhiều ảnh, cần thêm `capture suppression policy`: rate limit trong cùng track, quality gating, và `recent_identity_cache` để gộp các track bị đứt ngắn hạn thành cùng một `episode_id`.
- Với màn hình search mặc định, nên **search trên snapshot nhưng trả kết quả theo `episode_id`**. Đây là lớp dedup ở thời điểm truy vấn để top 10 không bị chiếm bởi nhiều ảnh gần như trùng nhau của cùng một người.
- Với trường hợp một người đứng quá lâu trong khung hình, sau khi đã lưu đủ snapshot tốt nhất thì nên **đóng băng việc ghi embedding mới**, chỉ kéo dài `end_ts`/`last_seen` và heartbeat metadata; chỉ mở lại ghi vector nếu appearance thay đổi đáng kể hoặc sau một khoảng thời gian đủ dài.

Khuyến nghị hạ tầng giai đoạn đầu:
- `Supabase` cho metadata + `pgvector` + storage
- `Vercel Hobby` cho Web UI demo
- `Hugging Face Spaces CPU Basic` cho text embedding service thử nghiệm

### Suy nghĩ & cách tiếp cận

- **Tách image path và text path** là quyết định quan trọng nhất. Board luôn bận với camera stream nên phải ưu tiên image embedding; text query là tải không liên tục và dễ đưa ra ngoài hơn.
- **Không lưu mọi frame** vì vector DB sẽ phình rất nhanh nhưng giá trị truy hồi tăng không tương xứng. Track-based dedup là bắt buộc.
- **Tracking là cần nhưng chưa đủ**. Nguồn duplicate lớn nhất thường là track bị đứt rồi sinh `track_id` mới. Vì vậy cần lớp suppression thứ hai dựa trên thời gian ngắn và độ giống embedding để gộp theo `episode_id`.
- **Search unit và display unit không nên giống nhau.** Search trên snapshot giữ recall tốt hơn, nhưng hiển thị cho user nên collapse theo `episode_id` để UX gọn và hữu ích hơn.
- **Long-dwell là một loại duplication riêng.** Ở đây tracker vẫn đúng, nhưng dữ liệu mới không còn nhiều giá trị. Cách đúng là tiếp tục duy trì sự kiện hiện diện của người đó bằng metadata, chứ không tiếp tục đẩy thêm embedding gần như trùng nhau vào vector DB.
- **Board outbound-only** giúp đơn giản bảo mật hơn nhiều so với việc public board cho user traffic. Tunnel hiện tại chỉ nên giữ vai trò admin/debug.
- **Supabase trước, Qdrant sau** là lộ trình hợp lý: prototype nên tối ưu cho tốc độ triển khai; khi số lượng vector tăng lớn mới tách sang vector DB chuyên dụng.
- Tài liệu chi tiết hơn về topology, schema, API và hosting được ghi tại `deployment/docs/end-to-end-system-design.md`.

## 8. Noisy Correspondence Injection — Tiêm nhiễu cặp ảnh-văn bản

> **Ngày:** 2026-05
> **Liên quan:** `data/bases.py`, `lightning_data.py`, `configs/dataset/*.yaml`, `run_noise_experiments.sh`

### Định nghĩa
- **Noisy Correspondence (Tương ứng nhiễu):** Tình huống trong tập huấn luyện mà cặp (ảnh, văn bản) bị ghép sai — văn bản mô tả người A nhưng lại gán với ảnh người B. Mô hình học sai tín hiệu: coi cặp sai là "positive pair".
- **Noise Injection (Tiêm nhiễu):** Cố tình tạo ra tương ứng nhiễu trong tập huấn luyện sạch để đánh giá độ bền của loss function dưới điều kiện nhiễu.
- **`inject_noisy_correspondence()`:** Hàm hoán đổi caption giữa các mẫu theo tỷ lệ `noisy_rate`. Giữ nguyên `pid` với ảnh, chỉ thay caption → mô hình vẫn coi cặp là positive nhưng nội dung không khớp.

### Vì sao (WHY)
Circle Loss hiện tại đạt R@1=52.28% trên VN3K sạch, nhưng chưa biết nó suy giảm thế nào khi có nhiễu. Cần:
1. Xây dựng đường cong suy giảm (degradation curve) tại noise rates 0.0–0.8
2. Có baseline để so sánh khi phát triển NACIR — không thể đo cải thiện nếu không biết mức suy giảm ban đầu
3. Thiết kế NACIR dựa trên dữ liệu thực: biết nhiễu loại nào (FN hay FP) gây suy giảm nhiều hơn

### Làm gì (WHAT)
Port hàm `inject_noisy_correspondence()` từ dự án RDE-CVPR2024 vào pipeline code/:
- Thêm hàm vào `data/bases.py`
- Thêm `noisy_rate` và `noisy_file` vào config dataset YAML
- Kết nối vào `TBPSDataModule.setup()` — tiêm nhiễu trước khi tạo `ImageTextDataset`
- Tạo script `run_noise_experiments.sh` chạy vòng lặp noise rates 0.0–0.8

### Làm như thế nào (HOW)
**Thuật toán tiêm nhiễu:**
1. Khởi tạo ánh xạ đồng nhất `noisy_inx = [0, 1, ..., N-1]`
2. Chọn ngẫu nhiên `noisy_rate * N` chỉ số → `c_noisy_inx`
3. Xáo trộn `c_noisy_inx` thành `shuffle_noisy_inx`
4. Gán `noisy_inx[c_noisy_inx] = shuffle_noisy_inx` → hoán đổi vị trí caption trong tập nhiễu
5. Với mỗi mẫu `i`: `dataset[i] = (pids[i], image_ids[i], images[i], captions[noisy_inx[i]])`
6. Lưu `noisy_inx` vào `.npy` để tái lập (reproducibility)

**Điểm tích hợp:** Trong `TBPSDataModule.setup()`, sau khối `proportion`, trước `if stage == "fit"`. Dùng `hydra.utils.get_original_cwd()` cho đường dẫn `artifacts/training/noiseindex/` vì Hydra thay đổi CWD sang `outputs/`.

**Lưu ý về PID:** `pid` đi cùng ảnh (không theo caption). Mẫu `(pid_A, img_A, caption_B)` được loss coi là positive pair nhưng nội dung sai → đây chính là noisy correspondence.

### Suy nghĩ & cách tiếp cận
- **Tiêm nhiễu ở `setup()` thay vì `__getitem__`:** Đơn giản, nhất quán với RDE, và cho phép lưu `.npy` tái lập. Nếu tiêm ở `__getitem__`, nhiễu sẽ khác nhau mỗi epoch (không tái lập được).
- **Chạy base Circle Loss trước, NACIR sau:** Cần baseline degradation curve trước khi đo cải thiện của NACIR. Xây NACIR mà không biết nhiễu ảnh hưởng thế nào là "giải bài toán chưa rõ".
- **`RandomIdentitySampler` an toàn:** Sampler nhóm theo `image_id` (vị trí [1] trong tuple), nhiễu chỉ thay caption ở vị trí [3] → sampler không bị ảnh hưởng.

## 9. INT8 Quantization cho HTP — `--preserve_io_datatype` và pipeline compile

> **Ngày:** 2026-05
> **Liên quan:** `deployment/docs/aihub-experiments.md`, `deployment/docs/deployment-plan.md`

### Định nghĩa

- **HTP (Hexagon Tensor Processor):** Bộ xử lý tensor chuyên dụng trên chip Qualcomm (VD: HTP V68 trên QCS6490). Chạy inference nhanh hơn CPU 18-30x, nhưng chỉ hỗ trợ INT8/INT16 ở biên I/O (input/output tensors).
- **INT8 Quantization:** Chuyển weights và activations từ FP32 (32-bit float) sang INT8 (8-bit integer). Giảm 4x dung lượng weights, tăng tốc inference trên HTP.
- **`--preserve_io_datatype`:** Flag của `qairt-converter` và `qairt-quantizer`. Khi có, I/O tensors giữ nguyên kiểu dữ liệu gốc (FP32/FP16) dù nội bộ đã quantize INT8. AI Hub tự inject flag này khi dùng `--quantize_full_type`.
- **Calibration data:** Tập dữ liệu mẫu dùng để xác định min/max range cho mỗi tensor khi quantize. Dummy calibration (`--calibration_data none`) dùng range ngẫu nhiên → accuracy rác, chỉ verify pipeline. Real calibration cần 200-500 mẫu đại diện.
- **QNN Context Binary (`.bin`):** File binary đã compile sẵn cho HTP. Chạy trực tiếp trên device qua `snpe-net-run --container model.bin --use_htp`.

### Vì sao (WHY)

HTP V68 trên QCS6490 **từ chối** bất kỳ float type nào (FP32 lẫn FP16) ở biên I/O. Đây là ràng buộc phần cứng/driver:
- Tensor transfers giữa CPU và DSP đi qua DMA channels tối ưu cho integer blocks.
- HTP instruction set load tensors dưới dạng INT8/INT16 tiles; float phải dequantize on-chip từ INT storage.
- Giữ I/O integer tránh chuyển đổi FP↔INT tốn kém ở tensor boundaries.

Sau 11 lần thử (experiments #1-#11), nguyên nhân thất bại chính là AI Hub **tự động inject** `--preserve_io_datatype` khi dùng `--quantize_full_type int8`, khiến I/O tensors vẫn là FP32 dù nội bộ đã INT8 → HTP từ chối ở bước tạo context binary.

### Làm gì (WHAT)

Pipeline deploy lên HTP gồm 5 bước:

1. **Merge LoRA + FP16 export** (local) → `model_fp16.pt`
2. **Export ONNX** (local) → `vision_onnx/`, `text_onnx/`
3. **INT8 quantize + compile** (AI Hub cloud) → QNN context binary `.bin`
4. **Download & transfer** `.bin` → RB3 device
5. **Benchmark** trên device: `snpe-net-run --container model.bin --use_htp`

Bước 3 hiện đã hoạt động với dummy calibration (job `jgkr7qwn5`). Cần thay bằng real calibration để có accuracy chấp nhận được.

### Làm như thế nào (HOW)

**Lệnh compile thành công (deprecated API, nhưng đang hoạt động):**
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

**Lưu ý quan trọng:**
- `--quantize_full_type` và `--target_runtime qnn_context_binary` đã deprecated. Nên migrate sang `submit_quantize_job` + `submit_compile_and_link_jobs`.
- `--preserve_io_datatype` KHÔNG được xuất hiện trong converter/quantizer commands. Nếu AI Hub tự inject → compile sẽ fail ở bước context binary.
- Dummy calibration (`--calibration_data none`) chỉ verify pipeline, accuracy rác. Production cần real calibration data.

**Pipeline nội bộ AI Hub (thành công, job jgkr7qwn5):**
1. ONNX graph optimization: 618→448 ops, 354.6→352.3 MiB
2. `qairt-converter` → DLC (KHÔNG có `--preserve_io_datatype`)
3. `qairt-quantizer --weights_bitwidth 8 --act_bitwidth 8` → INT8 DLC
4. `qnn-context-binary-generator` → `.bin` cho HTP V68
5. Stats: 353M MACs, 92M params, DDR spill=983KB, fill=983KB

**So sánh với job thất bại (jpyvrrv7p):**
- Converter command có `--preserve_io_datatype image output_0` (xuất hiện 2 lần!)
- Quantizer command cũng có `--preserve_io_datatype image output_0`
- Kết quả: I/O vẫn FP32 → HTP từ chối ở bước context binary

### Suy nghĩ & cách tiếp cận

- **Tại sao `--preserve_io_datatype` tự inject?** AI Hub thiết kế cho nhiều target (GPU, CPU, DSP). Với GPU/CPU, giữ I/O FP giúp tương thích với code inference hiện có. Nhưng với HTP, điều này gây fail. Đây là design decision của AI Hub, không phải bug.
- **Migration sang API mới:** `submit_quantize_job` cho phép kiểm soát chi tiết hơn quá trình quantize, có thể tránh auto-injection. Nên migrate khi làm production INT8.
- **Dummy vs Real calibration:** Dummy chỉ dùng để verify pipeline hoạt động. Real calibration cần ~200-500 ảnh từ VN3K training split, resize 256×256, normalize (0.5, 0.5, 0.5). Target accuracy: R@1 ≥ 48% (vs FP32 baseline 52.28%).
- **Text encoder:** Cần compile riêng với `--input_specs '{"input_ids": ((1, 64), "int64"), "attention_mask": ((1, 64), "int64")}'`. Text embedding table (~730 MB FP32, ~180 MB INT8) cần kiểm tra có vừa HTP memory không qua `submit-profile-job`.

---

## 10. Đánh giá output `notebooks/workspace.ipynb` cho NACIR

> **Ngày:** 2026-05-12
> **Liên quan:** `notebooks/workspace.ipynb`, `model/objectives.py:compute_noise_aware_circle`, `model/noise_aware.py`, `knowledge/noise_ideas_math.md`

### Định nghĩa

- **Notebook validation:** bước kiểm tra nhanh trên embedding đã extract từ checkpoint, dùng để quyết định có nên chạy full training tốn nhiều giờ hay không.
- **Degenerate equivalence:** điều kiện `NACIR(detectors off) == vanilla Circle Loss`. Nếu pass, có thể bật/tắt NACIR bằng config mà không đổi baseline.
- **FN posterior non-flat:** kiểm tra đường `P(FN|s)` có thực sự thay đổi theo similarity không. Nếu gần như hằng số, FN detector chưa phát hiện được pair đáng nghi.
- **GMM fallback:** cơ chế an toàn khi GMM clean/noisy separation thấp hơn ngưỡng. Khi fallback, `clean_weights = 1.0`, tức FP detection tạm thời không tác động.

### Vì sao (WHY)

Training mSigLIP tốn nhiều giờ, nên NACIR phải có tín hiệu tốt trong `notebooks/workspace.ipynb` trước khi chạy full training. Nếu notebook đã báo `STOP`, full training có nguy cơ chỉ tái tạo Circle Loss gốc hoặc thêm độ phức tạp mà không tạo lợi ích noise-aware thật sự.

### Làm gì (WHAT)

Output hiện tại **chưa đủ greenlight để train NACIR**:

- Checkpoint load đúng và retrieval tốt: T2I R@1 = 53.35, R@5 = 80.03, R@10 = 88.30, mAP = 58.17, mINP = 51.45.
- NACIR rollback an toàn: vanilla Circle = 213.834335 và NACIR-off = 213.834335, diff = 0.00e+00.
- FN detection chưa khả quan: `P(FN|s)` gần như phẳng, range chỉ `[0.001, 0.011]`, mean ≈ 0.0100.
- Batch dùng để fit FN Gaussian có phân phối pos/neg gần như chồng nhau: `mu_pos=-0.0768`, `mu_neg=-0.0668`, separation = -0.049.
- Gradient hard-negative hầu như không đổi: mean hard neg từ 0.034868 xuống 0.034670, ratio hard/easy từ 3131.3x xuống 2964.4x. Đây là suppression rất nhẹ, chưa chứng minh NACIR đang né FN thật.
- FP GMM có fallback an toàn: separation = 0.843 < 1.0, fallback = 1.0. Điều này không làm training hỏng, nhưng nghĩa là FP detector gần như inactive.

### Làm như thế nào (HOW)

Các bước cần kiểm tra lại trước full training:

1. Batch validation trong notebook đã được sửa để dùng PID đúng cho hai phía. Cell 14 hiện chọn `common_pids`, dựng batch identity-balanced, lấy `idx_img` và `idx_txt` theo cùng PID, rồi assert `torch.equal(image_batch_pids, text_batch_pids)` trước khi tạo `img_f`, `txt_f`, `pids`.
2. Dùng thống kê global hoặc EMA/queue thay vì chỉ một batch. Trên full test set, pos/neg có gap tốt hơn (`mean_p=0.2081`, `mean_n=-0.0795`, gap=0.2876), trong khi batch NACIR lại cho `mu_pos < mu_neg`, dấu hiệu batch hoặc pairing không đại diện.
3. Sweep `fn_prior` và/hoặc kiểm tra threshold bằng đồ thị `P(FN|s)`; kỳ vọng posterior phải tăng rõ ở vùng high-sim negatives.
4. Chạy Section 8 mini fine-tune sau khi FN detector có posterior non-flat. Hiện Section 8 chưa có output nên chưa có bằng chứng loss giữ R@1 ổn định sau cập nhật model.

### Suy nghĩ & cách tiếp cận

- **Tín hiệu tốt:** implementation path an toàn vì NACIR-off bằng Circle Loss tuyệt đối và không có gradient collapse (`NACIR/vanilla ≈ 0.994`).
- **Tín hiệu chưa tốt:** detector chính của NACIR chưa tạo quyết định noise-aware có ý nghĩa. `P(FN|s)` gần như bằng prior 0.01 trên toàn range, nên nhánh negative chỉ bị scale khoảng 0.99, gần như vanilla Circle.
- **Không nên train full ngay:** nếu train bây giờ, khả năng cao kết quả chỉ đo lại Circle Loss với overhead nhỏ, chưa kiểm nghiệm được hypothesis NACIR.
- **Ưu tiên tiếp theo:** làm cho FN posterior phản ánh đúng overlap full-set/queue trước, sau đó mới đánh giá GMM FP. Trong Idea C, FN branch là lợi ích trực tiếp nhất đối với Circle Loss; FP branch có fallback là chấp nhận được ở giai đoạn đầu.

---

## 11. Dataset có thể không có FN/FP không?

> **Ngày:** 2026-05-12
> **Liên quan:** `notebooks/workspace.ipynb`, `knowledge/noise_handling_analysis.md`, `knowledge/noise_ideas_math.md`, `data/vn3k_vi.py`

### Định nghĩa

- **Không có FN/FP thật:** annotation và PID trong dataset gần như đúng hoàn toàn; các cặp cùng PID là match thật, khác PID là non-match thật.
- **Không phát hiện được FN/FP:** detector hiện tại không đủ tín hiệu thống kê để tách noise, dù noise có thể vẫn tồn tại.
- **Semantic false negative:** hai người khác PID nhưng caption quá giống nhau hoặc ngoại hình rất giống nhau. Đây không nhất thiết là lỗi annotation, nhưng đối với retrieval/loss thì vẫn tạo hard negative nguy hiểm.
- **Annotation false positive:** ảnh và caption bị ghép sai hoặc caption mô tả không đúng người trong ảnh.

### Vì sao (WHY)

Nếu dataset thật sự rất sạch, NACIR sẽ khó tạo lợi ích trên split sạch vì detector không có gì để sửa. Khi đó kết quả tốt nhất của NACIR là suy biến về Circle Loss gốc nhờ các fallback. Tuy nhiên, nếu kết luận "không có noise" quá sớm từ một notebook batch sai hoặc thống kê yếu, ta có thể bỏ qua một hướng cải thiện quan trọng cho dữ liệu nhiễu và dữ liệu thực tế.

### Làm gì (WHAT)

Có thể dataset có rất ít FN/FP, nhưng output hiện tại **chưa đủ để kết luận là không có FN/FP**:

- GMM fallback với separation = 0.843 có thể nghĩa là ít FP thật, nhưng cũng có thể do LoRA không tạo memorization gap đủ rõ.
- FN posterior phẳng có thể nghĩa là ít FN thật, nhưng output cũ trong notebook được tạo trước khi sửa batch validation, nên thống kê `mu_pos < mu_neg` cần được chạy lại trước khi kết luận.
- Full-set similarity lại cho tín hiệu alignment tốt (`mean_p=0.2081`, `mean_n=-0.0795`, gap=0.2876), nên phải dùng thống kê đúng trước khi kết luận.
- Dataset benchmark thường được curate sạch hơn dữ liệu thực tế; absence of evidence trong benchmark không chứng minh NACIR vô dụng cho surveillance/noisy deployment.

### Làm như thế nào (HOW)

Các kiểm tra nên làm để phân biệt "dataset sạch" và "detector yếu":

1. **Manual audit top suspicious pairs:** lấy top high-sim negative pairs theo `text_pid != image_pid`, xem ảnh/caption có thực sự là cùng người hoặc mô tả quá tương đồng không.
2. **Audit high-loss positives:** lấy positive pairs có per-sample loss cao nhất hoặc positive similarity thấp nhất, xem caption có sai người/sai thuộc tính không.
3. **Noise injection control:** chạy baseline Circle và NACIR trên dữ liệu được tiêm FP noise có kiểm soát. Nếu NACIR không phản ứng cả khi có synthetic noise, vấn đề nằm ở detector/implementation.
4. **Use correct paired batch:** cell 14 đã dựng batch từ PID chung giữa image/text và assert alignment. Sau mỗi lần sửa logic batch, cần rerun từ Section 3 đến Section 4/NACIR để bỏ output cũ.
5. **Queue/global stats:** dùng nhiều batch hoặc queue để fit Gaussian, tránh kết luận từ một batch có ít positives.

### Suy nghĩ & cách tiếp cận

- **Khả năng dataset sạch là có thật:** VnPersonSearch/VN3K có thể được annotate khá kỹ, nên FP annotation nặng có thể hiếm.
- **FN vẫn có thể tồn tại dưới dạng semantic ambiguity:** ngay cả khi PID đúng, nhiều người mặc đồ giống nhau hoặc caption chung chung vẫn tạo false-negative-like hard negatives cho Circle Loss.
- **Nếu sạch, NACIR nên là no-op an toàn:** thiết kế đúng là khi không có noise, `P(FN|s)` thấp, `clean_weights=1`, và loss gần như Circle Loss.
- **Cách chứng minh mạnh nhất:** không dựa vào cảm giác từ một checkpoint sạch, mà dùng synthetic noise và manual audit. Nếu detector chỉ bật khi synthetic noise xuất hiện và im lặng trên clean split, đó là tín hiệu tốt chứ không phải thất bại.

---

## 12. Chủ động tạo nhiễu có giúp model cải thiện không?

> **Ngày:** 2026-05-12
> **Liên quan:** `data/bases.py:inject_noisy_correspondence`, `run_noise_experiments.sh`, `model/objectives.py`, `model/noise_aware.py`

### Định nghĩa

- **Noise injection để đánh giá:** cố tình làm bẩn training set theo tỷ lệ biết trước để đo robustness/degradation curve. Mục tiêu chính là kiểm thử loss/detector.
- **Noise injection để huấn luyện:** đưa dữ liệu nhiễu vào quá trình train với kỳ vọng model tổng quát tốt hơn.
- **Label noise augmentation:** augmentation ở tầng nhãn/cặp correspondence. Khác với image/text augmentation thông thường vì nó có thể tạo gradient sai hướng.
- **Known-noise mask:** vì ta tự tạo noise nên biết mẫu nào bị nhiễu; thông tin này có thể dùng để calibrate/evaluate detector hoặc down-weight trực tiếp.

### Vì sao (WHY)

Không phải mọi loại nhiễu đều là augmentation tốt. Nhiễu ảnh/text như crop, color jitter, random erasing, paraphrase có thể giúp model robust hơn vì vẫn giữ semantic label. Nhưng nhiễu correspondence kiểu ghép caption người B vào ảnh người A tạo **positive label sai**. Nếu train bằng loss thường, model sẽ bị ép học sai: kéo embedding của ảnh A và caption B lại gần.

### Làm gì (WHAT)

Chủ động tạo nhiễu **có thể giúp**, nhưng chỉ trong một số điều kiện:

- **Có ích nhất như stress test:** dùng noise rates 0.0-0.8 để vẽ đường suy giảm của baseline Circle/N-ITC và so với NACIR.
- **Có thể cải thiện robustness nếu loss noise-aware:** NACIR/RDE-style loss phải nhận ra hoặc down-weight cặp nhiễu; khi đó synthetic noise tạo môi trường để detector học/được kiểm chứng.
- **Không nên kỳ vọng cải thiện clean R@1 nếu train loss thường trên noisy labels:** với vanilla Circle Loss/N-ITC, FP noise thường làm giảm retrieval vì gradient positive bị sai hướng.
- **Có thể dùng nhỏ như regularization, nhưng rủi ro cao:** nếu noise rate rất thấp và curriculum nhẹ, nó có thể giống label smoothing ở mức hệ thống. Tuy nhiên với `gamma=128`, Circle Loss khuếch đại hard/noisy pairs nên rủi ro lớn hơn lợi ích.
- **Trên dữ liệu thật có nhiễu, mục tiêu là tăng robustness chứ không nhất thiết tăng clean benchmark:** nếu train/eval đều phản ánh môi trường thật có annotation noise, occlusion, caption thiếu/chung chung, noise-aware training có thể cải thiện hiệu năng thực tế. Nếu eval là VN3K sạch, thêm correspondence noise thường không làm R@1 sạch tăng.

### Làm như thế nào (HOW)

Thiết kế experiment hợp lý:

1. **Baseline degradation:** train Circle Loss hiện tại với `noisy_rate = 0.0, 0.1, 0.2, ...` để biết mô hình suy giảm thế nào.
2. **NACIR robustness:** train NACIR trên cùng noise rates; kỳ vọng NACIR thua ít hơn baseline khi noise tăng.
3. **Clean-set sanity:** so sánh `noisy_rate=0.0` giữa baseline và NACIR. Nếu NACIR làm tụt clean R@1 rõ rệt, detector quá aggressive.
4. **Known-noise audit:** vì synthetic noise có index `.npy`, đo riêng `clean_weight`/`P(FN)` trên mẫu bị nhiễu và mẫu sạch. Detector tốt phải phân biệt được hai nhóm.
5. **Không dùng synthetic noise như mục tiêu cuối:** nếu mục tiêu benchmark sạch VN3K, synthetic noise chủ yếu là công cụ kiểm chứng. Nếu mục tiêu deployment thực tế có annotation noise, khi đó training với noise-aware loss trên noisy distribution mới có ý nghĩa.
6. **Real-data validation matrix:** chạy 4 cấu hình tối thiểu: clean train→clean eval, noisy train→clean eval, clean train→noisy/real eval, noisy train+NACIR→noisy/real eval. Chỉ gọi là "cải thiện" nếu cấu hình cuối tốt hơn baseline trên eval phản ánh môi trường thật, đồng thời không tụt quá nhiều trên clean eval.

### Suy nghĩ & cách tiếp cận

- **Nhiễu đúng tầng mới giúp:** image/text augmentation giữ nhãn đúng nên thường giúp. Correspondence noise phá nhãn nên chỉ giúp khi loss biết nghi ngờ nhãn.
- **Với Circle Loss, noise nguy hiểm hơn bình thường:** hard-mining sẽ dồn gradient vào đúng các cặp đang mâu thuẫn với model; đó cũng là nơi synthetic/noisy labels nằm.
- **Best-case của synthetic noise:** không phải làm model clean tốt hơn ngay, mà chứng minh NACIR giảm tốc độ suy giảm khi data bẩn.
- **Chiến lược khuyến nghị:** trước mắt dùng noise injection như benchmark robustness, không dùng để tăng clean performance. Khi NACIR detector đã pass trên synthetic noise, mới cân nhắc train với noise nhẹ để mô phỏng dữ liệu thực tế.
- **Không có free lunch:** nếu synthetic noise khác phân phối noise thật, model có thể học tính bất biến sai và giảm hiệu năng. Noise rate nên bắt đầu thấp (`0.1-0.2`) và luôn có clean-set sanity check.

---

## 13. Notebook-only controlled validation cho NACIR

> **Ngày:** 2026-05-12
> **Liên quan:** `notebooks/workspace.ipynb`, `model/objectives.py:compute_noise_aware_circle`, `model/noise_aware.py`

### Định nghĩa

- **Controlled validation:** kiểm thử loss bằng các tình huống nhiễu do ta chủ động tạo và biết ground-truth, thay vì chờ dataset thật có FN/FP.
- **Synthetic FN trong notebook:** giữ embedding ảnh/văn bản, nhưng đổi PID label của một phần mẫu cùng người sang PID giả. Các cặp cùng true PID nhưng khác fake PID trở thành known false negatives.
- **Synthetic FP trong notebook:** giữ PID label, nhưng thay text embedding của một phần hàng bằng text khác PID. Mẫu đó trở thành known false positive/noisy correspondence.
- **Greenlight table:** bảng PASS/FAIL tổng hợp điều kiện clean no-op, FN suppression, FP down-weighting, và no-collapse.

### Vì sao (WHY)

Clean VN3K có thể ít noise thật, nên detector im lặng trên clean split không đủ để kết luận NACIR hỏng hay đúng. Cần kiểm thử bằng nhiễu có kiểm soát ngay trong notebook để trả lời câu hỏi: khi có FN/FP thật sự, NACIR có phản ứng đúng hướng không?

### Làm gì (WHAT)

Đã thêm Section 4.6 **NACIR Controlled Noise Validation** vào `notebooks/workspace.ipynb`:

- Clean no-op: so sánh Circle Loss và NACIR khi `fn_stats=None`, `clean_weights=None`.
- Synthetic FN: fake PID split trong batch `32 PIDs x 4 pairs/PID`, sweep `fn_prior=[0.01, 0.05, 0.10]`, đo `P(FN)`, `alpha_n_scale`, và gradient trên known-FN.
- Synthetic FP: corrupt 30% text embeddings bằng different-PID text, fit `NoiseAwareCircleState` GMM trên per-sample loss, đo clean weight của corrupted vs clean rows.
- Greenlight summary: in PASS/FAIL và dừng nếu detector không phản ứng hoặc gradient collapse.

### Làm như thế nào (HOW)

Quy trình chạy notebook:

1. Rerun Section 3 cell 14 để dựng batch aligned và thấy `Image/text PID aligned: True`.
2. Rerun Section 3/4/4.5 để cập nhật loss, gradient, NACIR diagnostics sau khi batch đã sửa.
3. Rerun Section 4.6 controlled validation.
4. Chỉ xem xét mini fine-tune hoặc full training nếu:
   - clean no-op pass (`diff < 1e-4`);
   - known-FN có `P(FN)` cao hơn true negatives;
   - known-FN gradient của NACIR thấp hơn vanilla Circle;
   - synthetic FP GMM không fallback và corrupted rows có clean weight thấp hơn clean rows;
   - tổng negative gradient còn >30% vanilla.

### Suy nghĩ & cách tiếp cận

- **Controlled FN bắt buộc phải có:** dataset-level noise injection hiện tại chủ yếu tạo FP/noisy correspondence, không chứng minh được negative branch của NACIR.
- **FP fallback là tín hiệu cần đọc kỹ:** nếu GMM fallback trên synthetic FP, chưa nên claim FP branch hoạt động; cần điều chỉnh per-sample loss, corruption construction, hoặc GMM threshold.
- **Clean no-op là điều kiện an toàn:** nếu không có noise, NACIR phải gần như Circle Loss. Đây là cách bảo vệ clean benchmark.
- **Notebook là gate trước training:** full training chỉ nên chạy sau khi Section 4.6 pass, vì training tốn giờ và khó debug nguyên nhân nếu detector chưa được kiểm chứng độc lập.

### README documentation

`README.md` đã thêm một phần riêng **Experimental Extension: NACIR (Noise-Aware Circle Loss)**. Phần này chỉ mô tả thuật toán, công thức, protocol kiểm chứng, cách chạy và bảng kết quả pending; không chỉnh hoặc diễn giải lại các kết quả Circle Loss đã báo cáo.

---

## 14. Nếu VN3K sạch thì tăng R@1 bằng hướng nào?

> **Ngày:** 2026-05-12
> **Liên quan:** `docs/EXPERIMENT_SUMMARY.md`, `notebooks/workspace.ipynb`, `data/sampler.py`, `model/objectives.py`

### Định nghĩa

- **Clean benchmark:** tập train/test có rất ít annotation FP/FN, nên lỗi chính không đến từ nhãn sai mà đến từ mô hình chưa phân biệt đủ tốt các người giống nhau.
- **Hard true negative:** người khác PID thật sự nhưng ngoại hình/caption rất giống nhau. Đây là vùng quyết định R@1.
- **Semantic-preserving augmentation:** tăng đa dạng ảnh/văn bản nhưng giữ đúng nhãn, khác với noisy correspondence.
- **Retrieval-side improvement:** cải thiện bước inference/ranking mà không đổi training loss, ví dụ TTA, checkpoint ensemble, query expansion.

### Vì sao (WHY)

Nếu VN3K đã sạch, NACIR chỉ nên là no-op an toàn; nó không tạo thêm tín hiệu học mới. Muốn tăng R@1 cần tăng khả năng phân biệt hard true negatives, cải thiện biểu diễn fine-grained, hoặc giảm variance của checkpoint/inference. Kết quả hiện tại cho thấy LoRA + Curriculum Circle đã khai thác hard-negative mining khá tốt, nhưng batch vẫn nhỏ và mô hình vẫn dùng embedding global, nên còn dư địa ở sampler, queue, local alignment và test-time ensemble.

### Làm gì (WHAT)

Thứ tự ưu tiên để tăng R@1 trên clean VN3K:

1. **Offline hard-negative sampler:** dùng checkpoint hiện tại để tìm các PID dễ nhầm, sau đó tạo batch chứa các PID hard với nhau. Mục tiêu: Circle Loss thấy nhiều hard true negatives hơn thay vì negatives ngẫu nhiên.
2. **Memory queue / cross-batch negatives:** mở rộng negative pool mà không cần tăng batch GPU. Nếu labels sạch, queue giúp hard-negative mining mạnh hơn và ổn định hơn.
3. **Fine-grained local/attribute alignment:** thêm tín hiệu patch/token hoặc attribute-level để phân biệt các chi tiết như màu áo, balo, mũ, quần, giày. Đây là hướng phù hợp khi lỗi R@1 là do người giống nhau.
4. **Semantic-preserving text augmentation:** paraphrase tiếng Việt, synonym, attribute dropout nhẹ, caption normalization. Chỉ dùng augment giữ nghĩa, không dùng ghép sai caption.
5. **LoRA/hyperparameter sweep có kiểm soát:** thử rank cao hơn ở các layer cuối, target thêm MLP cuối, hoặc schedule Circle khác; validate bằng notebook trước vì full FT đã underperform và Circle weight cao từng gây mất ổn định.
6. **Retrieval-time ensemble/TTA:** flip/multi-crop image embedding, checkpoint ensemble 3 seeds, hoặc averaging similarity. Đây thường là cách nhanh nhất để tăng leaderboard R@1 nhưng tăng chi phí inference.

### Làm như thế nào (HOW)

Thực nghiệm nên đi theo 3 mức:

1. **Không train:** thử flip-TTA, multi-crop, seed/checkpoint ensemble trong `notebooks/workspace.ipynb`; nếu R@1 tăng, có thể dùng cho báo cáo offline nhưng cần ghi rõ inference cost.
2. **Train ít thay đổi:** hard-negative sampler hoặc memory queue; giữ loss hiện tại để cô lập tác động của batch composition/negative pool.
3. **Train thay đổi lớn:** local/attribute alignment hoặc loss mới; chỉ làm sau khi phân tích failure cases cho thấy lỗi đến từ chi tiết hình ảnh/ngôn ngữ cụ thể.

Metric cần theo dõi:
- R@1 là chính, nhưng mAP/mINP không được tụt mạnh.
- Failure cases: top-1 sai có phải người rất giống nhau không.
- Gradient analysis: hard true negatives phải nhận nhiều gradient hơn, nhưng không collapse.
- Multi-seed: cải thiện phải vượt std hiện tại khoảng `0.68` R@1, nếu không có thể chỉ là seed noise.

### Suy nghĩ & cách tiếp cận

- **Hướng hứa hẹn nhất:** hard-negative sampler hoặc memory queue, vì nó đánh đúng bottleneck đã biết: batch=24 còn nhỏ so với contrastive learning và Circle Loss cần negative đa dạng.
- **Hướng high-impact nhưng tốn công:** local/attribute alignment. Nếu triển khai tốt, nó có thể tăng R@1 thật thay vì chỉ làm robustness.
- **Hướng nhanh nhất để có số đẹp:** TTA/checkpoint ensemble, nhưng không phải cải thiện thuật toán training và có chi phí deploy.
- **Không ưu tiên synthetic noise để tăng clean R@1:** nếu dataset sạch, thêm correspondence noise thường là regularization sai tầng và dễ làm giảm hiệu năng sạch.

---

## 15. Nhiễu được inject trong repo RDE-mSigLIP-3000VnPersonsearch

> **Ngày:** 2026-05-12
> **Liên quan:** `/Users/phamtunglam/Documents/Projects/mSigLIP/RDE-mSigLIP-3000VnPersonsearch/2024-CVPR-RDE/datasets/bases.py`

### Định nghĩa

Repo RDE inject **noisy correspondence** ở mức cặp ảnh-văn bản. Đây là nhiễu gán nhầm caption cho ảnh, không phải nhiễu pixel ảnh, không phải nhiễu token trong câu, và không đổi PID của ảnh.

Cụ thể, mỗi sample ban đầu có dạng:

```text
(pid_i, image_id_i, image_i, caption_i)
```

Sau khi inject noise, code giữ nguyên `pid_i`, `image_id_i`, `image_i`, nhưng thay caption bằng caption của sample khác:

```text
(pid_i, image_id_i, image_i, caption_{noisy_inx[i]})
```

### Vì sao (WHY)

Mục tiêu là mô phỏng lỗi phổ biến trong text-based person search: ảnh và mô tả bị ghép sai nhưng training vẫn xem cặp đó là positive pair. Loại nhiễu này tạo **false positive/noisy positive correspondence** cho loss contrastive: model bị kéo ảnh của người A lại gần caption của người B.

### Làm gì (WHAT)

Hàm `inject_noisy_correspondence(dataset, noisy_rate, noisy_file)` làm các bước:

1. Tạo `noisy_inx = np.arange(N)` cho toàn bộ train samples.
2. Chọn `int(noisy_rate * N)` index để làm noisy.
3. Shuffle nội bộ các index được chọn.
4. Gán `noisy_inx[c_noisy_inx] = shuffle_noisy_inx`.
5. Với mỗi sample `i`, thay caption bằng `captions[noisy_inx[i]]`.
6. Ghi `real_correspondences[i] = 1` nếu `noisy_inx[i] == i`, ngược lại `0`.

Các file `.npy` trong `artifacts/training/noiseindex/` là mapping index để tái lập cùng một pattern nhiễu. Nếu file tồn tại, repo load lại mapping; nếu chưa có, repo tạo mapping mới và save.

### Làm như thế nào (HOW)

Ví dụ trực quan:

```text
Clean:
  i = 10: (pid=1, image=A, caption="nguoi mac ao do")
  k = 47: (pid=8, image=B, caption="nguoi mac ao xanh")

Noisy sau shuffle:
  i = 10: (pid=1, image=A, caption="nguoi mac ao xanh")
```

Training vẫn xem sample `i=10` là positive của `pid=1`, nên gradient sẽ kéo ảnh A về caption của người khác. Đây là nhánh FP/noisy-positive mà các phương pháp như RDE hoặc NACIR cần giảm trọng số.

### Suy nghĩ & cách tiếp cận

- **Không trực tiếp tạo FN:** injection này không đổi PID để biến true match thành negative. Vì vậy nó không kiểm tra nhánh false-negative của NACIR.
- **Có thể có noisy index nhưng vẫn đúng nghĩa:** nếu caption được shuffle từ sample khác cùng PID hoặc mô tả vô tình vẫn đúng người, `real_correspondences=0` nhưng semantic noise có thể yếu. Code chỉ so sánh index, không kiểm tra PID/caption semantics.
- **`vn3k_attribute_noise_20pct.npy` vẫn được xử lý như index mapping nếu đưa vào `noisy_file`:** trong đường dataset đang dùng, không thấy injector riêng để sửa token/attribute trong caption. Các module attribute-aware là xử lý model/loss, không phải cơ chế inject noise dataset chính.
- **Ý nghĩa cho NACIR:** noise này chủ yếu là kiểm thử FP detector/clean-weight branch. Muốn kiểm thử FN detector phải tạo synthetic FN riêng trong notebook bằng cách đổi label PID hoặc mask quan hệ positive thành negative có kiểm soát.

---

## 16. Cập nhật README theo tiến độ NACIR và deployment

> **Ngày:** 2026-05-12
> **Liên quan:** `README.md`, `run_nacir.sh`, `run_noise_experiments.sh`, `deployment/docs/deployment-plan.md`, `deployment/docs/aihub-experiments.md`

### Định nghĩa

README cần phản ánh đúng ba trạng thái hiện tại:

- **Kết quả chính đã báo cáo:** LoRA + Curriculum Circle Loss là baseline ổn định, không bị thay đổi hoặc diễn giải lại.
- **NACIR:** đã được implement và có script `run_nacir.sh`, nhưng kết quả full training vẫn pending.
- **Deployment:** pipeline export local đã chạy được, vision encoder đã compile INT8 HTP qua Qualcomm AI Hub bằng dummy calibration; text encoder, benchmark RB3 và real calibration vẫn pending.

### Vì sao (WHY)

README là entrypoint cho người chạy lại thí nghiệm. Nếu README vẫn hướng dẫn nhập Hydra command thủ công cho NACIR trong khi repo đã có `run_nacir.sh`, người đọc dễ chạy sai cấu hình hoặc bỏ qua script chuẩn. Tương tự, nếu deployment chỉ ghi chung là "ongoing", nó không thể hiện blocker đã giải quyết: QCS6490 HTP không nhận floating-point I/O và cần INT8 I/O.

### Làm gì (WHAT)

Đã cập nhật README theo các nhóm:

1. Thêm bảng **Current Status Snapshot** cho training, NACIR, noisy correspondence và deployment.
2. Sửa phần **How to Run NACIR** để dùng `./run_nacir.sh` hoặc `bash run_nacir.sh`.
3. Cập nhật **Repository Structure** với `run_nacir.sh`, `run_noise_experiments.sh`, `model/noise_aware.py`, `artifacts/training/noiseindex/`, `reports/`, `changelog/`, và chi tiết deployment scripts.
4. Cập nhật **Training** để liệt kê các script chuẩn: Circle, NACIR, noise sweep, full fine-tuning.
5. Thay phần **Ongoing Work** bằng **Deployment Status** có bảng tiến độ rõ ràng.

### Làm như thế nào (HOW)

Các lệnh chạy chính trong README hiện là:

```bash
./run_cir_loss.sh
./run_nacir.sh
./run_noise_experiments.sh
./run_full_finetune.sh
```

Deployment quick commands:

```bash
python deployment/scripts/lora_fp16/export.py \
    --ckpt epoch=56-val_score=52.28.ckpt \
    --output-dir artifacts/deployment/exports/msiglip_lora

python deployment/scripts/onnx/export.py \
    --model-dir artifacts/deployment/exports/msiglip_lora \
    --precision fp32
```

### Suy nghĩ & cách tiếp cận

- **Không đụng kết quả cũ:** bảng kết quả Circle Loss/VN3K/CUHK/PRW được giữ nguyên vì NACIR chưa có kết quả training đáng tin để thay thế.
- **Tách rõ implemented vs pending:** NACIR có code và script, nhưng kết luận vẫn pending; deployment có vision compile thành công, nhưng end-to-end RB3 chưa hoàn tất.
- **README nên dẫn về nguồn chi tiết:** phần deployment trong README chỉ là status ngắn; chi tiết compile và lỗi AI Hub vẫn thuộc `deployment/docs/deployment-plan.md` và `deployment/docs/aihub-experiments.md`.

---

## 17. Dọn placeholder `my_new_loss` khỏi workspace notebook

> **Ngày:** 2026-05-13
> **Liên quan:** `notebooks/workspace.ipynb`

### Định nghĩa

`my_new_loss` là cell template cũ trong notebook, dùng để người dùng tự viết loss thử nghiệm. Hiện notebook đã chuyển trọng tâm sang NACIR và các loss đã implement sẵn, nên placeholder này không còn là nguồn kiểm chứng chính.

### Vì sao (WHY)

Giữ placeholder trong notebook gây hai rủi ro:

1. Dễ hiểu nhầm loss mẫu triplet-style là một phần của NACIR hoặc kết quả chính.
2. Nếu xóa mỗi cell định nghĩa nhưng không sửa các cell phụ thuộc, Section 4 và Section 8 sẽ lỗi `NameError`.

### Làm gì (WHAT)

Đã dọn toàn bộ đường phụ thuộc placeholder:

- Xóa cell markdown/code **Write Your New Loss Here**.
- Xóa wrapper `_my_new_loss_from_sim`.
- Bỏ `"Your New Loss"` khỏi bảng gradient `loss_fns`.
- Đổi Section 8 mini fine-tune sang dùng `compute_cross_modal_circle(...)` với `m=0.25`, `gamma=128`.
- Cập nhật quick reference để ưu tiên Section 4.6 controlled NACIR validation.

### Làm như thế nào (HOW)

Mini fine-tune hiện dùng loss thật:

```python
loss = compute_cross_modal_circle(
    img_feats,
    txt_feats,
    batch_pids,
    m=MINI_CIRCLE_M,
    gamma=MINI_CIRCLE_GAMMA,
)
```

Notebook được kiểm tra bằng:

```bash
jq empty notebooks/workspace.ipynb
python3 - <<'PY'
import ast, json
from pathlib import Path
nb = json.loads(Path("notebooks/workspace.ipynb").read_text())
for cell in nb["cells"]:
    if cell.get("cell_type") == "code":
        ast.parse("".join(cell.get("source", [])))
PY
```

### Suy nghĩ & cách tiếp cận

- **Không giữ template khi mục tiêu đã rõ:** notebook hiện là lab test cho Circle/NACIR, không phải nơi chứa loss mẫu chung chung.
- **Mini fine-tune nên dùng loss thật:** Cross-Modal Circle là lựa chọn an toàn vì đã có baseline và không cần detector state.
- **NACIR vẫn kiểm bằng Section 4.6 trước:** full training hoặc mini fine-tune với biến thể NACIR chỉ nên làm sau khi clean/FN/FP greenlight pass.

---

## 18. Chạy `vision_encoder.bin` trên RB3 bằng QNN HTP runtime

> **Ngày:** 2026-05-14
> **Liên quan:** `deployment/docs/aihub-experiments.md`, `artifacts/deployment/logs/jgkr7qwn5.log`, `artifacts/deployment/qnn_runs/legacy_root/out_qnn/execution_metadata.yaml`

### Định nghĩa

- **QNN context binary (`.bin`):** Artifact đã được Qualcomm AI Hub compile sẵn cho QNN runtime. Đây không phải SNPE DLC, nên không thể đọc bằng DLC reader của `snpe-net-run`.
- **DLC:** Container model của SNPE. Nếu muốn dùng `snpe-net-run --container model.dlc`, phải có artifact DLC hoặc compile theo target tương thích SNPE/DLC.
- **HTP backend:** Backend tăng tốc Hexagon Tensor Processor của Qualcomm, load qua `libQnnHtp.so`. Trên RB3/QCS6490, đây là đường chạy accelerator/NPU-style cho model INT8.
- **Skel library:** Thư viện phía DSP/HTP được load qua FastRPC. Version skel phải khớp với QNN host runtime; nếu host 2.45 nhưng DSP skel 2.43 thì device creation fail.

### Vì sao (WHY)

Sau khi tải `vision_encoder.bin` từ AI Hub lên RB3, mục tiêu là xác nhận model chạy thật trên phần cứng tăng tốc, không chỉ compile thành công trên cloud. Các lỗi gặp phải cho thấy có ba lớp dễ nhầm:

1. `snpe-net-run` đọc `.bin` như DLC nên báo `Dlc read failure`.
2. `qnn-net-run` hệ thống load nhầm backend library 2.43 trong `/lib` hoặc `/usr/lib`, trong khi tool/context binary là 2.45.
3. Sau khi dùng đúng backend 2.45, HTP vẫn có thể load nhầm skel 2.43 phía DSP nếu `ADSP_LIBRARY_PATH` chưa trỏ tới skel 2.45.

Nếu không tách rõ ba vấn đề này, rất dễ kết luận sai rằng model compile hỏng hoặc NPU không dùng được.

### Làm gì (WHAT)

Đường chạy đúng cho artifact hiện tại là:

- Không dùng `snpe-net-run` cho `vision_encoder.bin`.
- Dùng `qnn-net-run` cùng bộ QAIRT 2.45 trong `/opt/qcom/qairt/2.45.40.260406`.
- Trỏ backend tới `libQnnHtp.so` 2.45.
- Trỏ backend extension tới `libQnnHtpNetRunExtensions.so` 2.45.
- Trỏ `ADSP_LIBRARY_PATH` tới thư mục skel HTP 2.45 để tránh mismatch với skel 2.43.
- Chạy `--retrieve_context vision_encoder.bin`.

Kết quả smoke test đã tạo được:

```text
artifacts/deployment/qnn_runs/legacy_root/out_qnn/Result_0/output_0.raw
artifacts/deployment/qnn_runs/legacy_root/out_qnn/qnn-profiling-data_0.log
artifacts/deployment/qnn_runs/legacy_root/out_qnn/execution_metadata.yaml
```

`execution_metadata.yaml` ghi nhận:

```yaml
inferences_completed: 1
graph_name: graph_mn2_j93e
input_tensors:
  - tensor_name: image
    datatype: QNN_DATATYPE_UFIXED_POINT_8
    dimensions: [1, 3, 256, 256]
output_tensors:
  - tensor_name: output_0
    datatype: QNN_DATATYPE_UFIXED_POINT_8
    dimensions: [1, 768]
```

File `output_0.raw` có kích thước `3072` bytes = `768 * 4`, tức `qnn-net-run` đã ghi output float32 mặc định. Vector đọc được hữu hạn, không NaN/Inf:

```text
len = 768
min = -4.087
max = 2.376
mean = -0.0341
std = 0.459
norm = 12.759
NaN = false
Inf = false
```

Kết quả benchmark 100 inference trong `artifacts/deployment/qnn_runs/vision_bench/`:

```yaml
inferences_completed: 100
graph_name: graph_mn2_j93e
input:
  image: QNN_DATATYPE_UFIXED_POINT_8 [1, 3, 256, 256]
output:
  output_0: QNN_DATATYPE_UFIXED_POINT_8 [1, 768]
```

Thư mục có đủ `Result_0` đến `Result_99`, mỗi `output_0.raw` là `3072` bytes = `768 * 4`. Vì benchmark dùng cùng một dummy input lặp lại, 100 output giống nhau byte-by-byte. Các output vẫn finite:

```text
len = 768
min = -4.087
max = 2.376
mean = -0.0341
std = 0.459
norm = 12.759
NaN = false
Inf = false
all_identical_to_first = true
```

`qnn-profiling-data_0.log` là profiling binary/flatbuffer, không phải text log. Muốn lấy latency chính xác cần parse bằng tool profiling của QNN, ví dụ `qnn-profile-viewer` nếu có trên board hoặc trong QAIRT SDK.

Sau khi parse bằng `qnn-profile-viewer`, file `artifacts/deployment/qnn_runs/legacy_root/out_qnn/log.txt` cho kết quả:

```text
NetRun IPS: 32.3070 inf/sec
Average NetRun latency: 22610 us ~= 22.61 ms
Average QNN execute time: 22373 us ~= 22.37 ms
Average accelerator execute time: 20636 us ~= 20.64 ms
Average accelerator execute excluding wait: 19921 us ~= 19.92 ms
Min NetRun latency: 22001 us ~= 22.00 ms
Max NetRun latency: 25891 us ~= 25.89 ms
HVX threads used: 4
Init/load binary time: 54828 us ~= 54.83 ms
De-init time: 20126 us ~= 20.13 ms
```

Diễn giải: vision encoder INT8 dummy-cal trên HTP V68 đã đạt khoảng `32.3 FPS` end-to-end theo `qnn-net-run`, với compute accelerator khoảng `20.6 ms` mỗi ảnh. Đây là số đo cho dummy input lặp lại, chưa bao gồm preprocessing ảnh thật và chưa phản ánh accuracy.

### Làm như thế nào (HOW)

#### 1. Không chạy `.bin` bằng SNPE

Lệnh sau sai với artifact hiện tại:

```bash
snpe-net-run \
  --container vision_encoder.bin \
  --input_list input_list.txt \
  --use_dsp
```

Lỗi:

```text
Dlc read failure. Failed to initialize the DLC reader
```

Nguyên nhân: `vision_encoder.bin` là QNN context binary, không phải DLC.

#### 2. Dùng QAIRT 2.45 đồng bộ host runtime

```bash
cd ~/sigm/Lam

export QAIRT=/opt/qcom/qairt/2.45.40.260406
export QNN_BIN=$QAIRT/bin/aarch64-ubuntu-gcc9.4
export QNN_LIB=$QAIRT/lib/aarch64-ubuntu-gcc9.4
export LD_LIBRARY_PATH=$QNN_LIB:$LD_LIBRARY_PATH
```

Không dùng các thư viện package cũ:

```text
/lib/libQnnHtp.so
/usr/lib/libQnnHtp.so
```

vì package hệ thống đang là 2.43, trong khi `qnn-net-run` và context binary là 2.45.

#### 3. Cấu hình HTP backend extension

`htp_setting.json`:

```json
{
  "context": {
    "weight_sharing_enabled": false
  },
  "devices": [
    {
      "device_id": 0,
      "dsp_arch": "v68",
      "soc_model": 93
    }
  ],
  "graphs": [
    {
      "graph_names": ["graph_mn2_j93e"],
      "fp16_relaxed_precision": 0,
      "vtcm_mb": 0,
      "O": 3
    }
  ]
}
```

`deployment/config/qnn/htp_config_245.json`:

```bash
cat > deployment/config/qnn/htp_config_245.json <<EOF
{
  "backend_extensions": {
    "shared_library_path": "$QNN_LIB/libQnnHtpNetRunExtensions.so",
    "config_file_path": "htp_setting.json"
  }
}
EOF
```

#### 4. Trỏ skel 2.45 cho DSP/HTP

Nếu log có lỗi:

```text
Skel lib id mismatch: expected (v2.45.40...), detected (v2.43.0...)
```

cần tìm và set thư mục skel 2.45:

```bash
find $QAIRT -iname "*Skel*.so" -o -iname "*skel*.so"
export ADSP_LIBRARY_PATH=<thu_muc_chua_skel_2_45>
```

Ví dụ thường gặp:

```bash
export ADSP_LIBRARY_PATH=$QAIRT/lib/hexagon-v68/unsigned
```

hoặc thư mục signed tương ứng nếu board yêu cầu signed PD.

#### 5. Chạy QNN context binary trên HTP

```bash
rm -rf artifacts/deployment/qnn_runs/vision_single
mkdir -p artifacts/deployment/qnn_runs/vision_single

$QNN_BIN/qnn-net-run \
  --backend $QNN_LIB/libQnnHtp.so \
  --retrieve_context vision_encoder.bin \
  --config_file deployment/config/qnn/htp_config_245.json \
  --input_list input_list.txt \
  --output_dir artifacts/deployment/qnn_runs/vision_single \
  --profiling_level basic \
  --perf_profile high_performance \
  --log_level verbose
```

Input list smoke test:

```text
image -> image_256_float32.raw
```

Model native I/O trong metadata là `QNN_DATATYPE_UFIXED_POINT_8`. Khi không dùng `--use_native_input_files`, `qnn-net-run` đọc input file dạng floating-point và tự xử lý theo tensor metadata. Khi cần kiểm soát đúng INT8/uint8 native input cho benchmark chính xác, dùng thêm `--use_native_input_files` và tạo raw theo encoding native của graph.

#### 6. Tự động hóa test ảnh VN3K thật

Hai script hỗ trợ nằm trong `deployment/scripts/qnn/`:

```bash
# Tạo raw input từ ảnh VN3K thật
python deployment/scripts/qnn/prepare_vn3k_vision_inputs.py \
  --dataset-root VN3K \
  --split test \
  --num-samples 10 \
  --output-dir artifacts/deployment/qnn_inputs/vn3k_test_10

# Sau khi chạy qnn-net-run, tóm tắt output 768-d
python deployment/scripts/qnn/summarize_qnn_outputs.py \
  qnn_results \
  --manifest artifacts/deployment/qnn_inputs/vn3k_test_10/manifest.csv \
  --stats-csv qnn_stats.csv \
  --embeddings-csv qnn_embeddings_l2.csv
```

`prepare_vn3k_vision_inputs.py` dùng đúng preprocessing image path:

```text
RGB -> resize 256x256 bicubic -> ToTensor [0,1] -> Normalize(0.5,0.5) -> NCHW float32 raw
```

Script cũng tạo `manifest.csv`, `input_list.txt`, và `run_qnn_vision.sh` để copy sang RB3 rồi chạy trực tiếp.

Lưu ý triển khai:

- Ghi raw tensor bằng `array("f").tofile(file_object)` với file mở `wb`; không truyền trực tiếp `Path` vào `tofile()`.
- Dùng `Image.tobytes()` thay cho `Image.getdata()` để tránh deprecation warning từ Pillow 14, sau đó tách kênh RGB theo thứ tự NCHW.

### Suy nghĩ & cách tiếp cận

- **Phân biệt artifact trước khi chọn runner:** `.bin` từ `qnn_context_binary` phải chạy bằng QNN runtime. `snpe-net-run` chỉ phù hợp nếu có DLC/SNPE container.
- **Version phải đồng bộ cả host và DSP:** Chỉ set `LD_LIBRARY_PATH` chưa đủ. QNN host backend có thể đúng 2.45 nhưng DSP skel vẫn bị lấy từ system 2.43; cần set `ADSP_LIBRARY_PATH`.
- **HTP/NPU trên Qualcomm thường hiện dưới tên DSP/HTP:** `--use_dsp` trong SNPE hoặc `libQnnHtp.so` trong QNN là đường chạy accelerator, không phải CPU fallback.
- **Smoke test chưa phải accuracy test:** Lần chạy này xác nhận runtime chạy được và output đúng shape. Vì compile dùng dummy calibration, bước tiếp theo vẫn phải tạo calibration VN3K thật, benchmark latency nhiều inference, và so sánh embedding/R@1 với FP32 baseline.
- **Cần L2-normalize ngoài model:** `TBPS.encode_image()` trả projection vector chưa normalize. Khi dùng retrieval/cosine similarity, phải normalize output 768-d trước khi so sánh:

```python
emb = emb / np.linalg.norm(emb)
```

---

## 19. Bối cảnh nghiên cứu trước mSigLIP-CLoRA

> **Ngày:** 2026-05  
> **Liên quan:** `knowledge/paper/paper.tex`

### Định nghĩa

- **Text-Based Person Search (TBPS):** Bài toán truy hồi ảnh người đi bộ bằng mô tả ngôn ngữ tự nhiên thay vì ảnh truy vấn.
- **Multilingual TBPS:** Phiên bản TBPS cho nhiều ngôn ngữ, đặc biệt khó khi ngôn ngữ có ít dữ liệu gán nhãn như tiếng Việt.
- **Hard negative:** Cặp ảnh-văn bản không khớp nhưng rất giống nhau trong không gian embedding, ví dụ hai người mặc trang phục tương tự.
- **Sigmoid-based contrastive loss / N-ITC:** Hàm mất mát xem từng cặp ảnh-văn bản như một bài toán nhị phân độc lập, giúp ổn định và tiết kiệm bộ nhớ nhưng không nhấn mạnh đủ các hard negatives.

### Vì sao (WHY)

Các phương pháp TBPS trước đây đã đạt kết quả tốt trên benchmark tiếng Anh hoặc dữ liệu lớn, nhưng khi chuyển sang thiết lập đa ngôn ngữ và ít dữ liệu, mô hình dễ gặp căn chỉnh ảnh-văn bản yếu, thiếu chú thích, nhiễu nhãn và khả năng phân biệt kém giữa các danh tính có mô tả/ngoại hình gần giống nhau.

### Làm gì (WHAT)

Khi trình bày bối cảnh nghiên cứu trước mSigLIP-CLoRA, cần nhấn mạnh ba hạn chế chính: tập trung nhiều vào English/high-resource, mục tiêu contrastive sigmoid của TBPS-mSigLIP thiên về căn chỉnh toàn cục hơn là phân tách hình học chi tiết, và full fine-tuning nhiều tham số dễ tốn tài nguyên hoặc overfit trong dữ liệu ít.

### Làm như thế nào (HOW)

Thông điệp slide nên đi theo mạch:

```text
TBPS truyền thống/CLIP-style -> mạnh trên tiếng Anh và dữ liệu lớn.
Multilingual/low-resource -> thiếu annotation, cross-lingual alignment yếu.
TBPS-mSigLIP/N-ITC -> bền hơn với nhiễu và tiết kiệm bộ nhớ.
Vấn đề còn lại -> không phân biệt đủ hard negatives, embedding chưa compact/separable,
                  full fine-tuning tốn tài nguyên và dễ overfit.
```

Hình minh hoạ phù hợp nhất cho slide là một ví dụ retrieval thất bại: bên trái là câu mô tả tiếng Việt, bên phải là một gallery 3-5 ảnh người, trong đó ảnh đúng được viền xanh nhưng một ảnh hard negative rất giống bị model cũ chọn nhầm và viền đỏ. Dưới hình có thể thêm một inset nhỏ về embedding space: điểm positive và hard negative nằm quá gần nhau trước mSigLIP-CLoRA.

### Suy nghĩ & cách tiếp cận

Bối cảnh nên được viết như một "problem setup" dẫn thẳng vào đóng góp của mSigLIP-CLoRA. Không cần liệt kê quá nhiều tên phương pháp; quan trọng là làm rõ vì sao một baseline đã mạnh như TBPS-mSigLIP vẫn cần thêm Circle Loss và LoRA: Circle Loss xử lý hard-negative discrimination, còn LoRA giúp ổn định và giảm chi phí fine-tuning trong low-resource regime.

---

## 20. Slide hàm loss của mSigLIP-CLoRA

> **Ngày:** 2026-05  
> **Liên quan:** `knowledge/paper/paper.tex`, `model/objectives.py`, `model/tbps.py`

### Định nghĩa

- **Base objective:** Tổ hợp loss của TBPS-mSigLIP để học alignment ảnh-văn bản và regularization.
- **N-ITC:** Loss contrastive sigmoid chính, kéo cặp ảnh-văn bản đúng lại gần và đẩy cặp sai ra xa.
- **MVS:** Multi-View Supervision, giữ nhất quán giữa ảnh gốc và ảnh augment.
- **C-ITC:** Cyclic Image-Text Contrastive, giữ cấu trúc tương đồng trong/cross modality.
- **SS / SimCLR:** Self-supervision cho ảnh, tăng tính bất biến với augmentation.
- **Cross-modal Circle Loss:** Loss phụ tập trung vào hard positives và hard negatives bằng trọng số thích nghi theo độ khó của từng cặp.

### Vì sao (WHY)

4 loss base giúp mô hình có nền alignment ổn định, nhưng N-ITC sigmoid thường cho gradient khá đồng đều giữa negative dễ và negative khó. Trong TBPS, lỗi thường đến từ hard negatives: người khác nhưng mặc đồ, màu sắc, dáng người hoặc bối cảnh rất giống. Vì vậy cần thêm Circle Loss để tập trung gradient vào các cặp khó này.

### Làm gì (WHAT)

Slide nên trình bày công thức tổng:

```text
L = L_base + alpha5(t) * L_circle

L_base = 1.0*N-ITC + 1.0*MVS + 0.1*C-ITC + 0.4*SS
```

Nội dung chính:

```text
Base losses: tạo alignment ổn định và chống overfit/augmentation noise.
Circle Loss: tinh chỉnh hình học embedding.
Mục tiêu: tăng similarity của positive khó, giảm similarity của hard negative.
Điểm quan trọng: cặp càng khó -> trọng số càng lớn -> gradient càng mạnh.
```

### Làm như thế nào (HOW)

Phiên bản công thức nên đưa lên slide:

```text
L_circle = log(1 + sum_N exp(gamma * alpha_n * (s_n - m))
                 * sum_P exp(-gamma * alpha_p * (s_p - (1-m))))

alpha_p = [1 + m - s_p]+
alpha_n = [s_n + m]+
```

Cách giải thích bằng lời:

```text
s_p thấp  -> positive còn xa query -> tăng alpha_p -> kéo lại gần hơn.
s_n cao  -> negative quá giống query -> tăng alpha_n -> đẩy ra xa hơn.
Cặp đã dễ -> trọng số nhỏ -> không lãng phí gradient.
```

### Suy nghĩ & cách tiếp cận

Không nên biến slide loss thành một trang toàn công thức. Cấu trúc dễ hiểu hơn là: một dòng loss tổng ở trên, một khối nhỏ liệt kê 4 loss base, và phần lớn diện tích dành cho trực giác Circle Loss bằng hình embedding hoặc query-positive-hard negative. Với người nghe không chuyên loss, thông điệp cần nhớ là: base loss học alignment tổng quát, Circle Loss sửa phần khó nhất của retrieval bằng cách nhấn mạnh hard negatives.

---

## 21. Đánh giá output mới nhất của `notebooks/workspace.ipynb`

> **Ngày:** 2026-05-16  
> **Liên quan:** `notebooks/workspace.ipynb`, `model/objectives.py`, `model/noise_aware.py`, `run_nacir.sh`

### Định nghĩa

- **Section 4.5 NACIR validation:** kiểm tra nhanh NACIR trên batch thật, gồm clean equivalence, FN posterior, FP GMM fallback và gradient sanity.
- **Section 4.6 controlled validation:** kiểm tra có ground truth synthetic FN/FP, nên đáng tin hơn để quyết định có chạy full NACIR hay chưa.
- **No-collapse criterion:** NACIR phải giữ lại đủ tổng gradient negative, mặc định >30% so với Circle, để tránh detector làm mất tín hiệu học hard negatives.

### Vì sao (WHY)

Notebook mới nhất có hai tín hiệu trái chiều: Section 4.5 báo `GREENLIGHT`, nhưng Section 4.6 báo `STOP`. Khi có mâu thuẫn, phải ưu tiên Section 4.6 vì nó tạo known-FN/known-FP có nhãn kiểm soát và kiểm tra trực tiếp tiêu chí no-collapse.

### Làm gì (WHAT)

Đánh giá ngắn:

- Retrieval checkpoint tốt: T2I `R@1=53.35`, `R@5=80.03`, `mAP=58.17`; I2T `R@1=54.90`.
- Clean no-op pass tuyệt đối: NACIR-off bằng Circle, `diff=0.00e+00`, các scale đều `1.0`.
- FN detector phản ứng đúng hướng: known-FN có `P(FN)=0.7618` cao hơn true negatives `0.0184`, alpha scale known-FN giảm còn `0.2597`.
- FP detector synthetic hoạt động rất rõ: GMM separation `3.954`, corrupted clean weight `0.0000`, clean rows `0.9778`, alpha_p corrupted chạm floor `0.2`.
- Điểm fail chính: synthetic FN suppress quá mạnh, tổng negative gradient chỉ còn `0.075` vanilla, thấp hơn tiêu chí `0.3`.

### Làm như thế nào (HOW)

Quyết định hiện tại:

```text
Không nên launch full NACIR training ngay.
Giữ Circle baseline là loss chính cho đến khi no-collapse pass.
```

Các hướng sửa trước khi training:

1. Đánh giá no-collapse theo production default `fn_prior=0.01` thay vì chọn best prior `0.10`.
2. Nếu vẫn collapse, tăng `epsilon_n` hoặc giảm `fn_prior`.
3. Greenlight table nên báo riêng theo từng prior trong sweep, không chỉ chọn prior có P(FN) cao nhất.
4. Với data sạch, FP GMM fallback ở clean test (`sep=0.843`, fallback=True) là an toàn và không đáng lo; synthetic FP mới là kiểm tra detector thật.

### Suy nghĩ & cách tiếp cận

- **Không bị đánh lừa bởi Section 4.5:** greenlight ở Section 4.5 chỉ là sanity nhanh và cho phép fallback an toàn; nó chưa chứng minh detector hoạt động đúng trên noise có ground truth.
- **FN branch là blocker:** detector tìm đúng known-FN nhưng suppression quá mạnh. Đây là lỗi hyperparameter/safety floor, không phải lỗi ý tưởng NACIR.
- **FP branch khả quan:** synthetic noisy correspondence đúng loại FP mà RDE inject, và detector phân biệt clean/noisy rất tốt.
- **Benchmark vẫn tốt:** output retrieval 53.35 R@1 là tín hiệu checkpoint mạnh, nhưng chưa nên claim thay đổi kết quả canonical nếu chưa chạy cùng script đánh giá chuẩn.

---

## 22. Tune FN branch của NACIR trong notebook

> **Ngày:** 2026-05-16  
> **Liên quan:** `notebooks/workspace.ipynb`, `configs/loss/cir_msiglip.yaml`

### Định nghĩa

- **`fn_prior`:** prior xác suất một labeled-negative thật ra là false negative. Giá trị cao làm detector mạnh hơn nhưng dễ suppress quá nhiều hard negatives thật.
- **`epsilon_n`:** floor cho negative-branch scale `max(1 - P(FN), epsilon_n)`. Giá trị cao hơn bảo toàn gradient nhiều hơn.
- **No-collapse gate:** synthetic FN phải giữ tổng negative gradient >30% vanilla Circle.

### Vì sao (WHY)

Output cũ chọn `fn_prior=0.10` vì nó tối đa hóa chênh lệch `P(FN)` giữa known-FN và true negatives, nhưng lựa chọn này làm tổng negative gradient chỉ còn `0.075` vanilla Circle. Đây là tuning sai mục tiêu: detector nhận diện đúng nhưng quá hung hăng.

### Làm gì (WHAT)

Đã sửa Section 4.6 để sweep cả:

```text
fn_prior: [0.005, 0.01, 0.02, 0.05, 0.10]
epsilon_n: [0.10, 0.20, 0.30, 0.40, 0.50]
```

Tiêu chí chọn mới:

```text
known-FN P(FN) > true-negative P(FN)
known-FN alpha_n scale < true-negative alpha_n scale
known-FN gradient ratio < 0.80
total negative gradient ratio > 0.30
finite loss
```

Trong các candidate pass, notebook ưu tiên cấu hình gần production default `fn_prior=0.01` nhất, rồi chọn `epsilon_n` nhỏ nhất còn pass.

### Làm như thế nào (HOW)

Chạy lại trong notebook:

```text
Section 3  -> dựng aligned batch
Section 4.6 cell helpers
Section 4.6 Clean no-op
Section 4.6 Synthetic FN branch test
Section 4.6 Synthetic FP branch test
Section 4.6 Greenlight table
```

Kết quả cần đọc:

```text
Selected prior
Selected epsilon_n
Known-FN grad ratio
Total neg grad ratio
Selection status
```

Chỉ nên cập nhật `configs/loss/cir_msiglip.yaml` sau khi greenlight pass. Nếu candidate hợp lý là `fn_prior=0.01`, `epsilon_n=0.10`, có thể giữ config hiện tại; nếu cần floor cao hơn thì chỉ đổi `epsilon_n`.

### Suy nghĩ & cách tiếp cận

- **Bảo thủ hơn là đúng:** với dataset sạch, false negative thật có thể ít; detector nên giảm gradient vừa phải chứ không triệt tiêu hard-negative mining.
- **Không tune theo P(FN) cao nhất:** P(FN) cao mà làm mất gradient là tín hiệu xấu.
- **Giữ `fn_prior=0.01` làm neo:** đây là default sản xuất hợp lý cho clean VN3K cho đến khi có bằng chứng dataset có nhiều FN.

---

## 23. RB3-first modular demo cho hệ thống end-to-end

> **Ngày:** 2026-05-16  
> **Liên quan:** `deployment/demo/`, `deployment/docs/end-to-end-system-design.md`, `deployment/docs/deployment-plan.md`

### Định nghĩa

- **RB3-first demo:** demo mà local machine chỉ dùng để kiểm tra wiring/interface, còn tiêu chí deploy thật phải chạy trên board RB3 Gen2.
- **Plugin/adapter boundary:** ranh giới module rõ ràng để thay implementation mà không đổi pipeline chính, ví dụ thay `FakeVisionEncoder` bằng `QnnVisionEncoder`.
- **Local preflight:** kiểm tra nhanh bằng fake/ONNX/local JSONL để bắt lỗi code, không được coi là benchmark deployment.
- **QNN vision adapter:** module chạy `vision_encoder.bin` bằng `qnn-net-run`, đọc output `output_0.raw`, kiểm tra vector 768 chiều finite rồi L2-normalize.

### Vì sao (WHY)

Model vẫn đang trong giai đoạn training/optimization, còn deployment mới có vision encoder QNN dummy-cal chạy được trên RB3. Nếu đợi toàn bộ camera, backend, text service và cloud stack hoàn chỉnh mới bắt đầu demo thì rủi ro tích hợp sẽ dồn về cuối.

Cần dựng sớm một hệ thống module cắm được để:
- chạy được phần nào đã deploy được trên RB3, đặc biệt image encoder;
- thay dần các phần chưa sẵn sàng như live camera, detector thật, Supabase, text service;
- tránh viết demo monolithic khó debug và khó mở rộng;
- phân biệt rõ local preflight với acceptance thật trên board.

### Làm gì (WHAT)

Đã thêm package `deployment/demo/` với cấu trúc:

```text
deployment/demo/
  core/       # dataclass/protocol, utils, pipeline orchestration
  adapters/   # implementation có thể thay thế
  cli/        # implementation của command line
  tests/      # local preflight tests
  run_*.py    # wrapper giữ compatibility cho python -m deployment.demo.*
```

Các boundary chính:

```text
FrameSource -> PersonDetector -> Tracker -> CropSelector
  -> ImageEncoder -> DiskSpool -> Uploader/VectorStore
```

Các adapter ban đầu:
- `ImageDirectorySource`, `VideoFileSource`: đọc ảnh/video file trước khi nối USB/IP camera.
- `FullFramePersonDetector`: dùng cho ảnh crop người kiểu VN3K.
- `SimpleTracker`: tạo `track_id` và `episode_id` deterministic.
- `DefaultCropSelector`: giữ tối đa 3 snapshot mỗi track.
- `QnnVisionEncoder`: adapter thật cho RB3, gọi `qnn-net-run`.
- `OnnxVisionEncoder`, `FakeVisionEncoder`, `FakeTextEncoder`: chỉ phục vụ local preflight.
- `JsonlVectorStore`, `DiskSpool`, `LocalVectorStoreUploader`, `FailingUploader`: kiểm thử search, spool, retry/failure trước khi nối backend thật.

### Làm như thế nào (HOW)

Local preflight:

```bash
python -m compileall deployment/demo
python -m unittest discover deployment/demo/tests
python -m deployment.demo.run_ingest \
  --source /path/to/images \
  --encoder fake \
  --store artifacts/deployment/runtime/vectors.jsonl \
  --spool artifacts/deployment/runtime/spool
python -m deployment.demo.run_search \
  --query "người mặc áo đỏ" \
  --store artifacts/deployment/runtime/vectors.jsonl
```

RB3 acceptance path:

```bash
python -m deployment.demo.run_ingest \
  --source /path/to/images_or_video \
  --encoder qnn \
  --vision-bin artifacts/deployment/qnn_inputs/vision_encoder.bin \
  --htp-config deployment/config/qnn/htp_config_245.json \
  --qairt /opt/qcom/qairt/2.45.40.260406 \
  --board-id qc-rb3g2 \
  --camera-id cam-lab-01
```

`QnnVisionEncoder` dùng đúng preprocessing image path:

```text
RGB -> resize 256x256 bicubic -> ToTensor [0,1] -> Normalize(0.5,0.5) -> NCHW float32 raw
```

Sau đó adapter gọi:

```text
qnn-net-run --backend libQnnHtp.so --retrieve_context vision_encoder.bin ...
```

Output được đọc từ `Result_0/output_0.raw`, yêu cầu:
- đúng 768 phần tử float32;
- không NaN/Inf;
- norm khác 0;
- được L2-normalize trước khi lưu/search.

### Suy nghĩ & cách tiếp cận

- **RB3 là cổng acceptance thật:** mọi local test chỉ giúp bắt lỗi wiring sớm. Không được claim deploy thành công nếu chưa chạy QNN trên board.
- **Tách interface trước cloud backend:** local JSONL/spool đủ để kiểm tra data contract. Khi Supabase/API sẵn sàng, chỉ thay `Uploader` và `VectorStore`.
- **Fake encoder là công cụ kiểm thử, không phải model:** fake embedding giúp test pipeline deterministically nhưng không có ý nghĩa retrieval.
- **QNN artifact phải chạy bằng QNN runtime:** `vision_encoder.bin` là QNN context binary, không phải DLC, nên dùng `qnn-net-run` thay vì `snpe-net-run`.
- **Giữ camera là adapter:** v1 dùng image/video input để debug lặp lại; USB/IP camera sẽ được thêm sau qua `FrameSource` mới.
- **Search/display vẫn theo `episode_id`:** vector search chạy trên snapshot, nhưng kết quả UI nên collapse theo episode để tránh duplicate.

## 24. Tái cấu trúc repo theo layout AI project chuẩn

> **Ngày:** 2026-05-17  
> **Liên quan:** `src/msiglip/`, `configs/`, `artifacts/`, `deployment/config/qnn/`

### Định nghĩa

- **`src/` layout:** Cách tổ chức Python package trong thư mục `src/msiglip/` thay vì để module rải trực tiếp ở repo root.
- **Artifacts:** File sinh ra trong quá trình chạy như checkpoint, ONNX export, QNN raw input/output và runtime logs.
- **Wrapper compatibility:** Root `trainer.py` và `test.py` vẫn tồn tại, nhưng chỉ gọi implementation trong package để command cũ không bị gãy.

### Vì sao (WHY)

Repo trước đó trộn source code, notebook, checkpoint 1.4GB, ONNX export, QNN raw input, QNN output và HTP config ở root. Cách này làm khó đọc cấu trúc, dễ commit nhầm artifact lớn, và khiến deployment script sinh output ở nhiều nơi khác nhau.

### Làm gì (WHAT)

Tái cấu trúc theo hướng vừa phải:
- code chính chuyển vào `src/msiglip/`;
- Hydra config đổi sang `configs/`;
- notebook validation chuyển sang `notebooks/workspace.ipynb`;
- artifacts sinh ra gom về `artifacts/`;
- QNN runtime config nằm tại `deployment/config/qnn/`;
- root wrapper và shell script cũ vẫn chạy.

### Làm như thế nào (HOW)

Các default path mới:

```yaml
paths:
  artifacts_root: artifacts
  data_root: data/raw
  pretrained_root: artifacts/models/pretrained
  checkpoint_root: artifacts/models/checkpoints
  deployment_root: artifacts/deployment
```

Hydra single-run output:

```text
artifacts/training/runs/YYYY-MM-DD/HH-MM-SS
```

Deployment outputs:

```text
artifacts/deployment/exports/
artifacts/deployment/logs/
artifacts/deployment/qnn_inputs/
artifacts/deployment/qnn_runs/
artifacts/deployment/runtime/
```

Các env var có thể override local setup:

```bash
MSIGLIP_DATA_ROOT=/path/to/data
MSIGLIP_PRETRAINED_ROOT=/path/to/pretrained
MSIGLIP_ARTIFACTS_ROOT=/path/to/artifacts
```

Không thêm fallback theo máy local vào `trainer.py`; wrapper Python vẫn chỉ nạp package và Hydra config. Thay vào đó, các shell wrapper training (`run_cir_loss.sh`, `run_nacir.sh`, `run_noise_experiments.sh`, `run_full_finetune.sh`) source `scripts/training_paths.sh`.

Helper này dùng logic portable:
- nếu `MSIGLIP_DATA_ROOT` / `MSIGLIP_PRETRAINED_ROOT` đã được set thì giữ nguyên;
- nếu có layout chuẩn `data/raw/VN3K` và `artifacts/models/pretrained/m_siglip_checkpoints` thì dùng layout chuẩn;
- nếu server workspace còn layout cũ `VN3K/` và `m_siglip_checkpoints/` ở repo root thì dùng repo root làm data/pretrained root.

Vì vậy trên server `/mnt/data/user_data/lampt/PS/code`, nếu đang chạy qua shell wrapper thì chỉ cần:

```bash
cd /mnt/data/user_data/lampt/PS/code
./run_nacir.sh
```

Nếu gọi trực tiếp `trainer.py` hoặc `uv run trainer.py`, truyền env var rõ ràng:

```bash
cd /mnt/data/user_data/lampt/PS/code
MSIGLIP_DATA_ROOT=/mnt/data/user_data/lampt/PS/code \
MSIGLIP_PRETRAINED_ROOT=/mnt/data/user_data/lampt/PS/code \
uv run trainer.py -cn cir_msiglip
```

Nếu server đã chuyển dữ liệu/pretrained về layout chuẩn thì không cần override:

```text
/mnt/data/user_data/lampt/PS/code/data/raw/VN3K
/mnt/data/user_data/lampt/PS/code/artifacts/models/pretrained/m_siglip_checkpoints
```

### Suy nghĩ & cách tiếp cận

- **Không đổi hành vi model/loss:** refactor chỉ đổi layout, imports và path mặc định.
- **Giữ command cũ:** `python trainer.py -cn cir_msiglip` và các `run_*.sh` vẫn là entrypoint quen thuộc.
- **Không ép move dữ liệu thật:** dataset và pretrained checkpoint có thể ở path cũ qua env var; default mới chỉ định nơi chuẩn cho setup mới.
- **Artifacts bị ignore:** file nặng hoặc sinh ra khi chạy không nên sống ở root hoặc trong source tree.
- **Notebook validation cũng phải theo `src` layout:** `notebooks/workspace.ipynb` cần tự thêm `src/` vào `sys.path` và import qua `msiglip.*`; nếu còn `from model...` hoặc `from data...` thì notebook sẽ vỡ sau refactor.
- **Config cũ có relative path:** các `.hydra/config.yaml` sinh trước refactor có thể chứa `dataset_root_dir: .`. Khi notebook chạy từ `notebooks/`, dấu `.` không còn là repo root. Vì vậy notebook phải normalize `dataset_root_dir`, tokenizer path và backbone path theo `PROJECT_ROOT` trước khi tạo `TBPSDataModule`.

## 25. Đánh giá QNN HTP output đầu tiên trên VN3K test 10

> **Ngày:** 2026-05-17  
> **Liên quan:** `artifacts/deployment/qnn_outputs/vn3k_test_10`, `deployment/scripts/qnn/summarize_qnn_outputs.py`, `deployment/config/qnn/htp_config_245.json`

### Định nghĩa

- **QNN context binary:** File `vision_encoder.bin` đã được compile cho QNN/HTP, chạy bằng `qnn-net-run --retrieve_context`, không phải DLC cho SNPE.
- **Result output:** Mỗi input trong `input_list.txt` tạo một thư mục `Result_N/output_0.raw`.
- **Embedding vision:** Output cuối của vision encoder, kỳ vọng 768 chiều. Trong lần chạy này graph metadata là `QNN_DATATYPE_UFIXED_POINT_8`, nhưng `qnn-net-run` ghi output ra file float32 đã dequantize vì không dùng `--use_native_output_files`.

### Vì sao (WHY)

Sau khi QNN HTP chạy được trên board, cần kiểm tra output trước khi claim pipeline deploy ổn:
- số lượng output phải khớp số input;
- mỗi output phải đúng shape 768;
- file không rỗng, không NaN/Inf;
- các ảnh khác nhau không được cho output byte-identical;
- profile/timing phải được đọc đúng từ file profiling có dữ liệu;
- sau đó mới so sánh với PyTorch/ONNX baseline và tính retrieval metric.

### Làm gì (WHAT)

Đánh giá thư mục `artifacts/deployment/qnn_outputs/vn3k_test_10`:
- Có đủ 10 output cho 10 ảnh VN3K test.
- Mỗi `output_0.raw` có 3072 bytes = 768 float32.
- `summary.json` báo `any_nan=false`, `any_inf=false`.
- `all_outputs_byte_identical=false`, nghĩa là QNN không trả cùng một tensor cho mọi input.
- Norm output chưa normalize nằm trong khoảng 11.83 đến 13.10, trung bình 12.47.
- `embeddings_l2.csv` đã chứa 10 embedding L2-normalized để dùng cho bước kiểm tra cosine hoặc retrieval.

### Làm như thế nào (HOW)

Lệnh summarize output:

```bash
python3 deployment/scripts/qnn/summarize_qnn_outputs.py \
  artifacts/deployment/qnn_outputs/vn3k_test_10 \
  --manifest artifacts/deployment/qnn_inputs/vn3k_test_10/manifest.csv \
  --stats-csv artifacts/deployment/qnn_outputs/vn3k_test_10/stats.csv \
  --embeddings-csv artifacts/deployment/qnn_outputs/vn3k_test_10/embeddings_l2.csv \
  --json artifacts/deployment/qnn_outputs/vn3k_test_10/summary.json
```

Kết quả quan trọng:

```text
num_outputs = 10
bytes_per_output_unique = [3072]
expected_dim = 768
any_nan = false
any_inf = false
all_outputs_byte_identical = false
norm_min = 11.8331
norm_max = 13.1043
norm_mean = 12.4697
```

Sanity check cosine trên 10 ảnh:

```text
same_pid_pairs = 5
diff_pid_pairs = 40
same_mean = 0.9120
diff_mean = 0.9018
top_diff_max = 0.9238
```

Cosine cùng PID chỉ cao hơn khác PID một chút và vẫn có negative pair cao hơn positive pair. Đây chưa phải lỗi runtime, vì tập 10 ảnh quá nhỏ và chỉ kiểm tra vision embedding, chưa có text encoder/retrieval. Nó chỉ nói rằng chưa thể kết luận accuracy từ 10 ảnh này.

Profiling cập nhật sau khi chạy đúng `qnn-profile-viewer` trên `qnn-profiling-data_1.log`:

```text
Init / load binary NetRun: 54.269 ms
De-init NetRun: 16.766 ms
Execute NetRun average: 22.252 ms
Execute NetRun min/max: 21.774 / 23.013 ms
Backend QNN execute average: 22.155 ms
Accelerator execute average: 20.723 ms
Accelerator excluding wait average: 19.964 ms
RPC execute average: 21.928 ms
HVX threads used: 4
NetRun IPS: 38.2405 inference/sec
```

File `profile_1.txt` là CSV chi tiết. Các dòng summary text được in ra terminal bởi `qnn-profile-viewer`, còn file output chứa từng event riêng lẻ.

### Suy nghĩ & cách tiếp cận

- **Pass runtime:** QNN HTP đã load context binary, retrieve graph, chạy đủ 10 inference và sinh output đúng kích thước.
- **Chưa pass accuracy:** Cần so sánh QNN output với PyTorch/ONNX baseline trên cùng preprocessing để đo sai số cosine/L2. Sau đó chạy tập VN3K lớn hơn để đo retrieval.
- **Profile đã đọc đúng:** `profile.txt` cũ được tạo từ `qnn-profiling-data_0.log` nên rỗng; `profile_1.txt` mới đọc từ `qnn-profiling-data_1.log` và cho latency thực tế khoảng 22.25 ms/inference.
- **Output cần L2-normalize trước khi search:** norm raw không bằng 1, nên vector store/retrieval phải lưu embedding đã normalize hoặc normalize tại query time.
- **Bước tiếp theo đúng thứ tự:** tạo baseline PyTorch cho cùng VN3K subset bằng `deployment/scripts/qnn/compare_qnn_with_pytorch.py`, so sánh sai số embedding QNN-vs-PyTorch, sau đó chạy benchmark nhiều mẫu hơn và mở rộng sang text encoder hoặc end-to-end retrieval.

Command so sánh QNN với PyTorch baseline trên local:

```bash
python3 deployment/scripts/qnn/compare_qnn_with_pytorch.py \
  --model-dir artifacts/deployment/exports/exported_model \
  --input-dir artifacts/deployment/qnn_inputs/vn3k_test_10 \
  --qnn-output-dir artifacts/deployment/qnn_outputs/vn3k_test_10 \
  --precision fp32 \
  --csv artifacts/deployment/qnn_outputs/vn3k_test_10/qnn_vs_pytorch.csv \
  --json artifacts/deployment/qnn_outputs/vn3k_test_10/qnn_vs_pytorch_summary.json
```

Script này đọc chính raw tensor trong `input_list.txt`, không decode JPEG lại. Vì vậy sai khác đo được là sai khác giữa QNN HTP context binary và PyTorch model export trên cùng input.

Kết quả thực tế với binary dummy-calibration hiện tại:

```text
QNN vs PyTorch/ONNX cosine_l2_mean = 0.1727
cosine_l2_min/max = 0.1440 / 0.2424
l2_l2_mean = 1.2861
```

ONNX Runtime và PyTorch cho cùng raw input khớp với nhau; QNN lệch mạnh. Vì vậy không nên mở rộng benchmark với binary này để đánh giá accuracy. Nguyên nhân phù hợp nhất với log compile là INT8 dùng dummy calibration (`--calibration_data none`), nên quantization range không đại diện cho ảnh VN3K.

Chuẩn bị calibration thật từ VN3K train:

```bash
venv/bin/python deployment/scripts/qnn/prepare_vn3k_vision_inputs.py \
  --dataset-root VN3K \
  --split train \
  --selection random \
  --seed 2400 \
  --num-samples 500 \
  --output-dir artifacts/deployment/qnn_inputs/vn3k_train_calib_500 \
  --path-mode relative
```

Upload calibration dataset bằng Python API vì CLI `qai-hub 0.48.0` không có lệnh `upload-dataset`:

```bash
venv/bin/python deployment/scripts/qnn/upload_qaihub_calibration_dataset.py \
  --input-dir artifacts/deployment/qnn_inputs/vn3k_train_calib_500 \
  --name msiglip-vision-vn3k-train-calib-500
```

Sau khi script in `Dataset ID`, compile lại QNN context binary:

```bash
venv/bin/qai-hub submit-compile-job \
  --model artifacts/deployment/exports/exported_model/vision_onnx/ \
  --device "Dragonwing RB3 Gen 2 Vision Kit" \
  --compile_options " --target_runtime qnn_context_binary --quantize_full_type int8" \
  --input_specs '{"image": ((1, 3, 256, 256), "float32")}' \
  --calibration_data <DATASET_ID> \
  --name "mSigLIP-vision-int8-vn3k-calib-500" \
  --wait
```

Sau khi tải binary mới về, chạy lại cùng `vn3k_test_10`, rồi chạy lại `compare_qnn_with_pytorch.py`. Mục tiêu tối thiểu: cosine QNN-vs-PyTorch phải tăng rất cao so với `0.17`; nếu vẫn thấp thì cần kiểm tra thêm I/O quantization, graph output tensor, hoặc dùng `submit_quantize_job` + `submit_compile_and_link_jobs`.

## 26. Đánh giá output NACIR sau tuning FN sweep

> **Ngày:** 2026-05  
> **Liên quan:** `notebooks/workspace.ipynb`, `src/msiglip/model/objectives.py`, `src/msiglip/model/noise_aware.py`

### Định nghĩa

- **Controlled validation:** kiểm thử loss trên embedding cố định, có synthetic noise được biết trước, để xác nhận detector có phản ứng đúng trước khi chạy full training.
- **No-collapse ratio:** tỉ lệ tổng gradient hard-negative của NACIR so với vanilla Circle. Nếu quá thấp, loss có thể làm yếu tín hiệu học negative quá mức.
- **Synthetic FN:** cặp cùng người thật nhưng bị đổi PID giả để mô phỏng false negative.
- **Synthetic FP:** positive pair bị làm hỏng bằng cách thay text embedding bằng caption khác PID.

### Vì sao (WHY)

Output mới của `notebooks/workspace.ipynb` sau khi thêm sweep `fn_prior` và `epsilon_n` cần được đánh giá lại. Section 4.5 vẫn in `GREENLIGHT`, nhưng gate đáng tin hơn là Section 4.6 `NACIR Controlled Noise Validation`, vì section này kiểm tra clean no-op, synthetic FN, synthetic FP và collapse cùng lúc.

### Làm gì (WHAT)

Kết luận hiện tại: **chưa nên chạy full NACIR training**. NACIR đã tốt hơn output trước ở nhánh FN, nhưng vẫn chưa vượt ngưỡng no-collapse.

Các số chính:

```text
Retrieval sanity: R@1=53.35, R@5=80.03, R@10=88.30, mAP=58.17, mINP=51.45

Clean no-op:
Circle loss = 124.615601
NACIR off   = 124.615601
diff        = 0.00e+00

Synthetic FN best fallback:
selected fn_prior   = 0.100
selected epsilon_n  = 0.50
known-FN P(FN)      = 0.7903
true-neg P(FN)      = 0.0221
known-FN grad ratio = 0.181
total neg grad ratio= 0.258  (< 0.30 threshold)

Synthetic FP:
GMM separation          = 6.473
fallback                = False
clean weight corrupted  = 0.0000
clean weight clean      = 0.9778
alpha_p corrupt / clean = 0.2000 / 0.9910
```

### Làm như thế nào (HOW)

Đọc output trực tiếp từ notebook:

```bash
python3 - <<'PY'
import json
from pathlib import Path

p = Path("notebooks/workspace.ipynb")
nb = json.loads(p.read_text())
for i, cell in enumerate(nb["cells"]):
    text = "".join(
        "".join(out.get("text", "")) if isinstance(out.get("text", ""), list) else out.get("text", "")
        for out in cell.get("outputs", [])
    )
    if "NACIR Controlled Noise Validation Summary" in text:
        print(i)
        print(text)
PY
```

Hướng tune tiếp theo hợp lý hơn là mở rộng sweep `epsilon_n` thay vì hạ ngưỡng no-collapse ngay:

```python
FN_EPS_N_SWEEP = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
```

Nếu `epsilon_n=0.60` hoặc `0.70` đưa `total neg grad ratio` lên trên `0.30` mà `known-FN grad ratio` vẫn thấp hơn vanilla Circle rõ ràng, lúc đó mới coi NACIR đủ an toàn để chạy training thật.

### Suy nghĩ & cách tiếp cận

- Clean no-op pass tuyệt đối (`diff=0.00e+00`), nghĩa là khi tắt detector NACIR không làm lệch Circle Loss.
- FP branch pass rất mạnh: GMM separation cao và clean weight của corrupted rows gần 0. Đây đúng với loại noisy correspondence/caption shuffle mà repo đang inject.
- FN branch đã cải thiện so với output trước (`total neg grad ratio` từ khoảng `0.075` lên `0.258`), nhưng vẫn dưới ngưỡng `0.30`. Đây là dấu hiệu detector đang suppress FN đúng hướng nhưng hơi mạnh tay với tổng negative gradient.
- Không nên dùng Section 4.5 `GREENLIGHT` làm quyết định training. Section 4.6 mới là gate chính và hiện đang `STOP`.
- Các lỗi notebook ở A/B comparison (`CKPT_PATH_B` chưa set, `W_B` chưa có) và Section 8 OOM là lỗi vận hành của section tùy chọn, không phải lỗi logic NACIR. Tuy vậy nên dọn hoặc guard các cell này để notebook chạy sạch hơn.
