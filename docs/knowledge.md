# Knowledge Base

> Cơ sở kiến thức bền vững cho mSigLIP: định nghĩa, khái niệm, cơ chế kỹ thuật, trade-off và các nguyên tắc áp dụng lâu dài.
> File này không phải journal. Không ghi log chạy, metric theo ngày, quyết định tạm thời, reviewer-answer wording, changelog hoặc deployment job result vào đây.

## Quy tắc ghi vào file này

Chỉ thêm mục mới khi nội dung trả lời được câu hỏi: **"khái niệm này có còn đúng sau vài tháng không?"**

Ghi vào nơi khác nếu nội dung là:

- **Kết quả train/tối ưu model:** `docs/journal/[train]-YYYY-MM-DD.md`
- **Kết quả deploy/AI Hub/QNN/RB3:** `deployment/docs/journal/[deploy]-YYYY-MM-DD.md`
- **Reviewer response hoặc wording paper:** `knowledge/response.md` hoặc `knowledge/paper/`
- **Changelog thay đổi code/config/docs:** `changelog/{component}/changelog.md`

Trước khi agent ghi thêm docs/journal/changelog/README, agent phải nêu file đích, nội dung định ghi và lý do, rồi hỏi user xác nhận, trừ khi user đã yêu cầu trực tiếp việc ghi đó.

Không dùng trường ngày trong từng mục knowledge. Nếu cần biết một kết quả xảy ra lúc nào, link sang journal tương ứng.

<!-- TEMPLATE CHO MỤC KIẾN THỨC MỚI

Chỉ dùng template này khi nội dung là kiến thức bền vững, không phải kết quả theo ngày.

## N. [Tên khái niệm / cơ chế]

> **Loại:** concept / mechanism / design-rationale / trade-off
> **Liên quan:** `file/path`, `paper_or_component`

### Định nghĩa

- **Thuật ngữ chính:** Giải thích ngắn gọn, tự đủ nghĩa.
- **Thuật ngữ phụ:** Chỉ thêm nếu cần để hiểu thuật ngữ chính.

### Bối cảnh trong repo

Khái niệm này xuất hiện ở đâu trong mSigLIP, training, deployment hoặc paper.
Không ghi metric/run cụ thể ở đây; nếu cần, link sang journal.

### Cơ chế / nguyên lý

Giải thích cách nó hoạt động ở mức ổn định: công thức, data flow hoặc tương tác giữa các thành phần.

### Khi nào dùng / không dùng

- **Nên dùng khi:** Điều kiện áp dụng hợp lý.
- **Không nên dùng khi:** Giới hạn, phản ví dụ hoặc trường hợp dễ hiểu sai.

### Trade-off và lưu ý

Các đánh đổi kỹ thuật, rủi ro, giả định và điểm cần nhớ khi áp dụng.

### Tham chiếu

- Paper / file / journal liên quan nếu có.
-->

---

## Mục lục

1. [Text-Based Person Search và embedding chung](#1-text-based-person-search-và-embedding-chung)
2. [mSigLIP và LoRA fine-tuning](#2-msiglip-và-lora-fine-tuning)
3. [Checkpoint, state_dict và export artifact](#3-checkpoint-state_dict-và-export-artifact)
4. [ONNX như graph inference trung gian](#4-onnx-như-graph-inference-trung-gian)
5. [Bộ nhớ inference: parameters, activations và precision](#5-bộ-nhớ-inference-parameters-activations-và-precision)
6. [QNN/HTP và INT8 quantization](#6-qnnhtp-và-int8-quantization)
7. [Loss stack trong mSigLIP](#7-loss-stack-trong-msiglip)
8. [Circle Loss và hard-negative mining](#8-circle-loss-và-hard-negative-mining)
9. [False positive, false negative và noisy correspondence](#9-false-positive-false-negative-và-noisy-correspondence)
10. [NACIR - Noise-Aware Circle Loss](#10-nacir---noise-aware-circle-loss)
11. [Noise injection là stress test, không phải augmentation mặc định](#11-noise-injection-là-stress-test-không-phải-augmentation-mặc-định)
12. [Notebook controlled validation như research gate](#12-notebook-controlled-validation-như-research-gate)
13. [Kích thước ảnh 256x256 so với 384x128](#13-kích-thước-ảnh-256x256-so-với-384x128)
14. [PACLIP-TPS và prompting-adapting CLIP](#14-paclip-tps-và-prompting-adapting-clip)
15. [RB3-first deployment và adapter boundary](#15-rb3-first-deployment-và-adapter-boundary)

---

## 1. Text-Based Person Search và embedding chung

> **Loại:** concept
> **Liên quan:** `src/msiglip/model/tbps.py`, `src/msiglip/lightning_models.py`, `src/msiglip/utils/metrics.py`

### Định nghĩa

- **Text-Based Person Search (TBPS):** bài toán truy hồi ảnh người bằng mô tả văn bản, hoặc truy hồi mô tả bằng ảnh người.
- **Cross-modal embedding:** không gian vector chung nơi ảnh và text được biểu diễn sao cho cặp đúng nằm gần nhau.
- **Text-to-image R@1:** tỷ lệ truy vấn text mà ảnh đúng đứng hạng 1 trong danh sách truy hồi. Đây là metric chính của repo, nhưng giá trị cụ thể phải ghi trong journal hoặc experiment summary.

### Bối cảnh trong repo

mSigLIP dùng image encoder và text encoder để đưa ảnh người và câu mô tả tiếng Việt vào cùng embedding dimension. Retrieval được tính bằng similarity giữa hai tập embedding đã normalize.

### Cơ chế / nguyên lý

Quy trình TBPS cơ bản:

1. Encode ảnh thành `image_features`.
2. Encode câu mô tả thành `text_features`.
3. L2-normalize hai vector.
4. Tính similarity bằng dot product hoặc cosine similarity.
5. Sort similarity để lấy ranking cho text-to-image hoặc image-to-text.

### Trade-off và lưu ý

- R@1 nhạy với fine-grained alignment; R@5/R@10 có thể vẫn ổn ngay cả khi R@1 giảm.
- Nếu embedding chưa normalize, dot product bị ảnh hưởng bởi norm vector, làm metric khó diễn giải.
- TBPS khác image classification: model không dự đoán class cố định, mà học không gian tương đồng giữa hai modality.

## 2. mSigLIP và LoRA fine-tuning

> **Loại:** concept / mechanism
> **Liên quan:** `src/msiglip/model/build.py`, `src/msiglip/model/lora.py`, `configs/backbone/m_siglip.yaml`, `configs/lora/default.yaml`

### Định nghĩa

- **mSigLIP:** backbone SigLIP đa ngôn ngữ, dùng image encoder và text encoder để học alignment ảnh-văn bản.
- **LoRA:** phương pháp parameter-efficient fine-tuning, thêm các ma trận rank thấp vào một số linear layer thay vì cập nhật toàn bộ backbone.
- **LoRA merge:** cộng trọng số LoRA vào trọng số base để tạo model inference không còn phụ thuộc adapter runtime.

### Bối cảnh trong repo

Repo fine-tune mSigLIP cho mô tả người tiếng Việt. LoRA giúp giữ phần lớn prior của backbone pretrained, giảm số tham số trainable và thuận lợi hơn cho export/deploy sau training.

### Cơ chế / nguyên lý

Với một linear layer có trọng số base `W`, LoRA học hai ma trận rank thấp `A` và `B`. Output hiệu dụng tương đương:

```text
W_eff = W + scale * B @ A
```

Khi train, chỉ adapter LoRA và một số tham số được chọn cần cập nhật. Khi deploy, có thể merge để inference chỉ còn một bộ weight.

### Trade-off và lưu ý

- LoRA giảm chi phí fine-tune nhưng vẫn phụ thuộc vào lựa chọn target modules, rank và scale.
- Merge LoRA hữu ích cho inference vì loại bỏ dependency PEFT và giảm graph phức tạp.
- Nếu tiếp tục train sau merge, cần cẩn thận vì adapter gốc không còn tách riêng.

## 3. Checkpoint, state_dict và export artifact

> **Loại:** concept / mechanism
> **Liên quan:** `deployment/scripts/lora_fp16/export.py`, `deployment/scripts/analyze_checkpoint.py`, `src/msiglip/lightning_models.py`

### Định nghĩa

- **Lightning checkpoint `.ckpt`:** artifact để resume training, thường chứa model weights, optimizer state, scheduler state, epoch/step và config.
- **PyTorch state_dict `.pt`:** mapping tên tensor sang tensor, phù hợp hơn cho inference nếu chỉ cần model weights.
- **Export artifact:** bộ file dùng cho inference/deploy, thường gồm state_dict đã lọc/merge, config đã resolve và các graph trung gian như ONNX.

### Bối cảnh trong repo

Training tạo checkpoint giàu metadata. Deployment không cần optimizer/scheduler, nên pipeline export tách phần cần thiết cho inference, merge LoRA nếu có và chuẩn bị model cho ONNX/QNN.

### Cơ chế / nguyên lý

Một export inference tốt thường làm các việc:

- Rebuild model architecture từ config.
- Load checkpoint hoặc state_dict.
- Merge LoRA vào base weights nếu model dùng adapter.
- Bỏ optimizer/scheduler/training metadata.
- Chuyển precision nếu cần, ví dụ FP32 sang FP16.
- Lưu config đi kèm để inference tái dựng đúng architecture.

### Trade-off và lưu ý

- `.ckpt` phù hợp cho resume training, không phù hợp làm artifact inference trực tiếp trên edge device.
- `.pt` nhẹ hơn nhưng cần config chính xác để rebuild model.
- Nếu model có shared weights, export nên tránh nhân đôi tensor không cần thiết.
- Tách export thành nhiều bước giúp debug dễ hơn: checkpoint -> state_dict -> ONNX -> runtime-specific artifact.

## 4. ONNX như graph inference trung gian

> **Loại:** concept / mechanism
> **Liên quan:** `deployment/scripts/onnx/export.py`, `deployment/scripts/qnn/compare_onnx_with_pytorch.py`

### Định nghĩa

- **ONNX:** định dạng graph trung gian để trao đổi model giữa PyTorch và các runtime inference khác.
- **Opset:** phiên bản tập toán tử ONNX mà runtime cần hỗ trợ.
- **Dynamic axes:** khai báo chiều có thể thay đổi, thường là batch dimension.

### Bối cảnh trong repo

ONNX là bước trung gian trước khi đi sang runtime tối ưu hơn như QNN. Repo export image encoder và text encoder thành các graph riêng để kiểm tra fidelity từng phần.

### Cơ chế / nguyên lý

`torch.onnx.export()` trace một module với dummy input để tạo graph tĩnh. Với mSigLIP, thường cần wrapper quanh `encode_image()` hoặc `encode_text()` để graph chỉ chứa đường inference cần deploy.

### Trade-off và lưu ý

- ONNX pass không đảm bảo QNN pass; mỗi runtime có tập op và constraint riêng.
- Static ONNX vs PyTorch là control quan trọng. Nếu control fail, lỗi nằm trước quantization/runtime.
- FP32 ONNX thường ổn định hơn khi trace/debug; quantization nên được kiểm tra bằng QDQ hoặc runtime-specific artifact riêng.
- Text encoder có input kiểu token/mask, khác với image encoder có input tensor ảnh; không nên gộp hai đường deploy khi chưa cần.

## 5. Bộ nhớ inference: parameters, activations và precision

> **Loại:** concept
> **Liên quan:** `deployment/scripts/analyze_checkpoint.py`, `deployment/docs/system.md`

### Định nghĩa

- **Parameters:** trọng số model được lưu trong checkpoint/state_dict.
- **Activations:** tensor trung gian sinh ra trong forward pass.
- **Peak memory:** mức RAM cao nhất trong quá trình load hoặc inference, không chỉ bằng file size model.
- **Precision:** kiểu số như FP32, FP16, INT8; ảnh hưởng tới dung lượng, latency và fidelity.

### Bối cảnh trong repo

RB3 Gen2 có RAM hữu hạn, nên deployment phải xét cả size artifact, memory khi load model, activation memory và overhead của runtime.

### Cơ chế / nguyên lý

Memory inference gồm nhiều phần:

```text
peak_memory ~= model_weights + runtime_overhead + activations + input_output_buffers
```

Training checkpoint còn có optimizer state, nên load checkpoint trực tiếp thường tốn RAM hơn nhiều so với load state_dict inference.

### Trade-off và lưu ý

- FP16 giảm memory so với FP32 nhưng không thay thế được INT8 nếu target là accelerator yêu cầu quantized graph.
- INT8 giảm size và có thể tăng tốc, nhưng có rủi ro fidelity nếu calibration hoặc quantization config không phù hợp.
- Batch size làm activation memory tăng nhanh; edge inference nên bắt đầu với batch nhỏ và đo thực tế.

## 6. QNN/HTP và INT8 quantization

> **Loại:** concept / deployment-rationale
> **Liên quan:** `deployment/config/qnn/`, `deployment/scripts/qnn/`, `deployment/docs/journal/`

### Định nghĩa

- **QNN:** Qualcomm AI runtime/toolchain để chạy model trên các backend như CPU/GPU/HTP.
- **HTP:** accelerator Hexagon Tensor Processor trên Qualcomm SoC.
- **PTQ:** post-training quantization, lượng tử hóa model sau training bằng calibration data.
- **QDQ ONNX:** ONNX có cặp QuantizeLinear/DequantizeLinear, dùng để mô phỏng hoặc kiểm tra graph đã quantize.
- **QNN context binary:** artifact đã compile/link cho QNN runtime, khác với ONNX hoặc DLC.

### Bối cảnh trong repo

Deployment mSigLIP đi theo nguyên tắc kiểm tra fidelity theo tầng: PyTorch -> static ONNX -> QDQ ONNX -> QNN runtime. Không nên nhảy thẳng từ ONNX sang board benchmark nếu graph quantized đã sai ở QDQ.

### Cơ chế / nguyên lý

Một flow quantization an toàn:

1. Kiểm tra static ONNX giữ fidelity so với PyTorch.
2. Quantize bằng calibration data đúng preprocessing.
3. So QDQ ONNX với PyTorch trên cùng input.
4. Chỉ compile/link sang QNN khi QDQ đủ tốt.
5. Sau khi QNN chạy được, so QNN output với PyTorch/ONNX trước khi đo retrieval end-to-end.

### Trade-off và lưu ý

- Calibration data phải cùng preprocessing với inference; sai normalize/shape/layout có thể làm quantization hỏng.
- Một số op nhạy như normalization, LayerNorm, projection hoặc attention path có thể cần mixed precision hoặc exclude khỏi INT8.
- HTP runtime có constraint riêng về dtype, op support và layout; pass ở CPU/ONNX không đồng nghĩa pass trên HTP.
- Job ID, fidelity number và kết luận candidate phải nằm trong deployment journal, không nằm ở knowledge.

## 7. Loss stack trong mSigLIP

> **Loại:** mechanism
> **Liên quan:** `configs/loss/cir_msiglip.yaml`, `src/msiglip/model/tbps.py`, `src/msiglip/model/objectives.py`

### Định nghĩa

Loss stack hiện tại kết hợp nhiều tín hiệu:

- **N-ITC:** contrastive alignment ảnh-văn bản.
- **MVS:** augmentation branch dùng trong N-ITC/Circle để tăng tính ổn định.
- **Circle Loss:** hard-positive/hard-negative mining cross-modal.
- **C-ITC:** cyclic consistency giữa similarity nội modal và liên modal.
- **SimCLR:** self-supervised visual consistency giữa hai view ảnh.

### Bối cảnh trong repo

`TBPS.forward()` tính từng loss thành key riêng, Lightning cộng các key kết thúc bằng `loss` để tạo total loss. Config trong `configs/loss/cir_msiglip.yaml` điều khiển bật/tắt và trọng số.

### Cơ chế / nguyên lý

Vai trò của từng loss:

| Loss | Vai trò |
|---|---|
| N-ITC | Tạo alignment chính giữa ảnh và text |
| Circle | Tập trung gradient vào cặp khó |
| C-ITC | Regularize cấu trúc similarity |
| SimCLR | Giữ visual representation ổn định dưới augmentation |

### Trade-off và lưu ý

- Loss stack mạnh hơn một loss đơn lẻ nhưng khó debug hơn khi metric giảm.
- Circle Loss có hard-mining nên rất nhạy với label/correspondence noise.
- Khi thêm loss mới, nên có đường no-op hoặc ablation để chứng minh không phá baseline.
- Config weight là trạng thái repo, còn kết quả tốt/xấu của từng run phải ghi ở journal.

## 8. Circle Loss và hard-negative mining

> **Loại:** concept / mechanism
> **Liên quan:** `src/msiglip/model/objectives.py:compute_cross_modal_circle`, `src/msiglip/model/tbps.py`

### Định nghĩa

**Circle Loss** là loss metric learning đặt trọng số lớn hơn lên các positive chưa đủ gần và negative chưa đủ xa. Trong TBPS, positive thường là ảnh/text cùng PID, negative là khác PID.

### Bối cảnh trong repo

Circle Loss được dùng như loss hard-mining cross-modal. Curriculum trong `tbps.py` cho phép bật trọng số Circle từ thấp lên cao sau giai đoạn warmup.

### Cơ chế / nguyên lý

Circle Loss dùng margin `m` và scale `gamma` để điều chỉnh lực kéo/đẩy:

- Positive similarity thấp hơn ngưỡng mong muốn sẽ nhận gradient kéo gần mạnh hơn.
- Negative similarity cao hơn ngưỡng mong muốn sẽ nhận gradient đẩy xa mạnh hơn.
- Cặp đã dễ thường có trọng số nhỏ hơn.

### Trade-off và lưu ý

- Hard-mining giúp cải thiện fine-grained retrieval khi label đúng.
- Nếu train data có noisy correspondence, hard-mining có thể khuếch đại chính các cặp sai.
- Curriculum giúp giảm sốc đầu training, nhưng không tự giải quyết label noise.
- Circle Loss cần đọc cùng batch composition; sampler ảnh hưởng trực tiếp số positive/negative có trong batch.

## 9. False positive, false negative và noisy correspondence

> **Loại:** concept
> **Liên quan:** `src/msiglip/data/bases.py`, `src/msiglip/model/noise_aware.py`, `knowledge/noise_ideas_math.md`

### Định nghĩa

- **False positive / noisy positive:** cặp được loss xem là positive nhưng thực chất không mô tả cùng người, ví dụ ảnh người A ghép caption người B.
- **False negative:** cặp bị loss xem là negative nhưng thực chất nên gần nhau, ví dụ hai sample khác PID nhưng mô tả/ngoại hình quá giống hoặc annotation bị tách sai.
- **Noisy correspondence:** lỗi ghép ảnh-văn bản ở tầng correspondence, khác với nhiễu pixel ảnh hoặc nhiễu token trong câu.
- **Semantic false negative:** không nhất thiết là lỗi annotation; đây là trường hợp khác PID nhưng quá giống về semantic nên hard-negative mining có thể đẩy quá mạnh.

### Bối cảnh trong repo

VN3K/VnPersonSearch dùng PID để xác định positive/negative. Khi caption bị shuffle hoặc annotation không hoàn hảo, assumption "cùng PID là positive, khác PID là negative" có thể bị phá.

### Cơ chế / nguyên lý

Trong contrastive learning:

- False positive tạo gradient kéo sai hai embedding lại gần.
- False negative tạo gradient đẩy sai hai embedding ra xa.
- Hard-mining làm hai loại lỗi này nguy hiểm hơn vì các cặp mâu thuẫn thường bị xem là cặp khó.

### Trade-off và lưu ý

- Dataset benchmark có thể rất sạch, nhưng dữ liệu thực tế hoặc synthetic stress test vẫn cần noise-aware protocol.
- Detector im lặng trên clean split không đủ để kết luận detector hỏng; có thể đơn giản là không có tín hiệu noise rõ.
- Muốn phân biệt dataset sạch và detector yếu cần audit thủ công, synthetic control hoặc thống kê nhiều batch.

## 10. NACIR - Noise-Aware Circle Loss

> **Loại:** mechanism / research-concept
> **Liên quan:** `src/msiglip/model/objectives.py:compute_noise_aware_circle`, `src/msiglip/model/noise_aware.py`, `configs/loss/cir_msiglip.yaml`, `docs/journal/[train]-2026-05-27.md`

### Định nghĩa

**NACIR** là biến thể noise-aware của Circle Loss, nhằm giảm tác hại của false positive và false negative trong cross-modal retrieval.

NACIR có hai nhánh chính:

- **FN branch:** ước lượng xác suất một negative thật ra là false negative, rồi giảm lực đẩy negative đó.
- **FP branch:** theo dõi per-sample loss để phát hiện positive nghi nhiễu, rồi giảm lực kéo positive đó.

### Bối cảnh trong repo

Khi `loss.NACIR=true`, nhánh Circle thường được thay bằng `compute_noise_aware_circle()`. State liên quan đến noise nằm trong `NoiseAwareCircleState`.

### Cơ chế / nguyên lý

FN branch dùng thống kê similarity EMA để ước lượng phân phối positive/negative và tính posterior kiểu Bayesian cho negative đáng nghi.

FP branch dùng EMA của per-sample Circle loss. Sau khi đủ lịch sử, GMM 1D hai component được fit trên loss EMA:

- component loss thấp được hiểu là clean hơn;
- component loss cao được hiểu là nghi noisy positive hơn;
- nếu separation không đủ, fallback đặt clean weight gần no-op.

Khi detector tắt hoặc chưa active, NACIR phải suy biến gần Circle Loss thường.

### Trade-off và lưu ý

- NACIR chỉ có ý nghĩa nếu noise detector có tín hiệu đủ tốt.
- FP branch phù hợp nhất với caption-shuffle/noisy correspondence vì loại noise này tạo positive sai.
- FN branch khó chứng minh hơn trên dataset sạch; cần synthetic FN hoặc audit có kiểm soát.
- Notebook pass không đảm bảo full training tốt. Kết quả thực nghiệm cụ thể phải xem trong train journal.

## 11. Noise injection là stress test, không phải augmentation mặc định

> **Loại:** design-rationale
> **Liên quan:** `src/msiglip/data/bases.py:inject_noisy_correspondence`, `run_noise_experiments.sh`

### Định nghĩa

**Noise injection** trong repo là thao tác chủ động làm bẩn train correspondence bằng cách tráo caption giữa các sample. Đây là label/correspondence noise, không phải augmentation ảnh/text giữ nguyên semantic label.

### Bối cảnh trong repo

Noise index `.npy` giúp tái lập cùng pattern nhiễu giữa các run. Mục tiêu chính là stress test robustness của loss và detector, không phải mặc định để tăng clean benchmark.

### Cơ chế / nguyên lý

Với sample gốc:

```text
(pid_i, image_i, caption_i)
```

Noisy correspondence giữ `pid_i` và `image_i`, nhưng thay caption:

```text
(pid_i, image_i, caption_j)
```

Nếu `j` thuộc người khác, loss vẫn coi cặp mới là positive dù semantic đã sai.

### Trade-off và lưu ý

- Với vanilla contrastive/Circle Loss, correspondence noise thường làm gradient sai hướng.
- Với loss noise-aware, synthetic noise giúp kiểm tra detector có down-weight đúng sample nhiễu không.
- Synthetic noise không tự đại diện cho noise thật; kết luận robustness cần eval phù hợp.
- Noise rate cao là stress test nặng, không nên hiểu là augmentation an toàn.

## 12. Notebook controlled validation như research gate

> **Loại:** research-protocol
> **Liên quan:** `notebooks/workspace.ipynb`, `src/msiglip/model/objectives.py`, `src/msiglip/model/noise_aware.py`

### Định nghĩa

**Controlled validation** là kiểm tra ý tưởng loss trên embedding đã trích xuất hoặc batch có nhiễu tổng hợp trước khi chạy full training.

### Bối cảnh trong repo

Training tốn nhiều giờ. Notebook dùng embedding frozen để kiểm tra loss scale, gradient direction, synthetic FN/FP và retrieval sanity nhanh hơn nhiều so với full run.

### Cơ chế / nguyên lý

Một gate notebook tốt nên kiểm tra:

- no-op trên clean condition;
- loss value finite và cùng scale với baseline;
- gradient đi vào đúng nhóm hard positives/hard negatives;
- synthetic FP/FN làm detector phản ứng đúng hướng;
- không collapse tổng gradient;
- mini fine-tune chỉ là sanity, không thay thế full training.

### Trade-off và lưu ý

- Notebook giúp loại bỏ ý tưởng sai nhanh, nhưng không chứng minh được convergence dài hạn.
- Nếu notebook dùng batch không aligned hoặc output cũ sau khi sửa code, kết luận sẽ sai.
- Sau khi pass notebook, vẫn cần full training và journal để ghi metric thật.

## 13. Kích thước ảnh 256x256 so với 384x128

> **Loại:** design-rationale / trade-off
> **Liên quan:** `configs/backbone/m_siglip.yaml`, `configs/aug/img/siglip.yaml`, `src/msiglip/model/build.py`, `src/msiglip/utils/layer_resize.py`

### Định nghĩa

- **384x128:** resolution phổ biến trong person ReID/TBPS vì ảnh người thường là crop dọc.
- **256x256:** resolution vuông khớp với backbone `siglip-base-patch16-256-multilingual`.
- **Patch grid:** với patch size 16, ảnh 256x256 tạo grid 16x16; ảnh 384x128 tạo grid 24x8.
- **Position embedding resize:** nội suy positional embedding khi grid inference/fine-tune khác grid pretrained.

### Bối cảnh trong repo

mSigLIP hiện đi theo geometry vuông 256x256 của pretrained SigLIP. Đổi sang 384x128 không chỉ đổi resize ảnh mà còn đổi spatial prior của vision transformer.

### Cơ chế / nguyên lý

Với `patch_size=16`:

```text
256x256 -> 16 x 16 = 256 patches
384x128 -> 24 x 8  = 192 patches
```

Grid 384x128 giữ dáng người tốt hơn nhưng làm positional embedding vuông bị nội suy sang grid chữ nhật rất lệch.

### Trade-off và lưu ý

- 384x128 hợp với nhiều person ReID model vì giữ aspect ratio người.
- 256x256 hợp hơn với mSigLIP pretrained square input và giữ grid 2D cân bằng.
- 256x256 có thể làm méo người, nhưng giữ nhiều patch hơn và ít phá prior pretrained hơn.
- Nếu ablate resolution, nên thử các lựa chọn ít cực đoan hơn như 256x128, 320x160, 384x192 hoặc letterbox/pad-to-square.

## 14. PACLIP-TPS và prompting-adapting CLIP

> **Loại:** paper-concept / mechanism
> **Liên quan:** `ref/paclip-tps.md`, `ref/paclip-tps-summary.md`

### Định nghĩa

- **PACLIP-TPS:** hướng PEFT cho Text-Based Person Search, đóng băng phần lớn CLIP backbone và train các module prompt/adapter nhẹ.
- **Prompt token/vector:** vector học được chèn vào Transformer để điều hướng representation, không phải prompt ngôn ngữ tự nhiên.
- **CMCP:** Cross-Modal Collaborative Prompting, cơ chế prompt hai chiều giữa vision và text.
- **SVD-LoRA:** biến thể LoRA dùng thông tin cấu trúc từ trọng số pretrained để tăng khả năng thích nghi.

### Bối cảnh trong repo

PACLIP-TPS là tài liệu tham khảo cho hướng PEFT/prompting trong TBPS. Nó không tự động thay thế LoRA hiện tại; mọi ý tưởng lấy từ paper cần ablation riêng trên mSigLIP/VN3K.

### Cơ chế / nguyên lý

Prompt token được cập nhật bằng retrieval loss qua backprop. Khi được chèn vào Transformer, prompt có thể ảnh hưởng tới token ảnh/text thông qua attention. CMCP mở thêm kênh trao đổi cross-modal thay vì để vision/text thích nghi độc lập.

### Trade-off và lưu ý

- Prompt/adapters nhẹ giúp giảm rủi ro phá backbone pretrained.
- Thiết kế cross-modal prompt phức tạp hơn LoRA thường, cần kiểm soát compute và stability.
- Kết quả paper trên benchmark khác không đảm bảo chuyển nguyên sang mSigLIP multilingual.
- Phần liên quan paper/reviewer nên để ở `knowledge/paper/` hoặc `knowledge/response.md`, không để trong knowledge base này.

## 15. RB3-first deployment và adapter boundary

> **Loại:** architecture-principle
> **Liên quan:** `deployment/demo/`, `deployment/docs/deployment-plan.md`, `deployment/docs/journal/`

### Định nghĩa

- **RB3-first:** tiêu chí deploy thật phải được xác nhận trên RB3 Gen2, không chỉ local machine.
- **Adapter boundary:** tách interface khỏi implementation để thay encoder, source, uploader hoặc vector store mà không đổi pipeline chính.
- **Local preflight:** kiểm tra wiring bằng fake/local runtime, không được xem là benchmark deployment.

### Bối cảnh trong repo

Deployment có nhiều tầng: input source, detector/tracker, crop selector, image encoder, vector store/uploader và search. Tách adapter giúp thay dần fake/local implementation bằng QNN/RB3/backend thật.

### Cơ chế / nguyên lý

Một pipeline deploy dễ thay thế nên đi qua contract rõ:

```text
FrameSource -> PersonDetector -> Tracker -> CropSelector
  -> ImageEncoder -> VectorStore/Uploader
```

Mỗi boundary chỉ nên trao đổi data structure rõ ràng như frame, crop, embedding, metadata và artifact path.

### Trade-off và lưu ý

- Local fake encoder hữu ích để test pipeline deterministically, nhưng không có ý nghĩa retrieval thật.
- QNN context binary phải chạy bằng QNN runtime tương ứng, không lẫn với ONNX/DLC/SNPE artifact.
- Board runtime là acceptance gate cho deploy; local pass chỉ là preflight.
- Deployment progress, job ID, QNN/QDQ fidelity và runtime benchmark phải nằm trong `deployment/docs/journal/`.
