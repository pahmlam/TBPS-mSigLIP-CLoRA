# Deployment Journal

Thư mục này ghi tiến độ theo ngày cho phần deploy và tối ưu pipeline edge.

## Quy ước file

- Tên file: `[deploy]-YYYY-MM-DD.md`
- Tên file kế hoạch/checklist: `[deploy-plan]-YYYY-MM-DD.md`
- Tên file demo system: `[demo-system]-YYYY-MM-DD.md`
- Mỗi ngày append vào file cùng ngày nếu file đã tồn tại.
- Nội dung phù hợp: AI Hub job, QNN/QDQ fidelity, RB3 runtime, artifact, command, log quan trọng, kết luận tạm thời, quyết định deploy tiếp theo.
- Nội dung demo system phù hợp: scaffold/module boundary của `deployment/demo/`, adapter, CLI workflow, local preflight, trạng thái RB3 demo acceptance.
- Với một kế hoạch deploy đang chạy, chỉ giữ **một plan checklist** trong file `[deploy-plan]`, ví dụ `deployment/docs/journal/[deploy-plan]-2026-06-06.md`.
- Các file ngày sau khi chưa tạo kế hoạch mới chỉ ghi kết quả run trong ngày, chẩn đoán, và việc tiếp theo; không lặp lại checklist tổng.
- Nếu kết quả trong ngày làm thay đổi trạng thái kế hoạch, cập nhật checklist tổng ở file `[deploy-plan]` và ghi trong daily journal rằng trạng thái đã được phản ánh ở đó.

## Không ghi ở đây

- Định nghĩa/khái niệm bền vững như ONNX, QNN, HTP, PTQ: ghi vào `docs/knowledge.md`.
- Changelog của thay đổi code/config/docs: ghi vào `changelog/deployment/changelog.md` sau khi được user xác nhận.
- Trạng thái hiện hành cấp cao: cập nhật `deployment/docs/deployment-plan.md` khi user yêu cầu hoặc xác nhận.
- AI Hub jobs, QDQ/QNN fidelity, board benchmark logs không ghi vào `[demo-system]`; các nội dung đó thuộc `[deploy]`.
- Training/model optimization không ghi ở đây; dùng `docs/journal/[train]-YYYY-MM-DD.md`.

## Template file kế hoạch / plan checklist

Dùng khi bắt đầu một kế hoạch deploy mới có nhiều phase hoặc nhiều run nối tiếp. Khi kế hoạch này còn hiệu lực, các ngày sau cập nhật checklist ở chính file này.

````markdown
# [Deploy Plan] YYYY-MM-DD - Kế hoạch ...

> **Ngày:** YYYY-MM-DD
> **Thiết bị / Runtime:** RB3 / QNN HTP / ONNX Runtime / AI Hub
> **Model / Artifact nguồn:** checkpoint, ONNX, QDQ, hoặc QNN binary
> **Mục tiêu:** Câu ngắn mô tả kế hoạch đang mở khóa điều gì
> **Plan checklist hiện hành:** file này
> **Trạng thái:** DONE / PARTIAL / BLOCKED / FOLLOW-UP
> **Cập nhật checklist gần nhất:** YYYY-MM-DD

---

## 1. Tóm tắt quyết định

- Gate hiện tại.
- Blocker chính.
- Điều chưa được phép claim hoặc mở rộng.

## 2. Gate bắt buộc

| Gate | Ngưỡng | Ý nghĩa |
|---|---:|---|
| ... | ... | ... |

## 3. Checklist tổng của kế hoạch

### Phase A - ...

- [ ] Việc cần làm
- [ ] Việc cần làm

## 4. Command / Quy trình đang mở

```bash
...
```

## 5. Thứ tự ưu tiên

1. Việc tiếp theo gần nhất.
2. Nhánh xử lý nếu fail.
3. Gate để được đi tiếp.
````

## Template file theo ngày

Dùng cho daily journal khi đã có plan checklist. File ngày không chứa checklist tổng; chỉ link đến `[deploy-plan]` và ghi kết quả/chẩn đoán/việc tiếp theo của ngày đó.

```markdown
# [Deploy] YYYY-MM-DD - Chủ đề chính

> **Ngày:** YYYY-MM-DD
> **Thiết bị / Runtime:** RB3 / QNN HTP / ONNX Runtime / AI Hub
> **Model / Artifact nguồn:** checkpoint, ONNX, QDQ, hoặc QNN binary
> **Mục tiêu:** Câu ngắn mô tả đang muốn xác minh điều gì
> **Plan checklist:** `deployment/docs/journal/[deploy-plan]-YYYY-MM-DD.md`
> **Trạng thái cuối ngày:** DONE / PARTIAL / BLOCKED / FOLLOW-UP

---

## 1. Tóm tắt quyết định

- Kết luận quan trọng nhất trong ngày.
- Gate nào pass/fail?
- Có được phép mở rộng benchmark/compile/link tiếp không?

## 2. Bối cảnh

- Trạng thái pipeline trước ngày hôm nay.
- Artifact đầu vào và blocker hiện tại.
- File plan checklist đang được follow.
- Link tới knowledge nếu cần định nghĩa khái niệm.

## 3. Job / Command / Artifact

| # | Việc | Command/job ID | Input | Output | Trạng thái |
|---|---|---|---|---|---|
| 1 | ... | `...` | `...` | `...` | DONE |

## 4. Kết quả fidelity / runtime

| So sánh / Run | Metric chính | Metric phụ | Gate | Kết luận |
|---|---:|---:|---|---|
| ... | ... | ... | PASS/FAIL | ... |

## 5. Chẩn đoán

- Lỗi nằm ở tầng nào: export, preprocess, QDQ/PTQ, compile/link, runtime, hay compare script?
- Bằng chứng nào loại trừ các tầng còn lại?

## 6. Quyết định tiếp theo

- Việc tiếp theo gần nhất.
- Có cập nhật plan checklist không, và cập nhật ở file nào.
- Gate để được compile/link, benchmark rộng hơn, hoặc chuyển sang phase mới.

## 7. Câu hỏi mở / Rủi ro

- Rủi ro kỹ thuật hoặc thông tin cần xác minh thêm.
```

## Template demo system theo ngày

Dùng cho thay đổi ở `deployment/demo/`: scaffold, module boundary, adapter, CLI workflow, local preflight, và trạng thái acceptance demo trên RB3.

```markdown
# [Demo System] YYYY-MM-DD - Chủ đề chính

> **Ngày:** YYYY-MM-DD
> **Phạm vi:** Modular demo system / adapter scaffold / CLI workflow
> **Code liên quan:** `deployment/demo/`
> **Mục tiêu:** Câu ngắn mô tả muốn dựng hoặc xác minh phần nào của demo system
> **Trạng thái cuối ngày:** DONE / PARTIAL / BLOCKED / FOLLOW-UP

---

## 1. Tóm tắt trong ngày

- Việc quan trọng nhất đã làm.
- Module/adapter/workflow nào đã thay đổi?
- Local preflight pass/fail?

## 2. Vì sao làm việc này

- Demo system đang cần mở khóa điều gì?
- Phần nào đã deploy được, phần nào vẫn là placeholder/local preflight?

## 3. Code structure / module boundary

- Cấu trúc thư mục hoặc boundary mới.
- Interface/contract quan trọng.
- Quy tắc tách module cần giữ.

## 4. Adapter / CLI / workflow đã thêm hoặc đổi

| # | Thành phần | Việc đã làm | Trạng thái |
|---|---|---|---|
| 1 | ... | ... | DONE |

## 5. Local preflight

| Kiểm tra | Command | Kết quả | Ghi chú |
|---|---|---|---|
| ... | `...` | PASS/FAIL | ... |

## 6. RB3 acceptance status

- Đã chạy trên RB3 chưa?
- Nếu chưa, gate nào còn pending?
- Nếu có, artifact/runtime nào được dùng?

## 7. Điều không claim

- Những gì local preflight chưa chứng minh.
- Những phần chưa thể coi là deploy success.

## 8. Quyết định kỹ thuật

- Quyết định kiến trúc hoặc workflow trong ngày.
- Lý do chọn cách này thay vì hướng khác.

## 9. Việc tiếp theo

- [ ] Việc tiếp theo 1
- [ ] Việc tiếp theo 2

## 10. Rủi ro / câu hỏi mở

- Rủi ro kỹ thuật hoặc thông tin cần xác minh thêm.
```

## Template entry append nhanh

Dùng khi file ngày đã tồn tại và chỉ cần thêm một job hoặc kết quả mới.

```markdown
## HH:MM - Tên job / kiểm tra

### Mục tiêu

### Command / Job ID

### Input / Output artifact

### Kết quả

### Chẩn đoán

### Quyết định tiếp theo
```
