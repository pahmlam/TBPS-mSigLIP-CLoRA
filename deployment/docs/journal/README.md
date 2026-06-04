# Deployment Journal

Thư mục này ghi tiến độ theo ngày cho phần deploy và tối ưu pipeline edge.

## Quy ước file

- Tên file: `[deploy]-YYYY-MM-DD.md`
- Tên file demo system: `[demo-system]-YYYY-MM-DD.md`
- Mỗi ngày append vào file cùng ngày nếu file đã tồn tại.
- Nội dung phù hợp: AI Hub job, QNN/QDQ fidelity, RB3 runtime, artifact, command, log quan trọng, kết luận tạm thời, quyết định deploy tiếp theo.
- Nội dung demo system phù hợp: scaffold/module boundary của `deployment/demo/`, adapter, CLI workflow, local preflight, trạng thái RB3 demo acceptance.

## Không ghi ở đây

- Định nghĩa/khái niệm bền vững như ONNX, QNN, HTP, PTQ: ghi vào `docs/knowledge.md`.
- Changelog của thay đổi code/config/docs: ghi vào `changelog/deployment/changelog.md` sau khi được user xác nhận.
- Trạng thái hiện hành cấp cao: cập nhật `deployment/docs/deployment-plan.md` khi user yêu cầu hoặc xác nhận.
- AI Hub jobs, QDQ/QNN fidelity, board benchmark logs không ghi vào `[demo-system]`; các nội dung đó thuộc `[deploy]`.
- Training/model optimization không ghi ở đây; dùng `docs/journal/[train]-YYYY-MM-DD.md`.

## Template file theo ngày

```markdown
# [Deploy] YYYY-MM-DD - Chủ đề chính

> **Ngày:** YYYY-MM-DD
> **Thiết bị / Runtime:** RB3 / QNN HTP / ONNX Runtime / AI Hub
> **Model / Artifact nguồn:** checkpoint, ONNX, QDQ, hoặc QNN binary
> **Mục tiêu:** Câu ngắn mô tả đang muốn xác minh điều gì
> **Trạng thái cuối ngày:** DONE / PARTIAL / BLOCKED / FOLLOW-UP

---

## 1. Tóm tắt quyết định

- Kết luận quan trọng nhất trong ngày.
- Gate nào pass/fail?
- Có được phép mở rộng benchmark/compile/link tiếp không?

## 2. Bối cảnh

- Trạng thái pipeline trước ngày hôm nay.
- Artifact đầu vào và blocker hiện tại.
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

- [ ] Việc tiếp theo 1
- [ ] Việc tiếp theo 2

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
