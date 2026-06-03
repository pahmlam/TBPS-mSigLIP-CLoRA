# Training Journal

Thư mục này ghi tiến độ theo ngày cho phần training và tối ưu model.

## Quy ước file

- Tên file: `[train]-YYYY-MM-DD.md`
- Mỗi ngày append vào file cùng ngày nếu file đã tồn tại.
- Nội dung phù hợp: lệnh chạy, log quan trọng, metric, kết luận tạm thời, quyết định thử nghiệm tiếp theo.

## Không ghi ở đây

- Định nghĩa/khái niệm bền vững: ghi vào `docs/knowledge.md`.
- Changelog của thay đổi code/config/docs: ghi vào `changelog/training/changelog.md` sau khi được user xác nhận.
- Wording cho paper/reviewer: ghi vào `knowledge/response.md` hoặc `knowledge/paper/`.

## Template file theo ngày

```markdown
# [Train] YYYY-MM-DD - Chủ đề chính

> **Ngày:** YYYY-MM-DD
> **Phạm vi:** Training / loss / data / notebook / evaluation
> **Mục tiêu:** Câu ngắn mô tả đang muốn xác minh điều gì
> **Trạng thái cuối ngày:** DONE / PARTIAL / BLOCKED / FOLLOW-UP

---

## 1. Tóm tắt quyết định

- Kết luận quan trọng nhất trong ngày.
- Có nên tiếp tục hướng này không?
- Nếu có, bước tiếp theo là gì?

## 2. Bối cảnh

- Baseline hoặc kết quả trước đó.
- Vì sao hôm nay cần chạy/kiểm tra việc này.
- Link tới knowledge nếu cần định nghĩa khái niệm.

## 3. Thí nghiệm / thay đổi đã làm

| # | Việc | Command/config | Artifact/log | Trạng thái |
|---|---|---|---|---|
| 1 | ... | `...` | `...` | DONE |

## 4. Kết quả

| Run | Metric chính | Metric phụ | So với baseline | Ghi chú |
|---|---:|---:|---:|---|
| ... | ... | ... | ... | ... |

## 5. Diễn giải

- Điều gì đã được chứng minh.
- Điều gì chưa được chứng minh.
- Khả năng lỗi/variance/confounder nếu có.

## 6. Quyết định tiếp theo

- [ ] Việc tiếp theo 1
- [ ] Việc tiếp theo 2

## 7. Câu hỏi mở

- Câu hỏi còn cần user hoặc thí nghiệm tiếp theo trả lời.
```

## Template entry append nhanh

Dùng khi file ngày đã tồn tại và chỉ cần thêm một mục mới.

```markdown
## HH:MM - Tên thí nghiệm / phân tích

### Mục tiêu

### Command / Config

### Artifact / Log

### Kết quả

### Diễn giải

### Quyết định tiếp theo
```
