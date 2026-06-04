# [Demo System] 2026-06-04 - RB3-first modular demo scaffold + FastAPI backend shell

> **Ngày:** 2026-06-04  
> **Phạm vi:** Modular demo system / adapter scaffold / CLI workflow  
> **Code liên quan:** `deployment/demo/`  
> **Mục tiêu:** Ghi lại phần scaffold hệ thống demo và backend/API shell đã dựng để các module camera/model/backend có thể cắm thay thế dần.  
> **Trạng thái cuối ngày:** PARTIAL

---

## 1. Tóm tắt trong ngày

- Đã dựng scaffold demo modular dưới `deployment/demo/` theo hướng RB3-first.
- Demo system đã có pipeline ingest/search local để kiểm tra wiring trước khi nối camera, backend cloud hoặc Supabase.
- Đã tách code thành `core/`, `adapters/`, `cli/`, `tests/` để tránh gom toàn bộ logic vào một file.
- Đã thêm **FastAPI backend shell** để khóa luồng edge -> backend -> search trước khi model QNN/RB3 hoàn thiện.
- Local preflight và FastAPI smoke đã pass. RB3 QNN acceptance vẫn chỉ là gate của image encoder adapter, chưa phải blocker của system scaffold.

## 2. Vì sao làm việc này

Model vẫn đang trong quá trình training/optimization và vision QNN deployment vẫn còn blocker fidelity. Tuy vậy phần demo end-to-end cần được dựng dần để:

- phần nào deploy được thì cắm vào trước, ví dụ QNN vision encoder trên RB3;
- phần nào chưa sẵn sàng thì dùng adapter local/fake/ONNX để giữ pipeline chạy được;
- tránh chờ đến khi camera, detector thật, vector DB, text service, backend đều xong mới bắt đầu tích hợp;
- giữ ranh giới module rõ để sau này thay adapter mà không viết lại pipeline.

## 3. Code structure / module boundary

Cấu trúc hiện tại:

```text
deployment/demo/
  core/       # contracts, utilities, pipeline orchestration
  adapters/   # sources, detector, tracker, crop selector, encoders, vector store, spool, uploaders
  backend/    # FastAPI API shell, Pydantic schemas, SQLite local backend storage
  cli/        # ingest/search/health command implementations
  tests/      # local preflight tests
  run_*.py    # compatibility wrappers for python -m deployment.demo.*
```

Boundary chính:

```text
FrameSource
  -> PersonDetector
  -> Tracker
  -> CropSelector
  -> ImageEncoder
  -> DiskSpool
  -> Uploader / VectorStore / FastAPI backend
```

Các contract/dataclass chính nằm ở `deployment/demo/core/contracts.py`:

- `Frame`
- `Detection`
- `TrackObservation`
- `CropCandidate`
- `EmbeddingRecord`
- `SearchResult`
- protocol cho source, detector, tracker, selector, encoder, vector store, spool, uploader

## 4. Adapter / CLI / workflow đã thêm hoặc đổi

| # | Thành phần | Việc đã làm | Trạng thái |
|---|---|---|---|
| 1 | `core/` | Thêm contracts, utilities, pipeline orchestration | DONE |
| 2 | `adapters/sources.py` | Thêm image-directory source và video-file source | DONE |
| 3 | `adapters/detectors.py` | Thêm full-frame person detector cho ảnh crop/VN3K-style smoke test | DONE |
| 4 | `adapters/tracker.py` | Thêm simple deterministic tracker sinh `track_id`/`episode_id` | DONE |
| 5 | `adapters/crop_selector.py` | Thêm selector giữ tối đa 3 snapshot mỗi track | DONE |
| 6 | `adapters/encoders.py` | Thêm QNN vision encoder, ONNX fallback, fake vision/text encoder | DONE |
| 7 | `adapters/vector_store.py` | Thêm JSONL vector store với cosine search và collapse theo `episode_id` | DONE |
| 8 | `adapters/spool.py` | Thêm disk spool `pending/`, `sent/`, `failed/` | DONE |
| 9 | `adapters/uploaders.py` | Thêm local vector-store uploader, failing uploader, và `HttpUploader` gửi event sang FastAPI backend | DONE |
| 10 | `backend/` | Thêm FastAPI app, Pydantic schemas, SQLite store, ingest/search/heartbeat/health endpoints | DONE |
| 11 | `cli/` | Thêm `run_ingest`, `run_search`, `health`; `run_search` có thể gọi backend API | DONE |
| 12 | root wrappers | Giữ compatibility cho `python -m deployment.demo.run_ingest`, `run_search`, `health` | DONE |

## 5. Local preflight

Các kiểm tra local đã chạy trong lúc dựng scaffold:

| Kiểm tra | Command | Kết quả | Ghi chú |
|---|---|---|---|
| Compile | `python3 -m compileall deployment/demo` | PASS | Dùng sandbox-safe pycache path khi cần |
| Unit tests | `venv/bin/python -m unittest discover deployment/demo/tests` | PASS | `9 tests OK` |
| Ruff | `venv/bin/ruff check deployment/demo` | PASS | `All checks passed` |
| Fake ingest smoke | `venv/bin/python -m deployment.demo.run_ingest ... --encoder fake` | PASS | 3 frames, 3 detections, 3 crops, 3 embeddings, 0 upload failed |
| Fake search smoke | `venv/bin/python -m deployment.demo.run_search ...` | PASS | Trả top-k JSON, không in embedding mặc định, collapse theo `episode_id` |
| FastAPI ingest/search smoke | `python -m deployment.demo.backend.server` + HTTP ingest/search | PASS | Backend health: `record_count=3`, `heartbeat_count=1` |

Kết quả smoke ingest:

```text
frames_seen = 3
detections_seen = 3
crops_selected = 3
embeddings_written = 3
uploads_failed = 0
```

Kết quả FastAPI backend smoke:

```text
server = http://127.0.0.1:8765
POST /api/v1/ingest/track-embedding = 3 x 200 OK
POST /api/v1/search/text = 200 OK, 3 results, no embedding by default
POST /api/v1/boards/heartbeat = 200 OK
GET /api/v1/health = record_count 3, heartbeat_count 1
```

## 6. RB3 acceptance status

Chưa chạy acceptance trên RB3 cho demo system, và đây **không phải bước tiếp theo ngay**.

Lý do: model vẫn đang trong quá trình optimize/fidelity, nên không nên để QNN runtime trở thành blocker cho việc dựng hệ thống xung quanh model. Ở giai đoạn này, image encoder nên được xem là một adapter có thể thay thế:

```text
FakeVisionEncoder -> OnnxVisionEncoder -> QnnVisionEncoder
```

Hệ thống cần chạy được end-to-end với fake/ONNX trước để khóa API contract, storage contract, upload/retry, search flow và UI/search surface. Khi model candidate tốt hơn, chỉ thay adapter encoder.

RB3/QNN gate vẫn cần làm, nhưng chuyển thành bước sau của image-runtime validation:

- chạy `run_ingest` trên RB3 với `--encoder qnn`;
- dùng `vision_encoder.bin` bằng QNN `qnn-net-run`;
- xác minh output image embedding 768 chiều finite;
- xác minh vector được L2-normalize trước khi lưu/search;
- đo latency/FPS demo path trên board;
- xác minh disk spool hoạt động khi upload fail/retry trên board.

Command dự kiến:

```bash
python -m deployment.demo.run_ingest \
  --source /path/to/images_or_video \
  --encoder qnn \
  --vision-bin vision_encoder.bin \
  --htp-config htp_config_245.json \
  --qairt /opt/qcom/qairt/2.45.40.260406 \
  --board-id qc-rb3g2 \
  --camera-id cam-lab-01
```

## 7. Điều không claim

- Local fake/ONNX preflight không chứng minh deployment thành công.
- Fake embeddings không có ý nghĩa retrieval thật.
- JSONL vector store chỉ là backend local tạm thời, không phải Supabase/production store.
- FastAPI + SQLite backend chỉ là local API shell để khóa contract, không phải backend production.
- Full-frame detector chỉ phù hợp smoke test với ảnh crop người, chưa phải person detector thật từ camera.
- RB3 QNN acceptance chưa chạy trong ngày này.
- Vision QNN fidelity/retrieval accuracy vẫn là việc riêng của deploy/QNN journal `[deploy]`.
- QNN chưa pass không được chặn việc dựng backend/API/Web UI/search shell, miễn là encoder vẫn nằm sau adapter boundary.

## 8. Quyết định kỹ thuật

- Tách journal demo system thành prefix riêng `[demo-system]` để không lẫn với `[deploy]`.
- Giữ RB3 là acceptance gate thật cho image-runtime adapter, nhưng không coi đó là bước đầu tiên của demo system.
- Ưu tiên dựng hệ thống quanh model trước: backend ingest/search API, uploader, storage, result collapse, health, rồi mới swap encoder runtime khi model sẵn sàng.
- Chọn FastAPI + Pydantic + SQLite cho backend shell v1 để có contract rõ, dễ thay bằng Supabase/Qdrant sau.
- `HttpUploader` là đường edge -> backend chính cho demo từ giờ; `LocalVectorStoreUploader` vẫn giữ để local preflight không cần server.
- Dùng adapter boundary thay vì script monolithic để sau này cắm camera, detector, backend, text service, vector DB mà không viết lại pipeline.
- Giữ root wrappers cho CLI để command ngắn không đổi dù implementation đã chuyển vào `deployment/demo/cli/`.
- Search/display mặc định collapse theo `episode_id`, còn raw snapshot vẫn là đơn vị vector search.

## 9. Việc tiếp theo

- [x] Dựng backend demo/local API để khóa system contract quanh model:
  - `POST /api/v1/ingest/track-embedding`
  - `POST /api/v1/search/text`
  - `POST /api/v1/boards/heartbeat`
- [x] Thêm `HttpUploader` cho edge pipeline để gửi event từ `deployment/demo` sang backend API thay vì ghi trực tiếp local vector store.
- [x] Dùng local backend storage tạm thời:
  - metadata bằng SQLite
  - vector search brute-force cosine
  - crop/object artifact bằng local folder/path
- [x] Implement search API với fake text encoder trước, có filter camera/time và collapse kết quả theo `episode_id`.
- [x] Thêm CLI search client gọi backend API.
- [ ] Dựng Web UI mỏng gọi backend search API.
- [ ] Thêm backend endpoint hoặc CLI retry worker để đẩy lại events từ `spool/failed` khi backend/network hồi phục.
- [ ] Khi backend/search shell ổn, thêm camera source adapter USB/IP hoặc RTSP.
- [ ] Thêm detector/tracker thật sau khi source camera ổn định.
- [ ] Chạy `run_ingest --encoder qnn` trên RB3 như image-runtime adapter smoke test khi có model/runtime candidate phù hợp.
- [ ] Ghi kết quả RB3 demo acceptance vào journal `[demo-system]` cùng ngày chạy, nhưng chỉ sau khi bước backend/system shell đủ rõ.
- [ ] Nếu demo scaffold thay đổi code/config/docs, cập nhật changelog deployment sau khi user xác nhận.

## 10. Rủi ro / câu hỏi mở

- Nếu chạy QNN quá sớm, team dễ bị kéo về model deployment/debug và chậm dựng system shell quanh model.
- QNN vision binary hiện vẫn có blocker fidelity ở deploy path; demo QNN runtime có thể chạy nhưng chưa đảm bảo retrieval đúng.
- Chưa có live camera adapter nên demo hiện phù hợp ảnh/video file trước.
- Text embedding service thật chưa được cắm; fake text encoder chỉ dùng local wiring.
- FastAPI backend hiện chưa có auth enforcement thật; `board_token`/`user_token` mới là header wiring.
- Chưa có background retry worker tự động cho failed spool events.
- Cần tránh để local preflight bị hiểu nhầm là board deployment success.
- Cần chọn backend local tối giản đủ nhanh nhưng không khóa chặt production path: ưu tiên API contract rõ, adapter dễ thay hơn là tối ưu storage ngay từ đầu.
