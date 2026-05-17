# mSigLIP End-to-End System Design on Qualcomm RB3 Gen2

> **Status:** Proposed architecture  
> **Date:** 2026-04-26  
> **Scope:** Post-training, post-compression, production-oriented end-to-end retrieval system  
> **Target device:** Qualcomm RB3 Gen2 (`qc-rb3g2`, QCS6490, Ubuntu 24.04, ARM64)

---

## 1. Mục tiêu

Sau khi hoàn tất 2 giai đoạn:
- cải thiện chất lượng mô hình mSigLIP
- nén / deploy model để chạy được trên Qualcomm RB3 Gen2

mục tiêu tiếp theo là xây dựng một **hệ thống end-to-end** có thể:
- nhận video trực tiếp từ camera gắn với board
- phát hiện người trong khung hình
- crop person image
- sinh **image embedding** trên board
- lưu embedding vào **vector database** để truy hồi
- cung cấp **Web UI** cho người dùng nhập mô tả tiếng Việt
- thực hiện **text embedding ngoài board**
- trả về top-k kết quả phù hợp theo thời gian thực hoặc gần thời gian thực

Hệ thống phải ưu tiên:
- không gây OOM trên board
- không phụ thuộc cloud cho image embedding
- có khả năng demo từ xa qua Internet
- có đường phát triển từ prototype miễn phí sang production

---

## 2. Ràng buộc và giả định

### 2.1 Ràng buộc phần cứng

Theo `deployment/docs/system.md` và `deployment/docs/deployment-plan.md`:
- Board có khoảng **~4 GB RAM khả dụng** cho workload inference.
- Vision encoder và text encoder của mSigLIP là 2 nhánh lớn, không nên chạy đồng thời trên board trong giai đoạn đầu.
- HTP/DSP là mục tiêu tối ưu, nhưng hiện pipeline ổn định nhất về mặt thiết kế hệ thống vẫn nên có fallback `GPU -> CPU`.

### 2.2 Ràng buộc vận hành

- Board hiện đã có mạng ổn định và có thể truy cập từ xa qua tunnel/web admin.
- Board phù hợp cho **edge inference node**, không phù hợp để kiêm luôn toàn bộ backend public, vector DB, object storage và web app.
- Tunnel public chỉ nên dùng cho vận hành/admin, không nên là đường truy cập chính của người dùng cuối.

### 2.3 Ràng buộc mô hình

- **Text embedding bắt buộc phải dùng đúng text encoder của cùng checkpoint mSigLIP** đã dùng để tạo image embedding.
- Không được thay text encoder bằng model embedding khác vì sẽ làm lệch không gian embedding 768 chiều.
- Vector lưu trữ nên là embedding **L2-normalized** để truy hồi bằng cosine similarity ổn định.

### 2.4 Giả định sản phẩm

- Use case chính: truy hồi người từ mô tả văn bản tiếng Việt.
- Một thời điểm có thể có 1 hoặc vài camera, nhưng kiến trúc nên mở rộng được.
- Prototype ban đầu chấp nhận một phần dịch vụ dùng free tier, nhưng kiến trúc không được khóa cứng vào free tier.

---

## 3. Kiến trúc tổng thể đề xuất

```text
Camera
  |
  v
Qualcomm RB3 Gen2
  - capture
  - person detect
  - track
  - crop
  - image encoder
  - local spool / retry queue
  |
  | HTTPS (outbound only)
  v
Cloud Backend
  - ingest API
  - metadata DB
  - object storage
  - vector DB
  - search API
  |
  +--> Remote Text Embedding Service
  |      - same mSigLIP text encoder
  |      - query text -> 768-d vector
  |
  v
Web UI
  - user nhập text query
  - filter theo camera / thời gian
  - xem top-k kết quả
```

### Quyết định lõi

- **Board chỉ chạy image pipeline**.
- **Text embedding chạy ngoài board**.
- **Vector DB đặt ngoài board**.
- **User không truy cập trực tiếp vào board**.
- **Board chủ động push dữ liệu outbound lên backend**.

Đây là tách lớp hợp lý nhất để tránh OOM, tránh public board ra Internet cho user traffic, và giữ cho board luôn ưu tiên tài nguyên cho camera + image encoder.

---

## 4. Phân rã thành phần hệ thống

### 4.1 Edge Node: Qualcomm RB3 Gen2

Board chịu trách nhiệm cho toàn bộ luồng realtime gần camera:

1. Nhận stream từ camera USB/CSI/IP camera.
2. Chạy **person detection**.
3. Chạy **tracking** để gán `track_id` cho cùng một người qua nhiều frame.
4. Sinh crop ảnh người từ bbox.
5. Chọn frame tốt nhất hoặc một số frame đại diện cho mỗi track.
6. Chạy **image encoder mSigLIP** trên crop đã chọn.
7. Gửi embedding + metadata + thumbnail ra backend.
8. Ghi tạm vào local spool nếu mất mạng hoặc backend lỗi.

Board không nên:
- host public website chính
- chứa vector DB chính
- chạy text encoder thường trực
- giữ lâu dài toàn bộ lịch sử embeddings

### 4.2 Cloud Backend

Backend ngoài board chịu trách nhiệm:

- nhận ingest từ board
- lưu metadata về camera, track, event, thời gian
- lưu ảnh crop hoặc thumbnail
- lưu vector embeddings
- nhận text query từ Web UI
- gọi text-embedding service
- thực hiện vector search + filter theo metadata
- trả kết quả cho Web UI

Backend có thể được tách làm 2 lớp:
- **data layer:** database, vector index, storage
- **application layer:** API search, auth, business logic

### 4.3 Remote Text Embedding Service

Service này chỉ có 1 nhiệm vụ:
- nhận chuỗi text
- tokenize đúng tokenizer của mSigLIP
- chạy đúng text encoder của cùng checkpoint
- trả về vector 768 chiều đã normalize

Vì text encoder lớn và tốn RAM, tách nó ra khỏi board sẽ:
- giảm nguy cơ OOM
- tách tải image path và query path
- cho phép scale độc lập nếu query tăng

### 4.4 Web UI

Web UI là lớp tương tác người dùng:
- nhập mô tả người
- chọn camera / khoảng thời gian
- xem top-k kết quả
- xem crop, timestamp, camera, score
- có thể thêm màn hình giám sát health hệ thống cho admin

Web UI không nên gọi trực tiếp tới board. Nó chỉ nói chuyện với backend public.

---

## 5. Luồng dữ liệu end-to-end

### 5.1 Luồng ingest từ camera đến vector DB

```text
Camera stream
  -> Board capture
  -> Person detector
  -> Tracker
  -> Crop selector
  -> Image encoder
  -> Local queue
  -> Backend ingest API
  -> Object storage + metadata DB + vector DB
```

### Chi tiết

1. `capture` đọc frame từ camera.
2. `detector` tìm bbox của người.
3. `tracker` gán `track_id`, giúp gộp nhiều frame của cùng một người.
4. `crop selector` quyết định có nên lưu frame này không.
5. `encoder` sinh embedding từ crop.
6. `uploader` gửi:
   - `embedding`
   - `camera_id`
   - `track_id`
   - `timestamp`
   - `bbox`
   - `image_path` hoặc thumbnail
   - `quality_score`
7. Backend ghi vào storage + DB + vector index.

### 5.2 Luồng truy hồi text query

```text
User Web UI
  -> Search API
  -> Text embedding service
  -> Vector DB similarity search
  -> Metadata join / filters
  -> Top-k results
  -> Web UI
```

### Chi tiết

1. User nhập mô tả tiếng Việt.
2. Search API kiểm tra auth, validate filter.
3. API gọi text embedding service.
4. Nhận embedding text 768-d.
5. Query vector DB theo cosine similarity.
6. Lọc theo:
   - camera
   - time range
   - confidence / quality threshold
7. Join metadata để lấy ảnh, timestamp, location.
8. **Collapse / dedup** các kết quả trùng gần nhau theo `episode_id`.
9. Trả top-k cuối cùng cho Web UI.

### Query-time dedup / result collapsing

Đây là lớp rất quan trọng cho UX. Dù ingest đã có suppression policy, vector search raw vẫn có thể trả về:
- nhiều snapshot của cùng một track
- nhiều track của cùng một `episode_id`
- nhiều crop gần như giống nhau của cùng một người

Nếu trả thẳng raw top 10, user sẽ thấy danh sách bị chiếm chỗ bởi near-duplicates.

#### Nguyên tắc khuyến nghị

- **Search unit:** snapshot embedding
- **Display unit:** `episode_id`
- **Scoring unit:** snapshot score cao nhất trong episode

Nói cách khác:
1. Search trên tất cả snapshot để giữ recall.
2. Lấy `top_k_raw` lớn hơn nhu cầu hiển thị, ví dụ 50 hoặc 100.
3. Group kết quả theo `episode_id`.
4. Với mỗi episode, chọn:
   - snapshot có score cao nhất làm representative
   - thumbnail đẹp nhất hoặc score cao nhất để hiển thị
   - `first_seen`, `last_seen`, `camera_id`, `snapshot_count`
5. Sort lại theo representative score.
6. Trả ra `top_k_final = 10` episode.

#### Rule mặc định cho prototype

```yaml
search_result_policy:
  top_k_raw: 50
  top_k_final: 10
  collapse_key: episode_id
  representative_strategy: max_score
  thumbnail_strategy: best_quality
  allow_same_episode_multiple_rows: false
```

#### Khi nào cho phép không collapse?

Chỉ nên trả nhiều dòng của cùng một người nếu UI đang ở chế độ:
- xem timeline chi tiết
- điều tra forensic
- xem toàn bộ sightings của một episode

Với màn hình search mặc định, nên collapse theo `episode_id`.

### 5.3 Luồng khi board mất mạng

Board phải có chế độ offline-first tối thiểu:

1. Nếu upload thất bại, event được ghi vào local disk queue.
2. Ảnh crop và JSON metadata được giữ tạm.
3. Một background worker retry upload khi mạng hồi phục.
4. Chỉ xóa local artifact sau khi backend xác nhận ingest thành công.

Nếu không có local spool, demo sẽ mất dữ liệu ngay khi Wi-Fi chập chờn.

---

## 6. Thiết kế chi tiết trên board

### 6.1 Pipeline tiến trình khuyến nghị

Không nên gom toàn bộ vào 1 process lớn. Nên tách thành các worker với queue hữu hạn:

```text
Process A: Capture
  -> frame_queue

Process B: Detect + Track
  -> candidate_queue

Process C: Crop Selection / Dedup
  -> embed_queue

Process D: Image Encoder
  -> upload_queue

Process E: Upload / Retry / Health Report
```

### Lý do

- Dễ giới hạn bộ nhớ bằng queue bounded.
- Nếu upload chậm, encoder không bị block hoàn toàn.
- Nếu detector bị lỗi, có thể restart riêng.
- Dễ đo latency từng stage.

### 6.2 Chính sách queue để tránh OOM

Mỗi queue phải có giới hạn. Ví dụ:
- `frame_queue`: 2-4 frame
- `candidate_queue`: 16 items
- `embed_queue`: 16 items
- `upload_queue`: 64 items

Nếu queue đầy:
- ưu tiên **drop frame cũ**
- không tích vô hạn trong RAM

Nguyên tắc là hệ thống edge nên degrade bằng cách giảm sample rate, không phải bằng cách chết OOM.

### 6.3 Chính sách chọn crop

Không nên embed mọi bbox của mọi frame.

Nên áp dụng ít nhất 1 trong 3 chiến lược:

### Chiến lược A: Best frame per track

Mỗi `track_id` chỉ giữ 1 frame tốt nhất, chọn theo:
- bbox đủ lớn
- ảnh ít blur
- confidence cao
- không bị che khuất nhiều

### Chiến lược B: Top-N frames per track

Mỗi track giữ 3-5 frame tốt nhất, sau đó:
- lưu nhiều vector
- hoặc lấy mean embedding sau khi normalize

### Chiến lược C: Periodic snapshot

Nếu cần timeline liên tục:
- lưu 1 snapshot mỗi `N` giây trên cùng track
- ví dụ `N = 2s` hoặc `N = 5s`

### Khuyến nghị

Prototype nên bắt đầu với:
- `Top-3 frames per track`
- aggregate thành `1 embedding đại diện`
- vẫn giữ 3 thumbnail để UI hiển thị

Cách này cân bằng giữa chất lượng truy hồi và kích thước DB.

#### Chính sách chống capture quá nhiều ảnh cho cùng một người

Chỉ dùng tracking thôi là chưa đủ. Trong thực tế, một người vẫn có thể bị lưu quá nhiều ảnh vì:
- track bị đứt rồi nối lại thành track mới
- người đứng lâu trước camera
- người đi qua lại cùng một vùng nhiều lần trong thời gian ngắn
- crop mới không tốt hơn crop cũ nhưng pipeline vẫn tiếp tục lưu

Nên áp dụng **4 lớp suppression** theo thứ tự:

1. **Frame-level gating**
   - Chỉ xét lưu nếu bbox đủ lớn, detector confidence đủ cao, blur thấp.
   - Nếu crop hiện tại kém hơn crop tốt nhất đã có của track thì bỏ qua ngay.

2. **Intra-track rate limit**
   - Mỗi track có `max_snapshots_per_track`.
   - Mỗi lần lưu phải cách snapshot trước ít nhất `min_snapshot_interval_sec`.
   - Chỉ lưu snapshot mới nếu có thay đổi đáng kể về chất lượng hoặc góc nhìn.

3. **Track finalization rule**
   - Khi track kết thúc, chỉ push event nếu track đạt `min_track_duration_sec` hoặc `min_frames_seen`.
   - Các track quá ngắn hoặc crop quá xấu chỉ dùng cho online monitoring, không ghi vào DB chính.

4. **Cross-track duplicate suppression**
   - Sau khi finalize một track, giữ lại embedding đại diện của track đó trong `recent_identity_cache` theo từng camera.
   - Nếu một track mới xuất hiện trong cửa sổ thời gian ngắn và embedding quá giống track vừa lưu, coi đó là **continuation** thay vì người mới.
   - Khi đó có thể:
     - không tạo bản ghi mới
     - hoặc gắn nó vào cùng `episode_id`
     - hoặc chỉ update `end_ts` và thumbnail tốt hơn cho event cũ

#### Rule lưu ảnh khuyến nghị cho prototype

Board nên dùng rule mặc định như sau:

```yaml
capture_policy:
  max_snapshots_per_track: 3
  min_snapshot_interval_sec: 3
  min_track_duration_sec: 1.5
  min_frames_seen: 8
  min_bbox_area_ratio: 0.03
  min_quality_score: 0.55
  min_quality_improvement: 0.08
  recent_identity_window_sec: 60
  cross_track_same_camera_cosine_thresh: 0.92
  allow_episode_reopen: true
```

Ý nghĩa:
- `max_snapshots_per_track`: chặn việc một người đứng trước camera 30 giây nhưng bị lưu 30 ảnh.
- `min_snapshot_interval_sec`: nếu người vẫn là cùng track và gần như không đổi pose thì không lưu dày.
- `min_quality_improvement`: snapshot mới phải tốt hơn đủ đáng kể thì mới thay thế hoặc bổ sung.
- `recent_identity_window_sec`: chống việc tracker bị đứt ngắn hạn rồi tạo track mới cho cùng một người.
- `cross_track_same_camera_cosine_thresh`: chỉ suppress nếu giống rất cao và cùng camera; không nên dùng ngưỡng thấp vì dễ merge nhầm hai người ăn mặc giống nhau.

#### Tiêu chí "thay đổi đủ để đáng lưu"

Một snapshot mới chỉ nên được giữ nếu thỏa **ít nhất một** trong các điều kiện:
- quality score tăng hơn `min_quality_improvement`
- bbox area tăng đáng kể so với snapshot trước
- pose/appearance khác rõ rệt, ví dụ cosine với snapshot gần nhất của cùng track nhỏ hơn `0.98`
- đã vượt `min_snapshot_interval_sec` và track sắp kết thúc nên cần lưu frame cuối

#### Cách xử lý khi tracker bị đứt

Đây là nguồn duplicate lớn nhất.

Giải pháp thực tế:
- Khi một track kết thúc, tạo `track_summary_embedding`.
- Giữ embedding này trong `recent_identity_cache[camera_id]` khoảng 30-60 giây.
- Nếu track mới vào cùng camera trong khoảng thời gian này:
  - compute provisional embedding từ best crop đầu tiên
  - so với cache gần nhất
  - nếu similarity vượt ngưỡng và khoảng cách thời gian nhỏ, gộp vào cùng `episode_id`

Như vậy hệ thống sẽ lưu theo **episode của một người trong camera**, không lưu cứng theo từng `track_id` do tracker sinh ra.

#### Cách xử lý khi một người đứng quá lâu trong khung hình

Trường hợp này khác với duplicate do tracker bị đứt. Ở đây `track_id` có thể vẫn ổn định, nhưng nếu không chặn, pipeline sẽ tiếp tục lưu thêm ảnh của cùng một người dù thông tin mới gần như không có giá trị.

Hành vi khuyến nghị:

1. Trong 1 track dài, chỉ lưu các snapshot đầu tiên đủ tốt theo `max_snapshots_per_track`.
2. Khi đã đạt ngưỡng này, **không lưu thêm embedding mới** cho track đó nữa.
3. Trong lúc người vẫn đứng trong khung hình, chỉ cập nhật:
   - `last_seen`
   - `end_ts`
   - trạng thái track
4. Nếu cần monitoring dài hạn, chỉ ghi **metadata heartbeat** theo chu kỳ, không ghi thêm vector.
5. Chỉ cho phép lưu thêm 1 snapshot mới nếu:
   - pose / góc nhìn thay đổi đáng kể
   - chất lượng crop tăng rõ rệt
   - hoặc đã qua một cửa sổ thời gian dài đủ lớn để coi là context mới

Rule mặc định phù hợp cho prototype:

```yaml
long_dwell_policy:
  freeze_embeddings_after_max_snapshots: true
  metadata_heartbeat_sec: 30
  allow_refresh_if_appearance_changes: true
  same_track_refresh_min_sec: 120
  same_track_refresh_cosine_thresh: 0.98
```

Ý nghĩa:
- `freeze_embeddings_after_max_snapshots`: sau khi đã lưu đủ 3 snapshot tốt nhất, track vẫn sống nhưng embedding pipeline ngừng ghi thêm bản mới.
- `metadata_heartbeat_sec`: nếu người đứng rất lâu, hệ thống chỉ update sự kiện tồn tại theo chu kỳ 30 giây.
- `same_track_refresh_min_sec`: tránh việc một người đứng yên 20 giây đã được mở lại ghi thêm.
- `same_track_refresh_cosine_thresh`: chỉ khi appearance khác đủ nhiều so với snapshot đã lưu gần nhất mới cho phép refresh.

Đối với use case search mặc định, đây là hành vi nên dùng:
- **1 episode kéo dài**
- **ít snapshot**
- **nhiều cập nhật metadata, ít cập nhật vector**

Như vậy vừa tránh DB phình ra, vừa không làm top-k bị lặp vô ích bởi một người đứng lâu trước camera.

### 6.4 Chính sách tracking

Tracking rất quan trọng vì nó là lớp chống duplication đầu tiên.

Board nên lưu cho mỗi track:
- `track_id`
- `camera_id`
- `episode_id`
- `start_ts`
- `end_ts`
- `best_bbox`
- `num_frames_seen`
- `last_saved_at`
- `suppressed_frame_count`
- `best_crop_path`
- `embedding_status`

Track kết thúc khi:
- biến mất quá `M` frame
- timeout
- hoặc camera stream reset

Sau khi track kết thúc, board có thể:
- finalize embedding
- gửi 1 event summary lên backend

### 6.5 Image encoder runtime

Image encoder là workload nặng nhất trên board sau detector.

Thiết kế runtime nên có:
- `preferred`: HTP/DSP nếu compile thành công
- `fallback`: GPU
- `fallback cuối`: ONNX Runtime CPU

Ứng dụng trên board nên có cấu hình:

```yaml
runtime_priority:
  - htp
  - gpu
  - cpu
max_inflight_tracks: 16
snapshot_interval_sec: 3
```

### 6.6 Local spool trên board

Nên dùng local disk thay vì RAM để buffer:

```text
/home/ubuntu/sigm/Lam/runtime_spool/
  pending/
    2026-04-26/
      event_0001.json
      event_0001.jpg
  sent/
  failed/
```

Mỗi event JSON nên chứa:
- metadata
- đường dẫn ảnh local
- checksum
- retry_count
- created_at

---

## 7. Text embedding ngoài board

### 7.1 Yêu cầu bắt buộc

Text service phải dùng:
- cùng tokenizer
- cùng text encoder weights
- cùng preprocessing
- cùng normalization

với image encoder đã dùng trên board.

Nếu board đang dùng checkpoint `epoch=56-val_score=52.28.ckpt` sau khi merge/export, thì text service phải export từ đúng checkpoint đó.

### 7.2 Tại sao không chạy text trên board

- Board chỉ có ~4 GB RAM khả dụng.
- Vision encoder đã chiếm đáng kể RAM và compute budget.
- Text encoder multilingual có embedding table rất lớn.
- Search query từ user có thể đến bất kỳ lúc nào, dễ chồng lên image path và gây peak memory.

### 7.3 Cách triển khai text service

### Prototype

- Host text encoder ở ngoài board.
- Expose 1 API đơn giản:

```http
POST /embed/text
```

Input:

```json
{
  "text": "người mặc áo đỏ, đeo balo đen"
}
```

Output:

```json
{
  "embedding": [0.0123, -0.0421, "..."],
  "dim": 768,
  "model_version": "msiglip-vn3k-seed2400-epoch56"
}
```

### Production

Khi query traffic tăng, text service nên tách thành container riêng với:
- request queue
- warm process
- model versioning
- health check

---

## 8. Vector database và nơi lưu dữ liệu

### 8.1 Không nên lưu vector chính trên board

Lưu vector DB chính trên board có các vấn đề:
- chiếm RAM / disk ngay trên thiết bị inference
- khó mở rộng khi số lượng track tăng
- khó backup
- làm board kiêm quá nhiều vai trò
- rủi ro mất toàn bộ dữ liệu khi board lỗi

Board chỉ nên giữ:
- cache ngắn hạn
- spool tạm khi mất mạng
- log

### 8.2 Các lớp dữ liệu cần tồn tại

Hệ thống thực tế cần 3 lớp dữ liệu tách biệt:

1. **Vector store**
   - lưu embedding để search ANN / cosine similarity

2. **Metadata DB**
   - lưu camera, track, thời gian, bbox, score, trạng thái

3. **Object storage**
   - lưu crop ảnh / thumbnail / debug artifact

### 8.3 Lựa chọn khuyến nghị cho prototype

### Phương án 1: Supabase all-in-one

Thành phần:
- Postgres
- `pgvector`
- Auth
- Storage
- SQL/RPC

Ưu điểm:
- một nền tảng cho cả metadata + vector + storage
- thuận tiện cho prototype
- query lọc theo `camera_id`, `time range` dễ hơn vector DB thuần
- free plan hiện có **2 free projects**

Nhược điểm:
- scale ANN lớn không linh hoạt bằng vector DB chuyên dụng
- tuning index / performance sâu không mạnh bằng Qdrant chuyên biệt

### Phương án 2: Qdrant Cloud + Postgres/Supabase

Thành phần:
- Qdrant cho vector search
- Postgres/Supabase cho metadata
- object storage riêng

Ưu điểm:
- vector search chuyên dụng
- free cluster theo docs hiện hỗ trợ khoảng **1M vectors 768-d** ở free tier cấu hình nhỏ

Nhược điểm:
- thêm 1 hệ thống cần vận hành
- join metadata phức tạp hơn
- prototype phức tạp hơn phương án all-in-one

### Phương án 3: Self-host trên board

Không khuyến nghị, trừ khi chỉ làm demo rất nhỏ, offline, ngắn hạn.

### 8.4 Khuyến nghị chốt

### Prototype giai đoạn 1

- **Supabase** cho:
  - metadata DB
  - `pgvector`
  - object storage

### Giai đoạn scale hơn

- chuyển vector search sang **Qdrant**
- giữ metadata và auth ở Postgres/Supabase

---

## 9. Thiết kế schema dữ liệu

### 9.1 Bảng `cameras`

| Field | Type | Mô tả |
|---|---|---|
| `id` | uuid | camera id |
| `name` | text | tên camera |
| `location` | text | vị trí |
| `stream_uri` | text | URI stream nội bộ |
| `board_id` | text | board đang đọc camera |
| `status` | text | online/offline |
| `created_at` | timestamptz | thời gian tạo |

### 9.2 Bảng `tracks`

| Field | Type | Mô tả |
|---|---|---|
| `id` | uuid | track record id |
| `camera_id` | uuid | FK tới cameras |
| `board_track_id` | text | track id sinh tại board |
| `episode_id` | uuid | nhóm các track bị đứt nhưng vẫn được coi là cùng một lần xuất hiện |
| `start_ts` | timestamptz | lúc bắt đầu xuất hiện |
| `end_ts` | timestamptz | lúc kết thúc |
| `best_frame_url` | text | thumbnail / crop tốt nhất |
| `snapshot_count` | int | số snapshot đã lưu |
| `suppressed_frame_count` | int | số frame / candidate bị bỏ qua do suppression policy |
| `last_saved_at` | timestamptz | lần gần nhất track được phép lưu snapshot |
| `status` | text | active/finalized |
| `created_at` | timestamptz | thời gian tạo |

### 9.3 Bảng `track_embeddings`

| Field | Type | Mô tả |
|---|---|---|
| `id` | uuid | embedding row id |
| `track_id` | uuid | FK tới tracks |
| `camera_id` | uuid | denormalized filter key |
| `captured_at` | timestamptz | thời điểm snapshot |
| `embedding` | vector(768) | image embedding |
| `quality_score` | float | điểm chọn frame |
| `bbox_x1` | int | bbox |
| `bbox_y1` | int | bbox |
| `bbox_x2` | int | bbox |
| `bbox_y2` | int | bbox |
| `crop_url` | text | ảnh crop |
| `save_reason` | text | lý do lưu: `best_frame`, `periodic`, `track_end`, `episode_reopen` |
| `model_version` | text | version của image encoder |
| `created_at` | timestamptz | thời gian ingest |

### 9.4 Bảng `search_queries`

| Field | Type | Mô tả |
|---|---|---|
| `id` | uuid | query id |
| `user_id` | uuid | người dùng |
| `query_text` | text | mô tả nhập vào |
| `camera_filter` | jsonb | filter camera |
| `time_range` | tstzrange | filter thời gian |
| `top_k` | int | số kết quả yêu cầu |
| `created_at` | timestamptz | thời gian truy vấn |

### 9.5 Index khuyến nghị

Cho `pgvector`:

```sql
create index on track_embeddings
using hnsw (embedding vector_cosine_ops);
```

Index metadata:

```sql
create index idx_tracks_camera_time
on track_embeddings (camera_id, captured_at desc);
```

---

## 10. Chiến lược lưu vector để tránh phình dữ liệu

### 10.1 Không lưu mọi frame

Nếu camera 15 FPS và mỗi người xuất hiện 10 giây:
- 1 người = 150 frame
- 10 người/giờ/camera = 1,500 embeddings/giờ nếu lưu mọi frame
- vài camera là DB tăng cực nhanh và đa số vector trùng lặp

### 10.2 Đơn vị lưu đúng hơn: track-level event

Thay vì "mỗi frame = 1 vector", nên xem dữ liệu là:
- **một track người trong camera**
- có 1 hoặc vài snapshot đại diện

### 10.3 Hai phương án lưu

### Phương án A: 1 vector đại diện / track

Lưu:
- embedding aggregate
- 1 best crop
- metadata của cả track

Ưu điểm:
- DB gọn
- query nhanh

Nhược điểm:
- mất đa dạng pose/appearance trong cùng track

### Phương án B: nhiều snapshot / track

Lưu:
- 3 đến 5 snapshot cho mỗi track
- query trên tất cả snapshot
- post-process gộp kết quả theo `episode_id` (hoặc tối thiểu `track_id` nếu chưa có episode grouping)

Ưu điểm:
- recall tốt hơn
- bền vững hơn với pose / blur / occlusion

Nhược điểm:
- DB lớn hơn

### Khuyến nghị

Prototype nên dùng:
- **3 snapshot / track**
- vector search trên snapshot
- UI gộp kết quả theo `episode_id`

Đây là điểm cân bằng tốt giữa recall và kích thước DB.

Nếu muốn giảm duplicate mạnh hơn nữa, có thể nâng đơn vị hiển thị từ `track_id` lên `episode_id`, trong đó nhiều track bị đứt ngắn hạn của cùng một người trong cùng camera sẽ được gộp thành một episode.

#### Khuyến nghị chốt

- **Index/search unit:** `track_embeddings`
- **Business/view unit:** `episode_id`
- **Top 10 hiển thị cho user:** top 10 episode, không phải top 10 raw snapshot

Đây là phương án tốt nhất cho MVP vì:
- không yêu cầu tracker hoàn hảo tuyệt đối
- vẫn giữ recall nhờ search trên snapshot
- tránh UI bị lặp kết quả của cùng một người
- cho phép mở chế độ forensic sau này mà không phải đổi cách lưu dữ liệu

---

## 11. API contracts khuyến nghị

### 11.1 Board -> Backend ingest

```http
POST /api/v1/ingest/track-embedding
Authorization: Bearer <board_token>
Content-Type: application/json
```

Body:

```json
{
  "board_id": "qc-rb3g2",
  "camera_id": "cam-lab-01",
  "track_id": "trk-1842",
  "captured_at": "2026-04-26T10:14:23Z",
  "bbox": [120, 48, 260, 410],
  "quality_score": 0.91,
  "embedding": [0.01, -0.02, "..."],
  "crop_upload_key": "tracks/2026-04-26/cam-lab-01/trk-1842_01.jpg",
  "model_version": "msiglip-vn3k-seed2400-epoch56"
}
```

Response:

```json
{
  "status": "ok",
  "embedding_id": "8b5d6e8d-6e9d-40e3-9473-47fd4cb2e8d1"
}
```

### 11.2 Web UI -> Search API

```http
POST /api/v1/search/text
Authorization: Bearer <user_token>
Content-Type: application/json
```

Body:

```json
{
  "query_text": "người mặc áo đỏ, đội nón trắng, đeo balo đen",
  "camera_ids": ["cam-lab-01", "cam-gate-02"],
  "start_time": "2026-04-26T00:00:00Z",
  "end_time": "2026-04-26T23:59:59Z",
  "top_k": 20
}
```

Response:

```json
{
  "query_id": "6a62b0f1-10a2-4e3d-9b8e-d21800ef7a64",
  "results": [
    {
      "episode_id": "eps-9b2d",
      "track_id": "trk-1842",
      "score": 0.8421,
      "snapshot_count": 3,
      "camera_id": "cam-lab-01",
      "first_seen": "2026-04-26T10:14:18Z",
      "captured_at": "2026-04-26T10:14:23Z",
      "crop_url": "https://.../trk-1842_01.jpg"
    }
  ]
}
```

### 11.3 Health API

Board nên gửi heartbeat định kỳ:

```http
POST /api/v1/boards/heartbeat
```

Payload:
- CPU usage
- RAM usage
- encoder runtime đang dùng
- upload backlog
- camera status

---

## 12. Hosting chiến lược cho prototype

### 12.1 Web UI

### Khuyến nghị

- Host Web UI bằng **Vercel Hobby** cho prototype cá nhân / demo.

### Lý do

- triển khai Next.js rất nhanh
- HTTPS sẵn
- preview deployment thuận tiện
- phù hợp cho frontend mỏng và API nhẹ

### Cảnh báo

Theo tài liệu chính thức của Vercel tại thời điểm đọc:
- Hobby là **free**
- nhưng dành cho **personal, non-commercial use**
- có quota giới hạn theo chu kỳ sử dụng

Vì vậy Vercel Hobby phù hợp để demo, chưa phải câu trả lời production cuối cùng.

### 12.2 Data layer

### Khuyến nghị

- **Supabase** cho prototype end-to-end đầu tiên.

### Lý do

- có `pgvector`
- có storage
- có auth
- dễ quản lý metadata bằng SQL
- free plan có **2 free projects**

### 12.3 Text embedding service

### Prototype

- có thể host trên **Hugging Face Spaces CPU Basic**

### Lưu ý

- free CPU Basic hiện cho 2 vCPU / 16 GB RAM / 50 GB disk
- free Space có thể **sleep khi không dùng**

Điều này chấp nhận được cho prototype, nhưng nếu muốn query luôn nóng thì cần dịch vụ không sleep.

### 12.4 Vector DB chuyên dụng

Nếu cần tách vector khỏi Postgres:
- **Qdrant Cloud** là lựa chọn hợp lý cho prototype tiếp theo
- free tier theo docs hiện là 1 node, 0.5 vCPU, 1 GB RAM, 4 GB disk

---

## 13. Bảo mật và networking

### 13.1 Nguyên tắc kết nối

Nguyên tắc nên là:
- board **không** nhận user traffic public trực tiếp
- board chỉ mở outbound HTTPS đến backend
- web UI chỉ gọi backend public

### 13.2 Tunnel public

Tunnel hiện tại rất hữu ích cho:
- SSH / terminal web
- debug
- monitoring
- manual demo nội bộ

Nhưng không nên coi tunnel là API public chính cho sản phẩm.

### 13.3 Auth

Nên có ít nhất 2 lớp token:
- `board_token` cho ingest
- `user_token` cho Web UI

Nếu dùng Supabase:
- user auth có thể đi qua Supabase Auth
- board dùng service token riêng, rotate định kỳ

### 13.4 Bảo vệ dữ liệu ảnh

- Crop URL nên là signed URL hoặc object storage private + proxy qua backend
- Không nên để bucket public hoàn toàn nếu ảnh chứa dữ liệu nhạy cảm

---

## 14. Tính toán tài nguyên và tránh OOM

### 14.1 Phân bổ trách nhiệm theo tài nguyên

Board:
- detect
- track
- image embedding
- upload

Cloud:
- text embedding
- vector search
- metadata query
- UI

Tách như vậy giúp board giữ RAM cho luồng hình ảnh liên tục.

### 14.2 Các điểm dễ gây OOM trên board

1. Giữ quá nhiều frame trong RAM.
2. Chạy detector + image encoder + text encoder cùng lúc.
3. Không giới hạn queue.
4. Retry upload bằng cách giữ payload trong memory thay vì disk.
5. Dùng batch lớn.

### 14.3 Biện pháp kỹ thuật

- queue bounded
- batch = 1 cho realtime pipeline
- local spool trên disk
- chỉ giữ thumbnail / crop cần thiết
- drop frame cũ khi quá tải
- tách process thay vì 1 tiến trình monolithic

---

## 15. MVP roadmap đề xuất

## Phase 1: Single camera, single board, offline-ish demo

Mục tiêu:
- camera -> detect -> crop -> image embedding trên board
- upload vector lên backend
- Web UI search text

Phạm vi:
- 1 camera
- 1 board
- 1 text service ngoài board
- 1 vector DB ngoài board

### Phase 1 implementation scaffold hiện tại

Repo hiện có scaffold demo tại `deployment/demo/` để bắt đầu nối các module theo hướng plug-in:

```text
deployment/demo/
  core/       # contracts, shared utilities, pipeline orchestration
  adapters/   # source/detector/tracker/encoder/store/spool/uploader implementations
  cli/        # real command implementations
  tests/      # local preflight tests
  run_*.py    # compatibility wrappers for python -m deployment.demo.*
```

Pipeline runtime:

```text
FrameSource
  -> PersonDetector
  -> Tracker
  -> CropSelector
  -> ImageEncoder
  -> DiskSpool
  -> Uploader / VectorStore
```

Các adapter v1:
- `ImageDirectorySource` / `VideoFileSource`: dùng ảnh hoặc video file để phát triển lặp lại trước khi nối camera USB/IP.
- `FullFramePersonDetector`: coi ảnh crop VN3K là một người, phục vụ smoke test và demo dữ liệu crop.
- `SimpleTracker`: sinh `track_id` / `episode_id` deterministic.
- `DefaultCropSelector`: giữ tối đa 3 snapshot mỗi track, không embed mọi frame.
- `QnnVisionEncoder`: đường chạy thật trên RB3, dùng `qnn-net-run` với `vision_encoder.bin`.
- `OnnxVisionEncoder` và `FakeVisionEncoder`: chỉ dùng cho preflight/local wiring, không phải tiêu chí deploy.
- `JsonlVectorStore`, `DiskSpool`, `LocalVectorStoreUploader`: backend local tạm thời để kiểm thử ingest/search/retry trước khi thay bằng Supabase hoặc API thật.

CLI chính vẫn giữ dạng ngắn qua wrapper ở package root:

```bash
python -m deployment.demo.run_ingest \
  --source /path/to/images_or_video \
  --encoder qnn \
  --vision-bin vision_encoder.bin \
  --htp-config deployment/config/qnn/htp_config_245.json \
  --board-id qc-rb3g2 \
  --camera-id cam-lab-01

python -m deployment.demo.run_search \
  --query "người mặc áo đỏ" \
  --store artifacts/deployment/runtime/vectors.jsonl

python -m deployment.demo.health
```

**Local preflight không được tính là deploy thành công.** Local fake/ONNX chỉ kiểm tra interface, spool, JSONL vector store và result collapsing. Acceptance thật của Phase 1 phải chạy trên RB3 bằng QNN `qnn-net-run`, output 768 chiều finite, L2-normalized, có latency/FPS đo trên board.

## Phase 2: Stable remote demo

Bổ sung:
- local spool
- retry upload
- health dashboard
- signed URLs
- track aggregation tốt hơn

## Phase 3: Multi-camera / multi-board

Bổ sung:
- nhiều `camera_id`
- nhiều `board_id`
- backend phân quyền theo site/location
- monitoring tập trung

## Phase 4: Production hardening

Bổ sung:
- dedicated text service
- vector DB scale riêng
- retention policy
- audit logs
- alerting

---

## 16. Rủi ro kỹ thuật và cách giảm thiểu

| Rủi ro | Tác động | Giảm thiểu |
|---|---|---|
| Board OOM khi chạy nhiều stage cùng lúc | Pipeline chết | Chỉ giữ image path trên board, bounded queue, text embedding remote |
| DB phình quá nhanh do lưu mọi frame | Query chậm, tốn tiền | Track-based dedup, top-N snapshots |
| Text encoder remote dùng sai checkpoint | Search sai hoàn toàn | Version lock giữa image/text encoder |
| Internet chập chờn | Mất dữ liệu | Local spool + retry |
| Vercel/HF free tier ngủ hoặc chạm quota | Demo không ổn định | Chấp nhận cho prototype, chuẩn bị phương án nâng cấp |
| Board public trực tiếp | Rủi ro bảo mật | Chỉ cho outbound, UI qua backend |

---

## 17. Khuyến nghị chốt cho prototype đầu tiên

### Kiến trúc

- **Board RB3 Gen2**
  - camera ingest
  - person detect + track
  - crop selection
  - image embedding
  - upload + local spool

- **Supabase**
  - metadata DB
  - `pgvector`
  - object storage
  - auth

- **Web UI trên Vercel**
  - Next.js UI
  - search form
  - result gallery

- **Text embedding service ngoài board**
  - đúng checkpoint mSigLIP
  - prototype có thể host trên Hugging Face Space

### Lưu vector

- Không lưu mọi frame
- Lưu **3 snapshot / track**
- Search trên snapshot
- Gộp theo `episode_id` ở backend/UI

### Networking

- Board outbound only
- Tunnel chỉ cho admin

Đây là kiến trúc cân bằng tốt nhất giữa:
- giới hạn RAM của board
- độ đơn giản khi triển khai
- khả năng demo sớm
- đường nâng cấp về sau

---

## 18. Tài liệu tham chiếu

- Board và trạng thái runtime: `deployment/docs/system.md`
- Kế hoạch deploy model: `deployment/docs/deployment-plan.md`
- Benchmark proxy models: `deployment/docs/benchmark-rp.md`
- AI Hub compile learnings: `deployment/docs/aihub-experiments.md`

### Tài liệu chính thức đã tham chiếu cho lựa chọn hosting/vector

- Vercel Hobby plan: <https://vercel.com/docs/plans/hobby>
- Vercel account plans: <https://vercel.com/docs/plans>
- Supabase billing: <https://supabase.com/docs/guides/platform/billing-on-supabase>
- Supabase pgvector: <https://supabase.com/docs/guides/database/extensions/pgvector>
- Supabase vector indexes: <https://supabase.com/docs/guides/ai/vector-indexes>
- Supabase HNSW indexes: <https://supabase.com/docs/guides/ai/vector-indexes/hnsw-indexes>
- Hugging Face Spaces overview: <https://huggingface.co/docs/hub/en/spaces-overview>
- Qdrant Cloud pricing: <https://qdrant.tech/pricing/>
- Qdrant free cluster details: <https://qdrant.tech/documentation/cloud/create-cluster/>

> Các mức free tier và quota có thể thay đổi. Trước khi triển khai thật hoặc báo cáo chính thức, cần kiểm tra lại trang pricing hiện hành.
