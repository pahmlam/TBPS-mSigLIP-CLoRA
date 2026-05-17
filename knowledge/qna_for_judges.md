# Q&A Chuẩn bị cho Ban Giám Khảo (Nghiên cứu mSigLIP & Circle Loss)

> Tài liệu này tổng hợp các câu hỏi hóc búa mang tính phản biện (defense) mà ban giám khảo có thể đặt ra khi chấm dự án, kèm theo dàn ý trả lời dựa trên mạch tư duy và thiết kế hệ thống của bạn.

---

## Phần 1: Động lực & Tư duy Cốt lõi (Tại sao lại là Circle Loss?)

### 1. Tại sao em lại sử dụng Circle Loss thay vì các hàm loss truyền thống cho học đối chiếu (Contrastive Loss) hay Triplet Loss?
**Phân tích tư duy của bạn:** 
Ban đầu, tư duy của bạn không phải là "chọn đại" một hàm Loss phức tạp để làm màu. Bạn xuất phát từ hai nhu cầu thực tế của quá trình tối ưu hóa:
1. **Tự điều chỉnh Gradient (Auto-adjust gradients) theo độ khó của sample:** Trong mSigLIP gốc, N-ITC (Sigmoid) đối xử khá "bao dung" với các mẫu. Phân tích toán học cho thấy gradient của N-ITC biến mất (vanishing) tại các vùng "semi-hard negatives". Triplet Loss thì lại quá phụ thuộc vào cặp khó nhất (hardest pair), dễ gãy nếu gặp nhiễu. Bạn cần một cơ chế tự động: mẫu nào đang sai lệch nhiều thì model phải tập trung gradient mạnh vào mẫu đó (phạt nặng hơn), mẫu nào dễ (đã phân tách tốt) thì giảm gradient lại.
2. **Hướng hội tụ cụ thể mang tính hình học:** Contrastive Loss thông thường có biên giới quyết định là một siêu mặt phẳng (tuyến tính, $\alpha_p = \alpha_n = 1$). Nhìn vào không gian nhúng, bạn muốn các mẫu cùng người (positives) hội tụ lại một cụm kín, và những người khác (negatives) bị đẩy văng ra xa tạo thành một không gian có ranh giới hình tròn (hay mặt cầu). Circle Loss cung cấp tham số $\Delta_p$ (1-margin) và $\Delta_n$ (margin) làm mốc ranh giới rõ ràng.

**Cách trả lời (Gợi ý):** 
*"Thưa BGK, tư duy ban đầu của em rất đơn thuần: em muốn một cơ chế tự điều chỉnh gradient dựa trên từng cặp mẫu riêng biệt (adaptive re-weighting), sao cho mẫu nào càng khó thì mức độ phạt càng nặng, đồng thời hàm loss phải có một hướng hội tụ cụ thể về mặt không gian thay vì để tuyến tính tự do.* 

*Khi nghiên cứu mSigLIP, em nhận ra N-ITC làm rất tốt việc học khái quát chống nhiễu (Soft labels), nhưng do dùng hàm Sigmoid nên đạo hàm biên bị triệt tiêu sớm, dẫn đến thiếu nhạy bén với các chi tiết khác biệt nhỏ (Hard negatives). Ngược lại, Circle Loss cung cấp một mặt cầu hội tụ rõ ràng với các hệ số phạt mềm dẻo $\alpha_p$ và $\alpha_n$ tự động mở rộng gradient theo mức độ sai lệch của model. Vì vậy em chọn Circle loss làm 'mũi nhọn' đục vào vùng Hard Negatives."*

### 2. Có ý kiến cho rằng "N-ITC sẽ có mức phạt ngang nhau cho cả mẫu khó và mẫu dễ", điều đó có đúng không? Sự bù khuyết của Circle Loss ở đây là gì?
**Phân tích bảo vệ (Defense):** Dùng câu này để thể hiện kiến thức giải tích hàm của bạn. Nhận định "phạt ngang nhau giữa mẫu khó và dễ" là chưa chuẩn xác, phải nói đúng hàm ý là "bão hòa mức phạt giữa mẫu KHÁ KHÓ và mẫu CỰC KỲ KHÓ".
**Cách trả lời:** 
*"Thưa hội đồng, nhận định phạt ngang nhau cho mẫu 'khó' và 'dễ' là chưa chính xác hoàn toàn về mặt toán học, nhưng nó đã nhắm đúng vào điểm mù kiến trúc của N-ITC. Thực chất, N-ITC vẫn phân biệt được mẫu dễ (luôn cho mức phạt tiến về 0). Tuy nhiên, vì tối ưu trên hàm Sigmoid, đạo hàm sẽ bão hòa và nằm bẹt thành một đường trần (plateau) khi tiến về đuôi. Do đó, N-ITC đối xử với một mẫu khá khó (độ tương đồng 0.5) với một mức phạt y hệt một mẫu cực kỳ khó (độ tương đồng 0.9). Nó gọi là bão hòa mức phạt (vanishing gradient).

Ngược lại, Circle Loss đóng vai trò 'bàn tay sắt'. Với cơ chế `LogSumExp` và hệ số mở rộng $\alpha$, Circle Loss không bị bão hòa mà sẽ **nhân số mũ mức phạt (exponential penalty)**. Một điểm cực khó lọt vào sẽ làm hàm mũ phình to, chiếm lấy gần như 100% tỷ trọng gradient của cả nhánh, dồn toàn lực ép mô hình bóc tách điểm cứng đầu đó. Nhờ vậy, giới hạn phân biệt các distractors (người cực giống nhau) mới được phá vỡ."*

---

## Phần 2: Đánh đổi & Giải quyết rủi ro (Curriculum Learning)

### 3. Circle Loss tấn công mạnh vào Hard Negatives. Nhưng nhỡ sample khó đó thực chất là một nhãn bị sai (False Negative - hai bức ảnh của cùng một người nhưng ID sai) thì sao? Đẩy nó ra xa chẳng phải là phá hỏng mô hình à?
**Phân tích:** Giám khảo tinh ý sẽ thấy Hard mining + Label Noise = Catastrophe (Thảm họa triệt tiêu lặp).
**Cách trả lời:**
*"Đó là một rủi ro cực kỳ lớn do Label Noise và em hoàn toàn nhận thức được. Phương án giải quyết của em nằm ở 2 khía cạnh:

1. **Curriculum Learning (Bảo vệ ở mức độ thực tế triển khai):** 
   - **Epoch 0-5 (Giai đoạn hỗn loạn):** Ở những epoch đầu tiên, mô hình chưa nhận dạng được mục tiêu, không gian hình học rất lộn xộn. Lúc này, MỌI điểm dữ liệu trùng vào nhau đều có vẻ sai lệch và biến thành "Hard Negatives ảo". Nếu bật Circle Loss ngay với lực phạt hàm mũ khổng lồ, nó sẽ trừng phạt vô tội vạ, làm sụp đổ hoàn toàn quá trình hội tụ. Do đó, em khóa trọng số Circle = 0. Em dùng hàm N-ITC (bao dung và mềm mỏng) để gom các nhóm người có nét na ná nhau thành các cụm cơ bản (tạo 'bản lề').
   - **Epoch 6-20 (Giai đoạn tinh chỉnh):** Khi bản lề không gian đã vững, em mới dùng Curriculum tịnh tiến sức mạnh Circle Loss lên đà tối đa (0.1). Lúc này, những điểm nào còn găm chặt gây nhiễu thì chắc chắn mới là **Hard Negatives thực sự** (ví dụ 2 người cùng mặc áo trắng nhưng khác ID). Circle loss lúc này mới được thả xích để bóc tách chúng một cách tinh vi mà không sợ trừng phạt oan như giai đoạn không gian lộn xộn.

2. **Noise-Aware Circle Loss (Hướng giải quyết lý thuyết tận gốc cho tương lai):** Giải pháp lý thuyết lõi mà dự án đang xây dựng là tiêm (inject) thuật toán rà soát nhiễu vào nhánh âm của Circle Loss (dựa trên phân phối Bayesian). Khi một âm bản (Negative) bỗng dưng lọt thỏm vào vùng phân bố của hội Dương bản, nó sẽ bị nhận diện rủi ro mang nhãn sai False Negative $P(FN)$ rất cao. Mô hình sẽ dùng lượng $1-P(FN)$ để triệt tiêu trực tiếp hệ số hạt nhân $\alpha_n$ của Circle Loss. Nhờ vậy mô hình tự biết "nhún nhường" khi gặp nhãn bị gán sai thay vì đâm đầu vào dồn lực cắt đứt chúng."*

### 4. Tại sao lại dùng LoRA thay vì fine-tune toàn bộ mô hình (Full FT)? Có phải chỉ để tiết kiệm bộ nhớ phần cứng chạy Batch Size lớn?
**Phân tích:** Đừng chỉ nói LoRA là để đỡ tốn RAM. Nó mang ý nghĩa Tối ưu hóa Low-Resource.
**Cách trả lời:**
*"Viết lại toàn bộ trọng số (Full FT) cần GPU lớn là một hạn chế vật lý, nhưng lý do chính mang tính khoa học là **kiểm soát Overfitting trên low-resource data**. Dữ liệu VN3K khá nhỏ. Nếu em fine-tune toàn bộ 376 triệu tham số, mô hình sẽ bị 'quên' (Catastrophic Forgetting) những gì nó đã học trên hàng tỷ ảnh đa ngôn ngữ trước đó, và ghi nhớ vẹt (overfit) vào các chi tiết thừa của tập VN3K. Nhờ dùng LoRA đóng băng 98% mô hình gốc lại và chỉ huấn luyện 1.57% tham số mới, em có thể 'dạy' thêm cho mô hình kỹ năng tìm người (TBPS) mà vẫn giữ nguyên được bộ não hiểu biết chống nhiễu khổng lồ ban đầu của mSigLIP để nó không bị sụp đổ."*

### 5. Có đánh giá thiên lệch không khi batch size của LoRA quá khác batch size của Full Fine-tuning ở baseline? Lỡ do batch size lớn nên kết quả tốt thì sao?
**Cách trả lời:** 
*"Thưa BGK, để loại trừ nguyên nhân ảo này, em đã làm ablation study thực tế (Nêu trong bảng Experiment Summary). Em ép chạy Full FT với hệ thống Accumulate Grad batches để batch size thực tế lên 24 (bằng với quá trình train LoRA). Kết quả R@1 của Full FT trượt giảm (từ 49.70% xuống 49.18%). Điều này là minh chứng rằng Batch Size không phải là nguyên nhân đẻ ra Accuracy 52%+, mà chính tính kiềm chế Overfitting của mạng LoRA kết hợp Curriculum Circle Loss mới là nhân tố giải bài toán low-resource này."*

---

## Phần 3: Kiến trúc Hệ thống Toàn cảnh

### 6. Dùng song song 4 loss (N-ITC, Circle, CITC, SimCLR) có khiến quá trình học bị kéo giật ngược hay giẫm đạp lên biên gradient của nhau không?
**Cách trả lời:** 
*"Bốn hàm Loss trong kiến trúc là sự cộng hưởng, gánh lỗi bổ trợ cho nhau:
- **SimCLR** giữ ổn định thị giác (hai ảnh augment của cùng 1 người phải không đổi).
- **N-ITC (Sigmoid)** giăng lưới lớn định vị toàn cục một cách chung chung và từ chối nhiễu cục bộ.
- **CITC** làm chốt kiểm ngang Inter/Intra Modal bảo vệ độ đồng bộ chéo.
- Cuối cùng, **Circle Loss** như một bàn tay tiểu phẫu đi vào cái nền lưới đã được 3 thuật toán trên chăng cứng, búng mạnh các điểm cực khó, cắt gọt không gian thành ranh giới phân lớp rõ ràng. Do có Curriculum chờ 5 epochs cho các Loss nền ổn định xong mới tung Circle Loss ra nên mô hình không giẫm đạp gradient mà phân công hội tụ đồng điệu."*

---

## Phần 4: Thách thức Dữ liệu (Dataset & Low-Resource)

### 7. Đâu là câu mô tả bao quát nhất về rào cản của nguồn dữ liệu trong bài toán này?
**Cách trả lời / Tuyên bố (Statement):**
*"Thưa BGK, thách thức cốt lõi về nguồn dữ liệu không chỉ nằm ở sự **khan hiếm của ngữ liệu Tiếng Việt (Low-resource)**, mà cực đoan hơn là **tính mập mờ ẩn giấu của nhãn (Label Noise / Ambiguity)** và **sự tương đồng khốc liệt về trang phục trên đường phố (Hard Distractors)**. Chính môi trường nhiễu này khiến một mô hình khổng lồ rất dễ sụp đổ vì 'học vẹt' nhãn sai, đòi hỏi kiến trúc phải có sự đánh đổi chính xác qua nếp lọc LoRA."*
