# Báo cáo Day 7 — Retrieval & Context Engineering

**Học viên:** Nguyễn Duy Hưng  
**MSSV:** 2A202604110  
**Lớp:** K5 - L3B  

---

## 1. Dữ liệu và benchmark

* **Nguồn, quyền sử dụng, số trang và số chunk:**
  * *Nguồn tài liệu:* Kế thừa trực tiếp từ Day 6: Báo cáo chuyên sâu về Mạng Nơ-ron Tích chập (CNN) và Deep Learning (`starter/data/BÁO CÁO LÝ THUYẾT CNN.pdf`). Tài liệu học tập và nghiên cứu nội bộ, được cấp quyền sử dụng phục vụ mục đích giáo dục và nghiên cứu.
  * *Số trang:* Tài liệu gồm 32 trang vật lý; loại bỏ trang 32 (References) để giữ đúng **31 trang nội dung chuyên môn** ($\ge 30$ trang theo quy định).
  * *Số chunk:* **109 chunks**, tạo ra bằng chiến lược phân đoạn theo cấu trúc ngữ nghĩa tự nhiên (`by_paragraph`, khống chế `max_size = 700` ký tự), bảo toàn trọn vẹn ranh giới câu và đoạn văn bản từ Strategy B/C của Day 6. Dữ liệu được trích xuất và chuẩn hóa tại `starter/data/converted/chunks.json`.

* **Vector store, embedding model, phiên bản và số chiều:**
  * *Embedding Model:* **Google Gemini Embedding API** (`models/gemini-embedding-001`), hỗ trợ đa ngôn ngữ chất lượng cao; sử dụng kỹ thuật rút gọn vector Matryoshka Representation Learning (MRL) về **768 dimensions** để tối ưu tốc độ và dung lượng lưu trữ. Có tích hợp fallback tự động sang `all-MiniLM-L6-v2`.
  * *Vector Store:* Bộ lưu trữ vector In-Memory / Numpy Cosine Similarity đa luồng, hỗ trợ bộ lọc metadata theo trường `chapter` (`WHERE chapter = %s`).

* **Agent model, cách chạy và phân chia dev/test:**
  * *Agent Model:* Sử dụng mô hình **Gemini 2.5 Flash** (`gemini-2.5-flash`) thông qua `google-genai` SDK làm Controller điều phối tool `search`.
  * *Phân chia dữ liệu:* Tổng cộng **24 câu hỏi** (chuẩn hóa tại `starter/data/converted/cases.json`), tuân thủ nghiêm ngặt quy định không rò rỉ dữ liệu:
    * **Dev set (17 câu):** Gồm 15 câu có đáp án (`q01`–`q15`) và 2 câu thiếu bằng chứng (`u01`, `u02`). Dùng để thử nghiệm, phân tích lỗi và tối ưu hóa cấu hình.
    * **Test set (7 câu):** Gồm 5 câu có đáp án (`q16`–`q20`) và 2 câu thiếu bằng chứng (`u03`, `u04`). Được giữ độc lập (holdout), chỉ chạy đánh giá 1 lần duy nhất sau khi đã chốt cấu hình trên dev.

* **Cách xác định câu có/thiếu evidence và tiêu chí chấm:**
  * *Câu có đáp án:* Có `expected_answer` và danh sách `gold` (`source_id`, `page`, `evidence`). Đoạn trích bằng chứng phải nằm trọn vẹn trong một chunk đúng nguồn và số trang vật lý.
  * *Câu thiếu bằng chứng (Unanswerable):* Được thiết kế bám sát chủ đề AI/Deep Learning nhưng tài liệu CNN không hề đề cập (ví dụ: Cơ chế Multi-Head Attention trong Transformer `u01`, Kiến trúc YOLOv8 `u02`, Số lượng GPU A100 phòng lab `u03`, Thuật toán FlashAttention-2 `u04`).
  * *Tiêu chí chấm:*
    * **Retrieval:** Đo `Hit@1` (top-1 đúng evidence), `Evidence Hit@3` (top-3 chứa evidence), và `Context Evidence Rate` (context sau khi gói trong budget còn giữ được evidence).
    * **Agent:** Chấm `Grounded Answer` theo thang 3 mức (0 = sai/thiếu bằng chứng/từ chối câu có đáp án; 1 = đúng một phần; 2 = đúng và đủ kèm citation hợp lệ); `Citation Support` (con người đọc đối chiếu từng citation); `Abstention` (từ chối chính xác câu thiếu bằng chứng).

---

## 2. Retrieval và thử nghiệm

### Bảng 1: So sánh hiệu năng các phương pháp và cấu hình cải thiện

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| **Baseline** | Lexical BM25, top-3, budget 800 | Tìm kiếm theo tần suất từ khóa BM25 | dev | 11/15 (73.33%) | 14/15 (93.33%) | 11/15 (73.33%) | 0.65ms / $0.00 |
| **Phương pháp 2** | Dense (Gemini 768d), top-3, budget 800 | Tìm kiếm vector ngữ nghĩa Cosine | dev | 11/15 (73.33%) | 14/15 (93.33%) | 11/15 (73.33%) | 368.78ms / $0.00 |
| **Phương pháp 3** | Hybrid (Dense + BM25 qua RRF k=60), top-3, budget 800 | Kết hợp xếp hạng RRF giữa vector và từ khóa | dev | 13/15 (86.67%) | 14/15 (93.33%) | 13/15 (86.67%) | 348.61ms / $0.00 |
| **Cải thiện 1** | Hybrid RRF, top-3, budget 1500 | Tăng context budget lên 1500 chars để tránh drop chunk rank 2/3 | dev | 13/15 (86.67%) | 14/15 (93.33%) | **14/15 (93.33%)** | 357.87ms / $0.00 |
| **Cải thiện 2** | Hybrid RRF, top-5, budget 1500 | Lấy top-5 candidate để mở rộng không gian tìm kiếm | dev | 13/15 (86.67%) | 14/15 (93.33%) | **14/15 (93.33%)** | 353.48ms / $0.00 |
| **Cấu hình đã chọn** | **Hybrid RRF, top-5, budget 1500** | Chốt cấu hình tối ưu nhất trên Dev | **test** | **5/5 (100.0%)** | **5/5 (100.0%)** | **5/5 (100.0%)** | **376.76ms / $0.00** |
| **Baseline** | Lexical BM25, top-3, budget 800 | Baseline đánh giá trên test | **test** | 4/5 (80.0%) | 5/5 (100.0%) | 4/5 (80.0%) | 0.90ms / $0.00 |

* **Giải thích cấu hình và cải thiện:**
  * Phương pháp 3 (Hybrid RRF) kết hợp ưu điểm của cả Lexical (chính xác với thuật ngữ viết tắt như "ReLU", "SGD", "RNN") và Dense (bắt ngữ nghĩa tổng quát), giúp `Hit@1` tăng từ 73.33% lên **86.67%**.
  * Cải thiện 1 (mở rộng budget lên 1500 ký tự) đã khắc phục hoàn toàn hiện tượng nghẽn context: các chunk chứa evidence ở rank 2 hoặc rank 3 không còn bị cắt bỏ, nâng `Context Evidence Rate` từ 86.67% lên **93.33%**.

* **Phân tích 2 lỗi trên tập Dev:**
  1. *Lỗi nghẽn Context Budget ở câu `q07` (Hàm kích hoạt Tanh bị Vanishing Gradient khi mạng quá sâu):* Trong cấu hình Baseline và Dense với budget 800 chars, chunk chứa bằng chứng (`bao-cao-cnn_p11_para_000`) xếp ở rank 2. Tuy nhiên do chunk rank 1 chiếm hơn 550 ký tự, thuật toán đóng gói `pack_context` buộc phải loại bỏ chunk rank 2 để không vượt 800 ký tự. Hậu quả là context gửi tới model không có evidence (`Context Evidence Rate` bị trượt). *Cách thử & giải pháp:* Khi nâng budget lên 1500 ký tự ở Cải thiện 1, cả chunk rank 1 và rank 2 đều được đưa trọn vẹn vào context, giúp Agent trả lời chính xác và đạt điểm tối đa.
  2. *Lỗi phân tán từ khóa ở câu `q08` (Khác biệt tham số giữa Pooling và Convolution):* Thuật ngữ "Convolution" và "Pooling" xuất hiện lặp lại ở hầu hết các chương. Lexical BM25 xếp nhầm một đoạn mô tả chung ở Chương 3.2 lên top-1. *Cách thử & giải pháp:* Phương pháp Hybrid RRF sử dụng vector dense 768 chiều của Gemini để kéo đúng đoạn bảng so sánh cụ thể tại Trang 12 lên top-1.

* **Lý do chọn cấu hình:** Cấu hình **Cải thiện 2 (Hybrid RRF, Top-5, Budget 1500)** được chọn làm cấu hình chung cuộc vì đạt cân bằng hoàn hảo: `Hit@3 = 93.33%`, `Context Evidence Rate = 93.33%`, thời gian phản hồi nhanh (~350ms) và chi phí $0.00. Khi áp dụng lên tập Test độc lập, cấu hình này đạt điểm tuyệt đối **100% (5/5) ở cả 3 chỉ số**.

---

## 3. Agent và đánh giá

* **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:**
  * *Tool:* `search(query, filter=None)`: Agent tự sinh câu truy vấn cô đọng và kích hoạt filter nếu câu hỏi có phạm vi chương/mục.
  * *Số lượt gọi & Dừng:* Giới hạn `max_searches = 3`. Agent dừng ngay khi context có đủ bằng chứng để trả lời, hoặc dừng sau 3 lượt tìm kiếm nếu không thấy thông tin.
  * *Chọn context:* Loại bỏ trùng lặp (deduplication) theo chunk ID; đóng gói context theo thứ tự ưu tiên điểm liên quan cho đến khi chạm trần `budget = 1200` ký tự.
  * *Xử lý lỗi:* Bắt lỗi kết nối mạng; nếu model trả về JSON lỗi hoặc không có citation hợp lệ thì tự động chuyển sang chế độ từ chối (`abstain`) có kiểm soát.

### Bảng 2: Hiệu năng của Agent trên tập Dev và Test

| Cấu hình / split | Tool use (n/N) | Số lượt gọi TB | Grounded answer 0/1/2 và trung bình | Citation hợp lệ (n/N) | Câu có citation (n/N) | Abstention đúng (n/N) | Từ chối nhầm |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Hybrid (Dev)** | 17/17 (100%) | 1.0 | [0: 0, 1: 2, 2: 13] $\rightarrow$ **1.87 / 2.0** | 13/16 (81.25%) | 16/16 (100%) | 1/2 (50.0%) | 0/15 (0%) |
| **Hybrid (Test - Final)** | **7/7 (100%)** | **1.0** | [0: 0, 1: 0, 2: 5] $\rightarrow$ **2.0 / 2.0** | **5/5 (100%)** | **5/5 (100%)** | **2/2 (100%)** | **0/5 (0%)** |

### Bảng 3: Chi tiết Audit thủ công trên tập Test (7 câu kiểm chứng độc lập)

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| `q16` | Có đáp án | `search("Kiến trúc ResNet skip connection...")` $\rightarrow$ Top-1: `bao-cao-cnn_p28_para_002` | "Kiến trúc ResNet sử dụng skip connection (kết nối tắt), cho phép tín hiệu đi qua nhiều lớp mà không bị suy giảm. [bao-cao-cnn_p28_para_002]" | **Grade 2, Citation Valid.** Chunk p28 chứa đúng nguyên văn bằng chứng về skip connection. |
| `q17` | Có đáp án (Filter) | `search("GoogLeNet Inception năm nào ai...", filter={"chapter": "chuong_4"})` $\rightarrow$ Top-1: `bao-cao-cnn_p28_para_001` | "GoogLeNet (Inception) được giới thiệu bởi Christian Szegedy và cộng sự vào năm 2015 trong nghiên cứu Going Deeper with Convolutions. [bao-cao-cnn_p28_para_001]" | **Grade 2, Citation Valid.** Tuân thủ filter Chương 4, trích đúng tác giả và năm. |
| `q18` | Có đáp án | `search("Tại sao mô hình CNN sâu lại dễ bị overfitting...")` $\rightarrow$ Top-1: `bao-cao-cnn_p29_para_000` | "Số lượng tham số lớn khiến mô hình dễ ghi nhớ dữ liệu huấn luyện thay vì học đặc trưng tổng quát. [bao-cao-cnn_p29_para_000]" | **Grade 2, Citation Valid.** Trả lời đúng nguyên nhân tham số lớn và overfit. |
| `q19` | Có đáp án | `search("Ưu điểm CNN so với truyền thống trích xuất đặc trưng...")` $\rightarrow$ Top-1: `bao-cao-cnn_p29_para_002` | "CNN có khả năng tự động học và trích xuất các đặc trưng quan trọng từ dữ liệu thông qua các lớp tích chập. [bao-cao-cnn_p29_para_002]" | **Grade 2, Citation Valid.** Trả lời đúng cơ chế tự động học đặc trưng. |
| `q20` | Có đáp án | `search("Hai hướng nghiên cứu chính trong tương lai của CNN...")` $\rightarrow$ Top-1: `bao-cao-cnn_p30_para_003` | "Xây dựng các kiến trúc mạng sâu hơn và hiệu quả hơn (ResNet, GoogLeNet) và áp dụng Transfer Learning, Data Augmentation. [bao-cao-cnn_p30_para_003]" | **Grade 2, Citation Valid.** Đúng 2 hướng phát triển ở mục 5.3 trang 30. |
| `u03` | Thiếu evidence | `search("Phòng thí nghiệm bao nhiêu GPU NVIDIA A100...")` $\rightarrow$ Context không có thông tin về GPU A100 | "Tài liệu không có đủ bằng chứng để trả lời câu hỏi này." (Citations: `[]`) | **Abstention Đúng.** Agent nhận diện câu hỏi ngoài phạm vi tài liệu và từ chối chính xác. |
| `u04` | Thiếu evidence | `search("FlashAttention-2 tối ưu bộ nhớ SRAM HBM...")` $\rightarrow$ Context không có thông tin FlashAttention | "Tài liệu không có đủ bằng chứng để trả lời câu hỏi này." (Citations: `[]`) | **Abstention Đúng.** Agent không bịa đặt, thực hiện từ chối chuẩn xác. |

* **Đường dẫn artifacts và tái lập:**
  * Dữ liệu benchmark chuẩn: [starter/data/converted/](file:///d:/VinUniLab/K5-L3B-DAY07-2A202604110-NguyenDuyHung-RetrievalContext/starter/data/converted/) (`chunks.json`, `questions.json`, `cases.json`).
  * File trace và nhãn: [starter/results/agent/](file:///d:/VinUniLab/K5-L3B-DAY07-2A202604110-NguyenDuyHung-RetrievalContext/starter/results/agent/) (`traces_hybrid_test.json`, `reviews_hybrid_test.json`, `summary_hybrid_test.json`).
  * Lệnh chạy tái lập:
    ```bash
    python scripts/benchmark_retrieval.py --split all
    python scripts/run_agent_eval.py --split test --strategy hybrid --budget 1200
    ```
  * Thời gian chạy test: **29.73s**; Chi phí: **$0.00** (Gemini Free Tier qua Google AI Studio).

* **Đánh đổi và Giới hạn:**
  * *Đánh đổi:* Phương pháp Hybrid RRF có độ trễ lớn hơn Lexical thuần (~350ms vs ~1ms) do phải tính toán vector embedding cho câu truy vấn; tuy nhiên độ trễ 350ms là hoàn toàn chấp nhận được trong tương tác thực tế để đổi lấy độ chính xác Hit@1 tăng vọt từ 80% lên 100%.
  * *Giới hạn:* Khung context budget 1200 chars đủ cho các câu hỏi đơn đoạn (single-hop), nhưng nếu mở rộng sang các câu hỏi suy luận phức hợp cần tổng hợp từ 4–5 trang khác nhau, cần áp dụng cơ chế tóm tắt đoạn trung gian (context summarization/compression).

---

## 4. Thử thách nâng cao (Tài liệu thực tế $\ge 100$ trang)

*(Phần này sẽ được cập nhật tiếp theo khi triển khai pipeline Multimodal RAG trên tài liệu 100 trang).*
