# Ngày 1 — Bài Tập & Phản Ánh
## Nền Tảng LLM API | Phiếu Thực Hành

**Thời lượng:** 1:30 giờ  
**Cấu trúc:** Lập trình cốt lõi (60 phút) → Bài tập mở rộng (30 phút)

---

## Phần 1 — Lập Trình Cốt Lõi (0:00–1:00)

Chạy các ví dụ trong Google Colab tại: https://colab.research.google.com/drive/172zCiXpLr1FEXMRCAbmZoqTrKiSkUERm?usp=sharing

Triển khai tất cả TODO trong `template.py`. Chạy `pytest tests/` để kiểm tra tiến độ.

**Điểm kiểm tra:** Sau khi hoàn thành 4 nhiệm vụ, chạy:
```bash
python template.py
```
Bạn sẽ thấy output so sánh phản hồi của GPT-4o và GPT-4o-mini.

---

## Phần 2 — Bài Tập Mở Rộng (1:00–1:30)

### Bài tập 2.1 — Độ Nhạy Của Temperature
Gọi `call_openai` với các giá trị temperature 0.0, 0.5, 1.0 và 1.5 sử dụng prompt **"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi temperature tăng dần, output trở nên đa dạng và sáng tạo hơn nhưng cũng kém ổn định hơn: ở **0.0** mô hình gần như cho một câu trả lời cố định lặp lại các fact phổ biến nhất (ví dụ "Việt Nam có 54 dân tộc" hay "phở/áo dài"); ở **0.5–1.0** cách diễn đạt phong phú hơn, fact đa dạng hơn nhưng vẫn chính xác; ở **1.5** ngôn ngữ bắt đầu rời rạc, dễ "bịa" số liệu hoặc trộn lẫn dữ kiện sai. Nói cách khác, temperature càng cao → entropy phân phối token càng cao → trade-off giữa **đa dạng** và **độ tin cậy**.

**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ đặt **temperature ≈ 0.2** (hoặc thậm chí 0.0 cho các luồng tra cứu chính sách/quy định). Lý do: chatbot hỗ trợ khách hàng cần (1) **nhất quán** — cùng một câu hỏi phải cho ra câu trả lời giống nhau qua các phiên, (2) **bám chặt nguồn dữ liệu thật** để tránh hallucination về giá, chính sách hoàn tiền, SLA…, và (3) **dễ kiểm thử/QA** vì output ít ngẫu nhiên. Sự "sáng tạo" hầu như không cần thiết trong bối cảnh này; nếu muốn câu trả lời đỡ máy móc, có thể nâng nhẹ lên 0.3 nhưng không nên vượt 0.5.

---

### Bài tập 2.2 — Đánh Đổi Chi Phí
Xem xét kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người thực hiện 3 lần gọi API, mỗi lần trung bình ~350 token.

**Ước tính xem GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này:**
> **Tổng lượng token/ngày:** 10.000 user × 3 call × 350 token = **10,5 triệu token/ngày**.
>
> Giả định trung bình mỗi call có ~50% input và ~50% output (175 in / 175 out), tức là 5,25M input + 5,25M output mỗi ngày:
>
> | Model | Chi phí input ($/ngày) | Chi phí output ($/ngày) | **Tổng/ngày** | **Tổng/tháng (30 ngày)** |
> |---|---:|---:|---:|---:|
> | GPT-4o (5,00 / 20,00 per 1M) | 5,25 × 5,00 = **$26,25** | 5,25 × 20,00 = **$105,00** | **$131,25** | **≈ $3.937** |
> | GPT-4o-mini (0,150 / 0,600 per 1M) | 5,25 × 0,150 = **$0,79** | 5,25 × 0,600 = **$3,15** | **$3,94** | **≈ $118** |
>
> **Tỷ lệ chênh lệch ≈ 131,25 / 3,94 ≈ 33,3×.**
>
> Lưu ý đẹp: vì đơn giá `gpt-4o : gpt-4o-mini` của *cả input lẫn output* đều bằng 5/0,15 = 20/0,6 = 33,33, nên **bất kể tỷ lệ in/out giả định là bao nhiêu**, GPT-4o vẫn luôn đắt **đúng ~33,3×** so với GPT-4o-mini cho cùng một lượng token. Tiết kiệm khi đổi sang mini ở quy mô này là khoảng **$127/ngày ≈ $46.000/năm**.

**Mô tả một trường hợp mà chi phí cao hơn của GPT-4o là xứng đáng, và một trường hợp GPT-4o-mini là lựa chọn tốt hơn:**
> **Đáng dùng GPT-4o:** các tác vụ **suy luận phức tạp, rủi ro cao** mà sai sót có chi phí lớn — ví dụ phân tích hợp đồng pháp lý, sinh code cho hệ thống thanh toán, hỗ trợ chẩn đoán y tế dựa trên triệu chứng, hay agentic workflow nhiều bước (planning + tool use). Trong các bối cảnh này, một câu trả lời tốt hơn 5–10% có thể tiết kiệm hàng giờ review của con người, nên trả thêm 33× chi phí token vẫn rẻ hơn rất nhiều so với chi phí lỗi.
>
> **Nên dùng GPT-4o-mini:** các tác vụ **khối lượng cao nhưng đơn giản** — ví dụ chatbot FAQ, phân loại intent, tóm tắt ngắn email, kiểm duyệt nội dung, hoặc gắn nhãn dữ liệu hàng loạt. Ở đây, mini đã đủ chất lượng (>95% accuracy với prompt tốt), trong khi 33× chi phí của GPT-4o sẽ trực tiếp ăn vào biên lợi nhuận sản phẩm. Một chiến lược thực dụng là **routing**: dùng mini làm mặc định, escalate sang GPT-4o chỉ khi confidence thấp hoặc câu hỏi phức tạp.

---

### Bài tập 2.3 — Trải Nghiệm Người Dùng với Streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming **quan trọng nhất khi có người dùng cuối đang chờ** một phản hồi dài (chatbot hội thoại, code copilot, writing assistant, tutor): nó kéo *time-to-first-token* xuống còn vài trăm mili-giây và biến độ trễ tổng (vài giây sinh hết câu) thành thời gian "đọc song song" — về mặt tâm lý người dùng cảm giác app **nhanh và sống động** hơn rất nhiều so với việc nhìn spinner 5–10 giây dù tổng thời gian gần như nhau. Ngược lại, **non-streaming phù hợp khi không có người chờ, hoặc cần nguyên khối kết quả để xử lý tiếp**: các pipeline backend (gọi API → parse JSON → ghi DB), các tác vụ yêu cầu **structured output / function calling / tool use** (cần JSON hợp lệ trước khi dùng), batch jobs offline, hoặc khi cần áp dụng **content moderation / validation** trên toàn bộ phản hồi trước khi hiển thị. Streaming cũng làm code phức tạp hơn (xử lý chunk, reconnect, hủy giữa chừng), nên nếu không có lợi ích UX rõ ràng thì non-streaming là lựa chọn đơn giản và đáng tin cậy hơn.


## Danh Sách Kiểm Tra Nộp Bài
- [ ] Tất cả tests pass: `pytest tests/ -v`
- [ ] `call_openai` đã triển khai và kiểm thử
- [ ] `call_openai_mini` đã triển khai và kiểm thử
- [ ] `compare_models` đã triển khai và kiểm thử
- [ ] `streaming_chatbot` đã triển khai và kiểm thử
- [ ] `retry_with_backoff` đã triển khai và kiểm thử
- [ ] `batch_compare` đã triển khai và kiểm thử
- [ ] `format_comparison_table` đã triển khai và kiểm thử
- [ ] `exercises.md` đã điền đầy đủ
- [ ] Sao chép bài làm vào folder `solution` và đặt tên theo quy định 
