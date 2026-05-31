# 🤟 VSL AI Translator - Hệ thống Nhận diện và Dịch thuật Ngôn ngữ Ký hiệu Việt Nam

Dự án ứng dụng Deep Learning để nhận diện Ngôn ngữ Ký hiệu Việt Nam (Vietnamese Sign Language - VSL) qua Webcam theo thời gian thực. Hệ thống sử dụng mô hình học sâu **VSTGCN** để trích xuất đặc trưng khung xương và sử dụng **thuật toán xử lý ngôn ngữ thủ công (Rule-based)** để ghép các từ khóa (Gloss) thành câu tiếng Việt.

## ✨ Tính năng nổi bật
- **Nhận diện Real-time:** Trích xuất khung xương 3D liên tục qua MediaPipe Holistic và dự đoán từ khóa với độ trễ cực thấp.
- **Xử lý ngôn ngữ tự nhiên (NLP):** Xử lý từ khóa thủ công (Rule-based) nhanh chóng, chạy hoàn toàn offline và không phụ thuộc vào API bên thứ ba.
- **2 Chế độ hoạt động:** 
  - **Web App:** Giao diện thân thiện, nhận diện qua WebSocket siêu tốc, tích hợp "Từ điển Video" cho người học.
  - **Desktop App:** Chạy trực tiếp qua cửa sổ OpenCV siêu nhẹ.

---

## ⚙️ Hướng dẫn Cài đặt

### Bước 1: Clone dự án
Mở Terminal (Command Prompt / PowerShell) và chạy lệnh sau:
```bash
git clone https://github.com/TEN_GITHUB_CUA_BAN/VSL_Project.git
cd VSL_Project
```
*(Lưu ý: Bạn nên sử dụng **Python 3.9** hoặc **3.10** để tương thích tốt nhất với MediaPipe và PyTorch).*

### Bước 2: Cài đặt thư viện (Dependencies)
Hãy đảm bảo cài đặt chính xác các phiên bản thư viện dưới đây để tránh lỗi xung đột (đặc biệt là `protobuf`):
```bash
pip install mediapipe==0.10.5
pip install protobuf==3.20.3
pip install opencv-python pillow numpy pandas torch flask flask-socketio requests
```
*(Nếu máy bạn có Card đồ họa NVIDIA, hãy cài đặt PyTorch hỗ trợ CUDA để tăng tốc độ nhận diện).*

### Bước 3: Tải Dataset Video từ Google Drive
Do giới hạn dung lượng của GitHub, thư mục chứa Video mẫu từ điển (`Dataset/Videos/`) đã được đưa vào `.gitignore`. 
1. Truy cập vào link Google Drive sau: **[Chèn Link Google Drive của bạn vào đây]**
2. Tải xuống toàn bộ tệp nén (hoặc thư mục).
3. Giải nén và đặt các video vào đúng đường dẫn sau trong dự án:
   ```text
   VSL_Project/
   └── Dataset/
       └── Videos/
           ├── video_1.mp4
           ├── video_2.mp4
           └── ...
   ```

### Bước 4: Cấu hình API Key (Tùy chọn)
Hệ thống mặc định sử dụng Google Gemini để dịch câu. Bạn hãy mở tệp `core/nlp_translator.py`, tìm đến dòng số 15 và thay thế chuỗi API Key bằng Key cá nhân của bạn (Lấy miễn phí tại Google AI Studio):
```python
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
```
*Nếu bạn muốn chạy hoàn toàn Offline (không cần mạng), hãy đổi `USE_GEMINI = False`.*

---

## 🚀 Hướng dẫn Sử dụng

Bạn có thể khởi chạy hệ thống bằng 1 trong 2 cách sau:

### Cách 0: Chạy bằng 1 click (Dành cho người dùng Windows)
Chỉ cần mở thư mục dự án và click đúp chuột vào tệp:
👉 **`run_web.bat`**

*Hệ thống sẽ tự động cài đặt các thư viện còn thiếu và bật Server lên cho bạn. Sau khi thấy chữ báo thành công, hãy vào trình duyệt và gõ `http://127.0.0.1:5000`*

### Cách 1: Chạy phiên bản Web (Khuyên dùng)
Phiên bản Web cung cấp giao diện trực quan, cho phép tùy chỉnh ngưỡng hạ tay, xem từ điển mẫu và upload video có sẵn.
```bash
python app.py
```
👉 Sau khi Terminal báo chạy thành công, hãy mở trình duyệt và truy cập: `http://127.0.0.1:5000`

### Cách 2: Chạy phiên bản Desktop (OpenCV)
Dành cho việc kiểm thử nhanh, giao diện gọn nhẹ, không cần mở trình duyệt.
*(Lưu ý: Bạn vẫn phải mở `app.py` ở một Terminal khác để cấp API dịch thuật cho bản Desktop)*
```bash
# Mở Terminal 1
python app.py

# Mở Terminal 2
python webcam_demo.py
```
**Phím tắt phiên bản Desktop:**
- `SPACE`: Ép buộc hệ thống dịch và ghép các từ vừa múa ngay lập tức.
- `ESC`: Tắt Camera và thoát chương trình.

---

## 📁 Cấu trúc Dự án

```text
VSL_Project/
├── app.py                 # File khởi chạy Server Flask (Web Backend & API)
├── webcam_demo.py         # File khởi chạy Desktop App (OpenCV)
├── core/                  # Thư mục chứa lõi AI
│   ├── slr_engine.py      # Xử lý trích xuất khung xương và gọi Model VSTGCN
│   └── nlp_translator.py  # Xử lý ngôn ngữ tự nhiên (Gemini / Rule-based)
├── Model/                 # Chứa trọng số mô hình (.pth) và nhãn (JSON)
├── Dataset/               
│   ├── Videos/            # Nơi chứa video từ điển (Tải từ Drive)
│   └── Text/label.csv     # File mapping nhãn và tên video
├── static/                # Giao diện tĩnh (CSS, JS)
└── templates/             # Giao diện Web HTML (index.html)
```

---

## 🛠 Khắc phục Lỗi thường gặp
1. **Lỗi `TypeError: Descriptors cannot not be created directly...`**
   *Nguyên nhân:* Phiên bản Protobuf không tương thích với MediaPipe.
   *Cách sửa:* Chạy lệnh `pip install protobuf==3.20.3`

2. **Lỗi `[GEMINI API LIMIT] CẢNH BÁO...`**
   *Nguyên nhân:* Vượt quá giới hạn dịch 15 câu/phút của bản miễn phí.
   *Cách sửa:* Chờ 1 phút để API reset lại, hoặc kéo thanh "Ngưỡng hạ tay" trên web lên cao hơn (VD: 0.9) để tránh dịch liên tục.

---
*Dự án được phát triển nhằm mục đích hỗ trợ giao tiếp cho người khiếm thính tại Việt Nam.*