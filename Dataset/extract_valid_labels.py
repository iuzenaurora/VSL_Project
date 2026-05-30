import os
import pandas as pd
from collections import Counter

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "Dataset", "Text", "label.csv")
VIDEO_DIR = os.path.join(BASE_DIR, "Dataset", "Videos")
OUTPUT_PATH = os.path.join(BASE_DIR, "valid_labels.csv")

def main():
    if not os.path.exists(CSV_PATH):
        print(f"❌ Không tìm thấy file: {CSV_PATH}")
        return
        
    # Đọc file label.csv
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        print(f"❌ Lỗi đọc file CSV: {e}")
        return

    label_counts = Counter()
    valid_count = 0
    missing_count = 0

    print("Đang kiểm tra sự tồn tại của các video trong thư mục...")
    for index, row in df.iterrows():
        video_name = str(row['VIDEO']).strip()
        label = str(row['LABEL']).strip()
        
        video_path = os.path.join(VIDEO_DIR, video_name)
        if os.path.exists(video_path):
            label_counts[label] += 1
            valid_count += 1
        else:
            missing_count += 1

    # Tạo DataFrame từ dictionary đếm
    out_df = pd.DataFrame(label_counts.items(), columns=["LABEL", "SO_LUONG_VIDEO"])
    
    # Sắp xếp theo số lượng video giảm dần, sau đó theo tên từ vựng
    out_df = out_df.sort_values(by=["SO_LUONG_VIDEO", "LABEL"], ascending=[False, True])
    
    # Xuất ra file CSV
    out_df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8')
    
    # Lấy danh sách các từ có >= 2 video
    multi_video_df = out_df[out_df["SO_LUONG_VIDEO"] > 1]
    
    print("\n==========================================")
    print(f"✅ THÀNH CÔNG! Đã kiểm tra xong dữ liệu.")
    print(f"👉 Tổng số video thực tế tồn tại: {valid_count}")
    print(f"👉 Tổng số video bị thiếu trong thư mục: {missing_count}")
    print(f"👉 Tổng số lượng TỪ VỰNG (LABEL) hợp lệ: {len(out_df)}")
    print(f"👉 Số lượng từ vựng có NHIỀU HƠN 1 video: {len(multi_video_df)}")
    print(f"👉 Danh sách từ vựng đã được lưu vào: {OUTPUT_PATH}")
    print("==========================================\n")
    
    if not multi_video_df.empty:
        print("TOP NHỮNG TỪ VỰNG CÓ NHIỀU VIDEO NHẤT:")
        print(multi_video_df.head(20).to_string(index=False))
        print("...\n(Xem đầy đủ trong file valid_labels.csv)")

if __name__ == "__main__":
    main()