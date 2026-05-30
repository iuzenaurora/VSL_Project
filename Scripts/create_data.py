import os
import cv2
import json
import math
import random
import shutil
import numpy as np
import pandas as pd
import mediapipe as mp

from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# =========================================================
# 1. CẤU HÌNH HỆ THỐNG
# =========================================================

TARGET_LENGTH        = 60
RANDOM_SEED          = 42
VISIBILITY_THR       = 0.3

# --- ĐIỀU CHỈNH CHO 67 VIDEO ---
# Tại sao MIN=3: với 75/25 split, class có 1 video không thể chia được
# (75% của 1 = 0 video train → crash). Bắt buộc tối thiểu 3.
MIN_VIDEOS_PER_LABEL = 3

# Số bản aug cho từ vựng có NHIỀU VIDEO NHẤT (cơ sở tính target)
# Tại sao 5: Max=16 → target=16×6=96 samples. Đủ để model
# học nhưng không quá dư cho class mạnh.
NUM_AUG_BASE         = 5

# Cap tối đa aug per video để tránh 1 video bị nhân thành 60+ bản sao
# giống hệt nhau (quá nhiều bản sao → model học thuộc lòng 1 sample)
# Tại sao 12: với 2 video train, aug×12 = 26 samples — đủ đa dạng
# nếu augmentation đủ mạnh (6 kiểu, random combination).
MAX_AUG_CAP          = 12

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = os.path.join(BASE_DIR, "Dataset")
LABEL_PATH   = os.path.join(DATASET_DIR, "Text", "label.csv")
VIDEO_DIR    = os.path.join(DATASET_DIR, "Videos")
OUTPUT_DIR   = os.path.join(BASE_DIR, "processed_data")
FEATURE_DIR  = os.path.join(OUTPUT_DIR, "features")

os.makedirs(FEATURE_DIR, exist_ok=True)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# =========================================================
# 2. CẤU HÌNH MEDIAPIPE & KHUNG XƯƠNG (BONE TREE)
# =========================================================

mp_holistic    = mp.solutions.holistic
POSE_LANDMARKS = 25
HAND_LANDMARKS = 21
NUM_JOINTS     = POSE_LANDMARKS + HAND_LANDMARKS + HAND_LANDMARKS  # 67
IN_CHANNELS    = 9   # Joint(3) + Velocity(3) + Bone(3)
TOTAL_FEATURES = NUM_JOINTS * IN_CHANNELS                          # 603

PARENTS = np.array([
    0, 0, 1, 2, 0, 4, 5, 3, 6, 0, 0, 0, 0, 11, 12, 13, 14, 15, 16, 15, 16, 15, 16, 11, 12,
    15, 25, 26, 27, 28, 25, 30, 31, 32, 25, 34, 35, 36, 25, 38, 39, 40, 25, 42, 43, 44,
    16, 46, 47, 48, 49, 46, 51, 52, 53, 46, 55, 56, 57, 46, 59, 60, 61, 46, 63, 64, 65
])

# =========================================================
# 3. TRÍCH XUẤT KEYPOINTS & ĐIỀN KHUYẾT
# =========================================================

def extract_keypoints(results) -> np.ndarray:
    pose = np.full((POSE_LANDMARKS, 3), np.nan, dtype=np.float32)
    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark[:POSE_LANDMARKS]):
            if lm.visibility >= VISIBILITY_THR:
                pose[i] = [lm.x, lm.y, lm.z]

    lh = (np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark], dtype=np.float32)
          if results.left_hand_landmarks else np.full((HAND_LANDMARKS, 3), np.nan, dtype=np.float32))

    rh = (np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark], dtype=np.float32)
          if results.right_hand_landmarks else np.full((HAND_LANDMARKS, 3), np.nan, dtype=np.float32))

    return np.concatenate([pose, lh, rh])  # (67, 3)


def fill_missing_keypoints(sequence: np.ndarray) -> np.ndarray:
    T, V, C = sequence.shape
    df = pd.DataFrame(sequence.reshape(T, -1))
    df = df.ffill(axis=0).bfill(axis=0).fillna(0.0)
    return df.to_numpy(dtype=np.float32).reshape(T, V, C)

# =========================================================
# 4. MOTION TRIMMING
# =========================================================

def trim_static_frames(sequence: np.ndarray) -> np.ndarray:
    if len(sequence) < TARGET_LENGTH // 2:
        return sequence
    hands_seq = sequence[:, 25:, :]
    diffs     = np.sum(np.abs(np.diff(hands_seq, axis=0)), axis=(1, 2))
    smoothed  = np.convolve(diffs, np.ones(5) / 5, mode='same')
    threshold = np.mean(smoothed) * 0.3
    active    = np.where(smoothed > threshold)[0]
    if len(active) > 5:
        return sequence[max(0, active[0] - 2): min(len(sequence), active[-1] + 3)]
    return sequence

# =========================================================
# 5. CHUẨN HÓA VỊ TRÍ (GỐC TỌA ĐỘ = TRUNG ĐIỂM HAI VAI)
# =========================================================

def normalize_keypoints(data: np.ndarray) -> np.ndarray:
    norm = data.copy()
    for t in range(len(norm)):
        ls, rs = norm[t, 11], norm[t, 12]
        if np.all(ls == 0) and np.all(rs == 0):
            continue
        if not np.isnan(ls[0]) and ls[0] != 0:
            anchor = (ls + rs) / 2.0
            w      = np.linalg.norm(ls - rs)
            if w == 0: w = 1e-6
        else:
            anchor, w = norm[t, 0], 1.0
        norm[t] = (norm[t] - anchor) / w
    return norm

# =========================================================
# 6. MULTI-STREAM FEATURES (9 CHANNELS)
# =========================================================

def compute_multi_stream_features(joints: np.ndarray) -> np.ndarray:
    T, V, C = joints.shape
    vel        = np.zeros_like(joints)
    vel[:-1]   = joints[1:] - joints[:-1]
    vel[-1]    = vel[-2]
    bone       = np.zeros_like(joints)
    for v in range(V):
        bone[:, v, :] = joints[:, v, :] - joints[:, PARENTS[v], :]
    return np.concatenate([joints, vel, bone], axis=-1)  # (T, 67, 9)

# =========================================================
# 7. AUGMENTATION PIPELINE — 6 KỸ THUẬT ĐA DẠNG
# =========================================================

def random_noise(seq: np.ndarray, level: float = 0.005) -> np.ndarray:
    """Nhiễu Gaussian mịn — giả lập run tay nhẹ"""
    noise = np.random.normal(0, level, seq.shape).astype(np.float32)
    return np.where(seq != 0, seq + noise, seq)


def scale_sequence(seq: np.ndarray) -> np.ndarray:
    """Thay đổi tỷ lệ toàn cục — giả lập khoảng cách camera khác nhau"""
    return (seq * np.random.uniform(0.88, 1.12)).astype(np.float32)


def spatial_jitter(seq: np.ndarray) -> np.ndarray:
    """
    Nhiễu phân vùng: thân người (khớp 0-24) ít nhiễu hơn tay (25-66)
    Tại sao: khi làm ký hiệu, thân người khá tĩnh, tay mới chuyển động
    → nhiễu tay lớn hơn giúp model tổng quát hóa tốt hơn với các ngón tay.
    """
    seq = seq.copy()
    T, V, C = seq.shape
    body_noise = np.random.normal(0, 0.003, (T, 25, C)).astype(np.float32)
    hand_noise = np.random.normal(0, 0.008, (T, 42, C)).astype(np.float32)
    seq[:, :25, :] = np.where(seq[:, :25, :] != 0, seq[:, :25, :] + body_noise, 0.0)
    seq[:, 25:, :] = np.where(seq[:, 25:, :] != 0, seq[:, 25:, :] + hand_noise, 0.0)
    return seq


def joint_dropout(seq: np.ndarray) -> np.ndarray:
    """
    Zero ngẫu nhiên 1-2 khớp hoàn toàn — giả lập bị che khuất
    Tại sao: camera thực tế đôi khi mất track 1 ngón tay.
    Giúp model không phụ thuộc tuyệt đối vào bất kỳ khớp nào.
    """
    seq  = seq.copy()
    T, V, _ = seq.shape
    for j in random.sample(range(V), random.randint(1, 2)):
        seq[:, j, :] = 0.0
    return seq


def temporal_crop(seq: np.ndarray) -> np.ndarray:
    """
    Cắt ngẫu nhiên 80-100% trục thời gian rồi nội suy tuyến tính về TARGET_LENGTH
    Tại sao: giúp model nhận diện ký hiệu được thực hiện nhanh hoặc chậm hơn bình thường.
    Crop tối thiểu 80% (48 frame) để không mất quá nhiều thông tin động tác.
    """
    T, V, C = seq.shape
    min_len = max(int(TARGET_LENGTH * 0.80), 4)  # tối thiểu 80% hoặc 4 frames

    if T <= min_len:
        # Sequence quá ngắn → chỉ resample về TARGET_LENGTH, không crop thêm
        crop = seq
    else:
        crop_len = random.randint(min_len, T)
        start    = random.randint(0, T - crop_len)
        crop     = seq[start: start + crop_len]

    # Nội suy tuyến tính từng channel về đúng TARGET_LENGTH frames
    idx       = np.linspace(0, len(crop) - 1, TARGET_LENGTH)
    resampled = np.zeros((TARGET_LENGTH, V, C), dtype=np.float32)
    for v in range(V):
        for c in range(C):
            resampled[:, v, c] = np.interp(idx, np.arange(len(crop)), crop[:, v, c])
    return resampled


def time_warp(seq: np.ndarray) -> np.ndarray:
    """
    Giãn / nén cục bộ trục thời gian — làm lệch nhịp điệu của động tác
    Tại sao: người thực hiện ký hiệu thực tế không đều tay, đoạn đầu
    có thể nhanh, đoạn giữa chậm → tạo ra nhiều biến thể thực tế hơn.
    Khác temporal_crop: time_warp giữ TOÀN BỘ sequence nhưng biến dạng tốc độ.
    """
    T, V, C = seq.shape
    # Tạo lưới thời gian bị lệch nhẹ (±15% tốc độ)
    orig_pts  = np.linspace(0, T - 1, 6)          # 6 điểm neo
    warp_pts  = orig_pts + np.random.uniform(-T * 0.15, T * 0.15, 6)
    warp_pts  = np.clip(warp_pts, 0, T - 1)
    warp_pts[0] = 0; warp_pts[-1] = T - 1         # giữ đầu/cuối cố định

    new_grid  = np.interp(np.linspace(0, T - 1, T), orig_pts, warp_pts)
    warped    = np.zeros_like(seq)
    for v in range(V):
        for c in range(C):
            warped[:, v, c] = np.interp(new_grid, np.arange(T), seq[:, v, c])
    return warped.astype(np.float32)


def augment_sequence(seq: np.ndarray) -> np.ndarray:
    """
    Pipeline tổng hợp 6 kỹ thuật — mỗi kỹ thuật có xác suất độc lập.
    Mỗi lần gọi sẽ tạo ra một biến thể NGẪU NHIÊN KHÁC NHAU.
    Thứ tự: temporal trước (thay đổi trục thời gian) → spatial sau.
    Không dùng horizontal_flip vì ngôn ngữ ký hiệu quy định đúng tay —
    lật gương tạo ra ký hiệu sai hoàn toàn về mặt ngữ nghĩa.
    """
    # 1. Temporal crop (80% xác suất — quan trọng nhất với dữ liệu nhỏ)
    if random.random() < 0.80:
        seq = temporal_crop(seq)

    # 2. Time warp (60% xác suất — biến dạng nhịp điệu)
    if random.random() < 0.60:
        seq = time_warp(seq)

    # 3. Gaussian noise (50%)
    if random.random() < 0.50:
        seq = random_noise(seq)

    # 4. Scale (50%)
    if random.random() < 0.50:
        seq = scale_sequence(seq)

    # 5. Spatial jitter phân vùng (50%)
    if random.random() < 0.50:
        seq = spatial_jitter(seq)

    # 6. Joint dropout (40% — không quá thường xuyên tránh mất info)
    if random.random() < 0.40:
        seq = joint_dropout(seq)

    return seq.astype(np.float32)

# =========================================================
# 8. PIPELINE XỬ LÝ VIDEO
# =========================================================

def process_video(video_path: str, holistic):
    cap, sequence = cv2.VideoCapture(video_path), []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        sequence.append(extract_keypoints(holistic.process(rgb)))
    cap.release()

    if not sequence:
        zero = np.zeros((TARGET_LENGTH, NUM_JOINTS, 3), dtype=np.float32)
        return zero, zero

    seq = np.array(sequence, dtype=np.float32)
    seq = fill_missing_keypoints(seq)
    seq = trim_static_frames(seq)
    seq = normalize_keypoints(seq)

    # raw: giữ nguyên độ dài thực sau trim, dùng cho augmentation
    raw = seq.copy()

    # fixed: pad/truncate về đúng TARGET_LENGTH, dùng cho bản gốc
    T = len(seq)
    if T >= TARGET_LENGTH:
        fixed = seq[:TARGET_LENGTH]
    else:
        fixed        = np.zeros((TARGET_LENGTH, NUM_JOINTS, 3), dtype=np.float32)
        fixed[:T]    = seq
    return raw, fixed


def save_npz(filename: str, seq_3d: np.ndarray, label: int, text: str) -> dict:
    """
    seq_3d có shape (T, 67, 3) — hàm tự tính multi-stream và lưu (T, 603).
    Trích xuất tại đây thay vì trước để augmentation hoạt động trên tọa độ gốc,
    không trên velocity/bone đã tính (tránh hiệu ứng kép khi aug làm lệch tọa độ).
    """
    # Đảm bảo đúng TARGET_LENGTH trước khi tính multi-stream
    T = len(seq_3d)
    if T != TARGET_LENGTH:
        idx   = np.linspace(0, T - 1, TARGET_LENGTH)
        final = np.zeros((TARGET_LENGTH, NUM_JOINTS, 3), dtype=np.float32)
        for v in range(NUM_JOINTS):
            for c in range(3):
                final[:, v, c] = np.interp(idx, np.arange(T), seq_3d[:, v, c])
        seq_3d = final

    multi = compute_multi_stream_features(seq_3d)
    flat  = multi.reshape(TARGET_LENGTH, TOTAL_FEATURES)
    path  = os.path.join(FEATURE_DIR, filename)
    np.savez_compressed(path, sequence=flat, label=label, text=text, length=TARGET_LENGTH)
    return {"path": os.path.join("features", filename), "label": label, "text": text}

# =========================================================
# 9. DYNAMIC BALANCER — CÂN BẰNG MẪU THEO CLASS
# =========================================================

def process_dataset_dynamic(df: pd.DataFrame, split_name: str, holistic) -> pd.DataFrame:
    csv_rows = []
    is_train = (split_name == "train")

    # Tập val: CHỈ bản gốc — không aug để tránh đánh giá bị lạc quan
    if not is_train:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name}"):
            v_path = os.path.join(VIDEO_DIR, row["VIDEO"])
            if not os.path.exists(v_path): continue
            _, fixed = process_video(v_path, holistic)
            csv_rows.append(save_npz(f"val_{idx}_orig.npz", fixed, row["LABEL_ENCODED"], row["LABEL"]))
        return pd.DataFrame(csv_rows)

    # --- Tập train: Dynamic Balancing ---
    label_groups     = {label: g for label, g in df.groupby("LABEL")}
    max_train_count  = max(len(g) for g in label_groups.values())

    # Target = max_class_videos × (1 + NUM_AUG_BASE)
    # Ví dụ: 12 video × (1+5) = 72 samples cho class lớn nhất
    TARGET_SAMPLES = max_train_count * (1 + NUM_AUG_BASE)

    print(f"\n[Dynamic Balancer] max_train_videos={max_train_count}  →  target={TARGET_SAMPLES} samples/class")
    print(f"[Dynamic Balancer] MAX_AUG_CAP per video = {MAX_AUG_CAP}\n")

    for label_name, group in label_groups.items():
        T_c         = len(group)  # số video train gốc của class này
        aug_raw     = math.ceil(TARGET_SAMPLES / T_c) - 1
        aug_per_vid = min(aug_raw, MAX_AUG_CAP)  # cap chống nhân bản quá nhiều

        # Số mẫu thực tế sau khi bị cap (thường thấp hơn TARGET với class ít video)
        actual_total = T_c * (1 + aug_per_vid)
        gap_pct      = abs(actual_total - TARGET_SAMPLES) / TARGET_SAMPLES * 100
        print(f"  {label_name:20s}: {T_c:2d} train vid → aug×{aug_per_vid:2d} "
              f"→ {actual_total:3d} samples  (gap {gap_pct:.0f}% from target)")

        # Phân bổ đều aug copies cho từng video trong group
        base_copies = aug_per_vid
        remainder   = (TARGET_SAMPLES - T_c) % T_c  # phân bổ phần dư lần lượt

        for i, (_, row) in enumerate(group.iterrows()):
            v_path = os.path.join(VIDEO_DIR, row["VIDEO"])
            if not os.path.exists(v_path): continue

            raw, fixed = process_video(v_path, holistic)

            # Bản gốc
            csv_rows.append(save_npz(
                f"train_{row['LABEL_ENCODED']}_{i}_orig.npz",
                fixed, row["LABEL_ENCODED"], row["LABEL"]
            ))

            # Augmented — dùng raw (chưa pad) để temporal_crop có frame dự phòng
            extra = 1 if (i < remainder and aug_raw <= MAX_AUG_CAP) else 0
            for aug_i in range(base_copies + extra):
                aug = augment_sequence(raw.copy())
                csv_rows.append(save_npz(
                    f"train_{row['LABEL_ENCODED']}_{i}_aug{aug_i}.npz",
                    aug, row["LABEL_ENCODED"], row["LABEL"]
                ))

    return pd.DataFrame(csv_rows)

# =========================================================
# 10. MAIN
# =========================================================

def main():
    print("=" * 65)
    print("  CSLR — Dynamic Balance Augmentation Pipeline (67-video edition)")
    print("=" * 65)

    df = pd.read_csv(LABEL_PATH)
    df["LABEL"] = df["LABEL"].astype(str).str.strip()
    df["VIDEO"] = df["VIDEO"].astype(str).str.strip()

    # Lọc video tồn tại thực sự trên disk
    df = df[df["VIDEO"].apply(lambda x: os.path.exists(os.path.join(VIDEO_DIR, x)))]

    # Giữ lại class có ít nhất MIN_VIDEOS_PER_LABEL=2
    # (cần ít nhất 2 video để chia train/val)
    counts       = df["LABEL"].value_counts()
    valid_labels = counts[counts >= MIN_VIDEOS_PER_LABEL].index
    df           = df[df["LABEL"].isin(valid_labels)].reset_index(drop=True)

    print(f"\n  Tổng video hợp lệ  : {len(df)}")
    print(f"  Số từ vựng giữ lại : {df['LABEL'].nunique()}")
    print(f"  (Loại class có < {MIN_VIDEOS_PER_LABEL} video vì không thể chia train/val)\n")

    le = LabelEncoder()
    df["LABEL_ENCODED"] = le.fit_transform(df["LABEL"])
    mapping_path = os.path.join(OUTPUT_DIR, "label_mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump({l: int(i) for i, l in enumerate(le.classes_)}, f, ensure_ascii=False, indent=4)

    # Chia 75% train / 25% val — stratified theo label
    train_rows, val_rows = [], []
    for _, group in df.groupby("LABEL"):
        group     = group.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        split_idx = max(1, int(len(group) * 0.75))
        if split_idx >= len(group):
            split_idx = len(group) - 1
        train_rows.extend(group.iloc[:split_idx].to_dict("records"))
        val_rows.extend(group.iloc[split_idx:].to_dict("records"))

    train_df = pd.DataFrame(train_rows)
    val_df   = pd.DataFrame(val_rows)
    print(f"  Train gốc : {len(train_df)} video")
    print(f"  Val gốc   : {len(val_df)} video")

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    ) as holistic:
        train_csv = process_dataset_dynamic(train_df, "train", holistic)
        val_csv   = process_dataset_dynamic(val_df,   "val",   holistic)

    train_csv.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    val_csv.to_csv(os.path.join(OUTPUT_DIR, "val.csv"),   index=False)

    print(f"\n  Train samples sau aug : {len(train_csv)}")
    print(f"  Val samples           : {len(val_csv)}")
    print(f"  Ratio train/val       : {len(train_csv)/max(len(val_csv),1):.1f}x")

    shutil.make_archive(os.path.join(BASE_DIR, "cslr_dataset"), "zip", OUTPUT_DIR)

    print(f"\n{'=' * 65}")
    print("  HOÀN TẤT — Dataset cân bằng và sẵn sàng huấn luyện!")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()