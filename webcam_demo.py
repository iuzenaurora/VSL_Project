# ==========================================
# HƯỚNG DẪN CÀI ĐẶT (chạy 1 lần trong terminal)
# ==========================================
# pip install mediapipe==0.10.5
# pip install protobuf==3.20.3         ← QUAN TRỌNG: phải đúng version này
# pip install google-generativeai      ← đúng tên package
# pip install opencv-python pillow numpy pandas torch
# ==========================================

import cv2
import json
import math
import time
import requests
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import mediapipe as mp
import os
from collections import deque, Counter
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG (ĐỒNG BỘ VỚI CANVAS)
# ==========================================
SEQ_LEN          = 60
NUM_JOINTS       = 67
IN_CHANNELS      = 9    # x,y,z + velocity + bone = 3*3
HIDDEN_DIM       = 256   # Đồng bộ 256 từ TrainModel
NUM_GCN_LAYERS   = 3
NUM_ATTN_HEADS   = 4
NUM_TRANS_LAYERS = 2
DROPOUT          = 0.3  # Đồng bộ Dropout = 0.3 chống học vẹt

CONFIDENCE_THRESHOLD = 0.4
PREDICTION_COOLDOWN  = 18  # Nghỉ 18 frames sau khi đoán trúng để tránh spam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Nạp nhãn label mapping linh hoạt
mapping_path = "Model/label_mapping.json" if os.path.exists("Model/label_mapping.json") else "processed_data/label_mapping_2.json"
with open(mapping_path, "r", encoding="utf-8") as f:
    label_map = json.load(f)
idx_to_label = {v: k for k, v in label_map.items()}
NUM_CLASSES = len(label_map)
print(f"NUM_CLASSES: {NUM_CLASSES}")

# ==========================================
# 2. ADJACENCY MATRIX (MA TRẬN KỀ ĐỒ THỊ)
# ==========================================
def build_adjacency(num_joints=67):
    A = np.zeros((num_joints, num_joints), dtype=np.float32)
    def connect(i, j): A[i, j] = A[j, i] = 1

    pose_edges = [
        (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
        (9,10),(11,12),(11,13),(13,15),(12,14),(14,16),
        (11,23),(12,24),(23,24),(0,11),(0,12),
    ]
    for i, j in pose_edges:
        if i < 25 and j < 25: connect(i, j)

    hand_edges = [
        (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
    ]
    for i, j in hand_edges: connect(25 + i, 25 + j)
    for i, j in hand_edges: connect(46 + i, 46 + j)
    connect(15, 25); connect(16, 46)
    np.fill_diagonal(A, 1)

    D_inv = np.diag(1.0 / np.maximum(A.sum(axis=1), 1e-6) ** 0.5)
    return torch.from_numpy(D_inv @ A @ D_inv).float()

ADJ = build_adjacency().to(device)

# ==========================================
# 3. KIẾN TRÚC MÔ HÌNH VSTGCN ĐỒNG BỘ SOTA
# ==========================================
class GraphConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.W    = nn.Linear(in_ch, out_ch, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_ch))
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x, A):
        out = torch.einsum("btvc,vw->btwc", self.W(x), A) + self.bias
        return self.act(self.bn(out.permute(0,3,1,2))).permute(0,2,3,1)


class STGCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout):
        super().__init__()
        self.gcn = GraphConv(in_ch, out_ch)
        self.tcn = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, (3,1), padding=(1,0), groups=out_ch),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Dropout(dropout),
        )
        self.residual = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1), nn.BatchNorm2d(out_ch))
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x, A):
        res = x.permute(0,3,1,2)
        out = self.gcn(x, A).permute(0,3,1,2)
        return (self.tcn(out) + self.residual(res)).permute(0,2,3,1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe       = torch.zeros(1, max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class VSTGCN(nn.Module):
    def __init__(self):
        super().__init__()
        chs = [IN_CHANNELS] + [HIDDEN_DIM] * NUM_GCN_LAYERS
        self.stgcn       = nn.ModuleList([STGCNBlock(chs[i], chs[i+1], DROPOUT) for i in range(NUM_GCN_LAYERS)])
        self.pos_encoder = PositionalEncoding(HIDDEN_DIM, max_len=SEQ_LEN)
        enc_layer        = nn.TransformerEncoderLayer(
            HIDDEN_DIM, NUM_ATTN_HEADS, HIDDEN_DIM * 4,
            DROPOUT, "gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, NUM_TRANS_LAYERS)
        
        # ĐỒNG BỘ SOTA: Đưa LayerNorm vào spatial_proj để khớp với TrainModel Checkpoint
        self.spatial_proj = nn.Sequential(
            nn.Linear(NUM_JOINTS * HIDDEN_DIM, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM)
        )
        self.fc          = nn.Linear(HIDDEN_DIM, NUM_CLASSES)

    def forward(self, x, A):
        x = x.permute(0, 2, 3, 1) # Shape: (B, T, V, C)
        for blk in self.stgcn:
            x = blk(x, A)
        
        B, T, V, C = x.shape
        x = x.reshape(B, T, V * C)
        x = self.spatial_proj(x)

        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.fc(x.mean(dim=1))


# Nạp trọng số tốt nhất
model = VSTGCN().to(device)
model_path = "Model/best_model.pth" if os.path.exists("Model/best_model.pth") else "Model/best_model_2.pth"
ckpt  = torch.load(model_path, map_location=device)
state = ckpt.get("model", ckpt)
model.load_state_dict(state)
model.eval()
print("Mô hình đã nạp thành công.")

# ==========================================
# 4. HẬU XỬ LÝ NLP DỊCH THUẬT (SỬA LỖI PAYLOAD)
# ==========================================
class NLPTranslator:
    def __init__(self):
        self.buffer = []
        self.vocab_mapping = {
            "0 (số không)": "không",
            "tp. hồ chí minh": "TP. Hồ Chí Minh",
            "đồng nai": "Đồng Nai",
        }
        self.api_url = "http://127.0.0.1:5000/translate"

    def add_gloss(self, word: str):
        w = word.lower()
        if not self.buffer or self.buffer[-1] != w:
            self.buffer.append(w)

    def _build_raw_sentence(self) -> str:
        processed = []
        spell_buf = ""

        for word in self.buffer:
            if len(word) == 1 or word.isdigit():
                spell_buf += word
            else:
                if spell_buf:
                    processed.append(spell_buf.capitalize())
                    spell_buf = ""
                processed.append(self.vocab_mapping.get(word, word))

        if spell_buf:
            processed.append(spell_buf.capitalize())

        return " ".join(processed)

    def finalize_sentence(self) -> str:
        if not self.buffer:
            return ""

        raw = self._build_raw_sentence()
        print(f"[Gloss Raw] {raw}")
        self.buffer.clear()

        try:
            # Sửa payload chính xác thành {"text": raw} khớp với server Flask
            response = requests.post(
                self.api_url,
                json={"text": raw},
                timeout=15
            )
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    final_text = result.get("result", "").strip()
                    print(f"[Gemini] {final_text}")
                    return final_text
                else:
                    print(f"[Gemini Error Backend] {result}")
            return raw.capitalize() + "."
        except Exception as e:
            print(f"[API ERROR] {e}")
            return raw.capitalize() + "."


translator = NLPTranslator()

# ==========================================
# 5. TIỀN XỬ LÝ KHỚP REAL-TIME
# ==========================================
PARENTS = np.array([
    0,0,1,2,0,4,5,3,6,0,0,0,0,11,12,13,14,15,16,15,16,15,16,11,12,
    15,25,26,27,28,25,30,31,32,25,34,35,36,25,38,39,40,25,42,43,44,
    16,46,47,48,49,46,51,52,53,46,55,56,57,46,59,60,61,46,63,64,65,
])


def extract_keypoints(results) -> np.ndarray:
    pose = np.full((25, 3), np.nan, dtype=np.float32)
    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark[:25]):
            if lm.visibility >= 0.3:
                pose[i] = [lm.x, lm.y, lm.z]

    lh = (np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark], dtype=np.float32)
          if results.left_hand_landmarks else np.full((21, 3), np.nan, dtype=np.float32))
    rh = (np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark], dtype=np.float32)
          if results.right_hand_landmarks else np.full((21, 3), np.nan, dtype=np.float32))

    return np.concatenate([pose, lh, rh])


def process_features(raw: np.ndarray) -> torch.Tensor:
    flat = pd.DataFrame(raw.reshape(SEQ_LEN, -1)).ffill().bfill().fillna(0.0).to_numpy()
    seq  = flat.reshape(SEQ_LEN, NUM_JOINTS, 3).astype(np.float32)

    for t in range(SEQ_LEN):
        ls, rs = seq[t, 11], seq[t, 12]
        if np.all(ls == 0) and np.all(rs == 0):
            continue
        anchor = (ls + rs) / 2.0
        w      = np.linalg.norm(ls - rs)
        seq[t] = (seq[t] - anchor) / max(w, 1e-6)

    vel       = np.zeros_like(seq)
    vel[:-1]  = seq[1:] - seq[:-1]
    vel[-1]   = vel[-2]

    bone = seq - seq[:, PARENTS, :]

    multi  = np.concatenate([seq, vel, bone], axis=-1)
    tensor = torch.from_numpy(np.transpose(multi, (2, 0, 1))).float()
    return tensor.unsqueeze(0).to(device)


# ==========================================
# 6. HIỂN THỊ VIỆT HÓA CHỮ CÓ DẤU
# ==========================================
def _find_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def put_text(img: np.ndarray, text: str, pos: tuple,
             size: int = 30, color_bgr: tuple = (0, 255, 0)) -> np.ndarray:
    try:
        pil   = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw  = ImageDraw.Draw(pil)
        font  = _find_font(size)
        rgb   = (color_bgr[2], color_bgr[1], color_bgr[0])
        draw.text(pos, text, font=font, fill=rgb)
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.putText(img, text, (pos[0], pos[1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_bgr, 2)
        return img


# ==========================================
# 7. CAMERA LOOP (TÍCH HỢP COOLDOWN & VOTE SỐ ĐÔNG)
# ==========================================
mp_holistic     = mp.solutions.holistic
frames_buffer   = deque(maxlen=SEQ_LEN)
current_word    = ""
final_sentence  = ""
hand_down_frames = 0

prediction_history = deque(maxlen=3) # Bầu chọn số đông trên 3 frames suy luận gần nhất
cooldown_counter   = 0
word_display_counter = 0               # Bộ đếm giữ chữ hiển thị trên UI không bị biến mất nhanh

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1,
) as holistic:

    print("Hệ thống nhận diện Webcam đã sẵn sàng! SPACE = Dịch câu, ESC = Thoát.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        kpts = extract_keypoints(results)
        frames_buffer.append(kpts)

        # ── Suy luận kết hợp Cooldown & Majority Voting ─────────────
        # ── Tìm và THAY THẾ TOÀN BỘ khối lệnh này bên trong vòng lặp while ───
        if len(frames_buffer) == SEQ_LEN:
            cooldown_counter = max(0, cooldown_counter - 1)
            
            if cooldown_counter == 0:
                with torch.no_grad():
                    tensor = process_features(np.array(frames_buffer))
                    logits = model(tensor, ADJ)
                    probs  = torch.softmax(logits, dim=1)
                    max_prob, pred_idx = probs.max(1)

                    conf  = max_prob.item()
                    label = idx_to_label[pred_idx.item()]

                    if conf > CONFIDENCE_THRESHOLD:
                        prediction_history.append(label)
                        
                        # Chỉ chấp nhận từ khi có sự đồng thuận 2/3 khung hình gần nhất
                        if len(prediction_history) == 3:
                            most_common, count = Counter(prediction_history).most_common(1)[0]
                            if count >= 2:
                                translator.add_gloss(most_common) # Thêm vào hàng đợi dịch
                                current_word = most_common         # Gán từ hiển thị lên UI
                                cooldown_counter = PREDICTION_COOLDOWN # Kích hoạt thời gian nghỉ (18 frames)
                                word_display_counter = 30          # Giữ chữ trên màn hình hiển thị trong 30 frames (~1 giây)
                                prediction_history.clear()

        # LOGIC GIỮ CHỮ UI: Giảm bộ đếm thời gian giữ chữ theo từng khung hình
        if word_display_counter > 0:
            word_display_counter -= 1
        else:
            current_word = "" # Chỉ xóa chữ khi thời gian găm chữ trên UI kết thúc

        # ── Nhận diện hạ tay múa tự động ───
        if results.pose_landmarks:
            lw = results.pose_landmarks.landmark[15].y
            rw = results.pose_landmarks.landmark[16].y
            if lw > 0.65 and rw > 0.65:
                hand_down_frames += 1
            else:
                hand_down_frames = 0

            if hand_down_frames > 30 and translator.buffer:
                print("  → Tự động ghép câu khi hạ tay...")
                final_sentence   = translator.finalize_sentence()
                current_word     = ""
                hand_down_frames = 0
                frames_buffer.clear()
                prediction_history.clear()

        # ── Giao diện UI ─────────────────────────────────────
        frame = put_text(frame, f"Từ nhận diện: {current_word}",    (20, 20),  size=35, color_bgr=(0, 255, 0))
        frame = put_text(frame, f"Câu dịch: {final_sentence}",      (20, 70),  size=30, color_bgr=(0, 255, 255))
        frame = put_text(frame, f"Bộ đệm: {len(translator.buffer)} từ khóa", (20, 115), size=25, color_bgr=(200, 200, 200))

        cv2.imshow("VSL AI Translator", frame)

        key = cv2.waitKey(10) & 0xFF
        if key == 27:
            break
        elif key == 32: # SPACE
            if translator.buffer:
                print("  → Ép buộc ghép câu (SPACE)...")
                final_sentence   = translator.finalize_sentence()
                current_word     = ""
                hand_down_frames = 0
                frames_buffer.clear()
                prediction_history.clear()

cap.release()
cv2.destroyAllWindows()