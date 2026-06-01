# -*- coding: utf-8 -*-
"""
SLR Engine đồng bộ 100% với webcam_demo.py
"""

import os
import cv2
import json
import math
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import mediapipe as mp
import threading

from collections import deque, Counter

# =========================================================
# CẤU HÌNH
# =========================================================

SEQ_LEN = 60
NUM_JOINTS = 67
IN_CHANNELS = 9
HIDDEN_DIM = 256
NUM_GCN_LAYERS = 3
NUM_ATTN_HEADS = 4
NUM_TRANS_LAYERS = 2
DROPOUT = 0.3

CONFIDENCE_THRESHOLD = 0.4
PREDICTION_COOLDOWN = 18

# =========================================================
# ADJACENCY MATRIX
# =========================================================

def build_adjacency(num_joints=67):
    A = np.zeros((num_joints, num_joints), dtype=np.float32)

    def connect(i, j):
        A[i, j] = 1
        A[j, i] = 1

    pose_edges = [
        (0,1),(1,2),(2,3),(3,7),
        (0,4),(4,5),(5,6),(6,8),
        (9,10),
        (11,12),
        (11,13),(13,15),
        (12,14),(14,16),
        (11,23),(12,24),(23,24),
        (0,11),(0,12),
    ]

    for i, j in pose_edges:
        connect(i, j)

    hand_edges = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17)
    ]

    for i, j in hand_edges:
        connect(25+i, 25+j)

    for i, j in hand_edges:
        connect(46+i, 46+j)

    connect(15, 25)
    connect(16, 46)

    np.fill_diagonal(A, 1)

    D_inv = np.diag(
        1.0 / np.maximum(A.sum(axis=1), 1e-6) ** 0.5
    )

    return torch.from_numpy(
        D_inv @ A @ D_inv
    ).float()

# =========================================================
# GRAPH CONV
# =========================================================

class GraphConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.W = nn.Linear(
            in_ch,
            out_ch,
            bias=False
        )

        self.bias = nn.Parameter(
            torch.zeros(out_ch)
        )

        self.bn = nn.BatchNorm2d(out_ch)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x, A):

        out = torch.einsum(
            "btvc,vw->btwc",
            self.W(x),
            A
        )

        out = out + self.bias

        out = self.bn(
            out.permute(0,3,1,2)
        )

        out = self.act(out)

        return out.permute(0,2,3,1)

# =========================================================
# STGCN BLOCK
# =========================================================

class STGCNBlock(nn.Module):

    def __init__(
        self,
        in_ch,
        out_ch,
        dropout=0.3
    ):
        super().__init__()

        self.gcn = GraphConv(
            in_ch,
            out_ch
        )

        self.tcn = nn.Sequential(
            nn.Conv2d(
                out_ch,
                out_ch,
                kernel_size=(3,1),
                padding=(1,0),
                groups=out_ch
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        if in_ch != out_ch:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x, A):

        residual = x.permute(0,3,1,2)

        out = self.gcn(
            x,
            A
        ).permute(0,3,1,2)

        out = (
            self.tcn(out)
            + self.residual(residual)
        )

        return out.permute(0,2,3,1)

# =========================================================
# POSITIONAL ENCODING
# =========================================================

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=200):
        super().__init__()

        pe = torch.zeros(
            1,
            max_len,
            d_model
        )

        position = torch.arange(
            max_len
        ).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2
            ).float()
            *
            (
                -math.log(10000.0)
                /
                d_model
            )
        )

        pe[0,:,0::2] = torch.sin(
            position * div_term
        )

        pe[0,:,1::2] = torch.cos(
            position * div_term
        )

        self.register_buffer(
            "pe",
            pe
        )

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# =========================================================
# VSTGCN
# =========================================================

class VSTGCN(nn.Module):

    def __init__(
        self,
        output_size
    ):
        super().__init__()

        channels = [
            IN_CHANNELS
        ] + [
            HIDDEN_DIM
        ] * NUM_GCN_LAYERS

        self.stgcn = nn.ModuleList([
            STGCNBlock(
                channels[i],
                channels[i+1],
                DROPOUT
            )
            for i in range(NUM_GCN_LAYERS)
        ])

        self.pos_encoder = PositionalEncoding(
            HIDDEN_DIM,
            max_len=SEQ_LEN
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_DIM,
            nhead=NUM_ATTN_HEADS,
            dim_feedforward=HIDDEN_DIM * 4,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=NUM_TRANS_LAYERS
        )

        self.spatial_proj = nn.Sequential(
            nn.Linear(
                NUM_JOINTS * HIDDEN_DIM,
                HIDDEN_DIM
            ),
            nn.LayerNorm(HIDDEN_DIM)
        )

        self.fc = nn.Linear(
            HIDDEN_DIM,
            output_size
        )

    def forward(self, x, A):

        x = x.permute(
            0,
            2,
            3,
            1
        )

        for block in self.stgcn:
            x = block(x, A)

        B,T,V,C = x.shape

        x = x.reshape(
            B,
            T,
            V*C
        )

        x = self.spatial_proj(x)

        x = self.pos_encoder(x)

        x = self.transformer(x)

        return self.fc(
            x.mean(dim=1)
        )
    
# =========================================================
# FEATURE EXTRACTION
# ĐỒNG BỘ 100% webcam_demo.py
# =========================================================

PARENTS = np.array([
    0,0,1,2,0,4,5,3,6,0,0,0,0,
    11,12,13,14,15,16,15,16,15,16,11,12,

    15,25,26,27,28,
    25,30,31,32,
    25,34,35,36,
    25,38,39,40,
    25,42,43,44,

    16,46,47,48,49,
    46,51,52,53,
    46,55,56,57,
    46,59,60,61,
    46,63,64,65,
])

# =========================================================
# EXTRACT KEYPOINTS
# =========================================================

def extract_keypoints(results):

    pose = np.full(
        (25,3),
        np.nan,
        dtype=np.float32
    )

    if results.pose_landmarks:

        for i,lm in enumerate(
            results.pose_landmarks.landmark[:25]
        ):

            if lm.visibility >= 0.3:

                pose[i] = [
                    lm.x,
                    lm.y,
                    lm.z
                ]

    if results.left_hand_landmarks:

        left_hand = np.array([
            [lm.x,lm.y,lm.z]
            for lm in
            results.left_hand_landmarks.landmark
        ],dtype=np.float32)

    else:

        left_hand = np.full(
            (21,3),
            np.nan,
            dtype=np.float32
        )

    if results.right_hand_landmarks:

        right_hand = np.array([
            [lm.x,lm.y,lm.z]
            for lm in
            results.right_hand_landmarks.landmark
        ],dtype=np.float32)

    else:

        right_hand = np.full(
            (21,3),
            np.nan,
            dtype=np.float32
        )

    return np.concatenate([
        pose,
        left_hand,
        right_hand
    ])

# =========================================================
# PROCESS FEATURES
# Position + Velocity + Bone
# =========================================================

def process_features(
    raw,
    seq_len,
    num_joints,
    device
):

    flat = (
        pd.DataFrame(
            raw.reshape(
                seq_len,
                -1
            )
        )
        .ffill()
        .bfill()
        .fillna(0.0)
        .to_numpy()
    )

    seq = flat.reshape(
        seq_len,
        num_joints,
        3
    ).astype(np.float32)

    # =====================================
    # SHOULDER NORMALIZATION
    # =====================================

    for t in range(seq_len):

        ls = seq[t,11]
        rs = seq[t,12]

        if (
            np.all(ls == 0)
            and
            np.all(rs == 0)
        ):
            continue

        anchor = (
            ls + rs
        ) / 2.0

        width = np.linalg.norm(
            ls - rs
        )

        seq[t] = (
            seq[t] - anchor
        ) / max(width,1e-6)

    # =====================================
    # VELOCITY
    # =====================================

    vel = np.zeros_like(seq)

    vel[:-1] = (
        seq[1:]
        -
        seq[:-1]
    )

    vel[-1] = vel[-2]

    # =====================================
    # BONE
    # =====================================

    bone = (
        seq
        -
        seq[:,PARENTS,:]
    )

    multi = np.concatenate(
        [
            seq,
            vel,
            bone
        ],
        axis=-1
    )

    tensor = torch.from_numpy(
        np.transpose(
            multi,
            (2,0,1)
        )
    ).float()

    return tensor.unsqueeze(0).to(device)

# =========================================================
# DRAW SKELETON
# =========================================================

def draw_skeleton_glow(
    frame,
    results
):

    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    landmark_style = (
        mp_drawing.DrawingSpec(
            color=(0,255,0),
            thickness=1,
            circle_radius=1
        )
    )

    connection_style = (
        mp_drawing.DrawingSpec(
            color=(0,200,0),
            thickness=2
        )
    )

    if results.left_hand_landmarks:

        mp_drawing.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_style,
            connection_style
        )

    if results.right_hand_landmarks:

        mp_drawing.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_style,
            connection_style
        )

    if results.pose_landmarks:

        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_style,
            connection_style
        )

    return frame

# =========================================================
# SLR ENGINE
# =========================================================

class SLREngine:

    def __init__(self):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # ==========================
        # LABEL MAPPING
        # ==========================

        mapping_path = (
            "Model/label_mapping.json"
            if os.path.exists(
                "Model/label_mapping.json"
            )
            else
            "processed_data/label_mapping_2.json"
        )

        with open(
            mapping_path,
            "r",
            encoding="utf-8"
        ) as f:

            label_map = json.load(f)

        self.idx_to_label = {
            v:k
            for k,v
            in label_map.items()
        }

        num_classes = len(label_map)

        # ==========================
        # GRAPH
        # ==========================

        self.ADJ = build_adjacency().to(
            self.device
        )

        # ==========================
        # MODEL
        # ==========================

        self.model = VSTGCN(
            num_classes
        ).to(self.device)

        model_path = (
            "Model/best_model.pth"
            if os.path.exists(
                "Model/best_model.pth"
            )
            else
            "Model/best_model_2.pth"
        )

        checkpoint = torch.load(
            model_path,
            map_location=self.device
        )

        state_dict = checkpoint.get(
            "model",
            checkpoint
        )

        self.model.load_state_dict(
            state_dict
        )

        self.model.eval()

        print(
            "SLR Model loaded successfully"
        )

        # ==========================
        # MEDIAPIPE
        # ==========================

        self.mp_holistic = (
            mp.solutions.holistic
        )

        self.holistic = (
            self.mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=1
            )
        )

        # ==========================
        # BUFFER
        # ==========================

        self.frames_buffer = deque(
            maxlen=SEQ_LEN
        )

        self.prediction_history = deque(
            maxlen=3
        )

        self.cooldown_counter = 0

        self.word_display_counter = 0

        self.hand_down_frames = 0

        self.current_word = ""

        self.final_sentence = ""

        self.last_confidence = 0.0

        self.translator = None

        self.hand_down_threshold = 0.85
        
        self.mp_lock = threading.Lock()

    # =====================================
    # SET TRANSLATOR
    # =====================================

    def set_translator(
        self,
        translator
    ):
        self.translator = translator

    # =====================================
    # RESET BUFFER
    # =====================================

    def reset_buffer(self):

        self.frames_buffer.clear()

        self.prediction_history.clear()

        self.cooldown_counter = 0

        self.hand_down_frames = 0

        self.current_word = ""

    # =====================================
    # CLOSE
    # =====================================

    def close(self):

        if self.holistic:
            self.holistic.close()

    # =====================================
    # PROCESS FRAME
    # =====================================

    def process_frame(
        self,
        frame,
        draw_skeleton=True
    ):

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        with self.mp_lock:
            results = self.holistic.process(
                rgb
            )

        skeleton_frame = None

        if draw_skeleton:

            skeleton_frame = np.zeros_like(frame) # Nền đen
            skeleton_frame = draw_skeleton_glow(
                skeleton_frame,
                results
            )

        keypoints = extract_keypoints(
            results
        )

        self.frames_buffer.append(
            keypoints
        )

        # =====================================
        # INFERENCE
        # =====================================

        if len(self.frames_buffer) == SEQ_LEN:

            self.cooldown_counter = max(
                0,
                self.cooldown_counter - 1
            )

            if self.cooldown_counter == 0:

                with torch.no_grad():

                    tensor = process_features(
                        np.array(
                            self.frames_buffer
                        ),
                        SEQ_LEN,
                        NUM_JOINTS,
                        self.device
                    )

                    logits = self.model(
                        tensor,
                        self.ADJ
                    )

                    probs = torch.softmax(
                        logits,
                        dim=1
                    )

                    max_prob, pred_idx = probs.max(
                        dim=1
                    )

                    confidence = (
                        max_prob.item()
                    )

                    label = (
                        self.idx_to_label[
                            pred_idx.item()
                        ]
                    )

                    self.last_confidence = (
                        confidence
                    )

                    if (
                        confidence >
                        CONFIDENCE_THRESHOLD
                    ):

                        self.prediction_history.append(
                            label
                        )

                        if (
                            len(
                                self.prediction_history
                            ) == 3
                        ):

                            most_common, count = (
                                Counter(
                                    self.prediction_history
                                )
                                .most_common(1)[0]
                            )

                            if count >= 2:

                                self.current_word = (
                                    most_common
                                )

                                if self.translator:

                                    self.translator.add_gloss(
                                        most_common
                                    )

                                self.cooldown_counter = (
                                    PREDICTION_COOLDOWN
                                )

                                self.word_display_counter = (
                                    30
                                )

                                self.prediction_history.clear()

        # =====================================
        # GIỮ CHỮ HIỂN THỊ
        # =====================================

        if self.word_display_counter > 0:

            self.word_display_counter -= 1

        else:

            self.current_word = ""

        # =====================================
        # HAND DOWN DETECTION
        # =====================================

        translated_sentence = None

        if results.pose_landmarks:

            lw = (
                results
                .pose_landmarks
                .landmark[15]
                .y
            )

            rw = (
                results
                .pose_landmarks
                .landmark[16]
                .y
            )

            if lw > self.hand_down_threshold and rw > self.hand_down_threshold:

                self.hand_down_frames += 1

            else:

                self.hand_down_frames = 0

            if (
                self.hand_down_frames > 45
                and
                self.translator
                and
                self.translator.buffer
            ):

                translated_sentence = (
                    self.translator.finalize_sentence()
                )

                self.final_sentence = (
                    translated_sentence
                )

                self.current_word = ""

                self.hand_down_frames = 0

                self.frames_buffer.clear()

                self.prediction_history.clear()

        return {
            "frame": frame,
            "skeleton": skeleton_frame,
            "word": self.current_word,
            "sentence": self.final_sentence,
            "confidence": round(
                self.last_confidence,
                4
            ),
            "buffer_size":
                len(self.frames_buffer),
            "translated":
                translated_sentence
        }

    # =====================================
    # VIDEO PREDICTION
    # =====================================

    def predict_video(
        self,
        video_path
    ):

        self.reset_buffer()

        if self.translator:

            self.translator.buffer.clear()

        cap = cv2.VideoCapture(
            video_path
        )

        final_words = []

        while cap.isOpened():

            success, frame = cap.read()

            if not success:
                break

            result = self.process_frame(
                frame,
                draw_skeleton=False
            )

            if result["word"]:

                if (
                    len(final_words) == 0
                    or
                    final_words[-1]
                    != result["word"]
                ):

                    final_words.append(
                        result["word"]
                    )

        cap.release()

        sentence = ""

        if self.translator:

            self.translator.buffer = (
                final_words.copy()
            )

            sentence = (
                self.translator.finalize_sentence()
            )

        return {
            "words": final_words,
            "sentence": sentence
        }