# -*- coding: utf-8 -*-

import os
import cv2
import json
import tempfile
import base64

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_from_directory
)
from flask_socketio import SocketIO, emit

from core.slr_engine import SLREngine
from core.nlp_translator import NLPTranslator

# ==========================================
# FLASK
# ==========================================

app = Flask(__name__)

socketio = SocketIO(app, cors_allowed_origins="*")

UPLOAD_FOLDER = "uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

# ==========================================
# ENGINE
# ==========================================

translator = NLPTranslator()

engine = SLREngine()

engine.set_translator(
    translator
)

# ==========================================
# HOME
# ==========================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )

# ==========================================
# RESET
# ==========================================

@app.route(
    "/reset",
    methods=["POST"]
)
def reset():

    translator.clear()

    engine.reset_buffer()

    return jsonify({
        "success": True
    })

# ==========================================
# TRANSLATE
# ==========================================

@app.route(
    "/translate_now",
    methods=["POST"]
)
def translate_now():

    sentence = (
        translator.finalize_sentence()
    )

    engine.final_sentence = sentence
    engine.reset_buffer()

    return jsonify({
        "success": True,
        "sentence": sentence
    })

# ==========================================
# EXTERNAL API TRANSLATE (Dành cho webcam_demo.py)
# ==========================================

@app.route(
    "/translate",
    methods=["POST"]
)
def api_translate():
    data = request.json
    if not data or "text" not in data:
        return jsonify({"success": False, "message": "Missing text"})
        
    raw_text = data["text"]
    result_sentence = translator.translate_text(raw_text)
    
    return jsonify({
        "success": True,
        "result": result_sentence
    })

# ==========================================
# UPDATE THRESHOLD
# ==========================================

@app.route(
    "/update_threshold",
    methods=["POST"]
)
def update_threshold():
    try:
        data = request.json
        threshold = float(data.get("threshold", 0.85))
        engine.hand_down_threshold = threshold
        return jsonify({"success": True, "threshold": threshold})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

# ==========================================
# PROCESS FRAME
# ==========================================

@app.route(
    "/predict_frame",
    methods=["POST"]
)
def predict_frame():

    if "frame" not in request.files:

        return jsonify({
            "success": False,
            "message": "No frame"
        })

    file = request.files["frame"]

    data = file.read()

    import numpy as np

    image = cv2.imdecode(
        np.frombuffer(
            data,
            np.uint8
        ),
        cv2.IMREAD_COLOR
    )

    draw_skeleton = request.form.get("draw_skeleton", "true") == "true"

    result = engine.process_frame(
        image,
        draw_skeleton=draw_skeleton
    )

    response_data = {
        "success": True,
        "word": result["word"],
        "sentence": result["sentence"],
        "confidence": result["confidence"],
        "buffer_size": translator.buffer_length(),
        "frame_buffer_size": result["buffer_size"],
        "translated": result["translated"]
    }

    if draw_skeleton and result.get("skeleton") is not None:
        # Mã hoá khung xương thành định dạng Base64
        _, buffer = cv2.imencode('.jpg', result["skeleton"])
        skeleton_b64 = base64.b64encode(buffer).decode('utf-8')
        response_data["skeleton_img"] = "data:image/jpeg;base64," + skeleton_b64

    return jsonify(response_data)

# ==========================================
# WEBSOCKET FRAME PROCESS
# ==========================================

@socketio.on("process_frame")
def handle_process_frame(data):
    import numpy as np
    
    image_data = data.get("image")
    draw_skeleton = data.get("draw_skeleton", True)
    
    # Tách chuỗi Base64
    if "," in image_data:
        _, encoded = image_data.split(",", 1)
    else:
        encoded = image_data
        
    img_bytes = base64.b64decode(encoded)
    image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    
    result = engine.process_frame(
        image,
        draw_skeleton=draw_skeleton
    )
    
    response_data = {
        "success": True,
        "word": result["word"],
        "sentence": result["sentence"],
        "confidence": result["confidence"],
        "buffer_size": translator.buffer_length(),
        "frame_buffer_size": result["buffer_size"],
        "translated": result["translated"]
    }

    if draw_skeleton and result.get("skeleton") is not None:
        _, buffer = cv2.imencode('.jpg', result["skeleton"])
        skeleton_b64 = base64.b64encode(buffer).decode('utf-8')
        response_data["skeleton_img"] = "data:image/jpeg;base64," + skeleton_b64
        
    emit("frame_result", response_data)

# ==========================================
# VIDEO UPLOAD
# ==========================================

@app.route(
    "/predict_video",
    methods=["POST"]
)
def predict_video():

    if "video" not in request.files:

        return jsonify({
            "success": False,
            "message": "No video"
        })

    video = request.files["video"]

    temp_path = os.path.join(
        UPLOAD_FOLDER,
        video.filename
    )

    video.save(temp_path)

    result = engine.predict_video(
        temp_path
    )

    try:
        os.remove(temp_path)
    except:
        pass

    return jsonify({
        "success": True,
        "words":
            result["words"],
        "sentence":
            result["sentence"]
    })

# ==========================================
# BUFFER INFO
# ==========================================

@app.route(
    "/buffer"
)
def buffer_info():

    return jsonify({
        "buffer":
            translator.get_buffer(),
        "count":
            translator.buffer_length()
    })

# ==========================================
# LABELS
# ==========================================

@app.route(
    "/labels"
)
def labels():

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

    return jsonify({
        "count":
            len(label_map),
        "labels":
            sorted(
                list(
                    label_map.keys()
                )
            )
    })

# ==========================================
# DICTIONARY API
# ==========================================

@app.route("/api/vocabulary")
def api_vocabulary():

    mapping_path = (
        "Model/label_mapping.json"
        if os.path.exists("Model/label_mapping.json")
        else "processed_data/label_mapping_2.json"
    )

    valid_labels = set()
    if os.path.exists(mapping_path):
        with open(mapping_path, "r", encoding="utf-8") as f:
            label_map = json.load(f)
            valid_labels = set(label_map.keys())

    csv_path = os.path.join("Dataset", "Text", "label.csv")
    label_to_video = {}
    
    if os.path.exists(csv_path):
        import csv
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = str(row.get("LABEL", "")).strip()
                video = str(row.get("VIDEO", "")).strip()
                # Khớp với nhãn đã huấn luyện và bỏ qua nếu từ đó đã có video rồi (lấy video đầu tiên)
                if label in valid_labels and label not in label_to_video:
                    video_path = os.path.join("Dataset", "Videos", video)
                    if os.path.exists(video_path):
                        label_to_video[label] = video

    sorted_labels = sorted(label_to_video.keys())
    result = [{"label": lbl, "video": label_to_video[lbl]} for lbl in sorted_labels]
    
    return jsonify({
        "success": True, 
        "vocabulary": result
    })

@app.route("/dataset_video/<path:filename>")
def serve_dataset_video(filename):
    return send_from_directory(os.path.join("Dataset", "Videos"), filename)

# ==========================================
# SHUTDOWN
# ==========================================

@app.route(
    "/shutdown",
    methods=["POST"]
)
def shutdown():

    try:

        engine.close()

        return jsonify({
            "success": True
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "message": str(e)
        })

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    socketio.run(
        app,
        host="0.0.0.0",
        port=5000,
        debug=True
    )