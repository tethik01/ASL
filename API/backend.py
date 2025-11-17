import argparse
import json
import math
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import eventlet
import io
import os
import cv2
import tempfile
from typing import List
import traceback

# --- NEW: Import MediaPipe and extraction logic ---
try:
    import mediapipe as mp
    mp_holistic = mp.solutions.holistic
    print("[INFO] MediaPipe imported successfully.")
except Exception as e:
    print(f"[ERROR] Failed to import MediaPipe: {e}")
    print("Please install MediaPipe: pip install mediapipe")
    mp = None

# ---------------------------
# Constants from robust_extract_513.py
# ---------------------------
POSE_LEN = 33 * 4       # (world_x,y,z, visibility)
HAND_LEN = 21 * 3       # (image_x,y,z)
FACE_LANDMARK_INDICES = [
    361, 323, 340, 389, 347, 288, 356, 346, 261, 454, 397, 365, 435, 
    367, 401, 379, 364, 366, 447, 433, 394, 378, 416, 264, 376, 434, 
    265, 448, 395, 352, 430, 400, 345, 411, 368, 431, 432, 372, 427, 
    422, 251, 0, 248, 166, 405, 383, 10, 179, 369, 301, 5, 142, 424, 
    280, 177, 100, 16, 446, 440, 58, 302, 132, 129, 172, 377, 137, 
    93, 420, 262, 234, 158, 102, 13, 287, 45, 126, 151, 436, 396, 
    127, 244, 425, 354, 215, 338
]
FACE_LEN = len(FACE_LANDMARK_INDICES) * 3  # 85 * 3 = 255 features
POSE_LANDMARKS = 33
HAND_LANDMARKS = 21
# ---------------------------

# --- Model Definition (Must match training) ---
class TemporalTransformerClassifier(nn.Module):
    def __init__(self, input_dim=513, d_model=384, nhead=6, num_layers=6,
                 dim_ff=768, dropout=0.3, max_len=128, num_classes=100):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
        
        self.norm = nn.LayerNorm(d_model)
        self.cls = nn.Linear(d_model, num_classes)
        self.max_len = max_len

    def forward(self, x, length=None):
        B, T, _ = x.shape
        h = self.input_proj(x) + self.pe[:, :T, :]
        
        pad_mask = torch.zeros(B, T, dtype=torch.bool, device=h.device)
        if length is not None:
            pad_mask = torch.arange(T, device=h.device)[None, :] >= length[:, None]
            
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        h = self.norm(h)
        
        valid_mask = ~pad_mask
        pooled = (h * valid_mask.unsqueeze(-1)).sum(1) / valid_mask.sum(1).clamp(min=1).unsqueeze(-1)
        return self.cls(pooled)

# --- Globals ---
app = Flask(__name__)
CORS(app) 
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

model = None
labels = {}
max_len = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
holistic_model = None # NEW: Global holistic model
user_sessions = {}

# --- Model Loading ---
def load_model_config(ckpt_path):
    global model, labels, max_len, holistic_model
    print(f"Loading checkpoint from: {ckpt_path}")
    pkg = torch.load(ckpt_path, map_location=device)
    cfg = pkg["config"]
    
    model = TemporalTransformerClassifier(
        input_dim=cfg["input_dim"],
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_ff=cfg["dim_ff"],
        dropout=cfg.get("dropout", 0.3),
        max_len=cfg["max_len"],
        num_classes=cfg["num_classes"]
    )
    
    model.load_state_dict(pkg["state_dict"])
    model.to(device)
    model.eval()
    
    labels = pkg.get("label_map", {})
    if labels:
        labels = {str(k): v for k, v in labels.items()}
        print(f"Loaded {len(labels)} labels from checkpoint.")
        
    max_len = cfg["max_len"]
    
    print(f"Successfully loaded model with {cfg['num_classes']} classes.")
    print(f"Model max_len set to {max_len}.")

    # --- NEW: Initialize Holistic Model ---
    if mp:
        print("Initializing MediaPipe Holistic model...")
        holistic_model = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1, 
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        print("MediaPipe Holistic model initialized.")
    else:
        print("[WARNING] MediaPipe not found. /predict_mp4_video endpoint will not work.")


# --- Prediction Logic ---
@torch.no_grad()
def predict_frames(frames_list):
    if not isinstance(frames_list, np.ndarray):
        frames_list = np.array(frames_list, dtype=np.float32)

    X = frames_list
    L = X.shape[0]
    
    if L == 0:
        print("[ERROR] predict_frames called with 0 frames.")
        return []
    
    # Pad or trim
    if L > max_len:
        print(f"[INFO] Trimming frames from {L} to {max_len}")
        X = X[:max_len]
        L = max_len
    
    X_tensor = torch.from_numpy(X).float().to(device).unsqueeze(0)  # [1, T, D]
    L_tensor = torch.tensor([L], dtype=torch.long).to(device)       # [1]
    
    logits = model(X_tensor, L_tensor)
    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    
    top_indices = probs.argsort()[::-1][:5]
    
    predictions = [
        {"label": labels.get(str(i), f"class_{i}"), "prob": float(probs[i])}
        for i in top_indices
    ]
    return predictions

# --- NEW: Feature Extraction Logic (from robust_extract_513.py) ---
def _landmarks_to_vec(result) -> np.ndarray:
    """
    Build per-frame feature vector (D,) using the 513-feature recipe.
    Order: Pose (132), Face (255), LHand (63), RHand (63)
    """
    
    # 1. POSE (132 features)
    pose_vec = np.zeros((POSE_LEN,), dtype=np.float32)
    pose_world = result.pose_world_landmarks
    pose_image = result.pose_landmarks
    if pose_world and pose_image:
        for i in range(POSE_LANDMARKS):
            offset = i * 4
            pose_vec[offset + 0] = pose_world.landmark[i].x
            pose_vec[offset + 1] = pose_world.landmark[i].y
            pose_vec[offset + 2] = pose_world.landmark[i].z
            pose_vec[offset + 3] = pose_image.landmark[i].visibility
    
    # 2. FACE (255 features)
    face_vec = np.zeros((FACE_LEN,), dtype=np.float32)
    face_image = result.face_landmarks
    if face_image:
        for i, lm_index in enumerate(FACE_LANDMARK_INDICES):
            if 0 <= lm_index < len(face_image.landmark):
                offset = i * 3
                lm = face_image.landmark[lm_index]
                face_vec[offset + 0] = lm.x
                face_vec[offset + 1] = lm.y
                face_vec[offset + 2] = lm.z

    # 3. LEFT HAND (63 features)
    lhand_vec = np.zeros((HAND_LEN,), dtype=np.float32)
    lhand_image = result.left_hand_landmarks
    if lhand_image:
        for i in range(HAND_LANDMARKS):
            offset = i * 3
            lm = lhand_image.landmark[i]
            lhand_vec[offset + 0] = lm.x
            lhand_vec[offset + 1] = lm.y
            lhand_vec[offset + 2] = lm.z

    # 4. RIGHT HAND (63 features)
    rhand_vec = np.zeros((HAND_LEN,), dtype=np.float32)
    rhand_image = result.right_hand_landmarks
    if rhand_image:
        for i in range(HAND_LANDMARKS):
            offset = i * 3
            lm = rhand_image.landmark[i]
            rhand_vec[offset + 0] = lm.x
            rhand_vec[offset + 1] = lm.y
            rhand_vec[offset + 2] = lm.z
    
    # 5. Concatenate
    parts = [pose_vec, face_vec, lhand_vec, rhand_vec]
    vec = np.concatenate(parts, axis=0).astype(np.float32)
    np.nan_to_num(vec, copy=False) # Replace NaNs with 0
    return vec

# --- Socket.IO Events ---
@socketio.on('connect')
def on_connect():
    print(f"Client connected: {request.sid}")
    user_sessions[request.sid] = {"mode": "live", "frames": []}

@socketio.on('disconnect')
def on_disconnect():
    print(f"Client disconnected: {request.sid}")
    if request.sid in user_sessions:
        del user_sessions[request.sid]

@socketio.on('set_mode')
def on_set_mode(data):
    mode = data.get('mode', 'live')
    if request.sid in user_sessions:
        user_sessions[request.sid]["mode"] = mode
        user_sessions[request.sid]["frames"] = [] # Clear buffer on mode change
    print(f"Client {request.sid} set mode to {mode}")

@socketio.on('frame')
def on_frame(data):
    """Handles live frame streaming for webcam."""
    session = user_sessions.get(request.sid)
    if not session:
        return

    if session["mode"] == 'live':
        session["frames"].append(data)
        if len(session["frames"]) > max_len:
            session["frames"] = session["frames"][-max_len:]
        if len(session["frames"]) > 10: 
            predictions = predict_frames(session["frames"])
            emit('prediction', predictions)
            
# --- HTTP Endpoints for File Processing ---

# --- NEW: MP4 Video Processing Endpoint ---
@app.route('/predict_mp4_video', methods=['POST'])
def predict_mp4_video():
    global holistic_model
    if 'mp4_file' not in request.files:
        print("[ERROR] '/predict_mp4_video' called but 'mp4_file' not in request.files")
        return jsonify({"error": "No file part"}), 400
    
    if not holistic_model:
        print("[ERROR] MediaPipe model is not initialized.")
        return jsonify({"error": "MediaPipe model not initialized on server."}), 500
        
    file = request.files['mp4_file']
    if file.filename == '':
        print("[ERROR] 'mp4_file' was empty.")
        return jsonify({"error": "No selected file"}), 400

    print(f"[INFO] Received file: {file.filename}, Content-Type: {file.content_type}, Size: {file.content_length} bytes")
    
    # --- FIX: Save temp file in the *current directory* to avoid PermissionError ---
    temp_dir = os.path.dirname(os.path.abspath(__file__))
    temp_f_name = None
    try:
        # Create a temp file handle
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir=temp_dir) as temp_f:
            temp_f_name = temp_f.name
            file.save(temp_f_name)
        
        print(f"[INFO] Saved to temp file: {temp_f_name}")
        
        # --- Use EXACT logic from robust_extract_513.py ---
        cap = cv2.VideoCapture(temp_f_name)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open temp video file")

        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if not src_fps or math.isnan(src_fps) or src_fps <= 0:
            src_fps = 30.0
        
        target_fps = 20 # From robust_extract_513.py default
        stride = max(1, int(round(src_fps / float(target_fps))))
        print(f"[INFO] Video FPS: {src_fps:.2f}, Target FPS: {target_fps}, Stride: {stride}")

        frames: List[np.ndarray] = []
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if (i % stride) != 0:
                i += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = holistic_model.process(rgb)
            x = _landmarks_to_vec(res)  # (D,)
            frames.append(x)
            i += 1

        cap.release()
        print(f"[INFO] Extracted {len(frames)} frames from MP4.")
        # --- End of robust_extract logic ---

        if not frames:
            print("[ERROR] No frames extracted from video.")
            return jsonify({"error": "No frames extracted from video."}), 400

        # Predict on the collected frames
        predictions = predict_frames(np.stack(frames, axis=0))
        
        if predictions:
            print(f"MP4 Prediction Successful. Top guess: {predictions[0]['label']}")
        
        return jsonify(predictions)

    except Exception as e:
        print(f"--- MP4 Video Prediction Error ---")
        print(f"Error: {e}")
        print(traceback.format_exc())
        print(f"----------------------------------")
        return jsonify({"error": str(e)}), 500
    finally:
        # --- NEW: Ensure temp file is always deleted ---
        if temp_f_name and os.path.exists(temp_f_name):
            try:
                os.remove(temp_f_name)
                print(f"[INFO] Cleaned up temp file: {temp_f_name}")
            except Exception as e:
                print(f"[WARNING] Failed to clean up temp file: {e}")


@app.route('/predict_npz', methods=['POST'])
def predict_npz():
    """Handles NPZ file upload for direct prediction."""
    if 'npz_file' not in request.files:
        print("[ERROR] '/predict_npz' called but 'npz_file' not in request.files")
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['npz_file']
    if file.filename == '':
        print("[ERROR] 'npz_file' was empty.")
        return jsonify({"error": "No selected file"}), 400

    print(f"[INFO] Received file: {file.filename}, Content-Type: {file.content_type}, Size: {file.content_length} bytes")

    try:
        file_bytes = file.read()
        file_like_object = io.BytesIO(file_bytes)
        print(f"[INFO] Read {len(file_bytes)} bytes into memory buffer.")
        
        data = np.load(file_like_object)
        print(f"[INFO] NPZ file loaded. Keys: {list(data.keys())}")

        if 'inputs' not in data:
            print("[ERROR] NPZ file missing 'inputs' key")
            return jsonify({"error": "NPZ file missing 'inputs' key"}), 400
            
        frames = data['inputs']
        
        frames_shape = frames.shape
        frames_dtype = frames.dtype
        print(f"[INFO] 'inputs' array loaded. Shape: {frames_shape}, DType: {frames_dtype}")

        if len(frames_shape) != 2 or frames_shape[1] != 513:
             print(f"[ERROR] Invalid 'inputs' shape. Expected (T, 513), got {frames_shape}")
             return jsonify({"error": f"Invalid NPZ shape. Expected (T, 513), got {frames_shape}"}), 400

        print(f"Processing NPZ. Total frames: {frames_shape[0]}")
        
        predictions = predict_frames(frames)
        
        if predictions:
            print(f"NPZ Prediction Successful. Top guess: {predictions[0]['label']}")
        
        return jsonify(predictions)

    except Exception as e:
        print(f"--- NPZ Prediction Error ---")
        print(f"Error: {e}")
        print(traceback.format_exc())
        print(f"----------------------------")
        return jsonify({"error": str(e)}), 500

# --- Main ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint (best.pt)")
    parser.add_argument("--labels", required=True, help="Path to labels.json")
    args = parser.parse_args()

    # Load the model from the checkpoint
    load_model_config(args.ckpt)
    
    # Load external labels to override/ensure string keys
    try:
        with open(args.labels, 'r') as f:
            labels_from_json = json.load(f)
            labels = {str(k): v for k, v in labels_from_json.items()}
        print(f"Loaded and verified {len(labels)} labels from {args.labels}")
    except Exception as e:
        print(f"Warning: Could not load external {args.labels}. Using labels from checkpoint. Error: {e}")

    print(f"Starting server on http://127.0.0.1:5001")
    socketio.run(app, host='127.0.0.1', port=5001)