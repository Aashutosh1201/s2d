import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
import cv2
import torch
import torch.nn as nn
import numpy as np
from models_arch.s2d_model import s2d_base_patch16_224
from models_arch.landmark import LandmarkDetector

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'))

def get_model_paths():
    s2d_path = os.path.join(MODELS_DIR, "s2d_dfew_best.pth")
    if not os.path.exists(s2d_path):
        from huggingface_hub import hf_hub_download
        dl = hf_hub_download(repo_id="cyinen/S2D", filename="ckpts/dfew/with_oversampler/split01_checkpoint-best.pth")
        import shutil
        shutil.copyfile(dl, s2d_path)
    yunet_path = os.path.join(MODELS_DIR, "face_detection_yunet.onnx")
    if not os.path.exists(yunet_path):
        import urllib.request
        urllib.request.urlretrieve("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx", yunet_path)
    landmark_path = os.path.join(MODELS_DIR, "mobilefacenet_model_best.pth.tar")
    return s2d_path, yunet_path, landmark_path

def load_models():
    s2d_path, yunet_path, landmark_path = get_model_paths()
    s2d = s2d_base_patch16_224(pretrained=False, num_classes=7)
    ckpt = torch.load(s2d_path, map_location=device, weights_only=False)
    sd = ckpt['model'] if 'model' in ckpt else ckpt
    s2d.load_state_dict(sd, strict=False)
    s2d.to(device).eval()
    lm = LandmarkDetector(weights_path=landmark_path, map_location=device)
    lm.to(device).eval()
    detector = cv2.FaceDetectorYN.create(
        model=yunet_path,
        config="",
        input_size=(320, 320),
        score_threshold=0.55,
        nms_threshold=0.3,
        top_k=5000
    )
    return s2d, lm, detector

def detect_face(detector, frame):
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    if faces is not None and len(faces) > 0:
        max_area = 0
        best_face = None
        for f in faces:
            fx, fy, fw, fh = f[0:4]
            area = fw * fh
            if area > max_area and fx >= 0 and fy >= 0:
                max_area = area
                best_face = (int(fx), int(fy), int(fw), int(fh))
        return best_face
    return None

def process_faces(detector, frames):
    n = len(frames)
    raw_boxes = [detect_face(detector, f) for f in frames]
    valid = [i for i, b in enumerate(raw_boxes) if b is not None]
    if not valid:
        return []
    smoothed = [None] * n
    for i in range(0, valid[0]):
        smoothed[i] = raw_boxes[valid[0]]
    for i in range(valid[-1] + 1, n):
        smoothed[i] = raw_boxes[valid[-1]]
    for idx in range(len(valid) - 1):
        s_i, e_i = valid[idx], valid[idx + 1]
        b_s, b_e = np.array(raw_boxes[s_i], dtype=float), np.array(raw_boxes[e_i], dtype=float)
        for k in range(s_i, e_i + 1):
            alpha = (k - s_i) / max(1, e_i - s_i)
            smoothed[k] = tuple(map(int, (1 - alpha) * b_s + alpha * b_e))
    cur = np.array(smoothed[0], dtype=float)
    final_boxes = []
    for b in smoothed:
        cur = 0.65 * np.array(b, dtype=float) + 0.35 * cur
        final_boxes.append(tuple(map(int, cur)))
    crops = []
    for i, frame in enumerate(frames):
        ih, iw = frame.shape[:2]
        fx, fy, fw, fh = final_boxes[i]
        cx, cy = fx + fw / 2.0, fy + fh / 2.0
        sz = max(fw, fh) * 1.30
        x1, y1 = int(round(cx - sz / 2.0)), int(round(cy - sz / 2.0))
        x2, y2 = int(round(cx + sz / 2.0)), int(round(cy + sz / 2.0))
        pl, pt = max(0, -x1), max(0, -y1)
        pr, pb = max(0, x2 - iw), max(0, y2 - ih)
        crop = frame[max(0, y1):min(ih, y2), max(0, x1):min(iw, x2)]
        if pl > 0 or pt > 0 or pr > 0 or pb > 0:
            crop = cv2.copyMakeBorder(crop, pt, pb, pl, pr, borderType=cv2.BORDER_REPLICATE)
        resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
        crops.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    return crops

def slice_clips(crops, fps):
    total = len(crops)
    step = max(1, round(fps / 16.0))
    indices = list(range(0, total, step))
    if len(indices) < 16:
        indices = np.linspace(0, total - 1, 16, dtype=int).tolist()
    sampled = [crops[i] for i in indices]
    n = len(sampled)
    stride = 8
    clips = []
    start = 0
    while start + 16 <= n:
        clips.append(sampled[start:start + 16])
        start += stride
    if start < n and (len(clips) == 0 or (start - stride + 16) < n):
        tail = max(0, n - 16)
        clip = sampled[tail:tail + 16]
        while len(clip) < 16:
            clip.append(clip[-1])
        clips.append(clip)
    return clips

def predict(video_path):
    cap = cv2.VideoCapture(video_path)
    if hasattr(cv2, 'CAP_PROP_ORIENTATION_AUTO'):
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frames.append(frame)
    cap.release()
    total_frames = len(frames)
    if total_frames == 0:
        raise ValueError("Could not decode any frames from video.")
    s2d, lm, detector = load_models()
    crops = process_faces(detector, frames)
    if not crops:
        raise ValueError("No face detected in video.")
    clips = slice_clips(crops, fps)
    if not clips:
        raise ValueError("Could not extract clips from video.")
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1, 1)
    clip_probs = []
    with torch.no_grad():
        for clip in clips:
            clip_np = np.stack(clip, axis=0).astype(np.float32) / 255.0
            t = torch.from_numpy(clip_np).permute(3, 0, 1, 2).unsqueeze(0).to(device)
            t = (t - mean) / std
            a = lm(t)
            logits, _ = s2d(t, a)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            clip_probs.append(probs)
    probs_mat = np.array(clip_probs)
    k, num_c = probs_mat.shape
    mean_p = np.mean(probs_mat, axis=0)
    k_top = max(1, int(np.ceil(k * 0.25)))
    peak_p = np.zeros(num_c)
    for c in range(num_c):
        col = np.sort(probs_mat[:, c])[::-1]
        peak_p[c] = np.mean(col[:k_top])
    emotion_names = ["happy", "sad", "neutral", "angry", "surprise", "disgust", "fear"]
    scores = np.zeros(num_c)
    for c, name in enumerate(emotion_names):
        if name == "neutral":
            base = 0.65 * mean_p[c] + 0.35 * peak_p[c]
            scores[c] = base * 0.60
        else:
            scores[c] = 0.45 * mean_p[c] + 0.55 * peak_p[c]
    total_score = np.sum(scores)
    percentages = (scores / total_score) * 100.0 if total_score > 0 else np.ones(num_c) * (100.0 / num_c)
    rounded = np.round(percentages, 1)
    dom_idx = int(np.argmax(rounded))
    diff = round(100.0 - float(np.sum(rounded)), 1)
    rounded[dom_idx] = round(rounded[dom_idx] + diff, 1)
    emotions = {}
    for c, name in enumerate(emotion_names):
        emotions[name] = float(rounded[c])
    dominant_emotion = emotion_names[dom_idx]
    return {
        "dominant_emotion": dominant_emotion,
        "emotions": emotions,
        "total_frames": total_frames
    }

if __name__ == "__main__":
    # Input your video file path here
    video_path = "conf3.mp4"
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    result = predict(video_path)
    print(json.dumps(result, indent=2))
