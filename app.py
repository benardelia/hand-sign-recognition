"""
ScribeSign web hub: data collection, training and model distribution.

Landmarks are extracted on the client (browser: static/js/pipeline.js, phone:
Flutter app), so the server only receives numbers:
  POST /api/samples          save a recorded (T, 1662) sequence + metadata
  POST /api/stream/frames    live translation for one session (binary float32 frames)
  POST /api/predict          classify one full sequence
  GET  /api/models           trained profiles; files under /api/models/<profile>/<file>
  POST /api/train            train a profile (streams logs on /stream_train_logs)

LOCAL_CAMERA=1 additionally enables the host's own webcam (server/local_camera.py).
"""
import base64
import functools
import hmac
import json
import os
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
from flask import Flask, Response, abort, jsonify, render_template, request, send_from_directory

from utils.dataset import (DATA_PATH, dataset_summary, delete_sample, load_meta, normalize_label,
                           save_sample)
from utils.features import HOLISTIC_SIZE, get_profile, load_profiles, resample_sequence
from utils.inference import MODELS_PATH, trained_profiles
from utils.logger_config import setup_logger
from utils.sessions import SessionStore, parse_frames

logger = setup_logger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024

PORT = 5001
LOCAL_CAMERA = os.environ.get('LOCAL_CAMERA', '0') == '1'
ADMIN_TOKEN = os.environ.get('ADMIN_TOKEN', '')      # required for delete/train when set
COLLECT_TOKEN = os.environ.get('COLLECT_TOKEN', '')  # required for sample upload when set
MODEL_FILES = {'model.tflite', 'labels.txt', 'profile.json', 'feature_spec.json'}
TRAIN_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml', 'train_model.py')

MEDIAPIPE_VERSION = '0.10.14'
MEDIAPIPE = {
    "bundle": f"https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@{MEDIAPIPE_VERSION}/vision_bundle.mjs",
    "wasm": f"https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@{MEDIAPIPE_VERSION}/wasm",
    "model": "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
             "holistic_landmarker/float16/latest/holistic_landmarker.task",
}

os.makedirs(DATA_PATH, exist_ok=True)

sessions = SessionStore()
local_camera = None
if LOCAL_CAMERA:
    try:
        from server.local_camera import LocalCamera
        local_camera = LocalCamera(record_seconds=load_profiles()['dataset']['record_seconds'])
    except Exception as e:
        logger.error(f"LOCAL_CAMERA=1 but local camera mode failed to initialize: {e}")

training_process = None
training_profile = None
training_lock = threading.Lock()


def require_token(kind):
    """Optional shared-secret check (X-Handsign-Token header) for public deployments."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            expected = ADMIN_TOKEN if kind == 'admin' else COLLECT_TOKEN
            given = request.headers.get('X-Handsign-Token', '')
            if expected and not hmac.compare_digest(given, expected):
                return jsonify({"status": "error", "message": "Token required", "token_required": kind}), 401
            return fn(*args, **kwargs)
        return wrapper
    return decorator


def error(message, code=400):
    return jsonify({"status": "error", "message": message}), code


def inside_data_path(path):
    abs_data = os.path.abspath(DATA_PATH)
    abs_path = os.path.abspath(path or '')
    return abs_path if abs_path.startswith(abs_data + os.sep) else None


def decode_frames(payload):
    """Accepts {"frames_b64": base64 float32 LE} or {"frames": [[...1662], ...]}."""
    if payload.get('frames_b64'):
        raw = base64.b64decode(payload['frames_b64'])
        if len(raw) % (HOLISTIC_SIZE * 4):
            raise ValueError(f"frames_b64 must hold N x {HOLISTIC_SIZE} float32 values")
        return np.frombuffer(raw, dtype='<f4').reshape(-1, HOLISTIC_SIZE)
    return np.asarray(payload.get('frames', []), dtype=np.float32)


# --- PAGES & CONFIG ---

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/config')
def get_config():
    cfg = load_profiles()
    trained = trained_profiles()
    profiles = {}
    for name, p in cfg['profiles'].items():
        meta = trained.get(name, {})
        profiles[name] = {"description": p['description'], "target": p['target'], "camera": p['camera'],
                          "sequence_length": p['sequence_length'], "trained": name in trained,
                          "version": meta.get('version'), "metrics": meta.get('metrics')}
    return jsonify({"local_camera": local_camera is not None, "default_profile": cfg['default_profile'],
                    "dataset": cfg['dataset'], "profiles": profiles, "mediapipe": MEDIAPIPE,
                    "collect_token_required": bool(COLLECT_TOKEN), "admin_token_required": bool(ADMIN_TOKEN)})


# --- DATA COLLECTION ---

@app.route('/api/samples', methods=['POST'])
@require_token('collect')
def upload_sample():
    payload = request.get_json(silent=True) or {}
    try:
        seq = decode_frames(payload)
        meta = payload.get('meta') or {}
        if not isinstance(meta, dict):
            raise ValueError("meta must be an object")
        meta['source'] = str(meta.get('source', 'web'))[:20]
        path = save_sample(payload.get('label', ''), seq, meta)
    except ValueError as e:
        return error(str(e))
    logger.info(f"Saved sample {path} ({len(seq)} frames, source={meta['source']})")
    return jsonify({"status": "success", "path": path, "frames": int(len(seq))})


@app.route('/api/signs')
def get_signs():
    return jsonify({"signs": dataset_summary()})


@app.route('/api/npy_files')
def get_npy_files():
    result = {}
    for sign in dataset_summary():
        folder = os.path.join(DATA_PATH, sign['name'])
        files = sorted(f for f in os.listdir(folder) if f.endswith('.npy'))
        result[sign['name']] = [{"filename": f, "path": os.path.join(folder, f)} for f in files]
    return jsonify(result)


@app.route('/api/npy_data')
def get_npy_data():
    filepath = inside_data_path(request.args.get('path', ''))
    if not filepath:
        return jsonify({"error": "Unauthorized path access."}), 403
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
    try:
        data = np.load(filepath)
    except Exception as e:
        logger.error(f"Error loading .npy file: {e}")
        return jsonify({"error": str(e)}), 500
    if data.ndim != 2 or data.shape[1] != HOLISTIC_SIZE:
        return jsonify({"error": f"Expected shape (T, {HOLISTIC_SIZE}), got {data.shape}"}), 400

    def pts(block, stride):
        block = block.reshape(-1, stride)
        if stride == 4:
            return [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2]), "visibility": float(p[3])} for p in block]
        return [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in block]

    frames = [{"pose": pts(f[0:132], 4), "face": pts(f[132:1536], 3),
               "left_hand": pts(f[1536:1599], 3), "right_hand": pts(f[1599:1662], 3)} for f in data]
    return jsonify({"filename": os.path.basename(filepath), "type": "holistic", "frames": frames,
                    "meta": load_meta(filepath)})


@app.route('/api/delete_file', methods=['POST'])
@require_token('admin')
def delete_file():
    path = inside_data_path((request.get_json(silent=True) or {}).get('path', ''))
    if not path or not path.endswith('.npy'):
        return error("Unauthorized path access.", 403)
    if not os.path.exists(path):
        return error("File not found.", 404)
    delete_sample(path)
    logger.info(f"Deleted sample: {path}")
    return jsonify({"status": "success", "message": "File deleted successfully."})


@app.route('/api/delete_sign', methods=['POST'])
@require_token('admin')
def delete_sign():
    label = (request.get_json(silent=True) or {}).get('label', '').strip()
    target = inside_data_path(os.path.join(DATA_PATH, label)) if label else None
    if not target:
        return error("Unauthorized path access.", 403)
    if not os.path.isdir(target):
        return error("Sign dataset directory not found.", 404)
    shutil.rmtree(target)
    logger.info(f"Deleted sign dataset folder: {target}")
    return jsonify({"status": "success", "message": "Sign dataset deleted successfully."})


# --- LIVE TRANSLATION (client-side landmarks) ---

@app.route('/api/stream/frames', methods=['POST'])
def stream_frames():
    session_id = request.headers.get('X-Session-Id', '')[:64]
    if len(session_id) < 8:
        return error("X-Session-Id header required")
    try:
        frames = parse_frames(request.get_data())
        session = sessions.get(session_id, request.headers.get('X-Profile') or None)
    except ValueError as e:
        return error(str(e))
    return jsonify({"status": "success", **session.push(frames)})


@app.route('/api/session/clear', methods=['POST'])
def clear_session():
    session_id = request.headers.get('X-Session-Id', '')[:64]
    try:
        sessions.get(session_id, request.headers.get('X-Profile') or None).clear()
    except ValueError as e:
        return error(str(e))
    return jsonify({"status": "success"})


@app.route('/api/predict', methods=['POST'])
def predict():
    payload = request.get_json(silent=True) or {}
    try:
        clf = sessions.classifier(payload.get('profile'))
        if not clf.loaded:
            return error(f"No trained model for profile '{clf.profile_name}'", 404)
        seq = decode_frames(payload)
        if seq.ndim != 2 or seq.shape[1] != HOLISTIC_SIZE or len(seq) < 2:
            raise ValueError(f"frames must be (T, {HOLISTIC_SIZE})")
        probs = clf.predict(resample_sequence(seq, clf.sequence_length))
    except ValueError as e:
        return error(str(e))
    idx = int(np.argmax(probs))
    return jsonify({"status": "success", "profile": clf.profile_name, "label": clf.labels[idx],
                    "confidence": float(probs[idx]),
                    "probabilities": {l: round(float(p), 4) for l, p in zip(clf.labels, probs)}})


# --- MODEL DISTRIBUTION (phone app) ---

@app.route('/api/models')
def list_models():
    out = {}
    for name, meta in trained_profiles().items():
        files = {f: f"/api/models/{name}/{f}" for f in MODEL_FILES
                 if os.path.exists(os.path.join(MODELS_PATH, name, f))}
        out[name] = {"version": meta.get('version'), "labels": meta.get('labels'),
                     "sequence_length": meta.get('sequence_length'), "camera": meta.get('camera'),
                     "target": meta.get('target'), "metrics": meta.get('metrics'), "files": files}
    return jsonify({"default_profile": load_profiles()['default_profile'], "profiles": out})


@app.route('/api/models/<profile>/<filename>')
def model_file(profile, filename):
    if profile not in load_profiles()['profiles'] or filename not in MODEL_FILES:
        abort(404)
    directory = os.path.abspath(os.path.join(MODELS_PATH, profile))
    if not os.path.exists(os.path.join(directory, filename)):
        abort(404)
    return send_from_directory(directory, filename, max_age=0)


# --- TRAINING ---

@app.route('/api/train', methods=['POST'])
@require_token('admin')
def run_train():
    global training_process, training_profile
    payload = request.get_json(silent=True) or {}
    profile = payload.get('profile') or load_profiles()['default_profile']
    if profile != 'all':
        try:
            profile = get_profile(profile)['name']
        except ValueError as e:
            return error(str(e))
    cmd = [sys.executable, TRAIN_SCRIPT, "--profile", profile]
    for key, flag in (('epochs', '--epochs'), ('batch_size', '--batch-size')):
        if payload.get(key):
            try:
                cmd += [flag, str(int(payload[key]))]
            except (TypeError, ValueError):
                return error(f"{key} must be an integer")

    with training_lock:
        if training_process and training_process.poll() is None:
            return error("Training is already in progress.", 409)
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        try:
            training_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                text=True, bufsize=1, env=env)
        except Exception as e:
            logger.error(f"Failed to start training: {e}")
            return error(str(e), 500)
        training_profile = profile
    logger.info(f"Training started: {' '.join(cmd)}")
    return jsonify({"status": "success", "message": f"Training started for profile '{profile}'."})


@app.route('/stream_train_logs')
def stream_train_logs():
    def generate_logs():
        proc = training_process
        if not proc:
            yield "data: {\"type\": \"log\", \"message\": \"No training process active.\"}\n\n"
            return
        while True:
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is None:
                    time.sleep(0.1)
                    continue
                rc = proc.returncode
                msg = "Training complete!" if rc == 0 else f"Training failed with exit code {rc}"
                if rc == 0:
                    names = list(load_profiles()['profiles']) if training_profile == 'all' else [training_profile]
                    for name in names:
                        sessions.reload(name)
                    if local_camera:
                        local_camera.reload_model()
                yield f"data: {json.dumps({'type': 'status', 'status': 'completed', 'code': rc, 'message': msg})}\n\n"
                break
            line = line.strip()
            if "EPOCH_END:" in line:
                parts = line.split("EPOCH_END:")[1].split(":")
                if len(parts) >= 5:
                    data = {"type": "epoch_metrics", "epoch": int(parts[0]), "accuracy": float(parts[1]),
                            "loss": float(parts[2]), "val_accuracy": float(parts[3]), "val_loss": float(parts[4])}
                    yield f"data: {json.dumps(data)}\n\n"
            elif line:
                yield f"data: {json.dumps({'type': 'log', 'message': line})}\n\n"
    return Response(generate_logs(), mimetype='text/event-stream')


# --- LOCAL CAMERA MODE (LOCAL_CAMERA=1) ---

def local_only(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if local_camera is None:
            return error("Local camera mode is disabled (set LOCAL_CAMERA=1).", 404)
        return fn(*args, **kwargs)
    return wrapper


@app.route('/video_feed')
@local_only
def video_feed():
    local_camera.start()

    def gen():
        while True:
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + local_camera.jpeg() + b'\r\n\r\n'
            time.sleep(0.03)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/status')
@local_only
def get_status():
    return jsonify(local_camera.status())


@app.route('/api/clear_sentence', methods=['POST'])
@local_only
def clear_sentence():
    local_camera.clear_sentence()
    return jsonify({"status": "success"})


@app.route('/api/start_recording', methods=['POST'])
@local_only
def start_recording():
    try:
        label = normalize_label((request.get_json(silent=True) or {}).get('label', ''))
    except ValueError as e:
        return error(str(e))
    ok, msg = local_camera.start_recording(label)
    return (jsonify({"status": "success", "message": msg}) if ok else error(msg))


@app.route('/api/cameras')
@local_only
def get_cameras():
    cams = local_camera.list_cameras() or [{"id": 0, "name": "Camera 0", "active": True}]
    return jsonify({"cameras": cams, "active_id": local_camera.camera_index})


@app.route('/api/select_camera', methods=['POST'])
@local_only
def select_camera():
    try:
        camera_id = int((request.get_json(silent=True) or {}).get('camera_id'))
    except (TypeError, ValueError):
        return error("Invalid Camera ID.")
    local_camera.pending_camera_index = camera_id
    time.sleep(0.6)
    if local_camera.camera_index == camera_id:
        return jsonify({"status": "success", "message": f"Switched to Camera {camera_id}."})
    return error(f"Failed to switch to Camera {camera_id}.", 500)


if __name__ == '__main__':
    import socket

    def find_free_port(start_port):
        for port in range(start_port, start_port + 10):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('127.0.0.1', port)) != 0:
                    return port
        return start_port

    target_port = int(os.environ.get('PORT', PORT))
    actual_port = find_free_port(target_port)
    if actual_port != target_port:
        logger.warning(f"Port {target_port} was in use. Falling back to port {actual_port}.")
    logger.info(f"Starting server on port {actual_port} (local camera: {local_camera is not None})...")
    try:
        app.run(host='0.0.0.0', port=actual_port, debug=False, threaded=True)
    finally:
        if local_camera:
            local_camera.stop()
