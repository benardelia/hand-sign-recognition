"""
Server-side webcam mode (LOCAL_CAMERA=1): the Flask host's own camera, processed
with Python MediaPipe Holistic. Meant for a local machine / the experimental
profile. Remote deployments use the browser pipeline (static/js/pipeline.js),
so cv2 and mediapipe are only imported here.
"""
import os
import threading
import time

os.environ.setdefault("OPENCV_LOG_LEVEL", "OFF")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from mediapipe.python.solutions import drawing_utils as mp_drawing  # noqa: E402
from mediapipe.python.solutions import face_mesh as mp_face_mesh  # noqa: E402
from mediapipe.python.solutions import holistic as mp_holistic  # noqa: E402

from utils.dataset import save_sample  # noqa: E402
from utils.features import get_profile  # noqa: E402
from utils.holistic_utils import extract_holistic_keypoints  # noqa: E402
from utils.inference import GlossDecoder, SignClassifier  # noqa: E402
from utils.logger_config import setup_logger  # noqa: E402

logger = setup_logger(__name__)


class LocalCamera:
    def __init__(self, profile_name=None, record_seconds=1.0):
        self.profile = get_profile(profile_name)
        self.record_seconds = record_seconds
        self.lock = threading.Lock()
        self.is_running = False
        self.thread = None
        self.cap = None
        self.camera_index = 0
        self.pending_camera_index = None
        self.annotated_frame = None

        self.classifier = SignClassifier(self.profile['name'])
        self.decoder = GlossDecoder(self.classifier.labels)
        self.sequence = []

        self.recording_label = None
        self.recording_status = "idle"  # idle, countdown, recording, saved
        self.recording_frames = []
        self.recording_times = []
        self.phase_start = 0.0
        self.countdown_duration = 3.0

    # --- lifecycle ---
    def start(self):
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

    def reload_model(self):
        with self.lock:
            self.classifier = SignClassifier(self.profile['name'])
            self.decoder = GlossDecoder(self.classifier.labels)
            self.decoder.reset("Model reloaded. Start signing...")
            self.sequence = []

    def _open(self, index):
        cap = cv2.VideoCapture(index)
        cam = self.profile['camera']
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam['height'])
        return cap

    def _run(self):
        self.cap = self._open(self.camera_index)
        if not self.cap.isOpened():
            logger.warning("No webcam on this host; local camera mode unavailable.")
            self.is_running = False
            return
        complexity = 2 if self.profile['target'] == 'desktop' else 1
        with mp_holistic.Holistic(model_complexity=complexity, min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5) as holistic:
            while self.is_running:
                if self.pending_camera_index is not None:
                    self.cap.release()
                    self.camera_index = self.pending_camera_index
                    self.pending_camera_index = None
                    self.cap = self._open(self.camera_index)
                    if not self.cap.isOpened():
                        logger.error(f"Camera {self.camera_index} failed; reverting to 0.")
                        self.camera_index = 0
                        self.cap = self._open(0)
                ok, frame = self.cap.read()
                if not ok:
                    time.sleep(0.03)
                    continue
                try:
                    self._process(frame, holistic)
                except Exception as e:
                    logger.error(f"Error processing camera frame: {e}")

    # --- per-frame work ---
    def _process(self, raw, holistic):
        frame = cv2.flip(raw, 1)  # selfie view: the pipeline-wide convention
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = holistic.process(rgb)
        image = frame
        if results.face_landmarks:
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                                      None, mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1))
        for lms, conns in ((results.pose_landmarks, mp_holistic.POSE_CONNECTIONS),
                           (results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS),
                           (results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)):
            if lms:
                mp_drawing.draw_landmarks(image, lms, conns)
        keypoints = extract_holistic_keypoints(results).astype(np.float32)
        now = time.time()
        h, w, _ = image.shape

        with self.lock:
            if self.recording_status == "countdown":
                remaining = self.countdown_duration - (now - self.phase_start)
                text = f"{int(remaining) + 1}" if remaining > 0 else "GO!"
                cv2.putText(image, text, (w // 2 - 30, h // 2 + 20), cv2.FONT_HERSHEY_DUPLEX, 3.0, (0, 255, 255), 5)
                if remaining <= 0:
                    self.recording_status = "recording"
                    self.phase_start = now
                    self.recording_frames, self.recording_times = [], []
            elif self.recording_status == "recording":
                self.recording_frames.append(keypoints)
                self.recording_times.append(now)
                cv2.rectangle(image, (0, 0), (w, h), (0, 0, 255), 3)
                cv2.putText(image, "RECORDING...", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                if now - self.phase_start >= self.record_seconds:
                    self._save_recording()
            elif self.recording_status == "saved":
                cv2.putText(image, "SEQUENCE SAVED!", (w // 2 - 150, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                if now - self.phase_start > 1.5:
                    self.recording_status, self.recording_label = "idle", None
            elif self.classifier.loaded:
                t = self.classifier.sequence_length
                self.sequence = (self.sequence + [keypoints])[-t:]
                if len(self.sequence) == t:
                    word, conf = self.decoder.update(self.classifier.predict(np.stack(self.sequence)))
                    if word:
                        cv2.putText(image, f"{word} ({int(conf * 100)}%)", (15, 45),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            self.annotated_frame = image

    def _save_recording(self):
        frames, times = self.recording_frames, self.recording_times
        try:
            t0 = times[0] if times else 0
            save_sample(self.recording_label, np.stack(frames), {
                "source": "pc", "profile": self.profile['name'], "mirrored": True,
                "fps": round(len(frames) / max(1e-3, times[-1] - t0), 2) if len(times) > 1 else None,
                "timestamps_ms": [round((t - t0) * 1000, 1) for t in times],
                "resolution": list(self.annotated_frame.shape[1::-1]) if self.annotated_frame is not None else None,
            })
            self.recording_status = "saved"
        except Exception as e:
            logger.error(f"Recording rejected: {e}")
            self.recording_status = "idle"
            self.recording_label = None
        self.phase_start = time.time()

    # --- API helpers ---
    def start_recording(self, label):
        with self.lock:
            if not self.is_running:
                return False, "Local camera is not running."
            if self.recording_status != "idle":
                return False, f"Camera is busy ({self.recording_status})."
            self.recording_label = label
            self.recording_status = "countdown"
            self.phase_start = time.time()
            return True, f"Countdown started for '{label}'"

    def clear_sentence(self):
        with self.lock:
            self.decoder.reset("Sentence Cleared.")

    def status(self):
        with self.lock:
            d = self.decoder
            progress = 0.0
            if self.recording_status == "recording":
                progress = min(1.0, (time.time() - self.phase_start) / self.record_seconds)
            return {
                "current_sentence": d.sentence, "last_prediction": d.last_prediction,
                "confidence": round(d.confidence, 3), "gloss_buffer": list(d.gloss_buffer),
                "is_recording": self.recording_status != "idle", "recording_label": self.recording_label or "",
                "recording_status": self.recording_status, "recording_progress": progress,
                "model_loaded": self.classifier.loaded, "labels": self.classifier.labels,
                "profile": self.profile['name'],
            }

    def jpeg(self):
        with self.lock:
            img = self.annotated_frame
            if img is None:
                img = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(img, "Starting camera..." if self.is_running else "Local camera offline",
                            (170, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            return cv2.imencode('.jpg', img)[1].tobytes()

    def list_cameras(self):
        cams = []
        for index in range(5):
            if index == self.camera_index and self.cap is not None and self.cap.isOpened():
                cams.append({"id": index, "name": f"Camera {index} (Active)", "active": True})
                continue
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                cams.append({"id": index, "name": f"Camera {index}", "active": False})
                cap.release()
        return cams
