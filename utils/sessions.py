"""
Per-client live translation sessions.

Browser and phone clients extract landmarks locally and push raw holistic
frames; each session keeps its own frame window and gloss decoder, so many
clients can use the server at once without mixing their streams.
"""
import threading
import time

import numpy as np

from utils.features import HOLISTIC_SIZE, get_profile
from utils.inference import GlossDecoder, SignClassifier

SESSION_TTL = 600  # seconds of inactivity before a session is dropped
MAX_SESSIONS = 200


class Session:
    def __init__(self, classifier):
        self.classifier = classifier
        self.frames = []
        self.decoder = GlossDecoder(classifier.labels)
        self.last_seen = time.time()
        self.lock = threading.Lock()

    def push(self, frames):
        t = self.classifier.sequence_length
        word = None
        with self.lock:
            self.last_seen = time.time()
            for frame in frames:
                self.frames = (self.frames + [frame])[-t:]
                if len(self.frames) == t and self.classifier.loaded:
                    word, _ = self.decoder.update(self.classifier.predict(np.stack(self.frames)))
            return self.state(word)

    def state(self, word=None):
        d = self.decoder
        return {
            "current_sentence": d.sentence,
            "last_prediction": word or d.last_prediction,
            "confidence": round(d.confidence, 3),
            "gloss_buffer": list(d.gloss_buffer),
            "model_loaded": self.classifier.loaded,
            "labels": self.classifier.labels,
            "profile": self.classifier.profile_name,
            "frames_buffered": len(self.frames),
            "sequence_length": self.classifier.sequence_length,
        }

    def clear(self):
        with self.lock:
            self.frames = []
            self.decoder.reset("Sentence Cleared.")


class SessionStore:
    def __init__(self):
        self._sessions = {}
        self._classifiers = {}
        self._lock = threading.Lock()

    def classifier(self, profile_name=None):
        name = get_profile(profile_name)['name']
        with self._lock:
            if name not in self._classifiers:
                self._classifiers[name] = SignClassifier(name)
            return self._classifiers[name]

    def reload(self, profile_name):
        """Called after training: reload the model and reset sessions using it."""
        name = get_profile(profile_name)['name']
        with self._lock:
            self._classifiers.pop(name, None)
            for key in [k for k, s in self._sessions.items() if s.classifier.profile_name == name]:
                del self._sessions[key]

    def get(self, session_id, profile_name=None):
        clf = self.classifier(profile_name)
        with self._lock:
            now = time.time()
            for key in [k for k, s in self._sessions.items() if now - s.last_seen > SESSION_TTL]:
                del self._sessions[key]
            s = self._sessions.get(session_id)
            if s is None or s.classifier is not clf:
                if len(self._sessions) >= MAX_SESSIONS:
                    oldest = min(self._sessions, key=lambda k: self._sessions[k].last_seen)
                    del self._sessions[oldest]
                s = Session(clf)
                self._sessions[session_id] = s
            return s


def parse_frames(raw_bytes):
    """Binary body of little-endian float32, N x 1662."""
    if len(raw_bytes) == 0 or len(raw_bytes) % (HOLISTIC_SIZE * 4) != 0:
        raise ValueError(f"Body must be N x {HOLISTIC_SIZE} float32 values")
    frames = np.frombuffer(raw_bytes, dtype='<f4').reshape(-1, HOLISTIC_SIZE)
    if frames.shape[0] > 60:
        raise ValueError("At most 60 frames per request")
    if not np.all(np.isfinite(frames)):
        raise ValueError("Frames contain NaN or infinite values")
    return [f.astype(np.float32) for f in frames]
