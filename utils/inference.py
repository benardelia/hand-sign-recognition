"""
Profile-aware model loading and prediction, shared by app.py and realtime_holistic.py.

A model directory (Model/<profile>/) is self-describing via profile.json, so
switching quality is just a different profile name.
"""
import json
import os
import threading

import numpy as np

from utils.features import HOLISTIC_SIZE, get_profile, load_profiles
from utils.logger_config import setup_logger
from utils.translator import translator

logger = setup_logger(__name__)

MODELS_PATH = 'Model'
LEGACY_MODEL = os.path.join(MODELS_PATH, 'hand_model.h5')
LEGACY_LABELS = os.path.join(MODELS_PATH, 'labels.txt')


def trained_profiles(models_path=MODELS_PATH):
    """Profiles that have a trained model on disk, with their metadata."""
    out = {}
    for name in load_profiles()['profiles']:
        meta_path = os.path.join(models_path, name, 'profile.json')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                out[name] = json.load(f)
    return out


class SignClassifier:
    """Loads Model/<profile>/ (TFLite preferred, Keras fallback). predict() takes a raw (T, 1662) sequence."""

    def __init__(self, profile_name=None, models_path=MODELS_PATH, prefer_tflite=True):
        self.models_path = models_path
        self.prefer_tflite = prefer_tflite
        self.profile_name = get_profile(profile_name)['name']
        self.meta = {}
        self.labels = []
        self.sequence_length = get_profile(self.profile_name)['sequence_length']
        self.backend = None
        self.loaded = False
        self._lock = threading.Lock()
        self._model = None
        self._interp = None
        self.load()

    def load(self):
        model_dir = os.path.join(self.models_path, self.profile_name)
        meta_path = os.path.join(model_dir, 'profile.json')
        self.loaded = False
        try:
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    self.meta = json.load(f)
                self.labels = self.meta['labels']
                self.sequence_length = self.meta['sequence_length']
                tfl = os.path.join(model_dir, 'model.tflite')
                if self.prefer_tflite and os.path.exists(tfl):
                    import tensorflow as tf
                    self._interp = tf.lite.Interpreter(model_path=tfl)
                    self._interp.allocate_tensors()
                    self.backend = 'tflite'
                else:
                    import tensorflow as tf
                    import utils.tf_layers  # noqa: F401  registers FeaturePreprocess
                    self._model = tf.keras.models.load_model(os.path.join(model_dir, 'model.keras'))
                    self.backend = 'keras'
                self.loaded = True
            elif os.path.exists(LEGACY_MODEL) and os.path.exists(LEGACY_LABELS):
                # Pre-profile model: raw (30, 1662) input, no in-graph preprocessing.
                from tensorflow.keras.models import load_model
                model = load_model(LEGACY_MODEL)
                if model.input_shape[-1] == HOLISTIC_SIZE:
                    self._model = model
                    self.sequence_length = model.input_shape[1] or 30
                    with open(LEGACY_LABELS) as f:
                        self.labels = f.read().splitlines()
                    self.meta = {"name": "legacy", "version": "legacy"}
                    self.backend = 'keras-legacy'
                    self.loaded = True
                    logger.warning(f"No model for profile '{self.profile_name}'; using legacy {LEGACY_MODEL}. "
                                   f"Run: python train_model.py --profile {self.profile_name}")
            if self.loaded:
                logger.info(f"Loaded model profile={self.profile_name} backend={self.backend} "
                            f"T={self.sequence_length} labels={self.labels}")
            else:
                logger.info(f"No trained model for profile '{self.profile_name}'. "
                            f"Train one with: python train_model.py --profile {self.profile_name}")
        except Exception as e:
            logger.error(f"Failed to load model for profile '{self.profile_name}': {e}")
        return self.loaded

    def predict(self, sequence):
        seq = np.asarray(sequence, dtype=np.float32)
        if seq.shape != (self.sequence_length, HOLISTIC_SIZE):
            raise ValueError(f"Expected sequence shape ({self.sequence_length}, {HOLISTIC_SIZE}), got {seq.shape}")
        with self._lock:
            if self.backend == 'tflite':
                inp = self._interp.get_input_details()[0]
                out = self._interp.get_output_details()[0]
                self._interp.set_tensor(inp['index'], seq[None])
                self._interp.invoke()
                return self._interp.get_tensor(out['index'])[0]
            return self._model.predict(seq[None], verbose=0)[0]

    def info(self):
        return {"profile": self.profile_name, "loaded": self.loaded, "backend": self.backend,
                "sequence_length": self.sequence_length, "labels": self.labels,
                "version": self.meta.get('version')}


class GlossDecoder:
    """Turns per-window probabilities into a stable gloss stream and a sentence."""

    def __init__(self, labels, threshold=0.85, stability_frames=5, max_glosses=5):
        self.labels = labels
        self.threshold = threshold
        self.stability_frames = stability_frames
        self.max_glosses = max_glosses
        self.reset()

    def reset(self, sentence="Start signing..."):
        self.gloss_buffer = []
        self.sentence = sentence
        self.last_prediction = ""
        self.confidence = 0.0
        self.counter = 0

    def update(self, probs):
        """Returns (label, confidence) of this window when above threshold, else (None, confidence)."""
        index = int(np.argmax(probs))
        self.confidence = float(probs[index])
        if self.confidence <= self.threshold:
            return None, self.confidence
        word = self.labels[index]
        if word == self.last_prediction:
            self.counter += 1
        else:
            self.counter = 0
            self.last_prediction = word
        if self.counter == self.stability_frames:
            if not self.gloss_buffer or self.gloss_buffer[-1] != word:
                self.gloss_buffer = (self.gloss_buffer + [word])[-self.max_glosses:]
                self.sentence = translator.translate(self.gloss_buffer)
                logger.info(f"Detected gloss: {word} | Sentence: {self.sentence}")
            self.counter = 0
        return word, self.confidence
