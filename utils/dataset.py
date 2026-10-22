"""Dataset storage: raw holistic sequences (.npy) with a metadata sidecar (.json)."""
import json
import os
import re
import time
import uuid

import numpy as np

from utils.features import HOLISTIC_SIZE

DATA_PATH = os.path.join('Data', 'Holistic_Landmarks')
LABEL_RE = re.compile(r"^[A-Z0-9][A-Z0-9 _\-']{0,39}$")
MIN_FRAMES, MAX_FRAMES = 6, 300
MAX_META_BYTES = 4096


def normalize_label(label):
    label = re.sub(r"\s+", " ", (label or "").strip().upper())
    if not LABEL_RE.match(label):
        raise ValueError("Label must be 1-40 characters: letters, digits, space, _ - '")
    return label


def validate_sequence(seq):
    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim != 2 or seq.shape[1] != HOLISTIC_SIZE:
        raise ValueError(f"Sequence must have shape (T, {HOLISTIC_SIZE}), got {seq.shape}")
    if not MIN_FRAMES <= seq.shape[0] <= MAX_FRAMES:
        raise ValueError(f"Sequence must have {MIN_FRAMES}-{MAX_FRAMES} frames, got {seq.shape[0]}")
    if not np.all(np.isfinite(seq)):
        raise ValueError("Sequence contains NaN or infinite values")
    if np.abs(seq).max() > 10:
        raise ValueError("Landmark values out of range (expected normalized coordinates)")
    if not np.any(seq[:, 1536:] != 0):
        raise ValueError("No hands detected in any frame; sample rejected")
    return seq


def save_sample(label, seq, meta=None, data_path=DATA_PATH):
    """Saves a validated sample, returns its path relative to data_path's parent."""
    label = normalize_label(label)
    seq = validate_sequence(seq)
    meta = dict(meta or {})
    if len(json.dumps(meta)) > MAX_META_BYTES:
        raise ValueError("Metadata too large")

    # Reuse an existing folder that differs only by case (e.g. legacy "Hello" vs "HELLO").
    if os.path.isdir(data_path):
        for existing in os.listdir(data_path):
            if existing.upper() == label and os.path.isdir(os.path.join(data_path, existing)):
                label = existing
                break
    target_dir = os.path.join(data_path, label)
    os.makedirs(target_dir, exist_ok=True)
    stem = f"holistic_seq_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
    np.save(os.path.join(target_dir, stem + '.npy'), seq)
    meta.update({"label": label, "frames": int(seq.shape[0]), "features": HOLISTIC_SIZE,
                 "saved_at": time.strftime('%Y-%m-%dT%H:%M:%S%z')})
    with open(os.path.join(target_dir, stem + '.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    return os.path.join(target_dir, stem + '.npy')


def load_meta(npy_path):
    meta_path = os.path.splitext(npy_path)[0] + '.json'
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                return json.load(f)
        except (OSError, ValueError):
            pass
    return {}


def dataset_summary(data_path=DATA_PATH):
    """Per-label sample counts, split by collection source."""
    out = []
    if not os.path.exists(data_path):
        return out
    for label in sorted(d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))):
        folder = os.path.join(data_path, label)
        sources = {}
        files = [f for f in os.listdir(folder) if f.endswith('.npy')]
        for f in files:
            src = load_meta(os.path.join(folder, f)).get('source', 'unknown')
            sources[src] = sources.get(src, 0) + 1
        out.append({"name": label, "samples": len(files), "sources": sources})
    return out


def delete_sample(npy_path):
    os.remove(npy_path)
    meta_path = os.path.splitext(npy_path)[0] + '.json'
    if os.path.exists(meta_path):
        os.remove(meta_path)
