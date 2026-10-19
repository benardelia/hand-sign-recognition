"""
Quality profiles and feature selection for the holistic landmark pipeline.

Every client (Python, browser, Flutter) produces the same raw frame layout as
utils/holistic_utils.extract_holistic_keypoints (1662 floats). A profile picks
a subset of landmarks from that frame and normalizes it. The selection and
normalization run inside the model graph (FeaturePreprocess), so clients only
have to feed raw (T, 1662) sequences. `select_features` is the numpy reference
implementation used for tests and offline tooling.
"""
import json
import os

import numpy as np

# Raw holistic frame layout (see utils/holistic_utils.py)
HOLISTIC_SIZE = 1662
POSE_COUNT, FACE_COUNT, HAND_COUNT = 33, 468, 21
POSE_OFFSET, POSE_STRIDE = 0, 4          # x, y, z, visibility
FACE_OFFSET, FACE_STRIDE = 132, 3
LH_OFFSET, RH_OFFSET, HAND_STRIDE = 1536, 1599, 3

GROUPS = {
    "pose": (POSE_OFFSET, POSE_STRIDE, POSE_COUNT),
    "face": (FACE_OFFSET, FACE_STRIDE, FACE_COUNT),
    "left_hand": (LH_OFFSET, HAND_STRIDE, HAND_COUNT),
    "right_hand": (RH_OFFSET, HAND_STRIDE, HAND_COUNT),
}
GROUP_ORDER = ["pose", "face", "left_hand", "right_hand"]

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
DEFAULT_CENTER = (0.5, 0.5, 0.0)
DEFAULT_SCALE = 0.25

# MediaPipe pose left/right pairs, used for mirror augmentation
POSE_MIRROR_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
                     (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILES_PATH = os.path.join(PROJECT_ROOT, 'config', 'profiles.json')


def load_profiles(path=PROFILES_PATH):
    with open(path, 'r') as f:
        return json.load(f)


def get_profile(name=None, path=PROFILES_PATH):
    cfg = load_profiles(path)
    name = name or os.environ.get('PROFILE') or cfg['default_profile']
    if name not in cfg['profiles']:
        raise ValueError(f"Unknown profile '{name}'. Available: {list(cfg['profiles'])}")
    profile = dict(cfg['profiles'][name])
    profile['name'] = name
    return profile


def _landmark_ids(group, spec):
    count = GROUPS[group][2]
    if spec == "all":
        return list(range(count))
    return [int(i) for i in (spec or [])]


def build_feature_spec(profile):
    """
    Returns a JSON-serializable description of the profile's features:
    the raw-frame column of x, y and z for every selected landmark, plus the
    normalization reference columns. Exported next to each model.
    """
    points = []
    for group in GROUP_ORDER:
        offset, stride, _ = GROUPS[group]
        for lm in _landmark_ids(group, profile['landmarks'].get(group)):
            base = offset + lm * stride
            points.append({"group": group, "id": lm, "cols": [base, base + 1, base + 2]})

    def pose_cols(lm):
        base = POSE_OFFSET + lm * POSE_STRIDE
        return [base, base + 1, base + 2]

    return {
        "raw_frame_size": HOLISTIC_SIZE,
        "sequence_length": profile['sequence_length'],
        "num_points": len(points),
        "num_features": len(points) * 3,
        "points": points,
        "columns": [c for p in points for c in p["cols"]],
        "normalize": bool(profile.get('normalize', True)),
        "normalization": {
            "method": "shoulder_center_scale",
            "left_shoulder_cols": pose_cols(LEFT_SHOULDER),
            "right_shoulder_cols": pose_cols(RIGHT_SHOULDER),
            "default_center": list(DEFAULT_CENTER),
            "default_scale": DEFAULT_SCALE,
            "note": "Missing landmarks are all-zero and stay zero. center = shoulder midpoint, "
                    "scale = 2D shoulder distance; defaults are used when a shoulder is missing.",
        },
    }


def select_features(seq, spec):
    """numpy reference of FeaturePreprocess. seq: (T, 1662) -> (T, num_features)."""
    seq = np.asarray(seq, dtype=np.float32)
    t = seq.shape[0]
    pts = seq[:, spec['columns']].reshape(t, -1, 3)
    if not spec['normalize']:
        return pts.reshape(t, -1)

    norm = spec['normalization']
    ls = seq[:, norm['left_shoulder_cols']]
    rs = seq[:, norm['right_shoulder_cols']]
    both = (np.abs(ls).max(axis=-1) > 0) & (np.abs(rs).max(axis=-1) > 0)
    center = np.where(both[:, None], (ls + rs) / 2.0, np.array(norm['default_center'], np.float32))
    dist = np.linalg.norm(ls[:, :2] - rs[:, :2], axis=-1)
    scale = np.maximum(np.where(both, dist, norm['default_scale']), 1e-3)

    present = np.abs(pts).max(axis=-1, keepdims=True) > 0
    out = (pts - center[:, None, :]) / scale[:, None, None]
    out = np.where(present, out, 0.0)
    return out.reshape(t, -1).astype(np.float32)


def resample_sequence(seq, length):
    """Linear interpolation along time to a fixed number of frames."""
    seq = np.asarray(seq, dtype=np.float32)
    n = seq.shape[0]
    if n == length:
        return seq
    if n == 1:
        return np.repeat(seq, length, axis=0)
    src = np.linspace(0, n - 1, length)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    w = (src - lo)[:, None].astype(np.float32)
    # Interpolating between a detected and a missing (zero) landmark would
    # create a fake point, so fall back to the nearest frame in that case.
    a, b = seq[lo], seq[hi]
    out = a * (1 - w) + b * w
    nearest = np.where(w < 0.5, a, b)
    mixed = (a == 0) ^ (b == 0)
    return np.where(mixed, nearest, out).astype(np.float32)


def mirror_holistic(seq):
    """Mirror a raw (T, 1662) sequence horizontally: x -> 1 - x, swap left/right hands and pose pairs."""
    seq = np.array(seq, dtype=np.float32, copy=True)
    out = seq.copy()

    def flip_x(block, offset, stride, count):
        cols = offset + np.arange(count) * stride
        vals = block[:, cols]
        block[:, cols] = np.where(vals != 0, 1.0 - vals, 0.0)

    flip_x(out, POSE_OFFSET, POSE_STRIDE, POSE_COUNT)
    flip_x(out, FACE_OFFSET, FACE_STRIDE, FACE_COUNT)
    flip_x(out, LH_OFFSET, HAND_STRIDE, HAND_COUNT)
    flip_x(out, RH_OFFSET, HAND_STRIDE, HAND_COUNT)

    lh = out[:, LH_OFFSET:LH_OFFSET + 63].copy()
    out[:, LH_OFFSET:LH_OFFSET + 63] = out[:, RH_OFFSET:RH_OFFSET + 63]
    out[:, RH_OFFSET:RH_OFFSET + 63] = lh

    for a, b in POSE_MIRROR_PAIRS:
        sa = slice(POSE_OFFSET + a * POSE_STRIDE, POSE_OFFSET + (a + 1) * POSE_STRIDE)
        sb = slice(POSE_OFFSET + b * POSE_STRIDE, POSE_OFFSET + (b + 1) * POSE_STRIDE)
        tmp = out[:, sa].copy()
        out[:, sa] = out[:, sb]
        out[:, sb] = tmp
    return out


def augment_sequence(seq, aug, rng):
    """Random time warp + optional mirror + coordinate noise on a raw (T, 1662) sequence."""
    length = seq.shape[0]
    warp = aug.get('time_warp', 0.0)
    if warp:
        n = max(2, int(round(length * rng.uniform(1 - warp, 1 + warp))))
        seq = resample_sequence(resample_sequence(seq, n), length)
    if aug.get('mirror') and rng.random() < 0.5:
        seq = mirror_holistic(seq)
    noise = aug.get('noise', 0.0)
    if noise:
        seq = np.where(seq != 0, seq + rng.normal(0, noise, seq.shape).astype(np.float32), 0.0)
    return seq.astype(np.float32)


def make_preprocess_layer(spec):
    """Keras layer implementing select_features inside the model graph (TFLite builtin ops only)."""
    from utils.tf_layers import FeaturePreprocess
    norm = spec['normalization']
    return FeaturePreprocess(
        columns=spec['columns'],
        normalize=spec['normalize'],
        left_shoulder_cols=norm['left_shoulder_cols'],
        right_shoulder_cols=norm['right_shoulder_cols'],
        default_center=norm['default_center'],
        default_scale=norm['default_scale'],
        name='feature_preprocess',
    )
