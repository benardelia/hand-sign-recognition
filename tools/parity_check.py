"""
Landmark parity check: legacy MediaPipe Holistic (Python collectors, existing data)
vs the Tasks HolisticLandmarker (same graph as the browser and the Flutter app).

The two APIs live in different MediaPipe versions (legacy `solutions` is gone in
1.x, and the 0.10.14 Python Tasks holistic is broken), so extract with each one
and compare the saved frames:

    # project venv (mediapipe 0.10.x, legacy API)
    python tools/parity_check.py extract --backend legacy --out legacy.npz img1.jpg img2.jpg
    # any venv with a recent mediapipe (Tasks API)
    python tools/parity_check.py extract --backend tasks --model holistic_landmarker.task --out tasks.npz img1.jpg img2.jpg
    python tools/parity_check.py compare legacy.npz tasks.npz

Images are mirrored before extraction, as in every collector.
holistic_landmarker.task: https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PARTS = {"pose": slice(0, 132), "face": slice(132, 1536), "left_hand": slice(1536, 1599),
         "right_hand": slice(1599, 1662)}


def load_mirrored_rgb(path):
    import cv2
    bgr = cv2.imread(path)
    if bgr is None:
        raise SystemExit(f"Cannot read image {path}")
    return np.ascontiguousarray(cv2.cvtColor(cv2.flip(bgr, 1), cv2.COLOR_BGR2RGB))


def extract_legacy(paths):
    from mediapipe.python.solutions import holistic as mp_holistic
    from utils.holistic_utils import extract_holistic_keypoints
    with mp_holistic.Holistic(static_image_mode=True, model_complexity=1) as legacy:
        return [extract_holistic_keypoints(legacy.process(load_mirrored_rgb(p))).astype(np.float32) for p in paths]


def tasks_frame(result):
    """Same layout as static/js/pipeline.js resultToFrame() and the Flutter plugin."""
    out = np.zeros(1662, dtype=np.float32)

    def first(lms):
        if not lms:
            return None
        return lms[0] if isinstance(lms[0], list) else lms

    pose = first(result.pose_landmarks)
    if pose:
        for i, p in enumerate(pose[:33]):
            out[i * 4:i * 4 + 4] = [p.x, p.y, p.z, p.visibility or 0.0]
    for lms, offset, count in ((result.face_landmarks, 132, 468), (result.left_hand_landmarks, 1536, 21),
                               (result.right_hand_landmarks, 1599, 21)):
        pts = first(lms)
        if pts:
            for i, p in enumerate(pts[:count]):
                out[offset + i * 3:offset + i * 3 + 3] = [p.x, p.y, p.z]
    return out


def extract_tasks(paths, model):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    frames = []
    for p in paths:
        # VIDEO mode over a few repeated frames lets the hand/face trackers lock on, like a live stream.
        lm = vision.HolisticLandmarker.create_from_options(vision.HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model), running_mode=vision.RunningMode.VIDEO))
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=load_mirrored_rgb(p))
        for i in range(5):
            result = lm.detect_for_video(img, i * 33)
        frames.append(tasks_frame(result))
        lm.close()
    return frames


def compare_frames(name, a, b):
    print(f"\n== {name}")
    for part, sl in PARTS.items():
        pa, pb = a[sl], b[sl]
        has_a, has_b = bool(np.any(pa)), bool(np.any(pb))
        if not (has_a and has_b):
            print(f"  {part:10s} legacy={'yes' if has_a else 'no ':3s} tasks={'yes' if has_b else 'no '}")
            continue
        stride = 4 if part == "pose" else 3
        ra, rb = pa.reshape(-1, stride), pb.reshape(-1, stride)
        xy = np.abs(ra[:, :2] - rb[:, :2])
        z = np.abs(ra[:, 2] - rb[:, 2])
        print(f"  {part:10s} xy mean {xy.mean():.4f} max {xy.max():.4f} | z mean {z.mean():.4f}")
    la, ra_ = a[PARTS["left_hand"]].reshape(-1, 3), a[PARTS["right_hand"]].reshape(-1, 3)
    lb, rb_ = b[PARTS["left_hand"]].reshape(-1, 3), b[PARTS["right_hand"]].reshape(-1, 3)
    for legacy_hand, label in ((la, "left"), (ra_, "right")):
        if not np.any(legacy_hand):
            continue
        d_left = np.abs(legacy_hand[:, :2] - lb[:, :2]).mean() if np.any(lb) else np.inf
        d_right = np.abs(legacy_hand[:, :2] - rb_[:, :2]).mean() if np.any(rb_) else np.inf
        if np.isinf(d_left) and np.isinf(d_right):
            continue
        matched = "left" if d_left < d_right else "right"
        verdict = "consistent" if matched == label else "SWAPPED"
        print(f"  legacy {label} hand -> tasks {matched} hand: {verdict}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    ex = sub.add_parser('extract')
    ex.add_argument('--backend', choices=['legacy', 'tasks'], required=True)
    ex.add_argument('--model', help="holistic_landmarker.task (tasks backend)")
    ex.add_argument('--out', required=True)
    ex.add_argument('images', nargs='+')
    cmp_ = sub.add_parser('compare')
    cmp_.add_argument('legacy')
    cmp_.add_argument('tasks')
    args = parser.parse_args()

    if args.cmd == 'extract':
        if args.backend == 'tasks' and not args.model:
            parser.error("--model is required for the tasks backend")
        frames = extract_legacy(args.images) if args.backend == 'legacy' else extract_tasks(args.images, args.model)
        np.savez(args.out, frames=np.stack(frames), names=np.array([os.path.basename(p) for p in args.images]))
        print(f"Saved {len(frames)} frames to {args.out}")
    else:
        a, b = np.load(args.legacy), np.load(args.tasks)
        for name, fa, fb in zip(a['names'], a['frames'], b['frames']):
            compare_frames(str(name), fa, fb)


if __name__ == '__main__':
    main()
