"""
Import downloaded data into the training dataset (Data/Holistic_Landmarks).

1. Images and videos, one folder per sign (the folder name is the label):

       downloads/
         THANK YOU/  clip1.mp4  clip2.mov  photo.jpg
         HELLO/      ...

       python tools/import_media.py media downloads/ --dataset-name "my-dictionary"

   Videos are processed frame by frame with MediaPipe Holistic (as the collectors do),
   trimmed to the part where hands are visible, and limited to --max-seconds.
   Images become a static sample (one frame repeated): only useful for signs that do
   not move, such as fingerspelling handshapes.

2. Kaggle "Google - Isolated Sign Language Recognition" landmark files:

       python tools/import_media.py kaggle path/to/asl-signs --labels hello thankyou --limit-per-label 50

   Reads train.csv + the parquet files (already MediaPipe Holistic landmarks), converts
   them to the 1662-float layout. Note: that dataset is American Sign Language.

Mirroring: our collectors mirror the camera image (selfie view, config/profiles.json).
Images and videos are flipped the same way unless --no-flip (use it for clips that were
already recorded mirrored). Whether a file was mirrored cannot be detected from the
landmarks: a mirrored signer looks like a left-handed signer. A wrong guess only swaps the
dominant hand, which the low/high profiles already cover with mirror augmentation.

Every sample is saved with source "import" plus its origin, and skipped when no hands are
found. Re-running an import skips files that were already imported.
"""
import argparse
import csv
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset import DATA_PATH, normalize_label, save_sample  # noqa: E402
from utils.features import (FACE_OFFSET, HOLISTIC_SIZE, LH_OFFSET, POSE_OFFSET, RH_OFFSET,  # noqa: E402
                            load_profiles, mirror_holistic)

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}
TARGET_FPS = 30.0


# ---------------------------------------------------------------------------
# Helpers shared by both importers
# ---------------------------------------------------------------------------

def has_hands(frame):
    return bool(np.any(frame[LH_OFFSET:] != 0))


def hand_window(seq, fps, max_seconds, pad_seconds=0.15):
    """(start, end) of the hand-active part of a clip (with a little padding), at most
    max_seconds long and centred on the hand activity. None if no hands."""
    active = np.flatnonzero([has_hands(f) for f in seq])
    if active.size == 0:
        return None
    pad = int(round(pad_seconds * fps))
    start, end = max(0, active[0] - pad), min(len(seq), active[-1] + pad + 1)
    max_frames = max(1, int(round(max_seconds * fps)))
    if end - start > max_frames:
        centre = int(round(active.mean()))
        start = min(max(start, centre - max_frames // 2), end - max_frames)
        end = start + max_frames
    return int(start), int(end)


def trim_to_hands(seq, fps, max_seconds):
    window = hand_window(seq, fps, max_seconds) if len(seq) else None
    return None if window is None else seq[window[0]:window[1]]


def existing_origins(data_path):
    """Origins of samples already imported, so re-running an import skips them."""
    seen = set()
    if not os.path.isdir(data_path):
        return seen
    for root, _, files in os.walk(data_path):
        for name in files:
            if name.endswith('.json'):
                try:
                    with open(os.path.join(root, name)) as f:
                        origin = json.load(f).get('origin')
                except (OSError, ValueError):
                    continue
                if origin:
                    seen.add(origin)
    return seen


class Report:
    def __init__(self):
        self.saved, self.skipped = {}, {}

    def save(self, label):
        self.saved[label] = self.saved.get(label, 0) + 1

    def skip(self, reason):
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def print(self, dry_run):
        verb = 'Would save' if dry_run else 'Saved'
        total = sum(self.saved.values())
        print(f"\n{verb} {total} samples:")
        for label, n in sorted(self.saved.items()):
            print(f"  {label:30s} {n}")
        if self.skipped:
            print("Skipped:")
            for reason, n in sorted(self.skipped.items()):
                print(f"  {reason:30s} {n}")


def store(label, seq, meta, args, report, seen):
    if meta['origin'] in seen:
        report.skip('already imported')
        return
    if not np.any(seq[:, LH_OFFSET:] != 0):
        report.skip('no hands detected')
        return
    if args.dry_run:
        report.save(label)
        return
    try:
        save_sample(label, seq.astype(np.float32), meta, data_path=args.data)
    except ValueError as e:
        report.skip(f'rejected: {str(e)[:40]}')
        return
    seen.add(meta['origin'])
    report.save(label)


# ---------------------------------------------------------------------------
# Images and videos
# ---------------------------------------------------------------------------

def file_sha1(path):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def extract_video(path, holistic_cls, extract, flip=True):
    """Holistic landmarks for every frame (sampled down to 30 fps), selfie-mirrored unless flip=False."""
    import cv2
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
    step = max(1, int(round(src_fps / TARGET_FPS)))
    frames, times = [], []
    with holistic_cls(static_image_mode=False, model_complexity=1) as holistic:
        index = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if index % step == 0:
                rgb = cv2.cvtColor(cv2.flip(bgr, 1) if flip else bgr, cv2.COLOR_BGR2RGB)
                frames.append(extract(holistic.process(rgb)).astype(np.float32))
                times.append(index / src_fps)
            index += 1
    cap.release()
    fps = src_fps / step
    return (np.stack(frames) if frames else np.zeros((0, HOLISTIC_SIZE), np.float32)), times, fps


def extract_image(path, holistic_cls, extract, flip=True):
    import cv2
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(cv2.flip(bgr, 1) if flip else bgr, cv2.COLOR_BGR2RGB)
    with holistic_cls(static_image_mode=True, model_complexity=1) as holistic:
        return extract(holistic.process(rgb)).astype(np.float32)


def import_media(args):
    from mediapipe.python.solutions import holistic as mp_holistic
    from utils.holistic_utils import extract_holistic_keypoints

    record_frames = int(round(load_profiles()['dataset']['record_seconds'] * TARGET_FPS))
    report, seen = Report(), existing_origins(args.data)
    folders = sorted(d for d in os.listdir(args.input) if os.path.isdir(os.path.join(args.input, d)))
    if not folders:
        raise SystemExit(f"No label folders in {args.input} (expected one sub-folder per sign).")

    for folder in folders:
        try:
            label = normalize_label(folder.replace('_', ' '))
        except ValueError:
            print(f"Skipping folder '{folder}': not a valid label")
            report.skip('invalid folder name')
            continue
        files = sorted(os.listdir(os.path.join(args.input, folder)))
        for name in files:
            path = os.path.join(args.input, folder, name)
            ext = os.path.splitext(name)[1].lower()
            if ext not in IMAGE_EXTS | VIDEO_EXTS:
                continue
            meta = {"source": "import", "dataset": args.dataset_name, "origin": f"file:{file_sha1(path)}",
                    "origin_name": f"{folder}/{name}"[:200]}
            if args.license:
                meta["license"] = args.license

            if ext in VIDEO_EXTS:
                seq, times, fps = extract_video(path, mp_holistic.Holistic, extract_holistic_keypoints,
                                                flip=not args.no_flip)
                window = hand_window(seq, fps, args.max_seconds) if len(seq) else None
                if window is None:
                    report.skip('no hands detected')
                    continue
                start, end = window
                meta.update({"kind": "video", "fps": round(fps, 2),
                             "timestamps_ms": [round((t - times[start]) * 1000, 1) for t in times[start:end]]})
                seq = seq[start:end]
            else:
                frame = extract_image(path, mp_holistic.Holistic, extract_holistic_keypoints,
                                      flip=not args.no_flip)
                if frame is None:
                    report.skip('unreadable image')
                    continue
                # A photo has no motion: repeat it to one recording's length (static signs only).
                seq = np.repeat(frame[None], record_frames, axis=0)
                meta.update({"kind": "image", "static": True})

            meta["flipped_on_import"] = not args.no_flip
            store(label, seq, meta, args, report, seen)
            print(f"  {folder}/{name}: {len(seq)} frames")
    report.print(args.dry_run)


# ---------------------------------------------------------------------------
# Kaggle Isolated Sign Language Recognition (landmark parquet files)
# ---------------------------------------------------------------------------

KAGGLE_TYPES = {  # type -> (offset in our frame, stride, landmark count)
    'pose': (POSE_OFFSET, 4, 33),
    'face': (FACE_OFFSET, 3, 468),
    'left_hand': (LH_OFFSET, 3, 21),
    'right_hand': (RH_OFFSET, 3, 21),
}


def kaggle_to_holistic(table):
    """Kaggle long format (frame, type, landmark_index, x, y, z) -> (T, 1662)."""
    frame = np.asarray(table['frame'])
    kind = np.asarray(table['type']).astype(str)
    index = np.asarray(table['landmark_index']).astype(int)
    xyz = np.stack([np.asarray(table[c], dtype=np.float64) for c in ('x', 'y', 'z')], axis=1)

    frames = np.unique(frame)
    row = np.searchsorted(frames, frame)
    out = np.zeros((len(frames), HOLISTIC_SIZE), np.float32)
    for name, (offset, stride, count) in KAGGLE_TYPES.items():
        sel = (kind == name) & (index < count)
        r, i, v = row[sel], index[sel], xyz[sel]
        # A part MediaPipe did not detect is NaN in the Kaggle files and zero in ours.
        ok = ~np.isnan(v).any(axis=1)
        r, i, v = r[ok], i[ok], v[ok]
        base = offset + i * stride
        for k in range(3):
            out[r, base + k] = v[:, k]
        if stride == 4:
            out[r, base + 3] = 1.0  # visibility is not in the dataset; present points count as visible
    return out


def import_kaggle(args):
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise SystemExit("pyarrow is needed for the Kaggle files: pip install pyarrow")

    csv_path = os.path.join(args.input, 'train.csv')
    if not os.path.exists(csv_path):
        raise SystemExit(f"{csv_path} not found (expected the Kaggle asl-signs folder).")
    wanted = {normalize_label(l) for l in args.labels} if args.labels else None
    report, seen = Report(), existing_origins(args.data)
    per_label = {}

    with open(csv_path, newline='') as f:
        for rec in csv.DictReader(f):
            try:
                label = normalize_label(rec['sign'])
            except ValueError:
                report.skip('invalid label')
                continue
            if wanted is not None and label not in wanted:
                continue
            if args.limit_per_label and per_label.get(label, 0) >= args.limit_per_label:
                continue
            path = os.path.join(args.input, rec['path'])
            if not os.path.exists(path):
                report.skip('missing parquet file')
                continue
            seq = kaggle_to_holistic(pq.read_table(path, columns=['frame', 'type', 'landmark_index',
                                                                  'x', 'y', 'z']).to_pydict())
            seq = trim_to_hands(seq, TARGET_FPS, args.max_seconds)
            if seq is None:
                report.skip('no hands detected')
                continue
            if args.mirror:
                seq = mirror_holistic(seq)
            meta = {"source": "import", "dataset": args.dataset_name or "kaggle-asl-signs",
                    "origin": f"kaggle:{rec['sequence_id']}", "participant": rec.get('participant_id'),
                    "kind": "landmarks", "fps": TARGET_FPS, "mirrored_on_import": bool(args.mirror),
                    "license": args.license or "Kaggle competition data: check its rules before reuse"}
            before = sum(report.saved.values())
            store(label, seq, meta, args, report, seen)
            if sum(report.saved.values()) > before:
                per_label[label] = per_label.get(label, 0) + 1
    report.print(args.dry_run)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    for name, helptext in (('media', 'folder of label sub-folders with images / videos'),
                           ('kaggle', 'Kaggle asl-signs folder (train.csv + train_landmark_files/)')):
        p = sub.add_parser(name, help=helptext)
        p.add_argument('input')
        p.add_argument('--data', default=DATA_PATH, help="dataset folder (default: Data/Holistic_Landmarks)")
        p.add_argument('--dataset-name', default='', help="stored with every sample, e.g. 'wlasl'")
        p.add_argument('--license', default='', help="licence / terms of the source, stored with every sample")
        p.add_argument('--max-seconds', type=float, default=2.0, help="longest clip kept per sample")
        p.add_argument('--dry-run', action='store_true', help="report what would be imported")
        if name == 'media':
            p.add_argument('--no-flip', action='store_true',
                           help="do not mirror the frames (the clips are already mirrored / selfie view)")
        if name == 'kaggle':
            p.add_argument('--mirror', action='store_true',
                           help="mirror the landmarks (swap left/right), e.g. to balance the dominant hand")
            p.add_argument('--labels', nargs='*', help="only these signs (default: all 250)")
            p.add_argument('--limit-per-label', type=int, default=0,
                           help="cap per sign, to keep the dataset balanced (0 = no cap)")
    args = parser.parse_args()
    import_media(args) if args.cmd == 'media' else import_kaggle(args)


if __name__ == '__main__':
    main()
