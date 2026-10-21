"""Run: python -m unittest discover tests"""
import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.import_media import (KAGGLE_TYPES, Report, existing_origins, hand_window,  # noqa: E402
                                import_kaggle, kaggle_to_holistic, store)
from utils.features import HOLISTIC_SIZE, LH_OFFSET, mirror_holistic  # noqa: E402

try:
    import pyarrow  # noqa: F401
    HAVE_PYARROW = True
except ImportError:
    HAVE_PYARROW = False


def sample(t=20, seed=0, shoulders=(0.6, 0.4)):
    """Random frames with a person facing the camera (left shoulder x > right shoulder x)."""
    seq = np.random.default_rng(seed).uniform(0.2, 0.8, (t, HOLISTIC_SIZE)).astype(np.float32)
    seq[:, 44], seq[:, 48] = shoulders
    return seq


def to_kaggle_rows(seq):
    """Inverse of kaggle_to_holistic: our frames -> Kaggle long-format rows."""
    rows = {'frame': [], 'row_id': [], 'type': [], 'landmark_index': [], 'x': [], 'y': [], 'z': []}
    for f, frame in enumerate(seq):
        for name, (offset, stride, count) in KAGGLE_TYPES.items():
            part = frame[offset:offset + count * stride].reshape(count, stride)
            missing = not np.any(part[:, :3])
            for i in range(count):
                x, y, z = (np.nan, np.nan, np.nan) if missing else part[i, :3]
                rows['frame'].append(f + 20)  # Kaggle frame numbers do not start at 0
                rows['row_id'].append(f"{f}-{name}-{i}")
                rows['type'].append(name)
                rows['landmark_index'].append(i)
                for k, v in zip('xyz', (x, y, z)):
                    rows[k].append(float(v))
    return rows


class HelpersTest(unittest.TestCase):
    def test_hand_window_trims_and_caps(self):
        seq = sample(t=120)
        seq[:30, LH_OFFSET:] = 0
        seq[100:, LH_OFFSET:] = 0
        start, end = hand_window(seq, fps=30, max_seconds=10)
        self.assertEqual((start, end), (30 - 4, 100 + 4))  # 0.15 s padding = 4 frames at 30 fps
        start, end = hand_window(seq, fps=30, max_seconds=1)
        self.assertEqual(end - start, 30)
        self.assertTrue(25 <= start and end <= 104)

    def test_hand_window_none_without_hands(self):
        seq = sample()
        seq[:, LH_OFFSET:] = 0
        self.assertIsNone(hand_window(seq, 30, 2))

    def test_mirroring_keeps_shoulder_layout(self):
        # Why the importer cannot auto-detect mirrored files: MediaPipe names body sides by
        # appearance, so a mirrored signer keeps the same shoulder layout (only hands swap).
        ours = sample()
        mirrored = mirror_holistic(ours)
        np.testing.assert_allclose(mirrored[:, [44, 48]], ours[:, [44, 48]], atol=1e-6)


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.args = argparse.Namespace(data=self.tmp, dry_run=False)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_store_skips_duplicates_and_handless(self):
        report, seen = Report(), existing_origins(self.tmp)
        meta = {"source": "import", "origin": "file:abc"}
        store('HELLO', sample(), dict(meta), self.args, report, seen)
        store('HELLO', sample(), dict(meta), self.args, report, seen)
        no_hands = sample()
        no_hands[:, LH_OFFSET:] = 0
        store('HELLO', no_hands, {"source": "import", "origin": "file:def"}, self.args, report, seen)
        self.assertEqual(report.saved, {'HELLO': 1})
        self.assertEqual(report.skipped, {'already imported': 1, 'no hands detected': 1})
        # A new run sees the earlier import through the saved metadata.
        self.assertIn('file:abc', existing_origins(self.tmp))


@unittest.skipUnless(HAVE_PYARROW, "pyarrow not installed")
class KaggleTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_conversion_roundtrip(self):
        seq = sample(t=5)
        seq[2, LH_OFFSET:LH_OFFSET + 63] = 0  # left hand missing in one frame -> NaN in Kaggle
        seq[:, 3::4][:, :33] = 1.0  # visibility column is not in Kaggle data; present = 1.0
        out = kaggle_to_holistic(to_kaggle_rows(seq))
        np.testing.assert_allclose(out, seq, atol=1e-6)

    def test_import_unmirrored_dataset(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        root = os.path.join(self.tmp, 'asl-signs')
        os.makedirs(os.path.join(root, 'train_landmark_files', '1'))
        ours = sample(t=40)
        pq.write_table(pa.table(to_kaggle_rows(ours)),
                       os.path.join(root, 'train_landmark_files', '1', '100.parquet'))
        with open(os.path.join(root, 'train.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['path', 'participant_id', 'sequence_id', 'sign'])
            w.writerow(['train_landmark_files/1/100.parquet', '1', '100', 'hello'])
            w.writerow(['train_landmark_files/1/missing.parquet', '1', '101', 'hello'])
        data = os.path.join(self.tmp, 'Data')
        args = argparse.Namespace(input=root, data=data, dataset_name='', license='', mirror=False,
                                  max_seconds=2.0, dry_run=False, labels=None, limit_per_label=0)
        import_kaggle(args)
        files = [f for f in os.listdir(os.path.join(data, 'HELLO')) if f.endswith('.npy')]
        self.assertEqual(len(files), 1)
        saved = np.load(os.path.join(data, 'HELLO', files[0]))
        np.testing.assert_allclose(saved[:, 132:], ours[:, 132:], atol=1e-6)
        meta = json.load(open(os.path.join(data, 'HELLO', files[0][:-4] + '.json')))
        self.assertEqual((meta['source'], meta['origin'], meta['mirrored_on_import']), ('import', 'kaggle:100', False))
        import_kaggle(args)  # second run: nothing new
        self.assertEqual(len([f for f in os.listdir(os.path.join(data, 'HELLO')) if f.endswith('.npy')]), 1)


if __name__ == '__main__':
    unittest.main()
