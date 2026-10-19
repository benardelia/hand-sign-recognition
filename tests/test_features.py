"""Run: python -m unittest discover tests"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.features import (HOLISTIC_SIZE, LH_OFFSET, RH_OFFSET, build_feature_spec, get_profile,  # noqa: E402
                            load_profiles, mirror_holistic, resample_sequence, select_features)


def random_sequence(t=30, seed=0, drop_left_hand=False):
    rng = np.random.default_rng(seed)
    seq = rng.uniform(0.1, 0.9, (t, HOLISTIC_SIZE)).astype(np.float32)
    if drop_left_hand:
        seq[:, LH_OFFSET:LH_OFFSET + 63] = 0
    return seq


class FeatureSpecTest(unittest.TestCase):
    def test_profile_sizes(self):
        sizes = {n: build_feature_spec(get_profile(n))['num_features'] for n in load_profiles()['profiles']}
        self.assertEqual(sizes['low'], (7 + 42) * 3)
        self.assertEqual(sizes['high'], (25 + 60 + 42) * 3)
        self.assertEqual(sizes['experimental'], (33 + 468 + 42) * 3)

    def test_columns_in_range_and_unique(self):
        for name in load_profiles()['profiles']:
            cols = build_feature_spec(get_profile(name))['columns']
            self.assertEqual(len(cols), len(set(cols)))
            self.assertTrue(all(0 <= c < HOLISTIC_SIZE for c in cols))

    def test_missing_landmarks_stay_zero(self):
        spec = build_feature_spec(get_profile('low'))
        out = select_features(random_sequence(drop_left_hand=True), spec)
        lh_idx = [i for i, p in enumerate(spec['points']) if p['group'] == 'left_hand']
        feats = out.reshape(out.shape[0], -1, 3)
        self.assertTrue(np.all(feats[:, lh_idx] == 0))
        self.assertTrue(np.any(feats != 0))

    def test_normalization_is_translation_invariant(self):
        spec = build_feature_spec(get_profile('high'))
        seq = random_sequence()
        shifted = seq.copy()
        cols = np.arange(HOLISTIC_SIZE)
        # Shift x of every landmark (pose x cols step 4, the rest step 3).
        x_cols = np.concatenate([np.arange(0, 132, 4), np.arange(132, HOLISTIC_SIZE, 3)])
        self.assertTrue(set(x_cols) <= set(cols))
        shifted[:, x_cols] += 0.05
        np.testing.assert_allclose(select_features(seq, spec), select_features(shifted, spec), atol=1e-4)


class SequenceOpsTest(unittest.TestCase):
    def test_resample_lengths(self):
        seq = random_sequence(30)
        self.assertEqual(resample_sequence(seq, 15).shape, (15, HOLISTIC_SIZE))
        self.assertEqual(resample_sequence(seq, 45).shape, (45, HOLISTIC_SIZE))
        np.testing.assert_array_equal(resample_sequence(seq, 30), seq)

    def test_resample_does_not_invent_landmarks(self):
        seq = random_sequence(4)
        seq[1:3, RH_OFFSET:RH_OFFSET + 63] = 0
        out = resample_sequence(seq, 7)
        rh = out[:, RH_OFFSET:RH_OFFSET + 63]
        # Each frame's hand is either fully missing or equal-ish to a real frame, never a blend with zero.
        for frame in rh:
            self.assertTrue(np.all(frame == 0) or np.all(frame > 0.05))

    def test_mirror_twice_is_identity(self):
        seq = random_sequence()
        np.testing.assert_allclose(mirror_holistic(mirror_holistic(seq)), seq, atol=1e-6)

    def test_mirror_swaps_hands(self):
        seq = random_sequence(drop_left_hand=True)
        m = mirror_holistic(seq)
        self.assertTrue(np.all(m[:, RH_OFFSET:RH_OFFSET + 63] == 0))
        self.assertTrue(np.all(m[:, LH_OFFSET:LH_OFFSET + 63] != 0))


if __name__ == '__main__':
    unittest.main()
