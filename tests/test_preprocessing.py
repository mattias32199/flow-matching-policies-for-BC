import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from train import EvalAgent, Features, evaluation_record


class PreprocessingTest(unittest.TestCase):
    def test_scaling_matches_block_layout_and_preserves_other_features(self):
        observations = np.random.default_rng(4).normal(size=(100, 37)).astype(np.float32)
        base = Features(observations, 'standardized')(observations)
        enhanced = Features(observations, 'xyz10')(observations)
        xyz = [12, 13, 14, 19, 20, 21, 28, 29, 30]
        other = [index for index in range(37) if index not in xyz]
        np.testing.assert_allclose(enhanced[:, xyz], base[:, xyz] * 10, rtol=1e-6)
        np.testing.assert_array_equal(enhanced[:, other], base[:, other])
        np.testing.assert_allclose(base.mean(axis=0), 0, atol=1e-6)
        np.testing.assert_allclose(base.std(axis=0), 1, atol=1e-6)

    def test_handcrafted_is_identity_and_constant_features_remain_finite(self):
        observations = np.ones((10, 37), dtype=np.float32)
        np.testing.assert_array_equal(Features(observations, 'handcrafted')(observations), observations)
        self.assertTrue(np.isfinite(Features(observations, 'standardized')(observations)).all())

    def test_zero_success_evaluation_serializes_undefined_duration(self):
        import json
        record = evaluation_record(1, {'success': 0., 'mean_success_steps': np.nan})
        self.assertIsNone(record['mean_success_steps'])
        self.assertEqual(record['success'], 0.)
        json.dumps(record, allow_nan=False)

    def test_eval_uses_frozen_training_transform(self):
        train = np.random.default_rng(5).normal(size=(100, 37)).astype(np.float32)
        transform = Features(train, 'standardized')
        shifted = train[:5] + 3
        class IdentityPolicy:
            def sample_actions(self, observations, seed=None):
                return observations
        policy = EvalAgent(IdentityPolicy(), transform)
        np.testing.assert_array_equal(policy.sample_actions(shifted), transform(shifted))
        self.assertGreater(policy.sample_actions(shifted).mean(), 2)


if __name__ == '__main__':
    unittest.main()
