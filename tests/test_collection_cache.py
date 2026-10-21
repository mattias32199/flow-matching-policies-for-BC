import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from collection_cache import CollectionCache, CacheFull, CONSUMERS


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = CollectionCache(self.tmp.name, limit=100000)
        self.desc = dict(episodes=1, val_episodes=0, seed=0)
        self.calls = 0

    def tearDown(self):
        self.tmp.cleanup()

    def generate(self):
        self.calls += 1
        time.sleep(.02)
        return {'observations': np.arange(8, dtype=np.float32)}, None, {'collection_time': .02}

    def test_private_arrays_and_four_consumer_cleanup(self):
        first = self.cache.get(self.desc, self.generate, CONSUMERS[0], 30000)
        first[0]['observations'][0] = -100
        for feature in CONSUMERS[1:]:
            value, _, metrics = self.cache.get(self.desc, self.generate, feature, 30000)
            self.assertTrue(metrics['cache_hit'])
            np.testing.assert_array_equal(value['observations'], np.arange(8))
        self.assertEqual(self.calls, 1)
        self.assertEqual(self.cache.status()['used_bytes'], 0)

    def test_concurrent_miss_collects_once(self):
        with ThreadPoolExecutor(2) as pool:
            jobs = [pool.submit(self.cache.get, self.desc, self.generate, feature, 30000) for feature in CONSUMERS[:2]]
            records = [job.result()[2] for job in jobs]
        self.assertEqual(self.calls, 1)
        self.assertEqual(sum(r['cache_hit'] for r in records), 1)

    def test_budget_prevents_write_and_prefill_generation(self):
        with self.assertRaises(CacheFull):
            self.cache.get(self.desc, self.generate, reservation_bytes=100001, allow_uncached=False)
        self.assertEqual(self.calls, 0)
        _, _, metrics = self.cache.get(self.desc, self.generate, reservation_bytes=100001)
        self.assertFalse(metrics['cache_stored'])
        self.assertEqual(self.cache.status()['used_bytes'], 0)

    def test_seed_identity_isolated(self):
        for seed in [0, 1]:
            self.cache.get(dict(self.desc, seed=seed), self.generate, reservation_bytes=30000)
        self.assertEqual(self.calls, 2)

    def test_failure_releases_reservation(self):
        def fail():
            raise ValueError('synthetic failure')
        with self.assertRaises(ValueError):
            self.cache.get(self.desc, fail, reservation_bytes=30000)
        self.assertEqual(self.cache.status()['reservations'], {})

    def test_oversize_payload_is_not_written(self):
        def large():
            return {'observations': np.zeros(40000, np.float32)}, None, {}
        _, _, record = self.cache.get(self.desc, large, reservation_bytes=30000)
        self.assertFalse(record['cache_stored'])
        self.assertEqual(self.cache.status()['used_bytes'], 0)
        self.assertFalse(list(Path(self.tmp.name).rglob('*.npz')))

    def test_finished_seed_cleanup_preserves_other_seed(self):
        for seed in [0, 1]:
            self.cache.get(dict(self.desc, data_seed=seed), self.generate, reservation_bytes=30000)
        before = self.cache.status()['used_bytes']
        self.assertEqual(self.cache.discard_seed(0), 1)
        self.assertGreater(self.cache.status()['used_bytes'], 0)
        self.assertLess(self.cache.status()['used_bytes'], before)
        self.assertEqual(self.cache.discard_seed(0), 0)

    def test_seed_cleanup_ignores_inflight_staging(self):
        import json
        pending = Path(self.tmp.name) / ('a' * 64 + '.pending')
        pending.mkdir()
        (pending / 'manifest.json').write_text(json.dumps(dict(description={'data_seed': 0}, bytes=123)))
        self.assertEqual(self.cache.discard_seed(0), 0)
        self.assertTrue(pending.exists())
        self.assertEqual(self.cache.status()['used_bytes'], 0)
