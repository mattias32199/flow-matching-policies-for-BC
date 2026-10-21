from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from collection_cache import CollectionCache
from loader_prefetch import CachedCollectionPrefetch


class LoaderPrefetchTests(unittest.TestCase):
    def test_reads_cached_data_once_and_never_generates_on_a_miss(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = CollectionCache(directory, consumers=('a', 'b'))
            identity = dict(episodes=1, val_episodes=0)
            reader = CachedCollectionPrefetch()
            try:
                self.assertFalse(reader.schedule(1, cache, identity, 'a'))
                self.assertIsNone(reader.take(1))
                cache.get(identity, lambda: ({'observations': np.arange(4)}, None, {}),
                          consumer='a', reservation_bytes=30000)
                self.assertTrue(reader.schedule(1, cache, identity, 'b'))
                self.assertFalse(reader.schedule(2, cache, identity, 'b'))
                data, _, metrics = reader.take(1)
                np.testing.assert_array_equal(data['observations'], np.arange(4))
                self.assertTrue(metrics['cache_hit'])
                self.assertEqual(cache.status()['used_bytes'], 0)
                self.assertIsNone(reader.take(1))
            finally:
                reader.close()

    def test_disappearing_entry_falls_back_without_generation(self):
        class VanishingCache:
            def contains(self, identity):
                return True

            def get(self, identity, generate, consumer):
                return generate()

        reader = CachedCollectionPrefetch()
        try:
            self.assertTrue(reader.schedule(1, VanishingCache(), {}, 'a'))
            self.assertIsNone(reader.take(1))
        finally:
            reader.close()
