"""Read one already-cached collection ahead, without background simulation."""
from concurrent.futures import ThreadPoolExecutor


class CacheMiss(RuntimeError):
    pass


class CachedCollectionPrefetch:
    def __init__(self):
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='demo-reader')
        self.future = None
        self.index = None

    def schedule(self, index, cache, identity, consumer):
        if self.future is not None or not cache.contains(identity):
            return False

        def no_generation():
            raise CacheMiss('Entry disappeared before background read.')

        self.index = index
        self.future = self.pool.submit(cache.get, identity, no_generation, consumer)
        return True

    def take(self, index):
        if self.future is None:
            return None
        if index != self.index:
            raise ValueError('Prefetched collection index mismatch.')
        future, self.future = self.future, None
        self.index = None
        try:
            return future.result()
        except CacheMiss:
            return None

    def close(self):
        self.pool.shutdown(wait=True)
