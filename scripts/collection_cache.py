"""Bounded, cross-process raw-demo cache shared by feature comparisons.

Readers receive private arrays. Every run still consumes each transition once.
Reservations include in-progress writes; completed entries are removed after all
four feature consumers have loaded them. A full cache falls back to collection.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np

CONSUMERS = ('handcrafted', 'standardized', 'xyz0.1', 'xyz10')
DEFAULT_LIMIT = 80_000_000_000  # Decimal bytes, below the user's 100 GB ceiling.


class CacheFull(RuntimeError):
    pass


def cache_key(description):
    return hashlib.sha256(json.dumps(description, sort_keys=True).encode()).hexdigest()


@contextmanager
def locked(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True))
    temporary.replace(path)


class CollectionCache:
    def __init__(self, root, limit=DEFAULT_LIMIT, consumers=CONSUMERS):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.limit = min(int(limit), DEFAULT_LIMIT)
        self.consumers = tuple(consumers)
        if not self.consumers or len(set(self.consumers)) != len(self.consumers):
            raise ValueError("Consumers must be nonempty and unique.")
        if self.limit <= 0:
            raise ValueError('Cache limit must be positive.')

    def _state(self):
        p = self.root / 'budget.json'
        return json.loads(p.read_text()) if p.exists() else dict(used_bytes=0, reservations={})

    def _save(self, state):
        write_json(self.root / 'budget.json', state)

    def status(self):
        with locked(self.root / 'budget.lock'):
            state = self._state()
            return dict(limit_bytes=self.limit, **state)

    def contains(self, description):
        return (self.root / cache_key(description) / 'manifest.json').is_file()

    def discard_seed(self, seed):
        """Release unused lookahead only after caller verifies all seed runs DONE."""
        removed = 0
        for target in list(self.root.iterdir()):
            if len(target.name) != 64 or any(c not in '0123456789abcdef' for c in target.name):
                continue
            if not (target / 'manifest.json').is_file():
                continue
            with locked(self.root / 'locks' / (target.name + '.lock')):
                manifest_path = target / 'manifest.json'
                if not manifest_path.exists():
                    continue
                manifest = json.loads(manifest_path.read_text())
                if manifest['description'].get('data_seed') != seed:
                    continue
                with locked(self.root / 'budget.lock'):
                    state = self._state()
                    shutil.rmtree(target)
                    state['used_bytes'] -= manifest['bytes']
                    self._save(state)
                removed += 1
        return removed

    def get(self, description, generate, consumer=None, reservation_bytes=None, allow_uncached=True):
        if consumer is not None and consumer not in self.consumers:
            raise ValueError('Unknown feature consumer.')
        started = time.monotonic()
        key = cache_key(description)
        target = self.root / key
        temporary = self.root / (key + '.pending')
        # Conservative bound for the state-based block datasets used here.
        reserve = int(reservation_bytes or (description['episodes'] + description['val_episodes']) * 1_000_000 + 1_000_000)
        with locked(self.root / 'locks' / (key + '.lock')):
            # Recover a crashed writer only after obtaining its per-key lock.
            with locked(self.root / 'budget.lock'):
                state = self._state()
                if key in state['reservations']:
                    if target.exists():
                        state['used_bytes'] += json.loads((target / 'manifest.json').read_text())['bytes']
                    elif temporary.exists():
                        shutil.rmtree(temporary)
                    state['reservations'].pop(key)
                    self._save(state)
            hit = target.exists()
            stored = hit
            if hit:
                manifest = json.loads((target / 'manifest.json').read_text())
                with np.load(target / 'train.npz', allow_pickle=False) as f:
                    train = dict(f)
                val = None
                if manifest['has_val']:
                    with np.load(target / 'val.npz', allow_pickle=False) as f:
                        val = dict(f)
                metrics = manifest['metrics']
            else:
                with locked(self.root / 'budget.lock'):
                    state = self._state()
                    available = state['used_bytes'] + sum(state['reservations'].values()) + reserve <= self.limit
                    if available:
                        state['reservations'][key] = reserve
                        self._save(state)
                if not available and not allow_uncached:
                    raise CacheFull('Bounded cache has no reservation space.')
                try:
                    train, val, metrics = generate()
                    if available:
                        arrays = list(train.values()) + (list(val.values()) if val is not None else [])
                        payload_bound = sum(np.asarray(a).nbytes + 4096 for a in arrays) + 16384
                        if payload_bound > reserve:
                            with locked(self.root / 'budget.lock'):
                                state = self._state()
                                extra = payload_bound - reserve
                                available = state['used_bytes'] + sum(state['reservations'].values()) + extra <= self.limit
                                if available:
                                    state['reservations'][key] = payload_bound
                                    reserve = payload_bound
                                else:
                                    state['reservations'].pop(key)
                                self._save(state)
                    if available:
                        temporary.mkdir()
                        np.savez(temporary / 'train.npz', **train)
                        if val is not None:
                            np.savez(temporary / 'val.npz', **val)
                        # Reserve space for the manifest and consumer record.
                        size = sum(p.stat().st_size for p in temporary.iterdir()) + 16384
                        if size <= reserve:
                            manifest = dict(description=description, metrics=metrics, has_val=val is not None, bytes=size)
                            write_json(temporary / 'manifest.json', manifest)
                            temporary.replace(target)
                            with locked(self.root / 'budget.lock'):
                                state = self._state()
                                state['reservations'].pop(key)
                                state['used_bytes'] += size
                                self._save(state)
                            stored = True
                        else:
                            shutil.rmtree(temporary)
                finally:
                    with locked(self.root / 'budget.lock'):
                        state = self._state()
                        if not target.exists():
                            if temporary.exists():
                                shutil.rmtree(temporary)
                            state['reservations'].pop(key, None)
                            self._save(state)
            entry_bytes = json.loads((target / 'manifest.json').read_text())['bytes'] if stored else 0
            if stored and consumer is not None:
                path = target / 'consumers.json'
                consumed = set(json.loads(path.read_text()) if path.exists() else [])
                consumed.add(consumer)
                write_json(path, sorted(consumed))
                if consumed == set(self.consumers):
                    with locked(self.root / 'budget.lock'):
                        state = self._state()
                        shutil.rmtree(target)
                        state['used_bytes'] -= entry_bytes
                        self._save(state)
            return train, val, dict(metrics, collection_time=time.monotonic() - started,
                                    cache_hit=hit, cache_stored=stored, cache_key=key,
                                    cache_entry_bytes=entry_bytes,
                                    source_collection_time=metrics.get('collection_time', 0))
