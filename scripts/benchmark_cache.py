"""End-to-end full-model fresh-demo comparison and bounded future prefill."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'ocbench/impls')]
import jax
import numpy as np
from train import collect, dataset, Features
from simulation import configure_iteration_warnings
from agents.fbc import FBCAgent, get_config
from envs.env_utils import ActionNormalizer
from collection_cache import CollectionCache, CONSUMERS, CacheFull
from demo_identity import description


def digest(raw):
    h = hashlib.sha256()
    for k, v in sorted(raw.items()):
        h.update(k.encode())
        h.update(np.asarray(v).tobytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=ROOT / '.local/cache-benchmark' / os.environ.get('SLURM_JOB_ID', 'manual'))
    p.add_argument('--updates', type=int, default=1024)
    p.add_argument('--prefill', type=int, default=8)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configure_iteration_warnings()
    device = jax.devices()[0]
    if device.platform != 'gpu' or 'A6000' not in device.device_kind:
        raise RuntimeError('Requires an allocated A6000.')
    output = args.output / 'cache-benchmark.jsonl'

    def emit(row):
        print(json.dumps(row), flush=True)
        with output.open('a') as f:
            f.write(json.dumps(row) + '\n')

    # Warm executable outside both comparison arms.
    config = get_config()
    warm = FBCAgent.create(0, np.zeros((1, 37), np.float32), np.zeros((1, 175), np.float32), config)
    warm, info = warm.update(dict(observations=np.zeros((1024, 37), np.float32), actions=np.zeros((1024, 175), np.float32)))
    jax.block_until_ready((warm, info))
    del warm, info
    gc.collect()
    shared_hashes = []
    for cached in (False, True):
        for feature in CONSUMERS:
            settings = SimpleNamespace(env='block-double-task1-v0', mode='streaming',
                collection_cache=cached, features=feature, data_seed=991, parallel=2500)
            started = time.monotonic()
            raw, val, metrics = collect(settings, 0, 1000, 128)
            collection_seconds = time.monotonic() - started
            sha = digest(raw)
            if cached:
                shared_hashes.append(sha)
            prep = time.monotonic()
            normalizer = ActionNormalizer.fit(raw['actions'], 1., 5.)
            training, validation = dataset(raw, normalizer), dataset(val, normalizer)
            transform = Features(training['observations'], feature)
            del raw, val, validation
            ids = np.random.default_rng(0).permutation(training.size)
            if args.updates * 1024 > len(ids):
                raise RuntimeError('Requested benchmark exceeds unique transition starts.')
            example = training.sample(1, keys=('observations', 'actions'))
            agent = FBCAgent.create(0, transform(example['observations']), example['actions'], get_config())
            preprocessing_seconds = time.monotonic() - prep
            started = time.monotonic()
            for step in range(args.updates):
                batch = training.sample(1024, idxs=ids[step * 1024:(step + 1) * 1024], keys=('observations', 'actions'))
                batch['observations'] = transform(batch['observations'])
                agent, info = agent.update(batch)
            jax.block_until_ready((agent, info))
            training_seconds = time.monotonic() - started
            loss = float(info['actor/actor_loss'])
            if not np.isfinite(loss):
                raise RuntimeError('Nonfinite benchmark loss.')
            emit(dict(arm='shared' if cached else 'independent', feature=feature,
                      updates=args.updates, collection_seconds=collection_seconds,
                      preprocessing_seconds=preprocessing_seconds, training_seconds=training_seconds,
                      total_phase_seconds=collection_seconds + preprocessing_seconds + training_seconds,
                      training_updates_per_second=args.updates / training_seconds,
                      flow_loss=loss, dataset_sha256=sha, retained_transitions=training.size,
                      cache_hit=metrics.get('cache_hit', False), cache_entry_bytes=metrics.get('cache_entry_bytes', 0)))
            del training, agent, info, batch, ids, example
            gc.collect()
    if len(set(shared_hashes)) != 1:
        raise RuntimeError('Shared consumers received different demonstration arrays.')
    (args.output / 'BENCHMARK_DONE').write_text('Four independent and four shared full-model consumers completed.\n')
    # A finite ahead-of-time producer, not another watcher or model wake-up.
    settings = SimpleNamespace(env='block-double-task1-v0', mode='streaming',
        collection_cache=False, features='handcrafted', data_seed=1, parallel=2500)
    cache = CollectionCache(ROOT / 'data/shared-demos')
    for index in range(args.prefill):
        val_count = 128 if index == 0 else 0
        try:
            raw, val, metrics = cache.get(description(settings.env, 1000, val_count, 1, index, 2500),
                lambda: collect(settings, index, 1000, val_count), consumer=None, allow_uncached=False)
        except CacheFull:
            emit(dict(event='prefill_budget_stop', index=index))
            break
        emit(dict(event='prefill', index=index, data_seed=1, cache_hit=metrics['cache_hit'],
                  cache_entry_bytes=metrics['cache_entry_bytes'], seconds=metrics['collection_time']))
        del raw, val
        gc.collect()
    emit(dict(event='cache_status', **cache.status()))
    (args.output / 'DONE').write_text('Benchmark and finite future prefill completed.\n')


if __name__ == '__main__':
    main()
