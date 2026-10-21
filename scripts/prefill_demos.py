"""Finite strided collection prefill for independent single-GPU workers."""
import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--start', type=int, required=True)
    p.add_argument('--stop', type=int, required=True)
    p.add_argument('--stride', type=int, default=1)
    args = p.parse_args()
    from train import collect
    from simulation import configure_iteration_warnings
    from collection_cache import CollectionCache, CacheFull
    from demo_identity import description
    configure_iteration_warnings()
    settings = SimpleNamespace(env='block-double-task1-v0', mode='streaming',
        collection_cache=False, features='handcrafted', data_seed=args.seed, parallel=2500)
    cache = CollectionCache(ROOT / 'data/shared-demos')
    for index in range(args.start, args.stop, args.stride):
        val_count = 128 if index == 0 else 0
        try:
            raw, val, metrics = cache.get(description(settings.env, 1000, val_count, args.seed, index, 2500),
                lambda: collect(settings, index, 1000, val_count), consumer=None, allow_uncached=False)
        except CacheFull:
            print(json.dumps(dict(event='budget_stop', index=index)), flush=True)
            return
        print(json.dumps(dict(event='prefill', index=index, seed=args.seed, seconds=metrics['collection_time'],
                              cache_hit=metrics['cache_hit'], bytes=metrics['cache_entry_bytes'])), flush=True)
        del raw, val
        gc.collect()
    print(json.dumps(dict(event='completed', seed=args.seed, start=args.start, stop=args.stop, stride=args.stride)), flush=True)


if __name__ == '__main__':
    main()
