"""Finite-window Slurm producer; scheduling and polling never invoke AI."""
import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace
import time
ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--window', type=int, default=24)
    p.add_argument('--max-collections', type=int, default=4096)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--start-index', type=int, default=0)
    args = p.parse_args()
    from train import collect
    from simulation import configure_iteration_warnings
    from collection_cache import CollectionCache, CONSUMERS, CacheFull, write_json
    from demo_identity import description
    configure_iteration_warnings()
    cache = CollectionCache(ROOT / 'data/shared-demos')
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        raise ValueError('Invalid producer shard.')
    first = args.start_index + (args.shard - args.start_index) % args.shards
    state_path = ROOT / '.local' / f'demo-producer-seed{args.seed}-n{args.shards}-s{args.shard}.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'next_index': first}
    index = max(state['next_index'], first)
    settings = SimpleNamespace(env='block-double-task1-v0', mode='streaming',
        collection_cache=False, features='handcrafted', data_seed=args.seed, parallel=2500)
    last_active = time.monotonic()
    full_since = None
    while index < min(args.max_collections, 4096):
        runs = [ROOT / 'runs' / f'm4-{f}-s{args.seed}' for f in CONSUMERS]
        if all((r / 'DONE').exists() for r in runs):
            removed = cache.discard_seed(args.seed)
            print(json.dumps(dict(event='all_consumers_complete', seed=args.seed, unused_entries_removed=removed)), flush=True)
            return
        try:
            active = json.loads((ROOT / '.local/queue-status.json').read_text()).get('active', {}).values()
            if any(r.name in active for r in runs):
                last_active = time.monotonic()
            elif time.monotonic() - last_active > 300:
                print(json.dumps(dict(event='no_active_consumers', seed=args.seed)), flush=True)
                return
        except (OSError, json.JSONDecodeError):
            pass
        # Bound lookahead by fastest consumer. Budget separately bounds lagging-reader storage.
        seen = []
        for r in runs:
            if (r / 'collections.jsonl').exists():
                for line in reversed((r / 'collections.jsonl').read_text().splitlines()):
                    try:
                        seen.append(json.loads(line)['index'])
                        break
                    except (json.JSONDecodeError, KeyError):
                        continue
        if index > max(seen, default=0) + args.window:
            time.sleep(15)
            continue
        val_count = 128 if index == 0 else 0
        try:
            raw, val, metrics = cache.get(description(settings.env, 1000, val_count, args.seed, index, 2500),
                lambda: collect(settings, index, 1000, val_count), consumer=None, allow_uncached=False)
        except CacheFull:
            full_since = full_since or time.monotonic()
            if time.monotonic() - full_since > 300:
                print(json.dumps(dict(event='budget_pause_release_gpu', seed=args.seed)), flush=True)
                return
            time.sleep(15)
            continue
        full_since = None
        print(json.dumps(dict(event='prefetched', seed=args.seed, index=index,
                              seconds=metrics['collection_time'], hit=metrics['cache_hit'],
                              entry_bytes=metrics['cache_entry_bytes'])), flush=True)
        del raw, val
        index += args.shards
        write_json(state_path, dict(next_index=index))
        gc.collect()
    print(json.dumps(dict(event='collection_limit_reached', seed=args.seed, next_index=index)), flush=True)


if __name__ == '__main__':
    main()
