"""Collect a sparse visual demonstration dataset for any OCBench MJWarp env, in the OCBench HF shard layout.

OCBench only releases visual datasets for block tasks. This produces the same layout for other envs (hanoi, bowling)
with OCBench's own in-loop rendering (3 cameras, 224 px, a frame every --interval steps; mocap state such as hanoi
peg poses is rendered correctly, which re-rendering from qpos would not do):

  data/ocbench-hf/visual/visual-<env>/NNN.npz      100 episodes each (all episodes, success filtered later)
  data/ocbench-hf/visual/visual-<env>/NNN-val.npz  validation episodes

Collection i uses generate_mjwarp_datasets(visual env, root_seed=--seed, collection_idx=i), i.e. the same reset and
oracle seeds as the state collection store (scripts/collections_store.py), so visual collection i shows the same
initial states as state collection i. Afterwards scripts/vis_prep.py / embed_probe.py / train_vis.py work unchanged.

  python scripts/vis_collect.py --env hanoi-single-task1-v0 --idx 0-9      (JAX venv, GPU)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KEYS = ('actions', 'rewards', 'masks', 'terminals', 'qpos', 'qvel', 'observations', 'observation_interval')


def split_episodes(d, per):
    """Yield sub-datasets of `per` whole episodes from a sparse visual dataset."""
    interval = int(d['observation_interval'])
    offsets = np.concatenate([[0], np.flatnonzero(d['terminals']) + 1])
    frames_per = [len(range(0, offsets[e + 1] - offsets[e], interval)) + 1 for e in range(len(offsets) - 1)]
    foff = np.concatenate([[0], np.cumsum(frames_per)])
    assert foff[-1] == len(d['observations']), (foff[-1], len(d['observations']))
    for a in range(0, len(offsets) - 1, per):
        b = min(a + per, len(offsets) - 1)
        out = {k: d[k][offsets[a]:offsets[b]] for k in KEYS if k not in ('observations', 'observation_interval') and k in d}
        out['observations'] = d['observations'][foff[a]:foff[b]]
        out['observation_interval'] = np.asarray(interval, np.int32)
        yield out


def save(path, d):
    tmp = path.with_name(path.name + f'.tmp{os.getpid()}.npz')
    np.savez_compressed(tmp, **d)
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--env', required=True, help='state env name, e.g. hanoi-single-task1-v0')
    p.add_argument('--idx', required=True, help='collection range a-b (1000 episodes each)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--interval', type=int, default=25)
    p.add_argument('--episodes', type=int, default=1000)
    p.add_argument('--val-episodes', type=int, default=20, help='validation episodes per collection')
    p.add_argument('--per-shard', type=int, default=100)
    p.add_argument('--parallel', type=int, default=520, help='max parallel worlds (pixel frames are held in RAM)')
    args = p.parse_args()
    sys.path.insert(0, str(ROOT / 'ocbench' / 'impls'))
    sys.path.insert(0, str(ROOT / 'scripts'))
    from envs.streaming_data import generate_mjwarp_datasets
    from simulation import configure_iteration_warnings
    configure_iteration_warnings(False)
    venv = f'visual-{args.env}'
    out = ROOT / 'data' / 'ocbench-hf' / 'visual' / venv
    out.mkdir(parents=True, exist_ok=True)
    a, b = (int(x) for x in args.idx.split('-')) if '-' in args.idx else (int(args.idx),) * 2
    shards_per = args.episodes // args.per_shard
    for i in range(a, b + 1):
        names = [out / f'{i * shards_per + j:03d}.npz' for j in range(shards_per)]
        if all(n.exists() and n.with_name(n.stem + '-val.npz').exists() for n in names):
            continue
        t0 = time.time()
        train, val, metrics = generate_mjwarp_datasets(venv, args.episodes, args.val_episodes, args.seed, i,
                                                       max_parallel_episodes=args.parallel,
                                                       observation_interval=args.interval)
        val_per = max(1, -(-args.val_episodes // shards_per))
        vparts = list(split_episodes(val, val_per))
        for j, part in enumerate(split_episodes(train, args.per_shard)):
            save(names[j].with_name(names[j].stem + '-val.npz'), vparts[min(j, len(vparts) - 1)])
            save(names[j], part)
        rec = dict(idx=i, seconds=round(time.time() - t0, 1), frames=int(len(train['observations'])),
                   **{k: float(v) for k, v in metrics.items() if isinstance(v, (int, float, np.floating))})
        print(json.dumps(rec), flush=True)
        with open(out / 'generation.jsonl', 'a') as f:
            f.write(json.dumps(rec) + '\n')
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
