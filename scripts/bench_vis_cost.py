"""Measure the compute cost of turning block-double state BC into visual BC.

--part sim   (JAX venv, GPU): expert episode collection with and without rendering, i.e. the
             "stream render" cost of the OCBench visual pipeline (3 cameras, 224 px) at observation
             intervals 25 and 5, versus state-only collection.
--part torch (torch venv, GPU): re-rendering saved qpos with the CPU/EGL renderer (per core), and
             frozen-encoder embedding throughput (DINOv2-S CLS / 4x4 patch pooling, 3 cameras).
Appends JSON lines to results/diagnostics/vis_cost.jsonl.
"""
import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'diagnostics' / 'vis_cost.jsonl'


def emit(rec):
    rec = dict(rec, host=socket.gethostname(), job=os.environ.get('SLURM_JOB_ID'), time=time.time())
    try:
        import subprocess
        rec['gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).split('\n')[0]
    except Exception:
        pass
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    print(json.dumps(rec), flush=True)


def part_sim(args):
    sys.path.insert(0, str(ROOT / 'ocbench' / 'impls'))
    from envs.streaming_data import generate_mjwarp_datasets
    sys.path.insert(0, str(ROOT / 'scripts'))
    from simulation import configure_iteration_warnings
    configure_iteration_warnings(False)
    for env, interval, n in [('block-double-task1-v0', 1, args.episodes), ('visual-block-double-task1-v0', 25, args.episodes),
                             ('visual-block-double-task1-v0', 5, args.episodes // 2)]:
        for rep in range(2):  # first repetition includes kernel compilation
            t0 = time.time()
            kw = dict(observation_interval=interval) if interval > 1 else {}
            train, _, m = generate_mjwarp_datasets(env, n, 0, 123, 900 + rep, max_parallel_episodes=n, **kw)
            dt = time.time() - t0
            frames = len(train['observations'])
            emit(dict(kind='collect', env=env, interval=interval, episodes=n, rep=rep, seconds=round(dt, 1),
                      transitions=int(len(train['actions'])), frames=int(frames),
                      sec_per_1k_episodes=round(dt / n * 1000, 1), frames_per_sec=round(frames / dt, 1)))
            del train


def part_torch(args):
    import torch
    # (1) CPU/EGL re-render of saved states, one process (multiply by cores for a pool).
    os.environ.setdefault('MUJOCO_GL', 'egl')
    sys.path.insert(0, str(ROOT / 'scripts'))
    import mujoco
    from dr_gen import CAMS
    import gymnasium
    import ocbench  # noqa: F401
    env = gymnasium.make('visual-block-cpu-double-task1-v0', disable_env_checker=True).unwrapped
    env.reset(seed=0)
    m, d = env.model, env.data
    qs = np.load(sorted((ROOT / 'data' / 'collections' / 'block-double-task1-v0-d0').glob('c00000.npz'))[0])['qpos'][::997][:args.renders].astype(np.float64)
    t0 = time.time()
    for q in qs:
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        for c in CAMS:
            env.render(camera=c)
    dt = time.time() - t0
    emit(dict(kind='rerender_cpu_egl', states=len(qs), cams=3, size=224, seconds=round(dt, 2),
              states_per_sec_per_process=round(len(qs) / dt, 1)))
    # (2) frozen-encoder embedding throughput.
    from vis_common import Embedder
    for enc in ('dinov2_s', 'dinov2_s_p4'):
        e = Embedder(enc)
        x = torch.randint(0, 255, (256, 3, 112, 112, 3), dtype=torch.uint8).cuda()  # 256 frames x 3 cams
        for rep in range(3):
            torch.cuda.synchronize(); t0 = time.time()
            for _ in range(10):
                e(x)
            torch.cuda.synchronize(); dt = time.time() - t0
        emit(dict(kind='embed', encoder=enc, frames3cam_per_sec=round(2560 / dt, 1), images_per_sec=round(3 * 2560 / dt, 1)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--part', choices=['sim', 'torch'], required=True)
    p.add_argument('--episodes', type=int, default=200)
    p.add_argument('--renders', type=int, default=300)
    args = p.parse_args()
    part_sim(args) if args.part == 'sim' else part_torch(args)


if __name__ == '__main__':
    main()
