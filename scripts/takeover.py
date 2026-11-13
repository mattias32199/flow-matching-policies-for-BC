"""Expert takeover diagnostic on MJWarp block tasks (state policies).

For each takeover time k, the same N initial states (eval seeding) are rolled out by a trained
student FBC policy for k steps (25-step open-loop chunks, as in evaluation), then the OCBench
scripted expert (BlockMjWarpController, re-planned from the current simulator state) controls the
rest of the episode. Reports per k:
  student_success_by_k   task already solved by the student before the handoff
  final_success          solved by the end of the episode (student or expert)
  expert_rescue          final success among episodes still unsolved and alive at k
k=0 is the expert alone; k=H is the student alone. --students none runs only the expert, e.g. with
OCB_MISTAKE_PROB=0 to measure the expert without its deliberate 10% mistakes.

  python scripts/takeover.py --students m4-handcrafted-s0-cont2,m1-double-n10000-s0 --episodes 500
Writes results/diagnostics/takeover/<env>.jsonl (one line per student x k).
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'ocbench' / 'impls'))

import flax  # noqa: E402
import jax  # noqa: E402
import numpy as np  # noqa: E402
import ocbench  # noqa: E402
from agents.fbc import FBCAgent, get_config  # noqa: E402
from envs.streaming_data import make_collection_seeds  # noqa: E402
from ocbench.mjwarp import make_env  # noqa: E402
from ocbench.mjwarp.controllers.block import BlockMjWarpController  # noqa: E402
from utils.evaluation import supply_rng  # noqa: E402

from train import EvalAgent, saved_preprocessing  # noqa: E402

OUT = ROOT / 'results' / 'diagnostics' / 'takeover'


def load_student(run):
    d = ROOT / 'runs' / run
    cfg = json.loads((d / 'config.json').read_text())
    features, normalizer = saved_preprocessing(d / 'preprocessing.npz', cfg.get('features', 'handcrafted'))
    ck = d / 'ckpt.pkl'
    if not ck.exists():
        ck = max((d / 'checkpoints').glob('params_*.pkl'), key=lambda p: int(p.stem.split('_')[-1]))
    with open(ck, 'rb') as f:
        state = pickle.load(f)['agent']
    c = get_config()
    c.actor_hidden_dims = (cfg.get('width', 4096),) * cfg.get('depth', 8)
    obs_dim = len(features.center) + 2 * len(features.xyz) * len(features.frequencies)
    agent = FBCAgent.create(0, np.zeros((1, obs_dim), np.float32), np.zeros((1, 25 * 7), np.float32), c)
    agent = flax.serialization.from_state_dict(agent, state)
    return EvalAgent(agent, features), normalizer, str(ck)


def episode(env_name, n, k, student, normalizer, seed):
    family, _, kwargs, H = ocbench.parse_env_spec(env_name)
    env = make_env(family, kwargs, n)
    np.random.seed(seed)
    seeds = make_collection_seeds(np.random.randint(0, 2**32), 0, 10, n)
    obs, _ = env.reset(seeds=seeds)
    done = np.zeros(n, bool)
    solved = np.zeros(n, bool)
    t = 0
    if student is not None and k > 0:
        actor = supply_rng(student.sample_actions, rng=jax.random.PRNGKey(seed))
        lo, hi = env.single_action_space.low, env.single_action_space.high
        while t < k and not done.all():
            a = np.asarray(actor(observations=obs), np.float32)
            a = np.clip(a, np.tile(normalizer.action_low, 25), np.tile(normalizer.action_high, 25))
            chunk = normalizer.unnormalize(a.reshape(-1, 7)).reshape(n, 25, 7)
            chunk = np.clip(chunk, lo, hi)
            for c in range(25):
                if t >= k:
                    break
                act = chunk[:, c].copy()
                act[done] = 0
                obs, _, term, _, info = env.step(act)
                t += 1
                solved |= info['success'] & ~done
                done |= term | (t >= H)
    by_k = solved.copy()
    alive_unsolved = ~done
    rescued = np.zeros(n, bool)
    if t < H and alive_unsolved.any():
        env._ensure_ik_buffers()
        oracle_seeds = make_collection_seeds(seed + 7, 0, 11, n)
        ctrl = BlockMjWarpController(env, seed=oracle_seeds, max_steps=H - t)
        ctrl.reset()
        ctrl.done.assign(done.astype(np.int32))
        for s in range(H - t):
            tp, tx, tg = ctrl.make_targets()
            act = env.ee_target_arrays_to_joint_actions_gpu(tp, tx, tg)
            env.step_joint_actions_gpu(act, ctrl.done)
            ctrl.update_done()
            if s % 100 == 99 and ctrl.done.numpy().all():
                break
        rescued = ctrl.episode_success.numpy().astype(bool) & alive_unsolved
    env.close()
    final = by_k | rescued
    return dict(student_success_by_k=float(by_k.mean()), final_success=float(final.mean()),
                alive_unsolved_at_k=float(alive_unsolved.mean()),
                expert_rescue=float(rescued.sum() / max(alive_unsolved.sum(), 1)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--env', default='block-double-task1-v0')
    p.add_argument('--students', default='none')
    p.add_argument('--ks', default='0,125,250,500,750,1000,1250,1500,1750,2000,2500')
    p.add_argument('--episodes', type=int, default=500)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--tag', default='')
    args = p.parse_args()
    from simulation import configure_iteration_warnings
    configure_iteration_warnings(False)
    OUT.mkdir(parents=True, exist_ok=True)
    mistake = float(os.environ.get('OCB_MISTAKE_PROB', '0.1'))
    out = OUT / f'{args.env}.jsonl'
    done = set()
    if out.exists():  # requeue-safe: skip (student, k, mistake, episodes, tag) already measured
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r['student'], r['k'], r['expert_mistake_prob'], r['episodes'], r['tag']))
    for name in args.students.split(','):
        student, normalizer, ck = (None, None, None) if name == 'none' else load_student(name)
        ks = [0] if student is None else [int(x) for x in args.ks.split(',')]
        for k in ks:
            if (name, k, mistake, args.episodes, args.tag) in done:
                continue
            t0 = time.time()
            res = episode(args.env, args.episodes, k, student, normalizer, args.seed)
            rec = dict(env=args.env, student=name, checkpoint=ck, k=k, episodes=args.episodes,
                       expert_mistake_prob=mistake, tag=args.tag, seconds=round(time.time() - t0, 1), **res)
            print(json.dumps(rec), flush=True)
            with open(out, 'a') as f:
                f.write(json.dumps(rec) + '\n')


if __name__ == '__main__':
    main()
