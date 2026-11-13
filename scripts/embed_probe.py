"""Cache frozen pretrained-encoder embeddings for OCBench visual datasets, then linear-probe state.

Run with .local/venv-torch/bin/python. Each shard is read once (allow_pickle=False) and passed
through every encoder. Outputs, per dataset and encoder, under data/emb/<env>/<encoder>/:
  {split}_emb.npy   float16 (frames, 3 cameras, dim)
  {split}_meta.npz  frame -> transition index (step), qpos label, episode id, is_final, shard
Probe: ridge regression from concatenated 3-camera embeddings to qpos groups, R^2 on held-out
validation shards; written to results/diagnostics/probe/<env>.jsonl. With --fractions, also the
probe as a function of how many training episodes carry state labels (<env>-fraction.jsonl).
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from vis_common import EMB_ROOT, ROOT, Embedder, frame_meta, shard_files

p = argparse.ArgumentParser()
p.add_argument('--env', default='visual-block-double-task1-v0')
p.add_argument('--shards', type=int, default=100)
p.add_argument('--encoders', default='dinov2_s,dinov2_b,resnet50_ce,clip_b16,random_vit_s')
p.add_argument('--batch', type=int, default=384)
p.add_argument('--probe-only', action='store_true')
p.add_argument('--no-probe', action='store_true')
p.add_argument('--fractions', default='', help='comma list of labelled-episode fractions, e.g. 0.001,0.01,0.1,1')
args = p.parse_args()

names = args.encoders.split(',')
out_root = EMB_ROOT / args.env
probe_out = ROOT / 'results' / 'diagnostics' / 'probe'
probe_out.mkdir(parents=True, exist_ok=True)
dev = torch.device('cuda')


def load(path):
    with np.load(path, allow_pickle=False) as d:
        d = {k: d[k] for k in ('observations', 'observation_interval', 'terminals', 'qpos')}
    steps, eps, final, _ = frame_meta(d['terminals'], int(d['observation_interval']), len(d['observations']))
    return d['observations'], dict(step=steps, episode=eps, final=final, qpos=d['qpos'][steps])


def run_embedding():
    models = {n: Embedder(n, dev, args.batch) for n in names}
    for split in ('train', 'val'):
        files = shard_files(split, args.env, args.shards)
        embs = {n: [] for n in names}
        metas = []
        t0, nframes = time.time(), 0
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(load, f) for f in files[:2]]
            for i, f in enumerate(files):
                frames, meta = futures[i].result()
                if i + 2 < len(files):
                    futures.append(pool.submit(load, files[i + 2]))
                meta['shard'] = np.full(len(frames), i)
                metas.append(meta)
                for n in names:
                    embs[n].append(models[n](frames).cpu().numpy())
                nframes += len(frames)
                print(json.dumps(dict(split=split, shard=i, frames=nframes,
                                      img_per_s=nframes * 3 * len(names) / (time.time() - t0))), flush=True)
                futures[i] = None
        meta = {k: np.concatenate([m[k] for m in metas]) for k in metas[0]}
        for n in names:
            d = out_root / n
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / f'{split}_emb.npy', np.concatenate(embs[n]))
            np.savez(d / f'{split}_meta.npz', **meta)


def r2(pred, y):
    return 1 - ((pred - y) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0).clip(1e-12)


def groups():
    g = {'arm_qpos': list(range(0, 6)), 'gripper_qpos': [6], 'cube1_xyz': [14, 15, 16]}
    if 'double' in args.env:
        g['cube2_xyz'] = [21, 22, 23]
    return g


def chunks(X, rows, size=32768):
    for i in range(0, len(rows), size):
        yield torch.from_numpy(np.asarray(X[rows[i:i + size]], np.float32).reshape(len(rows[i:i + size]), -1)).to(dev)


def ridge_fit(X, Y, rows, mu, sd, lams):
    """Chunked normal equations so wide (e.g. 4x4 patch-pooled) features fit on the GPU."""
    ym = torch.from_numpy(Y[rows].mean(0)).to(dev)
    G = B = None
    for i, x in zip(range(0, len(rows), 32768), chunks(X, rows)):
        x = (x - mu) / sd
        y = torch.from_numpy(Y[rows[i:i + 32768]]).to(dev) - ym
        G = x.T @ x if G is None else G + x.T @ x
        B = x.T @ y if B is None else B + x.T @ y
    eye = torch.eye(G.shape[0], device=dev)
    return {lam: torch.linalg.solve(G + lam * len(rows) * 1e-3 * eye, B) for lam in lams}, ym


def probe_encoder(n, fractions):
    d = out_root / n
    X = {s: np.load(d / f'{s}_emb.npy', mmap_mode='r') for s in ('train', 'val')}
    meta = {s: np.load(d / f'{s}_meta.npz') for s in ('train', 'val')}
    Y = {s: meta[s]['qpos'].astype(np.float32) for s in ('train', 'val')}
    all_rows = np.arange(len(X['train']))
    s1 = s2 = None
    for x in chunks(X['train'], all_rows):
        s1 = x.sum(0) if s1 is None else s1 + x.sum(0)
        s2 = (x * x).sum(0) if s2 is None else s2 + (x * x).sum(0)
    mu = s1 / len(all_rows)
    sd = (s2 / len(all_rows) - mu ** 2).clamp_min(1e-12).sqrt().clamp_min(1e-6)
    val_rows = np.arange(len(X['val']))
    lams = (1e-3, 1e-2, 1e-1, 1, 10, 100)
    uid = meta['train']['shard'] * 100000 + meta['train']['episode']
    episodes = np.unique(uid)
    perm = np.random.default_rng(0).permutation(episodes)
    results = []
    for frac in fractions:
        chosen = perm[:max(1, int(round(frac * len(episodes))))]
        rows = all_rows if frac >= 1 else np.flatnonzero(np.isin(uid, chosen))
        Ws, ym = ridge_fit(X['train'], Y['train'], rows, mu, sd, lams)
        preds = {lam: [] for lam in lams}
        for x in chunks(X['val'], val_rows):
            x = (x - mu) / sd
            for lam, W in Ws.items():
                preds[lam].append((x @ W + ym).cpu().numpy())
        best = None
        for lam in lams:
            score = r2(np.concatenate(preds[lam]), Y['val'])
            per_group = {g: float(np.mean(score[i])) for g, i in groups().items()}
            mean_score = float(np.mean(list(per_group.values())))
            if best is None or mean_score > best['mean_r2']:
                best = dict(encoder=n, lam=lam, mean_r2=mean_score, r2=per_group, dim=int(mu.shape[0]),
                            n_train=int(len(rows)), n_val=int(len(val_rows)), env=args.env,
                            label_fraction=frac, label_episodes=int(len(chosen) if frac < 1 else len(episodes)))
        results.append(best)
        print(json.dumps(best), flush=True)
    return results


if not args.probe_only:
    run_embedding()
if not args.no_probe:
    for n in names:
        (full,) = probe_encoder(n, [1.0])
        with (probe_out / f'{args.env}.jsonl').open('a') as f:
            f.write(json.dumps(full) + '\n')
        if args.fractions:
            for r in probe_encoder(n, [float(x) for x in args.fractions.split(',')]):
                with (probe_out / f'{args.env}-fraction.jsonl').open('a') as f:
                    f.write(json.dumps(r) + '\n')
print('DONE', flush=True)
