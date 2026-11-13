"""Build the visual-BC frame index, action normalizer and 112px image cache (torch venv).

Outputs under data/vis/<env>/:
  normalizer.npz        ActionNormalizer.fit(all train actions, 1, 5) as in scripts/train.py
  {split}_index.npz     keep (all cached frames, same order as data/emb), and for kept frames
                        (successful episodes, non-final frames): normalized 25-step action chunks,
                        qpos label, episode uid, step
  {split}_img112.u8     raw uint8 (kept, 3, 112, 112, 3) + {split}_img112.json shape
"""
import argparse
import json
import sys
import time

import numpy as np
import torch

from vis_common import CHUNK, ENV, ROOT, VIS_ROOT, frame_meta, resize_uint8, shard_files

sys.path.insert(0, str(ROOT / 'ocbench' / 'impls'))
from envs.env_utils import ActionNormalizer  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--env', default=ENV)
p.add_argument('--shards', type=int, default=100)
p.add_argument('--img', type=int, default=112)
p.add_argument('--no-images', action='store_true', help='index/normalizer only (frozen-embedding training)')
args = p.parse_args()
out = VIS_ROOT / args.env
out.mkdir(parents=True, exist_ok=True)

actions = []
for f in shard_files('train', args.env, args.shards):
    with np.load(f, allow_pickle=False) as d:
        actions.append(d['actions'])
normalizer = ActionNormalizer.fit(np.concatenate(actions), 1., 5.)
np.savez(out / 'normalizer.npz', q_low=normalizer.q_low, q_high=normalizer.q_high,
         clip=normalizer.clip, gripper_dims=np.flatnonzero(normalizer.gripper_dims))
del actions

for split in ('train', 'val'):
    t0 = time.time()
    keep_all, chunks, qpos, uid, steps_out = [], [], [], [], []
    shape = None
    img_path = out / f'{split}_img{args.img}.u8'
    n_img = 0
    with open(str(img_path) + '.tmp', 'wb') as img_file:
        for i, f in enumerate(shard_files(split, args.env, args.shards)):
            with np.load(f, allow_pickle=False) as d:
                d = {k: d[k] for k in ('observations', 'observation_interval', 'terminals', 'rewards', 'actions', 'qpos')}
            steps, eps, final, offsets = frame_meta(d['terminals'], int(d['observation_interval']), len(d['observations']))
            success = np.array([d['rewards'][a:b].max() >= 1 for a, b in zip(offsets[:-1], offsets[1:])])
            keep = success[eps] & ~final
            keep_all.append(keep)
            s = steps[keep]; e = eps[keep]
            term = offsets[e + 1] - 1
            idx = np.minimum(s[:, None] + np.arange(CHUNK)[None], term[:, None])
            chunks.append(normalizer.normalize(d['actions'][idx]).astype(np.float32))
            qpos.append(d['qpos'][s].astype(np.float32))
            uid.append(i * 100000 + e)
            steps_out.append(s - offsets[e])
            n_img += 0 if not args.no_images else int(keep.sum())
            for j in range(0, keep.sum() if not args.no_images else 0, 2048):
                sel = np.flatnonzero(keep)[j:j + 2048]
                small = resize_uint8(d['observations'][sel], args.img).cpu().numpy()
                img_file.write(small.tobytes())
                n_img += len(small)
            print(json.dumps(dict(split=split, shard=i, kept=n_img, seconds=round(time.time() - t0))), flush=True)
    shape = [n_img, 3, args.img, args.img, 3]
    (out / f'{split}_img{args.img}.json').write_text(json.dumps({'shape': shape}))
    if args.no_images:
        (img_path.with_name(img_path.name + '.tmp')).unlink()
    else:
        (img_path.with_name(img_path.name + '.tmp')).replace(img_path)
    np.savez(out / f'{split}_index.npz', keep=np.concatenate(keep_all), chunks=np.concatenate(chunks),
             qpos=np.concatenate(qpos), uid=np.concatenate(uid), step=np.concatenate(steps_out))
    print(f'{split}: {n_img} kept frames of {sum(len(k) for k in keep_all)}', flush=True)
print('DONE', flush=True)
