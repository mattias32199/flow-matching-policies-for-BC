"""Which yaw quantities do frozen visual features expose? (block-single, torch venv, GPU)

For each cached embedding (data/emb/<env>/<name>/{train,val}_emb.npy, shape (N, 3 cams, D)) fit ridge and small-MLP
probes, per camera (front, side, wrist) and on all three, for
  sin/cos(k * (cube yaw - effector yaw)), k = 1, 2, 4   (relative yaw; k = 4 is the cube's 90-degree symmetry)
  sin/cos(k * cube yaw), k = 1, 4                         (world-frame cube yaw)
  sin/cos(effector yaw)
Effector yaw is the pinch-site yaw from MuJoCo forward kinematics of the frame's qpos (as in the scripted expert).
Hypotheses: a uniformly coloured cube looks identical under 90-degree rotation, so k = 1 relative yaw should be
undecodable from any image feature while k = 4 is; the wrist camera (gripper frame) should expose relative yaw best.
Output: results/diagnostics/yaw_probe.jsonl (one line per encoder x camera x probe x target).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vis_common import EMB_ROOT, ROOT  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--env', default='visual-block-single-task1-v0')
p.add_argument('--encoders', default='dinov2_s,dinov2_s_p4,pre:h-sgl-enc-nodr-dinov2s-world,'
                                     'pre:h-sgl-enc-nodr-dinov2s-ego,pre:v-F-enc-nodr-dinov2s')
p.add_argument('--n-train', type=int, default=80000)
p.add_argument('--n-val', type=int, default=20000)
p.add_argument('--mlp-steps', type=int, default=4000)
p.add_argument('--out', default=str(ROOT / 'results' / 'diagnostics' / 'yaw_probe.jsonl'))
args = p.parse_args()
dev = torch.device('cuda')
CAMS = ('front', 'side', 'wrist')


def quat_yaw(q):
    w, x, y, z = q.T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def effector_yaw(qpos):
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import gymnasium
    import mujoco
    import ocbench  # noqa: F401
    env = gymnasium.make(args.env.replace('visual-block-', 'block-cpu-'), disable_env_checker=True).unwrapped
    env.reset(seed=0)
    m, d = env.model, env.data
    pinch = m.site('ur5e/robotiq/pinch').id
    out = np.zeros(len(qpos))
    for i, q in enumerate(qpos):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        R = d.site_xmat[pinch].reshape(3, 3)
        out[i] = np.arctan2(R[1, 0], R[0, 0])
    return out


def targets(qpos):
    cube = quat_yaw(qpos[:, 17:21].astype(np.float64))
    eff = effector_yaw(qpos.astype(np.float64))
    rel = cube - eff
    T = {}
    for k in (1, 2, 4):
        T[f'rel_k{k}'] = np.stack([np.sin(k * rel), np.cos(k * rel)], 1)
    for k in (1, 4):
        T[f'cube_k{k}'] = np.stack([np.sin(k * cube), np.cos(k * cube)], 1)
    T['eff_k1'] = np.stack([np.sin(eff), np.cos(eff)], 1)
    return {k: v.astype(np.float32) for k, v in T.items()}


def r2(pred, y):
    return float(np.mean(1 - ((pred - y) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0).clip(1e-12)))


def ridge(Xtr, Ytr, Xva, Yva):
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    ym = Ytr.mean(0)
    G, B = Xtr.T @ Xtr, Xtr.T @ (Ytr - ym)
    eye = torch.eye(G.shape[0], device=dev)
    best = -1e9
    for lam in (1e-2, 1e-1, 1, 10, 100, 1000):
        W = torch.linalg.solve(G + lam * len(Xtr) * 1e-3 * eye, B)
        best = max(best, r2((Xva @ W + ym).cpu().numpy(), Yva.cpu().numpy()))
    return best


def mlp_probe(Xtr, Ytr, Xva, Yva):
    torch.manual_seed(0)
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], 1024), torch.nn.GELU(), torch.nn.LayerNorm(1024),
                              torch.nn.Linear(1024, 1024), torch.nn.GELU(), torch.nn.LayerNorm(1024),
                              torch.nn.Linear(1024, Ytr.shape[1])).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.mlp_steps)
    g = torch.Generator(device=dev).manual_seed(0)
    for _ in range(args.mlp_steps):
        idx = torch.randint(len(Xtr), (512,), device=dev, generator=g)
        loss = ((net(Xtr[idx]) - Ytr[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        return r2(net(Xva).cpu().numpy(), Yva.cpu().numpy())


def main():
    rng = np.random.default_rng(0)
    meta = {s: np.load(EMB_ROOT / args.env / 'dinov2_s' / f'{s}_meta.npz') for s in ('train', 'val')}
    rows = {'train': np.sort(rng.choice(len(meta['train']['qpos']), min(args.n_train, len(meta['train']['qpos'])), replace=False)),
            'val': np.sort(rng.choice(len(meta['val']['qpos']), min(args.n_val, len(meta['val']['qpos'])), replace=False))}
    T = {s: targets(meta[s]['qpos'][rows[s]]) for s in rows}
    print('targets ready', {k: v.shape for k, v in T['train'].items()}, flush=True)
    # State reference: probes on the yaw part of the privileged state (cos/sin of cube and effector yaw).
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    state = {s: torch.from_numpy(np.concatenate([T[s]['cube_k1'], T[s]['eff_k1']], 1)).to(dev) for s in T}
    recs = []
    for tname in T['train']:
        Ytr, Yva = (torch.from_numpy(T[s][tname]).to(dev) for s in ('train', 'val'))
        for kind, fn in (('ridge', ridge), ('mlp', mlp_probe)):
            recs.append(dict(encoder='state:cos/sin(cube,eff yaw)', camera='-', probe=kind, target=tname,
                             r2=fn(state['train'], Ytr, state['val'], Yva)))
            print(json.dumps(recs[-1]), flush=True)
    for name in args.encoders.split(','):
        X = {s: np.load(EMB_ROOT / args.env / name / f'{s}_emb.npy', mmap_mode='r') for s in rows}
        X = {s: torch.from_numpy(np.asarray(X[s][rows[s]], np.float32)).to(dev) for s in rows}
        for cam in (0, 1, 2, 'all'):
            sel = {s: (X[s][:, cam] if cam != 'all' else X[s].flatten(1)) for s in X}
            for tname in T['train']:
                Ytr, Yva = (torch.from_numpy(T[s][tname]).to(dev) for s in ('train', 'val'))
                for kind, fn in (('ridge', ridge), ('mlp', mlp_probe)):
                    if kind == 'mlp' and cam == 'all' and sel['train'].shape[1] > 4000:
                        continue
                    recs.append(dict(encoder=name, camera=CAMS[cam] if cam != 'all' else 'all', probe=kind, target=tname,
                                     r2=fn(sel['train'], Ytr, sel['val'], Yva), dim=int(sel['train'].shape[1])))
                    print(json.dumps(recs[-1]), flush=True)
        del X
    with out.open('a') as f:
        for r in recs:
            f.write(json.dumps(r | {'env': args.env, 'n_train': len(rows['train'])}) + '\n')
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
