"""Group F data: (3-camera image, state-target) pairs rendered in sim, with and without visual DR.

States come from the visual-block-single training demos (data/vis/<env>/train_index.npz qpos),
perturbed so the set covers more than the demo manifold: half the samples get a uniformly
re-placed cube (xy within the demo range, random yaw, resting on the table), and all samples get
arm-joint noise (sigma 0.05 rad) and a random gripper opening. Every state is rendered twice:
  nodr  nominal OCBench appearance and cameras (same look as the demo dataset)
  dr    random geom/material colours, light/headlight, camera position/orientation jitter
Targets (27-d, in targets.npy, documented in meta.json): cube position in each camera frame
(3x3), cube position in the gripper (pinch-site) frame (3), sin/cos(4*cube yaw) (2), pinch-site
world position (3), gripper opening (1), arm joints (6), cube world position (3).

Output: data/dr/<env>/{nodr,dr}_img112.u8 (N, 3, 112, 112, 3), targets.npy, qpos.npy, meta.json.
Run on a GPU node (EGL), torch venv: python scripts/dr_gen.py --n 200000 --workers 24
"""
import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAMS = ('front', 'side', 'ur5e/wrist')
def target_layout(ncubes):
    """Per cube: position in each camera frame (9), in the gripper frame (3), sin/cos(4 yaw) (2), world xyz (3);
    then pinch xyz (3), gripper opening (1), arm joints (6). ncubes=1 reproduces the single-task layout's groups."""
    names, groups = [], {}
    def add(group, new):
        groups.setdefault(group, []).extend(range(len(names), len(names) + len(new)))
        names.extend(new)
    for k in range(ncubes):
        p = f'cube{k}_' if ncubes > 1 else 'cube_'
        add('cube_in_camera', [f'{p}in_{c.split("/")[-1]}_{a}' for c in CAMS for a in 'xyz'])
        add('cube_in_gripper', [f'{p}in_gripper_{a}' for a in 'xyz'])
        add('cube_yaw', [f'{p}sin4yaw', f'{p}cos4yaw'])
        add('cube_xyz', [f'{p}{a}' for a in 'xyz'])
    add('pinch_xyz', [f'pinch_{a}' for a in 'xyz'])
    add('gripper', ['gripper_open'])
    add('arm_qpos', [f'arm_q{i}' for i in range(6)])
    return names, groups


def make_env(env_name='visual-block-single-task1-v0'):
    import gymnasium
    import ocbench  # noqa: F401  (registers envs)
    env = gymnasium.make(env_name.replace('visual-block-', 'visual-block-cpu-'), disable_env_checker=True).unwrapped
    env.reset(seed=0)
    return env


def quat_yaw(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def yaw_quat(yaw):
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


def worker(args):
    wid, lo, hi, qpos_src, cfg = args
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import mujoco
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    env = make_env(cfg['env'])
    m, d = env.model, env.data
    rng = np.random.default_rng(cfg['seed'] * 1000 + wid)
    cam_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, c) for c in CAMS]
    pinch = m.site('ur5e/robotiq/pinch').id
    qas = [int(m.jnt_qposadr[j]) for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    names, _ = target_layout(len(qas))
    arm = np.asarray(env._arm_joint_ids)
    grip = env._gripper_opening_joint_id
    nom = dict(geom_rgba=m.geom_rgba.copy(), mat_rgba=m.mat_rgba.copy(), light_diffuse=m.light_diffuse.copy(),
               light_ambient=m.light_ambient.copy(), light_specular=m.light_specular.copy(), light_pos=m.light_pos.copy(),
               cam_pos=m.cam_pos.copy(), cam_quat=m.cam_quat.copy(),
               head_amb=m.vis.headlight.ambient.copy(), head_dif=m.vis.headlight.diffuse.copy())
    imgs = {k: np.memmap(cfg['out'] / f'{k}_img112.u8', np.uint8, 'r+', shape=(cfg['n'], 3, 112, 112, 3)) for k in cfg['variants']}
    targets = np.zeros((hi - lo, len(names)), np.float32)
    qpos_out = np.zeros((hi - lo, m.nq), np.float32)
    lim = cfg['cube_lim']

    def restore():
        m.geom_rgba[:] = nom['geom_rgba']; m.mat_rgba[:] = nom['mat_rgba']
        m.light_diffuse[:] = nom['light_diffuse']; m.light_ambient[:] = nom['light_ambient']
        m.light_specular[:] = nom['light_specular']; m.light_pos[:] = nom['light_pos']
        m.cam_pos[:] = nom['cam_pos']; m.cam_quat[:] = nom['cam_quat']
        m.vis.headlight.ambient[:] = nom['head_amb']; m.vis.headlight.diffuse[:] = nom['head_dif']

    def randomize():
        n = m.ngeom
        hue = rng.uniform(0, 1, (n, 3))
        mix = rng.uniform(0.3, 1.0, (n, 1))
        m.geom_rgba[:, :3] = np.clip((1 - mix) * nom['geom_rgba'][:, :3] + mix * hue, 0, 1)
        if m.nmat:
            m.mat_rgba[:, :3] = np.clip(nom['mat_rgba'][:, :3] * rng.uniform(0.4, 1.6, (m.nmat, 3)), 0, 1)
        if m.nlight:
            m.light_diffuse[:] = np.clip(nom['light_diffuse'] * rng.uniform(0.3, 1.5, (m.nlight, 1)), 0, 1)
            m.light_ambient[:] = np.clip(nom['light_ambient'] + rng.uniform(0, 0.3, (m.nlight, 1)), 0, 1)
            m.light_pos[:] = nom['light_pos'] + rng.normal(0, 0.5, nom['light_pos'].shape)
        m.vis.headlight.ambient[:] = rng.uniform(0.0, 0.5, 3)
        m.vis.headlight.diffuse[:] = rng.uniform(0.2, 0.9, 3)
        for c in cam_ids:
            m.cam_pos[c] = nom['cam_pos'][c] + rng.normal(0, 0.02 if c != cam_ids[2] else 0.005, 3)
            axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
            ang = rng.normal(0, np.deg2rad(3 if c != cam_ids[2] else 1.5))
            dq = np.concatenate([[np.cos(ang / 2)], np.sin(ang / 2) * axis])
            q = np.zeros(4); mujoco.mju_mulQuat(q, dq, nom['cam_quat'][c]); m.cam_quat[c] = q

    def render_all():
        frames = np.stack([env.render(camera=c) for c in CAMS])  # (3, 224, 224, 3)
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        x = F.interpolate(x, size=(112, 112), mode='bilinear', antialias=True)
        return x.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()

    t0 = time.time()
    for i in range(lo, hi):
        q = qpos_src[rng.integers(len(qpos_src))].astype(np.float64).copy()
        if rng.random() < 0.5:  # re-place every cube on the table (pairwise >= min_dist apart)
            placed = []
            for qa in qas:
                for _ in range(100):
                    xy = rng.uniform(lim[0], lim[1])
                    if all(np.linalg.norm(xy - o) >= cfg['min_dist'] for o in placed):
                        break
                placed.append(xy)
                q[qa:qa + 2] = xy
                q[qa + 2] = lim[2]
                q[qa + 3:qa + 7] = yaw_quat(rng.uniform(-np.pi, np.pi))
        q[arm] += rng.normal(0, 0.05, len(arm))
        d.qpos[:] = q
        d.qpos[grip] = rng.uniform(0, 0.8)
        d.qvel[:] = 0
        restore()
        mujoco.mj_forward(m, d)
        if 'nodr' in imgs:
            imgs['nodr'][i] = render_all()
        if 'dr' in imgs:
            randomize()
            mujoco.mj_forward(m, d)  # camera jitter changes cam_xpos/xmat
            imgs['dr'][i] = render_all()
            restore()
            mujoco.mj_forward(m, d)  # targets are computed with the nominal cameras
        t = []
        for qa in qas:
            cube = d.qpos[qa:qa + 3].copy()
            for c in cam_ids:
                t.extend(d.cam_xmat[c].reshape(3, 3).T @ (cube - d.cam_xpos[c]))
            t.extend(d.site_xmat[pinch].reshape(3, 3).T @ (cube - d.site_xpos[pinch]))
            yaw = quat_yaw(d.qpos[qa + 3:qa + 7])
            t.extend([np.sin(4 * yaw), np.cos(4 * yaw)])
            t.extend(cube)
        t.extend(d.site_xpos[pinch]); t.append(d.qpos[grip] / 0.8); t.extend(d.qpos[arm])
        targets[i - lo] = t
        qpos_out[i - lo] = d.qpos
        if wid == 0 and (i - lo) % 500 == 0:
            print(json.dumps(dict(worker=0, done=i - lo, of=hi - lo, seconds=round(time.time() - t0))), flush=True)
    restore()
    for v in imgs.values():
        v.flush()
    return lo, targets, qpos_out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--env', default='visual-block-single-task1-v0')
    p.add_argument('--n', type=int, default=200000)
    p.add_argument('--workers', type=int, default=24)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--variants', default='nodr,dr')
    args = p.parse_args()
    out = ROOT / 'data' / 'dr' / args.env
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'DONE').exists():
        print('already generated', flush=True)
        return
    with np.load(ROOT / 'data' / 'vis' / args.env / 'train_index.npz') as z:
        qpos_src = z['qpos']
    env = make_env(args.env)
    import mujoco
    qas = [int(env.model.jnt_qposadr[j]) for j in range(env.model.njnt) if env.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    cube = np.concatenate([qpos_src[:, qa:qa + 3] for qa in qas])
    lim = (cube[:, :2].min(0).tolist(), cube[:, :2].max(0).tolist(), float(np.percentile(cube[:, 2], 1)))
    names, groups = target_layout(len(qas))
    print(json.dumps(dict(nq=env.model.nq, cube_qpos_adr=qas, cube_lim=lim, ngeom=env.model.ngeom,
                          nmat=env.model.nmat, nlight=env.model.nlight, targets=len(names))), flush=True)
    del env
    for k in args.variants.split(','):
        np.memmap(out / f'{k}_img112.u8', np.uint8, 'w+', shape=(args.n, 3, 112, 112, 3)).flush()
    cfg = dict(seed=args.seed, out=out, n=args.n, cube_lim=lim, env=args.env, min_dist=0.08, variants=args.variants.split(','))
    bounds = [int(b) for b in np.linspace(0, args.n, args.workers + 1)]
    targets = np.zeros((args.n, len(names)), np.float32)
    qpos = None
    t0 = time.time()
    with mp.get_context('spawn').Pool(args.workers) as pool:
        # map_async + timeout: a worker killed by a native crash fails the job instead of hanging it.
        results = pool.map_async(worker, [(w, bounds[w], bounds[w + 1], qpos_src, cfg) for w in range(args.workers)]).get(timeout=4 * 3600)
        for lo, t, q in results:
            targets[lo:lo + len(t)] = t
            qpos = np.zeros((args.n, q.shape[1]), np.float32) if qpos is None else qpos
            qpos[lo:lo + len(q)] = q
    np.save(out / 'targets.npy', targets)
    np.save(out / 'qpos.npy', qpos)
    (out / 'meta.json').write_text(json.dumps(dict(n=args.n, shape=[args.n, 3, 112, 112, 3], cameras=CAMS,
                                                   target_names=names, groups=groups, seconds=time.time() - t0), indent=1))
    (out / 'DONE').write_text('ok\n')
    print('DONE', time.time() - t0, flush=True)


if __name__ == '__main__':
    main()
