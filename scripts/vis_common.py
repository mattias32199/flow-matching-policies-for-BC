"""Shared pieces for the Mystery-4 visual experiments (torch venv: .local/venv-torch).

* Frozen-encoder registry used both to cache embeddings and to embed live eval images.
* Torch port of the upstream FBC actor (ResidualMLP [4096]x8, LayerNorm, GELU-tanh, 65-d time
  features, Beta(1, 1.5) flow times, 10 Euler steps) so visual and state runs share the policy.
* Trainable pixel encoders (IMPALA-small port, timm ResNet-18 / DINOv2-S) for end-to-end runs.
* Visual MJWarp evaluation with sparse (every 25 steps) 3-camera observations, mirroring
  ocbench/impls/utils/evaluation_mjwarp.py's sparse branch.
"""
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
ENV = 'visual-block-single-task1-v0'
HF_ROOT = ROOT / 'data' / 'ocbench-hf' / 'visual'
EMB_ROOT = ROOT / 'data' / 'emb'
VIS_ROOT = ROOT / 'data' / 'vis'
CHUNK = 25
SIMCLR_FILE = 'resnet50-1x.pth'
SIMCLR_REPO = 'lightly-ai/simclrv1-imagenet1k-resnet50-1x'

# name -> (timm arch, weights, create kwargs, pooling)
ENC = {
    'dinov2_s': ('vit_small_patch14_dinov2.lvd142m', 'timm', dict(img_size=224), 'cls'),
    'dinov2_b': ('vit_base_patch14_dinov2.lvd142m', 'timm', dict(img_size=224), 'cls'),
    'resnet50_ce': ('resnet50.a1_in1k', 'timm', {}, 'cls'),
    'clip_b16': ('vit_base_patch16_clip_224.openai', 'timm', {}, 'cls'),
    'random_vit_s': ('vit_small_patch14_dinov2.lvd142m', 'random', dict(img_size=224), 'cls'),
    'mae_b16': ('vit_base_patch16_224.mae', 'timm', {}, 'cls'),
    'simclr_r50': ('resnet50', 'simclr', {}, 'cls'),
    'dinov2_s_p2': ('vit_small_patch14_dinov2.lvd142m', 'timm', dict(img_size=224), 'p2'),
    'dinov2_s_p4': ('vit_small_patch14_dinov2.lvd142m', 'timm', dict(img_size=224), 'p4'),
}


def device():
    return torch.device('cuda')


def amp_dtype():
    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


# ----------------------------------------------------------------------------- frozen encoders
def load_pre(run_name):
    """Backbone exported by scripts/pretrain_enc.py (runs/<run>/encoder.pt) -> (timm model, meta)."""
    import timm
    ckpt = torch.load(ROOT / 'runs' / run_name / 'encoder.pt', map_location='cpu', weights_only=True)
    meta = json.loads(ckpt['meta'])
    model = timm.create_model(meta['arch'], pretrained=False, num_classes=0, **meta['kwargs'])
    model.load_state_dict(ckpt['state_dict'])  # strict: no missing / unexpected keys
    return model, meta


class Embedder:
    """uint8 frames (F, cams, H, W, C) -> float16 features (F, cams, D), exactly as cached."""

    def __init__(self, name, dev=None, batch=384):
        import timm
        dev = dev or device()
        self.name, self.batch, self.dev = name, batch, dev
        if name.startswith('pre:'):  # pre:<run> (CLS) or pre:<run>@p4 (4x4 average-pooled patch tokens)
            rname, _, pool = name[4:].partition('@')
            model, meta = load_pre(rname)
            mean, std, size, pool = meta['mean'], meta['std'], meta['size'], pool or 'cls'
        else:
            arch, weights, kw, pool = ENC[name]
            if weights == 'random':
                torch.manual_seed(0)  # reproducible random weights for embedding and evaluation
            model = timm.create_model(arch, pretrained=weights == 'timm', num_classes=0, **kw)
            if weights == 'simclr':
                from huggingface_hub import hf_hub_download
                state = torch.load(hf_hub_download(SIMCLR_REPO, SIMCLR_FILE), map_location='cpu', weights_only=True)['state_dict']
                model.load_state_dict({k: v for k, v in state.items() if not k.startswith('fc.')})
                mean, std = (0., 0., 0.), (1., 1., 1.)  # converted SimCLR v1 expects [0, 1] inputs
            else:
                cfg = timm.data.resolve_data_config({}, model=model)
                mean, std = cfg['mean'], cfg['std']
            # timm's DINOv2 config says 518 even when built at 224.
            size = kw.get('img_size', timm.data.resolve_data_config({}, model=model)['input_size'][-1])
        self.model = model.eval().to(dev).half()
        self.mean = torch.tensor(mean, device=dev).view(1, 3, 1, 1).half()
        self.std = torch.tensor(std, device=dev).view(1, 3, 1, 1).half()
        self.size, self.pool = size, pool

    @torch.no_grad()
    def _forward(self, x):
        if self.pool == 'cls':
            return self.model(x)
        tokens = self.model.forward_features(x)[:, self.model.num_prefix_tokens:]
        side = int(math.isqrt(tokens.shape[1]))
        grid = tokens.transpose(1, 2).reshape(len(x), -1, side, side)
        k = int(self.pool[1:])
        return F.adaptive_avg_pool2d(grid.float(), k).flatten(1)

    @torch.no_grad()
    def __call__(self, frames):
        flat = frames.reshape(-1, *frames.shape[2:])
        outs = []
        for i in range(0, len(flat), self.batch):
            chunk = flat[i:i + self.batch]
            x = (torch.from_numpy(chunk) if isinstance(chunk, np.ndarray) else chunk).to(self.dev, non_blocking=True)
            x = x.permute(0, 3, 1, 2).half() / 255
            if x.shape[-1] != self.size:
                x = F.interpolate(x, size=(self.size, self.size), mode='bilinear', antialias=True)
            outs.append(self._forward((x - self.mean) / self.std).half())
        e = torch.cat(outs)
        return e.reshape(frames.shape[0], frames.shape[1], -1)


# ----------------------------------------------------------------------------- data
def resize_uint8(frames, size):
    """uint8 (..., H, W, C) array/tensor -> uint8 tensor (..., size, size, C) on GPU (antialiased)."""
    x = torch.as_tensor(frames).to(device(), non_blocking=True)
    lead = x.shape[:-3]
    x = x.reshape(-1, *x.shape[-3:]).permute(0, 3, 1, 2).float()
    if x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode='bilinear', antialias=True)
    x = x.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
    return x.reshape(*lead, size, size, 3)


def load_images(split, env=ENV, size=112):
    meta = json.loads((VIS_ROOT / env / f'{split}_img{size}.json').read_text())
    return np.fromfile(VIS_ROOT / env / f'{split}_img{size}.u8', dtype=np.uint8).reshape(meta['shape'])


def shard_files(split, env=ENV, shards=100):
    src = HF_ROOT / env
    train = sorted(src.glob('[0-9][0-9][0-9].npz'))[:shards]
    return train if split == 'train' else [s.with_name(s.stem + '-val.npz') for s in train]


def frame_meta(terminals, interval, n_frames):
    """Map each stored frame to (step, episode, is_final) in OCBench's sparse layout."""
    offsets = np.concatenate([[0], np.flatnonzero(terminals) + 1])
    steps, eps, final = [], [], []
    for e in range(len(offsets) - 1):
        s, t = offsets[e], offsets[e + 1]
        L = t - s
        for k in list(range(0, L, interval)) + [L]:
            steps.append(s + min(k, L - 1)); eps.append(e); final.append(k == L)
    assert len(steps) == n_frames, 'frame layout mismatch'
    return np.asarray(steps), np.asarray(eps), np.asarray(final), offsets


def load_index(split, env=ENV):
    with np.load(VIS_ROOT / env / f'{split}_index.npz', allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def load_normalizer(env=ENV):
    import sys
    sys.path.insert(0, str(ROOT / 'ocbench' / 'impls'))
    from envs.env_utils import ActionNormalizer
    with np.load(VIS_ROOT / env / 'normalizer.npz', allow_pickle=False) as d:
        return ActionNormalizer(d['q_low'], d['q_high'], float(d['clip']), d['gripper_dims'])


def load_embeddings(name, split, keep, env=ENV):
    emb = np.load(EMB_ROOT / env / name / f'{split}_emb.npy', mmap_mode='r')
    if len(emb) != len(keep):
        raise ValueError(f'{name}/{split}: {len(emb)} cached frames but index has {len(keep)}')
    return np.ascontiguousarray(emb[keep])


# ----------------------------------------------------------------------------- policy
def xavier_(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class ResidualMLP(nn.Module):
    """Port of ocbench impls ResidualMLP (layer_norm=True, flax LayerNorm eps 1e-6)."""

    def __init__(self, din, width, depth, dout):
        super().__init__()
        assert depth % 2 == 0
        self.inp, self.ln0 = nn.Linear(din, width), nn.LayerNorm(width, eps=1e-6)
        self.blocks = nn.ModuleList(nn.ModuleList([nn.Linear(width, width), nn.LayerNorm(width, eps=1e-6),
                                                   nn.Linear(width, width), nn.LayerNorm(width, eps=1e-6)])
                                    for _ in range(depth // 2))
        self.out = nn.Linear(width, dout)
        xavier_(self)

    def forward(self, x):
        g = lambda v: F.gelu(v, approximate='tanh')
        x = self.ln0(g(self.inp(x)))
        for a, la, b, lb in self.blocks:
            r = x
            x = b(la(g(a(x))))
            x = lb(g(r + x))
        return self.out(x)


def time_features(t):
    freqs = torch.arange(1, 33, device=t.device, dtype=t.dtype)
    ang = 2 * math.pi * t * freqs
    return torch.cat([t, torch.cos(ang), torch.sin(ang)], -1)


class FlowPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, width=4096, depth=8, flow_steps=10, betas=(1.0, 1.5)):
        super().__init__()
        self.action_dim, self.flow_steps, self.betas = action_dim, flow_steps, betas
        self.mlp = ResidualMLP(obs_dim + action_dim + 65, width, depth, action_dim)

    def velocity(self, obs, x, t):
        return self.mlp(torch.cat([obs, x, time_features(t)], -1))

    def loss(self, obs, actions):
        b = len(actions)
        x0 = torch.randn_like(actions)
        u = torch.rand(b, 1, device=actions.device)
        t = 1 - (1 - u) ** (1 / self.betas[1])  # Beta(1, b) by inverse CDF, as upstream
        xt = (1 - t) * x0 + t * actions
        pred = self.velocity(obs, xt, t)
        return ((pred.float() - (actions - x0)) ** 2).mean()

    @torch.no_grad()
    def sample(self, obs):
        x = torch.randn(len(obs), self.action_dim, device=obs.device)
        for i in range(self.flow_steps):
            t = torch.full((len(obs), 1), i / self.flow_steps, device=obs.device)
            x = x + self.velocity(obs, x, t).float() / self.flow_steps
        return x


# ----------------------------------------------------------------------------- trainable pixel encoders
class ImpalaSmall(nn.Module):
    """Port of upstream impala_small: stacks (16, 32, 32), 1 block each, LayerNorm, MLP(512)."""

    def __init__(self, img=112, stacks=(16, 32, 32), mlp=512):
        super().__init__()
        layers, c = [], 3
        for f in stacks:
            layers.append(nn.ModuleDict(dict(conv=nn.Conv2d(c, f, 3, padding=1),
                                             b1=nn.Conv2d(f, f, 3, padding=1), b2=nn.Conv2d(f, f, 3, padding=1))))
            c = f
        self.stacks = nn.ModuleList(layers)
        side = img
        for _ in stacks:
            side = (side + 1) // 2
        flat = side * side * c
        self.ln = nn.LayerNorm(c, eps=1e-6)  # flax LayerNorm on NHWC normalizes channels only
        self.fc, self.ln2 = nn.Linear(flat, mlp), nn.LayerNorm(mlp, eps=1e-6)
        self.out_dim = mlp
        xavier_(self)

    def forward(self, x):  # x float in [0, 1], NCHW
        for s in self.stacks:
            x = F.max_pool2d(s['conv'](x), 3, 2, 1)
            x = x + s['b2'](F.relu(s['b1'](F.relu(x))))
        x = self.ln(F.relu(x).permute(0, 2, 3, 1)).flatten(1)  # NHWC flatten order as in flax
        return self.ln2(F.gelu(self.fc(x), approximate='tanh'))


class TimmPixel(nn.Module):
    def __init__(self, arch, img):
        super().__init__()
        import timm
        if arch.startswith('pre:'):  # fine-tune a backbone from scripts/pretrain_enc.py
            self.net, meta = load_pre(arch[4:])
            assert meta['size'] == img, (meta['size'], img)
            mean, std = meta['mean'], meta['std']
        else:
            kw = dict(img_size=img) if 'vit' in arch else {}
            self.net = timm.create_model(arch, pretrained=True, num_classes=0, **kw)
            cfg = timm.data.resolve_data_config({}, model=self.net)
            mean, std = cfg['mean'], cfg['std']
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
        self.out_dim = self.net.num_features
        self.norm = nn.LayerNorm(self.out_dim, elementwise_affine=False)

    def forward(self, x):
        return self.norm(self.net((x - self.mean) / self.std))


PIXEL_ENC = {
    'impala_small': lambda img: ImpalaSmall(img),
    'resnet18': lambda img: TimmPixel('resnet18.a1_in1k', img),
    'dinov2_s': lambda img: TimmPixel('vit_small_patch14_dinov2.lvd142m', img),
}


def pixel_encoder(name, img):
    return TimmPixel(name, img) if name.startswith('pre:') else PIXEL_ENC[name](img)


def random_shift(x, pad=4):
    """DrQ-style random shift on NCHW float images."""
    n, _, h, w = x.shape
    x = F.pad(x, (pad,) * 4, mode='replicate')
    eps = 1.0 / (h + 2 * pad)
    arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * pad, device=x.device)[:h]
    base = torch.stack(torch.meshgrid(arange, arange, indexing='xy'), -1)[None].repeat(n, 1, 1, 1)
    shift = torch.randint(0, 2 * pad + 1, (n, 1, 1, 2), device=x.device).float() * 2.0 / (h + 2 * pad)
    return F.grid_sample(x, base + shift, padding_mode='zeros', align_corners=False)


# ----------------------------------------------------------------------------- evaluation
class Appearance:
    """Per-episode appearance shift of the CPU render model (MJWarp renders through env._cpu_env), with the same
    ranges as the 'dr' variant of scripts/dr_gen.py. mode: color (geom/material colours), light (scene lights and
    headlight), camera (pose jitter: 2 cm / 3 deg fixed cameras, 0.5 cm / 1.5 deg wrist), or all."""
    CAMS = ('front', 'side', 'ur5e/wrist')

    def __init__(self, model, mode):
        import mujoco
        assert mode in ('color', 'light', 'camera', 'all'), mode
        self.m, self.mode = model, mode
        self.cam_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, c) for c in self.CAMS]
        m = model
        self.nom = dict(geom_rgba=m.geom_rgba.copy(), mat_rgba=m.mat_rgba.copy(), light_diffuse=m.light_diffuse.copy(),
                        light_ambient=m.light_ambient.copy(), light_pos=m.light_pos.copy(), cam_pos=m.cam_pos.copy(),
                        cam_quat=m.cam_quat.copy(), head_amb=m.vis.headlight.ambient.copy(),
                        head_dif=m.vis.headlight.diffuse.copy())
        self.worlds = []

    def draw(self, n, seed):
        import mujoco
        rng, m, nom = np.random.default_rng(seed), self.m, self.nom
        self.worlds = []
        for _ in range(n):
            w = {}
            if self.mode in ('color', 'all'):
                hue, mix = rng.uniform(0, 1, (m.ngeom, 3)), rng.uniform(0.3, 1.0, (m.ngeom, 1))
                w['geom_rgba'] = nom['geom_rgba'].copy()
                w['geom_rgba'][:, :3] = np.clip((1 - mix) * nom['geom_rgba'][:, :3] + mix * hue, 0, 1)
                w['mat_rgba'] = nom['mat_rgba'].copy()
                if m.nmat:
                    w['mat_rgba'][:, :3] = np.clip(nom['mat_rgba'][:, :3] * rng.uniform(0.4, 1.6, (m.nmat, 3)), 0, 1)
            if self.mode in ('light', 'all'):
                if m.nlight:
                    w['light_diffuse'] = np.clip(nom['light_diffuse'] * rng.uniform(0.3, 1.5, (m.nlight, 1)), 0, 1)
                    w['light_ambient'] = np.clip(nom['light_ambient'] + rng.uniform(0, 0.3, (m.nlight, 1)), 0, 1)
                    w['light_pos'] = nom['light_pos'] + rng.normal(0, 0.5, nom['light_pos'].shape)
                w['head_amb'], w['head_dif'] = rng.uniform(0.0, 0.5, 3), rng.uniform(0.2, 0.9, 3)
            if self.mode in ('camera', 'all'):
                w['cam_pos'], w['cam_quat'] = nom['cam_pos'].copy(), nom['cam_quat'].copy()
                for c in self.cam_ids:
                    wrist = c == self.cam_ids[2]
                    w['cam_pos'][c] = nom['cam_pos'][c] + rng.normal(0, 0.005 if wrist else 0.02, 3)
                    axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
                    ang = rng.normal(0, np.deg2rad(1.5 if wrist else 3))
                    q = np.zeros(4)
                    mujoco.mju_mulQuat(q, np.concatenate([[np.cos(ang / 2)], np.sin(ang / 2) * axis]), nom['cam_quat'][c])
                    w['cam_quat'][c] = q
            self.worlds.append(w)

    def _set(self, vals):
        m = self.m
        for k, v in vals.items():
            if k == 'head_amb':
                m.vis.headlight.ambient[:] = v
            elif k == 'head_dif':
                m.vis.headlight.diffuse[:] = v
            else:
                getattr(m, k)[:] = v

    def apply(self, world):
        self._set(self.worlds[world])

    def restore(self):
        self._set(self.nom)


class VisualEval:
    """Batched MJWarp rollouts with 3-camera 224px images every 25 steps (sparse protocol).
    shift: optional Appearance mode applied to the rendered images (constant within an episode)."""

    def __init__(self, num_episodes, env_name=ENV, shift=None):
        import ocbench
        from ocbench.mjwarp import make_env
        family, backend, kwargs, self.max_steps = ocbench.parse_env_spec(env_name)
        assert backend == 'mjwarp' and kwargs.get('ob_type') == 'pixels'
        kwargs = dict(kwargs, ob_type='states', visualize_info=False)
        self.env = make_env(family, kwargs, num_episodes)
        self.n = num_episodes
        self.shift_mode, self.shift = shift, None  # the CPU render model exists only after reset

    def draw_shift(self, seed):
        if self.shift_mode:
            if self.shift is None:
                self.shift = Appearance(self.env._cpu_env.model, self.shift_mode)
            self.shift.draw(self.n, seed)

    def pixels(self, world_ids=None):
        if self.shift is None:
            return self.env.get_pixel_observation(world_ids)
        ids = range(self.n) if world_ids is None else world_ids
        arrays, out = self.env._get_arrays(), []
        try:
            for w in ids:
                self.shift.apply(int(w))
                self.env.copy_world_to_cpu_env(int(w), arrays=arrays)  # mj_forward picks up the camera pose
                out.append(self.env._cpu_env.get_pixel_observation())
        finally:
            self.shift.restore()
        return np.stack(out)

    def run(self, policy, normalizer, seed):
        """policy: uint8 (n, 3, 224, 224, 3) -> normalized action chunks (n, 25 * action_dim)."""
        import time
        env, n = self.env, self.n
        seeds = np.random.default_rng(seed).integers(0, 2**32, n, dtype=np.uint64).astype(np.uint32)
        env.reset(seeds=seeds)
        self.draw_shift(seed + 1)
        obs = self.pixels()
        adim = env.single_action_space.shape[-1]
        raw_low, raw_high = env.single_action_space.low, env.single_action_space.high
        done = np.zeros(n, bool); success = np.zeros(n, bool)
        steps = np.zeros(n, np.int32); returns = np.zeros(n, np.float64)
        t0 = time.time()
        while not done.all():
            chunk = np.asarray(policy(obs), np.float32).reshape(n, CHUNK, adim)
            chunk = np.clip(chunk, normalizer.action_low, normalizer.action_high)
            chunk = normalizer.unnormalize(chunk.reshape(-1, adim)).reshape(n, CHUNK, adim)
            chunk = np.clip(chunk, raw_low, raw_high)
            for k in range(CHUNK):
                active = ~done
                if not active.any():
                    break
                action = chunk[:, k].copy(); action[done] = 0.0
                _, reward, terminated, _, info = env.step(action)
                returns[active] += np.asarray(reward, np.float64)[active]
                steps[active] += 1
                new_done = active & (terminated | (steps >= self.max_steps))
                success |= new_done & np.asarray(info['success'], bool)
                capture = np.flatnonzero(active & ((steps % CHUNK == 0) | new_done))
                if len(capture):
                    obs = obs.copy()
                    obs[capture] = self.pixels(capture)
                done |= new_done
        return {'success': float(success.mean()), 'mean_steps': float(steps.mean()),
                'mean_success_steps': float(steps[success].mean()) if success.any() else None,
                'return': float(returns.mean()), 'eval_seconds': time.time() - t0, 'episodes': n}


# ----------------------------------------------------------------------------- run bookkeeping
def emit(path, record):
    with open(path, 'a') as f:
        f.write(json.dumps(record) + '\n')
    print(json.dumps(record), flush=True)


def atomic_save(obj, path):
    tmp = Path(str(path) + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def job_info():
    return {k: os.environ.get(k) for k in ('SLURM_JOB_ID', 'SLURM_JOB_PARTITION', 'SLURMD_NODENAME',
                                            'SLURM_RESTART_COUNT')} | {
        'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None}
