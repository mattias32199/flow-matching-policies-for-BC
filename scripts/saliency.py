"""Visual analogue of the M4 sensitivity diagnostic: where in the image do policy actions depend?

For each finished train_vis run, rebuild the policy (frozen encoders are re-created in fp32 with
gradients enabled), sample action chunks on held-out val frames with fixed flow noise, and take
|d ||a||^2 / d pixel| summed over colour channels. Reports, per camera:
  cube_ratio  mean saliency on cube pixels / mean saliency over the image (1 = no preference)
  cube_mass   fraction of total saliency on (dilated) cube pixels
The cube is segmented by its saturated blue colour. Writes results/diagnostics/saliency/<run>.json
and a PNG of frames + saliency overlays.

  python scripts/saliency.py --runs v-B-dinov2_b-s0,v-D-resnet18-ft-s0 [--frames 256]
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vis_common import (CHUNK, ENC, PIXEL_ENC, ROOT, FlowPolicy, load_images, load_index)

OUT = ROOT / 'results' / 'diagnostics' / 'saliency'


def mlp(din, dims):
    layers = []
    for d in dims[:-1]:
        layers += [nn.Linear(din, d), nn.GELU(approximate='tanh'), nn.LayerNorm(d, eps=1e-6)]
        din = d
    return nn.Sequential(*layers, nn.Linear(din, dims[-1]))


class DiffEmbedder(nn.Module):
    """fp32, differentiable version of vis_common.Embedder (same preprocessing and pooling)."""

    def __init__(self, name):
        super().__init__()
        import timm
        if name.startswith('pre:'):
            ck = torch.load(ROOT / 'runs' / name[4:] / 'encoder.pt', map_location='cpu', weights_only=True)
            meta = json.loads(ck['meta'])
            self.model = timm.create_model(meta['arch'], pretrained=False, num_classes=0, **meta['kwargs'])
            self.model.load_state_dict(ck['state_dict'])
            mean, std, self.size, self.pool = meta['mean'], meta['std'], meta['size'], 'cls'
        else:
            arch, weights, kw, self.pool = ENC[name]
            if weights == 'random':
                torch.manual_seed(0)
            self.model = timm.create_model(arch, pretrained=weights == 'timm', num_classes=0, **kw)
            if weights == 'simclr':
                from huggingface_hub import hf_hub_download
                from vis_common import SIMCLR_FILE, SIMCLR_REPO
                st = torch.load(hf_hub_download(SIMCLR_REPO, SIMCLR_FILE), map_location='cpu', weights_only=True)['state_dict']
                self.model.load_state_dict({k: v for k, v in st.items() if not k.startswith('fc.')})
                mean, std = (0., 0., 0.), (1., 1., 1.)
            else:
                cfg = timm.data.resolve_data_config({}, model=self.model)
                mean, std = cfg['mean'], cfg['std']
            self.size = kw.get('img_size', timm.data.resolve_data_config({}, model=self.model)['input_size'][-1])
        self.model.eval().requires_grad_(False)
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):  # x (N, 3, H, W) in [0, 1]
        if x.shape[-1] != self.size:
            x = F.interpolate(x, size=(self.size, self.size), mode='bilinear', antialias=True)
        x = (x - self.mean) / self.std
        if self.pool == 'cls':
            return self.model(x)
        tok = self.model.forward_features(x)[:, self.model.num_prefix_tokens:]
        side = int(math.isqrt(tok.shape[1]))
        return F.adaptive_avg_pool2d(tok.transpose(1, 2).reshape(len(x), -1, side, side), int(self.pool[1:])).flatten(1)


def build(run):
    cfg = json.loads((ROOT / 'runs' / run / 'config.json').read_text())
    state = torch.load(ROOT / 'runs' / run / 'ckpt.pt', map_location='cpu', weights_only=False)['model']
    m = nn.Module()
    if cfg['input'] == 'emb':
        m.emb = DiffEmbedder(cfg['encoder'])
        m.register_buffer('mu', state['mu']); m.register_buffer('sd', state['sd'])
        obs = state['mu'].shape[0]
        if cfg['adapter']:
            m.adapter = mlp(obs, [1024, 1024, 256]); obs = 256
    else:
        m.enc = PIXEL_ENC[cfg['encoder']](cfg['img'])
        obs = 3 * m.enc.out_dim
    if cfg['aux_frac'] > 0:
        m.aux = mlp(obs, [512, 21])
    m.policy = FlowPolicy(obs, CHUNK * 7, cfg['width'], cfg['depth'])
    missing, unexpected = m.load_state_dict(state, strict=False)
    assert not unexpected and all(k.startswith('emb.') for k in missing), (missing, unexpected)
    mult = {'std': 1., 'raw': 1., 'x0.1': .1, 'x10': 10.}[cfg['scale']]

    def features(v):  # v (B, 3cams, 3, H, W) float [0, 1]
        b = v.shape[0]
        flat = v.flatten(0, 1)
        if cfg['input'] == 'emb':
            f = m.emb(flat).reshape(b, -1)
            if cfg['scale'] != 'raw':
                f = (f - m.mu) / m.sd * mult
            return m.adapter(f) if cfg['adapter'] else f
        return m.enc(flat).reshape(b, -1)
    return m.cuda().eval(), features, cfg


def cube_mask(img):  # uint8 (..., H, W, 3) -> bool, saturated blue cube pixels (dilated by 2 px)
    x = img.astype(np.int16)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mask = torch.from_numpy(((b - r > 60) & (b - g > 30) & (b > 120)).astype(np.float32))
    shp = mask.shape
    mask = F.max_pool2d(mask.reshape(-1, 1, *shp[-2:]), 5, 1, 2).reshape(shp)
    return mask.numpy() > 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', required=True)
    p.add_argument('--frames', type=int, default=256)
    p.add_argument('--batch', type=int, default=16)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    images = load_images('val')
    rows = np.sort(np.random.default_rng(0).choice(len(images), args.frames, replace=False))
    frames = np.ascontiguousarray(images[rows])  # (N, 3, 112, 112, 3)
    mask = cube_mask(frames)
    for run in args.runs.split(','):
        if not (ROOT / 'runs' / run / 'ckpt.pt').exists():
            print('skip (no checkpoint)', run); continue
        m, features, cfg = build(run)
        g = torch.Generator(device='cuda').manual_seed(0)
        sal = []
        for i in range(0, len(frames), args.batch):
            v = torch.from_numpy(frames[i:i + args.batch]).cuda().permute(0, 1, 4, 2, 3).float() / 255
            v.requires_grad_(True)
            f = features(v)
            x = torch.randn(len(v), m.policy.action_dim, device='cuda', generator=g)
            for k in range(m.policy.flow_steps):
                t = torch.full((len(v), 1), k / m.policy.flow_steps, device='cuda')
                x = x + m.policy.velocity(f, x, t) / m.policy.flow_steps
            (x ** 2).sum().backward()
            sal.append(v.grad.abs().sum(2).cpu().numpy())  # (B, 3cams, H, W)
        s = np.concatenate(sal)
        res = {'run': run, 'encoder': cfg['encoder'], 'input': cfg['input'], 'frames': len(s), 'cameras': {}}
        for c, cam in enumerate(('front', 'side', 'wrist')):
            sc, mc = s[:, c], mask[:, c]
            has = mc.reshape(len(mc), -1).any(1)
            ratio = [sc[i][mc[i]].mean() / sc[i].mean() for i in np.flatnonzero(has)]
            mass = [sc[i][mc[i]].sum() / sc[i].sum() for i in np.flatnonzero(has)]
            res['cameras'][cam] = dict(cube_ratio=float(np.mean(ratio)) if ratio else None,
                                       cube_mass=float(np.mean(mass)) if mass else None,
                                       cube_area=float(mc[has].mean()) if has.any() else None,
                                       frames_with_cube=int(has.sum()),
                                       share_of_total=float(sc.sum() / s.sum()))
        (OUT / f'{run}.json').write_text(json.dumps(res, indent=1))
        print(json.dumps(res), flush=True)
        try:
            from PIL import Image
            tiles = []
            for i in range(4):
                top = np.concatenate(list(frames[i]), 1)
                h = s[i] / (np.percentile(s[i], 99.5, axis=(1, 2), keepdims=True) + 1e-12)
                heat = (np.clip(np.concatenate(list(h), 1), 0, 1) * 255).astype(np.uint8)
                over = (0.4 * top + 0.6 * np.stack([heat, heat * 0, heat * 0], -1)).astype(np.uint8)
                tiles.append(np.concatenate([top, over], 0))
            Image.fromarray(np.concatenate(tiles, 0)).save(OUT / f'{run}.png')
        except Exception as e:  # figure is best-effort
            print('figure failed', e)
        del m
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
