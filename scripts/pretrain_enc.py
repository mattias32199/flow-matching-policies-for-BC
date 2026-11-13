"""Group F: pretrain a robot visual encoder by regressing sim state targets (scripts/dr_gen.py data).

A shared timm backbone embeds each of the 3 cameras; an MLP head on the concatenated features
regresses the standardized 27-d targets (cube in camera / gripper frames, yaw, pinch, gripper, arm,
cube xyz). The last 5% of samples are held out; per-group R^2 goes to runs/<run>/encoder.jsonl.
Resumable (ckpt.pt every --ckpt-interval steps; SIGTERM/SIGUSR1 checkpoint and exit 143).
Writes runs/<run>/encoder.pt in the format vis_common.Embedder('pre:<run>') loads.

  python scripts/pretrain_enc.py --run enc-dr-dinov2s --data dr --arch dinov2_s
"""
import argparse
import json
import queue
import signal
import sys
import threading
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vis_common import ROOT, amp_dtype, atomic_save, emit, job_info, random_shift

ARCH = {'dinov2_s': ('vit_small_patch14_dinov2.lvd142m', dict(img_size=112)),
        'resnet18': ('resnet18.a1_in1k', {})}

p = argparse.ArgumentParser()
p.add_argument('--run', required=True)
p.add_argument('--data', choices=['dr', 'nodr'], required=True)
p.add_argument('--arch', choices=list(ARCH), required=True)
p.add_argument('--scratch', action='store_true', help='random init instead of pretrained weights')
p.add_argument('--env', default='visual-block-single-task1-v0')
p.add_argument('--steps', type=int, default=40000)
p.add_argument('--batch', type=int, default=128)
p.add_argument('--lr', type=float, default=3e-4)
p.add_argument('--enc-lr', type=float, default=5e-5)
p.add_argument('--log-interval', type=int, default=250)
p.add_argument('--eval-interval', type=int, default=2000)
p.add_argument('--ckpt-interval', type=int, default=2000)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--group', default='m4-vision-single')
p.add_argument('--tags', default='')
args = p.parse_args()

run = ROOT / 'runs' / args.run
run.mkdir(parents=True, exist_ok=True)
if (run / 'DONE').exists():
    print('Run already complete.', flush=True)
    sys.exit(0)
config = vars(args) | {'wandb_group': args.group, 'tags': [t for t in args.tags.split(',') if t], 'kind': 'encoder_pretrain'}
(run / 'config.json').write_text(json.dumps(config, indent=2))
ckpt_path = run / 'ckpt.pt'
emit(run / 'jobs.jsonl', job_info() | {'time': time.time(), 'resume': ckpt_path.exists()})
stop = {'flag': False}
signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
signal.signal(signal.SIGUSR1, lambda *_: stop.update(flag=True))

torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
dev = torch.device('cuda')
src = ROOT / 'data' / 'dr' / args.env
meta = json.loads((src / 'meta.json').read_text())
print('loading images', flush=True)
images = np.fromfile(src / f'{args.data}_img112.u8', dtype=np.uint8).reshape(meta['shape'])
Y = np.load(src / 'targets.npy').astype(np.float32)
n = len(Y)
n_val = n // 20
train_rows, val_rows = np.arange(n - n_val), np.arange(n - n_val, n)
y_mu, y_sd = Y[train_rows].mean(0), Y[train_rows].std(0).clip(1e-4)
Yn = torch.from_numpy((Y - y_mu) / y_sd).to(dev)

import timm  # noqa: E402
arch, kw = ARCH[args.arch]
backbone = timm.create_model(arch, pretrained=not args.scratch, num_classes=0, **kw)
cfg = timm.data.resolve_data_config({}, model=backbone)
mean = torch.tensor(cfg['mean'], device=dev).view(1, 3, 1, 1)
std = torch.tensor(cfg['std'], device=dev).view(1, 3, 1, 1)
D = backbone.num_features
head = nn.Sequential(nn.Linear(3 * D, 1024), nn.GELU(), nn.LayerNorm(1024), nn.Linear(1024, 1024), nn.GELU(),
                     nn.LayerNorm(1024), nn.Linear(1024, Y.shape[1]))
backbone, head = backbone.to(dev), head.to(dev)
enc_lr = args.lr if args.scratch else args.enc_lr
opt = torch.optim.AdamW([{'params': backbone.parameters(), 'lr': enc_lr}, {'params': head.parameters(), 'lr': args.lr}],
                        weight_decay=1e-4)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 1000) * 0.5 * (1 + np.cos(np.pi * min(s, args.steps) / args.steps)))
dtype = amp_dtype()
scaler = torch.amp.GradScaler(enabled=dtype == torch.float16)
rng = np.random.default_rng(args.seed)
step = 0
if ckpt_path.exists():
    s = torch.load(ckpt_path, map_location=dev, weights_only=False)
    backbone.load_state_dict(s['backbone']); head.load_state_dict(s['head']); opt.load_state_dict(s['opt'])
    sched.load_state_dict(s['sched']); scaler.load_state_dict(s['scaler']); step = s['step']
    rng.bit_generator.state = s['np_rng']
    print(f'RESUMED from step {step}', flush=True)


def save():
    atomic_save({'backbone': backbone.state_dict(), 'head': head.state_dict(), 'opt': opt.state_dict(),
                 'sched': sched.state_dict(), 'scaler': scaler.state_dict(), 'step': step,
                 'np_rng': rng.bit_generator.state}, ckpt_path)


q = queue.Queue(maxsize=8)


def loader():
    r = np.random.default_rng(int(rng.integers(2**31)))
    while True:
        ids = np.sort(r.choice(train_rows, args.batch, replace=False))
        q.put((torch.from_numpy(ids), torch.from_numpy(images[ids]).pin_memory()))


for _ in range(3):
    threading.Thread(target=loader, daemon=True).start()


def embed(x, train):
    b = x.shape[0]
    v = x.reshape(-1, *x.shape[2:]).permute(0, 3, 1, 2).float() / 255
    if train:
        v = random_shift(v)
    return backbone((v - mean) / std).reshape(b, -1)


@torch.no_grad()
def evaluate():
    backbone.eval(); head.eval()
    preds = []
    for i in range(0, len(val_rows), 256):
        ids = val_rows[i:i + 256]
        with torch.autocast('cuda', dtype=dtype):
            preds.append(head(embed(torch.from_numpy(images[ids]).to(dev), False)).float())
    pred = torch.cat(preds); y = Yn[val_rows[0]:]
    r2 = 1 - ((pred - y) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0).clamp_min(1e-12)
    backbone.train(); head.train()
    rec = {f'r2_{g}': float(r2[idx].mean()) for g, idx in meta['groups'].items()}
    rec['r2_mean'] = float(np.mean(list(rec.values())))
    return rec


backbone.train(); head.train()
t0, last = time.time(), step
while step < args.steps:
    step += 1
    ids, x = q.get()
    x = x.to(dev, non_blocking=True)
    with torch.autocast('cuda', dtype=dtype):
        loss = F.mse_loss(head(embed(x, True)).float(), Yn[ids.to(dev)])
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(head.parameters()), 1.0)
    scaler.step(opt); scaler.update(); sched.step()
    if step % args.log_interval == 0 or step == 1:
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at step {step}')
        emit(run / 'metrics.jsonl', {'step': step, 'training/target_mse': float(loss), 'seconds': time.time() - t0})
    if step % args.eval_interval == 0 or step == args.steps:
        emit(run / 'encoder.jsonl', {'step': step, **evaluate()})
    if step % args.ckpt_interval == 0 or step == args.steps or stop['flag']:
        save()
    if stop['flag']:
        print(f'Signal received; checkpointed at step {step}.', flush=True)
        sys.exit(143)

meta_out = dict(arch=arch, kwargs=kw, mean=list(cfg['mean']), std=list(cfg['std']), size=112,
                data=args.data, scratch=args.scratch, target_mean=y_mu.tolist(), target_std=y_sd.tolist())
atomic_save({'state_dict': backbone.state_dict(), 'meta': json.dumps(meta_out)}, run / 'encoder.pt')
(run / 'DONE').write_text('Encoder pretraining completed.\n')
print('DONE', flush=True)
