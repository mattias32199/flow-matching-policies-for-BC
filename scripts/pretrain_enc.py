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
p.add_argument('--groups', default='', help='comma list of target groups to regress (default: all), e.g. cube_xyz,pinch_xyz,gripper,arm_qpos')
p.add_argument('--exclude-groups', default='', help='regress all target groups except these, e.g. cube_yaw')
p.add_argument('--n-pairs', type=int, default=0, help='train on the first N training pairs only (0 = all); the '
                                                       'held-out set (last 5%% of the file) is the same for every N')
p.add_argument('--dr-dir', default='', help='data/dr/<this> instead of data/dr/<env>')
p.add_argument('--select', choices=['last', 'best'], default='last',
               help='best: export the backbone with the highest held-out mean R^2 (evaluated every --eval-interval)')
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
src = ROOT / 'data' / 'dr' / (args.dr_dir or args.env)
meta = json.loads((src / 'meta.json').read_text())
print('loading images', flush=True)
images = np.fromfile(src / f'{args.data}_img112.u8', dtype=np.uint8).reshape(meta['shape'])
Y = np.load(src / 'targets.npy').astype(np.float32)
if args.exclude_groups:
    args.groups = ','.join(g for g in meta['groups'] if g not in args.exclude_groups.split(','))
if args.groups:  # target ablation: regress only the selected groups
    sel = args.groups.split(',')
    cols = sorted(i for g in sel for i in meta['groups'][g])
    Y = Y[:, cols]
    meta['groups'] = {g: [cols.index(i) for i in meta['groups'][g]] for g in sel}
    print('target groups', meta['groups'], flush=True)
n = len(Y)
n_val = n // 20
train_rows, val_rows = np.arange(n - n_val), np.arange(n - n_val, n)
if args.n_pairs:
    assert args.n_pairs <= len(train_rows), (args.n_pairs, len(train_rows))
    train_rows = train_rows[:args.n_pairs]
    args.batch = min(args.batch, args.n_pairs)
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
    var = ((y - y.mean(0)) ** 2).sum(0)
    r2 = 1 - ((pred - y) ** 2).sum(0) / var.clamp_min(1e-12)
    live = (var / len(y) > 1e-4).cpu().numpy()  # skip (near-)constant targets, e.g. fixed hanoi-task1 pegs
    backbone.train(); head.train()
    rec = {f'r2_{g}': float(r2[[i for i in idx if live[i]]].mean()) for g, idx in meta['groups'].items()
           if any(live[i] for i in idx)}
    rec['r2_mean'] = float(np.mean(list(rec.values())))
    return rec


best_path = run / 'best_backbone.pt'
best = json.loads((run / 'best.json').read_text()) if (run / 'best.json').exists() else {'r2_mean': -1e9, 'step': 0}
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
        rec = evaluate()
        emit(run / 'encoder.jsonl', {'step': step, **rec})
        if args.select == 'best' and rec['r2_mean'] > best['r2_mean']:
            best = {'r2_mean': rec['r2_mean'], 'step': step}
            atomic_save(backbone.state_dict(), best_path)
            (run / 'best.json').write_text(json.dumps(best))
    if step % args.ckpt_interval == 0 or step == args.steps or stop['flag']:
        save()
    if stop['flag']:
        print(f'Signal received; checkpointed at step {step}.', flush=True)
        sys.exit(143)

meta_out = dict(arch=arch, kwargs=kw, mean=list(cfg['mean']), std=list(cfg['std']), size=112,
                data=args.data, scratch=args.scratch, target_mean=y_mu.tolist(), target_std=y_sd.tolist(),
                select=args.select, selected_step=best['step'] if args.select == 'best' else step,
                n_pairs=int(len(train_rows)))
state = torch.load(best_path, map_location='cpu', weights_only=True) if args.select == 'best' else backbone.state_dict()
atomic_save({'state_dict': state, 'meta': json.dumps(meta_out)}, run / 'encoder.pt')
best_path.unlink(missing_ok=True)
ckpt_path.unlink(missing_ok=True)  # retention: only encoder.pt is needed downstream
(run / 'DONE').write_text('Encoder pretraining completed.\n')
print('DONE', flush=True)
