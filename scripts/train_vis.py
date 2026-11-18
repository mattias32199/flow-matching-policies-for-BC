"""Visual flow-BC on visual-block-single (torch venv). Resumable: safe under Slurm preemption/requeue.

Inputs
  --input emb     cached frozen embeddings (data/emb/<env>/<encoder>); eval embeds live images with
                  the same frozen encoder. --scale picks the M4-style input scaling; --adapter adds a
                  trainable 3-layer bottleneck (256-d) between embeddings and the policy.
  --input pixels  112px uint8 images (data/vis/<env>/*_img112.u8) through a pixel encoder
                  (impala_small from scratch, or timm resnet18 / dinov2_s, frozen or --finetune).
Auxiliary state supervision: --aux-frac F gives qpos labels to a fixed random F of the training
episodes; an MLP head on the policy's observation features regresses standardized qpos.
Logs: runs/<run>/{config.json,metrics.jsonl,eval.jsonl}; scripts/wandb_follow.py mirrors to W&B.
"""
import argparse
import json
import signal
import sys
import threading
import queue
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vis_common import (CHUNK, ENV, PIXEL_ENC, ROOT, Embedder, FlowPolicy, VisualEval, amp_dtype,
                        atomic_save, emit, job_info, load_embeddings, load_images, load_index,
                        load_normalizer, random_shift, resize_uint8)

p = argparse.ArgumentParser()
p.add_argument('--run', required=True)
p.add_argument('--input', choices=['emb', 'pixels'], required=True)
p.add_argument('--encoder', required=True)
p.add_argument('--scale', choices=['std', 'raw', 'x0.1', 'x10'], default='std')
p.add_argument('--adapter', action='store_true')
p.add_argument('--finetune', action='store_true')
p.add_argument('--img', type=int, default=112)
p.add_argument('--aux-frac', type=float, default=0.0)
p.add_argument('--aux-weight', type=float, default=1.0)
p.add_argument('--steps', type=int, default=300000)
p.add_argument('--batch', type=int, default=1024)
p.add_argument('--lr', type=float, default=1e-4)
p.add_argument('--enc-lr', type=float, default=1e-5)
p.add_argument('--width', type=int, default=4096)
p.add_argument('--depth', type=int, default=8)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--log-interval', type=int, default=1000)
p.add_argument('--eval-interval', type=int, default=50000)
p.add_argument('--eval-episodes', type=int, default=300)
p.add_argument('--ckpt-interval', type=int, default=5000)
p.add_argument('--keep-every', type=int, default=50000, help='also keep a copy of params at these steps')
p.add_argument('--group', default='m4v')
p.add_argument('--tags', default='')
p.add_argument('--env', default=ENV)
p.add_argument('--episodes', type=int, default=0, help='train on K demo episodes only (fixed random subset, seed 0; '
                                                       'nested across K); 0 = all')
p.add_argument('--final-eval-episodes', type=int, default=0, help='extra large evaluation after the last step')
p.add_argument('--drop-ckpt', action='store_true', help='delete ckpt.pt when the run is DONE (retention)')
args = p.parse_args()

run = ROOT / 'runs' / args.run
run.mkdir(parents=True, exist_ok=True)
if (run / 'DONE').exists():
    print('Run already complete.', flush=True)
    sys.exit(0)
ckpt_path = run / 'ckpt.pt'
config = vars(args) | {'wandb_group': args.group, 'tags': [t for t in args.tags.split(',') if t],
                       'framework': 'torch', 'dataset': args.env}
if (run / 'config.json').exists():
    old = json.loads((run / 'config.json').read_text())
    if {k: old.get(k) for k in config} != config:
        raise RuntimeError('Existing run directory has a different config; use a new --run name.')
(run / 'config.json').write_text(json.dumps(config, indent=2))
emit(run / 'jobs.jsonl', job_info() | {'time': time.time(), 'resume': ckpt_path.exists()})

stop = {'flag': False}
signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
signal.signal(signal.SIGUSR1, lambda *_: stop.update(flag=True))

torch.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
dev = torch.device('cuda')
rng = np.random.default_rng(args.seed)
normalizer = load_normalizer(args.env)

# ----------------------------------------------------------------------------- data
idx = {s: load_index(s, args.env) for s in ('train', 'val')}
sub = None
if args.episodes:
    episodes = np.random.default_rng(0).permutation(np.unique(idx['train']['uid']))
    assert args.episodes <= len(episodes), (args.episodes, len(episodes))
    sub = np.flatnonzero(np.isin(idx['train']['uid'], episodes[:args.episodes]))
    idx['train'] = {k: (v[sub] if k != 'keep' else v) for k, v in idx['train'].items()}
n_train, n_val = len(idx['train']['chunks']), len(idx['val']['chunks'])
chunks = {s: torch.from_numpy(idx[s]['chunks'].reshape(len(idx[s]['chunks']), -1)).to(dev) for s in idx}
qpos_mu = idx['train']['qpos'].mean(0)
qpos_sd = idx['train']['qpos'].std(0).clip(1e-3)
qpos = {s: torch.from_numpy((idx[s]['qpos'] - qpos_mu) / qpos_sd).float().to(dev) for s in idx}
labelled = None
if args.aux_frac > 0:
    uid = idx['train']['uid']
    episodes = np.random.default_rng(0).permutation(np.unique(uid))
    chosen = episodes[:max(1, int(round(args.aux_frac * len(episodes))))] if args.aux_frac < 1 else episodes
    labelled = np.flatnonzero(np.isin(uid, chosen))
    emit(run / 'dataset.jsonl', {'label_fraction': args.aux_frac, 'label_episodes': int(len(chosen)),
                                 'label_frames': int(len(labelled))})

if args.input == 'emb':
    X = {s: torch.from_numpy(load_embeddings(args.encoder, s, idx[s]['keep'], args.env)).to(dev) for s in idx}
    if sub is not None:
        X['train'] = X['train'][torch.from_numpy(sub).to(dev)]
    emb_dim = X['train'].shape[1] * X['train'].shape[2]
    flat = X['train'].reshape(n_train, -1)
    mu = torch.zeros(emb_dim, device=dev); sq = torch.zeros(emb_dim, device=dev)
    for i in range(0, n_train, 65536):
        x = flat[i:i + 65536].float(); mu += x.sum(0); sq += (x * x).sum(0)
    mu /= n_train
    sd = (sq / n_train - mu ** 2).clamp_min(1e-12).sqrt().clamp_min(1e-6)
    del flat
else:
    images = {s: load_images(s, args.env, args.img) for s in idx}
    assert sub is None, '--episodes is only implemented for --input emb'
    assert len(images['train']) == n_train and len(images['val']) == n_val

emit(run / 'dataset.jsonl', {'train_frames': n_train, 'val_frames': n_val,
                             'train_episodes': int(len(np.unique(idx['train']['uid'])))})


# ----------------------------------------------------------------------------- model
def mlp(din, dims):
    layers = []
    for d in dims[:-1]:
        layers += [nn.Linear(din, d), nn.GELU(approximate='tanh'), nn.LayerNorm(d, eps=1e-6)]
        din = d
    return nn.Sequential(*layers, nn.Linear(din, dims[-1]))


class VisualPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        if args.input == 'emb':
            self.register_buffer('mu', mu)
            self.register_buffer('sd', sd)
            self.mult = {'std': 1., 'raw': 1., 'x0.1': .1, 'x10': 10.}[args.scale]
            obs_dim = emb_dim
            if args.adapter:
                self.adapter = mlp(emb_dim, [1024, 1024, 256])
                obs_dim = 256
        else:
            self.enc = PIXEL_ENC[args.encoder](args.img)
            if not args.finetune:
                self.enc.requires_grad_(False)
            obs_dim = 3 * self.enc.out_dim
        self.aux = mlp(obs_dim, [512, 21]) if args.aux_frac > 0 else None
        self.policy = FlowPolicy(obs_dim, CHUNK * 7, args.width, args.depth)

    def features(self, x, train):
        if args.input == 'emb':
            f = x.float().flatten(1)
            if args.scale != 'raw':
                f = (f - self.mu) / self.sd * self.mult
            return self.adapter(f) if args.adapter else f
        b = x.shape[0]
        v = x.reshape(-1, *x.shape[2:]).permute(0, 3, 1, 2).float() / 255
        if train:
            v = random_shift(v)
        if not args.finetune:
            self.enc.eval()
        return self.enc(v).reshape(b, -1).float()


model = VisualPolicy().to(dev)
if args.input == 'pixels':
    model = model.to(memory_format=torch.channels_last)
enc_params = [p_ for n_, p_ in model.named_parameters() if n_.startswith('enc.') and p_.requires_grad]
other = [p_ for n_, p_ in model.named_parameters() if not n_.startswith('enc.') and p_.requires_grad]
pretrained_enc = args.input == 'pixels' and args.encoder != 'impala_small'
groups = [{'params': other, 'lr': args.lr}]
if enc_params:
    groups.append({'params': enc_params, 'lr': args.enc_lr if pretrained_enc else args.lr})
opt = torch.optim.Adam(groups, eps=1e-8)
use_amp = args.input == 'pixels'
dtype = amp_dtype()
scaler = torch.amp.GradScaler(enabled=use_amp and dtype == torch.float16)
emit(run / 'dataset.jsonl', {'trainable_parameters': int(sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)),
                             'amp': str(dtype) if use_amp else 'tf32'})

step = 0
if ckpt_path.exists():
    state = torch.load(ckpt_path, map_location=dev, weights_only=False)
    model.load_state_dict(state['model']); opt.load_state_dict(state['opt']); scaler.load_state_dict(state['scaler'])
    step = state['step']
    rng.bit_generator.state = state['np_rng']
    torch.set_rng_state(state['torch_rng'].cpu()); torch.cuda.set_rng_state(state['cuda_rng'].cpu())
    print(f'RESUMED from step {step}', flush=True)


def save(step):
    atomic_save({'model': model.state_dict(), 'opt': opt.state_dict(), 'scaler': scaler.state_dict(), 'step': step,
                 'np_rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state(),
                 'cuda_rng': torch.cuda.get_rng_state()}, ckpt_path)


# ----------------------------------------------------------------------------- batches
class PixelLoader:
    """Background thread gathering random uint8 image batches into pinned memory."""

    def __init__(self, split, size, rows=None):
        self.split, self.size, self.rows = split, size, rows
        self.q = queue.Queue(maxsize=6)
        self.rng = np.random.default_rng(int(rng.integers(2**31)))
        threading.Thread(target=self.work, daemon=True).start()

    def work(self):
        n = len(images[self.split])
        while True:
            ids = self.rng.integers(n, size=self.size) if self.rows is None else self.rows[self.rng.integers(len(self.rows), size=self.size)]
            ids.sort()
            batch = torch.from_numpy(images[self.split][ids]).pin_memory()
            self.q.put((torch.from_numpy(ids), batch))

    def get(self):
        ids, batch = self.q.get()
        return ids.to(dev), batch.to(dev, non_blocking=True)


def emb_batch(split, size, rows=None):
    n = n_train if split == 'train' else n_val
    ids = rng.integers(n, size=size) if rows is None else rows[rng.integers(len(rows), size=size)]
    ids = torch.from_numpy(ids).to(dev)
    return ids, X[split][ids]


if args.input == 'pixels':
    loaders = {'train': PixelLoader('train', args.batch), 'val': PixelLoader('val', min(args.batch, 512))}
    if labelled is not None:
        loaders['aux'] = PixelLoader('train', max(32, args.batch // 4), labelled)
    get = lambda which: loaders[which].get()
else:
    get = lambda which: {'train': lambda: emb_batch('train', args.batch), 'val': lambda: emb_batch('val', 1024),
                         'aux': lambda: emb_batch('train', 256, labelled)}[which]()


def r2(pred, y):
    return (1 - ((pred - y) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0).clamp_min(1e-12))


# ----------------------------------------------------------------------------- eval
evaluator = {'env': None, 'embedder': None}


@torch.no_grad()
def policy_fn(obs):
    model.eval()
    if args.input == 'emb':
        if evaluator['embedder'] is None:
            evaluator['embedder'] = Embedder(args.encoder, dev)
        x = evaluator['embedder'](obs)
    else:
        x = resize_uint8(obs, args.img)
    out = []
    for i in range(0, len(x), 512):
        with torch.autocast('cuda', dtype=dtype, enabled=use_amp):
            f = model.features(x[i:i + 512], train=False)
        out.append(model.policy.sample(f.float()).cpu().numpy())
    model.train()
    return np.concatenate(out)


def evaluate(step):
    if evaluator['env'] is None:
        evaluator['env'] = VisualEval(args.eval_episodes, args.env)
    metrics = evaluator['env'].run(policy_fn, normalizer, seed=7_000_000 + step)
    emit(run / 'eval.jsonl', {'step': step, **metrics})


# ----------------------------------------------------------------------------- train
model.train()
t0, last_log, last_step = time.time(), time.time(), step
while step < args.steps:
    step += 1
    ids, x = get('train')
    with torch.autocast('cuda', dtype=dtype, enabled=use_amp):
        f = model.features(x, train=True)
        loss = model.policy.loss(f, chunks['train'][ids])
        aux_loss = torch.zeros((), device=dev)
        if labelled is not None:
            aids, ax = get('aux')
            aux_loss = F.mse_loss(model.aux(model.features(ax, train=True)).float(), qpos['train'][aids])
        total = loss + args.aux_weight * aux_loss
    opt.zero_grad(set_to_none=True)
    scaler.scale(total).backward()
    scaler.step(opt)
    scaler.update()
    if step % args.log_interval == 0 or step == 1:
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at step {step}')
        rec = {'step': step, 'seconds': time.time() - t0, 'training/flow_loss': float(loss),
               'steps_per_s': (step - last_step) / max(time.time() - last_log, 1e-9)}
        last_log, last_step = time.time(), step
        model.eval()
        with torch.no_grad(), torch.autocast('cuda', dtype=dtype, enabled=use_amp):
            vids, vx = get('val')
            vf = model.features(vx, train=False)
            rec['validation/flow_loss'] = float(model.policy.loss(vf, chunks['val'][vids]))
            rec['validation/action_mse'] = float(((model.policy.sample(vf.float()) - chunks['val'][vids]) ** 2).mean())
            if labelled is not None:
                rec['training/aux_loss'] = float(aux_loss)
                score = r2(model.aux(vf).float(), qpos['val'][vids])
                rec['validation/aux_r2_arm'] = float(score[:6].mean())
                rec['validation/aux_r2_gripper'] = float(score[6])
                rec['validation/aux_r2_cube_xyz'] = float(score[14:17].mean())
        model.train()
        emit(run / 'metrics.jsonl', rec)
    if step % args.eval_interval == 0 or step == args.steps:
        evaluate(step)
    if step % args.ckpt_interval == 0 or step == args.steps or stop['flag']:
        save(step)
        if step % args.keep_every == 0:
            torch.save({k: v for k, v in model.state_dict().items() if not k.startswith('enc.') or args.finetune},
                       run / f'params_{step}.pt')
    if stop['flag']:
        print(f'Signal received; checkpointed at step {step}, exiting for requeue.', flush=True)
        sys.exit(143)

if args.final_eval_episodes and not (run / 'final_eval.jsonl').exists():
    metrics = VisualEval(args.final_eval_episodes, args.env).run(policy_fn, normalizer, seed=9_000_000)
    emit(run / 'final_eval.jsonl', {'step': step, **metrics})
    torch.save({k: v for k, v in model.state_dict().items() if not k.startswith('enc.') or args.finetune},
               run / f'params_{step}.pt')
(run / 'DONE').write_text('Training and final evaluation completed.\n')
if args.drop_ckpt:
    ckpt_path.unlink(missing_ok=True)
print('DONE', flush=True)
