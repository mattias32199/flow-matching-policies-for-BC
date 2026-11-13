"""Submit the Mystery-4 vision (visual-block-single) experiment matrix, W&B group m4-vision-single.

Every job is requeue-safe:
  * sbatch --requeue: preemption (PreemptMode=REQUEUE) requeues it automatically;
  * --signal=B:USR1@300: before the time limit the launcher forwards SIGTERM to the trainer, which
    checkpoints and exits, then the launcher runs `scontrol requeue` unless runs/<run>/DONE exists;
  * trainers resume from runs/<run>/ckpt.pt; scripts/wandb_follow.py resumes the same W&B run id.
Training goes to A6000/A5000; render/embedding-only jobs may also use rtx6000.

  python scripts/launch_m4v.py --list
  python scripts/launch_m4v.py --only B,C            # submit groups
  python scripts/launch_m4v.py --only F --dry-run
"""
import argparse
import json
import os
import secrets
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT.parent / '.private-jobs'
TORCH = '.local/venv-torch/bin/python'
JAX = '.venv/bin/python'
TRAIN_GPUS = 'A6000|A5000'
ANY_GPUS = 'A6000|A5000|rtx6000'


def vis(run, group, extra, tags, **kw):
    cmd = [TORCH, '-u', 'scripts/train_vis.py', '--run', run, '--group', 'm4-vision-single',
           '--tags', ','.join([f'group:{group}'] + tags)] + extra
    kw.setdefault('cpus', 16 if 'pixels' in extra else 12)
    return dict(run=run, group=group, cmd=cmd, torch=True, **kw)


def experiments():
    E = []
    # A. state reference + M4 reproduction on block-single, chunk starts every 25 steps (handcrafted = 4088208).
    for feat in ('standardized', 'xyz0.1', 'xyz10', 'fourier4'):
        run = f'v-state-single-{feat}-n10000-int25-s0'
        E.append(dict(run=run, group='A', torch=False, time='14:00:00', mem='64G', cpus=16,
                      cmd=[JAX, '-u', 'scripts/train.py', '--run', run, '--env', 'block-single-task1-v0', '--mode', 'fixed',
                           '--features', feat, '--episodes', '10000', '--data-seed', '0', '--steps', '500000',
                           '--start-interval', '25', '--keep-checkpoints', '10', '--save-interval', '50000',
                           '--no-collection-cache']))
    def emb(enc, scale='std', extra=(), tags=()):
        return (['--input', 'emb', '--encoder', enc, '--scale', scale, *extra],
                [f'encoder:{enc}', 'frozen:true', f'scale:{scale}', 'input:emb', *tags])
    # B. frozen pretrained encoders with different objectives.
    for enc in ('dinov2_s', 'dinov2_b', 'resnet50_ce', 'clip_b16', 'mae_b16', 'simclr_r50', 'random_vit_s'):
        a, t = emb(enc)
        E.append(vis(f'v-B-{enc}-s0', 'B', a, t + ['aux_frac:0'], time='10:00:00', mem='64G'))
    # C. visual Mystery 4: input scaling of DINOv2-B embeddings; CLS vs pooled patch tokens.
    for scale in ('raw', 'x0.1', 'x10'):
        a, t = emb('dinov2_b', scale)
        E.append(vis(f'v-C-dinov2_b-{scale}-s0', 'C', a, t + ['aux_frac:0'], time='10:00:00', mem='64G'))
    for enc in ('dinov2_s_p2', 'dinov2_s_p4'):
        a, t = emb(enc)
        E.append(vis(f'v-C-{enc}-s0', 'C', a, t + ['aux_frac:0', 'pool:' + enc[-2:]], time='12:00:00', mem='96G'))
    # D. frozen vs finetune vs scratch, 112px end-to-end (batch 256).
    pix = ['--input', 'pixels', '--batch', '256', '--steps', '200000']
    E.append(vis('v-D-impala-scratch-s0', 'D', pix + ['--encoder', 'impala_small', '--finetune'],
                 ['encoder:impala_small', 'frozen:false', 'init:scratch', 'input:pixels', 'aux_frac:0'], time='16:00:00', mem='96G'))
    E.append(vis('v-D-resnet18-ft-s0', 'D', pix + ['--encoder', 'resnet18', '--finetune', '--enc-lr', '1e-4'],
                 ['encoder:resnet18', 'frozen:false', 'init:imagenet', 'input:pixels', 'aux_frac:0'], time='16:00:00', mem='96G'))
    E.append(vis('v-D-dinov2_s-ft-s0', 'D', pix + ['--encoder', 'dinov2_s', '--finetune', '--enc-lr', '1e-5'],
                 ['encoder:dinov2_s', 'frozen:false', 'init:dinov2', 'input:pixels', 'aux_frac:0'], time='20:00:00', mem='96G'))
    # E. paired (image, state) data: aux state regression with 1% / 10% / 100% of episodes labelled.
    a, t = emb('dinov2_b', extra=['--adapter'], tags=['adapter:true'])
    E.append(vis('v-E-dinov2_b-adapter-aux0-s0', 'E', a, t + ['aux_frac:0'], time='10:00:00', mem='64G'))
    for frac in ('0.01', '0.1', '1'):
        a, t = emb('dinov2_b', extra=['--adapter', '--aux-frac', frac], tags=['adapter:true'])
        E.append(vis(f'v-E-dinov2_b-adapter-aux{frac}-s0', 'E', a, t + [f'aux_frac:{frac}'], time='10:00:00', mem='64G'))
        E.append(vis(f'v-E-impala-aux{frac}-s0', 'E', pix + ['--encoder', 'impala_small', '--finetune', '--aux-frac', frac],
                     ['encoder:impala_small', 'frozen:false', 'init:scratch', 'input:pixels', f'aux_frac:{frac}'],
                     time='16:00:00', mem='96G'))
    # F. sim (image, state) encoder pretraining with / without visual DR, then frozen BC.
    E.append(dict(run='v-F-drgen', group='F', torch=True, time='06:00:00', mem='96G', cpus=32, gpus=TRAIN_GPUS,
                  cmd=[TORCH, '-u', 'scripts/dr_gen.py', '--n', '200000', '--workers', '30'], wandb=False))
    encs = [('enc-dr-dinov2s', ['--data', 'dr', '--arch', 'dinov2_s'], ['dr:true', 'encoder:dinov2_s', 'init:dinov2']),
            ('enc-nodr-dinov2s', ['--data', 'nodr', '--arch', 'dinov2_s'], ['dr:false', 'encoder:dinov2_s', 'init:dinov2']),
            ('enc-dr-resnet18', ['--data', 'dr', '--arch', 'resnet18', '--scratch'], ['dr:true', 'encoder:resnet18', 'init:scratch'])]
    for name, a, t in encs:
        E.append(dict(run=f'v-F-{name}', group='F', torch=True, time='12:00:00', mem='96G', cpus=16, after=['v-F-drgen'],
                      cmd=[TORCH, '-u', 'scripts/pretrain_enc.py', '--run', f'v-F-{name}', *a,
                           '--tags', ','.join(['group:F', 'stage:encoder'] + t)]))
    E.append(dict(run='v-F-embed', group='F', torch=True, time='06:00:00', mem='96G', cpus=16, gpus=TRAIN_GPUS, wandb=False,
                  after=[f'v-F-{n}' for n, _, _ in encs],
                  cmd=[TORCH, '-u', 'scripts/embed_probe.py', '--env', 'visual-block-single-task1-v0',
                       '--encoders', ','.join(f'pre:v-F-{n}' for n, _, _ in encs),
                       '--fractions', '0.001,0.003,0.01,0.03,0.1,0.3,1']))
    for name, _, t in encs:
        a, tt = emb(f'pre:v-F-{name}')
        E.append(vis(f'v-F-bc-{name}-s0', 'F', a, tt + t + ['aux_frac:0', 'stage:bc'], time='10:00:00', mem='64G',
                     after=['v-F-embed']))
    return E


LAUNCHER = r'''#!/usr/bin/env bash
#SBATCH --signal=B:USR1@300
set -uo pipefail
umask 0077
cd {root}
source scripts/env.sh
export WARP_CACHE_PATH="$REPO_ROOT/.local/cache/warp-job${{SLURM_JOB_ID:-local}}"
{unset}
RUN={run}
mkdir -p runs/$RUN
echo "$(date -Is) start job=$SLURM_JOB_ID node=$SLURMD_NODENAME restart=${{SLURM_RESTART_COUNT:-0}}" >> runs/$RUN/launcher.log
# Some GPUs on this cluster fail CUDA init ("CUDA driver initialization failed"). Detect it before
# training, exclude this node for the job and put it back in the queue instead of failing.
if ! timeout 180 env -u LD_LIBRARY_PATH {torch_py} -c "import torch; torch.zeros(1).cuda(); torch.cuda.synchronize()" >/dev/null 2>&1; then
    trap "" USR1 TERM
    EXC=$(scontrol show job "$SLURM_JOB_ID" | grep -oP 'ExcNodeList=\K\S+')
    [ "$EXC" = "(null)" ] && EXC="$SLURMD_NODENAME" || EXC="$EXC,$SLURMD_NODENAME"
    echo "$(date -Is) bad GPU on $SLURMD_NODENAME (CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-}}); requeue excluding $EXC" >> runs/$RUN/launcher.log
    scontrol requeuehold "$SLURM_JOB_ID"; scontrol update jobid="$SLURM_JOB_ID" ExcNodeList="$EXC"; scontrol release "$SLURM_JOB_ID"
    exit 0
fi
TRAIN=""; FOLLOW=""
finish() {{
    trap "" USR1 TERM
    echo "$(date -Is) signal: stopping trainer for requeue" >> runs/$RUN/launcher.log
    [ -n "$TRAIN" ] && kill -TERM "$TRAIN" 2>/dev/null && wait "$TRAIN"
    [ -n "$FOLLOW" ] && wait "$FOLLOW"
    if [ ! -f runs/$RUN/DONE ]; then scontrol requeue "$SLURM_JOB_ID"; fi
    exit 0
}}
trap finish USR1 TERM
{cmd} &
TRAIN=$!
{follow}
wait "$TRAIN"; rc=$?
[ -n "$FOLLOW" ] && wait "$FOLLOW"
echo "$(date -Is) trainer exit $rc" >> runs/$RUN/launcher.log
if [ ! -f runs/$RUN/DONE ] && [ "$rc" -eq 143 ]; then scontrol requeue "$SLURM_JOB_ID"; exit 0; fi
exit $rc
'''


def submit(e, dep_ids, partition, dry):
    d = JOBS / secrets.token_hex(16)
    follow = '' if e.get('wandb', True) is False else (
        f'env -u LD_LIBRARY_PATH {TORCH} -u scripts/wandb_follow.py --run {e["run"]} --pid $TRAIN '
        f'>> runs/{e["run"]}/wandb_follow.log 2>&1 &\nFOLLOW=$!')
    script = LAUNCHER.format(root=shlex.quote(str(ROOT)), torch_py=TORCH, unset='unset LD_LIBRARY_PATH' if e['torch'] else '',
                             run=e['run'], cmd=shlex.join(e['cmd']), follow=follow)
    args = ['sbatch', '--parsable', '--job-name=' + secrets.token_hex(5), '--partition=' + partition,
            '--constraint=' + e.get('gpus', TRAIN_GPUS), '--gres=gpu:1', f'--cpus-per-task={e.get("cpus", 12)}',
            '--mem=' + e['mem'], '--time=' + e['time'], '--requeue' if e.get('requeue', True) else '--no-requeue',
            '--open-mode=append',
            '--chdir=' + str(d), '--output=' + str(d / 'o'), '--error=' + str(d / 'e')]
    if not e.get('requeue', True):
        script = script.replace('#SBATCH --signal=B:USR1@300\n', '').replace('scontrol requeue', 'true')
    if dep_ids:
        args.append('--dependency=afterok:' + ':'.join(dep_ids))
    if dry:
        print(' '.join(args), '\n', script)
        return 'DRY'
    d.mkdir(parents=True, mode=0o700)
    launcher = d / 'launch.sh'
    launcher.write_text(script)
    launcher.chmod(0o700)
    out = subprocess.run(args + [str(launcher)], cwd=d, capture_output=True, text=True, check=True, timeout=60)
    jid = out.stdout.strip().split(';')[0]
    rec = dict(job_id=jid, run=e['run'], group=e['group'], partition=partition, private_directory=str(d),
               command=e['cmd'], dependency=dep_ids)
    with (ROOT / '.local' / 'm4v-submissions.jsonl').open('a') as f:
        f.write(json.dumps(rec) + '\n')
    return jid


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--only', default='A,B,C,D,E,F')
    p.add_argument('--runs', default='', help='comma list of run names (overrides --only)')
    p.add_argument('--all-partition', default='', help='comma list of run names that may also use partition "all"')
    p.add_argument('--list', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    exps = experiments()
    if args.list:
        for e in exps:
            print(e['group'], e['run'], e['time'], ' '.join(e['cmd'][2:]))
        print(len(exps), 'jobs')
        return
    groups = set(args.only.split(','))
    names = set(filter(None, args.runs.split(',')))
    allp = set(filter(None, args.all_partition.split(',')))
    ids = {}
    for e in exps:
        if not (e['run'] in names if names else e['group'] in groups):
            continue
        if (ROOT / 'runs' / e['run'] / 'DONE').exists():
            print('done already:', e['run'])
            continue
        deps = [ids[a] for a in e.get('after', []) if a in ids and ids[a] != 'DRY']
        part = 'deepaklong,all' if e['run'] in allp else 'deepaklong'
        ids[e['run']] = submit(e, deps, part, args.dry_run)
        print(e['group'], e['run'], ids[e['run']], part, 'after=' + ','.join(deps) if deps else '', flush=True)


if __name__ == '__main__':
    main()
