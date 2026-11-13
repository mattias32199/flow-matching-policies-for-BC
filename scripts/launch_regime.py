"""Submit the block-double data-regime / vision-cost study (W&B group double-data-regime).

Jobs (same requeue-safe launcher as scripts/launch_m4v.py: --requeue, USR1 before the time limit,
trainer checkpoints and the job requeues itself; GPU self-check excludes broken GPUs):
  P*  prefill the shared collection store (collections 0..99 = 100K expert episodes + val)
  R*  state FBC, 1M updates: fixed N, start interval (= visual frame density), replay refresh
  T*  expert-alone and student->expert takeover diagnostics
  B*  vision cost benchmarks (stream render, re-render, embedding)
  V*  visual (frozen DINOv2-S embedding) BC on the OCBench visual-block-double dataset

  python scripts/launch_regime.py --list
  python scripts/launch_regime.py --only P,R,T,B,V [--dry-run]
"""
import argparse
import json
import secrets
import shlex
import subprocess

from launch_m4v import ANY_GPUS, JAX, JOBS, LAUNCHER, ROOT, TORCH, TRAIN_GPUS

LOG = ROOT / '.local' / 'regime-submissions.jsonl'
VENV = 'visual-block-double-task1-v0'
PREFILL = [('P0', '0-9', True), ('P1', '10-29', False), ('P2', '30-59', False), ('P3', '60-99', False)]


def prefill_deps(n):
    last = n // 1000 - 1
    return [name for name, rng, _ in PREFILL if int(rng.split('-')[0]) <= last]


def state(run, base, interval=1, refresh=0, mem=None, tags=()):
    cmd = [JAX, '-u', 'scripts/train_regime.py', '--run', run, '--base', str(base), '--start-interval', str(interval),
           '--refresh-every', str(refresh), '--steps', '1000000',
           '--tags', ','.join(['input:state', f'base:{base}', f'start_interval:{interval}', f'refresh_every:{refresh}', *tags])]
    mem = mem or ('160G' if base >= 100000 else '96G' if base >= 30000 else '64G')
    return dict(run=run, group='R', cmd=cmd, torch=False, time='24:00:00', mem=mem, cpus=12, after=prefill_deps(base))


def experiments():
    E = []
    for name, rng, val in PREFILL:
        E.append(dict(run=f'regime-{name}', group='P', torch=False, time='06:00:00', mem='64G', cpus=8, wandb=False,
                      cmd=[JAX, '-u', 'scripts/collections_store.py', '--idx', rng] + (['--val'] if val else [])))
    for n in (1000, 3000, 10000, 30000, 100000):
        E.append(state(f'r-fixed-n{n // 1000}k', n))
    for n, k in ((10000, 5), (30000, 5), (30000, 25), (100000, 5), (100000, 25)):
        E.append(state(f'r-fixed-n{n // 1000}k-int{k}', n, interval=k))
    # replay: buffer of `base` episodes, +1K fresh episodes every `refresh` updates (FIFO)
    E.append(state('r-replay-b10k-r10k', 10000, refresh=10000))   # 110K episodes seen
    E.append(state('r-replay-b10k-r2k', 10000, refresh=2000))     # 510K episodes seen
    E.append(state('r-replay-b30k-r5k-int25', 30000, interval=25, refresh=5000))  # 230K, sparse frames
    students = 'm4-handcrafted-s0-cont2,m1-double-n10000-s0,v-state-double-n10000-int25-s0'
    E.append(dict(run='regime-T-expert', group='T', torch=False, time='06:00:00', mem='64G', cpus=8, wandb=False,
                  cmd=['bash', '-c', f'{JAX} -u scripts/takeover.py --students none --episodes 1000 && '
                                     f'OCB_MISTAKE_PROB=0 {JAX} -u scripts/takeover.py --students none --episodes 1000 && '
                                     f'{JAX} -u scripts/takeover.py --students {students} --episodes 500 && touch runs/regime-T-expert/DONE']))
    E.append(dict(run='regime-B-sim', group='B', torch=False, time='04:00:00', mem='96G', cpus=8, wandb=False,
                  cmd=['bash', '-c', f'{JAX} -u scripts/bench_vis_cost.py --part sim && touch runs/regime-B-sim/DONE']))
    E.append(dict(run='regime-B-torch', group='B', torch=True, time='02:00:00', mem='32G', cpus=8, wandb=False, after=['P0'],
                  cmd=['bash', '-c', f'{TORCH} -u scripts/bench_vis_cost.py --part torch && touch runs/regime-B-torch/DONE']))
    E.append(dict(run='regime-V-prep', group='V', torch=True, time='06:00:00', mem='96G', cpus=16, wandb=False, gpus=ANY_GPUS,
                  cmd=['bash', '-c', f'{TORCH} -u scripts/vis_prep.py --env {VENV} --no-images && '
                                     f'{TORCH} -u scripts/embed_probe.py --env {VENV} --encoders dinov2_s_p2 --no-probe && '
                                     'touch runs/regime-V-prep/DONE']))
    for enc in ('dinov2_s', 'dinov2_s_p2'):
        run = f'v-dbl-{enc}-s0'
        E.append(dict(run=run, group='V', torch=True, time='14:00:00', mem='96G', cpus=12, after=['V-prep'],
                      cmd=[TORCH, '-u', 'scripts/train_vis.py', '--run', run, '--env', VENV, '--group', 'double-data-regime',
                           '--input', 'emb', '--encoder', enc, '--scale', 'std', '--steps', '300000',
                           '--tags', f'input:emb,encoder:{enc},frozen:true,base:10000,start_interval:25']))
    # X: do the visual-block-single findings transfer to block-double? (OCBench visual double data, 10K eps, frames every 25)
    def vbc(run, enc, tags, after=()):
        return dict(run=run, group='X', torch=True, time='16:00:00', mem='96G', cpus=12, after=list(after),
                    cmd=[TORCH, '-u', 'scripts/train_vis.py', '--run', run, '--env', VENV, '--group', 'double-data-regime',
                         '--input', 'emb', '--encoder', enc, '--scale', 'std', '--steps', '300000',
                         '--tags', ','.join(['transfer:single->double', f'encoder:{enc}', 'frozen:true', 'base:10000',
                                             'start_interval:25', *tags])])
    for enc in ('dinov2_b', 'clip_b16', 'resnet50_ce', 'random_vit_s'):
        E.append(vbc(f'v-dbl-{enc}-s0', enc, ['finding:encoder-choice']))
    E.append(dict(run='regime-X-embed-p4', group='X', torch=True, time='06:00:00', mem='96G', cpus=16, wandb=False, gpus=ANY_GPUS,
                  cmd=['bash', '-c', f'{TORCH} -u scripts/embed_probe.py --env {VENV} --encoders dinov2_s_p4 && touch runs/regime-X-embed-p4/DONE']))
    E.append(vbc('v-dbl-dinov2_s_p4-s0', 'dinov2_s_p4', ['finding:patch-pooling'], after=['X-embed-p4']))
    E.append(dict(run='regime-X-drgen', group='X', torch=True, time='08:00:00', mem='96G', cpus=32, wandb=False,
                  cmd=['bash', '-c', f'{TORCH} -u scripts/dr_gen.py --env {VENV} --n 200000 --workers 30 --variants nodr '
                                     '&& touch runs/regime-X-drgen/DONE']))
    E.append(dict(run='v-dbl-F-enc-nodr-dinov2s', group='X', torch=True, time='12:00:00', mem='96G', cpus=16, after=['X-drgen'],
                  cmd=[TORCH, '-u', 'scripts/pretrain_enc.py', '--run', 'v-dbl-F-enc-nodr-dinov2s', '--env', VENV,
                       '--data', 'nodr', '--arch', 'dinov2_s', '--group', 'double-data-regime',
                       '--tags', 'transfer:single->double,stage:encoder,dr:false,encoder:dinov2_s,init:dinov2']))
    E.append(dict(run='regime-X-embed-F', group='X', torch=True, time='06:00:00', mem='96G', cpus=16, wandb=False,
                  after=['v-dbl-F-enc-nodr-dinov2s'],
                  cmd=['bash', '-c', f'{TORCH} -u scripts/embed_probe.py --env {VENV} --encoders pre:v-dbl-F-enc-nodr-dinov2s '
                                     '&& touch runs/regime-X-embed-F/DONE']))
    E.append(vbc('v-dbl-F-bc-enc-nodr-dinov2s-s0', 'pre:v-dbl-F-enc-nodr-dinov2s', ['finding:sim-pretraining'],
                 after=['X-embed-F']))
    return E


def submit(e, deps, partition, dry):
    d = JOBS / secrets.token_hex(16)
    follow = '' if e.get('wandb', True) is False else (
        f'env -u LD_LIBRARY_PATH {TORCH} -u scripts/wandb_follow.py --run {e["run"]} --pid $TRAIN '
        f'>> runs/{e["run"]}/wandb_follow.log 2>&1 &\nFOLLOW=$!')
    script = LAUNCHER.format(root=shlex.quote(str(ROOT)), torch_py=TORCH, unset='unset LD_LIBRARY_PATH' if e['torch'] else '',
                             run=e['run'], cmd=shlex.join(e['cmd']), follow=follow)
    args = ['sbatch', '--parsable', '--job-name=' + secrets.token_hex(5), '--partition=' + partition,
            '--constraint=' + e.get('gpus', TRAIN_GPUS), '--gres=gpu:1', f'--cpus-per-task={e["cpus"]}',
            '--mem=' + e['mem'], '--time=' + e['time'], '--requeue', '--open-mode=append',
            '--chdir=' + str(d), '--output=' + str(d / 'o'), '--error=' + str(d / 'e')]
    if deps:
        args.append('--dependency=afterok:' + ':'.join(deps))
    if dry:
        print(' '.join(args), '\n   ', shlex.join(e['cmd']))
        return 'DRY'
    d.mkdir(parents=True, mode=0o700)
    (d / 'launch.sh').write_text(script)
    (d / 'launch.sh').chmod(0o700)
    out = subprocess.run(args + [str(d / 'launch.sh')], cwd=d, capture_output=True, text=True, check=True, timeout=60)
    jid = out.stdout.strip().split(';')[0]
    with LOG.open('a') as f:
        f.write(json.dumps(dict(job_id=jid, run=e['run'], group=e['group'], partition=partition, private_directory=str(d),
                                command=e['cmd'], dependency=deps)) + '\n')
    return jid


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--only', default='P,R,T,B,V')
    p.add_argument('--runs', default='')
    p.add_argument('--partition', default='deepaklong')
    p.add_argument('--list', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    E = experiments()
    if args.list:
        for e in E:
            print(f"{e['group']}  {e['run']:32s} after={e.get('after', [])}  {shlex.join(e['cmd'][2:])[:110]}")
        return
    groups, runs = set(args.only.split(',')), set(filter(None, args.runs.split(',')))
    ids = {}
    if LOG.exists():  # dependencies on jobs from earlier submissions that are still queued/running
        for line in LOG.read_text().splitlines():
            r = json.loads(line)
            if subprocess.run(['squeue', '-h', '-j', r['job_id']], capture_output=True, text=True).stdout.strip():
                ids[r['run'].replace('regime-', '')] = r['job_id']
    for e in E:
        key = e['run'].replace('regime-', '')
        if (runs and e['run'] not in runs) or (not runs and e['group'] not in groups):
            continue
        deps = [ids[a] for a in e.get('after', []) if a in ids]
        jid = submit(e, deps, args.partition, args.dry_run)
        ids[key] = jid
        print(e['group'], e['run'], jid, 'after', deps, flush=True)


if __name__ == '__main__':
    main()
