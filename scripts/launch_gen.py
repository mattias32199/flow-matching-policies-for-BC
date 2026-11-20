"""Submit the follow-up study (W&B group vision-followup): why the visual encoder exposes relative yaw, whether
sim-pretrained visual BC beats state BC on other OCBench manipulation tasks, and a data scaling grid on block-single.

Groups
  Q  yaw probes on cached block-single embeddings; encoder trained on all targets except cube_yaw (+ its BC)
  G  generalization to hanoi-single / hanoi-double / bowling: state collections, state BC at 10K/30K episodes with
     chunk starts every 25 steps; visual demos (OCBench in-loop rendering), frozen DINOv2-S 4x4 and sim-pretrained
     DINOv2-S visual BC (hanoi-double: state only until a gap is seen)
  S  block-single scaling: encoder pretraining pairs P x demo episodes E, frozen-CLS visual BC to 300K updates
  S  low-pairs extension: P = 20 / 60 / 200 / 600 (regenerates the 200K-pair file first, then the same grid)
  X1 pairs from states NOT taken from the demos (dr_gen --states uniform), P = 200 / 2K / 20K, E = 1K / all
  X2 hanoi-single pairs scaling, P = 200 / 2K / 20K, E = 1K / all (+ P = 190K and generic 4x4 at E = 1K)
  X3 appearance-shift evaluation (colour / light / camera / all) of nodr- vs dr-pretrained and generic 4x4 encoders
  X4 hanoi state BC + sin/cos(4 (disk yaw - effector yaw)) (train.py 'diskyaw'), 10K / 30K demos
  X5 fine-tuning the low-pair encoders end to end (P = 20 / 200, pixels)
  X6 sim-pretrained encoder with a 4x4 patch readout (pre:<run>@p4): block-single low E, block-double all demos
  X7 replicate of the low-P rows with a fresh pair sample and seeds (dr_gen --seed 1)
Same requeue-safe launcher and submission log as scripts/launch_regime.py; partitions deepaklong,all.

  python scripts/launch_gen.py --list
  python scripts/launch_gen.py --only Q,G,S,SL [--dry-run]
"""
import argparse
import json
import os
import shlex
import subprocess

from launch_m4v import ANY_GPUS, JAX, ROOT, TORCH
from launch_regime import LOG, submit

GROUP = 'vision-followup'
SENV = 'visual-block-single-task1-v0'
# Measured 2026-11-17: frozen-embedding BC peaks at ~6 GB GPU memory (2080ti: 9.9 updates/s vs A5000 26, rtx6000 11);
# JAX state BC peaks at ~6 GB (3080ti 25 updates/s). grogu-3-20 / grogu-3-30 GPUs fail CUDA init; grogu-1-20's broken
# GPU is held by .local/gpu-hold/hold.sh.
CHEAP_OK = 'A6000|A5000|rtx6000|rtx3090|A6000Ada|rtx2080ti|3080ti|H200|6000Blackwell'  # torch 2.15 has sm_90/sm_120
os.environ.setdefault('SLURM_EXCLUDE', 'grogu-3-20,grogu-3-30,grogu-1-10,grogu-3-25,grogu-1-20')  # bad GPUs seen 2026-11-17/18
FAST = 'A6000|A5000|A6000Ada|rtx3090|3080ti|H200|6000Blackwell'  # jobs > ~2 h: avoid Turing (2080 Ti / rtx6000)
BIG = 'A6000|A6000Ada|H200|6000Blackwell'                       # >= 48 GB GPU memory (4x4 patch caches on GPU)
DENV = 'visual-block-double-task1-v0'
HENV = 'visual-hanoi-single-task1-v0'
TASKS = {'hanoi-single-task1-v0': dict(tag='hs', steps=500000, visual=True),
         'bowling-task1-v0': dict(tag='bw', steps=500000, visual=True),
         'hanoi-double-task1-v0': dict(tag='hd', steps=1000000, visual=False)}
PAIRS = (2000, 6000, 20000, 60000, 190000)       # 190K = every training pair of the 200K file (5% held out)
EPISODES = (250, 500, 1000, 2000, 4000, 8818)    # 8818 = all successful visual-block-single demos
LOW_PAIRS = (20, 60, 200, 600)                   # push P down until E = 8818 falls to ~0.9 (P = 2K gives 0.95)


def pname(P):
    return f'{P // 1000}k' if P >= 1000 else str(P)


def bc(run, env, enc, group, tags, after=(), episodes=0, seed=0, steps=300000):
    cmd = [TORCH, '-u', 'scripts/train_vis.py', '--run', run, '--env', env, '--group', GROUP, '--input', 'emb',
           '--encoder', enc, '--scale', 'std', '--steps', str(steps), '--seed', str(seed), '--eval-interval', '25000',
           '--keep-every', str(steps), '--final-eval-episodes', '1000', '--drop-ckpt',
           '--tags', ','.join([f'group:{group}', f'encoder:{enc}', 'frozen:true', f'episodes:{episodes or "all"}',
                               f'seed:{seed}', *tags])]
    if episodes:
        cmd[cmd.index('--steps'):cmd.index('--steps')] = ['--episodes', str(episodes)]
    # frozen-embedding BC measured at ~3 GB RSS, ~1 core, 4-5.5 GB GPU memory
    return dict(run=run, group=group, torch=True, time='16:00:00', mem='16G', cpus=3, after=list(after), cmd=cmd,
                gpus=CHEAP_OK)


def job(run, group, cmd, torch=True, time='06:00:00', mem='96G', cpus=16, after=(), gpus=None):
    e = dict(run=run, group=group, torch=torch, time=time, mem=mem, cpus=cpus, wandb=False, after=list(after),
             cmd=['bash', '-c', cmd + f' && touch runs/{run}/DONE'])
    if gpus:
        e['gpus'] = gpus
    return e


def enc(run, env, group, tags, after=(), extra=(), mem='96G', cpus=16):
    # pretrain_enc reads the whole pair file into RAM (200K pairs = 22 GB; 25K = 2.8 GB)
    return dict(run=run, group=group, torch=True, time='12:00:00', mem=mem, cpus=cpus, after=list(after),
                cmd=[TORCH, '-u', 'scripts/pretrain_enc.py', '--run', run, '--env', env, '--data', 'nodr',
                     '--arch', 'dinov2_s', '--group', GROUP, *extra, '--tags', ','.join(['stage:encoder', *tags])])


def experiments():
    E = []
    # ------------------------------------------------------------------ Q
    E.append(job('gen-Q-yawprobe', 'Q', f'{TORCH} -u scripts/yaw_probe.py', time='04:00:00', gpus=ANY_GPUS))
    E.append(enc('q-sgl-enc-noyaw', SENV, 'Q', ['targets:all-minus-cube_yaw'], extra=['--exclude-groups', 'cube_yaw']))
    E.append(job('gen-Q-embed', 'Q', f'{TORCH} -u scripts/embed_probe.py --env {SENV} --encoders pre:q-sgl-enc-noyaw '
                 f'&& {TORCH} -u scripts/yaw_probe.py --encoders pre:q-sgl-enc-noyaw', after=['q-sgl-enc-noyaw']))
    E.append(bc('q-sgl-bc-noyaw-s0', SENV, 'pre:q-sgl-enc-noyaw', 'Q', ['targets:all-minus-cube_yaw'], after=['Q-embed']))
    # ------------------------------------------------------------------ G
    for env, t in TASKS.items():
        tag, venv = t['tag'], f'visual-{env}'
        E.append(job(f'gen-G-prefill-{tag}', 'G', f'{JAX} -u scripts/collections_store.py --env {env} --idx 0-29 --val',
                     torch=False, time='12:00:00', mem='64G', cpus=8))
        for n in (10000, 30000):
            run = f'g-{tag}-state-n{n // 1000}k-int25-s0'
            E.append(dict(run=run, group='G', torch=False, time='24:00:00', mem='64G' if tag == 'hd' else '32G', cpus=6,
                          gpus=CHEAP_OK, after=[f'G-prefill-{tag}'],
                          cmd=[JAX, '-u', 'scripts/train_regime.py', '--run', run, '--env', env, '--base', str(n),
                               '--start-interval', '25', '--features', 'handcrafted', '--steps', str(t['steps']),
                               '--group', GROUP, '--tags', f'group:G,input:state,env:{env},base:{n},start_interval:25']))
        if not t['visual']:
            continue
        E.append(job(f'gen-G-viscol-{tag}', 'G', f'{JAX} -u scripts/vis_collect.py --env {env} --idx 0-9',
                     torch=False, time='24:00:00', mem='160G', cpus=16))
        E.append(job(f'gen-G-visprep-{tag}', 'G', f'{TORCH} -u scripts/vis_prep.py --env {venv} --no-images && '
                     f'{TORCH} -u scripts/embed_probe.py --env {venv} --encoders dinov2_s_p4 --no-probe',
                     after=[f'G-viscol-{tag}'], gpus=ANY_GPUS))
        E.append(job(f'gen-G-drgen-{tag}', 'G', f'{TORCH} -u scripts/dr_gen.py --env {venv} --n 200000 --workers 30 '
                     '--variants nodr --source collections --collections 0-9 --no-replace',
                     time='10:00:00', cpus=32, after=[f'G-prefill-{tag}']))
        E.append(enc(f'g-{tag}-enc-nodr-dinov2s', venv, 'G', [f'env:{env}'], after=[f'G-drgen-{tag}'],
                     extra=['--select', 'best'] + (['--exclude-groups', 'peg_xyz,peg_yaw'] if 'hanoi' in env else [])))
        E.append(job(f'gen-G-embedpre-{tag}', 'G', f'{TORCH} -u scripts/embed_probe.py --env {venv} '
                     f'--encoders pre:g-{tag}-enc-nodr-dinov2s --no-probe',
                     after=[f'G-visprep-{tag}', f'g-{tag}-enc-nodr-dinov2s'], gpus=ANY_GPUS))
        E.append(bc(f'g-{tag}-vis-dinov2_s_p4-s0', venv, 'dinov2_s_p4', 'G', [f'env:{env}'], after=[f'G-visprep-{tag}'])
                 | {'mem': '48G'})
        E.append(bc(f'g-{tag}-vis-pre-s0', venv, f'pre:g-{tag}-enc-nodr-dinov2s', 'G', [f'env:{env}'],
                    after=[f'G-embedpre-{tag}']))
    # ------------------------------------------------------------------ S
    encs = []
    for P in PAIRS:
        name = f's-sgl-enc-p{P // 1000}k'
        encs.append(name)
        E.append(enc(name, SENV, 'S', [f'pairs:{P}', 'select:best'], extra=['--n-pairs', str(P), '--select', 'best']))
    E.append(job('gen-S-embed', 'S', f'{TORCH} -u scripts/embed_probe.py --env {SENV} --encoders '
                 + ','.join(f'pre:{e}' for e in encs) + ' --no-probe', after=encs, gpus=ANY_GPUS))
    for P, name in [(0, 'dinov2_s')] + [(P, f'pre:{e}') for P, e in zip(PAIRS, encs)]:
        for ep in EPISODES:
            for seed in ((0, 1) if ep in (1000, 8818) else (0,)):
                run = f's-sgl-bc-p{P // 1000}k-e{ep}-s{seed}'
                E.append(bc(run, SENV, name, 'S', [f'pairs:{P}'], after=[] if P == 0 else ['S-embed'],
                            episodes=0 if ep == 8818 else ep, seed=seed))
    # ------------------------------------------------------------------ S, low pairs (group SL)
    E.append(job('gen-SL-drgen', 'SL', f'{TORCH} -u scripts/dr_gen.py --env {SENV} --n 200000 --workers 30 '
                 '--variants nodr', time='10:00:00', cpus=32))
    low = []
    for P in LOW_PAIRS:
        name = f's-sgl-enc-p{pname(P)}'
        low.append(name)
        E.append(enc(name, SENV, 'SL', [f'pairs:{P}', 'select:best'], after=['SL-drgen'],
                     extra=['--n-pairs', str(P), '--select', 'best']))
    for P, name in zip(LOW_PAIRS, low):  # one embed job per encoder, so each row's BC starts as soon as it can
        E.append(job(f'gen-SL-embed-p{pname(P)}', 'SL', f'{TORCH} -u scripts/embed_probe.py --env {SENV} '
                     f'--encoders pre:{name} --no-probe', after=[name], gpus=ANY_GPUS))
        for ep in EPISODES:
            E.append(bc(f's-sgl-bc-p{pname(P)}-e{ep}-s0', SENV, f'pre:{name}', 'SL', [f'pairs:{P}'],
                        after=[f'SL-embed-p{pname(P)}'], episodes=0 if ep == 8818 else ep))
    # ------------------------------------------------------------------ X (round 3)
    def drgen(name, grp, extra, env=SENV):
        return job(name, grp, f'{TORCH} -u scripts/dr_gen.py --env {env} --n 25000 --workers 30 --variants nodr {extra}',
                   time='04:00:00', cpus=32)

    def embed(name, grp, env, encs, after):  # embeddings are held in RAM: 4x4 caches need ~2x their size
        big = '@p4' in encs
        return job(name, grp, f'{TORCH} -u scripts/embed_probe.py --env {env} --encoders {encs} --no-probe',
                   after=after, gpus=CHEAP_OK, mem='96G' if big else '24G', cpus=16 if big else 6)

    def fast(e, gpus=FAST):
        return e | {'gpus': gpus}

    # X1: states not from the demos
    E.append(drgen('gen-X1-drgen', 'X1', '--states uniform --tag uniform'))
    for P in (200, 2000, 20000):
        name = f'x1-sgl-enc-uni-p{pname(P)}'
        E.append(enc(name, SENV, 'X1', [f'pairs:{P}', 'states:uniform'], after=['X1-drgen'], mem='32G', cpus=8,
                     extra=['--n-pairs', str(P), '--select', 'best', '--dr-dir', f'{SENV}-uniform']))
        E.append(embed(f'gen-X1-embed-p{pname(P)}', 'X1', SENV, f'pre:{name}', [name]))
        for ep in (1000, 8818):
            E.append(fast(bc(f'x1-sgl-bc-uni-p{pname(P)}-e{ep}-s0', SENV, f'pre:{name}', 'X1',
                             [f'pairs:{P}', 'states:uniform'], after=[f'X1-embed-p{pname(P)}'],
                             episodes=0 if ep == 8818 else ep)))
    # X2: hanoi-single pairs scaling
    E.append(drgen('gen-X2-drgen', 'X2', '--source collections --collections 0-9 --no-replace', env=HENV))
    for P in (200, 2000, 20000):
        name = f'x2-hs-enc-p{pname(P)}'
        E.append(enc(name, HENV, 'X2', [f'pairs:{P}', 'env:hanoi-single'], after=['X2-drgen'], mem='32G', cpus=8,
                     extra=['--n-pairs', str(P), '--select', 'best', '--exclude-groups', 'peg_xyz,peg_yaw']))
        E.append(embed(f'gen-X2-embed-p{pname(P)}', 'X2', HENV, f'pre:{name}', [name]))
        for ep in (1000, 9962):
            E.append(fast(bc(f'x2-hs-bc-p{pname(P)}-e{ep}-s0', HENV, f'pre:{name}', 'X2', [f'pairs:{P}'],
                             after=[f'X2-embed-p{pname(P)}'], episodes=0 if ep == 9962 else ep)))
    E.append(fast(bc('x2-hs-bc-p190k-e1000-s0', HENV, 'pre:g-hs-enc-nodr-dinov2s', 'X2', ['pairs:190000'],
                     episodes=1000)))
    E.append(fast(bc('x2-hs-bc-dinov2_s_p4-e1000-s0', HENV, 'dinov2_s_p4', 'X2', ['pairs:0'], episodes=1000)
                  | {'mem': '48G'}))
    # X3: appearance shift at evaluation (keeps ckpt.pt and params for later evaluations)
    for tag, encoder in (('nodr', 'pre:v-F-enc-nodr-dinov2s'), ('dr', 'pre:v-F-enc-dr-dinov2s'), ('p4', 'dinov2_s_p4')):
        e = fast(bc(f'x3-sgl-bc-{tag}-s0', SENV, encoder, 'X3', [f'pretrain:{tag}']))
        cmd = [c for c in e['cmd'] if c != '--drop-ckpt'] + ['--shift-eval', 'nominal,color,light,camera,all']
        E.append(e | {'cmd': cmd, 'mem': '48G' if tag == 'p4' else '24G'})
    # X4: hanoi state BC + relative disk yaw
    for n in (10000, 30000):
        run = f'x4-hs-state-diskyaw-n{n // 1000}k-int25-s0'
        E.append(dict(run=run, group='X4', torch=False, time='24:00:00', mem='32G', cpus=6, gpus=FAST,
                      cmd=[JAX, '-u', 'scripts/train_regime.py', '--run', run, '--env', 'hanoi-single-task1-v0',
                           '--base', str(n), '--start-interval', '25', '--features', 'handcrafted+diskyaw',
                           '--steps', '500000', '--group', GROUP,
                           '--tags', f'group:X4,input:state,features:handcrafted+diskyaw,base:{n},start_interval:25']))
    # X5: fine-tune low-pair encoders end to end (recipe of v-D-dinov2_s-ft-s0: pixels, batch 256, 200K, enc lr 1e-5)
    for P in (20, 200):
        run = f'x5-sgl-bc-ft-p{P}-s0'
        E.append(dict(run=run, group='X5', torch=True, time='24:00:00', mem='64G', cpus=8, gpus=FAST,
                      cmd=[TORCH, '-u', 'scripts/train_vis.py', '--run', run, '--env', SENV, '--group', GROUP,
                           '--input', 'pixels', '--encoder', f'pre:s-sgl-enc-p{P}', '--finetune', '--enc-lr', '1e-5',
                           '--batch', '256', '--steps', '200000', '--seed', '0', '--eval-interval', '25000',
                           '--keep-every', '200000', '--final-eval-episodes', '1000', '--drop-ckpt',
                           '--tags', f'group:X5,encoder:pre:s-sgl-enc-p{P},frozen:false,pairs:{P},input:pixels']))
    # X6: sim-pretrained backbone + 4x4 patch readout
    E.append(embed('gen-X6-embed-sgl', 'X6', SENV, 'pre:s-sgl-enc-p190k@p4,pre:s-sgl-enc-p2k@p4', []))
    for P in ('190k', '2k'):
        for ep in (250, 1000):
            E.append(fast(bc(f'x6-sgl-bc-p{P}p4-e{ep}-s0', SENV, f'pre:s-sgl-enc-p{P}@p4', 'X6', ['readout:p4'],
                             after=['X6-embed-sgl'], episodes=ep) | {'mem': '48G'}))
    E.append(embed('gen-X6-embed-dbl', 'X6', DENV, 'pre:v-dbl-F-enc-nodr-dinov2s@p4', []))
    E.append(fast(bc('x6-dbl-bc-prep4-s0', DENV, 'pre:v-dbl-F-enc-nodr-dinov2s@p4', 'X6', ['readout:p4'],
                     after=['X6-embed-dbl']) | {'mem': '96G'}, gpus=BIG))
    # X7: replicate low-P rows with a new pair sample (same distribution, dr_gen --seed 1) and seed 1
    E.append(drgen('gen-X7-drgen', 'X7', '--seed 1 --tag seed1'))
    for P in LOW_PAIRS:
        name = f's-sgl-enc-p{P}-r1'
        E.append(enc(name, SENV, 'X7', [f'pairs:{P}', 'replicate:1'], after=['X7-drgen'], mem='32G', cpus=8,
                     extra=['--n-pairs', str(P), '--select', 'best', '--dr-dir', f'{SENV}-seed1', '--seed', '1']))
        E.append(embed(f'gen-X7-embed-p{P}', 'X7', SENV, f'pre:{name}', [name]))
        for ep in (1000, 8818):
            E.append(fast(bc(f's-sgl-bc-p{P}-r1-e{ep}-s1', SENV, f'pre:{name}', 'X7', [f'pairs:{P}', 'replicate:1'],
                             after=[f'X7-embed-p{P}'], episodes=0 if ep == 8818 else ep, seed=1)))
    return E


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--only', default='Q,G,S')
    p.add_argument('--runs', default='')
    p.add_argument('--partition', default='deepaklong,all')
    p.add_argument('--list', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    E = experiments()
    if args.list:
        for e in E:
            print(f"{e['group']}  {e['run']:34s} after={e.get('after', [])}  {shlex.join(e['cmd'][2:])[:100]}")
        print(len(E), 'jobs')
        return
    groups, runs = set(args.only.split(',')), set(filter(None, args.runs.split(',')))
    ids = {}
    if LOG.exists():
        for line in LOG.read_text().splitlines():
            r = json.loads(line)
            if subprocess.run(['squeue', '-h', '-j', r['job_id']], capture_output=True, text=True).stdout.strip():
                ids[r['run'].removeprefix('gen-')] = r['job_id']
    for e in E:
        key = e['run'].removeprefix('gen-')
        if (runs and e['run'] not in runs) or (not runs and e['group'] not in groups):
            continue
        deps = [ids[a] for a in e.get('after', []) if a in ids]
        jid = submit(e, deps, args.partition, args.dry_run)
        ids[key] = jid
        print(e['group'], e['run'], jid, 'after', deps, flush=True)


if __name__ == '__main__':
    main()
