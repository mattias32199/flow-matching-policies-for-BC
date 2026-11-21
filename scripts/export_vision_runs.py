"""Export every visual-BC / block-double study run to experiments/vision-bc/{runs.csv,configs/}.

Per run: task, study part, trainer, key settings, updates, batch, samples seen, passes over the training starts/frames,
wall-clock GPU hours (summed over Slurm segments in runs/<run>/launcher.log, 1 GPU per job), GPU models used, final
success (the 1000-rollout final evaluation when the run has one, else the mean of the last 3 evaluations; bowling
reports the episode return), the final-evaluation rollout count, and the best evaluation.

  .venv/bin/python scripts/export_vision_runs.py
"""
import csv
import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments' / 'vision-bc'
PATTERNS = ['v-state-single-*', 'v-B-*', 'v-C-*', 'v-D-*', 'v-E-*', 'v-F-*', 'h-single-*', 'h3-single-*', 'h-sgl-*',
            'v-state-double-*', 'r-*', 'h-double-*', 'h2-double-*', 'v-dbl-*', 'h-dbl-*', 'regime-*',
            # follow-up (followup.md, scripts/launch_gen.py): Q, G, S, round-3 X1-X7 and their support jobs
            'q-*', 'g-*', 's-*', 'x[1-7]-*', 'gen-*']
# Training frames / chunk starts available to the visual trainers (successful episodes, non-final frames).
VIS_FRAMES = {'visual-block-single-task1-v0': 235673}


def node_gpus():
    out = subprocess.run(['sinfo', '-h', '-N', '-o', '%N %f'], capture_output=True, text=True).stdout
    return {l.split()[0]: l.split()[1] for l in out.splitlines() if len(l.split()) == 2}


def segments(run_dir):
    """Wall-clock hours and nodes from launcher.log (start ... signal/exit pairs)."""
    p = run_dir / 'launcher.log'
    if not p.exists():
        return None, []
    hours, start, nodes = 0.0, None, []
    for line in p.read_text().splitlines():
        t = datetime.fromisoformat(line.split()[0])
        if ' start ' in line:
            if start is not None:  # previous segment ended without a log line (preempted hard)
                pass
            start = t
            m = re.search(r'node=(\S+)', line)
            if m:
                nodes.append(m.group(1))
        elif start is not None and ('signal' in line or 'trainer exit' in line or 'bad GPU' in line):
            hours += (t - start).total_seconds() / 3600
            start = None
    return round(hours, 2), nodes


def last_metric(run_dir, name):
    p = run_dir / name
    if not p.exists():
        return {}
    lines = [l for l in p.read_text().splitlines() if l.strip()]
    return json.loads(lines[-1]) if lines else {}


def evals(run_dir):
    p = run_dir / 'eval.jsonl'
    if not p.exists():
        return []
    return [(r['step'], r.get('success', 0.0)) for r in map(json.loads, p.read_text().splitlines()) if r]


def part(run):
    if run.startswith(('q-', 'g-', 's-', 'x1-', 'x2-', 'x3-', 'x4-', 'x5-', 'x6-', 'x7-', 'gen-')):
        return 'D follow-up'
    if run.startswith(('h-', 'h2-', 'h3-')):
        return 'C explanation'
    if run.startswith(('r-', 'v-state-double', 'regime-')) or run.startswith('v-dbl-') or 'dbl' in run:
        return 'B block-double'
    return 'A block-single'


def task(run, env):
    for k in ('hanoi-single', 'hanoi-double', 'bowling'):
        if k in env:
            return k
    if run.startswith(('g-hs', 'x2-hs', 'x4-hs')):
        return 'hanoi-single'
    if run.startswith('g-hd'):
        return 'hanoi-double'
    if run.startswith('g-bw'):
        return 'bowling'
    return 'double' if 'double' in (env + run) or 'dbl' in run or run.startswith('r-') else 'single'


def main():
    (OUT / 'configs').mkdir(parents=True, exist_ok=True)
    gpu = node_gpus()
    vis_frames = dict(VIS_FRAMES)
    dbl_index = ROOT / 'data' / 'vis' / 'visual-block-double-task1-v0' / 'train_index.npz'
    if dbl_index.exists():
        import numpy as np
        with np.load(dbl_index) as z:
            vis_frames['visual-block-double-task1-v0'] = int(z['chunks'].shape[0])
    rows = []
    dirs = sorted({d for pat in PATTERNS for d in (ROOT / 'runs').glob(pat) if d.is_dir() and (d / 'launcher.log').exists()})
    for d in dirs:
        run = d.name
        cfg = json.loads((d / 'config.json').read_text()) if (d / 'config.json').exists() else {}
        if cfg:
            shutil.copy(d / 'config.json', OUT / 'configs' / f'{run}.json')
        hours, nodes = segments(d)
        e = evals(d)
        trainer = cfg.get('trainer') or cfg.get('kind') or ('train_vis' if 'input' in cfg else 'train.py' if 'features' in cfg else 'job')
        if run.startswith(('regime-', 'gen-')):
            trainer = 'support job'
        steps = cfg.get('steps')
        batch = cfg.get('batch_size') or cfg.get('batch')
        env = cfg.get('env', '')
        passes = None
        data = last_metric(d, 'data.jsonl')
        if trainer == 'train_regime' and data and steps and batch:
            passes = steps * batch / data['allowed_starts']
        elif 'input' in cfg and steps and batch and env in vis_frames:
            passes = steps * batch / vis_frames[env]
        elif trainer == 'encoder_pretrain' and steps and batch:
            passes = steps * batch / ((cfg.get('n_pairs') or 200000) * 0.95)
        elif 'features' in cfg and cfg.get('start_interval') and steps and batch:
            ds = last_metric(d, 'dataset.jsonl')
            if 'allowed_train_starts' in ds:
                passes = steps * batch / ds['allowed_train_starts']
        settings = {k: cfg[k] for k in ('features', 'base', 'episodes', 'start_interval', 'refresh_every', 'input', 'encoder',
                                         'scale', 'adapter', 'finetune', 'aux_frac', 'lr', 'enc_lr', 'img', 'data', 'arch',
                                         'groups', 'exclude_groups', 'n_pairs', 'dr_dir', 'extra', 'shift_eval', 'scratch',
                                         'seed') if k in cfg and cfg[k] not in (None, '', False)}
        fe = last_metric(d, 'final_eval.jsonl')
        if 'bowling' in env:  # no success flag; the score is the episode return
            e = [(r['step'], r.get('return', r.get('episode.return', 0.0))) for r in map(json.loads, (d / 'eval.jsonl').read_text().splitlines()) if r] \
                if (d / 'eval.jsonl').exists() else []
        rows.append(dict(
            run=run, part=part(run), task=task(run, env),
            trainer=trainer, settings=json.dumps(settings, separators=(',', ':')), updates=steps, batch=batch,
            samples_seen=steps * batch if steps and batch else None,
            passes_over_training_data=round(passes, 2) if passes else None,
            gpu_hours=hours, gpus='/'.join(sorted({gpu.get(n, '?') for n in nodes})), slurm_segments=len(nodes),
            final_success=round(fe['return' if 'bowling' in env else 'success'], 3) if fe else
            round(sum(v for _, v in e[-3:]) / len(e[-3:]), 3) if e else None,
            final_eval_rollouts=fe.get('episodes'),
            best_success=round(max(v for _, v in e), 3) if e else None, done=(d / 'DONE').exists()))
    with open(OUT / 'runs.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tot = sum(r['gpu_hours'] or 0 for r in rows)
    print(f'{len(rows)} runs, {tot:.0f} GPU-hours total')
    for p in sorted({r['part'] for r in rows}):
        print(p, round(sum(r['gpu_hours'] or 0 for r in rows if r['part'] == p)), 'GPU-h')


if __name__ == '__main__':
    main()
