"""Mirror one run directory's JSONL logs to W&B; safe across Slurm requeues.

Started in the background by each m4-vision-single launcher (scripts/launch_m4v.py). The W&B run
id is derived from the run name, so a requeued job resumes the same W&B run. Progress is kept in
runs/<run>/wandb_state.json as the last logged gradient step per stream: after a resume from an
older checkpoint, re-trained steps are skipped, so curves stay monotonic with no duplicates.

  python scripts/wandb_follow.py --run NAME [--pid PID] [--interval 60]
Exits after a final flush when PID (the trainer) is gone, or after one pass without --pid.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTITY, PROJECT = 'yzchhr', 'ocbench-bc-mysteries-1-4'
API_URL = 'https://api.forge.coreweave.com'
STREAMS = {'metrics.jsonl': '', 'eval.jsonl': 'evaluation/', 'encoder.jsonl': 'encoder/'}


def read(path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            break  # partially written last line
    return out


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--pid', type=int)
    p.add_argument('--interval', type=int, default=60)
    args = p.parse_args()
    os.environ['WANDB_MODE'] = 'online'
    import wandb
    d = ROOT / 'runs' / args.run
    while not (d / 'config.json').exists():
        if args.pid and not alive(args.pid):
            return
        time.sleep(10)
    config = json.loads((d / 'config.json').read_text())
    state_path = d / 'wandb_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    rid = 'm4v-' + hashlib.sha256(args.run.encode()).hexdigest()[:12]
    tags = config.get('tags') or []
    if 'features' in config and not tags:  # JAX state-policy runs (scripts/train.py)
        tags = ['group:A', 'input:state', 'encoder:state', 'frozen:na', 'aux_frac:0',
                f"scale:{config['features']}", f"start_interval:{config.get('start_interval', 1)}"]
        config = config | {'tags': tags, 'wandb_group': 'm4-vision-single'}
    run = wandb.init(entity=ENTITY, project=PROJECT, id=rid, name=args.run, resume='allow',
                     group=config.get('wandb_group', 'm4-vision-single'), tags=tags, config=config,
                     dir=str(ROOT / '.local' / 'wandb'),
                     settings=wandb.Settings(base_url=API_URL, console='off', disable_git=True, save_code=False,
                                             x_disable_stats=True, x_disable_machine_info=True, init_timeout=90))
    run.define_metric('gradient_step')
    for prefix in ('training/*', 'validation/*', 'evaluation/*', 'encoder/*', 'perf/*'):
        run.define_metric(prefix, step_metric='gradient_step')
    jobs = read(d / 'jobs.jsonl')
    run.summary['requeues'] = max(0, len(jobs) - 1)
    run.summary['wandb_follow_job'] = os.environ.get('SLURM_JOB_ID')

    def flush():
        n = 0
        for stream, prefix in STREAMS.items():
            last = state.get(stream, -1)
            for rec in read(d / stream):
                step = rec.get('step')
                if step is None or step <= last:
                    continue
                row = {'gradient_step': step}
                for k, v in rec.items():
                    if k == 'step' or not isinstance(v, (int, float)) or isinstance(v, bool):
                        continue
                    key = k if '/' in k else (prefix + k if prefix else f'perf/{k}')
                    row[key] = v
                run.log(row)
                last = step
                n += 1
            state[stream] = last
        tmp = state_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(state))
        tmp.replace(state_path)
        return n

    try:
        while True:
            flush()
            if not args.pid or not alive(args.pid):
                break
            time.sleep(args.interval)
        flush()
        done = (d / 'DONE').exists()
        run.summary['status'] = 'completed' if done else 'interrupted'
        evals = read(d / 'eval.jsonl')
        if evals:
            run.summary['final_success'] = evals[-1].get('success')
            run.summary['best_success'] = max(e.get('success', 0) for e in evals)
    finally:
        run.finish(exit_code=0)


if __name__ == '__main__':
    main()
