"""Durable four-GPU queue with periodic metric export and Git synchronization."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from matrix import matrix
from summarize import summarize

ROOT = Path(__file__).resolve().parents[1]


def progress(message):
    path = ROOT / 'PROGRESS.md'
    if path.exists():
        path.read_text()  # Always read before append; history stays intact.
    with path.open('a') as file:
        file.write(f'- {datetime.datetime.now(datetime.timezone.utc).isoformat()}: {message}\n')


def sync():
    summarize()
    # Explicit allowlist excludes local logs, paths, data, environments and checkpoints.
    subprocess.run(['git', 'add', 'results/curves.csv'], cwd=ROOT, check=True)
    changes = subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=ROOT)
    if changes.returncode == 1:
        subprocess.run(['git', 'commit', '-m', 'Record experiment metrics'], cwd=ROOT, check=True)
    subprocess.run(['git', 'push', 'origin', 'HEAD'], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--steps', type=int, default=2000000)
    parser.add_argument('--pilot-first', action='store_true')
    parser.add_argument('--sync-seconds', type=int, default=1800)
    args = parser.parse_args()
    pending = matrix()
    if args.pilot_first:
        pilots = []
        for job in pending[:4]:
            pilots.append(dict(job, run='pilot-' + job['run'], episodes=64,
                               steps=100, eval_interval=100, eval_episodes=16,
                               parallel=64, log_interval=10, save_interval=100))
        pending = pilots + pending
    active = {}
    handles = []
    failures = []
    last_sync = time.monotonic()
    for gpu in args.gpus.split(','):
        active[gpu] = None
    while pending or any(active.values()):
        for gpu, entry in list(active.items()):
            if entry is not None:
                process, job, logfile = entry
                code = process.poll()
                if code is None:
                    continue
                logfile.close()
                progress(f"Run {job['run']} exited {code} on GPU {gpu}.")
                if code:
                    failures.append(job['run'])
                    # Stop assigning this device on failure, but let other GPUs continue.
                    del active[gpu]
                    continue
                active[gpu] = None
            if pending and gpu in active and active[gpu] is None:
                job = pending.pop(0)
                run = ROOT / 'runs' / job['run']
                if (run / 'DONE').exists():
                    continue
                if (run / 'config.json').exists():
                    failures.append(job['run'])
                    progress(f"Refusing automatic overwrite of incomplete run {job['run']}.")
                    continue
                cmd = [sys.executable, '-u', 'scripts/train.py', '--steps', str(args.steps)]
                for key, value in job.items():
                    cmd.extend(['--' + key.replace('_', '-'), str(value)])
                logpath = ROOT / '.local' / f"{job['run']}.log"
                logfile = logpath.open('a')
                process = subprocess.Popen(cmd, cwd=ROOT, stdout=logfile, stderr=subprocess.STDOUT,
                                           env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu))
                active[gpu] = process, job, logfile
                progress(f"Launched {job['run']} on GPU {gpu}, PID {process.pid}.")
        if time.monotonic() - last_sync > args.sync_seconds:
            try:
                sync()
            except subprocess.CalledProcessError as error:
                progress(f'Periodic Git sync failed with exit {error.returncode}; retry next interval.')
            last_sync = time.monotonic()
        state = {gpu: entry[1]['run'] if entry else None for gpu, entry in active.items()}
        (ROOT / '.local' / 'queue-status.json').write_text(json.dumps(
            dict(active=state, pending=len(pending), failures=failures), indent=2))
        if not active:
            progress('All GPU workers stopped after errors. Queue remains incomplete.')
            break
        time.sleep(30)
    try:
        sync()
    except subprocess.CalledProcessError as error:
        progress(f'Final Git sync failed with exit {error.returncode}; local results retained.')
    progress(f'Supervisor stopped. Failed runs: {failures}; pending jobs: {len(pending)}.')


if __name__ == '__main__':
    main()
