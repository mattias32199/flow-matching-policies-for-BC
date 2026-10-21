"""Maintain a bounded, privately submitted pool of data producers."""
import json
import os
from pathlib import Path
import subprocess
import time
from collection_cache import locked, write_json, CollectionCache
from private_slurm import submit
ROOT = Path(__file__).resolve().parents[1]
_last_check = 0.


def ensure_producer(args, needed_index=0):
    global _last_check
    path = ROOT / '.local/demo-producer-config.json'
    if not path.exists() or time.monotonic() - _last_check < 60:
        return
    _last_check = time.monotonic()
    config = json.loads(path.read_text())
    if not config.get('enabled') or args.data_seed not in config.get('seeds', []):
        return
    if not args.collection_cache or args.mode != 'streaming':
        return
    status = CollectionCache(ROOT / 'data/shared-demos').status()
    if status['used_bytes'] + sum(status['reservations'].values()) + 1_129_000_000 > status['limit_bytes']:
        return
    try:
        with locked(ROOT / '.local/demo-producer.lock'):
            state_path = ROOT / '.local/demo-producer-job.json'
            previous = json.loads(state_path.read_text()) if state_path.exists() else {}
            if previous and time.time() - previous.get('checked_at', 0) < 60:
                return
            query = subprocess.run(['squeue', '-h', '-u', str(os.getuid()), '-o', '%i %T'],
                                   capture_output=True, text=True, check=True, timeout=15)
            active = {line.split()[0] for line in query.stdout.splitlines() if line.split()}
            live = [j for j in previous.get('jobs', []) if str(j['job_id']) in active]
            if live and any(j['seed'] != args.data_seed for j in live):
                write_json(state_path, dict(jobs=live, checked_at=time.time()))
                return
            workers = min(4, max(1, int(config.get('workers', 2))))
            for shard in range(workers):
                if any(j['shard'] == shard for j in live):
                    continue
                command = ['.venv/bin/python', '-u', 'scripts/demo_producer.py',
                           '--seed', str(args.data_seed), '--window', str(config.get('window', 24)),
                           '--shard', str(shard), '--shards', str(workers), '--start-index', str(needed_index)]
                job_id = submit(command, partition=config['partition'], time_limit='2-00:00:00')
                live.append(dict(job_id=job_id, seed=args.data_seed, shard=shard))
                write_json(state_path, dict(jobs=live, checked_at=time.time()))
            write_json(state_path, dict(jobs=live, checked_at=time.time()))
            message = f"Demo producer pool for data seed {args.data_seed}: " + ', '.join(j['job_id'] for j in live)
    except (subprocess.SubprocessError, OSError, ValueError) as error:
        message = f'Demo producer submission/check failed ({type(error).__name__}); training uses synchronous cache fallback.'
    progress = ROOT / 'PROGRESS.md'
    if progress.exists():
        progress.read_text()
    with progress.open('a') as f:
        f.write(f'\n- {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}: {message}\n')
    print(message, flush=True)
