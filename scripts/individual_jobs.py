"""Prepare and submit private individual jobs; retain an auditable local ledger."""
import argparse
import json
import os
from pathlib import Path
import secrets
import shlex
import subprocess
import sys
import time

from collection_cache import locked, write_json

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / '.local/individual-launch-plan.json'
LEDGER = ROOT / '.local/individual-job-status.json'
PREPARED = ROOT / '.local/individual-prepared-jobs.json'


def record(name, **values):
    with locked(LEDGER.with_suffix('.lock')):
        data = json.loads(LEDGER.read_text()) if LEDGER.exists() else {}
        data.setdefault(name, {}).update(values, updated_at=time.time())
        write_json(LEDGER, data)


def active_runs(root=ROOT):
    path = root / '.local/individual-job-status.json'
    data = json.loads(path.read_text()) if path.exists() else {}
    return {name for name, item in data.items()
            if item.get('state') in ('submitted', 'running', 'submitting')}


def log(message):
    path = ROOT / 'PROGRESS.md'
    path.read_text()
    with path.open('a') as file:
        file.write(f'\n- {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}: {message}\n')


def prepare():
    if ROOT.stat().st_mode & 0o077 or ROOT.parent.stat().st_mode & 0o077:
        raise PermissionError('Project and parent must be owner-only.')
    if PREPARED.exists():
        return json.loads(PREPARED.read_text())
    prepared = []
    for job in json.loads(PLAN.read_text()):
        name = job['run']
        if (ROOT / 'runs' / name / 'DONE').exists():
            continue
        if (ROOT / 'runs' / name / 'metrics.jsonl').exists():
            raise RuntimeError(f'Existing incomplete run requires explicit checkpoint continuation: {name}')
        memory = '48G' if name.startswith('m4-') else ('96G' if 'n50000' in name else '64G')
        directory = ROOT.parent / '.private-jobs' / secrets.token_hex(16)
        directory.mkdir(parents=True, mode=0o700)
        directory.parent.chmod(0o700)
        launcher = directory / (name + '.sbatch')
        job_name = name
        lines = ['#!/bin/bash', f'#SBATCH --job-name={job_name}',
                 '#SBATCH --requeue', '#SBATCH --partition=deepaklong', '#SBATCH --constraint=A6000|A5000',
                 '#SBATCH --gres=gpu:1', '#SBATCH --ntasks=1', '#SBATCH --cpus-per-task=8',
                 f'#SBATCH --mem={memory}', '#SBATCH --time=2-00:00:00',
                 f'#SBATCH --chdir={directory}', f'#SBATCH --output={directory / "o"}',
                 f'#SBATCH --error={directory / "e"}',
                 'set -euo pipefail', 'umask 0077', 'cd ' + shlex.quote(str(ROOT)),
                 'source scripts/env.sh',
                 'exec .venv/bin/python -u scripts/individual_jobs.py run --name ' + shlex.quote(name)]
        launcher.write_text('\n'.join(lines) + '\n')
        launcher.chmod(0o700)
        subprocess.run(['/bin/bash', '-n', str(launcher)], check=True)
        prepared.append(dict(run=name, launcher=str(launcher), private_directory=str(directory),
                             job_name=job_name, gpu_constraint='A6000|A5000', cpus=8, memory=memory, partition='deepaklong',
                             time_limit='2-00:00:00', gpus=1, command=job['command']))
    write_json(PREPARED, prepared)
    log(f'Prepared {len(prepared)} owner-private individual launchers; explicit /bin/bash, '
        'deepaklong,1 A6000 or A5000 and8 CPUs each,48/64/96GiB by dataset size,2-day limits. No submissions in prepare mode.')
    return prepared


def submit_all():
    jobs = prepare()
    with locked(ROOT / '.local/individual-submit.lock'):
        for job in jobs:
            data = json.loads(LEDGER.read_text()) if LEDGER.exists() else {}
            previous = data.get(job['run'], {})
            if previous.get('job_id'):
                print(f"Already submitted {job['run']}: {previous['job_id']}", flush=True)
                continue
            if previous.get('state') == 'submission_uncertain':
                raise RuntimeError('Reconcile uncertain submission with scheduler before retrying.')
            record(job['run'], state='submitting')
            try:
                result = subprocess.run(['sbatch', '--parsable', job['launcher']],
                    cwd=job['private_directory'], check=True, text=True, capture_output=True, timeout=45)
                job_id = result.stdout.strip().split(';')[0]
                if not job_id.isdigit():
                    raise ValueError('Unrecognized scheduler acknowledgement.')
            except Exception as error:
                record(job['run'], state='submission_uncertain', error=type(error).__name__)
                log(f"Submission of {job['run']} did not return a verified job ID ({type(error).__name__}); "
                    'no automatic retry to avoid duplicate allocation. Reconcile with scheduler first.')
                if isinstance(error, subprocess.CalledProcessError):
                    print(error.stderr[-2000:], file=sys.stderr)
                raise
            # A fast-starting worker may already have changed its state.
            with locked(LEDGER.with_suffix('.lock')):
                data = json.loads(LEDGER.read_text())
                item = data[job['run']]
                if item.get('state') == 'submitting':
                    item['state'] = 'submitted'
                item.update(job_id=job_id, updated_at=time.time())
                write_json(LEDGER, data)
            log(f"Submitted individual run {job['run']} as job{job_id},2-day deepaklong allocation.")
            print(f"{job['run']} {job_id}", flush=True)


def resolve_run(name, root=ROOT):
    path = root / '.local/individual-replacements.json'
    replacements = json.loads(path.read_text()) if path.exists() else {}
    seen = set()
    while name in replacements:
        if name in seen:
            raise ValueError('Cycle in individual replacement mapping.')
        seen.add(name)
        name = replacements[name]
    return name


def run(name):
    jobs = json.loads(PREPARED.read_text())
    job = next(job for job in jobs if job['run'] == name)
    # Protect against duplicate launchers and preemption restarting an incomplete directory.
    with locked(ROOT / '.local/individual-run-locks' / (name + '.lock')):
        directory = ROOT / 'runs' / name
        if (directory / 'DONE').exists():
            record(name, state='completed')
            return 0
        if (directory / 'metrics.jsonl').exists():
            record(name, state='checkpoint_continuation_required')
            log(f'{name} was restarted after producing metrics; preserving artifacts and requiring '
                'a new explicit continuation segment rather than overwriting checkpoints/logs.')
            return 1
        record(name, state='running', job_id=os.environ.get('SLURM_JOB_ID'))
        log(f'Individual worker started {name}. Shared-cache disk prefetch enabled; no extra GPU allocation.')
        result = subprocess.run(job['command'], cwd=ROOT)
        state = 'completed' if result.returncode == 0 and (directory / 'DONE').exists() else 'failed'
        record(name, state=state, exit_code=result.returncode)
        log(f'Individual worker {name} exited{result.returncode}; state={state}.')
        return result.returncode


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'submit', 'run'])
    parser.add_argument('--name')
    args = parser.parse_args()
    if args.action == 'prepare':
        print(json.dumps(prepare(), indent=2))
    elif args.action == 'submit':
        submit_all()
    else:
        if not args.name:
            parser.error('--name is required for run')
        name = resolve_run(args.name)
        if name != args.name:
            record(args.name, state='superseded', replacement_run=name)
        raise SystemExit(run(name))


if __name__ == '__main__':
    main()
