"""Submit owner-private launchers with opaque names; administrators retain access."""
import json
import os
from pathlib import Path
import secrets
import shlex
import subprocess
ROOT = Path(__file__).resolve().parents[1]


def submit(command, partition, time_limit='02:00:00', dependency=None, constraint='A6000', cpus=8, mem='48G',
           exclude=None, requeue=False, torch_env=False):
    if ROOT.stat().st_mode & 0o077:
        raise PermissionError('Workspace must be owner-only before private submission.')
    old_mask = os.umask(0o077)
    try:
        directory = ROOT.parent / '.private-jobs' / secrets.token_hex(16)
        directory.mkdir(parents=True, mode=0o700)
        directory.parent.chmod(0o700)
        launcher = directory / (secrets.token_hex(12) + '.sh')
        launcher.write_text('#!/usr/bin/env bash\nset -euo pipefail\numask 0077\ncd '
            + shlex.quote(str(ROOT)) + '\nsource scripts/env.sh\n'
            # Per-job Warp kernel cache: concurrent jobs racing on one NFS cache directory fail.
            + 'export WARP_CACHE_PATH="$REPO_ROOT/.local/cache/warp-job${SLURM_JOB_ID:-local}"\n'
            + ('unset LD_LIBRARY_PATH\n' if torch_env else '')
            + 'exec ' + shlex.join(command) + '\n')
        launcher.chmod(0o700)
        arguments = ['sbatch', '--parsable', '--job-name=' + secrets.token_hex(5),
                     '--partition=' + partition, '--constraint=' + constraint, '--gres=gpu:1',
                     '--cpus-per-task=' + str(cpus), '--mem=' + mem, '--time=' + time_limit,
                     '--chdir=' + str(directory), '--output=' + str(directory / 'o'),
                     '--error=' + str(directory / 'e')]
        if dependency:
            arguments.append('--dependency=' + dependency)
        if exclude:
            arguments.append('--exclude=' + exclude)
        if requeue:
            arguments.append('--requeue')
        else:
            arguments.append('--no-requeue')
        result = subprocess.run(arguments + [str(launcher)], cwd=directory, capture_output=True, text=True, check=True, timeout=30)
        job_id = result.stdout.strip().split(';')[0]
        if not job_id.isdigit():
            raise ValueError('Invalid scheduler job ID.')
        metadata = dict(job_id=job_id, private_directory=str(directory), command=command, partition=partition,
                        constraint=constraint, cpus=cpus, mem=mem, time=time_limit)
        with (ROOT / '.local/private-submissions.jsonl').open('a') as f:
            f.write(json.dumps(metadata) + '\n')
        return job_id
    finally:
        os.umask(old_mask)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--partition', required=True)
    parser.add_argument('--time', default='02:00:00')
    parser.add_argument('--dependency')
    parser.add_argument('--constraint', default='A6000')
    parser.add_argument('--cpus', type=int, default=8)
    parser.add_argument('--mem', default='48G')
    parser.add_argument('--exclude')
    parser.add_argument('--requeue', action='store_true')
    parser.add_argument('--torch-env', action='store_true')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('A command is required.')
    print(submit(command, args.partition, args.time, args.dependency, args.constraint, args.cpus, args.mem,
                 args.exclude, args.requeue, args.torch_env))
