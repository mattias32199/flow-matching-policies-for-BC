"""Publish completed portable profiling evidence to the existing W&B project."""
import argparse
import hashlib
import json
from pathlib import Path
import wandb
from wandb_mirror import settings, ensure_credentials, ENTITY, PROJECT, APP_URL


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--summary', type=Path, default=Path('results/profiling/summary.json'))
    args = p.parse_args()
    ensure_credentials()
    payload = args.summary.read_bytes()
    data = json.loads(payload)
    identity = 'profile-' + hashlib.sha256(payload).hexdigest()[:16]
    directory = Path('.local/wandb')
    directory.mkdir(parents=True, exist_ok=True)
    run = wandb.init(entity=ENTITY, project=PROJECT, id=identity,
                     name='Isolated A6000 training profile', group='profiling',
                     job_type='profiling', tags=['A6000', 'synthetic', 'profiling'],
                     mode='online', resume='allow', dir=str(directory), settings=settings(),
                     config={'batch_size': 1024, 'parameter_count': 136179887,
                             'timing_repetitions': 3, 'synthetic_batches': True,
                             'mfu_method': 'Analytical dense-matrix estimate, not hardware counters'})
    columns = ['case', 'updates_per_second', 'relative_speed', 'approximate_dense_mfu']
    table = wandb.Table(columns=columns, data=[
        [r['case'], r['median_updates_per_second'], r['speedup_vs_original'], r['approximate_dense_mfu']]
        for r in data['training']])
    run.log({'training_cases': table,
             'throughput_comparison': wandb.plot.bar(table, 'case', 'updates_per_second', title='Full-model updates/s')})
    run.summary['records_url'] = 'https://github.com/mattias32199/flow-matching-policies-for-BC/tree/reproduce-ocbench-mysteries-1-4/results/profiling'
    run.summary['experiment_status'] = 'completed'
    run.summary['learning_equivalence_validated'] = False
    run.finish()
    url = f'{APP_URL}/runs/{identity}'
    (args.summary.parent / 'tracking.json').write_text(json.dumps({'url': url, 'summary_sha256': hashlib.sha256(payload).hexdigest()}, indent=2) + '\n')
    print(url)


if __name__ == '__main__':
    main()
