"""Publish completed cache measurements without machine metadata."""
import hashlib
import json
from pathlib import Path
import wandb
from wandb_mirror import settings, ensure_credentials, ENTITY, PROJECT, APP_URL


def main():
    root = Path('results/cache-performance')
    payload = (root / 'summary.json').read_bytes()
    summary = json.loads(payload)
    records = [json.loads(s) for s in (root / 'measurements.jsonl').read_text().splitlines()]
    ensure_credentials()
    identity = 'cache-' + hashlib.sha256(payload).hexdigest()[:16]
    directory = Path('.local/wandb')
    directory.mkdir(parents=True, exist_ok=True)
    run = wandb.init(entity=ENTITY, project=PROJECT, id=identity, name='Shared demonstration cache',
                     group='profiling', job_type='profiling', tags=['A6000', 'cache', 'real-data'],
                     mode='online', resume='allow', dir=str(directory), settings=settings(),
                     config=summary['benchmark'] | {'cache_limit_bytes': summary['storage_cap_bytes']})
    table = wandb.Table(columns=['case', 'collection_seconds', 'preprocessing_seconds', 'training_seconds', 'total_phase_seconds'],
        data=[[r['arm'] + '/' + r['feature'], r['collection_seconds'], r['preprocessing_seconds'], r['training_seconds'], r['total_phase_seconds']]
              for r in records if 'arm' in r])
    run.log({'phase_measurements': table,
             'phase_costs': wandb.plot.bar(table, 'case', 'total_phase_seconds', title='One collection and 1024 updates')})
    for k in ('measured_serial_work_speedup', 'cache_read_seconds_median', 'projected_steady_overlap_speedup', 'projected_two_producer_overlap_speedup', 'shared_hashes_identical'):
        run.summary[k] = summary[k]
    run.summary['production_wall_time_speedup_measured'] = False
    run.summary['experiment_status'] = 'completed'
    run.summary['records_url'] = 'https://github.com/mattias32199/flow-matching-policies-for-BC/tree/reproduce-ocbench-mysteries-1-4/results/cache-performance'
    run.finish()
    url = f'{APP_URL}/runs/{identity}'
    (root / 'tracking.json').write_text(json.dumps({'url': url}, indent=2) + '\n')
    print(url)


if __name__ == '__main__':
    main()
