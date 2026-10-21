"""Backfill and follow existing JSONL logs without restarting GPU workers."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
import time

from catalog import (API_URL, APP_URL, BRANCH, ENTITY, FIELDS, GITHUB, PROJECT, ROOT,
                     clean_config, export_catalog, read_records, run_id)


class MissingCredentials(RuntimeError):
    pass


def ensure_credentials():
    from wandb.sdk.lib import wbauth
    if not os.environ.get('WANDB_API_KEY') and not wbauth.read_netrc_auth(host=API_URL):
        raise MissingCredentials('No WANDB_API_KEY or SDK-managed saved credential is configured.')


def payload(stream, record):
    clean = {key: value for key, value in record.items() if key in FIELDS[stream]}
    identity = hashlib.sha256(json.dumps([stream, clean], sort_keys=True).encode()).hexdigest()[:24]
    if stream == 'metrics.jsonl':
        result = {'gradient_step': clean.pop('step'), **clean}
    elif stream == 'eval.jsonl':
        result = {'gradient_step': clean.pop('step'),
                  **{f'evaluation/{key}': value for key, value in clean.items()}}
    elif stream == 'collections.jsonl':
        result = {f'collection/{key}': value for key, value in clean.items()}
    elif stream == 'continuation.jsonl':
        result = {f'continuation/{key}': value for key, value in clean.items()}
    else:
        result = {f'dataset/{key}': value for key, value in clean.items()}
    result['record_type'] = stream.removesuffix('.jsonl')
    result['record_uid'] = identity
    return result


def publish_records(run, directory, cursors):
    count = 0
    for stream in FIELDS:
        records = read_records(directory / stream)
        offset = cursors.get(stream, 0)
        if offset > len(records):
            raise RuntimeError(f'Refusing truncated log: {directory.name}/{stream}')
        for record in records[offset:]:
            # Internal W&B steps always advance. gradient_step is a custom x axis,
            # so late evaluation results never get dropped as backward steps.
            run.log(payload(stream, record))
            count += 1
        cursors[stream] = len(records)
    return count


def settings():
    import wandb
    return wandb.Settings(
        base_url=API_URL, console='off', disable_git=True, disable_code=True,
        save_code=False, disable_job_creation=True, x_disable_meta=True,
        x_disable_stats=True, x_disable_machine_info=True, init_timeout=45,
    )


def configure_metrics(run):
    run.define_metric('gradient_step')
    for pattern in ['training/*', 'validation/*', 'evaluation/*']:
        run.define_metric(pattern, step_metric='gradient_step')
    run.define_metric('collection/index')
    run.define_metric('collection/*', step_metric='collection/index')


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def progress(message):
    path = ROOT / 'PROGRESS.md'
    if path.exists():
        path.read_text()
    with path.open('a') as file:
        file.write(f'- W&B mirror: {message}\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=int, default=60)
    args = parser.parse_args()
    import wandb
    directory = ROOT / '.local' / 'wandb'
    directory.mkdir(parents=True, exist_ok=True)
    ensure_credentials()
    api = wandb.Api(overrides={'base_url': API_URL}, timeout=30)
    api.create_project(PROJECT, ENTITY)
    write_json(ROOT / '.local' / 'wandb-project.json', {'url': APP_URL, 'entity': ENTITY, 'project': PROJECT})
    print(f'Project ready: {APP_URL}', flush=True)
    state_path = ROOT / '.local' / 'wandb-mirror-state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    links_path = ROOT / '.local' / 'wandb-links.json'
    links = json.loads(links_path.read_text()) if links_path.exists() else {}
    sessions = {}
    catalog_run = wandb.init(
        entity=ENTITY, project=PROJECT, id='ocbench-catalog-v1', name='Run catalog',
        job_type='catalog', resume='allow', mode='online', dir=str(directory),
        settings=settings(), reinit='create_new',
    )
    last_catalog = None
    try:
        while True:
            queue_path = ROOT / '.local' / 'queue-status.json'
            queue = json.loads(queue_path.read_text()) if queue_path.exists() else {}
            from individual_jobs import active_runs
            active = set(queue.get('active', {}).values()) | active_runs()
            changed = 0
            for path in sorted((ROOT / 'runs').glob('*/config.json')):
                config = clean_config(json.loads(path.read_text()))
                name, local = config['run'], path.parent
                identity = run_id(config)
                entry = state.setdefault(identity, dict(cursors={}, finished=False))
                if entry['finished']:
                    continue
                if identity not in sessions:
                    group = 'mystery-1' if name.startswith('m1-') else 'mystery-4'
                    kind = 'reproduction'
                    if name.startswith(('pilot-', 'smoke-')):
                        group, kind = 'diagnostics', 'diagnostic'
                    run = wandb.init(
                        entity=ENTITY, project=PROJECT, id=identity, name=name,
                        group=group, job_type=kind, tags=[config.get('features', 'handcrafted'),kind],
                        config=config, resume='allow', mode='online', dir=str(directory),
                        settings=settings(), reinit='create_new',
                    )
                    configure_metrics(run)
                    sessions[identity] = run
                    url = f'{APP_URL}/runs/{identity}'
                    links[name] = dict(id=identity, url=url)
                    run.summary['records_url'] = f'{GITHUB}/tree/{BRANCH}/results/runs/{name}'
                    run.summary['results_url'] = f'{GITHUB}/blob/{BRANCH}/RESULTS.md'
                    run.summary['evaluation_protocol'] = 'Full action chunks; metrics indexed by gradient_step.'
                    write_json(links_path, links)
                run = sessions[identity]
                changed += publish_records(run, local, entry['cursors'])
                complete = (local / 'DONE').exists()
                inactive = name not in active
                status = 'completed' if complete else ('active' if not inactive else 'incomplete')
                run.summary['experiment_status'] = status
                run.summary['training_records'] = entry['cursors'].get('metrics.jsonl', 0)
                run.summary['evaluation_records'] = entry['cursors'].get('eval.jsonl', 0)
                if complete or inactive:
                    # Distinguish failed/incomplete diagnostics from successful runs.
                    run.finish(exit_code=0 if complete else 1)
                    del sessions[identity]
                    entry['finished'] = True
                write_json(state_path, state)
            entries = export_catalog()
            rows = [[name, item['status'], item['config'].get('seed'),
                     item['config'].get('features'), item['config'].get('mode'),
                     item['wandb_url'], f'{GITHUB}/tree/{BRANCH}/results/runs/{name}',
                     len(read_records(ROOT / 'runs' / name / 'eval.jsonl'))]
                    for name, item in sorted(entries.items())]
            signature = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
            if signature != last_catalog:
                table = wandb.Table(columns=['run', 'status', 'seed', 'features', 'data_mode',
                                             'wandb_run', 'exact_records', 'evaluation_count'], data=rows)
                catalog_run.log({'catalog/runs': table})
                catalog_run.summary['results_url'] = f'{GITHUB}/blob/{BRANCH}/RESULTS.md'
                last_catalog = signature
            subprocess.run([sys.executable, 'scripts/summarize.py'], cwd=ROOT, check=True)
            print(f'Mirrored {changed} new records; {len(sessions)} active runs.', flush=True)
            if not args.watch or not active:
                break
            time.sleep(args.interval)
    finally:
        for run in sessions.values():
            # This process mirrors training; finishing it does not finish the experiment.
            run.finish()
        catalog_run.finish()


if __name__ == '__main__':
    try:
        main()
    except MissingCredentials as error:
        progress(str(error))
        print(str(error), flush=True)
        raise SystemExit(2)
    except Exception as error:
        progress(f'Upload failed ({type(error).__name__}); local scalar logs preserved.')
        raise
