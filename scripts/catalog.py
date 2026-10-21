"""Export portable run records and stable links without copying machine logs."""
import hashlib
import json
from pathlib import Path

from matrix import matrix

ROOT = Path(__file__).resolve().parents[1]
PROJECT = 'ocbench-bc-mysteries-1-4'
ENTITY = 'yzchhr'
API_URL = 'https://api.forge.coreweave.com'
APP_URL = f'https://forge.coreweave.com/wandb/{ENTITY}/{PROJECT}'
GITHUB = 'https://github.com/mattias32199/flow-matching-policies-for-BC'
BRANCH = 'reproduce-ocbench-mysteries-1-4'
CONFIG_KEYS = {
    'run', 'env', 'mode', 'features', 'seed', 'data_seed', 'episodes', 'val_episodes',
    'collection_cache', 'demo_prefetch', 'demo_producer_workers', 'demo_cache_limit_bytes', 'parallel', 'steps', 'batch_size', 'width', 'depth', 'log_interval', 'eval_interval',
    'eval_episodes', 'save_interval', 'keep_checkpoints', 'solver_iteration_warnings',
    'upstream_commit', 'code_commit', 'training_code_sha256',
}
FIELDS = {
    'metrics.jsonl': {'step', 'seconds', 'collection', 'training/flow_loss', 'training/action_mse',
                      'validation/flow_loss', 'validation/action_mse'},
    'eval.jsonl': {'step', 'success', 'failure', 'healthy', 'episode.final_reward',
                   'episode.return', 'episode.length', 'episode.duration', 'mean_steps',
                   'mean_success_steps', 'throughput', 'success_weighted_speed'},
    'collections.jsonl': {'index', 'collection_idx', 'collection_time', 'cached',
                         'cache_hit', 'cache_stored', 'cache_key', 'cache_entry_bytes', 'source_collection_time',
                         'train/transitions', 'train/success_rate', 'train/avg_episode_length',
                         'train/avg_return', 'val/transitions', 'val/success_rate',
                         'val/avg_episode_length', 'val/avg_return'},
    'dataset.jsonl': {'train_episodes', 'val_episodes', 'train_transitions', 'val_transitions'},
}


def read_records(path):
    records = []
    if path.exists():
        content = path.read_text()
        lines = content.splitlines()
        for index, line in enumerate(lines):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                if index == len(lines) - 1 and not content.endswith('\n'):
                    break  # Only an unfinished last record may be deferred.
                raise
    return records


def clean_config(config):
    return {key: value for key, value in config.items() if key in CONFIG_KEYS}


def run_id(config):
    # Config identity prevents accidental merging of pilots, smoke tests and full runs.
    digest = hashlib.sha256(json.dumps(clean_config(config), sort_keys=True).encode()).hexdigest()
    return 'ocb' + digest[:17]


def read_mapping(root=ROOT):
    path = root / '.local' / 'wandb-links.json'
    return json.loads(path.read_text()) if path.exists() else {}


def export_catalog(root=ROOT):
    target = root / 'results' / 'runs'
    target.mkdir(parents=True, exist_ok=True)
    mapping = read_mapping(root)
    marker = root / '.local' / 'wandb-project.json'
    verified = json.loads(marker.read_text()) if marker.exists() else {}
    connected = verified.get('url') == APP_URL
    tracking = dict(status='connected' if connected else 'credentials_pending',
                    project_url=APP_URL,
                    catalog_url=f'{APP_URL}/runs/ocbench-catalog-v1' if connected else None)
    (root / 'results' / 'tracking.json').write_text(json.dumps(tracking, indent=2) + '\n')
    entries = {job['run']: dict(name=job['run'], status='queued', config=clean_config(job)) for job in matrix()}
    queue_path = root / '.local' / 'queue-status.json'
    queue = json.loads(queue_path.read_text()) if queue_path.exists() else {}
    active = set(queue.get('active', {}).values())
    failed = set(queue.get('failures', []))
    for config_path in sorted((root / 'runs').glob('*/config.json')):
        local = config_path.parent
        config = clean_config(json.loads(config_path.read_text()))
        name = config['run']
        records = {filename: [{k: v for k, v in record.items() if k in allowed}
                              for record in read_records(local / filename)]
                   for filename, allowed in FIELDS.items()}
        if (local / 'DONE').exists():
            status = 'completed'
        elif name in active:
            status = 'training' if records['metrics.jsonl'] else 'collecting'
        elif name in failed or name.startswith(('pilot-', 'smoke-')):
            status = 'incomplete'
        else:
            status = 'inactive'
        entries[name] = dict(name=name, status=status, config=config, records=records)
    for name, entry in entries.items():
        directory = target / name
        directory.mkdir(exist_ok=True)
        metadata = mapping.get(name, {})
        url = metadata.get('url') if metadata.get('id') == run_id(entry['config']) else None
        config = entry['config']
        (directory / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        for filename in FIELDS:
            records = entry.get('records', {}).get(filename, [])
            (directory / filename).write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in records))
        evaluations = entry.get('records', {}).get('eval.jsonl', [])
        lines = [f'# {name}', '', f"Status: **{entry['status']}**.", '',
                 '[Results](../../../RESULTS.md) · [Run catalog](../README.md)', '',
                 f"W&B: [training charts and evaluations]({url})" if url else 'W&B: awaiting authenticated upload (or run has not started).', '',
                 '[Exact configuration](config.json) · [Training and validation log](metrics.jsonl) · '
                 '[Evaluation log](eval.jsonl) · [Collection log](collections.jsonl) · [Retained dataset counts](dataset.jsonl)', '',
                 'These JSONL records are the complete portable scalar logs, not sampled chart exports. '
                 'Machine stdout/stderr, credentials, data and checkpoints remain local.', '',
                 '## Evaluations', '',
                 '| Gradient update | Success | Episodes | Record | W&B |',
                 '| ---: | ---: | ---: | --- | --- |']
        for index, evaluation in enumerate(evaluations, 1):
            step = evaluation['step']
            lines.append(f"| {step} | {evaluation.get('success', '—')} | {config.get('eval_episodes', '—')} | "
                         f"[JSONL line {index}](eval.jsonl#L{index}) | "
                         f"{'[Evaluation charts](' + url + ')' if url else 'pending upload'} |")
        if not evaluations:
            lines += ['', 'No evaluation record exists yet; no success rate is inferred.']
        (directory / 'README.md').write_text('\n'.join(lines) + '\n')
        entry['wandb_url'] = url
        entry['wandb_id'] = metadata.get('id') if url else None
        entry.pop('records', None)
    lines = ['# Run catalog', '', '[Results](../../RESULTS.md) · [W&B project](' + APP_URL + ')', '',
             ('[W&B catalog table](' + tracking['catalog_url'] + ')' if connected
              else 'Remote project connection has not been verified.'), '',
             'Full runs, queued conditions, and diagnostic runs are separate entries. '
             'Evaluation links identify the exact gradient update and JSONL line.', '',
             '| Run | Status | Exact records | W&B |', '| --- | --- | --- | --- |']
    for name, entry in sorted(entries.items()):
        url = entry['wandb_url']
        lines.append(f"| {name} | {entry['status']} | [Config and logs]({name}/README.md) | "
                     f"{'[Charts](' + url + ')' if url else 'not uploaded'} |")
    (target / 'README.md').write_text('\n'.join(lines) + '\n')
    (root / 'results' / 'catalog.json').write_text(json.dumps(list(entries.values()), indent=2) + '\n')
    return entries


if __name__ == '__main__':
    print(f'Exported {len(export_catalog())} catalog entries.')
