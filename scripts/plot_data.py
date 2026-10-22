"""Lossless, pickle-free per-experiment plot archives; keep all restart branches."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from catalog import CONFIG_KEYS, FIELDS, read_records

ROOT = Path(__file__).resolve().parents[1]


def load_run(root, name):
    directory = root / 'runs' / name
    config = json.loads((directory / 'config.json').read_text())
    return {'config': {k: v for k, v in config.items() if k in CONFIG_KEYS},
            'streams': {stream: [{k: v for k, v in row.items() if k in allowed}
                         for row in read_records(directory / stream)]
                        for stream, allowed in FIELDS.items()},
            'completed': (directory / 'DONE').exists()}


def check_complete(root, plan):
    for source, selected in {plan['m1']: plan['m1'], **plan['m4']}.items():
        run = load_run(root, selected)
        metrics = run['streams']['metrics.jsonl']; evaluations = run['streams']['eval.jsonl']
        if (not run['completed'] or not metrics or metrics[-1]['step'] != 2_000_000
                or not evaluations or evaluations[-1]['step'] != 2_000_000):
            raise RuntimeError(f'Final publication blocked: {source} / {selected} is not complete at 2M with final evaluation.')
        for stream, rows in run['streams'].items():
            for row in rows:
                for key, value in row.items():
                    if isinstance(value, (float, int)) and not np.isfinite(value):
                        raise ValueError(f'Nonfinite public record: {selected}/{stream}/{key}')


def export(root=ROOT, plan=None, output=None, require_complete=False):
    root = Path(root)
    plan = plan or json.loads((root / '.local/final-results-plan.json').read_text())
    if require_complete:
        check_complete(root, plan)
    output = Path(output or root / 'results/plot-data')
    output.mkdir(parents=True, exist_ok=True)
    index = {'schema': 1, 'description': 'Unsmoothed scalar measurements, not robot demonstrations.', 'runs': {}}
    for source, selected in {plan['m1']: plan['m1'], **plan['m4']}.items():
        names = sorted(p.name for p in (root / 'runs').iterdir()
                       if p.name == source or p.name.startswith(source + '-cont'))
        runs = {name: load_run(root, name) for name in names if (root/'runs'/name/'config.json').exists()}
        if source not in runs or selected not in runs:
            raise ValueError(f'Missing source/selected segment: {source}/{selected}')
        metadata = {'schema': 1, 'experiment': source, 'selected_segment': selected,
                    'smoothing': 'None in archives. Figures use 50K-update means of the unified selected history.',
                    'branch_policy': 'Follow selected checkpoint ancestry; truncate each parent at its child resume_step; never merge validation sets.',
                    'segments': {name: {'config': run['config'], 'completed': run['completed']}
                                 for name, run in runs.items()}}
        arrays = {'manifest_json': np.asarray(json.dumps(metadata, sort_keys=True))}
        for i, (name, run) in enumerate(runs.items()):
            metadata['segments'][name]['prefix'] = f's{i}'
            for stream, rows in run['streams'].items():
                prefix = f's{i}__{stream.removesuffix(".jsonl")}'
                # Complete records preserve missing/null/string values exactly, without pickle.
                arrays[prefix+'__jsonl'] = np.asarray([json.dumps(row, allow_nan=False, sort_keys=True)
                                                      for row in rows], dtype=np.str_)
                # Numeric columns make direct NumPy plotting convenient; JSONL is authoritative for nulls.
                for key in sorted({k for row in rows for k in row}):
                    values = [row.get(key) for row in rows]
                    if all(v is None or isinstance(v, (int, float, bool)) for v in values):
                        arrays[prefix+'__'+key.replace('/', '__')] = np.asarray(
                            [np.nan if v is None else v for v in values], dtype=np.float64)
        metadata['unified_prefix'] = 'history'
        metadata['unified_policy'] = 'One ordered selected training history; checkpoint ancestry resolves overlapping steps. Validation data may change at recovery.'
        for stream in ['metrics.jsonl', 'eval.jsonl']:
            rows = unified(metadata, runs, stream)
            prefix = 'history__' + stream.removesuffix('.jsonl')
            arrays[prefix+'__jsonl'] = np.asarray([json.dumps(row, allow_nan=False, sort_keys=True) for row in rows], dtype=np.str_)
            for key in sorted({k for row in rows for k in row}):
                values = [row.get(key) for row in rows]
                if all(v is None or isinstance(v, (int, float, bool)) for v in values):
                    arrays[prefix+'__'+key.replace('/', '__')] = np.asarray([np.nan if v is None else v for v in values], dtype=np.float64)
        arrays['manifest_json'] = np.asarray(json.dumps(metadata, sort_keys=True))
        path = output / (source + '.npz')
        np.savez_compressed(path, **arrays)
        with np.load(path, allow_pickle=False) as saved:
            for key, value in arrays.items():
                np.testing.assert_array_equal(saved[key], value)
        index['runs'][source] = {'file': path.name, 'bytes': path.stat().st_size,
                                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                'selected_segment': selected, 'segments': list(runs)}
    (output/'manifest.json').write_text(json.dumps(index, indent=2)+'\n')
    return index


def load(path):
    with np.load(path, allow_pickle=False) as arrays:
        meta = json.loads(str(arrays['manifest_json']))
        runs = {}
        for name, info in meta['segments'].items():
            runs[name] = {'config': info['config'], 'completed': info['completed'],
                          'streams': {stream: [json.loads(str(row)) for row in
                            arrays[info['prefix']+'__'+stream.removesuffix('.jsonl')+'__jsonl']]
                            for stream in FIELDS}}
    return meta, runs


def branches(meta, runs, stream):
    def chain(name, seen):
        if name in seen:
            raise ValueError('Cycle in checkpoint ancestry.')
        config = runs[name]['config']
        parent = config.get('resume_source_run')
        if not parent:
            return [(name, runs[name]['streams'][stream])]
        if parent not in runs:
            raise ValueError('Missing ancestor segment in archive: ' + parent)
        boundary = config['resume_step']
        prior = [(segment, [row for row in rows if row.get('step', 0) <= boundary])
                 for segment, rows in chain(parent, seen | {name})]
        return prior + [(name, [row for row in runs[name]['streams'][stream]
                               if row.get('step', 0) > boundary])]
    return chain(meta['selected_segment'], set())


def unified(meta, runs, stream):
    """One measured history, following checkpoint ancestry and never inventing points."""
    rows = [row for _, records in branches(meta, runs, stream) for row in records]
    by_step = {row['step']: row for row in rows}
    return [by_step[step] for step in sorted(by_step)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    index = export(output=args.output, require_complete=args.require_complete)
    print(json.dumps(index, indent=2))
