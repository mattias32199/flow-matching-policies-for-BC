"""Pin complete local checkpoints before releasing an allocation; never submit jobs."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def leaves(tree):
    if isinstance(tree, dict):
        for value in tree.values():
            yield from leaves(value)
    elif isinstance(tree, (list, tuple)):
        for value in tree:
            yield from leaves(value)
    else:
        yield np.asarray(tree)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare(name, root=ROOT):
    if Path(name).name != name:
        raise ValueError('Expected a run name, not a path.')
    source = root / 'runs' / name
    config = json.loads((source / 'config.json').read_text())
    if config['mode'] not in ('streaming', 'fixed'):
        raise ValueError('Unsupported data mode for continuation.')
    candidates = sorted((source / 'checkpoints').glob('params_*.pkl'),
                        key=lambda p: int(p.stem.split('_')[-1]), reverse=True)
    failures = []
    for checkpoint in candidates:
        step = int(checkpoint.stem.split('_')[-1])
        target = root / '.local/migration-snapshots' / name / str(step)
        target.mkdir(parents=True, exist_ok=True, mode=0o700)
        pinned = target / checkpoint.name
        try:
            if not pinned.exists():
                os.link(checkpoint, pinned)  # Survives source checkpoint retention.
            # Only trusted checkpoints written by our own runner are loaded.
            with pinned.open('rb') as file:
                state = pickle.load(file)['agent']
            numeric = 0
            for array in leaves(state):
                if array.dtype.kind in 'biufc':
                    if not np.isfinite(array).all():
                        raise ValueError('Nonfinite checkpoint state.')
                    numeric += array.size
            if int(state['network']['step']) != step + 1:
                raise ValueError('Checkpoint network step does not match filename.')
            del state
            gc.collect()
        except (OSError, EOFError, pickle.UnpicklingError, ValueError, KeyError) as error:
            failures.append(dict(checkpoint=checkpoint.name, error=type(error).__name__))
            pinned.unlink(missing_ok=True)
            continue
        for filename in ['config.json', 'preprocessing.npz', 'metrics.jsonl',
                         'eval.jsonl', 'collections.jsonl', 'dataset.jsonl']:
            if (source / filename).exists():
                shutil.copy2(source / filename, target / filename)
        records = []
        for line in (target / 'collections.jsonl').read_text().splitlines():
            try:
                records.append(json.loads(line)['index'])
            except (ValueError, KeyError):
                continue
        # A disjoint collection-index namespace prevents later overlap even if the
        # original allocation keeps running for a while after this snapshot.
        next_index = max(2048, max(records, default=0) + 1) if config['mode'] == 'streaming' else 0
        if next_index >= 4096:
            raise ValueError('No collection-index space remains for continuation.')
        manifest = dict(source_run=name, checkpoint=checkpoint.name, step=step,
                        checkpoint_bytes=pinned.stat().st_size,
                        checkpoint_sha256=sha256(pinned), numeric_elements=numeric,
                        fresh_stream_index=next_index, prepared_at=time.time(),
                        continuation_kind=('agent_optimizer_rng_fresh_stream' if config['mode'] == 'streaming'
                                           else 'agent_optimizer_rng_fixed_dataset_sampler_reset'),
                        exact_stream_resume=False, validation_reset=config['mode'] == 'streaming',
                        rejected_candidates=failures)
        temporary = target / 'manifest.json.tmp'
        temporary.write_text(json.dumps(manifest, indent=2) + '\n')
        temporary.replace(target / 'manifest.json')
        return target, manifest
    raise RuntimeError(f'No complete checkpoint for {name}: {failures}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', nargs='+')
    args = parser.parse_args()
    os.umask(0o077)
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    for name in args.runs:
        target, manifest = prepare(name)
        print(json.dumps(dict(bundle=str(target.relative_to(ROOT)), **manifest)), flush=True)
