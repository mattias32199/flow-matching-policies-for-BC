"""Explicit, inexact stream continuation with exact agent-state restoration."""
import json
from pathlib import Path
import pickle
from prepare_migration import sha256


def read_bundle(path, args):
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    source = json.loads((path / 'config.json').read_text())
    for key in ['env', 'mode', 'features', 'seed', 'data_seed', 'episodes',
                'batch_size', 'width', 'depth', 'val_episodes']:
        if getattr(args, key) != source[key]:
            raise ValueError(f'Continuation must preserve source {key}.')
    if args.mode != 'streaming' or args.steps <= manifest['step']:
        raise ValueError('Expected a streaming continuation with remaining updates.')
    if not 0 <= manifest['fresh_stream_index'] < 4096:
        raise ValueError('Invalid continuation collection index.')
    if sha256(path / manifest['checkpoint']) != manifest['checkpoint_sha256']:
        raise ValueError('Pinned checkpoint checksum mismatch.')
    return manifest


def restore_state(agent, bundle, manifest):
    import flax.serialization
    with (Path(bundle) / manifest['checkpoint']).open('rb') as file:
        state = pickle.load(file)['agent']
    restored = flax.serialization.from_state_dict(agent, state)
    if int(restored.network.step) != manifest['step'] + 1:
        raise ValueError('Restored optimizer/network step mismatch.')
    return restored
