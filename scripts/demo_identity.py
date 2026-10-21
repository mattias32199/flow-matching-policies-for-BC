"""Portable generator identity independent of feature transformations."""
import functools
import hashlib
import importlib.metadata
from pathlib import Path
import subprocess
ROOT = Path(__file__).resolve().parents[1]


@functools.lru_cache(maxsize=1)
def generator_identity():
    versions = {}
    for name in ('mujoco', 'mujoco-warp', 'warp-lang', 'jax', 'numpy'):
        versions[name] = importlib.metadata.version(name)
    return dict(schema=1, upstream_commit=subprocess.check_output(
        ['git', '-C', str(ROOT / 'ocbench'), 'rev-parse', 'HEAD'], text=True).strip(),
        versions=versions, generator_sha256=hashlib.sha256(
            (ROOT / 'ocbench/impls/envs/streaming_data.py').read_bytes()
            + (ROOT / 'scripts/simulation.py').read_bytes()).hexdigest())


def description(env, episodes, val_episodes, seed, index, parallel):
    return dict(generator_identity(), env=env, episodes=episodes, val_episodes=val_episodes,
                data_seed=seed, collection_index=index, parallel=parallel)
