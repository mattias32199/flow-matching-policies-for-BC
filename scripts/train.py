"""State-based FBC reproductions using the pinned upstream agent and simulator."""
import argparse
import hashlib
import faulthandler
import signal
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ocbench' / 'impls'))

import jax
import numpy as np
from agents.fbc import FBCAgent, get_config
from envs.env_utils import ActionNormalizer, postprocess_datasets
from envs.streaming_data import generate_mjwarp_datasets
from utils.datasets import Dataset, filter_success_trajectories
from utils.evaluation_mjwarp import evaluate_mjwarp
from utils.flax_utils import save_agent
from ocbench.dataset_utils import DATASET_KEYS


class Features:
    """Fit on initial training data only; apply identically at train and test."""
    def __init__(self, observations, mode):
        self.mode = mode
        self.frequencies = {'fourier1': (1.,), 'fourier4': (1., 2., 4., 8.)}.get(mode, ())
        dim = observations.shape[-1]
        self.center = np.zeros(dim, np.float32)
        self.scale = np.ones(dim, np.float32)
        if mode != 'handcrafted':
            # Streaming statistics limit peak memory with large fixed datasets.
            total = np.zeros(dim, np.float64)
            square = np.zeros(dim, np.float64)
            for block in np.array_split(observations, max(1, len(observations) // 100000)):
                total += block.sum(axis=0, dtype=np.float64)
                square += np.einsum('ij,ij->j', block.astype(np.float64), block.astype(np.float64))
            self.center = (total / len(observations)).astype(np.float32)
            variance = np.maximum(square / len(observations) - self.center.astype(np.float64)**2, 0)
            self.scale = (1 / np.maximum(np.sqrt(variance), 1e-3)).astype(np.float32)
            # Upstream block observation: 19 proprio dimensions, then 9 per cube.
            xyz = list(range(12, 15))
            for start in range(19, dim, 9):
                xyz.extend(range(start, start + 3))
            self.xyz = xyz
            multiplier = {'standardized': 1., 'xyz0.1': .1, 'xyz10': 10.,
                          'fourier1': 1., 'fourier4': 1.}[mode]
            self.scale[xyz] *= multiplier

    def __call__(self, observations):
        standardized = (observations - self.center) * self.scale
        if not self.frequencies:
            return standardized
        # Preserve the complete standardized state; append fixed coordinate features.
        extra = []
        for frequency in self.frequencies:
            phase = np.float32(2 * np.pi * frequency) * standardized[..., self.xyz]
            extra.extend((np.sin(phase), np.cos(phase)))
        return np.concatenate([standardized, *extra], axis=-1)


class EvalAgent:
    def __init__(self, agent, features):
        self.agent, self.features = agent, features

    def sample_actions(self, observations, seed=None):
        return self.agent.sample_actions(self.features(observations), seed=seed)


def saved_preprocessing(path, mode):
    """Restore observation and action transforms without refitting any statistic."""
    with np.load(path, allow_pickle=False) as saved:
        features = Features.__new__(Features)
        features.mode = mode
        features.center, features.scale = saved['center'].copy(), saved['scale'].copy()
        features.frequencies = tuple(saved['fourier_frequencies']) if 'fourier_frequencies' in saved else ()
        features.xyz = list(range(12, 15))
        for start in range(19, len(features.center), 9):
            features.xyz.extend(range(start, start + 3))
        normalizer = ActionNormalizer(saved['q_low'], saved['q_high'], saved['clip'], saved['gripper_dims'])
    return features, normalizer


def evaluation_record(step, metrics):
    # Upstream mean_success_steps is undefined when every rollout fails.
    return {'step': step, **{k: float(v) if np.isfinite(v) else None for k, v in metrics.items()}}


def emit(path, record):
    with path.open('a') as file:
        file.write(json.dumps(record, allow_nan=False) + '\n')
    print(json.dumps(record), flush=True)


def cache_request(args, index, episodes, val_episodes):
    from collection_cache import CollectionCache
    from demo_identity import description
    fourier = args.features in ('fourier1', 'fourier4')
    kwargs = {'consumers': ('fourier1', 'fourier4')} if fourier else {}
    cache = CollectionCache(ROOT / 'data' / 'shared-demos', **kwargs)
    identity = description(args.env, episodes, val_episodes, args.data_seed, index, args.parallel)
    if fourier:
        identity['consumer_group'] = 'fourier-pair-v1'
    return cache, identity


def collect(args, index, episodes, val_episodes):
    print(f'COLLECTION_START index={index} episodes={episodes}', flush=True)
    def generate():
        train, val, metrics = generate_mjwarp_datasets(
            args.env, episodes, val_episodes, args.data_seed, index,
            max_parallel_episodes=args.parallel,
        )
        # Simulator state is not needed for BC.
        train = {k: v for k, v in train.items() if k in DATASET_KEYS}
        val = {k: v for k, v in val.items() if k in DATASET_KEYS} if val is not None else None
        return train, val, metrics
    if args.mode == 'streaming' and args.collection_cache:
        from prefetch_control import ensure_producer
        fourier = args.features in ('fourier1', 'fourier4')
        cache, identity = cache_request(args, index, episodes, val_episodes)
        if not cache.contains(identity) and not fourier:
            ensure_producer(args, needed_index=index)
        return cache.get(identity, generate, consumer=args.features)
    return generate()


def dataset(raw, normalizer):
    processed, _ = postprocess_datasets(raw, None, normalizer)
    result = Dataset.create(**filter_success_trajectories(processed))
    result.action_chunk_length = 25
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='block-double-task1-v0')
    parser.add_argument('--mode', choices=['fixed', 'streaming'], default='streaming')
    parser.add_argument('--features', choices=['handcrafted', 'standardized', 'xyz0.1', 'xyz10', 'fourier1', 'fourier4'], default='handcrafted')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data-seed', type=int)
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--val-episodes', type=int, default=128)
    parser.add_argument('--parallel', type=int, default=2500)
    parser.add_argument('--collection-cache', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--loader-prefetch', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--steps', type=int, default=2000000)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--width', type=int, default=4096)
    parser.add_argument('--depth', type=int, default=8)
    parser.add_argument('--log-interval', type=int, default=1000)
    parser.add_argument('--eval-interval', type=int, default=50000)
    parser.add_argument('--eval-episodes', type=int, default=500)
    parser.add_argument('--save-interval', type=int, default=100000)
    parser.add_argument('--keep-checkpoints', type=int, default=2)
    parser.add_argument('--solver-iteration-warnings', action='store_true')
    parser.add_argument('--run', required=True)
    parser.add_argument('--resume-bundle', type=Path)
    args = parser.parse_args()
    faulthandler.register(signal.SIGUSR1)
    from simulation import configure_iteration_warnings
    configure_iteration_warnings(args.solver_iteration_warnings)
    if args.data_seed is None:
        args.data_seed = args.seed
    resume = None
    if args.resume_bundle:
        from migration import read_bundle
        resume = read_bundle(args.resume_bundle, args)
    run = ROOT / 'runs' / args.run
    run.mkdir(parents=True, exist_ok=True)
    (run / 'checkpoints').mkdir(exist_ok=True)
    if (run / 'metrics.jsonl').exists():
        raise RuntimeError('Refusing to overwrite an existing run; use a new run name.')
    config_record = {k: v for k, v in vars(args).items() if k != 'resume_bundle'} | {'upstream_commit': subprocess.check_output(
        ['git', '-C', str(ROOT / 'ocbench'), 'rev-parse', 'HEAD'], text=True).strip()}
    if resume:
        config_record.update(resume_source_run=resume['source_run'], resume_step=resume['step'],
            resume_checkpoint_sha256=resume['checkpoint_sha256'],
            resume_stream_index=resume['fresh_stream_index'],
            resume_kind=resume['continuation_kind'], validation_reset=True)
    config_record['fourier_frequencies'] = list({'fourier1': (1,), 'fourier4': (1, 2, 4, 8)}.get(args.features, ()))
    producer_config = ROOT / '.local/demo-producer-config.json'
    producer_settings = json.loads(producer_config.read_text()) if producer_config.exists() else {}
    config_record['demo_prefetch'] = bool(args.collection_cache and args.mode == 'streaming'
        and not config_record['fourier_frequencies']
        and producer_settings.get('enabled') and args.data_seed in producer_settings.get('seeds', []))
    config_record['demo_producer_workers'] = int(producer_settings.get('workers', 2)) if config_record['demo_prefetch'] else 0
    config_record['demo_cache_limit_bytes'] = 80_000_000_000 if args.collection_cache and args.mode == 'streaming' else 0
    config_record['code_commit'] = subprocess.check_output(
        ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    config_record['training_code_sha256'] = hashlib.sha256(
        Path(__file__).read_bytes() + (ROOT / 'scripts' / 'simulation.py').read_bytes()
        + (ROOT / 'scripts' / 'collection_cache.py').read_bytes() + (ROOT / 'scripts' / 'demo_identity.py').read_bytes()
        + (ROOT / 'scripts' / 'prefetch_control.py').read_bytes()
        + (ROOT / 'scripts' / 'private_slurm.py').read_bytes() + (ROOT / 'scripts' / 'demo_producer.py').read_bytes()
        + (ROOT / 'scripts' / 'migration.py').read_bytes()
        + (ROOT / 'scripts' / 'loader_prefetch.py').read_bytes()).hexdigest()
    (run / 'config.json').write_text(json.dumps(config_record, indent=2))
    print(f'JAX_DEVICES {jax.devices()}', flush=True)
    if not any(device.platform == 'gpu' for device in jax.devices()):
        raise RuntimeError('GPU is required for these experiments.')
    # Separate sampling, diagnostic, and evaluation RNGs for paired comparisons.
    sampler = np.random.default_rng(args.seed)
    diagnostic = np.random.default_rng(args.seed + 10000)
    np.random.seed(args.seed)
    start = time.monotonic()
    index = resume['fresh_stream_index'] if resume else 0
    data_dir = ROOT / 'data' / f'{args.env}-n{args.episodes}-d{args.data_seed}'
    train_file, val_file = data_dir / 'train.npz', data_dir / 'val.npz'
    if args.mode == 'fixed' and train_file.exists() and val_file.exists():
        raw_train, raw_val = dict(np.load(train_file)), dict(np.load(val_file))
        collection_metrics = {'cached': True}
    else:
        raw_train, raw_val, collection_metrics = collect(args, index, args.episodes, args.val_episodes)
        if args.mode == 'fixed':
            data_dir.mkdir(parents=True, exist_ok=True)
            np.savez(train_file, **raw_train)
            np.savez(val_file, **raw_val)
    emit(run / 'collections.jsonl', {'index': index, **collection_metrics})
    if resume:
        features, normalizer = saved_preprocessing(args.resume_bundle / 'preprocessing.npz', args.features)
    else:
        normalizer = ActionNormalizer.fit(raw_train['actions'], 1., 5.)
    train, val = dataset(raw_train, normalizer), dataset(raw_val, normalizer)
    del raw_train, raw_val
    emit(run / 'dataset.jsonl', {'train_episodes': int(train['terminals'].sum()),
                               'val_episodes': int(val['terminals'].sum()),
                               'train_transitions': train.size, 'val_transitions': val.size})
    if not resume:
        features = Features(train['observations'], args.features)
    np.savez(run / 'preprocessing.npz', center=features.center, scale=features.scale,
             fourier_frequencies=np.asarray(features.frequencies, dtype=np.float32),
             q_low=normalizer.q_low, q_high=normalizer.q_high, clip=normalizer.clip,
             gripper_dims=np.flatnonzero(normalizer.gripper_dims))
    config = get_config()
    config.actor_hidden_dims = (args.width,) * args.depth
    config.batch_size = args.batch_size
    example = train.sample(1, keys=('observations', 'actions'))
    agent = FBCAgent.create(args.seed, features(example['observations']), example['actions'], config)
    if resume:
        from migration import restore_state
        agent = restore_state(agent, args.resume_bundle, resume)
        emit(run / 'continuation.jsonl', {key: config_record[key] for key in
             ['resume_source_run', 'resume_step', 'resume_checkpoint_sha256',
              'resume_stream_index', 'resume_kind', 'validation_reset']})
    print(f'PARAMETERS {sum(x.size for x in jax.tree_util.tree_leaves(agent.network.params))}', flush=True)
    permutation = sampler.permutation(train.size) if args.mode == 'streaming' else None
    cursor = 0
    prefetch = None
    if args.mode == 'streaming' and args.collection_cache and args.loader_prefetch:
        from loader_prefetch import CachedCollectionPrefetch
        prefetch = CachedCollectionPrefetch()
    first_step = resume['step'] + 1 if resume else 1
    for step in range(first_step, args.steps + 1):
        if args.mode == 'streaming':
            if cursor + args.batch_size > train.size:
                index += 1
                del train
                loaded = prefetch.take(index) if prefetch else None
                raw_train, _, metrics = loaded if loaded is not None else collect(args, index, args.episodes, 0)
                del loaded
                train = dataset(raw_train, normalizer)
                del raw_train
                permutation = sampler.permutation(train.size)
                cursor = 0
                emit(run / 'collections.jsonl', {'index': index, **metrics})
            idxs = permutation[cursor:cursor + args.batch_size]
            if len(idxs) != args.batch_size:
                raise RuntimeError('Collection has fewer transitions than a batch.')
            cursor += args.batch_size
            if prefetch and step % 100 == 0:
                cache, identity = cache_request(args, index + 1, args.episodes, 0)
                prefetch.schedule(index + 1, cache, identity, args.features)
        else:
            idxs = sampler.integers(train.size, size=args.batch_size)
        batch = train.sample(args.batch_size, idxs=idxs, keys=('observations', 'actions'))
        batch['observations'] = features(batch['observations'])
        agent, info = agent.update(batch)
        if step == first_step or step % args.log_interval == 0:
            loss = float(info['actor/actor_loss'])
            if not np.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss at step {step}')
            record = {'step': step, 'seconds': time.monotonic() - start, 'collection': index,
                      'training/flow_loss': loss}
            for name, source in [('training', train), ('validation', val)]:
                ids = diagnostic.integers(source.size, size=args.batch_size)
                metric_batch = source.sample(args.batch_size, idxs=ids, keys=('observations', 'actions'))
                metric_batch['observations'] = features(metric_batch['observations'])
                rng = jax.random.PRNGKey(int(diagnostic.integers(2**31)))
                _, flow_info = agent.total_loss(metric_batch, grad_params=None, rng=rng)
                metrics = agent.compute_metrics(metric_batch, rng=rng)
                record[f'{name}/flow_loss'] = float(flow_info['actor/actor_loss'])
                record[f'{name}/action_mse'] = float(metrics['actor/mse'])
            emit(run / 'metrics.jsonl', record)
        if step % args.eval_interval == 0 or step == args.steps:
            # Same rollout reset/noise sequence at a given step across conditions.
            np.random.seed(args.seed * 100000 + step)
            metrics, _, _ = evaluate_mjwarp(
                EvalAgent(agent, features), args.env, config=agent.config,
                action_normalizer=normalizer, num_eval_episodes=args.eval_episodes,
                action_chunk_length=25,
            )
            emit(run / 'eval.jsonl', evaluation_record(step, metrics))
        if step % args.save_interval == 0 or step == args.steps:
            save_agent(agent, str(run / 'checkpoints'), step)
            saved = sorted((run / 'checkpoints').glob('params_*.pkl'),
                           key=lambda path: int(path.stem.split('_')[-1]))
            for old in saved[:-max(1, args.keep_checkpoints)]:
                old.unlink()
    if prefetch:
        prefetch.close()
    (run / 'DONE').write_text('Training and final evaluation completed.\n')


if __name__ == '__main__':
    main()
