"""Isolated full-size FBC timing, compiler FLOP estimates, and JAX traces.

Synthetic batches isolate training computation; they do not validate convergence.
No checkpoints or production run state are modified.
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ocbench' / 'impls'))
import jax
import jax.numpy as jnp
import numpy as np
from agents.fbc import FBCAgent, get_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=128)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = jax.devices()[0]
    if device.platform != 'gpu' or 'A6000' not in device.device_kind:
        raise RuntimeError(f'Requires an allocated A6000 GPU, got {device.device_kind}')
    emit_path = args.output / 'training.jsonl'

    def emit(row):
        print(json.dumps(row), flush=True)
        with emit_path.open('a') as f:
            f.write(json.dumps(row) + '\n')

    rng = np.random.default_rng(123)
    host = dict(observations=rng.normal(size=(1024, 37)).astype(np.float32),
                actions=rng.normal(size=(1024, 175)).astype(np.float32))
    batch = jax.device_put(host)
    emit(dict(event='environment', device=device.device_kind, jax=jax.__version__,
              batch_size=1024, updates=args.updates, repeats=args.repeats,
              note='Dedicated allocated GPU; node may share CPU and IO. FLOPs are compiler estimates, not hardware counters.'))
    cases = [('fp32-original-host', 'float32', 'default', 'original', 1),
             ('fp32-default-device', 'float32', 'default', 'device', 1),
             ('fp32-highest-device', 'float32', 'highest', 'device', 1),
             ('fp32-tf32-device', 'float32', 'tensorfloat32', 'device', 1),
             ('fp32-default-host', 'float32', 'default', 'host', 1),
             ('fp32-default-scan8', 'float32', 'default', 'device', 8),
             ('bf16-device', 'bfloat16', 'default', 'device', 1),
             ('bf16-scan8', 'bfloat16', 'default', 'device', 8)]
    for name, dtype, precision, feed, group in cases:
        jax.config.update('jax_default_matmul_precision', precision)
        config = get_config()
        config.compute_dtype = dtype
        agent = FBCAgent.create(0, host['observations'][:1], host['actions'][:1], config)
        if group == 1:
            update = jax.jit(lambda a, b: a.update(b), donate_argnums=() if feed == 'original' else (0,))
        else:
            def grouped(a, b):
                def body(state, _):
                    return state.update(b)
                return jax.lax.scan(body, a, None, length=group)
            update = jax.jit(grouped, donate_argnums=(0,))
        started = time.monotonic()
        compiled = update.lower(agent, batch).compile()
        compile_seconds = time.monotonic() - started
        # Some XLA backends omit library GEMMs from cost_analysis.
        dense_weights = sum(x.size for x in jax.tree_util.tree_leaves(agent.network.params) if x.ndim == 2)
        dense_flops = float(6 * 1024 * dense_weights)
        analysis = compiled.cost_analysis()
        if isinstance(analysis, list):
            analysis = analysis[0]
        flops = float(analysis.get('flops', 0)) / group
        for _ in range(8):
            agent, info = compiled(agent, batch)
        jax.block_until_ready((agent, info))
        rates = []
        for repeat in range(args.repeats):
            started = time.monotonic()
            for _ in range(args.updates // group):
                agent, info = compiled(agent, jax.device_put(host) if feed in ('host', 'original') else batch)
            jax.block_until_ready((agent, info))
            elapsed = time.monotonic() - started
            rate = (args.updates // group * group) / elapsed
            rates.append(rate)
            emit(dict(event='timing', case=name, repeat=repeat, updates_per_second=rate,
                      seconds=elapsed, compiler_flops_per_update=flops,
                      dense_matmul_estimated_flops_per_update=dense_flops,
                      compiler_estimated_tflops_per_second=flops * rate / 1e12,
                      finite_loss=bool(np.isfinite(np.asarray(info['actor/actor_loss'])).all())))
        # Warm execution trace, outside timed regions; stays local because traces contain paths.
        if name in ('fp32-original-host', 'fp32-default-host', 'fp32-default-scan8', 'bf16-scan8'):
            with jax.profiler.trace(str(args.output / 'traces' / name), create_perfetto_link=False):
                for _ in range(4):
                    agent, info = compiled(agent, jax.device_put(host) if feed in ('host', 'original') else batch)
                jax.block_until_ready((agent, info))
        emit(dict(event='summary', case=name, median_updates_per_second=float(np.median(rates)),
                  compile_seconds=compile_seconds, compiler_flops_per_update=flops,
                      dense_matmul_estimated_flops_per_update=dense_flops,
                  all_losses_finite=bool(np.isfinite(np.asarray(info['actor/actor_loss'])).all())))
        del agent, info, compiled, update
        jax.clear_caches()
        gc.collect()
    (args.output / 'TRAINING_DONE').write_text('All training cases completed.\n')


if __name__ == '__main__':
    main()
