# Recreate the experiments with OCBench

Start in this repository's root on branch `reproduce-ocbench-mysteries-1-4`.
OCBench is pinned at `ab6a2c2602b9ac2196e786aea7b7fab2fe779d98`.
Use a fresh checkout/output location and one GPU per invocation.

```bash
git submodule update --init --recursive
bash scripts/setup.sh
source scripts/env.sh
```

The pinned [native FBC configuration](ocbench/impls/agents/fbc.py) supplies batch 1024, eight 4096-wide residual layers, Adam learning rate 1e-4, float32 network computation and 10 flow steps. Both trainers use OCBench's agent, dataset utilities and simulator.

## Mystery 1: use the native trainer

Generate each fixed dataset with OCBench's own API. This small shell function only saves its output: the stock generation CLI always requests 10% validation episodes, whereas our runs request **128**. Training and validation arrays remain raw until the native trainer normalizes actions and filters successful trajectories.

```bash
make_fixed() {
  .venv/bin/python - "$@" <<'PY'
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, 'ocbench/impls')
from envs.streaming_data import generate_mjwarp_datasets
from ocbench.dataset_utils import DATASET_KEYS

env, episodes, seed, parallel, filename = sys.argv[1:]
p = Path(filename)
v = p.with_name(p.stem + '-val.npz')
assert not p.exists() and not v.exists(), 'Use new output filenames.'
p.parent.mkdir(parents=True, exist_ok=True)
train, val, _ = generate_mjwarp_datasets(
    env, collection_episodes=int(episodes), collection_val_episodes=128,
    root_seed=int(seed), collection_idx=0, max_parallel_episodes=int(parallel))
np.savez(p, **{k: x for k, x in train.items() if k in DATASET_KEYS})
np.savez(v, **{k: x for k, x in val.items() if k in DATASET_KEYS})
PY
}

# Arguments: environment, requested demos, data seed, parallel cap, output.
make_fixed block-double-task1-v0 10000 0 512  data/native-repro/double-10k-s0.npz
make_fixed block-double-task1-v0 10000 1 2500 data/native-repro/double-10k-s1.npz
make_fixed block-single-task1-v0 10000 0 2500 data/native-repro/single-10k-s0.npz
make_fixed block-single-task1-v0 50000 10000 2500 data/native-repro/single-50k-s0.npz
```

Each `native_fixed` call below is one independent 2M-update job. Run separately or submit separately to your scheduler; the displayed calls execute sequentially if pasted together.

```bash
native_fixed() {
  # Arguments: label, environment, training seed, dataset filename.
  .venv/bin/python ocbench/impls/main.py \
    --agent=ocbench/impls/agents/fbc.py \
    --run_group="$1" --env_name="$2" --seed="$3" --dataset_path="$4" \
    --streaming_data=0 --dataset_replace_interval=0 \
    --offline_steps=2000000 --online_steps=0 \
    --action_chunk_length=25 --success_only=1 \
    --action_norm_percentile=1 --action_norm_clip=5 \
    --log_interval=1000 --eval_interval=50000 --eval_episodes=500 \
    --save_interval=100000 --save_dir=.local/native-repro
}

native_fixed m1-double-10k-s0 block-double-task1-v0 0 data/native-repro/double-10k-s0.npz
native_fixed m1-double-10k-s1 block-double-task1-v0 1 data/native-repro/double-10k-s1.npz
native_fixed m1-single-10k-s0 block-single-task1-v0 0 data/native-repro/single-10k-s0.npz
native_fixed m1-single-50k-s0 block-single-task1-v0 0 data/native-repro/single-50k-s0.npz
```

These commands match the task, data counts/seeds, architecture, loss, action normalization and update budget. They **do not replay our exact numerical trajectory**: the native trainer draws minibatches/diagnostics differently and also evaluates at step 1. GPU simulation is not bitwise deterministic. Native saving retains all checkpoints; our wrapper retains two. Native CSV/W&B field names also differ.

## Mystery 4: what the native trainer can do

This is an exact runnable command for a **native streaming baseline**, not an exact recreation of our one-use stream:

```bash
.venv/bin/python ocbench/impls/main.py \
  --agent=ocbench/impls/agents/fbc.py \
  --run_group=m4-native-handcrafted --env_name=block-double-task1-v0 --seed=0 \
  --streaming_data=1 --collection_episodes=1000 --collection_val_episodes=128 \
  --collection_interval=1000 --collection_max_parallel_episodes=512 \
  --offline_steps=2000000 --online_steps=0 \
  --action_chunk_length=25 --success_only=1 \
  --action_norm_percentile=1 --action_norm_clip=5 \
  --log_interval=1000 --eval_interval=50000 --eval_episodes=500 \
  --save_interval=100000 --save_dir=.local/native-repro
```

**Difference:** this command replaces data every 1,000 updates and samples with replacement. Our runner exhausts a shuffled collection without reusing starting points, so its refresh interval varies with retained data size. Native streaming replaces validation data too; ours holds the initial validation set fixed. No native flag makes those policies identical. The native default is 25K episodes per 100K updates; the explicit settings above are a closer-size alternative, not the default or an exact match.

## Exact experiment settings: the small wrapper is needed for M4

There are no native flags for our frozen z-score, XYZ scaling or Fourier inputs. Our [runner](scripts/train.py) adds those transforms, the one-use stream and reporting around the native agent/simulator. Use it for all six M4 conditions if you want our experiment design. It also handles the exact M1 sampling/diagnostic policy when matching our runner matters.

```bash
ours=(--steps 2000000 --batch-size 1024 --width 4096 --depth 8
      --val-episodes 128 --log-interval 1000 --eval-interval 50000
      --eval-episodes 500 --save-interval 100000 --keep-checkpoints 2
      --no-collection-cache --no-loader-prefetch)

# Four M1 jobs using our sampling/diagnostic policy.
.venv/bin/python -u scripts/train.py "${ours[@]}" --run rerun-m1-double-10k-s0 \
  --env block-double-task1-v0 --mode fixed --features handcrafted \
  --episodes 10000 --seed 0 --data-seed 0 --parallel 512
.venv/bin/python -u scripts/train.py "${ours[@]}" --run rerun-m1-double-10k-s1 \
  --env block-double-task1-v0 --mode fixed --features handcrafted \
  --episodes 10000 --seed 1 --data-seed 1 --parallel 2500
.venv/bin/python -u scripts/train.py "${ours[@]}" --run rerun-m1-single-10k-s0 \
  --env block-single-task1-v0 --mode fixed --features handcrafted \
  --episodes 10000 --seed 0 --data-seed 0 --parallel 2500
.venv/bin/python -u scripts/train.py "${ours[@]}" --run rerun-m1-single-50k-s0 \
  --env block-single-task1-v0 --mode fixed --features handcrafted \
  --episodes 50000 --seed 0 --data-seed 10000 --parallel 2500

# Six M4 jobs; each loop iteration is a separate independent run.
for feature in handcrafted standardized xyz0.1 xyz10 fourier1 fourier4; do
  parallel=512
  case "$feature" in xyz10|fourier1|fourier4) parallel=2500 ;; esac
  .venv/bin/python -u scripts/train.py "${ours[@]}" \
    --run "rerun-m4-${feature}-s0" --env block-double-task1-v0 \
    --mode streaming --features "$feature" --episodes 1000 \
    --seed 0 --data-seed 0 --parallel "$parallel"
done
```

Use a fresh checkout/data directory: M1 otherwise reuses existing fixed dataset files. Cache/prefetch are disabled here to keep independently generated streams like the original source jobs; they do not change the native learner. Fourier jobs are extensions without a confirmed benefit. Fresh 2M runs do not replay our checkpoint-continuation branches, which restart the stream and validation set. No command guarantees the same success percentages.

See [the short guide and timestamped job status](EXPERIMENT_GUIDE.md), [saved configurations](results/runs/README.md), and [measured findings](RESULTS.md).
