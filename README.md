# OCBench behavioral cloning reproductions

Reproduce mysteries 1 (overfitting and dataset size) and 4 (observation scaling with streaming data) from [Behavioral cloning mystery](https://seohong.me/blog/behavioral-cloning-mystery/).

```bash
git submodule update --init --recursive
bash scripts/setup.sh
```

Source `scripts/env.sh` before running experiments. Python environment, caches, temporary downloads, datasets, logs, and checkpoints stay inside this checkout. Four independent GPU workers run conditions and seeds in parallel. Checkpoints and machine-specific progress logs are ignored by Git. Settings and findings are recorded in RESULTS.md.

Run a short integration test:

```bash
source scripts/env.sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train.py --run smoke-example \
  --episodes 8 --val-episodes 4 --parallel 12 --steps 2 --batch-size 16 \
  --width 128 --depth 2 --log-interval 1 --eval-interval 2 --eval-episodes 4
```

Run the full four-GPU matrix in a persistent session:

```bash
tmux -S "$REPO_ROOT/.local/tmux.sock" new-session -d -s ocbench-reproduction \
  'bash -c "source scripts/env.sh; exec .venv/bin/python -u scripts/supervise.py --pilot-first > .local/supervisor.log 2>&1"'
```

`results/matrix.json` records all 28 conditions/seeds. The supervisor uses one GPU per run, exports scalar metrics every 30 minutes, and commits/pushes `results/curves.csv` and the live table in `RESULTS.md`. Configure Git author identity and remote credentials before starting. It stops assigning a GPU if its job fails and preserves incomplete runs for inspection. Inspect `.local/queue-status.json` and `.local/*.log`; run `scripts/summarize.py` to refresh curves manually.

Mystery 4 uses freshly generated collections without reusing training start indices. Scaling statistics and action normalization are fitted on the initial training collection and frozen; the initial validation set is held out throughout. Standardization here is z-score with a minimum standard deviation of 0.001; exact blog normalization details are not published in the reference commands. Input xyz indices follow the pinned block environment observation layout. This is a documented reproduction choice.

The latest two checkpoints per run are retained locally to bound disk usage. `results/environment.txt` pins the installed package versions and is reused by setup. Long runs may take days; a completed first training step establishes launch health, while reproduction conclusions require full curves and rollout evaluations.

`--pilot-first` adds four 100-step full-model task-learning pilots (64 generated episodes each) before the full queue. Pilot metrics are labeled with `pilot-` and do not support the blog conclusions.
