# OCBench behavioral cloning reproductions

Reproduce mysteries 1 (overfitting and dataset size) and 4 (observation scaling with streaming data) from [Behavioral cloning mystery](https://seohong.me/blog/behavioral-cloning-mystery/).

```bash
git submodule update --init --recursive
bash scripts/setup.sh
```

Source `scripts/env.sh` before running experiments. Python environment, caches, temporary downloads, datasets, logs, and checkpoints stay inside this checkout. Four independent GPU workers run conditions and seeds in parallel. Checkpoints and machine-specific progress logs are ignored by Git. Start with [RESULTS.md](RESULTS.md), then open the [run catalog](results/runs/README.md) or a run link in its table for the exact configuration, training log, collection log, and evaluation records.

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

`results/matrix.json` records all 28 conditions/seeds. The supervisor uses one GPU per run, exports local metrics every 30 minutes. Git commits/pushes are reserved for code changes and completed experiment results; metric refreshes do not create commits. Configure Git author identity and remote credentials before starting. It stops assigning a GPU if its job fails and preserves incomplete runs for inspection. Inspect `.local/queue-status.json` and `.local/*.log`; run `scripts/summarize.py` to refresh curves manually.

Mystery 4 uses freshly generated collections without reusing training start indices. Scaling statistics and action normalization are fitted on the initial training collection and frozen; the initial validation set is held out throughout. Standardization here is z-score with a minimum standard deviation of 0.001; exact blog normalization details are not published in the reference commands. Input xyz indices follow the pinned block environment observation layout. This is a documented reproduction choice.

The latest two checkpoints per run are retained locally to bound disk usage. `results/environment.txt` pins the installed package versions and is reused by setup. Long runs may take days; a completed first training step establishes launch health, while reproduction conclusions require full curves and rollout evaluations.

`--pilot-first` adds four 100-step full-model task-learning pilots (64 generated episodes each) before the full queue. Pilot metrics are labeled with `pilot-` and do not support the blog conclusions.

`results/curves.csv` contains portable scalar curves; `scripts/plot.py` generates `results/curves.png` with success, validation flow loss, and validation action MSE, excluding pilots. Shading is sample standard deviation across available seeds, not a significance claim.

The [paper implementation details](https://arxiv.org/html/2610.07056v1#A4.SS1) refresh 25K streaming episodes every 100K updates. This implementation instead collects 1K episodes at a time and uses each retained transition start once, matching the blog's conceptual fresh-data description. This and the explicitly chosen z-score scaling prevent a claim of exact numerical reproduction. The requested blog uses 10K block-double episodes; the later paper's corresponding figure uses 30K.

## Experiment tracking

Navigation: **[RESULTS.md](RESULTS.md) → [run catalog](results/runs/README.md) → per-run configuration and full scalar logs**. Each evaluation row identifies its training update and exact JSONL line. Once uploaded, the same run record links to its W&B charts, and the W&B run links back to the exact repository records. Git records are snapshots updated with code/result commits; W&B can follow active runs without creating commits.

The live W&B project is [yzchhr/ocbench-bc-mysteries-1-4](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4). Historical training and evaluation logs are backfilled, and active runs are mirrored without Git commits. The [W&B catalog table](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocbench-catalog-v1) links run charts to the exact repository records. Use the normal `WANDB_API_KEY` environment variable or SDK-managed saved credentials. Keys belong outside Git.

```bash
source scripts/env.sh
# Load your credential setup into this shell first.
bash scripts/start_tracking.sh
```

The tracking launcher uses a separate persistent tmux session and checks credentials before starting. The mirror backfills existing JSONL files, then follows new records without restarting GPU jobs. Training, validation, and evaluation charts use `gradient_step`; collection charts use the collection index. Late evaluations retain their original training step. It uploads portable scalar records and a run-catalog table. Machine metadata, automatic code capture, console capture, and checkpoint uploads are disabled.

Refresh a local snapshot at any time (this does not commit or push):

```bash
.venv/bin/python scripts/summarize.py
.venv/bin/python scripts/plot.py
```

## Completion watcher

`bash scripts/start_watcher.sh` starts a separate persistent watcher using the current `CODEX_THREAD_ID`. Every 60 seconds it checks for new completed runs, supervisor failures, missing supervisor heartbeats, stale training logs, and a stalled W&B mirror. It uses the supported `codex queue --thread … --message …` command to queue follow-up work in the existing conversation. No new task or model is selected, and no timed Git commits are made.

The watcher requires the local Codex daemon to remain available. If queuing fails, it retains the pending event locally and retries; it does not silently mark the event delivered. See `.local/watcher-status.json`, `.local/watch-events.jsonl`, and `.local/watcher.log`. A queue acknowledgement means a follow-up was accepted; model execution follows the thread's normal scheduling and approval policy. Existing diagnostic completions are ignored when first establishing the baseline.
