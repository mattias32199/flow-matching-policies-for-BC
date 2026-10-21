# OCBench behavioral cloning reproductions

Reproduce mysteries 1 (overfitting and dataset size) and 4 (observation scaling with streaming data) from [Behavioral cloning mystery](https://seohong.me/blog/behavioral-cloning-mystery/).

```bash
git submodule update --init --recursive
bash scripts/setup.sh
```

Source `scripts/env.sh` before running experiments. Python environment, caches, temporary downloads, datasets, logs, and checkpoints stay inside this checkout. Each scientific run uses one GPU. The current scope is [ten experiments](EXPERIMENTS.md): four for Mystery 1 and six for Mystery 4, including two fixed XYZ Fourier feature variants. Checkpoints and machine-specific progress logs are ignored by Git. Start with [RESULTS.md](RESULTS.md), then open the [run catalog](results/runs/README.md) or a run link in its table for the exact configuration, training log, collection log, and evaluation records.

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

`results/matrix.json` records the ten planned experiments. Existing supervisors load their queue at startup; changing this file does not modify an already-running supervisor. The supervisor uses one GPU per run, exports local metrics every 30 minutes. Git commits/pushes are reserved for code changes and completed experiment results; metric refreshes do not create commits. Configure Git author identity and remote credentials before starting. It stops assigning a GPU if its job fails and preserves incomplete runs for inspection. Inspect `.local/queue-status.json` and `.local/*.log`; run `scripts/summarize.py` to refresh curves manually.

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

`bash scripts/start_watcher.sh` starts a separate persistent watcher using the current `CODEX_THREAD_ID`. Every 10 minutes it checks locally; only a newly completed full experiment queues an AI follow-up. Routine checks, evaluation milestones, failures, and stalls do not wake the model or create timed AI turns. It uses the supported `codex queue --thread … --message …` command to queue follow-up work in the existing conversation. No new task or model is selected, and no timed Git commits are made.

The watcher requires the local Codex daemon to remain available. The local file checks do not call a model. If queuing a completion event fails, it retains the pending event locally and retries; it does not silently mark the event delivered. See `.local/watcher-status.json`, `.local/watch-events.jsonl`, and `.local/watcher.log`. A queue acknowledgement means a follow-up was accepted; model execution follows the thread's normal scheduling and approval policy. Existing diagnostic completions are ignored when first establishing the baseline.

## Throughput tuning

Future runs default to a cap of 2,500 parallel simulator worlds, limited by the requested collection size. Existing active runs retain their recorded settings. Episode counts, one-use streaming sampling, FP32 training, and training horizons are unchanged. [Throughput measurements](results/performance.json) showed a 1.77× collection speedup for the same 1,000 episodes with 1,024 versus 512 parallel worlds; larger 2,500-episode collections reached 3.13× the baseline episodes/second. Those larger collections are benchmark-only.

Synthetic BF16 updates were 1.57× faster with FP32 parameters, but BF16 has not been applied to reproduction runs. Measurements shared GPUs with ongoing jobs and are not isolated production rates. Reproduce the finite benchmarks with `source scripts/env.sh` and `CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python scripts/benchmark_speed.py train` or `sim`; they temporarily contend with any job on that GPU.

## Single-GPU profiling

Set `PROFILE_PARTITION` to an appropriate GPU partition and submit `.venv/bin/python scripts/private_slurm.py --partition "$PROFILE_PARTITION" --time 01:00:00 -- bash scripts/profile_training.sbatch` from this checkout on a Slurm cluster offering an `A6000` constraint. Adjust scheduler directives for your cluster. The job requests one GPU and leaves production runs untouched.

The profiler uses the full reference model and batch size, warms each executable, and records three timing repetitions. It compares the original host-transfer path, buffer donation, resident batches, eight-update compiled groups, FP32 precision modes, and BF16. It also runs the simulator-parallelism sweep. Results, telemetry, and traces stay under `.local/profile/`; all caches remain in this checkout. Synthetic repeated-batch measurements isolate compute and transfer costs; they do not include real dataset sampling or establish equivalent learning. Compiler FLOP estimates are reported separately from GPU utilization and must not be presented as measured hardware MFU. Numerical and task-learning checks are required before changing reproduction arithmetic.

After the job finishes, run `.venv/bin/python scripts/analyze_profile.py .local/profile/<job-id>` to export portable timing evidence and the throughput figure. See the [completed isolated profile](results/profiling/README.md). Float32 parameters/network dtype do not force strict FP32 GEMMs; the default XLA precision policy can use tensor cores.

## Shared demonstration cache

Future streaming runs share raw collections across the four feature variants using `data/shared-demos`. The cache has an **80GB decimal cap**, including in-progress writes. Each run samples its transition starts once; entries are removed after all four variants load them. Cache identity includes the environment, seed, collection index, episode counts, parallelism, generator revision and package versions. A full cache falls back to fresh collection. Use `--no-collection-cache` to disable it.

An optional pair of Slurm GPUs can precompute up to 24 collections ahead. Local deployment settings in `.local/demo-producer-config.json` select an enabled flag, a GPU partition, and eligible data seeds; these cluster settings are ignored by Git. The first eligible cache miss submits a pool of two private producer jobs using `scripts/private_slurm.py`. It stops when its consumers finish and releases its allocation after prolonged budget/no-consumer waits. The original seed-0 workers continue unchanged.

See the [measured performance report](results/cache-performance/README.md). To reproduce the benchmark, submit `.venv/bin/python scripts/private_slurm.py --partition "$PROFILE_PARTITION" --time 01:00:00 -- .venv/bin/python -u scripts/benchmark_cache.py --prefill 0`, then run `.venv/bin/python scripts/report_cache.py .local/cache-benchmark/<job-id>`. Aggregate work savings and projected concurrent throughput are reported separately.

All automated Slurm submissions use `scripts/private_slurm.py`: random job names, random owner-only launcher/work/output directories, and `umask 0077`. Keep the workspace owner-only. Administrators retain access; scheduler resource records are not anonymous. Deployment partition names and private job inventories remain under ignored local settings.
