# Results

Status: GPU integration validated; long-run reproduction in progress. No reproduction claims yet.

## Evidence and exact runs

Open the [run catalog](results/runs/README.md) for every planned condition, diagnostic, configuration, and complete portable scalar log. Run names in the table below link directly to their records. Each run page lists individual evaluations by gradient update, with links to exact JSONL lines and W&B evaluation charts when available.

The live W&B project is [yzchhr/ocbench-bc-mysteries-1-4](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4). Historical metrics and evaluations are uploaded; active runs continue to update there. The [W&B run catalog](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocbench-catalog-v1) links back to the exact run records. Queued runs are labeled and acquire W&B links when they start.

The completed [smoke-v3 evaluation](results/runs/smoke-v3/README.md#evaluations) verifies evaluation logging and checkpoint execution; its zero-success result is a two-update smoke test, not a reproduction result.

## Planned comparisons

- Mystery 1: block-double with fixed 10,000 episodes; track training/validation flow loss, action MSE, and success over training. Compare block-single with independently sampled 10,000 versus 50,000 episodes over four seeds.
- Mystery 4: block-double streaming demonstrations; compare handcrafted observations, standardized observations, and standardized observations with object/gripper xyz scaled by 0.1 or 10.

Reference defaults: eight 4096-wide residual MLP layers, action chunks of 25, batch size 1024, learning rate 1e-4, 10 flow steps. Blog-specific settings beyond the public reference commands remain unverified. Finite streaming runs approximate the infinite-data setting.

## Validation

- Local Python 3.11 environment installed from the pinned submodule; exact resolved versions are in `results/environment.txt`.
- Four preprocessing/metric tests pass (including zero-success serialization): xyz index mapping, constant-feature stability, and identical frozen training/evaluation transforms.
- Full reference FBC model has 136,179,887 parameters. Three synthetic GPU updates at batch size 1024 produced finite flow losses: 2.92764, 3.65450, 2.35684. This validates execution and memory capacity, not task learning.
- End-to-end GPU smoke test completed demonstration collection, two updates, four rollout evaluations, and checkpoint saving. Four full reference-model workers each reached 100 real updates with finite training and validation losses.
- Checkpoints remain local; the latest two per run are retained.

Dataset-size comparisons use independent generation seeds (0–3 for 10K, 10000–10003 for 50K), with training seeds 0–3 in both groups. Feature variants share generation and training seeds; GPU simulation is not bitwise deterministic, so separately generated streams can differ slightly despite paired seeds. Requested episode counts are before upstream success filtering; actual retained episodes/transitions are logged locally in each run.

<!-- LIVE_RESULTS_START -->
## Live experiment status

[W&B project](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4) · [W&B run catalog](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocbench-catalog-v1)

Pilot runs validate execution only. Reproduction runs target 2,000,000 updates; partial curves do not establish either mystery.

| Run records | W&B | Updates | Train flow | Val flow | Val action MSE | Success | Eval update | Finished |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| [m1-double-n10000-s0](results/runs/m1-double-n10000-s0/README.md) | [Charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb7d868146415f56201) | 371000 | 0.0090 | 0.0264 | 0.0128 | 0.0800 | 350000 | no |
| [m4-handcrafted-s0](results/runs/m4-handcrafted-s0/README.md) | [Charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb6555c1eef3e7c6a5b) | 102000 | 0.0196 | 0.0233 | 0.0191 | 0.0040 | 100000 | no |
| [m4-standardized-s0](results/runs/m4-standardized-s0/README.md) | [Charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocbdf09e0125b1d85354) | 104000 | 0.0228 | 0.0204 | 0.0233 | 0.0040 | 100000 | no |
| [m4-xyz0.1-s0](results/runs/m4-xyz0.1-s0/README.md) | [Charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocbb5e56eb1e20d223a2) | 102000 | 0.0233 | 0.0226 | 0.0200 | 0.0000 | 100000 | no |
| [pilot-m1-double-n10000-s0](results/runs/pilot-m1-double-n10000-s0/README.md) | [Charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb68488cbd9d27a930e) | 100 | 0.2445 | 0.2911 | 0.1626 | — | — | no |
| [pilot-m4-handcrafted-s0](results/runs/pilot-m4-handcrafted-s0/README.md) | [Charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocbe05d0f1b8fc3f1410) | 100 | 0.2729 | 0.3451 | 0.2120 | — | — | no |
| [pilot-m4-standardized-s0](results/runs/pilot-m4-standardized-s0/README.md) | [Charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb6d452dd304eeae07c) | 100 | 0.3114 | 0.3221 | 0.1346 | — | — | no |
| [pilot-m4-xyz0.1-s0](results/runs/pilot-m4-xyz0.1-s0/README.md) | [Charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocbff8e42a1745ad8ec8) | 100 | 0.3143 | 0.3025 | 0.1196 | — | — | no |

Blank evaluation cells mean no recorded evaluation yet. Undefined duration aggregates at zero success are stored as null; success itself remains zero.
<!-- LIVE_RESULTS_END -->
