# Results

Status: first full 2M-update run completed; remaining comparisons are in progress. Findings below are preliminary and single-seed.

## Evidence and exact runs

Open the [run catalog](results/runs/README.md) for every planned condition, diagnostic, configuration, and complete portable scalar log. Run names in the table below link directly to their records. Each run page lists individual evaluations by gradient update, with links to exact JSONL lines and W&B evaluation charts when available.

The live W&B project is [yzchhr/ocbench-bc-mysteries-1-4](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4). Historical metrics and evaluations are uploaded; active runs continue to update there. The [W&B run catalog](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocbench-catalog-v1) links back to the exact run records. Queued runs are labeled and acquire W&B links when they start.

The completed [smoke-v3 evaluation](results/runs/smoke-v3/README.md#evaluations) verifies evaluation logging and checkpoint execution; its zero-success result is a two-update smoke test, not a reproduction result.

## Completed result: Mystery 1, block-double, nominal 10K, seed 0

![Completed run: success and log-scale losses](results/runs/m1-double-n10000-s0/analysis.png)

The [completed run](results/runs/m1-double-n10000-s0/README.md) reached **25.8% success (129/500)** at 2M updates ([exact final evaluation](results/runs/m1-double-n10000-s0/eval.jsonl#L40)). Its last five evaluations at 1.8–2M updates averaged **25.08%**; these are correlated checkpoint evaluations, not five independent seeds. The best observed evaluation was 29.6% at 1.45M updates ([record](results/runs/m1-double-n10000-s0/eval.jsonl#L29)); choosing this maximum is optimistic and is not the final score.

Validation flow loss reached its recorded minimum of 0.017516 at 182K updates, then increased to 0.056173 at 2M (3.21×), while final training flow loss fell to 0.003367. Rollout success increased from 0.8% at the nearby 200K evaluation to 25.8% at 2M. This supports the qualitative Mystery 1 pattern that worsening validation flow loss can coexist with improving control. It does not yet establish the dataset-size reversal or robustness across seeds. Final validation action MSE was 0.014597. See the [computed analysis](results/runs/m1-double-n10000-s0/analysis.json), [complete loss log](results/runs/m1-double-n10000-s0/metrics.jsonl), and [W&B charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb7d868146415f56201).

The nominal 10,000 training episodes retained 7,866 successful episodes (12,465,118 transitions); validation retained 97 episodes (152,756 transitions). The run took approximately 16.74 hours through the final training log, including collection and earlier evaluations. All 40 scheduled 500-episode evaluations are recorded. The latest two checkpoints (1.9M and 2M) remain local and are excluded from Git. The final checkpoint deserialized on CPU with all 117 numeric leaves finite; optimizer count is 2,000,000. Its size and SHA-256 are recorded in the analysis; this integrity check does not constitute a fresh rollout evaluation.

At the completion snapshot, Mystery 4's 500K evaluations were 27.2% handcrafted, 2.2% standardized, and 0.2% xyz×0.1. These partial single-seed results suggest a substantial feature-scaling effect, but all runs remain unfinished and xyz×10 has only just started. The queue has one completed experiment, four active, 23 pending, and no recorded failures. Existing protocol limitations above still apply.

## Isolated A6000 training profile

A dedicated single-GPU Slurm profile tested eight full-model execution variants with three timing repeats each. [Exact timing records, approximate MFU, and trace aggregates](results/profiling/README.md) are available separately from the scientific runs; these are synthetic throughput tests.

The original host-fed update loop sustained **36.29 updates/s**. Device-resident inputs plus buffer donation gave **36.30/s**, explicit TF32 **36.13/s**, and eight-update compiled groups **37.16/s** (2.4% faster). BF16 reached **56.58/s** (1.56×), with finite losses but unverified task-learning equivalence. Strict FP32 precision was slower at **19.58/s**. Default-mode CUDA traces contain tensor-core GEMMs; GEMM events account for approximately 56% of recorded GPU-event durations. Thus high GPU activity does not mean all runtime is matrix multiplication.

Analytical dense-matrix FLOPs divided by the precision-specific dense peak yield approximately **39% training-loop MFU** for the default path and **31% for BF16**. These are estimates, not hardware counters: the default denominator assumes TF32 tensor-core execution, and the numerator omits elementwise, normalization, and optimizer operations. The profiler's raw XLA cost analysis omits substantial library/loop work and is explicitly rejected as an MFU numerator. Higher precision-dependent MFU does not necessarily mean faster training.

Production streaming runs spend roughly 75% of elapsed time collecting demonstrations (about 64% for the newer parallelism setting). A 1.56× update speedup therefore implies only approximately 10–15% end-to-end improvement if other costs stay unchanged. The isolated simulator sweep measured 115.28s for 1,000 episodes at parallel512 versus 63.33s at parallel1024 (1.82×); 2,500 episodes at parallel2500 took 88.08s (28.38 episodes/s), testing a larger collection rather than the current protocol. The larger simulator parallelism cap is already used by newer/future processes. Existing reproduction workers were preserved: grouped updates offer little benefit, and changing arithmetic requires learning validation. Existing checkpoints contain agent/optimizer/RNG state but omit the data-sampler cursor/permutation and streaming snapshot, so exact continuation cannot be asserted without additional resume-state support.

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
| [m1-double-n10000-s0](results/runs/m1-double-n10000-s0/README.md) | [Charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb7d868146415f56201) | 2000000 | 0.0034 | 0.0562 | 0.0146 | 0.2580 | 2000000 | yes |
| [m4-handcrafted-s0](results/runs/m4-handcrafted-s0/README.md) | [Charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb6555c1eef3e7c6a5b) | 1002000 | 0.0106 | 0.0172 | 0.0136 | 0.5160 | 1000000 | no |
| [m4-standardized-s0](results/runs/m4-standardized-s0/README.md) | [Charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocbdf09e0125b1d85354) | 1028000 | 0.0170 | 0.0138 | 0.0134 | 0.2480 | 1000000 | no |
| [m4-xyz0.1-s0](results/runs/m4-xyz0.1-s0/README.md) | [Charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocbb5e56eb1e20d223a2) | 1009000 | 0.0161 | 0.0146 | 0.0146 | 0.0080 | 1000000 | no |
| [m4-xyz10-s0](results/runs/m4-xyz10-s0/README.md) | [Charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb8f3a301437f09df88) | 711000 | 0.0129 | 0.0139 | 0.0134 | 0.5180 | 700000 | no |
| [pilot-m1-double-n10000-s0](results/runs/pilot-m1-double-n10000-s0/README.md) | [Charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb68488cbd9d27a930e) | 100 | 0.2445 | 0.2911 | 0.1626 | — | — | no |
| [pilot-m4-handcrafted-s0](results/runs/pilot-m4-handcrafted-s0/README.md) | [Charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocbe05d0f1b8fc3f1410) | 100 | 0.2729 | 0.3451 | 0.2120 | — | — | no |
| [pilot-m4-standardized-s0](results/runs/pilot-m4-standardized-s0/README.md) | [Charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb6d452dd304eeae07c) | 100 | 0.3114 | 0.3221 | 0.1346 | — | — | no |
| [pilot-m4-xyz0.1-s0](results/runs/pilot-m4-xyz0.1-s0/README.md) | [Charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocbff8e42a1745ad8ec8) | 100 | 0.3143 | 0.3025 | 0.1196 | — | — | no |

Blank evaluation cells mean no recorded evaluation yet. Undefined duration aggregates at zero success are stored as null; success itself remains zero.
<!-- LIVE_RESULTS_END -->
