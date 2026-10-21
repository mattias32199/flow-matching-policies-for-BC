# Mystery 1 and 4: what we have learned

**One experiment is complete; the other nine have been submitted.** This is a progress snapshot, not a finished multi-seed reproduction. The [original blog](https://seohong.me/blog/behavioral-cloning-mystery/) is the reference. Our plots use the same order: robot success, flow loss, action MSE.

## What is confirmed, and what is still open?

| Question | Evidence so far | What we can say |
| --- | --- | --- |
| Can control improve while validation flow loss gets worse? | One completed block-double 10K run | **Yes, in this run.** This does not show that overfitting causes better control. |
| Is 10K data better than 50K data? | Both block-single jobs are submitted; no results yet | **Unknown.** We cannot claim the dataset-size reversal. |
| Does input scaling matter with fresh data? | Four partial, single-seed source runs | **A large effect is observed.** XYZ ×0.1 is much worse; exact final rankings are unfinished. |
| Are all scaling variants' losses identical? | Similar early losses; later losses differ | **No.** The losses do not reliably rank robot success, but they are not identical. |
| Do Fourier position features help? | Two jobs submitted; no rollout results | **Unknown.** These are extensions beyond the blog comparison. |
| Are these conclusions robust across seeds? | One completed seed; repeat submitted | **Not established.** |

## Mystery 1: loss can be misleading

![Mystery 1: success, flow loss, and action MSE](results/mystery1.png)

In the completed [block-double, nominal 10K, seed 0 run](results/runs/m1-double-n10000-s0/README.md), validation flow loss reached its lowest recorded value at **182K updates: 0.01752**. It then rose to **0.05617 at 2M**, a **3.21× increase**, while training flow loss fell to **0.00337**. Yet robot success rose from **0.8% at the nearby 200K evaluation to 25.8% at 2M**. Each success measurement uses 500 episodes; the final result is 129 successes out of 500.

This reproduces the qualitative loss/control mismatch in one run: a worse validation flow score does not necessarily mean a worse robot policy. Validation action MSE stays much flatter after its initial decline, ending at **0.01460**. It is a different measure of prediction error, and it tells a different story from flow loss. We have not shown that either loss predicts control perfectly, or that deliberately causing more overfitting improves control.

The best checkpoint scored 29.6%; the **final score is 25.8%**, not that selected maximum. Nominal 10K means 10,000 requested demonstrations; successful-trajectory filtering retained 7,866 training episodes. The seed-1 repeat and the independent block-single 10K/50K comparison have no rollout results yet. One run per dataset size will still leave seed robustness unresolved. See [exact computed evidence](results/runs/m1-double-n10000-s0/analysis.json).

## Mystery 4: scaling affects behavior much more than the training scores suggest

### Handcrafted versus standardized

![Mystery 4 baseline comparison](results/mystery4_baseline.png)

### Changing only the XYZ scale

![Mystery 4 four-way scaling comparison](results/mystery4_scaling.png)

The table compares all four source runs at the **same 1.35M update**, the latest shared rollout checkpoint. Comparing each run's latest result would mix different training budgets.

| Input features | Robot success | Train flow | Val flow | Val action MSE |
| --- | ---: | ---: | ---: | ---: |
| [Handcrafted](results/runs/m4-handcrafted-s0/README.md) | 52.2% | 0.01279 | 0.01181 | 0.01107 |
| [Standardized](results/runs/m4-standardized-s0/README.md) | 37.4% | 0.01127 | 0.01220 | 0.01176 |
| [Standardized, XYZ ×0.1](results/runs/m4-xyz0.1-s0/README.md) | 2.8% | 0.01305 | 0.01438 | 0.01295 |
| [Standardized, XYZ ×10](results/runs/m4-xyz10-s0/README.md) | 61.2% | 0.01597 | 0.01556 | 0.02894 |

XYZ ×0.1 achieves **2.8% success**, versus **52.2%** for handcrafted inputs and **61.2%** for XYZ ×10. The training flow losses remain in a fairly narrow range. More tellingly, XYZ ×10 has a worse validation action MSE (**0.02894**) than XYZ ×0.1 (**0.01295**), while its robot success is far higher. A lower held-out prediction error therefore does not reliably identify the better robot policy here. [Held-out loss plots](results/mystery4_validation.png) show this directly.

This supports the qualitative scaling effect. It does **not** establish that all losses are identical, a reliable ranking between XYZ ×10 and handcrafted inputs, or a result at infinite training time. These are partial single-seed runs with finite data streams. Earlier workers used paired seeds but separately simulated data, so their trajectories can differ slightly. The original allocation timed out before they finished; their historical measurements remain valid partial results.

## Restarts and current job progress

![Continuation segments on their own validation sets](results/continuations.png)

The four original Mystery 4 experiments continue from pinned checkpoints: handcrafted, standardized and XYZ ×0.1 at **1.1M**, XYZ ×10 at **800K**. They restore model weights, optimizer state, agent RNG and preprocessing. They restart the data stream, sampler and validation set. These are **continuation segments of the same four conditions**, not extra experiments or exact uninterrupted resumes. Some source curves extend beyond the pinned updates; the resumed branches replay part of the update range using fresh data. We do not average those branches or join their validation curves.

The first three continuations have completed real GPU updates with finite losses. Their new validation numbers cannot be compared directly to the old held-out sets, and fresh rollout evaluations are required before claiming new control results. The fourth continuation and the five new conditions remain queued at this snapshot. Pending jobs accept **A6000 or A5000**, whichever Slurm can schedule first. This improves placement flexibility; it is not a measured A5000 speed claim.

Snapshot: **2026-10-21T22:57:03+00:00**. [Machine-readable findings and job states](results/findings.json). These scheduler states are a snapshot; [W&B](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4) follows active training.

| Scientific experiment | State | Latest update in current segment | GPU / eligibility |
| --- | --- | ---: | --- |
| m1-double-n10000-s0 | completed | 2,000,000 | — |
| m1-double-n10000-s1 | pending | — | A6000|A5000 |
| m1-single-n10000-s0 | pending | — | A6000|A5000 |
| m1-single-n50000-s0 | pending | — | A6000|A5000 |
| m4-handcrafted-s0 | running | 1,113,000 | A6000 |
| m4-standardized-s0 | running | 1,113,000 | A6000 |
| m4-xyz0.1-s0 | running | 1,113,000 | A6000 |
| m4-xyz10-s0 | pending | — | A6000|A5000 |
| m4-fourier1-s0 | pending | — | A6000|A5000 |
| m4-fourier4-s0 | pending | — | A6000|A5000 |

## How to read the figures

Success is the observed fraction of 500 evaluation episodes. We connect the measured checkpoints; there are no invented points, extrapolated tails or seed uncertainty bands. Every plotted condition currently has one training seed. Thin loss traces are the recorded measurements; thick lines are nonoverlapping 50K-update means, computed within each run. Loss panels omit the first 10K updates to make later trends readable and use logarithmic vertical axes. The raw [CSV](results/curves.csv) and [per-run JSONL records](results/runs/README.md) retain all measurements. Plots are also available as SVG: [Mystery 1](results/mystery1.svg), [M4 baseline](results/mystery4_baseline.svg), [M4 scaling](results/mystery4_scaling.svg), [M4 validation](results/mystery4_validation.svg), [continuations](results/continuations.svg).

## What this reproduction does not match exactly

The [experiment plan](EXPERIMENTS.md) records the settings and input/output contract. We use the public reference architecture (eight 4096-wide layers), batch 1024, action chunks of 25, and a 2M-update budget. The completed Mystery 1 run stops earlier than the blog's displayed 3M-update curve. Our standardization is a frozen training-set z-score; the blog's exact scaling constants are not available in the reference commands. Streaming uses finite 1K-episode collections with transition starts used once. Fourier features are our extensions. Those choices, limited seeds, and restarted validation sets prevent a claim of exact numerical reproduction or proof of the proposed distribution-shift explanation.

## Execution and performance evidence

All nine remaining scientific jobs were accepted by Slurm; three are training at this snapshot. The first two continuations restored their saved steps and completed subsequent collections and updates. W&B server history independently confirmed the resumed metrics. A separate CPU Slurm job maintains uploads without depending on the interactive agent. Descriptive job/script names are used inside owner-only directories. Data, checkpoints, credentials, exact local launch scripts and run directories remain outside Git.

Initial A6000 compute-only intervals were about **34 updates/s**. Intervals spanning a fresh collection plus training were about **10.8 updates/s** per job. These short observations are not long-run averages or A5000 benchmarks. See the earlier [isolated profile](results/profiling/README.md) and [shared-cache benchmark](results/cache-performance/README.md) for separate performance experiments; their synthetic/aggregate speedups are not production speed claims. The 80GB cache cap bounds retained cache entries, not the cumulative data needed over a whole training stream.

The 32-test suite passes, covering preprocessing, metrics, checkpoint restoration, cache/prefetch behavior and private individual launchers. Small-model and integrity checks are recorded under [validation](results/validation/). They establish execution correctness, not the scientific conclusions above.

## Exact records

[Run catalog](results/runs/README.md) · [W&B project](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4) · [W&B catalog](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocbench-catalog-v1)

The table below lists observed source and continuation records separately. Inactive source runs stopped at the allocation time limit. Blank evaluation cells mean that segment has no recorded rollout result yet. Pilots and smoke tests remain in the catalog as execution diagnostics.

<!-- LIVE_RESULTS_START -->
## Live experiment status

[W&B project](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4) · [W&B run catalog](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocbench-catalog-v1)

Pilot runs validate execution only. Reproduction runs target 2,000,000 updates; partial single-seed curves support only the limited findings explained above.

| Run records | W&B | Updates | Train flow | Val flow | Val action MSE | Success | Eval update | Finished |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| [m1-double-n10000-s0](results/runs/m1-double-n10000-s0/README.md) | [Charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb7d868146415f56201) | 2000000 | 0.0034 | 0.0562 | 0.0146 | 0.2580 | 2000000 | yes |
| [m4-handcrafted-s0](results/runs/m4-handcrafted-s0/README.md) | [Charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb6555c1eef3e7c6a5b) | 1468000 | 0.0118 | 0.0162 | 0.0170 | 0.5760 | 1450000 | no |
| [m4-handcrafted-s0-cont1](results/runs/m4-handcrafted-s0-cont1/README.md) | [Charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb89ddace69a79a65f6) | 1113000 | 0.0126 | 0.0137 | 0.0147 | — | — | no |
| [m4-standardized-s0](results/runs/m4-standardized-s0/README.md) | [Charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocbdf09e0125b1d85354) | 1500000 | 0.0131 | 0.0132 | 0.0131 | 0.5000 | 1450000 | no |
| [m4-standardized-s0-cont1](results/runs/m4-standardized-s0-cont1/README.md) | [Charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb33979b4ad2b65ef78) | 1113000 | 0.0127 | 0.0139 | 0.0150 | — | — | no |
| [m4-xyz0.1-s0](results/runs/m4-xyz0.1-s0/README.md) | [Charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocbb5e56eb1e20d223a2) | 1473000 | 0.0112 | 0.0151 | 0.0187 | 0.0260 | 1450000 | no |
| [m4-xyz0.1-s0-cont1](results/runs/m4-xyz0.1-s0-cont1/README.md) | [Charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocbf86da02146d369276) | 1113000 | 0.0134 | 0.0140 | 0.0158 | — | — | no |
| [m4-xyz10-s0](results/runs/m4-xyz10-s0/README.md) | [Charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb8f3a301437f09df88) | 1387000 | 0.0105 | 0.0208 | 0.0273 | 0.6120 | 1350000 | no |

Blank evaluation cells mean no recorded evaluation yet. Undefined duration aggregates at zero success are stored as null; success itself remains zero.
<!-- LIVE_RESULTS_END -->
