# m4-xyz0.1-s0-cont1

Status: **inactive**.

[Results](../../../RESULTS.md) · [Run catalog](../README.md)

W&B: [training charts and evaluations](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocbf86da02146d369276)

[Exact configuration](config.json) · [Training and validation log](metrics.jsonl) · [Evaluation log](eval.jsonl) · [Collection log](collections.jsonl) · [Retained dataset counts](dataset.jsonl)

These JSONL records are the complete portable scalar logs, not sampled chart exports. Machine stdout/stderr, credentials, data and checkpoints remain local.

## Evaluations

| Gradient update | Success | Episodes | Record | W&B |
| ---: | ---: | ---: | --- | --- |
| 1150000 | 0.008 | 500 | [JSONL line 1](eval.jsonl#L1) | [Evaluation charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocbf86da02146d369276) |

## Continuation boundary

Continues [m4-xyz0.1-s0](../m4-xyz0.1-s0/README.md) from update 1100000. Network, optimizer, agent RNG and preprocessing are restored. The streaming dataset, sampler and validation set restart. Validation losses across this boundary use different held-out samples. This is not an exact sampler resume.

[Boundary metadata](continuation.jsonl)
