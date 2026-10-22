# m4-handcrafted-s0-cont2

Status: **completed**.

[Results](../../../RESULTS.md) · [Run catalog](../README.md)

W&B: [training charts and evaluations](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb84f0a7faf0b586cf8)

[Exact configuration](config.json) · [Training and validation log](metrics.jsonl) · [Evaluation log](eval.jsonl) · [Collection log](collections.jsonl) · [Retained dataset counts](dataset.jsonl)

These JSONL records are the complete portable scalar logs, not sampled chart exports. Machine stdout/stderr, credentials, data and checkpoints remain local.

## Evaluations

| Gradient update | Success | Episodes | Record | W&B |
| ---: | ---: | ---: | --- | --- |
| 1450000 | 0.538 | 500 | [JSONL line 1](eval.jsonl#L1) | [Evaluation charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb84f0a7faf0b586cf8) |
| 1500000 | 0.62 | 500 | [JSONL line 2](eval.jsonl#L2) | [Evaluation charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb84f0a7faf0b586cf8) |
| 1550000 | 0.652 | 500 | [JSONL line 3](eval.jsonl#L3) | [Evaluation charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb84f0a7faf0b586cf8) |
| 1600000 | 0.562 | 500 | [JSONL line 4](eval.jsonl#L4) | [Evaluation charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb84f0a7faf0b586cf8) |
| 1650000 | 0.658 | 500 | [JSONL line 5](eval.jsonl#L5) | [Evaluation charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb84f0a7faf0b586cf8) |
| 1700000 | 0.568 | 500 | [JSONL line 6](eval.jsonl#L6) | [Evaluation charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb84f0a7faf0b586cf8) |
| 1750000 | 0.588 | 500 | [JSONL line 7](eval.jsonl#L7) | [Evaluation charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb84f0a7faf0b586cf8) |
| 1800000 | 0.616 | 500 | [JSONL line 8](eval.jsonl#L8) | [Evaluation charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb84f0a7faf0b586cf8) |
| 1850000 | 0.662 | 500 | [JSONL line 9](eval.jsonl#L9) | [Evaluation charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb84f0a7faf0b586cf8) |
| 1900000 | 0.638 | 500 | [JSONL line 10](eval.jsonl#L10) | [Evaluation charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb84f0a7faf0b586cf8) |
| 1950000 | 0.558 | 500 | [JSONL line 11](eval.jsonl#L11) | [Evaluation charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb84f0a7faf0b586cf8) |
| 2000000 | 0.564 | 500 | [JSONL line 12](eval.jsonl#L12) | [Evaluation charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb84f0a7faf0b586cf8) |

## Continuation boundary

Continues [m4-handcrafted-s0](../m4-handcrafted-s0/README.md) from update 1400000. Network, optimizer, agent RNG and preprocessing are restored. The streaming dataset, sampler and validation set restart. Validation losses across this boundary use different held-out samples. This is not an exact sampler resume.

[Boundary metadata](continuation.jsonl)
