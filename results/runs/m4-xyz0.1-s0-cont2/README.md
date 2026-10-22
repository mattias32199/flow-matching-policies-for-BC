# m4-xyz0.1-s0-cont2

Status: **completed**.

[Results](../../../RESULTS.md) · [Run catalog](../README.md)

W&B: [training charts and evaluations](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb608bbd66f8b803e14)

[Exact configuration](config.json) · [Training and validation log](metrics.jsonl) · [Evaluation log](eval.jsonl) · [Collection log](collections.jsonl) · [Retained dataset counts](dataset.jsonl)

These JSONL records are the complete portable scalar logs, not sampled chart exports. Machine stdout/stderr, credentials, data and checkpoints remain local.

## Evaluations

| Gradient update | Success | Episodes | Record | W&B |
| ---: | ---: | ---: | --- | --- |
| 1450000 | 0.016 | 500 | [JSONL line 1](eval.jsonl#L1) | [Evaluation charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb608bbd66f8b803e14) |
| 1500000 | 0.05 | 500 | [JSONL line 2](eval.jsonl#L2) | [Evaluation charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb608bbd66f8b803e14) |
| 1550000 | 0.038 | 500 | [JSONL line 3](eval.jsonl#L3) | [Evaluation charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb608bbd66f8b803e14) |
| 1600000 | 0.042 | 500 | [JSONL line 4](eval.jsonl#L4) | [Evaluation charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb608bbd66f8b803e14) |
| 1650000 | 0.038 | 500 | [JSONL line 5](eval.jsonl#L5) | [Evaluation charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb608bbd66f8b803e14) |
| 1700000 | 0.032 | 500 | [JSONL line 6](eval.jsonl#L6) | [Evaluation charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb608bbd66f8b803e14) |
| 1750000 | 0.058 | 500 | [JSONL line 7](eval.jsonl#L7) | [Evaluation charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb608bbd66f8b803e14) |
| 1800000 | 0.04 | 500 | [JSONL line 8](eval.jsonl#L8) | [Evaluation charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb608bbd66f8b803e14) |
| 1850000 | 0.054 | 500 | [JSONL line 9](eval.jsonl#L9) | [Evaluation charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb608bbd66f8b803e14) |
| 1900000 | 0.07 | 500 | [JSONL line 10](eval.jsonl#L10) | [Evaluation charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb608bbd66f8b803e14) |
| 1950000 | 0.07 | 500 | [JSONL line 11](eval.jsonl#L11) | [Evaluation charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb608bbd66f8b803e14) |
| 2000000 | 0.064 | 500 | [JSONL line 12](eval.jsonl#L12) | [Evaluation charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb608bbd66f8b803e14) |

## Continuation boundary

Continues [m4-xyz0.1-s0](../m4-xyz0.1-s0/README.md) from update 1400000. Network, optimizer, agent RNG and preprocessing are restored. The streaming dataset, sampler and validation set restart. Validation losses across this boundary use different held-out samples. This is not an exact sampler resume.

[Boundary metadata](continuation.jsonl)
