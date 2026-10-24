# m1-single-n10000-s0-cont1

Status: **completed**.

[Results](../../../RESULTS.md) · [Run catalog](../README.md)

W&B: [training charts and evaluations](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb45bf280c7ad3312b8)

[Exact configuration](config.json) · [Training and validation log](metrics.jsonl) · [Evaluation log](eval.jsonl) · [Collection log](collections.jsonl) · [Retained dataset counts](dataset.jsonl)

These JSONL records are the complete portable scalar logs, not sampled chart exports. Machine stdout/stderr, credentials, data and checkpoints remain local.

## Evaluations

| Gradient update | Success | Episodes | Record | W&B |
| ---: | ---: | ---: | --- | --- |
| 1150000 | 0.896 | 500 | [JSONL line 1](eval.jsonl#L1) | [Evaluation charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb45bf280c7ad3312b8) |
| 1200000 | 0.888 | 500 | [JSONL line 2](eval.jsonl#L2) | [Evaluation charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb45bf280c7ad3312b8) |
| 1250000 | 0.908 | 500 | [JSONL line 3](eval.jsonl#L3) | [Evaluation charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb45bf280c7ad3312b8) |
| 1300000 | 0.904 | 500 | [JSONL line 4](eval.jsonl#L4) | [Evaluation charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb45bf280c7ad3312b8) |
| 1350000 | 0.908 | 500 | [JSONL line 5](eval.jsonl#L5) | [Evaluation charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb45bf280c7ad3312b8) |
| 1400000 | 0.89 | 500 | [JSONL line 6](eval.jsonl#L6) | [Evaluation charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb45bf280c7ad3312b8) |
| 1450000 | 0.898 | 500 | [JSONL line 7](eval.jsonl#L7) | [Evaluation charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb45bf280c7ad3312b8) |
| 1500000 | 0.882 | 500 | [JSONL line 8](eval.jsonl#L8) | [Evaluation charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb45bf280c7ad3312b8) |
| 1550000 | 0.926 | 500 | [JSONL line 9](eval.jsonl#L9) | [Evaluation charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb45bf280c7ad3312b8) |
| 1600000 | 0.898 | 500 | [JSONL line 10](eval.jsonl#L10) | [Evaluation charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb45bf280c7ad3312b8) |
| 1650000 | 0.884 | 500 | [JSONL line 11](eval.jsonl#L11) | [Evaluation charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb45bf280c7ad3312b8) |
| 1700000 | 0.896 | 500 | [JSONL line 12](eval.jsonl#L12) | [Evaluation charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb45bf280c7ad3312b8) |
| 1750000 | 0.898 | 500 | [JSONL line 13](eval.jsonl#L13) | [Evaluation charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb45bf280c7ad3312b8) |
| 1800000 | 0.898 | 500 | [JSONL line 14](eval.jsonl#L14) | [Evaluation charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb45bf280c7ad3312b8) |
| 1850000 | 0.906 | 500 | [JSONL line 15](eval.jsonl#L15) | [Evaluation charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb45bf280c7ad3312b8) |
| 1900000 | 0.904 | 500 | [JSONL line 16](eval.jsonl#L16) | [Evaluation charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb45bf280c7ad3312b8) |
| 1950000 | 0.868 | 500 | [JSONL line 17](eval.jsonl#L17) | [Evaluation charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb45bf280c7ad3312b8) |
| 2000000 | 0.892 | 500 | [JSONL line 18](eval.jsonl#L18) | [Evaluation charts](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb45bf280c7ad3312b8) |

## Continuation boundary

Continues [m1-single-n10000-s0](../m1-single-n10000-s0/README.md) from update 1100000. Network, optimizer, agent RNG and preprocessing are restored. The fixed dataset and validation set are reused; sampler and diagnostic RNG restart. This is not an exact sampler resume.

[Boundary metadata](continuation.jsonl)
