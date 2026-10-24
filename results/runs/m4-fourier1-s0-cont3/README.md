# m4-fourier1-s0-cont3

Status: **training**.

[Results](../../../RESULTS.md) · [Run catalog](../README.md)

W&B: awaiting authenticated upload (or run has not started).

[Exact configuration](config.json) · [Training and validation log](metrics.jsonl) · [Evaluation log](eval.jsonl) · [Collection log](collections.jsonl) · [Retained dataset counts](dataset.jsonl)

These JSONL records are the complete portable scalar logs, not sampled chart exports. Machine stdout/stderr, credentials, data and checkpoints remain local.

## Evaluations

| Gradient update | Success | Episodes | Record | W&B |
| ---: | ---: | ---: | --- | --- |
| 650000 | 0.068 | 500 | [JSONL line 1](eval.jsonl#L1) | pending upload |
| 700000 | 0.106 | 500 | [JSONL line 2](eval.jsonl#L2) | pending upload |
| 750000 | 0.1 | 500 | [JSONL line 3](eval.jsonl#L3) | pending upload |
| 800000 | 0.15 | 500 | [JSONL line 4](eval.jsonl#L4) | pending upload |
| 850000 | 0.198 | 500 | [JSONL line 5](eval.jsonl#L5) | pending upload |
| 900000 | 0.194 | 500 | [JSONL line 6](eval.jsonl#L6) | pending upload |
| 950000 | 0.258 | 500 | [JSONL line 7](eval.jsonl#L7) | pending upload |
| 1000000 | 0.198 | 500 | [JSONL line 8](eval.jsonl#L8) | pending upload |
| 1050000 | 0.316 | 500 | [JSONL line 9](eval.jsonl#L9) | pending upload |
| 1100000 | 0.298 | 500 | [JSONL line 10](eval.jsonl#L10) | pending upload |
| 1150000 | 0.318 | 500 | [JSONL line 11](eval.jsonl#L11) | pending upload |

## Continuation boundary

Continues [m4-fourier1-s0](../m4-fourier1-s0/README.md) from update 600000. Network, optimizer, agent RNG and preprocessing are restored. The streaming dataset, sampler and validation set restart. Validation losses across this boundary use different held-out samples. This is not an exact sampler resume.

[Boundary metadata](continuation.jsonl)
