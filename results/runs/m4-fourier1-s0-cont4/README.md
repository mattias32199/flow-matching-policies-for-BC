# m4-fourier1-s0-cont4

Status: **completed**.

[Results](../../../RESULTS.md) · [Run catalog](../README.md)

W&B: awaiting authenticated upload (or run has not started).

[Exact configuration](config.json) · [Training and validation log](metrics.jsonl) · [Evaluation log](eval.jsonl) · [Collection log](collections.jsonl) · [Retained dataset counts](dataset.jsonl)

These JSONL records are the complete portable scalar logs, not sampled chart exports. Machine stdout/stderr, credentials, data and checkpoints remain local.

## Evaluations

| Gradient update | Success | Episodes | Record | W&B |
| ---: | ---: | ---: | --- | --- |
| 1850000 | 0.53 | 500 | [JSONL line 1](eval.jsonl#L1) | pending upload |
| 1900000 | 0.56 | 500 | [JSONL line 2](eval.jsonl#L2) | pending upload |
| 1950000 | 0.598 | 500 | [JSONL line 3](eval.jsonl#L3) | pending upload |
| 2000000 | 0.514 | 500 | [JSONL line 4](eval.jsonl#L4) | pending upload |

## Continuation boundary

Continues [m4-fourier1-s0-cont3](../m4-fourier1-s0-cont3/README.md) from update 1800000. Network, optimizer, agent RNG and preprocessing are restored. The fixed dataset and validation set are reused; sampler and diagnostic RNG restart. This is not an exact sampler resume.

[Boundary metadata](continuation.jsonl)
