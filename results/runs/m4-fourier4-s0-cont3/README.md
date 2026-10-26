# m4-fourier4-s0-cont3

Status: **inactive**.

[Results](../../../RESULTS.md) · [Run catalog](../README.md)

W&B: awaiting authenticated upload (or run has not started).

[Exact configuration](config.json) · [Training and validation log](metrics.jsonl) · [Evaluation log](eval.jsonl) · [Collection log](collections.jsonl) · [Retained dataset counts](dataset.jsonl)

These JSONL records are the complete portable scalar logs, not sampled chart exports. Machine stdout/stderr, credentials, data and checkpoints remain local.

## Evaluations

| Gradient update | Success | Episodes | Record | W&B |
| ---: | ---: | ---: | --- | --- |
| 650000 | 0.058 | 500 | [JSONL line 1](eval.jsonl#L1) | pending upload |
| 700000 | 0.09 | 500 | [JSONL line 2](eval.jsonl#L2) | pending upload |
| 750000 | 0.118 | 500 | [JSONL line 3](eval.jsonl#L3) | pending upload |
| 800000 | 0.162 | 500 | [JSONL line 4](eval.jsonl#L4) | pending upload |
| 850000 | 0.174 | 500 | [JSONL line 5](eval.jsonl#L5) | pending upload |
| 900000 | 0.174 | 500 | [JSONL line 6](eval.jsonl#L6) | pending upload |
| 950000 | 0.238 | 500 | [JSONL line 7](eval.jsonl#L7) | pending upload |
| 1000000 | 0.228 | 500 | [JSONL line 8](eval.jsonl#L8) | pending upload |
| 1050000 | 0.314 | 500 | [JSONL line 9](eval.jsonl#L9) | pending upload |
| 1100000 | 0.298 | 500 | [JSONL line 10](eval.jsonl#L10) | pending upload |
| 1150000 | 0.306 | 500 | [JSONL line 11](eval.jsonl#L11) | pending upload |
| 1200000 | 0.398 | 500 | [JSONL line 12](eval.jsonl#L12) | pending upload |
| 1250000 | 0.408 | 500 | [JSONL line 13](eval.jsonl#L13) | pending upload |
| 1300000 | 0.388 | 500 | [JSONL line 14](eval.jsonl#L14) | pending upload |
| 1350000 | 0.392 | 500 | [JSONL line 15](eval.jsonl#L15) | pending upload |
| 1400000 | 0.412 | 500 | [JSONL line 16](eval.jsonl#L16) | pending upload |
| 1450000 | 0.404 | 500 | [JSONL line 17](eval.jsonl#L17) | pending upload |
| 1500000 | 0.466 | 500 | [JSONL line 18](eval.jsonl#L18) | pending upload |
| 1550000 | 0.446 | 500 | [JSONL line 19](eval.jsonl#L19) | pending upload |
| 1600000 | 0.47 | 500 | [JSONL line 20](eval.jsonl#L20) | pending upload |
| 1650000 | 0.47 | 500 | [JSONL line 21](eval.jsonl#L21) | pending upload |
| 1700000 | 0.526 | 500 | [JSONL line 22](eval.jsonl#L22) | pending upload |
| 1750000 | 0.5 | 500 | [JSONL line 23](eval.jsonl#L23) | pending upload |
| 1800000 | 0.522 | 500 | [JSONL line 24](eval.jsonl#L24) | pending upload |
| 1850000 | 0.486 | 500 | [JSONL line 25](eval.jsonl#L25) | pending upload |

## Continuation boundary

Continues [m4-fourier4-s0](../m4-fourier4-s0/README.md) from update 600000. Network, optimizer, agent RNG and preprocessing are restored. The streaming dataset, sampler and validation set restart. Validation losses across this boundary use different held-out samples. This is not an exact sampler resume.

[Boundary metadata](continuation.jsonl)
