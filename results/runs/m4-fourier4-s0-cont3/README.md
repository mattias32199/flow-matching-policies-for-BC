# m4-fourier4-s0-cont3

Status: **training**.

[Results](../../../RESULTS.md) · [Run catalog](../README.md)

W&B: awaiting authenticated upload (or run has not started).

[Exact configuration](config.json) · [Training and validation log](metrics.jsonl) · [Evaluation log](eval.jsonl) · [Collection log](collections.jsonl) · [Retained dataset counts](dataset.jsonl)

These JSONL records are the complete portable scalar logs, not sampled chart exports. Machine stdout/stderr, credentials, data and checkpoints remain local.

## Evaluations

| Gradient update | Success | Episodes | Record | W&B |
| ---: | ---: | ---: | --- | --- |

No evaluation record exists yet; no success rate is inferred.

## Continuation boundary

Continues [m4-fourier4-s0](../m4-fourier4-s0/README.md) from update 600000. Network, optimizer, agent RNG and preprocessing are restored. The streaming dataset, sampler and validation set restart. Validation losses across this boundary use different held-out samples. This is not an exact sampler resume.

[Boundary metadata](continuation.jsonl)
