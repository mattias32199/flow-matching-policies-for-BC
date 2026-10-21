# m4-handcrafted-s0-cont1

Status: **training**.

[Results](../../../RESULTS.md) · [Run catalog](../README.md)

W&B: [training charts and evaluations](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4/runs/ocb89ddace69a79a65f6)

[Exact configuration](config.json) · [Training and validation log](metrics.jsonl) · [Evaluation log](eval.jsonl) · [Collection log](collections.jsonl) · [Retained dataset counts](dataset.jsonl)

These JSONL records are the complete portable scalar logs, not sampled chart exports. Machine stdout/stderr, credentials, data and checkpoints remain local.

## Evaluations

| Gradient update | Success | Episodes | Record | W&B |
| ---: | ---: | ---: | --- | --- |

No evaluation record exists yet; no success rate is inferred.

## Continuation boundary

Continues [m4-handcrafted-s0](../m4-handcrafted-s0/README.md) from update 1100000. Network, optimizer, RNG and preprocessing are restored. The streaming dataset, sampler and validation set restart; this is not an exact stream resume. Validation losses across this boundary use different held-out samples.

[Boundary metadata](continuation.jsonl)
