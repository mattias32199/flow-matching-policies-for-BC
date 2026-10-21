# Ten-experiment plan

Existing completed and active experiments count toward this budget. Placement is
independent of the scientific plan; do not restart an existing run from scratch.

| Mystery | Condition | Seed | Role |
| --- | --- | ---: | --- |
| 1 | Double task, 10K requested episodes | 0 | Completed loss/control comparison |
| 1 | Double task, 10K requested episodes | 1 | Repeat loss/control comparison |
| 1 | Single task, 10K requested episodes | 0 | Smaller independent dataset |
| 1 | Single task, 50K requested episodes | 0 | Larger independent dataset, data seed 10000 |
| 4 | Handcrafted observations | 0 | Existing active baseline |
| 4 | Standardized observations | 0 | Existing active comparison |
| 4 | Standardized XYZ multiplied by 0.1 | 0 | Existing active comparison |
| 4 | Standardized XYZ multiplied by 10 | 0 | Existing active comparison |
| 4 | Standardized plus XYZ Fourier, frequency 1 | 0 | New representation extension |
| 4 | Standardized plus XYZ Fourier, frequencies 1, 2, 4, 8 | 0 | New representation extension |

One seed per dataset size cannot establish seed robustness. Fourier variants are
extensions to the original reproduction, with no success claim before evaluations.

## Dataset and network contract

Each demonstration example contains a 37-dimensional observation `o` and a
25-by-7 action chunk. Existing action normalization maps the chunk to a flattened
175-dimensional target `A`; Fourier features never change actions.

Fit observation mean and standard deviation on the initial training collection,
then freeze them for training and evaluation:

```text
z = (o - mean) / max(std, 0.001)
xyz = z[[12,13,14, 19,20,21, 28,29,30]]
phi_F(o) = concat(z, sin(2*pi*f*xyz), cos(2*pi*f*xyz) for f in F)
```

The selected coordinates describe the end effector and both blocks. All original
standardized coordinates remain in the input, preserving absolute position
information despite periodic features. Frequencies are fixed, in cycles per
standardized coordinate unit; there are no learned or random projections.
Conditioning width is 37 for standardization, 55 for `fourier1`, and 109 for
`fourier4`. The first layer gains 73,728 or 294,912 weights, respectively; hidden
width/depth are unchanged. This modest parameter-count difference is explicit.

```text
epsilon ~ Normal(0, I_175)
t ~ Beta(1, 1.5)
x_t = (1-t)*epsilon + t*A
network input:  (phi_F(o), x_t, t)
network output: velocity in R^175
training target: A - epsilon
loss: mean squared velocity error
```

The optimizer, action normalization, 4096-wide eight-layer model, batch size1024,
float32 network dtype, 2M updates and evaluation schedule remain unchanged.
Input features are computed on CPU before the existing compiled training update.
The upstream agent's existing flow-time embedding is unchanged.

## Data reuse and scheduling

The two new Fourier consumers share identical raw collections through a separate
cache identity and release each entry after both have loaded it. They retain
private arrays and each uses transition starts once. The four existing active
runs continue their original independent generation; paired seeds do not imply
bitwise-identical simulation across those old runs and the new pair. Separate
data-producer jobs are not automatically launched for the Fourier pair.

New scientific runs should each use one GPU through `scripts/private_slurm.py`,
with opaque job names and owner-only launch/work directories. Existing active
runs require deadline and continuation planning before changing allocations.
Existing checkpoints omit exact sampler/stream state; a restart must not be
described as exact continuation without additional verified recovery support.

## Releasing the current allocation

`scripts/prepare_migration.py` pins the latest complete checkpoint with a hardlink,
copies its configuration, preprocessing and scalar records into a private local
bundle, and checks numeric finiteness, training step and SHA-256. Pinned files
survive the original runner's checkpoint retention. No scheduler jobs are
submitted or cancelled by preparation.

The prepared source checkpoints are at 1.1M updates for handcrafted, standardized
and XYZ multiplied by 0.1, and 800K for XYZ multiplied by 10. More recent unsaved
updates are lost when those processes stop. The completed Mystery 1 run is kept.

Individual continuation runs use a distinct name and `--resume-bundle`. They
restore network weights, optimizer state, agent RNG and the exact saved action
and observation transforms. The update counter continues toward 2M total; it
does not start another 2M updates. Collection indices restart at 2048, separate
from the observed source streams and within the generator's 4096-index limit.
Simulator parallelism uses the current default cap of 2500.

**This is an inexact stream continuation.** Source sampler state and raw held-out
arrays were not checkpointed. New training collections and a replacement held-out
set are generated; diagnostic RNG and sampler restart. Validation losses across
the boundary must not be stitched into an uninterrupted same-dataset curve.
Policy rollout success remains measured under the existing evaluation protocol.
Continuation configuration and exact boundary metadata link to the source run in
the catalog and W&B records. These are segments of four existing scientific
experiments, not four additional conditions.

Preparation leaves nine individual launches: four continuations, three new
Mystery 1 runs and two new Fourier variants. Placement and actual submission are
separate. Do not resume the legacy supervisor's obsolete in-memory queue.

See [RESULTS.md](RESULTS.md) for outcomes and the [matrix](results/matrix.json)
for machine-readable settings.
