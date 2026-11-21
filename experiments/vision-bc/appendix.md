# Appendix: exact settings, compute and run index

Part of the [visual BC study](README.md). Results: [single.md](single.md) · [double.md](double.md) · [explanation.md](explanation.md) · [followup.md](followup.md).
Machine-readable: [runs.csv](runs.csv) (one row per run) and [configs/](configs/) (each run's full `config.json`).

**Contents**
- A1 Compute budget
- A2 Environments and demonstration data
- A3 Data pipelines: fixed datasets, chunk-start interval, replay refresh, streaming
- A4 State inputs and feature transforms
- A5 Visual inputs, frozen embeddings and patch pooling
- A6 Policy head and optimization
- A7 Visual-encoder recipes: frozen, fine-tuned, sim-pretrained, auxiliary loss
- A8 Evaluation protocol
- A9 Diagnostics: probes, saliency, expert takeover, cost benchmark
- A10 Common setup, as originally reported
- A11 Run index (parts A–D)
- A12 Reproduction

## A1. Compute budget

Every job used **one GPU**: NVIDIA RTX A5000 (24 GB) or RTX A6000 (48 GB), on the `deepaklong` Slurm partition.
"GPU-h" is wall-clock time summed over Slurm segments (jobs requeue on preemption or time limit and resume from
checkpoints). It includes periodic evaluation rollouts.

| Part | Runs | GPU-h |
| --- | ---: | ---: |
| A block-single | 34 | 126 |
| B block-double | 36 | 171 |
| C explanation experiments | 31 | 183 |
| D follow-up ([followup.md](followup.md)), incl. 39 support jobs | 178 | 513 |
| **Total** | 279 | **993** |

Part D also ran on the `all` partition and on other GPU models: RTX 6000, 2080 Ti, 3080 Ti, 3090, A6000 Ada and
RTX PRO 6000 Blackwell. GPU-h there are therefore less comparable across runs; the `GPU` column names the models
each run used.

The run index (A11) omits three runs whose launch logs predate the launcher used here (see the note at the end of
A11). Embedding-cache and data-preparation jobs from the first session are also not counted; together they are
under 3 GPU-h.

Typical costs, from the run index:

| Run type | Updates | Batch | Wall-clock |
| --- | ---: | ---: | --- |
| State FBC, JAX (`train_regime.py`, `train.py`) | 1M (double) / 500K (single) | 1024 | ~9.1 h / ~4.5 h (≈31 updates/s + evals) |
| Visual policy on cached frozen embeddings (`train_vis.py --input emb`) | 300K | 1024 | 3.4–3.9 h CLS; 4.5–4.7 h 4×4 patches |
| End-to-end pixels (`train_vis.py --input pixels`), 112 px | 200K | 256 | Impala 3.7 h; ResNet-18 5.3 h; DINOv2-S 14.4 h |
| Encoder pretraining on sim (image, state) (`pretrain_enc.py`) | 40K | 128 states × 3 cameras | DINOv2-S 1.4 h; ResNet-18 scratch 0.6 h |
| Sim data rendering (`dr_gen.py`), 200K states, 30 CPU workers + EGL | — | — | 3–6 min |
| Expert collection, 1K episodes (MJWarp, 1000 parallel worlds) | — | — | block-double 67 s; block-single 29 s |

The run index has a **Passes** column:
- State runs: samples seen ÷ available chunk starts. For replay runs it is per buffer.
- Visual runs: samples seen ÷ training frames (single 235,673, double 508,934).
- Encoder pretraining: samples ÷ 190K training states.

Sampling is uniform with replacement, so passes are expected visits per training sample, not ordered epochs.

## A2. Environments and demonstration data

**Environments** (OCBench, MJWarp backend):

| | `block-single-task1-v0` | `block-double-task1-v0` |
| --- | --- | --- |
| Horizon | 1250 | 2500 |
| Cubes | 1 | 2 |
| State dim | 28 | 37 |
| Action | 7-d joint-space (6 arm + gripper) | same |

The visual variants (`visual-block-*`) render 3 cameras (front, side, wrist) at 224×224 RGB.

**Expert**: the OCBench scripted MJWarp controller (`BlockMjWarpController`). It re-plans from the current simulator
state and makes a deliberate mistake (wrong subgoal) with probability 0.1 per planned move.
- block-double: 79.8% success over 1000 episodes; 90.7% with mistakes disabled (`OCB_MISTAKE_PROB=0`, local edit,
  `patches/ocbench-mistake-prob.patch`).
- Collections: double 78.9% successful, successful episodes 1584 steps on average; single 88.0% successful,
  655 steps.

**Demonstration datasets** (only successful episodes are kept everywhere):

| Dataset | Used by | Content |
| --- | --- | --- |
| OCBench released `visual-block-single-task1-v0` | Part A visual and A-group state runs | 10K episodes. Images every 25 steps (`observation_interval=25`) plus per-step state/qpos/actions. 235,673 training frames (non-final frames of successful episodes). |
| OCBench released `visual-block-double-task1-v0` | Part B visual runs | 10K episodes, images every 25 steps; 508,934 training frames |
| `data/block-*-n10000-d0` (JAX `train.py` fixed mode) | A-group state runs, `v-state-double-*` | 10K episodes collected by `generate_mjwarp_datasets` |
| Collection store `data/collections/<env>-d0/c#####.npz` (`collections_store.py`) | Part B data-regime and Part C state runs | Collections of 1K episodes from `generate_mjwarp_datasets(env, 1000, 0, root_seed=0, collection_idx=i, max_parallel_episodes=1000)`, i.e. the seeding of the streaming pipeline. Stored arrays: observations, actions, rewards, masks, terminals, qpos (fp16). Validation: 128 episodes, collection 0 val split. |

Storage: about 300 MB per double collection; 100 collections (100K episodes) is 21 GB.

## A3. Data pipelines (`scripts/train_regime.py`)

- **Fixed N**: the buffer is collections 0..N/1000−1, concatenated once. The action normalizer and feature
  statistics are fit on this buffer.
- **Chunk-start interval k**: a training sample is a 25-step action chunk starting at episode step t. Only
  t ≡ 0 (mod k) are allowed: allowed index = idx where (idx − episode_start) mod k == 0. Chunks are truncated at
  episode end, as upstream. k = 25 reproduces exactly the frames that exist in the visual datasets.
- **Replay refresh** (`--refresh-every R --refresh-collections 1`): a FIFO window of N/1000 collections.
  - Every R updates, one new collection (index 1000 + j) is appended and the oldest is evicted.
  - New collections are generated ahead of time by a background subprocess on the same GPU (two in flight),
    written to `runs/<run>/stream/`, and deleted on eviction.
  - Normalizer and feature statistics stay frozen from the initial window.
  - Episodes seen = N + 1000·⌊updates/R⌋.
- **One-use streaming** (Mystery-4 reference runs, `train.py --mode streaming`): 1K-episode collections, every chunk
  start used exactly once (random permutation), then the next collection. About 800K episodes are consumed per
  1M updates.
- Sampling within a buffer: indices are drawn uniformly with replacement from the allowed starts, seeded by
  (seed, start step), so resumed runs stay reproducible.

## A4. State inputs and feature transforms (`scripts/train.py`: `Features`, `extra_features`)

**Raw OCBench block observation** (28-d single, 37-d double):

| Dims | Content |
| --- | --- |
| 0:6 | arm joint positions |
| 6:12 | arm joint velocities |
| 12:15 | effector (pinch site) position, (p − [0.425, 0, 0]) × 10 |
| 15:17 | cos/sin effector yaw |
| 17 | gripper opening × 3 |
| 18 | gripper contact |
| per cube, 9 dims | position (same ×10 scaling), quaternion (4), cos/sin yaw |

**Base transforms:**

| Mode | Transform |
| --- | --- |
| `handcrafted` | raw observation |
| `standardized` | per-dimension z-score with training mean/std (std floored at 1e-3) |
| `xyz0.1`, `xyz10` | z-score, then the effector and cube xyz dims multiplied by 0.1 or 10 |
| `fourier1`, `fourier4` | z-score plus sin/cos(2πf·z_xyz) for f ∈ {1} or {1, 2, 4, 8} appended |

**Hypothesis extras** (`base+extra+...`). Each extra is computed from the raw observation, z-scored with its own
training statistics, and appended:

| Extra | Definition (per cube unless stated) | Width per cube |
| --- | --- | ---: |
| `rel` | Δ = cube_pos − effector_pos, world frame | 3 |
| `egopos` | Δ rotated by −ψ_eff: [cos ψ Δx + sin ψ Δy, −sin ψ Δx + cos ψ Δy, Δz] | 3 |
| `relyaw` | sin and cos of 4·(ψ_cube − ψ_eff); the factor 4 encodes the cube's 90° symmetry | 2 |
| `ego` | `egopos` + `relyaw` | 5 |
| `pair` | cube_j − cube_0 and its norm (block-double only), once per task | 4 |
| `randproj` | control: tanh(W z), z = z-scored state, W ~ N(0, 1/dim) fixed (seed 12345), width = `ego` width | 5 per cube |

ψ is taken from atan2(sin, cos) of the observation's yaw entries.

## A5. Visual inputs, frozen embeddings and patch pooling

- **Cameras**: front, side and wrist, 224×224 RGB. One shared encoder per policy; per-camera features are
  concatenated in that order.
- **Frozen generic encoders** (`vis_common.ENC`, timm weights):

| Encoder | timm model |
| --- | --- |
| `dinov2_s` | `vit_small_patch14_dinov2.lvd142m` (img_size 224) |
| `dinov2_b` | `vit_base_patch14_dinov2.lvd142m` |
| `clip_b16` | `vit_base_patch16_clip_224.openai` |
| `mae_b16` | `vit_base_patch16_224.mae` |
| `resnet50_ce` | `resnet50.a1_in1k` |
| `simclr_r50` | ResNet-50 with SimCLR weights; no mean/std normalization |
| `random_vit_s` | DINOv2-S architecture with random init (seed 0) |

  Each encoder gets its own timm mean/std normalization at its native input size (224). Embeddings are computed
  once from the original 224 px frames (antialiased resize when needed), stored as fp16
  `data/emb/<env>/<encoder>/{split}_emb.npy`.
- **Readouts**:
  - **CLS / global**: timm `num_classes=0` pooled output, i.e. the class token for DINOv2/CLIP ViTs and global
    average pooling for ResNets.
  - **Patch pooling `p2` / `p4`**: last-layer patch tokens (prefix tokens dropped), reshaped to the 16×16 grid,
    adaptive-average-pooled to 2×2 or 4×4, flattened: 4×384 or 16×384 per camera.
- **Embedding scaling** (`--scale`): `std` = per-dimension z-score with training statistics (default); `raw`; or
  z-score multiplied by 0.1 or 10.
- **Sim-pretrained encoders** (`pre:<run>`): backbone weights from `runs/<run>/encoder.pt`, input 112 px, global
  feature (CLS).
- **Pixel policies** (end-to-end): frames resized to 112 px; DrQ random shift (replicate-pad 4 px, then a random
  crop back to 112) during training; encoder output passed through a non-affine LayerNorm per camera.
- **Storage per frame (3 cameras)**: uint8 112 px 113 KB; fp16 CLS-384 2.3 KB; 2×2 patches 9.2 KB; 4×4 patches
  37 KB. The 4×4 embeddings for visual-block-double (509K frames) are 27 GB and are kept on the GPU, which needs an
  A6000.

## A6. Policy head and optimization

**Flow-matching BC**, a port of the OCBench FBC actor:
- `ResidualMLP`: input Linear → LayerNorm, then 4 residual blocks of (Linear, LayerNorm, Linear, LayerNorm) with
  GELU-tanh, width 4096 (8 hidden layers), Xavier init.
- Input = observation features ⊕ noisy action chunk (25×7 = 175) ⊕ 65-d time features [t, cos(2πkt), sin(2πkt),
  k = 1..32].
- Training: x_t = (1−t)x₀ + t·a with x₀ ~ N(0, I); t ~ Beta(1, 1.5) via inverse CDF; MSE on the velocity a − x₀.
- Sampling: 10 Euler steps from Gaussian noise.

**Actions**: OCBench `ActionNormalizer.fit(actions, 1, 5)` (1st/99th-percentile scaling, clip at 5), fit on
training actions. Chunks are 25 steps and executed open-loop at evaluation.

**Optimizers:**

| Trainer | Optimizer | lr | Batch | Schedule / extras |
| --- | --- | --- | ---: | --- |
| JAX FBC (`train.py`, `train_regime.py`) | Adam | 1e-4 | 1024 | constant, fp32 |
| `train_vis.py`, frozen embeddings | Adam (eps 1e-8) | 1e-4 | 1024 | constant, TF32, no weight decay |
| `train_vis.py`, pixels | Adam | head 1e-4; encoder 1e-4 (ResNet-18), 1e-5 (DINOv2-S); Impala trained at 1e-4 | 256 | constant, bf16 autocast, DrQ shift |

Checkpointing: every 5K updates (visual) or 25K (state). On SIGTERM/USR1 the trainer saves and the job requeues.

## A7. Visual-encoder recipes

**Frozen** (Part A group B, C; Part B transfer): no encoder training. Stage 1 caches embeddings (A5). Stage 2 trains
the flow head for 300K updates on standardized embeddings (≈1300 passes over single frames, ≈600 over double).

**End-to-end fine-tuning** (Part A group D): the encoder and head are trained jointly from step 0 on 112 px pixels,
batch 256, 200K updates (≈217 passes over the 235,673 single-task frames), lr as in A6, DrQ shift.

| Encoder | Init |
| --- | --- |
| Impala-small | random |
| ResNet-18 | `resnet18.a1_in1k` |
| DINOv2-S | `vit_small_patch14_dinov2` at img_size 112 |

**Sim-pretrained encoders** (Part A group F, Part B transfer, Part C target ablation):
1. **Render the data** (`dr_gen.py`): 200K simulator states.
   - States are sampled uniformly from the demo training qpos.
   - In 50% of samples every cube is re-placed uniformly in the demo xy range (cubes ≥ 8 cm apart on double),
     resting on the table, with a random yaw.
   - Arm joints get N(0, 0.05 rad) noise; gripper opening ~ U(0, 0.8).
   - `mj_forward` only (no physics stepping). Render 3 cameras at 224 px, then antialiased resize to 112 px.
   - Variants: `nodr`, nominal appearance; `dr` (single only):
     - geom colours blended toward random hues (mix 0.3–1);
     - material colours ×U(0.4, 1.6);
     - light diffuse ×U(0.3, 1.5), ambient +U(0, 0.3), light position +N(0, 0.5 m);
     - headlight randomized;
     - camera position +N(0, 2 cm) (wrist 0.5 cm) and rotation N(0, 3°) (wrist 1.5°).
2. **Targets**, standardized per dimension:
   - per cube: position in each camera frame (9), in the pinch/gripper frame (3), sin/cos(4·yaw) (2), world xyz (3);
   - plus pinch xyz (3), gripper opening (1), arm joints (6).
   - 27-d on single, 44-d on double.
   - Target groups for ablations (`--groups`): world = cube_xyz, pinch_xyz, gripper, arm_qpos; egocentric =
     cube_in_camera, cube_in_gripper, cube_yaw.
3. **Regression** (`pretrain_enc.py`):
   - The timm backbone is shared over the 3 cameras. Head: MLP 3D→1024→1024→T, GELU + LayerNorm.
   - Loss: MSE on standardized targets.
   - AdamW (weight decay 1e-4); lr: head 3e-4, backbone 5e-5 (from scratch: 3e-4).
   - 1000-step linear warmup, then cosine decay to 0 at 40K steps.
   - Batch 128 states (384 images); DrQ shift (pad 4); bf16 autocast; gradient clipping 1.0.
   - Last 5% of samples (10K) held out for R²; about 27 passes over the 190K training states.
4. **Freeze**: save the backbone only (`encoder.pt`), cache demo-frame embeddings at 112 px.
5. **BC**: 300K updates on standardized embeddings, as for frozen encoders.

**Auxiliary state regression** (Part A group E):
- Head MLP features→512→21 regressing the (single-task) qpos, MSE, weight 1.
- Labels only for a fixed random fraction of training episodes (1%, 10%, 100%).
- Auxiliary batch: max(32, batch/4) for pixels, 256 for embeddings.
- With frozen encoders it only acts through a trainable adapter (MLP emb→1024→1024→256).

## A8. Evaluation protocol

- Batched MJWarp rollouts from evaluation seeds. A new observation every 25 steps; each 25-step action chunk is
  executed open-loop.
- State policies: 500 episodes per evaluation; visual policies: 300 (pixels rendered at 224 by MJWarp and
  preprocessed exactly as in training).
- Evaluations every 50K updates. Reported **success = mean of the last 3 evaluations**; **best** = maximum over all
  evaluations.
- Neighbouring evaluations differ by about ±0.05; most runs are single-seed.
- Data-regime tables additionally report means over the evaluations at 450–600K and 850K–1M updates.

## A9. Diagnostics

- **Ridge probes** (`embed_probe.py`): frozen 3-camera embeddings (z-scored) → demo state groups (arm qpos, gripper,
  cube xyz). λ ∈ {1e-3, 1e-2, 0.1, 1, 10, 100}. The reported R² is for the λ with the best mean R² on the validation shards; there is no separate test split, so values are slightly optimistic and comparable across encoders.
- **Saliency** (`saliency.py`):
  - Gradient of ‖sampled action chunk‖² with respect to input pixels (summed over channels), with fixed flow noise,
    on 256 validation frames.
  - The cube mask is a saturated-blue colour threshold dilated by 2 px.
  - Reported: `cube_ratio` = mean saliency on cube pixels ÷ image mean.
- **Expert takeover** (`takeover.py`):
  - Identical initial states for every k. The student acts for k steps (open-loop chunks).
  - Then a freshly constructed `BlockMjWarpController` re-plans from the current state, with
    `max_steps = H − k`.
  - 500 episodes per (student, k); expert alone: 1000.
- **Cost benchmark** (`bench_vis_cost.py`): OCBench in-loop rendering at observation interval 25 and 5 vs state-only
  collection (200 / 100 episodes), CPU/EGL re-render rate, DINOv2-S embedding throughput.

## A10. Common setup, as originally reported

**Environments**
- **OCBench `visual-block-single-task1-v0`**: one cube, horizon 1250. Three cameras (front, side, wrist); RGB is
  224 px, resized to 112 px for pixel policies.
- **`block-double-task1-v0` / `visual-block-double-task1-v0`**: two cubes, horizon 2500.

**Demonstrations**
- **Single**: OCBench's released visual dataset, 10K expert episodes. Images exist every 25 steps; only
  successful episodes are kept, giving 235,673 training frames.
- **Double (visual)**: the released visual-block-double dataset, also 10K episodes with images every 25 steps.
- **Double (state, data-regime study)**: freshly collected with the OCBench MJWarp scripted expert. See part B.

**Policy**
- Flow-matching BC, ported to PyTorch from OCBench FBC for the visual runs (state runs use the original JAX FBC).
- Network: residual MLP 8 × 4096 with LayerNorm. It predicts 25 × 7 action chunks.
- Flow: time ~ Beta(1, 1.5), 10 Euler steps at inference.
- Optimizer: Adam, learning rate 1e-4, batch 1024. Pixel policies use batch 256 and an encoder learning rate of
  1e-4 (DINOv2-S fine-tune: 1e-5). Pixel policies also use a DrQ random shift.
- Actions are normalized with the OCBench `ActionNormalizer` (1st/99th percentiles, clip at 5).

**Visual observation**
- All three cameras go through the same encoder; the per-camera features are concatenated.
- Frozen encoders are run once and cached as fp16 features (`data/emb/<env>/<encoder>`).
- Features are standardized per dimension unless stated otherwise.

**Evaluation**
- Batched MJWarp rollouts. The policy sees a new observation every 25 steps and executes each chunk open-loop;
  state policies also act on 25-step chunks.
- 300 episodes per evaluation for visual runs and 500 for state runs, every 50K updates.
- "Success" below is the mean of the **last three** evaluations; "best" is the maximum over all evaluations.

**Infrastructure**
- Slurm partition `deepaklong` (A6000 / A5000). Jobs requeue automatically on preemption or time limit,
  checkpoint every 5K–25K updates, and resume the same W&B run.
- Launchers: `scripts/launch_m4v.py` (part A), `scripts/launch_regime.py` (part B).

---


## A11. Run index

Generated by `scripts/export_vision_runs.py`.

### Part A: block-single

| Run | Trainer | Key settings | Updates | Batch | Passes | GPU-h | GPU | Success | Best |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `v-B-clip_b16-s0` | train_vis | input=emb, encoder=clip_b16, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 1303.5 | 3.57 | A5000 | 0.761 | 0.79 |
| `v-B-dinov2_b-s0` | train_vis | input=emb, encoder=dinov2_b, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 1303.5 | 3.55 | A5000/A6000 | 0.714 | 0.75 |
| `v-B-dinov2_s-s0` | train_vis | input=emb, encoder=dinov2_s, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 1303.5 | 3.39 | A5000 | 0.647 | 0.677 |
| `v-B-mae_b16-s0` | train_vis | input=emb, encoder=mae_b16, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 1303.5 | 3.56 | A5000 | 0.732 | 0.76 |
| `v-B-random_vit_s-s0` | train_vis | input=emb, encoder=random_vit_s, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 1303.5 | 3.54 | A5000 | 0.006 | 0.01 |
| `v-B-resnet50_ce-s0` | train_vis | input=emb, encoder=resnet50_ce, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 1303.5 | 3.74 | A5000 | 0.654 | 0.673 |
| `v-B-simclr_r50-s0` | train_vis | input=emb, encoder=simclr_r50, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 1303.5 | 3.72 | A5000 | 0.342 | 0.37 |
| `v-C-dinov2_b-raw-s0` | train_vis | input=emb, encoder=dinov2_b, scale=raw, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 1303.5 | 3.59 | A5000 | 0.694 | 0.713 |
| `v-C-dinov2_b-x0.1-s0` | train_vis | input=emb, encoder=dinov2_b, scale=x0.1, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 1303.5 | 3.59 | A5000 | 0.754 | 0.773 |
| `v-C-dinov2_b-x10-s0` | train_vis | input=emb, encoder=dinov2_b, scale=x10, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 1303.5 | 3.59 | A5000 | 0.782 | 0.797 |
| `v-C-dinov2_s_p2-s0` | train_vis | input=emb, encoder=dinov2_s_p2, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 1303.5 | 3.4 | A6000 | 0.831 | 0.88 |
| `v-C-dinov2_s_p4-s0` | train_vis | input=emb, encoder=dinov2_s_p4, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 1303.5 | 4.72 | A5000 | 0.902 | 0.917 |
| `v-D-dinov2_s-ft-s0` | train_vis | input=pixels, encoder=dinov2_s, scale=std, finetune=True, lr=0.0001, enc_lr=1e-05, img=112 | 200000 | 256 | 217.25 | 14.37 | A5000 | 0.877 | 0.953 |
| `v-D-impala-scratch-s0` | train_vis | input=pixels, encoder=impala_small, scale=std, finetune=True, lr=0.0001, enc_lr=1e-05, img=112 | 200000 | 256 | 217.25 | 3.7 | A5000 | 0.859 | 0.89 |
| `v-D-resnet18-ft-s0` | train_vis | input=pixels, encoder=resnet18, scale=std, finetune=True, lr=0.0001, enc_lr=0.0001, img=112 | 200000 | 256 | 217.25 | 5.25 | A5000 | 0.901 | 0.927 |
| `v-E-dinov2_b-adapter-aux0-s0` | train_vis | input=emb, encoder=dinov2_b, scale=std, adapter=True, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 1303.5 | 3.42 | A5000 | 0.603 | 0.637 |
| `v-E-dinov2_b-adapter-aux0.01-s0` | train_vis | input=emb, encoder=dinov2_b, scale=std, adapter=True, aux_frac=0.01, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 1303.5 | 3.55 | A5000 | 0.576 | 0.577 |
| `v-E-dinov2_b-adapter-aux0.1-s0` | train_vis | input=emb, encoder=dinov2_b, scale=std, adapter=True, aux_frac=0.1, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 1303.5 | 3.77 | A5000 | 0.606 | 0.62 |
| `v-E-dinov2_b-adapter-aux1-s0` | train_vis | input=emb, encoder=dinov2_b, scale=std, adapter=True, aux_frac=1.0, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 1303.5 | 3.55 | A5000 | 0.611 | 0.617 |
| `v-E-impala-aux0.01-s0` | train_vis | input=pixels, encoder=impala_small, scale=std, finetune=True, aux_frac=0.01, lr=0.0001, enc_lr=1e-05, img=112 | 200000 | 256 | 217.25 | 4.14 | A5000 | 0.816 | 0.913 |
| `v-E-impala-aux0.1-s0` | train_vis | input=pixels, encoder=impala_small, scale=std, finetune=True, aux_frac=0.1, lr=0.0001, enc_lr=1e-05, img=112 | 200000 | 256 | 217.25 | 3.69 | A5000/A6000 | 0.691 | 0.78 |
| `v-E-impala-aux1-s0` | train_vis | input=pixels, encoder=impala_small, scale=std, finetune=True, aux_frac=1.0, lr=0.0001, enc_lr=1e-05, img=112 | 200000 | 256 | 217.25 | 4.47 | A5000 | 0.698 | 0.767 |
| `v-F-bc-enc-dr-dinov2s-s0` | train_vis | input=emb, encoder=pre:v-F-enc-dr-dinov2s, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 1303.5 | 3.44 | A5000 | 0.968 | 0.973 |
| `v-F-bc-enc-dr-resnet18-s0` | train_vis | input=emb, encoder=pre:v-F-enc-dr-resnet18, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 1303.5 | 3.52 | A5000 | 0.947 | 0.96 |
| `v-F-bc-enc-nodr-dinov2s-s0` | train_vis | input=emb, encoder=pre:v-F-enc-nodr-dinov2s, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 1303.5 | 3.5 | A5000 | 0.967 | 0.973 |
| `v-F-drgen` | job |  |  |  |  | 0.1 | A6000 |  |  |
| `v-F-embed` | job |  |  |  |  | 0.09 | A5000 |  |  |
| `v-F-enc-dr-dinov2s` | encoder_pretrain | lr=0.0003, enc_lr=5e-05, data=dr, arch=dinov2_s | 40000 | 128 | 26.95 | 1.42 | A5000 |  |  |
| `v-F-enc-dr-resnet18` | encoder_pretrain | lr=0.0003, enc_lr=5e-05, data=dr, arch=resnet18, scratch=True | 40000 | 128 | 26.95 | 0.62 | A5000 |  |  |
| `v-F-enc-nodr-dinov2s` | encoder_pretrain | lr=0.0003, enc_lr=5e-05, data=nodr, arch=dinov2_s | 40000 | 128 | 26.95 | 1.4 | A5000 |  |  |
| `v-state-single-fourier4-n10000-int25-s0` | train.py | features=fourier4, episodes=10000, start_interval=25 | 500000 | 1024 | 2191.51 | 4.52 | A5000 | 0.855 | 0.878 |
| `v-state-single-standardized-n10000-int25-s0` | train.py | features=standardized, episodes=10000, start_interval=25 | 500000 | 1024 | 2191.51 | 4.46 | A5000 | 0.846 | 0.89 |
| `v-state-single-xyz0.1-n10000-int25-s0` | train.py | features=xyz0.1, episodes=10000, start_interval=25 | 500000 | 1024 | 2191.51 | 4.39 | A5000 | 0.823 | 0.85 |
| `v-state-single-xyz10-n10000-int25-s0` | train.py | features=xyz10, episodes=10000, start_interval=25 | 500000 | 1024 | 2191.51 | 4.42 | A5000 | 0.867 | 0.904 |

### Part B: block-double

| Run | Trainer | Key settings | Updates | Batch | Passes | GPU-h | GPU | Success | Best |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `r-fixed-n100k` | train_regime | features=handcrafted, base=100000, start_interval=1 | 1000000 | 1024 | 8.19 | 9.06 | A5000 | 0.569 | 0.62 |
| `r-fixed-n100k-int25` | train_regime | features=handcrafted, base=100000, start_interval=25 | 1000000 | 1024 | 203.27 | 9.34 | A5000/A6000 | 0.578 | 0.632 |
| `r-fixed-n100k-int5` | train_regime | features=handcrafted, base=100000, start_interval=5 | 1000000 | 1024 | 40.91 | 9.17 | A5000 | 0.456 | 0.554 |
| `r-fixed-n10k` | train_regime | features=handcrafted, base=10000, start_interval=1 | 1000000 | 1024 | 82.13 | 9.09 | A5000 | 0.313 | 0.322 |
| `r-fixed-n10k-int5` | train_regime | features=handcrafted, base=10000, start_interval=5 | 1000000 | 1024 | 410.13 | 9.09 | A5000 | 0.207 | 0.268 |
| `r-fixed-n1k` | train_regime | features=handcrafted, base=1000, start_interval=1 | 1000000 | 1024 | 807.59 | 9.52 | A5000 | 0.008 | 0.016 |
| `r-fixed-n30k` | train_regime | features=handcrafted, base=30000, start_interval=1 | 1000000 | 1024 | 27.4 | 9.34 | A5000 | 0.419 | 0.488 |
| `r-fixed-n30k-int25` | train_regime | features=handcrafted, base=30000, start_interval=25 | 1000000 | 1024 | 679.82 | 9.09 | A5000 | 0.42 | 0.446 |
| `r-fixed-n30k-int5` | train_regime | features=handcrafted, base=30000, start_interval=5 | 1000000 | 1024 | 136.82 | 9.39 | A5000 | 0.471 | 0.478 |
| `r-fixed-n3k` | train_regime | features=handcrafted, base=3000, start_interval=1 | 1000000 | 1024 | 274.77 | 9.11 | A5000 | 0.037 | 0.05 |
| `r-replay-b10k-r10k` | train_regime | features=handcrafted, base=10000, start_interval=1, refresh_every=10000 | 1000000 | 1024 | 82.24 | 10.91 | A5000 | 0.576 | 0.636 |
| `r-replay-b10k-r2k` | train_regime | features=handcrafted, base=10000, start_interval=1, refresh_every=2000 | 1000000 | 1024 | 82.03 | 18.92 | A5000 | 0.593 | 0.628 |
| `r-replay-b30k-r5k-int25` | train_regime | features=handcrafted, base=30000, start_interval=25, refresh_every=5000 | 1000000 | 1024 | 677.51 | 13.51 | A5000/A6000 | 0.505 | 0.57 |
| `regime-B-sim` | support job |  |  |  |  | 0.14 | A6000 |  |  |
| `regime-B-torch` | support job |  |  |  |  | 0.01 | A5000 |  |  |
| `regime-H-embed-dbl` | support job |  |  |  |  | 0.22 | A5000 |  |  |
| `regime-H-embed-sgl` | support job |  |  |  |  | 0.07 | A5000 |  |  |
| `regime-H-prefill-single` | support job |  |  |  |  | 0.1 | A5000 |  |  |
| `regime-P0` | support job |  |  |  |  | 0.18 | A5000 |  |  |
| `regime-P1` | support job |  |  |  |  | 0.38 | A5000 |  |  |
| `regime-P2` | support job |  |  |  |  | 0.55 | A5000 |  |  |
| `regime-P3` | support job |  |  |  |  | 0.74 | A5000 |  |  |
| `regime-T-expert` | support job |  |  |  |  | 0.6 | A5000 |  |  |
| `regime-V-prep` | support job |  |  |  |  | 0.48 | A5000 |  |  |
| `regime-X-drgen` | support job |  |  |  |  | 0.06 | A5000 |  |  |
| `regime-X-embed-F` | support job |  |  |  |  | 0.14 | A5000 |  |  |
| `regime-X-embed-p4` | support job |  |  |  |  | 0.29 | rtx6000 |  |  |
| `v-dbl-F-bc-enc-nodr-dinov2s-s0` | train_vis | input=emb, encoder=pre:v-dbl-F-enc-nodr-dinov2s, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 603.61 | 3.55 | A5000 | 0.802 | 0.827 |
| `v-dbl-F-enc-nodr-dinov2s` | encoder_pretrain | lr=0.0003, enc_lr=5e-05, data=nodr, arch=dinov2_s | 40000 | 128 | 26.95 | 1.4 | A5000 |  |  |
| `v-dbl-clip_b16-s0` | train_vis | input=emb, encoder=clip_b16, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 603.61 | 3.7 | A5000 | 0.344 | 0.36 |
| `v-dbl-dinov2_b-s0` | train_vis | input=emb, encoder=dinov2_b, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 603.61 | 3.67 | A5000 | 0.317 | 0.35 |
| `v-dbl-dinov2_s-s0` | train_vis | input=emb, encoder=dinov2_s, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 603.61 | 3.6 | A5000 | 0.25 | 0.26 |
| `v-dbl-dinov2_s_p2-s0` | train_vis | input=emb, encoder=dinov2_s_p2, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 603.61 | 3.89 | A5000 | 0.444 | 0.473 |
| `v-dbl-dinov2_s_p4-s0` | train_vis | input=emb, encoder=dinov2_s_p4, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 603.61 | 4.49 | A6000 | 0.696 | 0.71 |
| `v-dbl-random_vit_s-s0` | train_vis | input=emb, encoder=random_vit_s, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 603.61 | 3.47 | A5000 | 0.0 | 0.0 |
| `v-dbl-resnet50_ce-s0` | train_vis | input=emb, encoder=resnet50_ce, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 603.61 | 3.71 | A5000/A6000 | 0.16 | 0.177 |

### Part C: explanation experiments

| Run | Trainer | Key settings | Updates | Batch | Passes | GPU-h | GPU | Success | Best |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `h-dbl-bc-nodr-dinov2s-ego-s0` | train_vis | input=emb, encoder=pre:h-dbl-enc-nodr-dinov2s-ego, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 603.61 | 3.4 | A5000 | 0.776 | 0.8 |
| `h-dbl-bc-nodr-dinov2s-world-s0` | train_vis | input=emb, encoder=pre:h-dbl-enc-nodr-dinov2s-world, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 603.61 | 3.55 | A5000 | 0.691 | 0.727 |
| `h-dbl-enc-nodr-dinov2s-ego` | encoder_pretrain | lr=0.0003, enc_lr=5e-05, data=nodr, arch=dinov2_s, groups=cube_in_camera,cube_in_gripper,cube_yaw | 40000 | 128 | 26.95 | 1.44 | A5000 |  |  |
| `h-dbl-enc-nodr-dinov2s-world` | encoder_pretrain | lr=0.0003, enc_lr=5e-05, data=nodr, arch=dinov2_s, groups=cube_xyz,pinch_xyz,gripper,arm_qpos | 40000 | 128 | 26.95 | 1.4 | A5000 |  |  |
| `h-double-n10k-int25-standardized-s0` | train_regime | features=standardized, base=10000, start_interval=25 | 1000000 | 1024 | 2037.88 | 8.99 | A5000 | 0.02 | 0.024 |
| `h-double-n10k-int25-standardized_ego_pair-s0` | train_regime | features=standardized+ego+pair, base=10000, start_interval=25 | 1000000 | 1024 | 2037.88 | 9.42 | A5000 | 0.031 | 0.044 |
| `h-double-n10k-int25-standardized_randproj-s0` | train_regime | features=standardized+randproj, base=10000, start_interval=25 | 1000000 | 1024 | 2037.88 | 9.19 | A5000 | 0.018 | 0.024 |
| `h-double-n30k-int25-standardized-s0` | train_regime | features=standardized, base=30000, start_interval=25 | 1000000 | 1024 | 679.82 | 9.09 | A5000 | 0.1 | 0.106 |
| `h-double-n30k-int25-standardized_ego_pair-s0` | train_regime | features=standardized+ego+pair, base=30000, start_interval=25 | 1000000 | 1024 | 679.82 | 9.19 | A5000 | 0.13 | 0.162 |
| `h-double-n30k-int25-standardized_randproj-s0` | train_regime | features=standardized+randproj, base=30000, start_interval=25 | 1000000 | 1024 | 679.82 | 9.31 | A5000 | 0.125 | 0.144 |
| `h-double-n30k-int25-standardized_rel_pair-s0` | train_regime | features=standardized+rel+pair, base=30000, start_interval=25 | 1000000 | 1024 | 679.82 | 9.12 | A5000 | 0.205 | 0.216 |
| `h-sgl-bc-nodr-dinov2s-ego-s0` | train_vis | input=emb, encoder=pre:h-sgl-enc-nodr-dinov2s-ego, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 1303.5 | 3.4 | A5000 | 0.931 | 0.947 |
| `h-sgl-bc-nodr-dinov2s-world-s0` | train_vis | input=emb, encoder=pre:h-sgl-enc-nodr-dinov2s-world, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 1303.5 | 3.35 | A5000 | 0.892 | 0.893 |
| `h-sgl-enc-nodr-dinov2s-ego` | encoder_pretrain | lr=0.0003, enc_lr=5e-05, data=nodr, arch=dinov2_s, groups=cube_in_camera,cube_in_gripper,cube_yaw | 40000 | 128 | 26.95 | 1.35 | A5000 |  |  |
| `h-sgl-enc-nodr-dinov2s-world` | encoder_pretrain | lr=0.0003, enc_lr=5e-05, data=nodr, arch=dinov2_s, groups=cube_xyz,pinch_xyz,gripper,arm_qpos | 40000 | 128 | 26.95 | 1.42 | A5000 |  |  |
| `h-single-n10k-int25-standardized-s0` | train_regime | features=standardized, base=10000, start_interval=25 | 500000 | 1024 | 2180.21 | 4.52 | A5000 | 0.856 | 0.884 |
| `h-single-n10k-int25-standardized-s1` | train_regime | features=standardized, base=10000, start_interval=25, seed=1 | 500000 | 1024 | 2180.21 | 4.4 | A5000 | 0.849 | 0.882 |
| `h-single-n10k-int25-standardized_ego-s0` | train_regime | features=standardized+ego, base=10000, start_interval=25 | 500000 | 1024 | 2180.21 | 4.7 | A5000 | 0.96 | 0.97 |
| `h-single-n10k-int25-standardized_ego-s1` | train_regime | features=standardized+ego, base=10000, start_interval=25, seed=1 | 500000 | 1024 | 2180.21 | 4.4 | A5000/A6000 | 0.959 | 0.966 |
| `h-single-n10k-int25-standardized_randproj-s0` | train_regime | features=standardized+randproj, base=10000, start_interval=25 | 500000 | 1024 | 2180.21 | 4.56 | A5000 | 0.859 | 0.89 |
| `h-single-n10k-int25-standardized_rel-s0` | train_regime | features=standardized+rel, base=10000, start_interval=25 | 500000 | 1024 | 2180.21 | 4.51 | A5000 | 0.865 | 0.912 |
| `h2-double-n100k-int25-handcrafted_ego_pair-s0` | train_regime | features=handcrafted+ego+pair, base=100000, start_interval=25 | 1000000 | 1024 | 203.27 | 9.14 | A5000 | 0.769 | 0.798 |
| `h2-double-n10k-int25-handcrafted-s0` | train_regime | features=handcrafted, base=10000, start_interval=25 | 1000000 | 1024 | 2037.88 | 9.07 | A5000 | 0.112 | 0.142 |
| `h2-double-n10k-int25-handcrafted_ego_pair-s0` | train_regime | features=handcrafted+ego+pair, base=10000, start_interval=25 | 1000000 | 1024 | 2037.88 | 8.92 | A5000 | 0.094 | 0.124 |
| `h2-double-n30k-int25-handcrafted_ego_pair-s0` | train_regime | features=handcrafted+ego+pair, base=30000, start_interval=25 | 1000000 | 1024 | 679.82 | 9.21 | A5000 | 0.379 | 0.422 |
| `h2-double-n30k-int25-handcrafted_randproj-s0` | train_regime | features=handcrafted+randproj, base=30000, start_interval=25 | 1000000 | 1024 | 679.82 | 9.07 | A5000 | 0.364 | 0.41 |
| `h2-double-n30k-int25-handcrafted_rel_pair-s0` | train_regime | features=handcrafted+rel+pair, base=30000, start_interval=25 | 1000000 | 1024 | 679.82 | 8.97 | A5000 | 0.463 | 0.512 |
| `h3-single-n10k-int25-standardized_egopos-s0` | train_regime | features=standardized+egopos, base=10000, start_interval=25 | 500000 | 1024 | 2180.21 | 4.54 | A5000 | 0.816 | 0.872 |
| `h3-single-n10k-int25-standardized_egopos-s1` | train_regime | features=standardized+egopos, base=10000, start_interval=25, seed=1 | 500000 | 1024 | 2180.21 | 4.47 | A5000 | 0.869 | 0.886 |
| `h3-single-n10k-int25-standardized_relyaw-s0` | train_regime | features=standardized+relyaw, base=10000, start_interval=25 | 500000 | 1024 | 2180.21 | 4.61 | A5000 | 0.955 | 0.97 |
| `h3-single-n10k-int25-standardized_relyaw-s1` | train_regime | features=standardized+relyaw, base=10000, start_interval=25, seed=1 | 500000 | 1024 | 2180.21 | 4.55 | A5000 | 0.969 | 0.982 |


### Part D: follow-up (Q, G, S, R)

Success is the 1000-rollout final evaluation where `Rollouts` is filled, otherwise the mean of the last three
evaluations; bowling reports the episode return. `support job` rows are pair rendering (`dr_gen.py`) and embedding
caches (`embed_probe.py`); their GPU-h are included in the totals.

| Run | Task | Trainer | Key settings | Updates | Batch | GPU-h | GPU | Success | Rollouts |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| `g-bw-enc-nodr-dinov2s` | bowling | encoder_pretrain | lr=0.0003, enc_lr=5e-05, data=nodr, arch=dinov2_s | 40000 | 128 | 1.35 | A5000 |  |  |
| `g-bw-state-n10k-int25-s0` | bowling | train_regime | features=handcrafted, base=10000, start_interval=25 | 500000 | 1024 | 4.49 | A5000/A6000 | 0.399 |  |
| `g-bw-state-n30k-int25-s0` | bowling | train_regime | features=handcrafted, base=30000, start_interval=25 | 500000 | 1024 | 4.07 | A6000 | 0.329 |  |
| `g-bw-vis-dinov2_s_p4-s0` | bowling | train_vis | input=emb, encoder=dinov2_s_p4, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 4.32 | A6000 | 0.379 | 1000 |
| `g-bw-vis-pre-s0` | bowling | train_vis | input=emb, encoder=pre:g-bw-enc-nodr-dinov2s, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.2 | A6000 | 0.328 | 1000 |
| `g-hd-state-n10k-int25-s0` | hanoi-double | train_regime | features=handcrafted, base=10000, start_interval=25 | 1000000 | 1024 | 8.47 | A6000 | 0.157 |  |
| `g-hd-state-n30k-int25-s0` | hanoi-double | train_regime | features=handcrafted, base=30000, start_interval=25 | 1000000 | 1024 | 8.41 | A6000 | 0.147 |  |
| `g-hs-enc-nodr-dinov2s` | hanoi-single | encoder_pretrain | lr=0.0003, enc_lr=5e-05, data=nodr, arch=dinov2_s, exclude_groups=peg_xyz,peg_yaw | 40000 | 128 | 1.39 | A5000/A6000 |  |  |
| `g-hs-state-n10k-int25-s0` | hanoi-single | train_regime | features=handcrafted, base=10000, start_interval=25 | 500000 | 1024 | 4.25 | A5000/A6000 | 0.539 |  |
| `g-hs-state-n30k-int25-s0` | hanoi-single | train_regime | features=handcrafted, base=30000, start_interval=25 | 500000 | 1024 | 4.02 | A6000 | 0.55 |  |
| `g-hs-vis-dinov2_s_p4-s0` | hanoi-single | train_vis | input=emb, encoder=dinov2_s_p4, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 4.47 | A6000 | 0.963 | 1000 |
| `g-hs-vis-pre-s0` | hanoi-single | train_vis | input=emb, encoder=pre:g-hs-enc-nodr-dinov2s, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.62 | A5000/A6000 | 0.996 | 1000 |
| `gen-G-drgen-bw` | single | support job |  |  |  | 0.06 | A5000/A6000 |  |  |
| `gen-G-drgen-hs` | single | support job |  |  |  | 0.07 | A5000/A6000 |  |  |
| `gen-G-embedpre-bw` | single | support job |  |  |  | 0.04 | A5000 |  |  |
| `gen-G-embedpre-hs` | single | support job |  |  |  | 0.04 | A6000 |  |  |
| `gen-G-prefill-bw` | single | support job |  |  |  | 0.32 | A6000/rtx6000 |  |  |
| `gen-G-prefill-hd` | single | support job |  |  |  | 0.89 | A6000/rtx6000 |  |  |
| `gen-G-prefill-hs` | single | support job |  |  |  | 0.27 | A6000/rtx6000 |  |  |
| `gen-G-viscol-bw` | single | support job |  |  |  | 0.67 | A5000/A6000 |  |  |
| `gen-G-viscol-hs` | single | support job |  |  |  | 0.9 | A5000/A6000 |  |  |
| `gen-G-visprep-bw` | single | support job |  |  |  | 0.12 | A5000 |  |  |
| `gen-G-visprep-hs` | single | support job |  |  |  | 0.17 | A5000 |  |  |
| `gen-L-drgen-e1000` | single | support job |  |  |  | 0.06 | A5000/A6000 |  |  |
| `gen-L-drgen-e250` | single | support job |  |  |  | 0.01 | A5000/A6000 |  |  |
| `gen-L-embed-l-sgl-enc-p190k-e1000` | single | support job |  |  |  | 0.05 | A5000 |  |  |
| `gen-L-embed-l-sgl-enc-p20k-e1000` | single | support job |  |  |  | 0.05 | A5000/A6000 |  |  |
| `gen-L-embed-l-sgl-enc-p20k-e250` | single | support job |  |  |  | 0.04 | A5000 |  |  |
| `gen-Q-embed` | single | support job |  |  |  | 0.09 | A5000/A6000 |  |  |
| `gen-Q-yawprobe` | single | support job |  |  |  | 0.18 | A6000 |  |  |
| `gen-S-embed` | single | support job |  |  |  | 0.14 | A5000 |  |  |
| `gen-SL-drgen` | single | support job |  |  |  | 0.05 | A6000 |  |  |
| `gen-SL-embed-p20` | single | support job |  |  |  | 0.05 | A6000 |  |  |
| `gen-SL-embed-p200` | single | support job |  |  |  | 0.06 | A6000 |  |  |
| `gen-SL-embed-p60` | single | support job |  |  |  | 0.06 | rtx6000 |  |  |
| `gen-SL-embed-p600` | single | support job |  |  |  | 0.05 | A6000 |  |  |
| `gen-X1-drgen` | single | support job |  |  |  | 0.01 | A5000 |  |  |
| `gen-X1-embed-p200` | single | support job |  |  |  | 0.05 | A5000 |  |  |
| `gen-X1-embed-p20k` | single | support job |  |  |  | 0.05 | A5000 |  |  |
| `gen-X1-embed-p2k` | single | support job |  |  |  | 0.05 | A5000 |  |  |
| `gen-X2-drgen` | single | support job |  |  |  | 0.01 | A5000 |  |  |
| `gen-X2-embed-p200` | single | support job |  |  |  | 0.04 | A5000 |  |  |
| `gen-X2-embed-p20k` | single | support job |  |  |  | 0.05 | A5000 |  |  |
| `gen-X2-embed-p2k` | single | support job |  |  |  | 0.05 | A5000 |  |  |
| `gen-X6-embed-dbl` | double | support job |  |  |  | 0.15 | rtx6000 |  |  |
| `gen-X6-embed-sgl` | single | support job |  |  |  | 0.08 | rtx6000 |  |  |
| `gen-X7-drgen` | single | support job |  |  |  | 0.01 | A5000 |  |  |
| `gen-X7-embed-p20` | single | support job |  |  |  | 0.05 | A5000 |  |  |
| `gen-X7-embed-p200` | single | support job |  |  |  | 0.08 | A5000 |  |  |
| `gen-X7-embed-p60` | single | support job |  |  |  | 0.05 | A5000 |  |  |
| `gen-X7-embed-p600` | single | support job |  |  |  | 0.05 | A5000 |  |  |
| `q-sgl-bc-noyaw-s0` | single | train_vis | input=emb, encoder=pre:q-sgl-enc-noyaw, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.46 | A5000/A6000 | 0.93 | 1000 |
| `q-sgl-enc-noyaw` | single | encoder_pretrain | lr=0.0003, enc_lr=5e-05, data=nodr, arch=dinov2_s, exclude_groups=cube_yaw | 40000 | 128 | 1.22 | A6000 |  |  |
| `s-sgl-bc-p0k-e1000-s0` | single | train_vis | episodes=1000, input=emb, encoder=dinov2_s, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 4.42 | A5000/A6000/rtx6000 | 0.187 | 1000 |
| `s-sgl-bc-p0k-e1000-s1` | single | train_vis | episodes=1000, input=emb, encoder=dinov2_s, scale=std, lr=0.0001, enc_lr=1e-05, img=112, seed=1 | 300000 | 1024 | 3.56 | A5000/A6000/rtx6000 | 0.197 | 1000 |
| `s-sgl-bc-p0k-e2000-s0` | single | train_vis | episodes=2000, input=emb, encoder=dinov2_s, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 1.25 | A5000/A6000/rtx6000 | 0.377 | 1000 |
| `s-sgl-bc-p0k-e250-s0` | single | train_vis | episodes=250, input=emb, encoder=dinov2_s, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.68 | A5000/A6000/rtx6000 | 0.029 | 1000 |
| `s-sgl-bc-p0k-e4000-s0` | single | train_vis | episodes=4000, input=emb, encoder=dinov2_s, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 1.36 | A5000/A6000/rtx6000 | 0.53 | 1000 |
| `s-sgl-bc-p0k-e500-s0` | single | train_vis | episodes=500, input=emb, encoder=dinov2_s, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.52 | A5000/A6000 | 0.075 | 1000 |
| `s-sgl-bc-p0k-e8818-s0` | single | train_vis | input=emb, encoder=dinov2_s, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 7.84 | A6000/rtx6000 | 0.68 | 1000 |
| `s-sgl-bc-p0k-e8818-s1` | single | train_vis | input=emb, encoder=dinov2_s, scale=std, lr=0.0001, enc_lr=1e-05, img=112, seed=1 | 300000 | 1024 | 4.46 | A5000/A6000/rtx6000 | 0.659 | 1000 |
| `s-sgl-bc-p190k-e1000-s0` | single | train_vis | episodes=1000, input=emb, encoder=pre:s-sgl-enc-p190k, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.67 | A5000/A6000 | 0.908 | 1000 |
| `s-sgl-bc-p190k-e1000-s1` | single | train_vis | episodes=1000, input=emb, encoder=pre:s-sgl-enc-p190k, scale=std, lr=0.0001, enc_lr=1e-05, img=112, seed=1 | 300000 | 1024 | 3.47 | A5000/A6000 | 0.921 | 1000 |
| `s-sgl-bc-p190k-e2000-s0` | single | train_vis | episodes=2000, input=emb, encoder=pre:s-sgl-enc-p190k, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 4.84 | A6000/rtx6000 | 0.941 | 1000 |
| `s-sgl-bc-p190k-e250-s0` | single | train_vis | episodes=250, input=emb, encoder=pre:s-sgl-enc-p190k, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 1.03 | 3080ti/A6000/A6000Ada/rtx6000 | 0.68 | 1000 |
| `s-sgl-bc-p190k-e4000-s0` | single | train_vis | episodes=4000, input=emb, encoder=pre:s-sgl-enc-p190k, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 2.97 | 3080ti/A5000/A6000/rtx6000 | 0.962 | 1000 |
| `s-sgl-bc-p190k-e500-s0` | single | train_vis | episodes=500, input=emb, encoder=pre:s-sgl-enc-p190k, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 0.89 | A6000/rtx6000 | 0.822 | 1000 |
| `s-sgl-bc-p190k-e8818-s0` | single | train_vis | input=emb, encoder=pre:s-sgl-enc-p190k, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 0.72 | A5000/A6000/rtx6000 | 0.958 | 1000 |
| `s-sgl-bc-p190k-e8818-s1` | single | train_vis | input=emb, encoder=pre:s-sgl-enc-p190k, scale=std, lr=0.0001, enc_lr=1e-05, img=112, seed=1 | 300000 | 1024 | 3.67 | A5000/A6000 | 0.964 | 1000 |
| `s-sgl-bc-p20-e1000-s0` | single | train_vis | episodes=1000, input=emb, encoder=pre:s-sgl-enc-p20, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 5.77 | A5000/A6000/rtx6000 | 0.33 | 1000 |
| `s-sgl-bc-p20-e2000-s0` | single | train_vis | episodes=2000, input=emb, encoder=pre:s-sgl-enc-p20, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.53 | A5000/A6000/rtx2080ti | 0.453 | 1000 |
| `s-sgl-bc-p20-e250-s0` | single | train_vis | episodes=250, input=emb, encoder=pre:s-sgl-enc-p20, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 5.57 | A6000/rtx6000 | 0.057 | 1000 |
| `s-sgl-bc-p20-e4000-s0` | single | train_vis | episodes=4000, input=emb, encoder=pre:s-sgl-enc-p20, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.57 | A5000/A6000/rtx2080ti | 0.637 | 1000 |
| `s-sgl-bc-p20-e500-s0` | single | train_vis | episodes=500, input=emb, encoder=pre:s-sgl-enc-p20, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.29 | A6000 | 0.117 | 1000 |
| `s-sgl-bc-p20-e8818-s0` | single | train_vis | input=emb, encoder=pre:s-sgl-enc-p20, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.52 | A5000/A6000/rtx2080ti | 0.746 | 1000 |
| `s-sgl-bc-p20-r1-e1000-s1` | single | train_vis | episodes=1000, input=emb, encoder=pre:s-sgl-enc-p20-r1, scale=std, lr=0.0001, enc_lr=1e-05, img=112, seed=1 | 300000 | 1024 | 3.44 | A5000 | 0.278 | 1000 |
| `s-sgl-bc-p20-r1-e8818-s1` | single | train_vis | input=emb, encoder=pre:s-sgl-enc-p20-r1, scale=std, lr=0.0001, enc_lr=1e-05, img=112, seed=1 | 300000 | 1024 | 3.55 | A5000/rtx3090 | 0.738 | 1000 |
| `s-sgl-bc-p200-e1000-s0` | single | train_vis | episodes=1000, input=emb, encoder=pre:s-sgl-enc-p200, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.31 | A6000/rtx2080ti | 0.568 | 1000 |
| `s-sgl-bc-p200-e2000-s0` | single | train_vis | episodes=2000, input=emb, encoder=pre:s-sgl-enc-p200, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.68 | A5000/rtx2080ti | 0.738 | 1000 |
| `s-sgl-bc-p200-e250-s0` | single | train_vis | episodes=250, input=emb, encoder=pre:s-sgl-enc-p200, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.34 | A6000 | 0.179 | 1000 |
| `s-sgl-bc-p200-e4000-s0` | single | train_vis | episodes=4000, input=emb, encoder=pre:s-sgl-enc-p200, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.36 | A6000/rtx2080ti | 0.85 | 1000 |
| `s-sgl-bc-p200-e500-s0` | single | train_vis | episodes=500, input=emb, encoder=pre:s-sgl-enc-p200, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.46 | A5000/A6000/rtx2080ti | 0.349 | 1000 |
| `s-sgl-bc-p200-e8818-s0` | single | train_vis | input=emb, encoder=pre:s-sgl-enc-p200, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.48 | A5000/A6000/rtx2080ti | 0.891 | 1000 |
| `s-sgl-bc-p200-r1-e1000-s1` | single | train_vis | episodes=1000, input=emb, encoder=pre:s-sgl-enc-p200-r1, scale=std, lr=0.0001, enc_lr=1e-05, img=112, seed=1 | 300000 | 1024 | 3.72 | A5000 | 0.559 | 1000 |
| `s-sgl-bc-p200-r1-e8818-s1` | single | train_vis | input=emb, encoder=pre:s-sgl-enc-p200-r1, scale=std, lr=0.0001, enc_lr=1e-05, img=112, seed=1 | 300000 | 1024 | 3.64 | A5000 | 0.931 | 1000 |
| `s-sgl-bc-p20k-e1000-s0` | single | train_vis | episodes=1000, input=emb, encoder=pre:s-sgl-enc-p20k, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.91 | 3080ti/A6000 | 0.893 | 1000 |
| `s-sgl-bc-p20k-e1000-s1` | single | train_vis | episodes=1000, input=emb, encoder=pre:s-sgl-enc-p20k, scale=std, lr=0.0001, enc_lr=1e-05, img=112, seed=1 | 300000 | 1024 | 0.52 | A5000/A6000/rtx2080ti | 0.89 | 1000 |
| `s-sgl-bc-p20k-e2000-s0` | single | train_vis | episodes=2000, input=emb, encoder=pre:s-sgl-enc-p20k, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.99 | 3080ti/A6000 | 0.938 | 1000 |
| `s-sgl-bc-p20k-e250-s0` | single | train_vis | episodes=250, input=emb, encoder=pre:s-sgl-enc-p20k, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 8.71 | A6000/rtx2080ti | 0.576 | 1000 |
| `s-sgl-bc-p20k-e4000-s0` | single | train_vis | episodes=4000, input=emb, encoder=pre:s-sgl-enc-p20k, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.96 | 3080ti/A6000 | 0.96 | 1000 |
| `s-sgl-bc-p20k-e500-s0` | single | train_vis | episodes=500, input=emb, encoder=pre:s-sgl-enc-p20k, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 5.96 | A5000/A6000/A6000Ada/rtx2080ti | 0.78 | 1000 |
| `s-sgl-bc-p20k-e8818-s0` | single | train_vis | input=emb, encoder=pre:s-sgl-enc-p20k, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.59 | A5000/A6000 | 0.965 | 1000 |
| `s-sgl-bc-p20k-e8818-s1` | single | train_vis | input=emb, encoder=pre:s-sgl-enc-p20k, scale=std, lr=0.0001, enc_lr=1e-05, img=112, seed=1 | 300000 | 1024 | 3.91 | 3080ti/A6000 | 0.968 | 1000 |
| `s-sgl-bc-p2k-e1000-s0` | single | train_vis | episodes=1000, input=emb, encoder=pre:s-sgl-enc-p2k, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.26 | A6000 | 0.826 | 1000 |
| `s-sgl-bc-p2k-e1000-s1` | single | train_vis | episodes=1000, input=emb, encoder=pre:s-sgl-enc-p2k, scale=std, lr=0.0001, enc_lr=1e-05, img=112, seed=1 | 300000 | 1024 | 3.26 | A6000 | 0.815 | 1000 |
| `s-sgl-bc-p2k-e2000-s0` | single | train_vis | episodes=2000, input=emb, encoder=pre:s-sgl-enc-p2k, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.22 | A6000 | 0.908 | 1000 |
| `s-sgl-bc-p2k-e250-s0` | single | train_vis | episodes=250, input=emb, encoder=pre:s-sgl-enc-p2k, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.57 | A5000/A6000 | 0.478 | 1000 |
| `s-sgl-bc-p2k-e4000-s0` | single | train_vis | episodes=4000, input=emb, encoder=pre:s-sgl-enc-p2k, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.26 | A6000 | 0.928 | 1000 |
| `s-sgl-bc-p2k-e500-s0` | single | train_vis | episodes=500, input=emb, encoder=pre:s-sgl-enc-p2k, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 4.33 | A5000/A6000/rtx6000 | 0.697 | 1000 |
| `s-sgl-bc-p2k-e8818-s0` | single | train_vis | input=emb, encoder=pre:s-sgl-enc-p2k, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 8.62 | A6000/rtx2080ti/rtx6000 | 0.944 | 1000 |
| `s-sgl-bc-p2k-e8818-s1` | single | train_vis | input=emb, encoder=pre:s-sgl-enc-p2k, scale=std, lr=0.0001, enc_lr=1e-05, img=112, seed=1 | 300000 | 1024 | 7.76 | A6000/rtx6000 | 0.95 | 1000 |
| `s-sgl-bc-p60-e1000-s0` | single | train_vis | episodes=1000, input=emb, encoder=pre:s-sgl-enc-p60, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.65 | A5000/A6000/rtx2080ti | 0.341 | 1000 |
| `s-sgl-bc-p60-e2000-s0` | single | train_vis | episodes=2000, input=emb, encoder=pre:s-sgl-enc-p60, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.55 | A5000/A6000/rtx2080ti | 0.586 | 1000 |
| `s-sgl-bc-p60-e250-s0` | single | train_vis | episodes=250, input=emb, encoder=pre:s-sgl-enc-p60, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.7 | A5000/A6000/rtx2080ti | 0.101 | 1000 |
| `s-sgl-bc-p60-e4000-s0` | single | train_vis | episodes=4000, input=emb, encoder=pre:s-sgl-enc-p60, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.37 | A6000/rtx2080ti | 0.692 | 1000 |
| `s-sgl-bc-p60-e500-s0` | single | train_vis | episodes=500, input=emb, encoder=pre:s-sgl-enc-p60, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.35 | A6000/rtx2080ti | 0.18 | 1000 |
| `s-sgl-bc-p60-e8818-s0` | single | train_vis | input=emb, encoder=pre:s-sgl-enc-p60, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.63 | A5000/A6000/rtx2080ti | 0.835 | 1000 |
| `s-sgl-bc-p60-r1-e1000-s1` | single | train_vis | episodes=1000, input=emb, encoder=pre:s-sgl-enc-p60-r1, scale=std, lr=0.0001, enc_lr=1e-05, img=112, seed=1 | 300000 | 1024 | 3.49 | A5000 | 0.306 | 1000 |
| `s-sgl-bc-p60-r1-e8818-s1` | single | train_vis | input=emb, encoder=pre:s-sgl-enc-p60-r1, scale=std, lr=0.0001, enc_lr=1e-05, img=112, seed=1 | 300000 | 1024 | 1.41 | 6000Blackwell | 0.777 | 1000 |
| `s-sgl-bc-p600-e1000-s0` | single | train_vis | episodes=1000, input=emb, encoder=pre:s-sgl-enc-p600, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.36 | A6000/rtx2080ti | 0.765 | 1000 |
| `s-sgl-bc-p600-e2000-s0` | single | train_vis | episodes=2000, input=emb, encoder=pre:s-sgl-enc-p600, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.75 | A5000/A6000/rtx2080ti/rtx6000 | 0.896 | 1000 |
| `s-sgl-bc-p600-e250-s0` | single | train_vis | episodes=250, input=emb, encoder=pre:s-sgl-enc-p600, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.35 | A6000 | 0.382 | 1000 |
| `s-sgl-bc-p600-e4000-s0` | single | train_vis | episodes=4000, input=emb, encoder=pre:s-sgl-enc-p600, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.5 | A5000/A6000/rtx2080ti | 0.932 | 1000 |
| `s-sgl-bc-p600-e500-s0` | single | train_vis | episodes=500, input=emb, encoder=pre:s-sgl-enc-p600, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.46 | A5000/A6000/rtx2080ti | 0.563 | 1000 |
| `s-sgl-bc-p600-e8818-s0` | single | train_vis | input=emb, encoder=pre:s-sgl-enc-p600, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.55 | A5000/A6000/rtx2080ti | 0.948 | 1000 |
| `s-sgl-bc-p600-r1-e1000-s1` | single | train_vis | episodes=1000, input=emb, encoder=pre:s-sgl-enc-p600-r1, scale=std, lr=0.0001, enc_lr=1e-05, img=112, seed=1 | 300000 | 1024 | 3.54 | A5000 | 0.751 | 1000 |
| `s-sgl-bc-p600-r1-e8818-s1` | single | train_vis | input=emb, encoder=pre:s-sgl-enc-p600-r1, scale=std, lr=0.0001, enc_lr=1e-05, img=112, seed=1 | 300000 | 1024 | 3.35 | A6000 | 0.945 | 1000 |
| `s-sgl-bc-p60k-e1000-s0` | single | train_vis | episodes=1000, input=emb, encoder=pre:s-sgl-enc-p60k, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 0.52 | A5000/A6000/rtx2080ti | 0.935 | 1000 |
| `s-sgl-bc-p60k-e1000-s1` | single | train_vis | episodes=1000, input=emb, encoder=pre:s-sgl-enc-p60k, scale=std, lr=0.0001, enc_lr=1e-05, img=112, seed=1 | 300000 | 1024 | 3.94 | 3080ti/A6000 | 0.916 | 1000 |
| `s-sgl-bc-p60k-e2000-s0` | single | train_vis | episodes=2000, input=emb, encoder=pre:s-sgl-enc-p60k, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.99 | 3080ti/A5000/A6000 | 0.96 | 1000 |
| `s-sgl-bc-p60k-e250-s0` | single | train_vis | episodes=250, input=emb, encoder=pre:s-sgl-enc-p60k, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 4.54 | A5000/A6000/A6000Ada/rtx2080ti | 0.662 | 1000 |
| `s-sgl-bc-p60k-e4000-s0` | single | train_vis | episodes=4000, input=emb, encoder=pre:s-sgl-enc-p60k, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 5.13 | A5000/A6000/rtx2080ti | 0.966 | 1000 |
| `s-sgl-bc-p60k-e500-s0` | single | train_vis | episodes=500, input=emb, encoder=pre:s-sgl-enc-p60k, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 0.64 | A5000/A6000/rtx2080ti | 0.821 | 1000 |
| `s-sgl-bc-p60k-e8818-s0` | single | train_vis | input=emb, encoder=pre:s-sgl-enc-p60k, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.91 | 3080ti/A5000/A6000/rtx6000 | 0.965 | 1000 |
| `s-sgl-bc-p60k-e8818-s1` | single | train_vis | input=emb, encoder=pre:s-sgl-enc-p60k, scale=std, lr=0.0001, enc_lr=1e-05, img=112, seed=1 | 300000 | 1024 | 3.93 | 3080ti/A6000/A6000Ada | 0.952 | 1000 |
| `s-sgl-bc-p6k-e1000-s0` | single | train_vis | episodes=1000, input=emb, encoder=pre:s-sgl-enc-p6k, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 7.96 | A6000/rtx6000 | 0.889 | 1000 |
| `s-sgl-bc-p6k-e1000-s1` | single | train_vis | episodes=1000, input=emb, encoder=pre:s-sgl-enc-p6k, scale=std, lr=0.0001, enc_lr=1e-05, img=112, seed=1 | 300000 | 1024 | 3.46 | A5000/A6000 | 0.862 | 1000 |
| `s-sgl-bc-p6k-e2000-s0` | single | train_vis | episodes=2000, input=emb, encoder=pre:s-sgl-enc-p6k, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.28 | A6000 | 0.939 | 1000 |
| `s-sgl-bc-p6k-e250-s0` | single | train_vis | episodes=250, input=emb, encoder=pre:s-sgl-enc-p6k, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 8.19 | A6000/rtx6000 | 0.542 | 1000 |
| `s-sgl-bc-p6k-e4000-s0` | single | train_vis | episodes=4000, input=emb, encoder=pre:s-sgl-enc-p6k, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 8.86 | A6000/rtx2080ti | 0.953 | 1000 |
| `s-sgl-bc-p6k-e500-s0` | single | train_vis | episodes=500, input=emb, encoder=pre:s-sgl-enc-p6k, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 8.19 | A6000/rtx6000 | 0.776 | 1000 |
| `s-sgl-bc-p6k-e8818-s0` | single | train_vis | input=emb, encoder=pre:s-sgl-enc-p6k, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 1.94 | 3080ti/A5000/A6000/rtx6000 | 0.951 | 1000 |
| `s-sgl-bc-p6k-e8818-s1` | single | train_vis | input=emb, encoder=pre:s-sgl-enc-p6k, scale=std, lr=0.0001, enc_lr=1e-05, img=112, seed=1 | 300000 | 1024 | 3.69 | 3080ti/A5000/A6000 | 0.972 | 1000 |
| `s-sgl-enc-p190k` | single | encoder_pretrain | lr=0.0003, enc_lr=5e-05, data=nodr, arch=dinov2_s, n_pairs=190000 | 40000 | 128 | 1.2 | A6000 |  |  |
| `s-sgl-enc-p20` | single | encoder_pretrain | lr=0.0003, enc_lr=5e-05, data=nodr, arch=dinov2_s, n_pairs=20 | 40000 | 128 | 0.3 | A6000 |  |  |
| `s-sgl-enc-p20-r1` | single | encoder_pretrain | lr=0.0003, enc_lr=5e-05, data=nodr, arch=dinov2_s, n_pairs=20, dr_dir=visual-block-single-task1-v0-seed1, seed=1 | 40000 | 128 | 0.3 | A5000 |  |  |
| `s-sgl-enc-p200` | single | encoder_pretrain | lr=0.0003, enc_lr=5e-05, data=nodr, arch=dinov2_s, n_pairs=200 | 40000 | 128 | 1.22 | A6000 |  |  |
| `s-sgl-enc-p200-r1` | single | encoder_pretrain | lr=0.0003, enc_lr=5e-05, data=nodr, arch=dinov2_s, n_pairs=200, dr_dir=visual-block-single-task1-v0-seed1, seed=1 | 40000 | 128 | 1.4 | A5000 |  |  |
| `s-sgl-enc-p20k` | single | encoder_pretrain | lr=0.0003, enc_lr=5e-05, data=nodr, arch=dinov2_s, n_pairs=20000 | 40000 | 128 | 1.39 | A5000/A6000 |  |  |
| `s-sgl-enc-p2k` | single | encoder_pretrain | lr=0.0003, enc_lr=5e-05, data=nodr, arch=dinov2_s, n_pairs=2000 | 40000 | 128 | 1.4 | A5000 |  |  |
| `s-sgl-enc-p60` | single | encoder_pretrain | lr=0.0003, enc_lr=5e-05, data=nodr, arch=dinov2_s, n_pairs=60 | 40000 | 128 | 0.65 | A6000 |  |  |
| `s-sgl-enc-p60-r1` | single | encoder_pretrain | lr=0.0003, enc_lr=5e-05, data=nodr, arch=dinov2_s, n_pairs=60, dr_dir=visual-block-single-task1-v0-seed1, seed=1 | 40000 | 128 | 0.2 | 6000Blackwell |  |  |
| `s-sgl-enc-p600` | single | encoder_pretrain | lr=0.0003, enc_lr=5e-05, data=nodr, arch=dinov2_s, n_pairs=600 | 40000 | 128 | 1.22 | A6000 |  |  |
| `s-sgl-enc-p600-r1` | single | encoder_pretrain | lr=0.0003, enc_lr=5e-05, data=nodr, arch=dinov2_s, n_pairs=600, dr_dir=visual-block-single-task1-v0-seed1, seed=1 | 40000 | 128 | 1.35 | A5000 |  |  |
| `s-sgl-enc-p60k` | single | encoder_pretrain | lr=0.0003, enc_lr=5e-05, data=nodr, arch=dinov2_s, n_pairs=60000 | 40000 | 128 | 1.19 | A6000 |  |  |
| `s-sgl-enc-p6k` | single | encoder_pretrain | lr=0.0003, enc_lr=5e-05, data=nodr, arch=dinov2_s, n_pairs=6000 | 40000 | 128 | 1.39 | A5000/A6000 |  |  |
| `x1-sgl-bc-uni-p200-e1000-s0` | single | train_vis | episodes=1000, input=emb, encoder=pre:x1-sgl-enc-uni-p200, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.49 | A5000 | 0.586 | 1000 |
| `x1-sgl-bc-uni-p200-e8818-s0` | single | train_vis | input=emb, encoder=pre:x1-sgl-enc-uni-p200, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 1.42 | 6000Blackwell/A6000 | 0.922 | 1000 |
| `x1-sgl-bc-uni-p20k-e1000-s0` | single | train_vis | episodes=1000, input=emb, encoder=pre:x1-sgl-enc-uni-p20k, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.5 | A5000 | 0.861 | 1000 |
| `x1-sgl-bc-uni-p20k-e8818-s0` | single | train_vis | input=emb, encoder=pre:x1-sgl-enc-uni-p20k, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 4.29 | 3080ti/A6000 | 0.947 | 1000 |
| `x1-sgl-bc-uni-p2k-e1000-s0` | single | train_vis | episodes=1000, input=emb, encoder=pre:x1-sgl-enc-uni-p2k, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.52 | A5000 | 0.797 | 1000 |
| `x1-sgl-bc-uni-p2k-e8818-s0` | single | train_vis | input=emb, encoder=pre:x1-sgl-enc-uni-p2k, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.47 | A5000 | 0.942 | 1000 |
| `x1-sgl-enc-uni-p200` | single | encoder_pretrain | lr=0.0003, enc_lr=5e-05, data=nodr, arch=dinov2_s, n_pairs=200, dr_dir=visual-block-single-task1-v0-uniform | 40000 | 128 | 1.37 | A5000 |  |  |
| `x1-sgl-enc-uni-p20k` | single | encoder_pretrain | lr=0.0003, enc_lr=5e-05, data=nodr, arch=dinov2_s, n_pairs=20000, dr_dir=visual-block-single-task1-v0-uniform | 40000 | 128 | 1.35 | A5000 |  |  |
| `x1-sgl-enc-uni-p2k` | single | encoder_pretrain | lr=0.0003, enc_lr=5e-05, data=nodr, arch=dinov2_s, n_pairs=2000, dr_dir=visual-block-single-task1-v0-uniform | 40000 | 128 | 1.39 | A5000 |  |  |
| `x2-hs-bc-dinov2_s_p4-e1000-s0` | hanoi-single | train_vis | episodes=1000, input=emb, encoder=dinov2_s_p4, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 4.49 | A6000 | 0.393 | 1000 |
| `x2-hs-bc-p190k-e1000-s0` | hanoi-single | train_vis | episodes=1000, input=emb, encoder=pre:g-hs-enc-nodr-dinov2s, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.37 | A6000 | 0.989 | 1000 |
| `x2-hs-bc-p200-e1000-s0` | hanoi-single | train_vis | episodes=1000, input=emb, encoder=pre:x2-hs-enc-p200, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.49 | A5000 | 0.94 | 1000 |
| `x2-hs-bc-p200-e9962-s0` | hanoi-single | train_vis | input=emb, encoder=pre:x2-hs-enc-p200, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.44 | A5000 | 0.998 | 1000 |
| `x2-hs-bc-p20k-e1000-s0` | hanoi-single | train_vis | episodes=1000, input=emb, encoder=pre:x2-hs-enc-p20k, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.45 | A5000 | 0.984 | 1000 |
| `x2-hs-bc-p20k-e9962-s0` | hanoi-single | train_vis | input=emb, encoder=pre:x2-hs-enc-p20k, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.31 | A6000 | 0.994 | 1000 |
| `x2-hs-bc-p2k-e1000-s0` | hanoi-single | train_vis | episodes=1000, input=emb, encoder=pre:x2-hs-enc-p2k, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.45 | A5000 | 0.98 | 1000 |
| `x2-hs-bc-p2k-e9962-s0` | hanoi-single | train_vis | input=emb, encoder=pre:x2-hs-enc-p2k, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 3.6 | A5000 | 0.993 | 1000 |
| `x2-hs-enc-p200` | hanoi-single | encoder_pretrain | lr=0.0003, enc_lr=5e-05, data=nodr, arch=dinov2_s, exclude_groups=peg_xyz,peg_yaw, n_pairs=200 | 40000 | 128 | 1.35 | A5000 |  |  |
| `x2-hs-enc-p20k` | hanoi-single | encoder_pretrain | lr=0.0003, enc_lr=5e-05, data=nodr, arch=dinov2_s, exclude_groups=peg_xyz,peg_yaw, n_pairs=20000 | 40000 | 128 | 1.34 | A5000 |  |  |
| `x2-hs-enc-p2k` | hanoi-single | encoder_pretrain | lr=0.0003, enc_lr=5e-05, data=nodr, arch=dinov2_s, exclude_groups=peg_xyz,peg_yaw, n_pairs=2000 | 40000 | 128 | 1.35 | A5000 |  |  |
| `x3-sgl-bc-dr-s0` | single | train_vis | input=emb, encoder=pre:v-F-enc-dr-dinov2s, scale=std, lr=0.0001, enc_lr=1e-05, img=112, shift_eval=nominal,color,light,camera,all | 300000 | 1024 | 3.81 | A5000 | 0.964 | 1000 |
| `x3-sgl-bc-nodr-s0` | single | train_vis | input=emb, encoder=pre:v-F-enc-nodr-dinov2s, scale=std, lr=0.0001, enc_lr=1e-05, img=112, shift_eval=nominal,color,light,camera,all | 300000 | 1024 | 3.56 | A5000 | 0.968 | 1000 |
| `x3-sgl-bc-p4-s0` | single | train_vis | input=emb, encoder=dinov2_s_p4, scale=std, lr=0.0001, enc_lr=1e-05, img=112, shift_eval=nominal,color,light,camera,all | 300000 | 1024 | 5.19 | A6000 | 0.924 | 1000 |
| `x4-hs-state-diskyaw-n10k-int25-s0` | hanoi-single | train_regime | features=handcrafted+diskyaw, base=10000, start_interval=25 | 500000 | 1024 | 5.4 | 3080ti/A6000 | 0.475 |  |
| `x4-hs-state-diskyaw-n30k-int25-s0` | hanoi-single | train_regime | features=handcrafted+diskyaw, base=30000, start_interval=25 | 500000 | 1024 | 4.52 | A5000/A6000 | 0.549 |  |
| `x5-sgl-bc-ft-p20-s0` | single | train_vis | input=pixels, encoder=pre:s-sgl-enc-p20, scale=std, finetune=True, lr=0.0001, enc_lr=1e-05, img=112 | 200000 | 256 | 12.53 | 6000Blackwell/A6000 | 0.918 | 1000 |
| `x5-sgl-bc-ft-p200-s0` | single | train_vis | input=pixels, encoder=pre:s-sgl-enc-p200, scale=std, finetune=True, lr=0.0001, enc_lr=1e-05, img=112 | 200000 | 256 | 12.4 | A6000 | 0.923 | 1000 |
| `x6-dbl-bc-prep4-s0` | double | train_vis | input=emb, encoder=pre:v-dbl-F-enc-nodr-dinov2s@p4, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 4.75 | A6000 | 0.824 | 1000 |
| `x6-sgl-bc-p190kp4-e1000-s0` | single | train_vis | episodes=1000, input=emb, encoder=pre:s-sgl-enc-p190k@p4, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 4.75 | A5000 | 0.853 | 1000 |
| `x6-sgl-bc-p190kp4-e250-s0` | single | train_vis | episodes=250, input=emb, encoder=pre:s-sgl-enc-p190k@p4, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 4.96 | A5000 | 0.331 | 1000 |
| `x6-sgl-bc-p2kp4-e1000-s0` | single | train_vis | episodes=1000, input=emb, encoder=pre:s-sgl-enc-p2k@p4, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 4.82 | A5000 | 0.817 | 1000 |
| `x6-sgl-bc-p2kp4-e250-s0` | single | train_vis | episodes=250, input=emb, encoder=pre:s-sgl-enc-p2k@p4, scale=std, lr=0.0001, enc_lr=1e-05, img=112 | 300000 | 1024 | 4.84 | A5000 | 0.348 | 1000 |

The following runs are reported in the text but launched by older tooling and are not in the table:
- `v-state-single-n10000-int25-s0` (A-group handcrafted, success 0.85);
- `v-state-double-n10000-int25-s0` (double, 0.14);
- the Mystery-1/4 reference runs (`m1-*`, `m4-*`; see [RESULTS.md](../../RESULTS.md)).

## A12. Reproduction

The visual scripts (`vis_common.py`, `embed_probe.py`, `pretrain_enc.py`, `train_vis.py`, `dr_gen.py`,
`yaw_probe.py`) run in a separate PyTorch environment, `.local/venv-torch` (Python 3.11, torch 2.15 + CUDA 13,
timm 1.0.31). The JAX state trainers use `.venv` from `scripts/setup.sh`. The `ocbench` submodule carries one local
patch, `patches/ocbench-mistake-prob.patch` (default behaviour unchanged):

```bash
git submodule update --init && (cd ocbench && git apply ../patches/ocbench-mistake-prob.patch)
uv venv --python 3.11 .local/venv-torch
uv pip install --python .local/venv-torch/bin/python -r experiments/vision-bc/requirements-torch.txt -e ./ocbench
```

```bash
source scripts/env.sh
python scripts/launch_m4v.py --list          # part A matrix (34 jobs)
python scripts/launch_regime.py --list       # part B (prefill P, regime R, takeover T, bench B, visual V, transfer X)
.venv/bin/python scripts/report_vision.py    # figures + figures/summary.json
python scripts/launch_gen.py --list          # part D follow-up: lists all jobs; submit with --only Q,G,S (default) or --only X1,...,X7
.venv/bin/python scripts/report_scaling.py   # figures/scaling_single.{png,json}: P x E table and logit fit
.venv/bin/python scripts/yaw_probe.py        # followup Q probes -> results/diagnostics/yaw_probe.jsonl
.venv/bin/python scripts/export_vision_runs.py  # runs.csv and configs/
```

| Script | Purpose |
| --- | --- |
| `vis_common.py`, `vis_prep.py`, `embed_probe.py`, `train_vis.py` | visual data index, embedding cache, probes, visual flow-BC trainer |
| `dr_gen.py`, `pretrain_enc.py` | simulated (image, state) data and encoder pretraining (`dr_gen --states uniform`, `--seed`, `--tag`; `pretrain_enc --n-pairs`, `--select best`, `--exclude-groups`) |
| `vis_collect.py` | in-loop rendering of visual demonstrations for hanoi / bowling (OCBench HF shard format) |
| `yaw_probe.py`, `report_scaling.py` | follow-up Q probes; S scaling table, figure and fit |
| `saliency.py` | gradient saliency |
| `collections_store.py`, `train_regime.py` | block-double expert collections and data-regime trainer |
| `takeover.py`, `bench_vis_cost.py` | expert takeover diagnostic and vision cost benchmark |
| `wandb_follow.py` | mirrors JSONL logs to W&B with resume |
| `launch_m4v.py`, `launch_regime.py`, `launch_gen.py` | requeue-safe Slurm launchers (`launch_gen.py`: follow-up) |

