# Experiment guide

Snapshot: 2026-10-21 23:30 UTC. Target: **2M training updates per experiment, one GPU each**.

[Copyable commands: native OCBench where supported, our wrapper where needed](OCBENCH_COMMANDS.md).

## Terms and settings

- **Episode/demo:** one robot attempt, containing many observations/actions. 10K means 10,000 requested attempts; training keeps successful attempts only. **Single/double** means one/two blocks.
- **Data mode:** M1 generates data once and reuses it (**fixed**). M4 generates 1,000 episodes at a time, uses each retained starting point once, then collects more (**streaming**). This reuse/collection policy is our choice.
- **Step/batch:** one model update uses **1,024 starting points**, each paired with 25 actions. 2M updates process 2.048 billion examples—not episodes; fixed-data examples repeat.
- **Network:** OCBench defaults: residual neural network, eight processing layers, each 4,096 values wide, with skip connections (~136M adjustable parameters); Adam learning rate 0.0001; float32 with default GPU matmul precision; 10 flow integration steps to generate actions.
- **Features = inputs:** original uses OCBench's scales. Our standardized version subtracts training mean/divides by standard deviation; position ×0.1/×10 further rescales gripper/block XYZ. Our Fourier extensions append sine/cosine position features. Transforms are fitted once, then frozen.
- **Seeds:** random choices for training/data. Both normally 0; repeat uses 1/1, single-50K uses 0/10000 for independent data.
- **Our measurement settings:** hold aside 128 requested demos for prediction scores; test actual robot success on 500 attempts every 50K updates. Flow loss/action MSE are prediction errors, not success rates. Log every 1K; save every 100K; retain two checkpoints.
- **Simulation parallel cap:** concurrent simulated attempts, not batch size. Originally 512 except XYZ×10; new jobs use 2500.
- **Continuation:** restore saved model/optimizer, restart data and validation. Counts include saved updates; old/new validation curves cannot be directly joined.

## All ten experiments

| Experiment | Current state | Updates / 2M |
| --- | --- | ---: |
| M1 double, 10K, seed 0 | Completed | 2,000,000 |
| M1 double, 10K, seed 1 | Pending | — |
| M1 single, 10K | Pending | — |
| M1 single, 50K | Pending | — |
| M4 original inputs | Running | 1,138,000 |
| M4 standardized inputs | Running | 1,138,000 |
| M4 position ×0.1 | Running | 1,138,000 |
| M4 position ×10 | Pending | 800,000 (saved checkpoint) |
| M4 Fourier, frequency 1 | Pending | — |
| M4 Fourier, frequencies 1/2/4/8 | Pending | — |

**1 completed, 6 pending, 3 running.** M1 final success: 25.8%. No new continuation rollout results yet. Pending jobs allow A6000/A5000. Clear names, owner-only directories; W&B uploads run in a separate CPU job. This table is a snapshot, not live monitoring.
