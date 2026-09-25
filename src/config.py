from dataclasses import dataclass


# Config
@dataclass
class FlowPolicyConfig:
    obs_dim: int = 37        # state vector size
    act_dim: int = 7         # joints per action
    chunk_len: int = 25      # H: actions per chunk
    width: int = 1024        # MLP hidden size   (debug small; Park needs >= 4096)
    depth: int = 4           # residual blocks   (Park uses 8)
    time_dim: int = 64       # size of the time embedding
    flow_steps: int = 10     # Euler steps when sampling

@dataclass
class TrainConfig:
    total_steps: int = 100_000
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.0
    warmup_steps: int = 1_000
    # "constant" after warmup is the default on purpose: with cosine, the curve's
    # shape depends on total_steps, so "train longer" runs aren't comparable --
    # which is exactly the comparison Mystery 1 is about.
    lr_schedule: str = "constant"          # "constant" | "cosine"
    grad_clip: float = 1.0

    log_every: int = 100
    eval_every: int = 5_000
    save_every: int = 10_000

    val_batches: int = 8                   # fixed held-out set = val_batches * batch_size samples
    normalizer_batches: int = 32           # batches used to fit obs/action stats

    use_bf16: bool = True                  # autocast on CUDA; ignored on CPU
    seed: int = 0
    out_dir: str = "runs/debug"
