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
