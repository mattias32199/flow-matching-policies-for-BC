import torch
import torch.nn as nn

import math


def time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Turn a scalar time t in [0, 1] into a `dim`-sized vector of sines and cosines
    at different frequencies, so the network can easily tell nearby times apart.
        t: (B,)  ->  (B, dim)
    """
    half = dim // 2
    freqs = torch.exp(-math.log(10_000.0) * torch.arange(half, device=t.device) / half)
    angles = (t * 1000.0)[:, None] * freqs[None, :]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class ResidualBlock(nn.Module):
    """x + Linear(GELU(LayerNorm(x))).  One Linear per block, so width 8192 x depth 8
    gives ~0.5B params, matching the sizes quoted in the blog post."""
    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.linear = nn.Linear(width, width)

    def forward(self, x):
        return x + self.linear(F.gelu(self.norm(x)))


class VelocityMLP(nn.Module):
    """
    Simple MLP
    Inputs are concatenated into one flat vector.
    """

    def __init__(self, obs_dim: int, chunk_size: int, width: int, depth: int, time_dim: int):
        super().__init__()
        self.time_dim = time_dim
        self.input_layer = nn.Linear(obs_dim + chunk_size + time_dim, width)
        self.blocks = nn.Sequential(*[ResidualBlock(width) for _ in range(depth)])
        self.output_norm = nn.LayerNorm(width)
        self.output_layer = nn.Linear(width, chunk_size)

        # Start by predicting zero velocity everywhere -> stable early training.
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, state, noisy_chunk, t):
        x = torch.cat([state, noisy_chunk, time_embedding(t, self.time_dim)], dim=-1)
        x = self.input_layer(x)
        x = self.blocks(x)
        return self.output_layer(self.output_norm(x))
