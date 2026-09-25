"""
flow_policy.py -- a flow-matching behavior cloning policy that outputs action chunks.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from src.config import FlowPolicyConfig
from src.policy.mlp import VelocityMLP
from src.policy.normalizer import Normalizer
t

class FlowPolicy(nn.Module):
    """
    Public methods:
        loss(obs, actions)          -> scalar      used by Trainer
        action_mse(obs, actions)    -> scalar      used by Trainer (validation)
        act(obs)                    -> numpy chunk used by Evaluator
        sample(obs)                 -> normalized chunk (the Euler loop; used by the two above)

    Shapes (raw, unnormalized):
        obs      (B, obs_dim)
        actions  (B, H, act_dim)
    """

    def __init__(self, cfg: FlowPolicyConfig):
        super().__init__()
        self.cfg = cfg
        self.chunk_size = cfg.chunk_len * cfg.act_dim            # flattened H * act_dim
        self.normalizer = Normalizer(cfg.obs_dim, cfg.act_dim)
        self.net = VelocityMLP(cfg.obs_dim, self.chunk_size, cfg.width, cfg.depth, cfg.time_dim)

    # ----------------------------------------------------------------- utils
    @property
    def device(self):
        return next(self.parameters()).device

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _flatten(self, chunk):      # (B, H, A) -> (B, H*A)
        return chunk.reshape(chunk.shape[0], -1)

    def _unflatten(self, chunk):    # (B, H*A) -> (B, H, A)
        return chunk.reshape(chunk.shape[0], self.cfg.chunk_len, self.cfg.act_dim)

    # ----------------------------------------------------------------- training
    def loss(self, obs, actions, generator: torch.Generator | None = None):
        """
        Flow-matching loss.
        """
        state = self.normalizer.normalize_obs(obs)
        clean = self._flatten(self.normalizer.normalize_actions(actions))       # x1

        noise = torch.randn(clean.shape, device=clean.device, generator=generator)  # x0
        t = torch.rand(clean.shape[0], device=clean.device, generator=generator)
        t_col = t[:, None]                                                       # (B, 1) for broadcasting

        noisy = (1 - t_col) * noise + t_col * clean                              # x_t
        target_velocity = clean - noise                                          # v

        predicted_velocity = self.net(state, noisy, t)
        return F.mse_loss(predicted_velocity, target_velocity)

    @torch.no_grad() # inference
    def sample(self, obs, steps: int | None = None, generator: torch.Generator | None = None):
        """Raw obs (B, obs_dim) -> normalized action chunk (B, H, act_dim)."""
        steps = steps or self.cfg.flow_steps
        state = self.normalizer.normalize_obs(obs)
        batch = state.shape[0]
        dt = 1.0 / steps

        chunk = torch.randn(batch, self.chunk_size, device=state.device, generator=generator)
        for i in range(steps):
            t = torch.full((batch,), i * dt, device=state.device)
            chunk = chunk + self.net(state, chunk, t) * dt # Euler step
        return self._unflatten(chunk)

    @torch.no_grad()
    def act(self, obs, steps: int | None = None) -> np.ndarray:
        """
        What the Evaluator calls. Raw obs in, raw action chunk out.
            obs (obs_dim,)     -> (H, act_dim)
            obs (B, obs_dim)   -> (B, H, act_dim)
        """
        is_single = np.ndim(obs) == 1
        obs = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        if is_single:
            obs = obs[None]

        chunk = self.normalizer.unnormalize_actions(self.sample(obs, steps))
        chunk = chunk.float().cpu().numpy()
        return chunk[0] if is_single else chunk

    # ----------------------------------------------------------------- metrics
    @torch.no_grad()
    def action_mse(self, obs, actions, steps: int | None = None,
                   generator: torch.Generator | None = None) -> torch.Tensor:
        """
        Sample a chunk and compare to the demo chunk, in normalized space (so every
        joint counts equally).
        """
        predicted = self.sample(obs, steps, generator)
        target = self.normalizer.normalize_actions(actions)
        return F.mse_loss(predicted, target)
