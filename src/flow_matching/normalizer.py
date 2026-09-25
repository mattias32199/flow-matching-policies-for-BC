import torch
import torch.nn as nn


class Normalizer(nn.Module):
    """
    Scales inputs for the network and un-scales its outputs.
        obs:  (obs - obs_mean) / obs_std * obs_scale
        action:  (action - act_mean) / act_std
    """

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))
        self.register_buffer("obs_scale", torch.ones(obs_dim))   # Mystery 4 knob
        self.register_buffer("act_mean", torch.zeros(act_dim))
        self.register_buffer("act_std", torch.ones(act_dim))

    @torch.no_grad()
    def fit(self, obs: torch.Tensor, actions: torch.Tensor, eps: float = 1e-6):
        """Compute mean/std from data.  obs: (N, obs_dim)   actions: (N, H, act_dim)"""
        actions = actions.reshape(-1, actions.shape[-1])
        self.obs_mean.copy_(obs.mean(0))
        self.obs_std.copy_(obs.std(0).clamp_min(eps))
        self.act_mean.copy_(actions.mean(0))
        self.act_std.copy_(actions.std(0).clamp_min(eps))

    @torch.no_grad()
    def set_obs_scale(self, scale):
        """Extra per-dimension multiplier on obs, applied after standardizing."""
        self.obs_scale.copy_(torch.as_tensor(scale, dtype=self.obs_scale.dtype))

    def normalize_obs(self, obs):
        return (obs - self.obs_mean) / self.obs_std * self.obs_scale

    def normalize_actions(self, actions):
        return (actions - self.act_mean) / self.act_std

    def unnormalize_actions(self, actions):
        return actions * self.act_std + self.act_mean
