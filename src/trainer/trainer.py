# ./src/trainer/trainer.py
"""
Trainer for flow matching policy
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict
from pathlib import Path
import json

import torch

from src.policy.flow_policy import FlowPolicy
from src.config import TrainConfig
from src.trainer.utils import JsonlLogger


class Trainer:
    def __init__(self, policy: FlowPolicy, train_source, val_source, cfg: TrainConfig,
                 evaluator=None, logger=None, device: str | None = None):
        self.cfg = cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.manual_seed(cfg.seed)

        self.policy = policy.to(self.device)
        self.train_source = train_source
        self.evaluator = evaluator
        self.out_dir = Path(cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or JsonlLogger(self.out_dir / "metrics.jsonl")

        self.opt = torch.optim.AdamW(self.policy.parameters(), lr=cfg.lr,
                                     weight_decay=cfg.weight_decay)
        self.amp = cfg.use_bf16 and self.device.type == "cuda"
        self.step = 0

        # Freeze a validation set once, so every eval compares on identical data.
        self.val_set = [self._batch(val_source) for _ in range(cfg.val_batches)]

        # Fit normalizer on training data (obs_scale is left as-is, so a
        # Mystery 4 scale set before construction survives this).
        obs, act = zip(*[self._batch(train_source) for _ in range(cfg.normalizer_batches)])
        self.policy.normalizer.fit(torch.cat(obs), torch.cat(act))

        with open(self.out_dir / "config.json", "w") as f:
            json.dump({"train": asdict(cfg), "policy": asdict(policy.cfg)}, f, indent=2)
        print(f"FlowPolicy: {policy.num_params() / 1e6:.1f}M params on {self.device}")

    def _batch(self, source):
        obs, act = source.sample_batch(self.cfg.batch_size)
        return self._to_tensor(obs, self.device), self._to_tensor(act, self.device)

    def _lr_at(self, step: int) -> float:
        c = self.cfg
        if step < c.warmup_steps:
            return c.lr * (step + 1) / c.warmup_steps
        if c.lr_schedule == "cosine":
            p = (step - c.warmup_steps) / max(1, c.total_steps - c.warmup_steps)
            return c.lr * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
        return c.lr

    def _autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp)

    @staticmethod
    def _to_tensor(x, device):
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    def train_step(self) -> dict:
        """Single gradient step"""
        self.policy.train()
        obs, act = self._batch(self.train_source)
        for g in self.opt.param_groups:
            g["lr"] = self._lr_at(self.step)

        with self._autocast():
            loss = self.policy.loss(obs, act)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_clip)
        self.opt.step()
        return {"train/flow_loss": loss.item(), "train/grad_norm": gnorm.item(),
                "train/lr": self.opt.param_groups[0]["lr"]}

    @torch.no_grad()
    def validate(self) -> dict:
        """Train-time metrics"""
        self.policy.eval()
        g = torch.Generator(device=self.device).manual_seed(12345)  # same noise every eval
        fl, mse = [], []
        for obs, act in self.val_set:
            with self._autocast():
                fl.append(self.policy.loss(obs, act, generator=g).item())
                mse.append(self.policy.action_mse(obs, act, generator=g).item())
        return {"val/flow_loss": sum(fl) / len(fl), "val/action_mse": sum(mse) / len(mse)}

    def save(self, tag: str | None = None):
        """Save checkpoints"""
        path = self.out_dir / f"ckpt_{tag if tag is not None else f'{self.step:07d}'}.pt"
        torch.save({"step": self.step, "policy": self.policy.state_dict(),
                    "opt": self.opt.state_dict(), "policy_cfg": asdict(self.policy.cfg),
                    "train_cfg": asdict(self.cfg)}, path)
        return path

    def load(self, path):
        ck = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ck["policy"])
        self.opt.load_state_dict(ck["opt"])
        self.step = ck["step"]

    def evaluate_now(self) -> dict:
        """Main loop"""
        m = self.validate()
        if self.evaluator is not None:
            self.policy.eval()
            t0 = time.time()
            m.update(self.evaluator.evaluate(self.policy, self.step))
            m["eval/seconds"] = time.time() - t0
        self.logger.log(self.step, m)
        return m

    def train(self):
        c = self.cfg
        self.evaluate_now() # step-0 baseline
        running, t0 = [], time.time()
        while self.step < c.total_steps:
            running.append(self.train_step())
            self.step += 1

            if self.step % c.log_every == 0:
                avg = {k: sum(r[k] for r in running) / len(running) for k in running[0]}
                avg["train/steps_per_sec"] = len(running) / (time.time() - t0)
                self.logger.log(self.step, avg)
                running, t0 = [], time.time()
            if self.step % c.eval_every == 0 or self.step == c.total_steps:
                self.evaluate_now()
            if self.step % c.save_every == 0 or self.step == c.total_steps:
                self.save()
