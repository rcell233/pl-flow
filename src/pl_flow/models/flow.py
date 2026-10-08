# Seed-VC-derived code (GPL-3.0); modified for PL-Flow, 2026-10-07.
"""Uniform-time conditional flow matching and fixed-step Euler sampling."""

import torch
from torch import nn

from .dit import DiT


class CFM(nn.Module):
    prompt_condition = True
    sigma_min = 1e-6

    def __init__(self, config):
        super().__init__()
        self.in_channels = config.DiT.in_channels
        self.estimator = DiT(config)
        self.criterion = nn.L1Loss()

    @staticmethod
    def prompt_masks(x, lengths, prompt_lens):
        if prompt_lens is None or prompt_lens.shape != lengths.shape or len(lengths) != len(x):
            raise ValueError("One prefix length and total length are required per sample")
        if ((prompt_lens < 0) | (prompt_lens >= lengths) | (lengths > x.size(-1))).any():
            raise ValueError("A prompt must leave a nonempty target within the sequence")
        positions = torch.arange(x.size(-1), device=x.device)[None, None]
        prefix = positions < prompt_lens[:, None, None]
        generated = (positions >= prompt_lens[:, None, None]) & (positions < lengths[:, None, None])
        return prefix, generated

    def forward(self, x1, x_lens, mu, style, prompt_lens=None):
        t = torch.rand([x1.size(0), 1, 1], device=mu.device, dtype=x1.dtype)
        z = torch.randn_like(x1)
        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        velocity = x1 - (1 - self.sigma_min) * z
        prefix, generated = self.prompt_masks(x1, x_lens, prompt_lens)
        prompt = x1.masked_fill(~prefix, 0)
        estimate = self.estimator(
            y.masked_fill(~generated, 0),
            x_lens,
            t.squeeze(1).squeeze(1),
            style,
            mu,
            prompt_x=prompt,
        )
        loss = 0
        for b in range(len(x1)):
            start = int(prompt_lens[b])
            loss += self.criterion(
                estimate[b, :, start : x_lens[b]], velocity[b, :, start : x_lens[b]]
            )
        return loss / len(x1), estimate + (1 - self.sigma_min) * z

    @torch.inference_mode()
    def inference(
        self,
        mu,
        x_lens,
        style,
        f0=None,
        n_timesteps=10,
        temperature=1.0,
        inference_cfg_rate=2.8,
        prompt=None,
        prompt_lens=None,
    ):
        if n_timesteps < 1 or temperature < 0 or inference_cfg_rate < 0:
            raise ValueError("Steps must be positive; temperature and guidance must be nonnegative")
        x = torch.randn([len(mu), self.in_channels, mu.size(1)], device=mu.device) * temperature
        times = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        return self.solve_euler(
            x, x_lens, mu, style, f0, times, inference_cfg_rate, prompt, prompt_lens
        )

    def solve_euler(
        self,
        x,
        x_lens,
        mu,
        style,
        f0,
        t_span,
        inference_cfg_rate=2.8,
        prompt=None,
        prompt_lens=None,
    ):
        prefix, generated = self.prompt_masks(x, x_lens, prompt_lens)
        if (
            prompt is None
            or prompt.shape[:2] != x.shape[:2]
            or not int(prompt_lens.max()) <= prompt.size(-1) <= x.size(-1)
        ):
            raise ValueError("Prompt shape/length mismatch")
        prompt_x = torch.zeros_like(x)
        prompt_x[..., : prompt.size(-1)] = prompt
        prompt_x = prompt_x.masked_fill(~prefix, 0)
        x = x.masked_fill(~generated, 0)
        t = t_span[0]
        for step in range(1, len(t_span)):
            dt = t_span[step] - t_span[step - 1]
            if inference_cfg_rate > 0:
                doubled = torch.cat([x, x])
                estimate = self.estimator(
                    doubled,
                    torch.cat([x_lens, x_lens]),
                    t.expand(len(doubled)),
                    torch.cat([style, torch.zeros_like(style)]),
                    torch.cat([mu, torch.zeros_like(mu)]),
                    prompt_x=torch.cat([prompt_x, torch.zeros_like(prompt_x)]),
                )
                conditional, unconditional = estimate.chunk(2)
                estimate = (
                    1 + inference_cfg_rate
                ) * conditional - inference_cfg_rate * unconditional
            else:
                estimate = self.estimator(x, x_lens, t.expand(len(x)), style, mu, prompt_x=prompt_x)
            x = (x + dt * estimate).masked_fill(~generated, 0)
            t = t + dt
        return x
