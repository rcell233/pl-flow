# Seed-VC-derived code (GPL-3.0); modified for PL-Flow, 2026-10-07.
"""Non-causal prompt-completion DiT with QK RMS normalization."""

import torch
from torch import nn

from .transformer import FinalLayer, ModelArgs, TimestepEmbedder, Transformer
from .utils import sequence_mask
from .wavenet import WN


class DiT(nn.Module):
    def __init__(self, args):
        super().__init__()
        d, w = args.DiT, args.wavenet
        self.in_channels = d.in_channels
        self.class_dropout_prob = d.class_dropout_prob
        self.transformer = Transformer(
            ModelArgs(
                block_size=16384,
                n_layer=d.depth,
                n_head=d.num_heads,
                dim=d.hidden_dim,
                head_dim=d.hidden_dim // d.num_heads,
            )
        )
        self.t_embedder = TimestepEmbedder(d.hidden_dim)
        self.register_buffer("input_pos", torch.arange(16384))
        self.t_embedder2 = TimestepEmbedder(w.hidden_dim)
        self.conv1 = nn.Linear(d.hidden_dim, w.hidden_dim)
        self.conv2 = nn.Conv1d(w.hidden_dim, d.in_channels, 1)
        self.wavenet = WN(
            w.hidden_dim,
            w.kernel_size,
            w.dilation_rate,
            w.num_layers,
            gin_channels=w.hidden_dim,
            p_dropout=w.p_dropout,
            causal=False,
        )
        self.final_layer = FinalLayer(w.hidden_dim, 1, w.hidden_dim)
        self.res_projection = nn.Linear(d.hidden_dim, w.hidden_dim)
        self.skip_linear = nn.Linear(d.hidden_dim + d.in_channels, d.hidden_dim)
        self.cond_x_merge_linear = nn.Linear(
            d.content_dim + 2 * d.in_channels + args.style_encoder.dim, d.hidden_dim
        )

    def setup_caches(self, max_batch_size, max_seq_length):
        if max_seq_length > self.input_pos.numel():
            raise ValueError("Sequence exceeds the positional context")
        self.transformer.setup_caches(max_batch_size, max_seq_length)

    def forward(self, x, x_lens, t, style, cond, mask_content=False, prompt_x=None):
        drop = False
        # Preserve the original batch-wise draw on the CPU generator.
        if self.training and torch.rand(1) < self.class_dropout_prob:
            drop = True
        if not self.training and mask_content:
            drop = True
        if prompt_x is None or prompt_x.shape != x.shape:
            raise ValueError("A frame/token-aligned prompt tensor is required")
        _, _, width = x.shape
        t1 = self.t_embedder(t)
        x = x.transpose(1, 2)
        merged = torch.cat([x, prompt_x.transpose(1, 2), cond], dim=-1)
        merged = torch.cat([merged, style[:, None, :].repeat(1, width, 1)], dim=-1)
        if drop:
            merged[..., self.in_channels :] = merged[..., self.in_channels :] * 0
        merged = self.cond_x_merge_linear(merged)
        mask = sequence_mask(x_lens, width).to(x.device).unsqueeze(1)
        attention_mask = mask[:, None, :].repeat(1, 1, width, 1)
        hidden = self.transformer(merged, t1.unsqueeze(1), self.input_pos[:width], attention_mask)
        hidden = self.skip_linear(torch.cat([hidden, x], dim=-1))
        x = self.conv1(hidden).transpose(1, 2)
        t2 = self.t_embedder2(t)
        x = self.wavenet(x, mask, g=t2.unsqueeze(2)).transpose(1, 2)
        x = x + self.res_projection(hidden)
        return self.conv2(self.final_layer(x, t1).transpose(1, 2))
