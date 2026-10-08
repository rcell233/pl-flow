# Seed-VC-derived code (GPL-3.0); modified for PL-Flow, 2026-10-07.
# Derived from Seed-VC and Meta gpt-fast; see licenses/.
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Copyright 2023 Meta. Redistribution is subject to licenses/gpt-fast-BSD-3-Clause.txt.
import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - n % k


class AdaptiveLayerNorm(nn.Module):
    """Adaptive Layer Normalization"""

    def __init__(self, d_model, norm) -> None:
        super(AdaptiveLayerNorm, self).__init__()
        self.project_layer = nn.Linear(d_model, 2 * d_model)
        self.norm = norm
        self.d_model = d_model
        self.eps = self.norm.eps

    def forward(self, input: Tensor, embedding: Tensor = None) -> Tensor:
        if embedding is None:
            return self.norm(input)
        (weight, bias) = torch.split(
            self.project_layer(embedding), split_size_or_sections=self.d_model, dim=-1
        )
        return weight * self.norm(input) + bias


@dataclass
class ModelArgs:
    block_size: int = 2048
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-05

    def __post_init__(self):
        if self.dim != self.n_head * self.head_dim:
            raise ValueError("Model width must equal attention heads times head width")
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList((TransformerBlock(config) for _ in range(config.n_layer)))
        self.norm = AdaptiveLayerNorm(config.dim, RMSNorm(config.dim, eps=config.norm_eps))
        self.freqs_cis: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1

    @torch.inference_mode(False)
    def setup_caches(self, max_batch_size, max_seq_length):
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        dtype = self.norm.project_layer.weight.dtype
        device = self.norm.project_layer.weight.device
        self.freqs_cis = precompute_freqs_cis(
            self.config.block_size, self.config.head_dim, self.config.rope_base, dtype
        ).to(device)

    def forward(self, x: Tensor, c: Tensor, input_pos: Tensor, mask: Tensor) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        freqs_cis = self.freqs_cis[input_pos]
        skip_in_x_list = []
        for i, layer in enumerate(self.layers):
            if i > self.config.n_layer // 2:
                skip_in_x = skip_in_x_list.pop(-1)
            else:
                skip_in_x = None
            x = layer(x, c, freqs_cis, mask, skip_in_x)
            if i < self.config.n_layer // 2:
                skip_in_x_list.append(x)
        x = self.norm(x, c)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = AdaptiveLayerNorm(config.dim, RMSNorm(config.dim, eps=config.norm_eps))
        self.attention_norm = AdaptiveLayerNorm(
            config.dim, RMSNorm(config.dim, eps=config.norm_eps)
        )
        self.skip_in_linear = nn.Linear(config.dim * 2, config.dim)

    def forward(
        self,
        x: Tensor,
        c: Tensor,
        freqs_cis: Tensor,
        mask: Tensor,
        skip_in_x: Optional[Tensor] = None,
    ) -> Tensor:
        if skip_in_x is not None:
            x = self.skip_in_linear(torch.cat([x, skip_in_x], dim=-1))
        h = x + self.attention(self.attention_norm(x, c), freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h, c))
        return out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0
        self.wqkv = nn.Linear(config.dim, 3 * config.dim, bias=False)
        self.wo = nn.Linear(config.head_dim * config.n_head, config.dim, bias=False)
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.dim = config.dim
        self.q_norm = RMSNorm(config.head_dim, eps=config.norm_eps)
        self.k_norm = RMSNorm(config.head_dim, eps=config.norm_eps)

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Tensor) -> Tensor:
        (bsz, seqlen, _) = x.shape
        (q, k, v) = self.wqkv(x).split([self.dim, self.dim, self.dim], dim=-1)
        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim)
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)
        (q, k, v) = map(lambda x: x.transpose(1, 2), (q, k, v))
        k = k.repeat_interleave(1, dim=1)
        v = v.repeat_interleave(1, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.head_dim * self.n_head)
        y = self.wo(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-05):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(
    seq_len: int, n_elem: int, base: int = 10000, dtype: torch.dtype = torch.bfloat16
) -> Tensor:
    freqs = 1.0 / base ** (torch.arange(0, n_elem, 2)[: n_elem // 2].float() / n_elem)
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = 10000
        self.scale = 1000
        half = frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        )
        self.register_buffer("freqs", freqs)

    def timestep_embedding(self, t):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        args = self.scale * t[:, None].float() * self.freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t)
        t_emb = self.mlp(t_freq)
        return t_emb


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06)
        self.linear = weight_norm(
            nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        (shift, scale) = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
