# Adapted from MQTTS, Copyright (c) 2022 Li-Wei Chen (MIT).
# Modified for PL-Flow: dynamic positional biases and padding masks.
# See licenses/MIT-NOTICES.txt for the original notice.
"""Attention layers for phoneme alignment."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class AlibiPostionEmbedding:
    def __init__(self, nheads, maxpos):
        self.maxpos = maxpos
        self.slopes = torch.Tensor(self.get_slopes(nheads)) * -1

    def get_slopes(self, n):
        def get_slopes_power_of_2(power):
            start = 2 ** (-(2 ** (-(math.log2(power) - 3))))
            ratio = start
            return [start * ratio**i for i in range(power)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + self.get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )

    def __call__(self, x):
        if x.size(1) > self.maxpos:
            raise ValueError("Alignment context exceeds max_length")
        positions = torch.arange(x.size(1))
        relative = (positions[None, :] - positions[:, None]).abs()
        return (self.slopes[:, None, None] * relative[None]).to(x.device)


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, softmax_temp=1.0):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.d_model = d_model
        self.head_dim = d_model // nhead
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.softmax_temp = softmax_temp

    def reshape(self, x):
        x = x.view(x.size(0), x.size(1), self.nhead, self.head_dim).transpose(1, 2).contiguous()
        return x.view(-1, x.size(2), self.head_dim)

    def forward(self, q, k, v, attn_mask=None, key_padding_mask=None, attn_bias=None):
        batch_size = q.size(0)
        q = self.q_proj(q) * self.head_dim ** (-0.5)
        k = self.k_proj(k)
        v = self.v_proj(v)
        (q, k, v) = (self.reshape(q), self.reshape(k), self.reshape(v))
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        if attn_bias is not None:
            attn_weights = attn_weights + attn_bias.unsqueeze(0).expand(
                batch_size, -1, -1, -1
            ).reshape(batch_size * self.nhead, q.size(1), k.size(1))
        if attn_mask is not None:
            expanded_attn_mask = attn_mask.unsqueeze(0).expand(batch_size * self.nhead, -1, -1)
        else:
            expanded_attn_mask = None
        if key_padding_mask is not None:
            key_padding_mask = (
                key_padding_mask.unsqueeze(1).unsqueeze(1).expand(-1, self.nhead, -1, -1)
            )
            key_padding_mask = key_padding_mask.reshape(batch_size * self.nhead, 1, k.size(1))
            if expanded_attn_mask is None:
                expanded_attn_mask = key_padding_mask.expand(-1, q.size(1), -1)
            else:
                expanded_attn_mask = expanded_attn_mask.logical_or(key_padding_mask)
        if expanded_attn_mask is not None:
            additive_mask = torch.zeros_like(expanded_attn_mask, dtype=q.dtype)
            additive_mask.masked_fill_(expanded_attn_mask, float("-inf"))
            attn_weights = attn_weights + additive_mask
        attn_weights = F.softmax(attn_weights * self.softmax_temp, dim=-1, dtype=attn_weights.dtype)
        attn_weights_reshaped = attn_weights.view(batch_size, self.nhead, q.size(1), k.size(1))
        attn_probs = self.dropout(attn_weights)
        attn_output = torch.bmm(attn_probs, v)
        attn_output = attn_output.view(batch_size, self.nhead, q.size(1), self.head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, q.size(1), self.d_model)
        return (self.out_proj(attn_output), attn_weights_reshaped)


class CrossAttnOnlyLayer(nn.Module):
    def __init__(self, hp, dropout=0.1):
        super().__init__()
        self.multihead_attn = MultiheadAttention(
            hp.hidden_size, 1, dropout=dropout, softmax_temp=hp.aligner_softmax_temp
        )
        self.linear1 = nn.Linear(hp.hidden_size, hp.ffd_size)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hp.ffd_size, hp.hidden_size)
        self.norm1 = nn.LayerNorm(hp.hidden_size, eps=hp.layer_norm_eps)
        self.norm2 = nn.LayerNorm(hp.hidden_size, eps=hp.layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, tgt, memory, memory_key_padding_mask=None):
        (tgt2, attn) = self.multihead_attn(
            tgt, memory, memory, key_padding_mask=memory_key_padding_mask
        )
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        return (tgt, attn)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hp, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(hp.hidden_size, hp.nheads, dropout=dropout)
        self.linear1 = nn.Linear(hp.hidden_size, hp.ffd_size)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hp.ffd_size, hp.hidden_size)
        self.norm1 = nn.LayerNorm(hp.hidden_size, eps=hp.layer_norm_eps)
        self.norm2 = nn.LayerNorm(hp.hidden_size, eps=hp.layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src, src_mask=None, attn_bias=None, src_key_padding_mask=None):
        (res, self_attn) = self.self_attn(
            src,
            src,
            src,
            attn_mask=src_mask,
            attn_bias=attn_bias,
            key_padding_mask=src_key_padding_mask,
        )
        src = self.norm1(src + self.dropout1(res))
        res = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(res))
        return (src, self_attn)


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layers):
        super().__init__()
        self.layers = encoder_layers

    def forward(self, src, mask=None, attn_bias=None, src_key_padding_mask=None):
        output = src
        attns = []
        for mod in self.layers:
            (output, attn) = mod(
                output,
                src_mask=mask,
                attn_bias=attn_bias,
                src_key_padding_mask=src_key_padding_mask,
            )
            attns.append(attn.detach())
        return (output, attns)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, hp, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(hp.hidden_size, hp.nheads, dropout=dropout)
        self.linear1 = nn.Linear(hp.hidden_size, hp.ffd_size)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hp.ffd_size, hp.hidden_size)
        self.norm1 = nn.LayerNorm(hp.hidden_size, eps=hp.layer_norm_eps)
        self.norm2 = nn.LayerNorm(hp.hidden_size, eps=hp.layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, tgt, tgt_mask=None, attn_bias=None, tgt_key_padding_mask=None):
        (tgt2, self_attn) = self.self_attn(
            tgt,
            tgt,
            tgt,
            attn_mask=tgt_mask,
            attn_bias=attn_bias,
            key_padding_mask=tgt_key_padding_mask,
        )
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        return (tgt, self_attn)


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layers):
        super().__init__()
        self.layers = decoder_layers

    def forward(self, tgt, tgt_mask=None, attn_bias=None, tgt_key_padding_mask=None):
        output = tgt
        self_attns = []
        for mod in self.layers:
            (output, self_attn) = mod(
                output,
                tgt_mask=tgt_mask,
                attn_bias=attn_bias,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )
            self_attns.append(self_attn.detach())
        return (output, self_attns)
