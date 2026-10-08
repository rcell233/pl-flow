"""HuBERT-to-phoneme encoder/decoder aligner."""

import math
from argparse import Namespace

import torch
from torch import nn

from .transformer import (
    AlibiPostionEmbedding,
    CrossAttnOnlyLayer,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)


class ASRTransformerAligner(nn.Module):
    CTC_BLANK_ID = 178
    SOS_ID = 179
    EOS_ID = 180

    def __init__(
        self,
        input_dim=768,
        hidden_size=768,
        hidden_dim=None,
        n_token=181,
        token_embedding_dim=768,
        nheads=12,
        ffd_size=3072,
        enc_nlayers=6,
        dec_nlayers=6,
        layer_norm_eps=1e-05,
        aligner_softmax_temp=1.0,
        dropout=0.1,
        max_length=10000,
    ):
        super().__init__()
        if hidden_dim is not None:
            hidden_size = hidden_dim
        if token_embedding_dim != hidden_size:
            raise ValueError(
                f"token_embedding_dim must equal hidden_size for this model, got {token_embedding_dim} vs {hidden_size}"
            )
        hp = Namespace(
            hidden_size=hidden_size,
            nheads=nheads,
            ffd_size=ffd_size,
            layer_norm_eps=layer_norm_eps,
            aligner_softmax_temp=aligner_softmax_temp,
        )
        self.n_token = n_token
        self.n_down = 0
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.ssl_proj = nn.Sequential(
            nn.Conv1d(input_dim, hidden_size, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
        )
        self.encoder = TransformerEncoder(
            nn.ModuleList(
                [TransformerEncoderLayer(hp, dropout=dropout) for _ in range(enc_nlayers)]
            )
        )
        self.decoder = TransformerDecoder(
            nn.ModuleList(
                [TransformerDecoderLayer(hp, dropout=dropout) for _ in range(dec_nlayers)]
            )
        )
        self.aligner = CrossAttnOnlyLayer(hp, dropout=dropout)
        self.token_embedding = nn.Embedding(n_token, hidden_size)
        self.encoder_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.decoder_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.aligner_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.output_proj = nn.Linear(hidden_size, n_token)
        self.alibi = AlibiPostionEmbedding(nheads, max_length)
        val_range = math.sqrt(6 / hidden_size)
        with torch.no_grad():
            self.token_embedding.weight.uniform_(-val_range, val_range)

    def forward(self, x, src_key_padding_mask=None, text_input=None):
        (encoder_states, encoder_attention) = self.encode_ssl(x, src_key_padding_mask)
        if text_input is None:
            return {"encoder_states": encoder_states, "encoder_attention": encoder_attention}
        (s2s_hidden, s2s_logit, s2s_attn, decoder_attention) = self.decode_tokens(
            encoder_states, src_key_padding_mask, text_input
        )
        return (None, s2s_logit, s2s_attn)

    def encode_ssl(self, x, src_key_padding_mask=None):
        x = self.ssl_proj(x).transpose(1, 2)
        x = self.encoder_norm(x)
        attn_bias = self.alibi(x)
        (encoder_states, encoder_attention) = self.encoder(
            x, attn_bias=attn_bias, src_key_padding_mask=src_key_padding_mask
        )
        return (encoder_states, encoder_attention)

    def decode_tokens(self, memory, memory_mask, text_input):
        random_mask = (torch.rand(text_input.shape, device=text_input.device) < 0.1) & (
            text_input != 0
        )
        decoder_tokens = text_input.clone()
        decoder_tokens.masked_fill_(random_mask, 3)
        decoder_inputs = self.token_embedding(decoder_tokens)
        start_embedding = self.token_embedding(
            torch.full(
                (decoder_inputs.size(0),), self.SOS_ID, device=text_input.device, dtype=torch.long
            )
        )
        decoder_inputs = torch.cat([start_embedding.unsqueeze(1), decoder_inputs], dim=1)
        decoder_inputs = self.decoder_norm(decoder_inputs)
        tgt_mask = self.get_future_mask(decoder_inputs.size(1)).to(decoder_inputs.device)
        attn_bias = self.alibi(decoder_inputs)
        (decoder_states, decoder_attention) = self.decoder(
            decoder_inputs, tgt_mask=tgt_mask, attn_bias=attn_bias, tgt_key_padding_mask=None
        )
        (aligned_states, alignment) = self.aligner(
            decoder_states, memory, memory_key_padding_mask=memory_mask
        )
        aligned_states = self.aligner_norm(aligned_states)
        logits = self.output_proj(aligned_states)
        return (aligned_states, logits, alignment[:, 0], decoder_attention)

    def get_feature(self, x):
        (encoded, _) = self.encode_ssl(x)
        return encoded.transpose(1, 2)

    def length_to_mask(self, lengths):
        mask = (
            torch.arange(lengths.max(), device=lengths.device)
            .unsqueeze(0)
            .expand(lengths.shape[0], -1)
        )
        return torch.gt(mask + 1, lengths.unsqueeze(1))

    def get_future_mask(self, out_length, unmask_future_steps=0):
        index_tensor = torch.arange(out_length).unsqueeze(0).expand(out_length, -1)
        return torch.gt(index_tensor, index_tensor.T + unmask_future_steps)
