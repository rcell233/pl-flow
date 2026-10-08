# Seed-VC-derived code (GPL-3.0); modified for PL-Flow, 2026-10-07.
"""Phoneme prosody, duration prediction and frame expansion."""

# VITS/Matcha-TTS-derived layers; copyright holders and terms are retained in licenses/.
import math

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm

from .utils import fused_add_tanh_sigmoid_multiply


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = int(length.max().item())
    positions = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return positions.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration, mask):
    """Build a monotonic hard alignment from [B, text] durations."""
    (batch, text_length, frame_length) = mask.shape
    cumulative = torch.cumsum(duration, dim=1)
    path = sequence_mask(cumulative.reshape(batch * text_length), frame_length).to(mask.dtype)
    path = path.view(batch, text_length, frame_length)
    path = path - F.pad(path, (0, 0, 1, 0))[:, :-1]
    return path * mask


class ScalarQuantize9(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs):
        del ctx
        return torch.round(9 * inputs) / 9

    @staticmethod
    def backward(ctx, grad_output):
        del ctx
        return grad_output


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=0.0001):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        mean = torch.mean(x, 1, keepdim=True)
        variance = torch.mean((x - mean) ** 2, 1, keepdim=True)
        x = (x - mean) * torch.rsqrt(variance + self.eps)
        shape = [1, -1] + [1] * (x.ndim - 2)
        return x * self.gamma.view(*shape) + self.beta.view(*shape)


class ConvReluNorm(nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, kernel_size, n_layers, p_dropout
    ):
        super().__init__()
        self.n_layers = n_layers
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.conv_layers.append(
            nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size // 2)
        )
        self.norm_layers.append(LayerNorm(hidden_channels))
        for _ in range(n_layers - 1):
            self.conv_layers.append(
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2)
            )
            self.norm_layers.append(LayerNorm(hidden_channels))
        self.relu_drop = nn.Sequential(nn.ReLU(), nn.Dropout(p_dropout))
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask):
        residual = x
        for index in range(self.n_layers):
            x = self.conv_layers[index](x * x_mask)
            x = self.norm_layers[index](x)
            x = self.relu_drop(x)
        return (residual + self.proj(x)) * x_mask


class WN(nn.Module):
    """The duration-predictor WaveNet used by the reference conditioner."""

    def __init__(
        self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0
    ):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("WN kernel_size must be odd")
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)
        if gin_channels:
            self.cond_layer = weight_norm(
                nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1), name="weight"
            )
        for index in range(n_layers):
            dilation = dilation_rate**index
            padding = (kernel_size * dilation - dilation) // 2
            self.in_layers.append(
                weight_norm(
                    nn.Conv1d(
                        hidden_channels,
                        2 * hidden_channels,
                        kernel_size,
                        dilation=dilation,
                        padding=padding,
                    ),
                    name="weight",
                )
            )
            res_skip_channels = 2 * hidden_channels if index < n_layers - 1 else hidden_channels
            self.res_skip_layers.append(
                weight_norm(nn.Conv1d(hidden_channels, res_skip_channels, 1), name="weight")
            )

    def forward(self, x, x_mask, g=None):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])
        if g is not None:
            g = self.cond_layer(g)
        for index in range(self.n_layers):
            x_in = self.in_layers[index](x)
            if g is None:
                condition = torch.zeros_like(x_in)
            else:
                offset = index * 2 * self.hidden_channels
                condition = g[:, offset : offset + 2 * self.hidden_channels]
            activations = fused_add_tanh_sigmoid_multiply(x_in, condition, n_channels_tensor)
            activations = self.drop(activations)
            res_skip = self.res_skip_layers[index](activations)
            if index < self.n_layers - 1:
                x = (x + res_skip[:, : self.hidden_channels]) * x_mask
                output = output + res_skip[:, self.hidden_channels :]
            else:
                output = output + res_skip
        return output * x_mask


class DurationPredictor(nn.Module):
    def __init__(self, hidden_channels, n_bins=512, p_dropout=0.0):
        super().__init__()
        self.n_bins = int(n_bins)
        self.enc = WN(hidden_channels, 5, 1, 3, gin_channels=0, p_dropout=p_dropout)
        self.duration_proj = nn.Linear(hidden_channels, self.n_bins, bias=False)

    def logits(self, x, x_mask):
        x = self.enc(x, x_mask)
        return self.duration_proj(x.transpose(1, 2))

    def forward(self, x, x_mask, durations):
        logits = self.logits(x, x_mask)
        valid = x_mask.squeeze(1).bool()
        targets = durations.clamp(0, self.n_bins - 1).long()
        flat_logits = logits[valid].view(-1, self.n_bins)
        flat_targets = targets[valid].view(-1)
        loss = F.cross_entropy(flat_logits, flat_targets, reduction="mean")
        accuracy = (flat_logits.argmax(dim=-1) == flat_targets).float().mean() * 100.0
        return (loss, logits, accuracy)


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, dimensions, base=10000):
        super().__init__()
        self.base = base
        self.dimensions = int(dimensions)
        self.cos_cached = None
        self.sin_cached = None

    @torch.inference_mode(False)
    def _build_cache(self, x):
        if (
            self.cos_cached is not None
            and x.shape[0] <= self.cos_cached.shape[0]
            and (x.device == self.cos_cached.device)
        ):
            return
        theta = 1.0 / self.base ** (
            torch.arange(0, self.dimensions, 2, device=x.device).float() / self.dimensions
        )
        positions = torch.arange(x.shape[0], device=x.device).float()
        idx_theta = torch.einsum("n,d->nd", positions, theta)
        idx_theta = torch.cat([idx_theta, idx_theta], dim=1)
        self.cos_cached = idx_theta.cos()[:, None, None, :]
        self.sin_cached = idx_theta.sin()[:, None, None, :]

    def _neg_half(self, x):
        half = self.dimensions // 2
        return torch.cat([-x[:, :, :, half:], x[:, :, :, :half]], dim=-1)

    def forward(self, x):
        x = rearrange(x, "b h t d -> t b h d")
        self._build_cache(x)
        (x_rope, x_pass) = (x[..., : self.dimensions], x[..., self.dimensions :])
        x_rope = (
            x_rope * self.cos_cached[: x.shape[0]]
            + self._neg_half(x_rope) * self.sin_cached[: x.shape[0]]
        )
        return rearrange(torch.cat((x_rope, x_pass), dim=-1), "t b h d -> b h t d")


class MultiHeadAttention(nn.Module):
    def __init__(self, channels, out_channels, n_heads, p_dropout=0.0):
        super().__init__()
        if channels % n_heads:
            raise ValueError(f"channels={channels} must be divisible by n_heads={n_heads}")
        self.channels = channels
        self.n_heads = n_heads
        self.head_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.query_rotary = RotaryPositionalEmbeddings(self.head_channels * 0.5)
        self.key_rotary = RotaryPositionalEmbeddings(self.head_channels * 0.5)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)
        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)

    def forward(self, x, condition, attention_mask=None):
        query = rearrange(self.conv_q(x), "b (h c) t -> b h t c", h=self.n_heads)
        key = rearrange(self.conv_k(condition), "b (h c) t -> b h t c", h=self.n_heads)
        value = rearrange(self.conv_v(condition), "b (h c) t -> b h t c", h=self.n_heads)
        query = self.query_rotary(query)
        key = self.key_rotary(key)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_channels)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -10000.0)
        attention = self.drop(torch.softmax(scores, dim=-1))
        output = torch.matmul(attention, value)
        output = output.transpose(2, 3).contiguous().view(x.size(0), self.channels, x.size(2))
        return self.conv_o(output)


class FFN(nn.Module):
    def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0.0):
        super().__init__()
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.conv_2 = nn.Conv1d(
            filter_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = self.drop(torch.relu(x))
        return self.conv_2(x * x_mask) * x_mask


class Encoder(nn.Module):
    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout):
        super().__init__()
        self.drop = nn.Dropout(p_dropout)
        self.attention_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for _ in range(n_layers):
            self.attention_layers.append(
                MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout)
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout)
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask):
        attention_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        for index in range(len(self.attention_layers)):
            x = x * x_mask
            residual = self.attention_layers[index](x, x, attention_mask)
            x = self.norm_layers_1[index](x + self.drop(residual))
            residual = self.ffn_layers[index](x, x_mask)
            x = self.norm_layers_2[index](x + self.drop(residual))
        return x * x_mask


class MatchaTextConditioner(nn.Module):
    """Speaker-free copy of the reference text/prosody/duration conditioner."""

    def __init__(
        self,
        n_vocab,
        output_channels=32,
        ssl_dim=768,
        hidden_channels=192,
        prosody_dim=16,
        filter_channels=768,
        duration_bins=512,
        n_heads=2,
        n_layers=6,
        kernel_size=3,
        p_dropout=0.1,
        prenet=True,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.embedding = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.embedding.weight, 0.0, hidden_channels ** (-0.5))
        self.prenet = (
            ConvReluNorm(
                hidden_channels,
                hidden_channels,
                hidden_channels,
                kernel_size=5,
                n_layers=3,
                p_dropout=0.5,
            )
            if prenet
            else None
        )
        self.prosody_down = nn.Conv1d(ssl_dim, prosody_dim, 1)
        self.prosody_up = nn.Conv1d(prosody_dim, hidden_channels, 1)
        self.encoder = Encoder(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )
        self.condition_projection = nn.Conv1d(hidden_channels, output_channels, 1)
        self.duration_predictor = DurationPredictor(hidden_channels, duration_bins, p_dropout)

    def forward(self, text, text_lengths, pooled_ssl, durations):
        x_mask = sequence_mask(text_lengths, text.size(1)).unsqueeze(1).to(pooled_ssl.dtype)
        prosody_latent = self.prosody_down(pooled_ssl) * x_mask
        prosody_latent = ScalarQuantize9.apply(torch.tanh(prosody_latent)) * x_mask
        return self.forward_from_prosody(text, text_lengths, prosody_latent, durations)

    def forward_from_prosody(self, text, text_lengths, prosody_latent, durations=None):
        """Consume decoded scalar codes, bypassing SSL pooling and prosody_down."""
        if prosody_latent.shape != (text.size(0), self.prosody_up.in_channels, text.size(1)):
            raise ValueError("Prosody codes must align with the complete phoneme sequence")
        x_mask = sequence_mask(text_lengths, text.size(1)).unsqueeze(1).to(prosody_latent.dtype)
        x = self.embedding(text).transpose(1, 2) * math.sqrt(self.hidden_channels)
        x = x * x_mask
        if self.prenet is not None:
            x = self.prenet(x, x_mask)
        prosody_latent = prosody_latent * x_mask
        prosody = self.prosody_up(prosody_latent) * x_mask
        x = self.encoder(x + prosody, x_mask)
        token_condition = self.condition_projection(x) * x_mask
        if durations is None:
            duration_logits = self.duration_predictor.logits(x.detach(), x_mask)
            (duration_loss, duration_accuracy) = (None, None)
        else:
            (duration_loss, duration_logits, duration_accuracy) = self.duration_predictor(
                x.detach(), x_mask, durations
            )
        return {
            "token_condition": token_condition,
            "duration_logits": duration_logits,
            "duration_loss": duration_loss,
            "duration_accuracy": duration_accuracy,
            "prosody": prosody,
            "prosody_latent": prosody_latent,
            "text_mask": x_mask,
        }


class LatentTTSConditioner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.duration_bins = int(config.get("duration_bins", 512))
        self.max_token_duration = int(config.get("max_token_duration", self.duration_bins - 1))
        self.max_infer_frames = int(config.get("max_infer_frames", 1000))
        self.text_conditioner = MatchaTextConditioner(
            n_vocab=int(config.get("n_vocab", 178)),
            output_channels=int(config.get("output_channels", 32)),
            ssl_dim=int(config.get("ssl_dim", 768)),
            hidden_channels=int(config.get("hidden_channels", 192)),
            prosody_dim=int(config.get("prosody_dim", 16)),
            filter_channels=int(config.get("filter_channels", 768)),
            duration_bins=self.duration_bins,
            n_heads=int(config.get("n_heads", 2)),
            n_layers=int(config.get("n_layers", 6)),
            kernel_size=int(config.get("kernel_size", 3)),
            p_dropout=float(config.get("p_dropout", 0.1)),
            prenet=bool(config.get("prenet", True)),
        )

    @staticmethod
    def pool_ssl_to_phonemes(ssl, durations, text_lengths, frame_lengths):
        text_mask = sequence_mask(text_lengths, durations.size(1))
        frame_mask = sequence_mask(frame_lengths, ssl.size(-1))
        attention_mask = text_mask.unsqueeze(-1) * frame_mask.unsqueeze(1)
        attention = generate_path(durations, attention_mask.to(ssl.dtype))
        pooled = torch.bmm(ssl, attention.transpose(1, 2))
        pooled = pooled / durations.clamp_min(1).unsqueeze(1).to(pooled.dtype)
        pooled = pooled * text_mask.unsqueeze(1).to(pooled.dtype)
        return (pooled, attention)

    @staticmethod
    def fit_durations_to_targets(durations, text_lengths, target_lengths):
        aligned = durations.clone()
        for batch_index in range(durations.size(0)):
            text_length = int(text_lengths[batch_index].item())
            target_length = int(target_lengths[batch_index].item())
            duration_length = int(aligned[batch_index, :text_length].sum().item())
            residual = target_length - duration_length
            if residual < 0:
                raise ValueError(
                    f"duration frames exceed target frames: {duration_length} > {target_length}"
                )
            aligned[batch_index, text_length - 1] += residual
        return aligned

    @staticmethod
    def expand_token_condition(token_condition, durations, text_lengths, target_lengths):
        text_mask = sequence_mask(text_lengths, durations.size(1))
        frame_mask = sequence_mask(target_lengths, int(target_lengths.max().item()))
        attention_mask = text_mask.unsqueeze(-1) * frame_mask.unsqueeze(1)
        attention = generate_path(durations, attention_mask.to(token_condition.dtype))
        frame_condition = torch.bmm(attention.transpose(1, 2), token_condition.transpose(1, 2))
        frame_condition = frame_condition * frame_mask.unsqueeze(-1).to(frame_condition.dtype)
        return (frame_condition, attention)

    def forward(self, text, text_lengths, durations, ssl, target_lengths):
        (pooled_ssl, pool_attention) = self.pool_ssl_to_phonemes(
            ssl, durations, text_lengths, target_lengths
        )
        outputs = self.text_conditioner(text, text_lengths, pooled_ssl, durations)
        aligned_durations = self.fit_durations_to_targets(durations, text_lengths, target_lengths)
        (frame_condition, attention) = self.expand_token_condition(
            outputs["token_condition"], aligned_durations, text_lengths, target_lengths
        )
        return {
            **outputs,
            "condition": frame_condition,
            "attention": attention,
            "pool_attention": pool_attention,
            "aligned_durations": aligned_durations,
        }

    def forward_segments(self, text, text_lengths, codes, durations, target_lengths, prompt_tokens):
        """Encode reference/target independently, then restore the original frame timeline."""
        segments = []
        for b, (length, cut) in enumerate(zip(text_lengths.tolist(), prompt_tokens.tolist())):
            if not 0 <= cut < length:
                raise ValueError("Prompt must leave at least one target token")
            if cut:
                segments.append((b, 0, cut))
            segments.append((b, cut, length))
        lengths = text_lengths.new_tensor([end - start for (_, start, end) in segments])
        width = int(lengths.max())
        ids = text.new_zeros(len(segments), width)
        prosody = codes.new_zeros(len(segments), codes.size(1), width)
        dur = durations.new_zeros(len(segments), width)
        for s, (b, start, end) in enumerate(segments):
            ids[s, : end - start] = text[b, start:end]
            prosody[s, :, : end - start] = codes[b, :, start:end]
            dur[s, : end - start] = durations[b, start:end]
        out = self.text_conditioner.forward_from_prosody(ids, lengths, prosody, dur)
        tokens = out["token_condition"].new_zeros(
            text.size(0), out["token_condition"].size(1), text.size(1)
        )
        logits = out["duration_logits"].new_zeros(*text.shape, self.duration_bins)
        for s, (b, start, end) in enumerate(segments):
            tokens[b, :, start:end] = out["token_condition"][s, :, : end - start]
            logits[b, start:end] = out["duration_logits"][s, : end - start]
        aligned = self.fit_durations_to_targets(durations, text_lengths, target_lengths)
        (condition, _) = self.expand_token_condition(tokens, aligned, text_lengths, target_lengths)
        return {
            "condition": condition,
            "token_condition": tokens,
            "duration_logits": logits,
            "duration_loss": out["duration_loss"],
            "duration_accuracy": out["duration_accuracy"],
            "prosody_latent": codes,
        }

    def build_inference_condition(
        self, token_condition, duration_logits, text_length, prompt_length=0
    ):
        """Expand predicted token durations into a frame-level condition."""
        text_length = int(text_length)
        tokens = token_condition[:, :text_length]
        predicted = duration_logits[:text_length].argmax(dim=-1).long()
        predicted = predicted.clamp(max=self.max_token_duration)
        if int(predicted.sum().item()) == 0:
            predicted[0] = 1
        if int(predicted.sum()) + int(prompt_length) > self.max_infer_frames:
            raise ValueError("Reference + predicted target exceed frame budget; split target text")
        condition = torch.repeat_interleave(tokens.transpose(0, 1), predicted, dim=0)
        output_length = condition.size(0)
        return {
            "condition": condition.unsqueeze(0),
            "output_length": torch.tensor(
                [output_length], device=condition.device, dtype=torch.long
            ),
            "predicted_durations": predicted,
        }
