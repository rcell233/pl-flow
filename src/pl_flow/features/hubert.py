"""Frozen HuBERT with the training waveform normalization and length rules."""

import torch
from torch import nn
from torch.nn import functional as F
from transformers import HubertConfig, HubertModel


class HubertFeatures(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = HubertModel(HubertConfig(**config))
        self.requires_grad_(False).eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, waves, lengths):
        if waves.ndim != 2 or lengths.shape != (len(waves),) or (lengths < 400).any():
            raise ValueError("HuBERT expects [batch, samples] at 16 kHz and at least 400 samples")
        waves = waves.float()
        mask = torch.arange(waves.size(1), device=waves.device)[None] < lengths[:, None]
        floating = mask.to(waves)
        mean = (waves * floating).sum(1, keepdim=True) / lengths.to(waves)[:, None]
        centered = waves - mean
        variance = (centered.square() * floating).sum(1, keepdim=True) / lengths.to(waves)[:, None]
        normalized = (centered / torch.sqrt(variance + 1e-7)).masked_fill(~mask, 0)
        features = self.model(normalized, attention_mask=mask.long()).last_hidden_state.float()
        sizes = (
            self.model._get_feat_extract_output_lengths(lengths).long().clamp(1, features.size(1))
        )
        return features[:, : int(sizes.max())], sizes

    @torch.no_grad()
    def aligned(self, waves, lengths):
        features, sizes = self(waves, lengths)
        features = features.transpose(1, 2)
        if features.size(-1) % 2:
            features = features[..., :-1]
            sizes = sizes.clamp_max(features.size(-1))
        return features, sizes

    @torch.no_grad()
    def to_latent_frames(self, waves, lengths, frame_lengths):
        result = waves.new_zeros(
            len(waves), self.model.config.hidden_size, int(frame_lengths.max())
        )
        for b, frames in enumerate(frame_lengths.tolist()):
            # HuBERT's convolutional group norm is sensitive to padded time.
            features, sizes = self(waves[b : b + 1, : int(lengths[b])], lengths[b : b + 1])
            result[b, :, :frames] = F.interpolate(
                features[:, : int(sizes[0])].transpose(1, 2), size=frames, mode="nearest"
            )[0]
        return result
