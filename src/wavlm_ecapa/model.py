# SPDX-License-Identifier: CC-BY-SA-3.0
"""Frozen WavLM-Large/ECAPA speaker embedding from local weights.

The downstream architecture follows UniSpeech; see NOTICE.md.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .ecapa import AttentiveStatsPool, Conv1dReluBn, SE_Res2Block
from .wavlm.model import WavLM, WavLMConfig


class WavLMFeatures(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.normalize = bool(config["normalize"])
        self.model = WavLM(WavLMConfig(config))
        self.model.feature_grad_mult = 0.0
        self.model.encoder.layerdrop = 0.0
        self.hidden = []
        for layer in self.model.encoder.layers:
            layer.register_forward_hook(self._layer_hook)
        self.model.encoder.register_forward_hook(self._encoder_hook)

    def _layer_hook(self, module, inputs, output):
        self.hidden.append(inputs[0].transpose(0, 1))

    def _encoder_hook(self, module, inputs, output):
        self.hidden.append(output[0])

    def forward(self, audio):
        if self.normalize:
            audio = F.layer_norm(audio, audio.shape[-1:])
        self.hidden.clear()
        try:
            self.model.extract_features(
                audio, padding_mask=torch.zeros_like(audio, dtype=torch.bool), mask=False
            )
            return tuple(self.hidden)
        finally:
            self.hidden.clear()


class SpeakerEncoder(nn.Module):
    embedding_dim = 256

    def __init__(self, config):
        super().__init__()
        self.feature_extract = WavLMFeatures(config)
        self.feature_weight = nn.Parameter(torch.zeros(config["encoder_layers"] + 1))
        self.instance_norm = nn.InstanceNorm1d(1024)
        self.layer1 = Conv1dReluBn(1024, 512, kernel_size=5, padding=2)
        for name, dilation in [("layer2", 2), ("layer3", 3), ("layer4", 4)]:
            setattr(self, name, SE_Res2Block(512, 512, 3, 1, dilation, dilation, 8, 128))
        self.conv = nn.Conv1d(1536, 1536, 1)
        self.pooling = AttentiveStatsPool(1536, attention_channels=128, global_context_att=False)
        self.bn = nn.BatchNorm1d(3072)
        self.linear = nn.Linear(3072, 256)
        self.requires_grad_(False).eval()

    def train(self, mode=True):
        return super().train(False)

    def _embed(self, audio):
        features = torch.stack(self.feature_extract(audio))
        weights = F.softmax(self.feature_weight, dim=-1)[:, None, None, None]
        x = (weights * features).sum(0).transpose(1, 2) + 1e-6
        x = self.instance_norm(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return self.linear(self.bn(self.pooling(F.relu(self.conv(torch.cat([x2, x3, x4], 1))))))

    @torch.no_grad()
    def forward(self, waves, lengths):
        if waves.ndim != 2 or lengths.shape != (len(waves),) or (lengths <= 0).any():
            raise ValueError("Invalid speaker waveform batch")
        with torch.autocast(waves.device.type, enabled=False):
            return torch.cat(
                [self._embed(w[None, : int(n)].float()) for w, n in zip(waves, lengths)]
            )
