"""Mono loading, resampling and length fitting shared by every stage."""

import torch
import torchaudio
from torch.nn import functional as F


def load_audio(path, sample_rate):
    wave, source_rate = torchaudio.load(str(path))
    wave = wave.float().mean(0, keepdim=True)
    if source_rate != sample_rate:
        wave = torchaudio.functional.resample(wave, source_rate, sample_rate)
    if wave.size(-1) == 0 or not torch.isfinite(wave).all():
        raise ValueError("Empty or non-finite waveform")
    return wave.clamp(-1, 1)


def fit_length(wave, length):
    return F.pad(wave, (0, max(0, length - wave.size(-1))))[..., :length]
