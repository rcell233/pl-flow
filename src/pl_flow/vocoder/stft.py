"""STFT encoder and decoder with scalar-quantized latent codes.

Spectral arrays have shape [batch, channels, frequency, time]. Normalization
and attention treat each time position independently; only convolutions and
explicit strides mix time positions.
"""

from collections.abc import Sequence
from math import sqrt
from numbers import Integral

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from pl_flow.models.conditioner import WN, ScalarQuantize9, sequence_mask

__all__ = ["STFT2DEncoder", "STFT2DDecoder"]

_MAGNITUDE_EXPONENT = 0.65
_MAGNITUDE_SCALE = 0.34


def _integer(name: str, value: int, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}; got {value!r}")
    return int(value)


def _integer_sequence(name: str, values: Sequence[int], minimum: int) -> tuple[int, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of integers")
    return tuple(_integer(f"{name}[{i}]", value, minimum) for i, value in enumerate(values))


class _Geometry:
    """Validated static geometry shared by the analysis and synthesis builders."""

    def __init__(
        self,
        hop: int,
        window: int,
        base: int,
        depths: Sequence[int],
        multipliers: Sequence[int],
        attention: Sequence[int],
        factors: Sequence[int],
        heads: int,
        temporal_factor: int,
    ) -> None:
        self.hop = _integer("stft_hop_length", hop)
        self.window = _integer("stft_win_length", window)
        if self.window != 4 * self.hop:
            raise ValueError("stft_win_length must equal 4 * stft_hop_length")
        base = _integer("m2l_base_channels", base)
        self.depths = _integer_sequence("m2l_layers", depths, 0)
        multipliers = _integer_sequence("m2l_multipliers", multipliers, 1)
        if not isinstance(attention, Sequence) or isinstance(attention, (str, bytes)):
            raise ValueError("m2l_attention must be a sequence of 0/1 flags")
        if any(not isinstance(flag, Integral) or flag not in (0, 1) for flag in attention):
            raise ValueError("m2l_attention must contain only 0/1 flags")
        self.attention = tuple(bool(flag) for flag in attention)
        count = len(self.depths)
        if count == 0 or len(multipliers) != count or len(self.attention) != count:
            raise ValueError(
                "m2l_layers, m2l_multipliers, and m2l_attention need equal nonzero lengths"
            )
        self.factors = _integer_sequence("m2l_freq_downsample_factors", factors, 1)
        if len(self.factors) != count - 1:
            raise ValueError(
                "m2l_freq_downsample_factors must have one entry per resolution boundary"
            )
        self.heads = _integer("m2l_heads", heads)
        self.temporal_factor = _integer("m2l_last_time_downsample_factor", temporal_factor)
        if count == 1 and self.temporal_factor != 1:
            raise ValueError("a one-resolution model requires m2l_last_time_downsample_factor=1")
        self.widths = tuple(base * multiplier for multiplier in multipliers)
        frequencies = [2 * self.hop]
        for index, factor in enumerate(self.factors):
            if frequencies[-1] % factor:
                raise ValueError(f"frequency division at boundary {index} must be exact")
            frequencies.append(frequencies[-1] // factor)
        self.frequencies = tuple(frequencies)
        self.time_factors = tuple(
            self.temporal_factor if i == count - 2 else 1 for i in range(count - 1)
        )


class _PerTimeNorm(nn.GroupNorm):
    """Group normalization over channels/frequency, with time folded into batch."""

    def __init__(self, channels: int) -> None:
        groups = min(channels // 4, 32)
        if channels < 4 or channels % groups:
            raise ValueError(
                f"normalized width {channels} must be >= 4 and divisible by min(width // 4, 32)"
            )
        super().__init__(groups, channels, eps=1e-5, affine=True)

    def forward(self, value: Tensor) -> Tensor:
        batch, channels = value.shape[:2]
        time = value.shape[-1]
        # Both [B,C,T] and [B,C,F,T] become [B*T,C,...].
        separated = value.movedim(-1, 1).reshape(batch * time, channels, *value.shape[2:-1])
        normalized = super().forward(separated)
        return normalized.reshape(batch, time, channels, *value.shape[2:-1]).movedim(1, -1)


class _FrequencyAttention(nn.Module):
    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError(f"attention width {channels} must be divisible by m2l_heads={heads}")
        self.heads = heads
        self.normalization = _PerTimeNorm(channels)
        self.q = nn.Linear(channels, channels)
        self.k = nn.Linear(channels, channels)
        self.v = nn.Linear(channels, channels)
        self.projection = nn.Linear(channels, channels)
        bound = sqrt(1.5 / channels)
        for linear in (self.q, self.k, self.v):
            nn.init.uniform_(linear.weight, -bound, bound)
            nn.init.zeros_(linear.bias)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, value: Tensor) -> Tensor:
        batch, channels, frequencies, time = value.shape
        tokens = self.normalization(value).permute(0, 3, 2, 1)
        tokens = tokens.reshape(batch * time, frequencies, channels)
        head_width = channels // self.heads
        queries, keys, values = (
            linear(tokens)
            .reshape(batch * time, frequencies, self.heads, head_width)
            .transpose(1, 2)
            for linear in (self.q, self.k, self.v)
        )
        attended = F.scaled_dot_product_attention(queries, keys, values, dropout_p=0.0)
        attended = attended.transpose(1, 2).reshape(batch * time, frequencies, channels)
        attended = self.projection(attended).reshape(batch, time, frequencies, channels)
        return value + attended.permute(0, 3, 2, 1)


class _Residual(nn.Module):
    def __init__(
        self, incoming: int, outgoing: int, dimensions: int, heads: int | None = None
    ) -> None:
        super().__init__()
        convolution = nn.Conv2d if dimensions == 2 else nn.Conv1d
        self.norm_input = _PerTimeNorm(incoming)
        self.conv_input = convolution(incoming, outgoing, 3, padding=1)
        self.norm_output = _PerTimeNorm(outgoing)
        self.conv_output = convolution(outgoing, outgoing, 3, padding=1)
        self.shortcut = (
            nn.Identity() if incoming == outgoing else convolution(incoming, outgoing, 1)
        )
        self.frequency_attention = (
            nn.Identity() if heads is None else _FrequencyAttention(outgoing, heads)
        )
        nn.init.zeros_(self.conv_output.weight)
        nn.init.zeros_(self.conv_output.bias)

    def forward(self, value: Tensor) -> Tensor:
        main = self.conv_input(F.silu(self.norm_input(value)))
        main = self.conv_output(F.silu(self.norm_output(main)))
        return self.frequency_attention(self.shortcut(value) + main)


class _Resolution(nn.Module):
    def __init__(self, incoming: int, outgoing: int, depth: int, heads: int | None) -> None:
        super().__init__()
        self.residuals = nn.ModuleList(
            _Residual(incoming if index == 0 else outgoing, outgoing, 2, heads)
            for index in range(depth)
        )

    def forward(self, value: Tensor) -> Tensor:
        for residual in self.residuals:
            value = residual(value)
        return value


class _AnalysisTransition(nn.Module):
    def __init__(self, width: int, frequency_factor: int, time_factor: int) -> None:
        super().__init__()
        frequency_kernel = 5 if frequency_factor == 4 else 3
        self.normalization = _PerTimeNorm(width)
        self.filter = nn.Conv2d(
            width,
            width,
            (frequency_kernel, 3),
            stride=(frequency_factor, time_factor),
            padding=(frequency_kernel // 2, 1),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.filter(self.normalization(value))


class _SynthesisTransition(nn.Module):
    def __init__(
        self, incoming: int, outgoing: int, frequency_factor: int, time_factor: int
    ) -> None:
        super().__init__()
        frequency_kernel = 5 if frequency_factor == 4 else 3
        self.scale_factor = (frequency_factor, time_factor)
        self.filter = nn.Conv2d(
            incoming,
            outgoing,
            (frequency_kernel, 3),
            padding=(frequency_kernel // 2, 1),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.filter(F.interpolate(value, scale_factor=self.scale_factor, mode="nearest"))


class _Posterior(nn.Module):
    def __init__(
        self,
        features: int,
        hidden: int,
        latent: int,
        kernel: int,
        dilation: int,
        layers: int,
    ) -> None:
        super().__init__()
        self.input = nn.Conv1d(features, hidden, 1)
        self.network = WN(hidden, kernel, dilation, layers, gin_channels=0)
        self.output = nn.Conv1d(hidden, latent, 1)

    def forward(self, value: Tensor, lengths: Tensor) -> Tensor:
        mask = sequence_mask(lengths, value.shape[-1]).unsqueeze(1).to(dtype=value.dtype)
        hidden = self.input(value) * mask
        hidden = self.network(hidden, mask)
        projected = self.output(hidden) * mask
        return ScalarQuantize9.apply(torch.tanh(projected)) * mask


def _complex_magnitude_map(value: Tensor, inverse: bool) -> Tensor:
    magnitude = value.abs()
    nonzero = magnitude != 0
    # The forward power is singular at zero. Select a finite derivative there
    # without perturbing any nonzero magnitude or its phase.
    safe_magnitude = torch.where(nonzero, magnitude, torch.ones_like(magnitude))
    if inverse:
        mapped = (safe_magnitude / _MAGNITUDE_SCALE).pow(1 / _MAGNITUDE_EXPONENT)
    else:
        mapped = _MAGNITUDE_SCALE * safe_magnitude.pow(_MAGNITUDE_EXPONENT)
    mapped = torch.where(nonzero, mapped, torch.zeros_like(mapped))
    limits = torch.finfo(magnitude.dtype)
    extreme = nonzero & ((magnitude < sqrt(limits.tiny)) | (magnitude > sqrt(limits.max)))
    phase_scale = torch.where(extreme, safe_magnitude.detach(), torch.ones_like(magnitude))
    # Angle is invariant under positive scaling. A detached scale keeps the
    # same phase derivative while avoiding squared-norm under/overflow in
    # angle's backward operation for extreme, nonzero complex values.
    phase = torch.complex(value.real / phase_scale, value.imag / phase_scale).angle()
    return torch.polar(mapped, phase)


def _check_representation(value: Tensor, frequencies: int) -> None:
    if not isinstance(value, Tensor) or value.ndim != 4:
        raise ValueError("representation must have shape [batch, 2, frequency, time]")
    if value.is_complex() or value.shape[1] != 2 or value.shape[2] != frequencies:
        raise ValueError(
            f"representation must be real with 2 channels and {frequencies} frequency bins"
        )
    if value.shape[0] == 0 or value.shape[-1] == 0:
        raise ValueError("representation batch and time dimensions must be nonempty")


def _check_audio(audio: Tensor) -> None:
    if not isinstance(audio, Tensor) or audio.ndim not in (2, 3) or audio.is_complex():
        raise ValueError(
            "audio must be a real tensor of shape [batch, time] or [batch, channels, time]"
        )
    if any(size == 0 for size in audio.shape):
        raise ValueError("audio dimensions must be nonempty")


class STFT2DEncoder(nn.Module):
    """Convert a waveform into deterministic, masked scalar-quantized codes."""

    def __init__(
        self,
        latent_channels: int = 64,
        hidden_channels: int = 192,
        encoder_kernel_size: int = 5,
        encoder_dilation_rate: int = 1,
        encoder_layers: int = 16,
        stft_hop_length: int = 640,
        stft_win_length: int = 2560,
        m2l_base_channels: int = 64,
        m2l_layers: Sequence[int] = (1, 1, 1, 1, 1),
        m2l_multipliers: Sequence[int] = (1, 2, 4, 4, 4),
        m2l_attention: Sequence[int] = (0, 0, 1, 1, 1),
        m2l_freq_downsample_factors: Sequence[int] = (4, 2, 2, 2),
        m2l_heads: int = 4,
        m2l_last_time_downsample_factor: int = 1,
    ) -> None:
        super().__init__()
        geometry = _Geometry(
            stft_hop_length,
            stft_win_length,
            m2l_base_channels,
            m2l_layers,
            m2l_multipliers,
            m2l_attention,
            m2l_freq_downsample_factors,
            m2l_heads,
            m2l_last_time_downsample_factor,
        )
        latent_channels = _integer("latent_channels", latent_channels)
        hidden_channels = _integer("hidden_channels", hidden_channels)
        encoder_kernel_size = _integer("encoder_kernel_size", encoder_kernel_size)
        encoder_dilation_rate = _integer("encoder_dilation_rate", encoder_dilation_rate)
        encoder_layers = _integer("encoder_layers", encoder_layers, 0)
        if encoder_kernel_size % 2 != 1:
            raise ValueError("encoder_kernel_size must be odd")
        self.stft_hop_length = geometry.hop
        self.stft_win_length = geometry.window
        self.freq_bins = geometry.frequencies[0]
        self.time_downsampling_ratio = geometry.temporal_factor
        self.register_buffer(
            "window",
            torch.hann_window(geometry.window, dtype=torch.float32),
            persistent=False,
        )
        current_width = geometry.widths[0]
        self.spectral_input = nn.Conv2d(2, current_width, 3, padding=1)
        self.frequency_scale = nn.Parameter(torch.ones(self.freq_bins))
        self.resolutions = nn.ModuleList()
        self.transitions = nn.ModuleList()
        for index, (width, depth, attended) in enumerate(
            zip(geometry.widths, geometry.depths, geometry.attention)
        ):
            self.resolutions.append(
                _Resolution(current_width, width, depth, geometry.heads if attended else None)
            )
            if depth:
                current_width = width
            if index < len(geometry.factors):
                self.transitions.append(
                    _AnalysisTransition(
                        current_width,
                        geometry.factors[index],
                        geometry.time_factors[index],
                    )
                )
        self.final_normalization = _PerTimeNorm(current_width)
        self.posterior = _Posterior(
            current_width * geometry.frequencies[-1],
            hidden_channels,
            latent_channels,
            encoder_kernel_size,
            encoder_dilation_rate,
            encoder_layers,
        )

    def waveform_to_representation(self, audio: Tensor) -> Tensor:
        _check_audio(audio)
        hop = self.stft_hop_length
        padding = (self.stft_win_length - hop) // 2
        padded_length = ((audio.shape[-1] + hop - 1) // hop) * hop
        if padded_length <= padding:
            raise ValueError(
                f"hop-padded audio length {padded_length} must exceed reflection padding {padding}"
            )
        with torch.autocast(device_type=audio.device.type, enabled=False):
            waveform = audio.float()
            if waveform.ndim == 3:
                waveform = waveform.mean(dim=1)
            waveform = F.pad(waveform, (0, padded_length - waveform.shape[-1]))
            waveform = F.pad(waveform, (padding, padding), mode="reflect")
            spectrum = torch.stft(
                waveform,
                n_fft=self.stft_win_length,
                hop_length=hop,
                win_length=self.stft_win_length,
                window=self.window.float(),
                center=False,
                normalized=False,
                onesided=True,
                return_complex=True,
            )
            compressed = _complex_magnitude_map(spectrum[:, : self.freq_bins], inverse=False)
            return torch.stack((compressed.real, compressed.imag), dim=1)

    def spectral_features(self, representation: Tensor) -> Tensor:
        _check_representation(representation, self.freq_bins)
        value = self.spectral_input(representation.to(dtype=self.spectral_input.weight.dtype))
        value = value * self.frequency_scale[None, None, :, None]
        for index, resolution in enumerate(self.resolutions):
            value = resolution(value)
            if index < len(self.transitions):
                value = self.transitions[index](value)
        return self.final_normalization(value).flatten(1, 2)

    def forward(self, audio: Tensor, audio_lengths: Tensor | None = None) -> Tensor:
        _check_audio(audio)
        batch, sample_count = audio.shape[0], audio.shape[-1]
        hop = self.stft_hop_length
        if audio_lengths is None:
            lengths = torch.full(
                (batch,),
                (sample_count + hop - 1) // hop,
                device=audio.device,
                dtype=torch.long,
            )
        else:
            if not isinstance(audio_lengths, Tensor) or audio_lengths.shape != (batch,):
                raise ValueError("audio_lengths must be a tensor of shape [batch]")
            if audio_lengths.is_complex() or audio_lengths.dtype == torch.bool:
                raise ValueError("audio_lengths must contain nonnegative integer sample counts")
            if not torch.isfinite(audio_lengths).all() or (audio_lengths < 0).any():
                raise ValueError("audio_lengths must contain finite nonnegative sample counts")
            if audio_lengths.is_floating_point() and (audio_lengths != audio_lengths.round()).any():
                raise ValueError("audio_lengths must contain integer sample counts")
            lengths = audio_lengths.to(device=audio.device, dtype=torch.long)
            lengths = (lengths + hop - 1) // hop
        lengths = (lengths + self.time_downsampling_ratio - 1) // self.time_downsampling_ratio
        representation = self.waveform_to_representation(audio)
        return self.posterior(self.spectral_features(representation), lengths)


class STFT2DDecoder(nn.Module):
    """Synthesize a waveform by inverting the compressed complex representation."""

    def __init__(
        self,
        latent_channels: int = 64,
        stft_hop_length: int = 640,
        stft_win_length: int = 2560,
        m2l_base_channels: int = 64,
        m2l_layers: Sequence[int] = (1, 1, 1, 1, 1),
        m2l_multipliers: Sequence[int] = (1, 2, 4, 4, 4),
        m2l_attention: Sequence[int] = (0, 0, 1, 1, 1),
        m2l_freq_downsample_factors: Sequence[int] = (4, 2, 2, 2),
        m2l_bottleneck_base_channels: int = 512,
        m2l_num_bottleneck_layers: int = 4,
        m2l_heads: int = 4,
        m2l_last_time_downsample_factor: int = 1,
    ) -> None:
        super().__init__()
        geometry = _Geometry(
            stft_hop_length,
            stft_win_length,
            m2l_base_channels,
            m2l_layers,
            m2l_multipliers,
            m2l_attention,
            m2l_freq_downsample_factors,
            m2l_heads,
            m2l_last_time_downsample_factor,
        )
        latent_channels = _integer("latent_channels", latent_channels)
        bottleneck = _integer("m2l_bottleneck_base_channels", m2l_bottleneck_base_channels)
        bottleneck_layers = _integer("m2l_num_bottleneck_layers", m2l_num_bottleneck_layers, 0)
        self.stft_hop_length = geometry.hop
        self.stft_win_length = geometry.window
        self.freq_bins = geometry.frequencies[0]
        self._deepest_width = geometry.widths[-1]
        self._deepest_frequencies = geometry.frequencies[-1]
        self.register_buffer(
            "window",
            torch.hann_window(geometry.window, dtype=torch.float32),
            persistent=False,
        )
        self.latent_input = nn.Conv1d(latent_channels, bottleneck, 1)
        self.temporal_residuals = nn.ModuleList(
            _Residual(bottleneck, bottleneck, 1) for _ in range(bottleneck_layers)
        )
        self.spectral_expansion = nn.Conv1d(
            bottleneck, self._deepest_width * self._deepest_frequencies, 1
        )
        self.resolutions = nn.ModuleList(
            _Resolution(width, width, depth, geometry.heads if attended else None)
            for width, depth, attended in zip(geometry.widths, geometry.depths, geometry.attention)
        )
        self.transitions = nn.ModuleList(
            _SynthesisTransition(
                geometry.widths[index + 1],
                geometry.widths[index],
                factor,
                geometry.time_factors[index],
            )
            for index, factor in enumerate(geometry.factors)
        )
        self.final_normalization = _PerTimeNorm(geometry.widths[0])
        self.spectral_output = nn.Conv2d(geometry.widths[0], 2, 3, padding=1)

    def latent_to_representation(self, latent: Tensor) -> Tensor:
        if not isinstance(latent, Tensor) or latent.ndim != 3 or latent.is_complex():
            raise ValueError("latent must be a real tensor of shape [batch, latent_channels, time]")
        if (
            latent.shape[1] != self.latent_input.in_channels
            or latent.shape[0] == 0
            or latent.shape[-1] == 0
        ):
            raise ValueError(
                f"latent requires {self.latent_input.in_channels} channels and nonempty batch/time dimensions"
            )
        value = self.latent_input(latent.to(dtype=self.latent_input.weight.dtype))
        for residual in self.temporal_residuals:
            value = residual(value)
        value = self.spectral_expansion(value)
        value = value.reshape(
            value.shape[0],
            self._deepest_width,
            self._deepest_frequencies,
            value.shape[-1],
        )
        for index in range(len(self.resolutions) - 1, -1, -1):
            value = self.resolutions[index](value)
            if index:
                value = self.transitions[index - 1](value)
        return self.spectral_output(F.silu(self.final_normalization(value)))

    def representation_to_waveform(
        self, representation: Tensor, length: int | None = None
    ) -> Tensor:
        _check_representation(representation, self.freq_bins)
        if length is not None:
            length = _integer("length", length, 0)
        with torch.autocast(device_type=representation.device.type, enabled=False):
            values = representation.float()
            compressed = torch.complex(values[:, 0], values[:, 1])
            compressed = F.pad(compressed, (0, 0, 0, 1))
            spectrum = _complex_magnitude_map(compressed, inverse=True)
            frames = torch.fft.irfft(spectrum, n=self.stft_win_length, dim=1)
            window = self.window.float()
            frames = frames * window[None, :, None]
            total = (frames.shape[-1] - 1) * self.stft_hop_length + self.stft_win_length
            fold_options = dict(
                output_size=(1, total),
                kernel_size=(1, self.stft_win_length),
                stride=(1, self.stft_hop_length),
            )
            summed = F.fold(frames, **fold_options)[:, 0, 0]
            weights = window.square()[None, :, None].expand(1, -1, frames.shape[-1])
            envelope = F.fold(weights, **fold_options)[0, 0, 0].clamp_min(1e-8)
            waveform = summed / envelope
            padding = (self.stft_win_length - self.stft_hop_length) // 2
            waveform = waveform[:, padding : total - padding]
            if length is not None:
                waveform = waveform[:, :length]
            return waveform.clamp(-1, 1).unsqueeze(1)

    def forward(self, latent: Tensor, length: int | None = None) -> Tensor:
        if length is not None:
            length = _integer("length", length, 0)
        return self.representation_to_waveform(self.latent_to_representation(latent), length)
