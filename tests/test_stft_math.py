"""Mathematical, gradient and interface tests for the STFT codec."""

import io
import json
import math
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from pl_flow.vocoder import stft as codec

FIXTURES = Path(__file__).parent / "fixtures"


def configuration(**changes):
    options = dict(
        latent_channels=3,
        stft_hop_length=8,
        stft_win_length=32,
        m2l_base_channels=8,
        m2l_layers=(1, 1),
        m2l_multipliers=(1, 2),
        m2l_attention=(0, 1),
        m2l_freq_downsample_factors=(2,),
        m2l_heads=2,
        m2l_last_time_downsample_factor=2,
    )
    options.update(changes)
    return options


def encoder(**changes):
    options = dict(hidden_channels=8, encoder_kernel_size=3, encoder_layers=2)
    options.update(configuration(**changes))
    return codec.STFT2DEncoder(**options)


def decoder(**changes):
    options = dict(m2l_bottleneck_base_channels=8, m2l_num_bottleneck_layers=2)
    options.update(configuration(**changes))
    return codec.STFT2DDecoder(**options)


def reference_norm(value, layer):
    batch, channels = value.shape[:2]
    groups = min(channels // 4, 32)
    results = []
    for time in range(value.shape[-1]):
        item = value[..., time].reshape(batch, groups, -1)
        mean = item.mean(dim=-1, keepdim=True)
        variance = ((item - mean) ** 2).mean(dim=-1, keepdim=True)
        item = ((item - mean) / torch.sqrt(variance + 1e-5)).reshape(value[..., time].shape)
        affine_shape = (1, channels) + (1,) * (item.ndim - 2)
        results.append(item * layer.weight.reshape(affine_shape) + layer.bias.reshape(affine_shape))
    return torch.stack(results, dim=-1)


def reference_attention(value, layer):
    normalized = reference_norm(value, layer.normalization)
    outputs = []
    width = value.shape[1]
    head_width = width // layer.heads
    for time in range(value.shape[-1]):
        tokens = normalized[..., time].transpose(1, 2)
        projected = [
            F.linear(tokens, linear.weight, linear.bias) for linear in (layer.q, layer.k, layer.v)
        ]
        heads = []
        for head in range(layer.heads):
            start = head * head_width
            query, key, val = [item[..., start : start + head_width] for item in projected]
            scores = (query @ key.transpose(-2, -1)) / math.sqrt(head_width)
            heads.append(scores.softmax(dim=-1) @ val)
        mixed = F.linear(torch.cat(heads, dim=-1), layer.projection.weight, layer.projection.bias)
        outputs.append(mixed.transpose(1, 2))
    return value + torch.stack(outputs, dim=-1)


def reference_convolution(value, layer):
    operation = F.conv2d if value.ndim == 4 else F.conv1d
    return operation(value, layer.weight, layer.bias, stride=layer.stride, padding=layer.padding)


def reference_residual(value, layer):
    main = reference_convolution(F.silu(reference_norm(value, layer.norm_input)), layer.conv_input)
    main = reference_convolution(F.silu(reference_norm(main, layer.norm_output)), layer.conv_output)
    shortcut = (
        value
        if isinstance(layer.shortcut, nn.Identity)
        else reference_convolution(value, layer.shortcut)
    )
    result = main + shortcut
    if not isinstance(layer.frequency_attention, nn.Identity):
        result = reference_attention(result, layer.frequency_attention)
    return result


def reference_analysis(value, module):
    value = reference_convolution(value, module.spectral_input)
    value = value * module.frequency_scale[None, None, :, None]
    for index, resolution in enumerate(module.resolutions):
        for residual in resolution.residuals:
            value = reference_residual(value, residual)
        if index < len(module.transitions):
            transition = module.transitions[index]
            value = reference_convolution(
                reference_norm(value, transition.normalization), transition.filter
            )
    value = reference_norm(value, module.final_normalization)
    return torch.cat([value[:, channel] for channel in range(value.shape[1])], dim=1)


def reference_synthesis(value, module):
    value = reference_convolution(value, module.latent_input)
    for residual in module.temporal_residuals:
        value = reference_residual(value, residual)
    value = reference_convolution(value, module.spectral_expansion)
    value = torch.stack(value.split(module._deepest_frequencies, dim=1), dim=1)
    for index in range(len(module.resolutions) - 1, -1, -1):
        for residual in module.resolutions[index].residuals:
            value = reference_residual(value, residual)
        if index > 0:
            transition = module.transitions[index - 1]
            frequency, time = transition.scale_factor
            value = value.repeat_interleave(frequency, dim=2).repeat_interleave(time, dim=3)
            value = reference_convolution(value, transition.filter)
    return reference_convolution(
        F.silu(reference_norm(value, module.final_normalization)),
        module.spectral_output,
    )


def activate_residuals(module):
    """Exercise every trained branch rather than only its zero initialization."""
    with torch.no_grad():
        for layer in module.modules():
            if isinstance(layer, codec._Residual):
                layer.conv_output.weight.normal_(0, 0.04)
                layer.conv_output.bias.normal_(0, 0.02)
            if isinstance(layer, codec._FrequencyAttention):
                layer.projection.weight.normal_(0, 0.05)
                layer.projection.bias.normal_(0, 0.02)
            if isinstance(layer, codec._PerTimeNorm):
                layer.weight.uniform_(0.75, 1.25)
                layer.bias.uniform_(-0.2, 0.2)


def reference_representation(audio, hop):
    audio = audio.float()
    if audio.ndim == 3:
        audio = audio.mean(dim=1)
    padded_samples = hop * math.ceil(audio.shape[-1] / hop)
    audio = torch.cat(
        (audio, audio.new_zeros(audio.shape[0], padded_samples - audio.shape[-1])),
        dim=-1,
    )
    window_size = 4 * hop
    padding = (window_size - hop) // 2
    audio = torch.cat(
        (
            audio[:, 1 : padding + 1].flip(-1),
            audio,
            audio[:, -padding - 1 : -1].flip(-1),
        ),
        dim=-1,
    )
    window = torch.hann_window(window_size, dtype=torch.float32, device=audio.device)
    spectra = [
        torch.fft.rfft(audio[:, offset : offset + window_size] * window)
        for offset in range(0, audio.shape[-1] - window_size + 1, hop)
    ]
    spectrum = torch.stack(spectra, dim=-1)[:, : 2 * hop]
    amplitude = 0.34 * spectrum.abs().pow(0.65)
    return torch.stack(
        (amplitude * spectrum.angle().cos(), amplitude * spectrum.angle().sin()), dim=1
    )


def reference_waveform(representation, hop, length=None):
    representation = representation.float()
    spectrum = torch.complex(representation[:, 0], representation[:, 1])
    spectrum = torch.cat(
        (spectrum, spectrum.new_zeros(spectrum.shape[0], 1, spectrum.shape[-1])), dim=1
    )
    amplitude = (spectrum.abs() / 0.34).pow(1 / 0.65)
    spectrum = torch.complex(amplitude * spectrum.angle().cos(), amplitude * spectrum.angle().sin())
    window_size = 4 * hop
    frame_count = spectrum.shape[-1]
    total = (frame_count - 1) * hop + window_size
    window = torch.hann_window(window_size, dtype=torch.float32, device=spectrum.device)
    summed = representation.new_zeros(representation.shape[0], total)
    envelope = representation.new_zeros(total)
    for frame in range(frame_count):
        offset = frame * hop
        waveform = torch.fft.irfft(spectrum[..., frame], n=window_size) * window
        summed = summed + F.pad(waveform, (offset, total - offset - window_size))
        envelope = envelope + F.pad(window.square(), (offset, total - offset - window_size))
    padding = (window_size - hop) // 2
    reconstructed = (summed / envelope.clamp_min(1e-8))[:, padding : total - padding]
    return reconstructed[:, :length].clamp(-1, 1).unsqueeze(1)


class SpecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(1947)

    def assert_close(self, actual, expected, atol=1e-6, rtol=1e-5):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

    def test_normalization_matches_population_variance(self):
        for shape in ((2, 8, 5), (2, 20, 3, 5), (1, 128, 1, 2), (1, 256, 3)):
            with self.subTest(shape=shape):
                layer = codec._PerTimeNorm(shape[1]).double()
                with torch.no_grad():
                    layer.weight.uniform_(0.5, 1.5)
                    layer.bias.normal_()
                value = torch.randn(shape, dtype=torch.float64, requires_grad=True)
                actual = layer(value)
                expected = reference_norm(value, layer)
                self.assert_close(actual, expected, atol=2e-14, rtol=2e-14)
                weights = torch.randn_like(actual)
                parameters = [value, layer.weight, layer.bias]
                got_grads = torch.autograd.grad(
                    (actual * weights).sum(), parameters, retain_graph=True
                )
                ref_grads = torch.autograd.grad((expected * weights).sum(), parameters)
                for got, wanted in zip(got_grads, ref_grads):
                    self.assert_close(got, wanted, atol=5e-13, rtol=5e-13)

    def test_normalization_and_attention_do_not_mix_time(self):
        layer = codec._FrequencyAttention(8, 2).double()
        activate_residuals(layer)
        value = torch.randn(2, 8, 5, 4, dtype=torch.float64)
        changed = value.clone()
        changed[..., -1] += torch.randn_like(changed[..., -1]) * 20
        self.assert_close(layer(value)[..., :-1], layer(changed)[..., :-1], atol=0, rtol=0)
        self.assert_close(
            layer.normalization(value)[..., :-1],
            layer.normalization(changed)[..., :-1],
            atol=0,
            rtol=0,
        )

    def test_attention_math_and_all_gradients(self):
        layer = codec._FrequencyAttention(8, 2).double()
        activate_residuals(layer)
        value = torch.randn(2, 8, 5, 3, dtype=torch.float64, requires_grad=True)
        actual = layer(value)
        expected = reference_attention(value, layer)
        self.assert_close(actual, expected, atol=3e-14, rtol=3e-14)
        weights = torch.randn_like(actual)
        parameters = [value, *layer.parameters()]
        actual_grads = torch.autograd.grad((actual * weights).sum(), parameters, retain_graph=True)
        expected_grads = torch.autograd.grad((expected * weights).sum(), parameters)
        for got, wanted in zip(actual_grads, expected_grads):
            self.assert_close(got, wanted, atol=8e-13, rtol=8e-13)

    def test_residual_math_for_all_shortcuts(self):
        for dimensions in (1, 2):
            for outgoing in (8, 16):
                with self.subTest(dimensions=dimensions, outgoing=outgoing):
                    layer = codec._Residual(8, outgoing, dimensions).double()
                    activate_residuals(layer)
                    shape = (2, 8, 7) if dimensions == 1 else (2, 8, 3, 7)
                    value = torch.randn(shape, dtype=torch.float64)
                    self.assert_close(
                        layer(value),
                        reference_residual(value, layer),
                        atol=1e-13,
                        rtol=1e-13,
                    )

    def test_full_analysis_and_synthesis_reference(self):
        for depths, multipliers in (((1, 0, 2), (1, 3, 2)), ((0, 2, 0), (2, 1, 3))):
            with self.subTest(depths=depths):
                options = dict(
                    m2l_layers=depths,
                    m2l_multipliers=multipliers,
                    m2l_attention=(1, 1, 1),
                    m2l_freq_downsample_factors=(2, 2),
                    m2l_last_time_downsample_factor=3,
                )
                analysis = encoder(**options).double()
                synthesis = decoder(**options).double()
                activate_residuals(analysis)
                activate_residuals(synthesis)
                representation = torch.randn(2, 2, 16, 7, dtype=torch.float64, requires_grad=True)
                actual_features = analysis.spectral_features(representation)
                expected_features = reference_analysis(representation, analysis)
                self.assert_close(actual_features, expected_features, atol=5e-12, rtol=5e-12)
                weight = torch.randn_like(actual_features)
                actual_gradient = torch.autograd.grad(
                    (weight * actual_features).sum(), representation, retain_graph=True
                )[0]
                expected_gradient = torch.autograd.grad(
                    (weight * expected_features).sum(), representation
                )[0]
                self.assert_close(actual_gradient, expected_gradient, atol=2e-11, rtol=2e-11)
                latent = torch.randn(2, 3, 3, dtype=torch.float64, requires_grad=True)
                actual_spectrum = synthesis.latent_to_representation(latent)
                expected_spectrum = reference_synthesis(latent, synthesis)
                self.assert_close(actual_spectrum, expected_spectrum, atol=4e-12, rtol=4e-12)
                weight = torch.randn_like(actual_spectrum)
                actual_gradient = torch.autograd.grad(
                    (weight * actual_spectrum).sum(), latent, retain_graph=True
                )[0]
                expected_gradient = torch.autograd.grad((weight * expected_spectrum).sum(), latent)[
                    0
                ]
                self.assert_close(actual_gradient, expected_gradient, atol=2e-11, rtol=2e-11)

    def test_frequency_four_transition_kernel_and_stride(self):
        options = dict(m2l_freq_downsample_factors=(4,))
        analysis, synthesis = encoder(**options), decoder(**options)
        self.assertEqual(analysis.transitions[0].filter.kernel_size, (5, 3))
        self.assertEqual(analysis.transitions[0].filter.stride, (4, 2))
        self.assertEqual(synthesis.transitions[0].filter.kernel_size, (5, 3))
        self.assertEqual(synthesis.transitions[0].filter.stride, (1, 1))
        self.assertEqual(analysis.spectral_features(torch.randn(2, 2, 16, 5)).shape, (2, 64, 3))
        self.assertEqual(
            synthesis.latent_to_representation(torch.randn(2, 3, 3)).shape,
            (2, 2, 16, 6),
        )

    def test_waveform_analysis_matches_explicit_frames(self):
        for hop in (1, 3, 5, 8):
            options = dict(
                stft_hop_length=hop,
                stft_win_length=4 * hop,
                m2l_layers=(0,),
                m2l_multipliers=(1,),
                m2l_attention=(0,),
                m2l_freq_downsample_factors=(),
                m2l_last_time_downsample_factor=1,
            )
            model = encoder(**options)
            for samples in (2 * hop, 2 * hop + 1, 6 * hop - 1):
                with self.subTest(hop=hop, samples=samples):
                    audio = torch.randn(2, 2, samples) * 0.2
                    actual = model.waveform_to_representation(audio)
                    expected = reference_representation(audio, hop)
                    self.assert_close(actual, expected, atol=1e-6, rtol=2e-6)
                    padded_samples = hop * math.ceil(samples / hop)
                    padding = (4 * hop - hop) // 2
                    count = (padded_samples + 2 * padding - 4 * hop) // hop + 1
                    self.assertEqual(actual.shape, (2, 2, 2 * hop, count))
                    self.assertEqual(actual.dtype, torch.float32)

    def test_synthesis_matches_explicit_overlap_add(self):
        for hop in (1, 3, 5, 8):
            model = decoder(
                stft_hop_length=hop,
                stft_win_length=4 * hop,
                m2l_layers=(0,),
                m2l_multipliers=(1,),
                m2l_attention=(0,),
                m2l_freq_downsample_factors=(),
                m2l_last_time_downsample_factor=1,
            )
            for frames in (1, 2, 5):
                with self.subTest(hop=hop, frames=frames):
                    representation = torch.randn(2, 2, 2 * hop, frames) * 0.15
                    actual = model.representation_to_waveform(representation)
                    expected = reference_waveform(representation, hop)
                    self.assert_close(actual, expected, atol=1e-6, rtol=3e-6)
                    padding = (3 * hop) // 2
                    natural = (frames - 1) * hop + 4 * hop - 2 * padding
                    self.assertEqual(actual.shape, (2, 1, natural))
                    for requested in (0, 1, natural - 1, natural, natural + 20):
                        shortened = model.representation_to_waveform(representation, requested)
                        self.assert_close(shortened, actual[..., :requested], atol=0, rtol=0)

    def test_inverse_nyquist_restoration_and_clamping(self):
        model = decoder()
        representation = torch.randn(2, 2, 16, 5) * 20
        output = model.representation_to_waveform(representation)
        expected = reference_waveform(representation, 8)
        # At this stress amplitude, cancellation magnifies the FP32 difference
        # between separate cos/sin and the polar kernel (observed: 3.19e-6).
        self.assert_close(output, expected, atol=8e-6, rtol=2e-6)
        self.assertLessEqual(output.abs().max().item(), 1)
        self.assertTrue((output.abs() == 1).any())

    def test_compression_exact_nonzero_values_and_zero_gradients(self):
        values = torch.tensor(
            [0j, 1 + 0j, -1 + 0j, 0 + 1j, 3 - 4j, 1e-20 + 0j],
            dtype=torch.complex64,
            requires_grad=True,
        )
        result = codec._complex_magnitude_map(values, inverse=False)
        nonzero = values[1:]
        amplitude = 0.34 * nonzero.abs().pow(0.65)
        expected = torch.complex(
            amplitude * nonzero.angle().cos(), amplitude * nonzero.angle().sin()
        )
        self.assert_close(result[1:], expected, atol=0, rtol=1e-7)
        self.assertEqual(result[0].item(), 0j)
        restored = codec._complex_magnitude_map(result, inverse=True)
        self.assert_close(restored, values, atol=2e-6, rtol=2e-6)
        result.real.sum().backward()
        self.assertTrue(values.grad.isfinite().all())
        self.assertEqual(values.grad[0].item(), 0j)
        zero = torch.zeros(2, 2, 16, 3, requires_grad=True)
        waveform = decoder().representation_to_waveform(zero)
        waveform.sum().backward()
        self.assertEqual(waveform.count_nonzero().item(), 0)
        self.assertEqual(zero.grad.count_nonzero().item(), 0)

    def test_silent_audio_has_finite_zero_transform_gradient(self):
        audio = torch.zeros(2, 37, requires_grad=True)
        representation = encoder().waveform_to_representation(audio)
        representation.sum().backward()
        self.assertEqual(representation.count_nonzero().item(), 0)
        self.assertTrue(audio.grad.isfinite().all())
        self.assertEqual(audio.grad.count_nonzero().item(), 0)

    def test_waveform_and_synthesis_input_gradients(self):
        audio = (torch.randn(2, 37) * 0.2).requires_grad_()
        actual = encoder().waveform_to_representation(audio)
        expected = reference_representation(audio, 8)
        weights = torch.randn_like(actual)
        got = torch.autograd.grad((actual * weights).sum(), audio, retain_graph=True)[0]
        wanted = torch.autograd.grad((expected * weights).sum(), audio)[0]
        self.assert_close(got, wanted, atol=3e-5, rtol=3e-5)
        representation = (torch.randn(2, 2, 16, 5) * 0.1).requires_grad_()
        actual = decoder().representation_to_waveform(representation)
        expected = reference_waveform(representation, 8)
        weights = torch.randn_like(actual)
        got = torch.autograd.grad((actual * weights).sum(), representation, retain_graph=True)[0]
        wanted = torch.autograd.grad((expected * weights).sum(), representation)[0]
        self.assert_close(got, wanted, atol=2e-6, rtol=2e-5)

    def test_mask_ceilings_without_spectral_masking(self):
        audio = torch.randn(7, 57)
        sample_lengths = torch.tensor([0, 1, 8, 9, 16, 17, 57])
        for factor in (1, 2, 3):
            with self.subTest(factor=factor):
                model = encoder(m2l_last_time_downsample_factor=factor)
                with torch.no_grad():
                    model.posterior.output.weight.zero_()
                    model.posterior.output.bias.fill_(1)
                recorded = []
                handle = model.posterior.network.register_forward_pre_hook(
                    lambda module, args: recorded.append(args)
                )
                result = model(audio, sample_lengths)
                handle.remove()
                valid = torch.tensor(
                    [math.ceil(math.ceil(int(length) / 8) / factor) for length in sample_lengths]
                )
                mask = (
                    (torch.arange(result.shape[-1])[None, :] < valid[:, None]).unsqueeze(1).float()
                )
                self.assert_close(result, (mask * (7 / 9)).expand_as(result), atol=0, rtol=0)
                full_features = model.spectral_features(model.waveform_to_representation(audio))
                projected = model.posterior.input(full_features) * mask
                self.assert_close(recorded[0][0], projected, atol=0, rtol=0)
                self.assert_close(recorded[0][1], mask, atol=0, rtol=0)
                self.assert_close(model(audio), torch.full_like(result, 7 / 9), atol=0, rtol=0)

    def test_posterior_formula_and_straight_through_gradients(self):
        module = codec._Posterior(12, 8, 3, 3, 2, 3).double()
        features = torch.randn(2, 12, 5, dtype=torch.float64, requires_grad=True)
        lengths = torch.tensor([2, 5])
        actual = module(features, lengths)
        mask = torch.tensor([[[1, 1, 0, 0, 0]], [[1, 1, 1, 1, 1]]], dtype=torch.float64)
        inputs = F.conv1d(features, module.input.weight, module.input.bias) * mask
        hidden = module.network(inputs, mask)
        continuous = (F.conv1d(hidden, module.output.weight, module.output.bias) * mask).tanh()
        discrete = torch.round(9 * continuous) / 9
        expected = (continuous + (discrete - continuous).detach()) * mask
        self.assert_close(actual, discrete * mask, atol=0, rtol=0)
        weights = torch.randn_like(actual)
        parameters = [features, *module.parameters()]
        got = torch.autograd.grad((actual * weights).sum(), parameters, retain_graph=True)
        wanted = torch.autograd.grad((expected * weights).sum(), parameters)
        for got_gradient, wanted_gradient in zip(got, wanted):
            self.assert_close(got_gradient, wanted_gradient, atol=1e-12, rtol=1e-12)
        ties = (torch.arange(-8, 9, dtype=torch.float64) + 0.5) / 9
        self.assert_close(
            codec.ScalarQuantize9.apply(ties), torch.round(ties * 9) / 9, atol=0, rtol=0
        )

    def test_zero_depth_and_nonuniform_shapes(self):
        for depths, multipliers in (
            ((0, 0, 0), (1, 3, 2)),
            ((1, 0, 2), (2, 1, 3)),
            ((0, 2, 0), (2, 1, 3)),
        ):
            for factor in (1, 2, 3):
                with self.subTest(depths=depths, factor=factor):
                    options = dict(
                        m2l_layers=depths,
                        m2l_multipliers=multipliers,
                        m2l_attention=(0, 1, 1),
                        m2l_freq_downsample_factors=(2, 2),
                        m2l_last_time_downsample_factor=factor,
                    )
                    analysis, synthesis = encoder(**options), decoder(**options)
                    current_width = 8 * multipliers[0]
                    for index, depth in enumerate(depths):
                        self.assertEqual(len(analysis.resolutions[index].residuals), depth)
                        self.assertEqual(len(synthesis.resolutions[index].residuals), depth)
                        if depth:
                            current_width = 8 * multipliers[index]
                    self.assertEqual(analysis.posterior.input.in_channels, current_width * 4)
                    self.assertEqual(
                        synthesis.spectral_expansion.out_channels,
                        8 * multipliers[-1] * 4,
                    )
                    audio = torch.randn(2, 49)
                    result = analysis(audio)
                    self.assertEqual(result.shape, (2, 3, math.ceil(7 / factor)))
                    representation = synthesis.latent_to_representation(result)
                    self.assertEqual(representation.shape, (2, 2, 16, result.shape[-1] * factor))

    def test_single_resolution_and_empty_temporal_stack(self):
        options = dict(
            m2l_layers=(0,),
            m2l_multipliers=(1,),
            m2l_attention=(1,),
            m2l_freq_downsample_factors=(),
            m2l_last_time_downsample_factor=1,
        )
        analysis = encoder(**options)
        synthesis = decoder(**options, m2l_num_bottleneck_layers=0, m2l_bottleneck_base_channels=2)
        self.assertEqual(len(analysis.transitions), 0)
        self.assertEqual(len(synthesis.transitions), 0)
        self.assertEqual(len(synthesis.temporal_residuals), 0)
        self.assertEqual(analysis(torch.randn(2, 49)).shape, (2, 3, 7))
        self.assertEqual(synthesis(torch.randn(2, 3, 7)).shape, (2, 1, 56))

    def test_stereo_noncontiguous_and_double_module(self):
        analysis, synthesis = encoder(), decoder()
        stereo = torch.randn(2, 61, 3).transpose(1, 2)
        self.assertFalse(stereo.is_contiguous())
        self.assert_close(analysis(stereo), analysis(stereo.mean(dim=1)), atol=0, rtol=0)
        audio = torch.randn(2, 114)[:, ::2]
        self.assertFalse(audio.is_contiguous())
        self.assert_close(analysis(audio), analysis(audio.contiguous()), atol=0, rtol=0)
        latent = torch.randn(2, 3, 10)[:, :, ::2]
        self.assertFalse(latent.is_contiguous())
        self.assert_close(synthesis(latent), synthesis(latent.contiguous()), atol=0, rtol=0)
        representation = torch.randn(2, 2, 16, 10)[..., ::2]
        self.assert_close(
            synthesis.representation_to_waveform(representation),
            synthesis.representation_to_waveform(representation.contiguous()),
            atol=0,
            rtol=0,
        )
        analysis, synthesis = analysis.double(), synthesis.double()
        self.assertEqual(analysis.window.dtype, torch.float64)
        self.assertEqual(analysis.waveform_to_representation(audio.double()).dtype, torch.float32)
        self.assertEqual(analysis(audio.double()).dtype, torch.float64)
        self.assertEqual(synthesis(latent.double()).dtype, torch.float32)

    def test_initialization_invariants(self):
        for model in (encoder(), decoder()):
            self.assertEqual(model.window.dtype, torch.float32)
            self.assert_close(model.window, torch.hann_window(32), atol=0, rtol=0)
            self.assertNotIn("window", model.state_dict())
            for layer in model.modules():
                if isinstance(layer, codec._PerTimeNorm):
                    self.assertEqual(layer.weight.count_nonzero().item(), layer.weight.numel())
                    self.assert_close(layer.weight, torch.ones_like(layer.weight), atol=0, rtol=0)
                    self.assertEqual(layer.bias.count_nonzero().item(), 0)
                if isinstance(layer, codec._Residual):
                    self.assertEqual(layer.conv_output.weight.count_nonzero().item(), 0)
                    self.assertEqual(layer.conv_output.bias.count_nonzero().item(), 0)
                    shape = (
                        (2, layer.norm_input.num_channels, 5)
                        if layer.conv_input.weight.ndim == 3
                        else (2, layer.norm_input.num_channels, 3, 5)
                    )
                    value = torch.randn(shape)
                    self.assert_close(layer(value), layer.shortcut(value), atol=0, rtol=0)
                if isinstance(layer, codec._FrequencyAttention):
                    self.assertEqual(layer.projection.weight.count_nonzero().item(), 0)
                    self.assertEqual(layer.projection.bias.count_nonzero().item(), 0)
                    for linear in (layer.q, layer.k, layer.v):
                        self.assertLessEqual(
                            linear.weight.abs().max().item(),
                            math.sqrt(1.5 / linear.in_features),
                        )
                        self.assertGreater(linear.weight.abs().sum().item(), 0)
                        self.assertEqual(linear.bias.count_nonzero().item(), 0)
                    self.assertFalse(torch.equal(layer.q.weight, layer.k.weight))
                    self.assertFalse(torch.equal(layer.k.weight, layer.v.weight))
                if isinstance(layer, (nn.Conv1d, nn.Conv2d)):
                    bound = 1 / math.sqrt(layer.weight[0].numel())
                    self.assertLessEqual(layer.weight.abs().max().item(), bound + 1e-7)
                    self.assertLessEqual(layer.bias.abs().max().item(), bound + 1e-7)
        self.assert_close(encoder().frequency_scale, torch.ones(16), atol=0, rtol=0)

    def test_positional_and_keyword_constructor_contract(self):
        options = configuration(
            m2l_layers=(1, 0, 2),
            m2l_multipliers=(1, 3, 2),
            m2l_attention=(0, 1, 1),
            m2l_freq_downsample_factors=(2, 2),
            m2l_last_time_downsample_factor=3,
        )
        torch.manual_seed(927)
        positional_encoder = codec.STFT2DEncoder(
            3,
            8,
            3,
            2,
            2,
            8,
            32,
            8,
            (1, 0, 2),
            (1, 3, 2),
            (0, 1, 1),
            (2, 2),
            2,
            3,
        )
        torch.manual_seed(927)
        keyword_encoder = codec.STFT2DEncoder(
            **options,
            hidden_channels=8,
            encoder_kernel_size=3,
            encoder_dilation_rate=2,
            encoder_layers=2,
        )
        torch.manual_seed(928)
        positional_decoder = codec.STFT2DDecoder(
            3,
            8,
            32,
            8,
            (1, 0, 2),
            (1, 3, 2),
            (0, 1, 1),
            (2, 2),
            8,
            1,
            2,
            3,
        )
        torch.manual_seed(928)
        keyword_decoder = codec.STFT2DDecoder(
            **options,
            m2l_bottleneck_base_channels=8,
            m2l_num_bottleneck_layers=1,
        )
        for positional, keyword in (
            (positional_encoder, keyword_encoder),
            (positional_decoder, keyword_decoder),
        ):
            self.assertEqual(positional.state_dict().keys(), keyword.state_dict().keys())
            for name, value in positional.state_dict().items():
                self.assert_close(value, keyword.state_dict()[name], atol=0, rtol=0)
            for attribute in ("stft_hop_length", "stft_win_length", "freq_bins"):
                self.assertEqual(getattr(positional, attribute), getattr(keyword, attribute))
        self.assertEqual(positional_encoder.time_downsampling_ratio, 3)
        audio = torch.randn(2, 49)
        codes = positional_encoder(audio)
        self.assert_close(codes, keyword_encoder(audio), atol=0, rtol=0)
        self.assert_close(positional_decoder(codes), keyword_decoder(codes), atol=0, rtol=0)

    def test_target_tensor_manifest(self):
        target = json.loads((FIXTURES / "stft_parameter_shapes.json").read_text())
        model = nn.ModuleDict(
            {
                "encoder": codec.STFT2DEncoder(**target["config"]["encoder"]),
                "decoder": codec.STFT2DDecoder(**target["config"]["decoder"]),
            }
        )
        actual = {name: list(value.shape) for name, value in model.state_dict().items()}
        self.assertEqual(actual, target["parameter_shapes"])
        self.assertEqual(len(actual), 313)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 29_075_362)
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            target["parameter_count"],
        )

    def test_serialization_preserves_values_and_outputs(self):
        pair = nn.ModuleDict({"encoder": encoder(), "decoder": decoder()})
        activate_residuals(pair)
        memory = io.BytesIO()
        torch.save(pair.state_dict(), memory)
        memory.seek(0)
        restored = nn.ModuleDict({"encoder": encoder(), "decoder": decoder()})
        restored.load_state_dict(torch.load(memory, weights_only=True), strict=True)
        for name, value in pair.state_dict().items():
            self.assert_close(value, restored.state_dict()[name], atol=0, rtol=0)
        audio = torch.randn(2, 49)
        codes = pair["encoder"](audio)
        self.assert_close(codes, restored["encoder"](audio), atol=0, rtol=0)
        self.assert_close(pair["decoder"](codes), restored["decoder"](codes), atol=0, rtol=0)

    def test_cpu_full_forward_backward(self):
        for factor in (1, 2, 3):
            with self.subTest(factor=factor):
                analysis, synthesis = (
                    encoder(m2l_last_time_downsample_factor=factor),
                    decoder(m2l_last_time_downsample_factor=factor),
                )
                activate_residuals(analysis)
                activate_residuals(synthesis)
                audio = (torch.randn(2, 2, 53) * 0.1).requires_grad_()
                codes = analysis(audio, torch.tensor([53, 25]))
                output = synthesis(codes, length=53)
                output.square().mean().backward()
                self.assertTrue(audio.grad.isfinite().all())
                self.assertGreater(audio.grad.abs().sum().item(), 0)
                for name, parameter in list(analysis.named_parameters()) + list(
                    synthesis.named_parameters()
                ):
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(parameter.grad.isfinite().all(), name)

    def test_invalid_configurations_are_rejected_during_construction(self):
        invalid = (
            {"latent_channels": 0},
            {"stft_hop_length": 0},
            {"stft_hop_length": 8.0},
            {"stft_win_length": 31},
            {"m2l_base_channels": 3},
            {"m2l_base_channels": 11},
            {"m2l_layers": ()},
            {"m2l_layers": (1,)},
            {"m2l_layers": (1, -1)},
            {"m2l_layers": (1, 0.5)},
            {"m2l_layers": "11"},
            {"m2l_attention": (0, 2)},
            {"m2l_multipliers": (1, 0)},
            {"m2l_multipliers": (1,)},
            {"m2l_freq_downsample_factors": ()},
            {"m2l_freq_downsample_factors": (3,)},
            {"m2l_freq_downsample_factors": (0,)},
            {"m2l_heads": 0},
            {"m2l_heads": 3},
            {"m2l_last_time_downsample_factor": 0},
            {"m2l_last_time_downsample_factor": True},
            dict(
                m2l_layers=(0,),
                m2l_multipliers=(1,),
                m2l_attention=(0,),
                m2l_freq_downsample_factors=(),
                m2l_last_time_downsample_factor=2,
            ),
        )
        for changes in invalid:
            for constructor in (encoder, decoder):
                with self.subTest(constructor=constructor.__name__, changes=changes):
                    with self.assertRaisesRegex(ValueError, ".+"):
                        constructor(**changes)
        for changes in (
            {"encoder_kernel_size": 2},
            {"encoder_layers": -1},
            {"encoder_dilation_rate": 0},
            {"hidden_channels": 0},
        ):
            with self.assertRaises(ValueError):
                encoder(**changes)
        for changes in (
            {"m2l_num_bottleneck_layers": -1},
            {"m2l_bottleneck_base_channels": 2},
        ):
            with self.assertRaises(ValueError):
                decoder(**changes)

    def test_invalid_inputs_and_short_audio(self):
        analysis, synthesis = encoder(), decoder()
        for audio in (
            torch.randn(30),
            torch.randn(2, 1, 1, 30),
            torch.empty(2, 0),
            torch.empty(0, 30),
            torch.empty(2, 0, 30),
            torch.randn(2, 30, dtype=torch.complex64),
            torch.zeros(2, 8),
        ):
            with self.subTest(audio_shape=audio.shape):
                with self.assertRaises(ValueError):
                    analysis(audio)
        self.assertEqual(analysis.waveform_to_representation(torch.zeros(2, 9)).shape[-1], 2)
        for lengths in (
            torch.tensor([1]),
            torch.tensor([-1, 2]),
            torch.tensor([1.5, 2]),
            torch.tensor([float("nan"), 2]),
            torch.tensor([True, False]),
            torch.tensor([1j, 2j]),
        ):
            with self.assertRaises(ValueError):
                analysis(torch.randn(2, 30), lengths)
        for latent in (
            torch.randn(2, 3),
            torch.randn(2, 4, 3),
            torch.empty(2, 3, 0),
            torch.empty(0, 3, 5),
            torch.randn(2, 3, 5, dtype=torch.complex64),
        ):
            with self.assertRaises(ValueError):
                synthesis(latent)
        for representation in (
            torch.randn(2, 2, 16),
            torch.randn(2, 1, 16, 3),
            torch.randn(2, 2, 15, 3),
            torch.empty(2, 2, 16, 0),
            torch.randn(2, 2, 16, 3, dtype=torch.complex64),
        ):
            with self.assertRaises(ValueError):
                analysis.spectral_features(representation)
            with self.assertRaises(ValueError):
                synthesis.representation_to_waveform(representation)
        for length in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                synthesis(torch.randn(2, 3, 3), length)

    def test_cpu_autocast_stays_finite(self):
        analysis, synthesis = encoder(), decoder()
        audio = torch.randn(2, 49, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = synthesis(analysis(audio))
            loss = output.square().mean()
        loss.backward()
        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(output.isfinite().all())
        self.assertTrue(audio.grad.isfinite().all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_fp32_outputs_and_gradients_match_cpu(self):
        old_matmul = torch.backends.cuda.matmul.allow_tf32
        old_cudnn = torch.backends.cudnn.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            cpu_analysis, cpu_synthesis = encoder(), decoder()
            activate_residuals(cpu_analysis)
            activate_residuals(cpu_synthesis)
            gpu_analysis, gpu_synthesis = encoder().cuda(), decoder().cuda()
            gpu_analysis.load_state_dict(cpu_analysis.state_dict())
            gpu_synthesis.load_state_dict(cpu_synthesis.state_dict())
            audio_cpu = (torch.randn(2, 49) * 0.1).requires_grad_()
            audio_gpu = audio_cpu.detach().cuda().requires_grad_()
            codes_cpu, codes_gpu = cpu_analysis(audio_cpu), gpu_analysis(audio_gpu)
            self.assert_close(codes_gpu.cpu(), codes_cpu, atol=0, rtol=0)
            waveform_cpu = cpu_synthesis(codes_cpu)
            waveform_gpu = gpu_synthesis(codes_gpu)
            self.assert_close(waveform_gpu.cpu(), waveform_cpu, atol=2e-5, rtol=2e-4)
            weights = torch.randn_like(waveform_cpu)
            (waveform_cpu * weights).sum().backward()
            (waveform_gpu * weights.cuda()).sum().backward()
            self.assert_close(audio_gpu.grad.cpu(), audio_cpu.grad, atol=8e-5, rtol=8e-4)
            for (name, cpu), (gpu_name, gpu) in zip(
                list(cpu_analysis.named_parameters()) + list(cpu_synthesis.named_parameters()),
                list(gpu_analysis.named_parameters()) + list(gpu_synthesis.named_parameters()),
            ):
                self.assertEqual(name, gpu_name)
                self.assertTrue(gpu.grad.isfinite().all(), name)
                self.assert_close(gpu.grad.cpu(), cpu.grad, atol=1e-4, rtol=1e-3)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_matmul
            torch.backends.cudnn.allow_tf32 = old_cudnn

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_autocast_and_half_module(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                analysis, synthesis = encoder().cuda(), decoder().cuda()
                audio = torch.randn(2, 49, device="cuda", requires_grad=True)
                with torch.autocast("cuda", dtype=dtype):
                    waveform = synthesis(analysis(audio))
                waveform.square().mean().backward()
                self.assertEqual(waveform.dtype, torch.float32)
                self.assertTrue(waveform.isfinite().all())
                self.assertTrue(audio.grad.isfinite().all())
        analysis, synthesis = encoder().cuda().half(), decoder().cuda().half()
        self.assertEqual(analysis.window.dtype, torch.float16)
        waveform = synthesis(analysis(torch.randn(2, 49, device="cuda", dtype=torch.float16)))
        self.assertEqual(waveform.dtype, torch.float32)
        self.assertTrue(waveform.isfinite().all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
