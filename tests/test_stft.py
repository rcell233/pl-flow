import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F

from pl_flow.checkpoint import ModelBundle, save_component
from pl_flow.cli.export import export_stage
from pl_flow.config import write_json
from pl_flow.vocoder.convert_checkpoint import convert_stft_state_dict, main
from pl_flow.vocoder.model import SQCodec
from pl_flow.vocoder.stft import _FrequencyAttention, _PerTimeNorm


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def small_config(time_factor=2):
    spectral = {
        "latent_channels": 4,
        "stft_hop_length": 16,
        "stft_win_length": 64,
        "m2l_base_channels": 8,
        "m2l_layers": (2, 1, 2),
        "m2l_multipliers": (1, 2, 4),
        "m2l_attention": (0, 1, 1),
        "m2l_freq_downsample_factors": (4, 2),
        "m2l_heads": 2,
        "m2l_last_time_downsample_factor": time_factor,
    }
    return {
        "encoder": dict(spectral, hidden_channels=8, encoder_layers=2),
        "decoder": dict(spectral, m2l_bottleneck_base_channels=16, m2l_num_bottleneck_layers=2),
    }


@pytest.mark.parametrize("shape", [(2, 8, 5), (2, 8, 7, 5)])
def test_normalization_is_independent_at_each_time(shape):
    torch.manual_seed(7)
    norm = _PerTimeNorm(8)
    features = torch.randn(shape)
    reference = torch.stack(
        [F.group_norm(frame, 2, norm.weight, norm.bias, norm.eps) for frame in features.unbind(-1)],
        dim=-1,
    )
    torch.testing.assert_close(norm(features), reference)
    changed = features.clone()
    changed[..., 2] *= 100
    torch.testing.assert_close(norm(changed)[..., :2], norm(features)[..., :2], rtol=0, atol=0)


def test_attention_matches_per_frame_pytorch_attention_and_input_gradient():
    torch.manual_seed(19)
    attention = _FrequencyAttention(8, heads=2)
    reference = nn.MultiheadAttention(8, 2, batch_first=True)
    with torch.no_grad():
        for index, projection in enumerate((attention.q, attention.k, attention.v)):
            projection.weight.copy_(reference.in_proj_weight.chunk(3)[index])
            projection.bias.copy_(reference.in_proj_bias.chunk(3)[index])
        attention.projection.load_state_dict(reference.out_proj.state_dict())
    features = torch.randn(2, 8, 5, 3, requires_grad=True)
    expected = []
    for frame in attention.normalization(features).unbind(-1):
        tokens = frame.transpose(1, 2)
        expected.append(reference(tokens, tokens, tokens, need_weights=False)[0].transpose(1, 2))
    expected = torch.stack(expected, dim=-1) + features
    actual = attention(features)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    actual_gradient = torch.autograd.grad(actual.square().sum(), features, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected.square().sum(), features)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-5, atol=2e-6)
    changed = features.detach().clone()
    changed[..., 1] += 10 * torch.randn_like(changed[..., 1])
    torch.testing.assert_close(attention(changed)[..., 0], actual.detach()[..., 0], rtol=0, atol=0)


@pytest.mark.parametrize("time_factor", [1, 2])
def test_codec_lengths_mask_quantization_stereo_and_gradients(time_factor):
    torch.manual_seed(13)
    codec = SQCodec(small_config(time_factor))
    audio = (0.1 * torch.randn(2, 2, 129)).requires_grad_()
    lengths = torch.tensor([129, 65])
    latent = codec.encoder(audio, lengths)
    expected_frames = (9 + time_factor - 1) // time_factor
    valid_second = (5 + time_factor - 1) // time_factor
    assert latent.shape == (2, 4, expected_frames)
    assert torch.count_nonzero(latent[1, :, valid_second:]) == 0
    assert latent.abs().max() <= 1
    torch.testing.assert_close(latent * 9, (latent * 9).round())
    torch.testing.assert_close(latent, codec.encoder(audio.mean(1), lengths), rtol=0, atol=0)
    full = codec.decoder(latent)
    cropped = codec.decoder(latent, length=129)
    assert full.shape == (2, 1, expected_frames * time_factor * 16)
    torch.testing.assert_close(cropped, full[..., :129], rtol=0, atol=0)
    assert cropped.abs().max() <= 1
    cropped.square().mean().backward()
    assert torch.isfinite(audio.grad).all()
    assert audio.grad.abs().sum() > 0
    for parameter in codec.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize(
    "change, message",
    [
        ({"m2l_layers": (1,)}, "equal nonzero lengths"),
        ({"m2l_freq_downsample_factors": (2,)}, "per resolution boundary"),
        ({"m2l_freq_downsample_factors": (3, 2)}, "division.*exact"),
        ({"m2l_last_time_downsample_factor": 0}, "integer >= 1"),
        ({"m2l_heads": 3}, "divisible by m2l_heads"),
        ({"m2l_heads": 0}, "m2l_heads must be an integer"),
    ],
)
def test_invalid_architecture_fails_before_inference(change, message):
    config = small_config()
    config["encoder"].update(change)
    with pytest.raises(ValueError, match=message):
        SQCodec(config)


def test_conversion_rejects_incomplete_mixed_or_wrong_shape_states():
    config = small_config()
    state = SQCodec(config).state_dict()
    copied = convert_stft_state_dict(state, config)
    for key in state:
        torch.testing.assert_close(copied[key], state[key], rtol=0, atol=0)
    key = "encoder.spectral_input.weight"
    with pytest.raises(ValueError, match="missing="):
        convert_stft_state_dict({k: v for k, v in state.items() if k != key}, config)
    with pytest.raises(ValueError, match="shape_mismatches="):
        convert_stft_state_dict(dict(state, **{key: torch.zeros(1)}), config)
    with pytest.raises(ValueError, match="Mixed checkpoint parameter layouts"):
        convert_stft_state_dict(
            dict(state, **{"decoder.dec.conv_inp.bias": torch.zeros(16)}), config
        )
    with pytest.raises(ValueError, match="Mixed checkpoint parameter layouts"):
        convert_stft_state_dict(
            dict(state, **{"decoder.synthesis.temporal_input.bias": torch.zeros(16)}), config
        )
    with pytest.raises(ValueError, match="unexpected="):
        convert_stft_state_dict(dict(state, accidental_tensor=torch.ones(1)), config)


def legacy_fixture():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/stft_legacy_reference.json").read_text()
    )
    generator = torch.Generator().manual_seed(fixture["seed"])
    state = {}
    for name, shape in sorted(fixture["parameter_shapes"].items()):
        tensor = torch.randn(shape, generator=generator) * 0.08
        if ("norm" in name and name.endswith("weight")) or name.endswith("gain.scale"):
            tensor = tensor + 1
        state[name] = tensor
    return fixture, state


@pytest.mark.parametrize("schema", ["original", "staged"])
def test_legacy_checkpoint_matches_recorded_numerical_contract(schema):
    fixture, state = legacy_fixture()
    converted = convert_stft_state_dict(state, fixture["config"])
    if schema == "staged":
        layout = json.loads(
            (Path(__file__).parent / "fixtures/stft_staged_layout.json").read_text()
        )
        staged = {}
        for entry in layout["entries"]:
            tensor = state[entry["original"]]
            if entry["operation"] == "slice":
                tensor = tensor[entry["start"] : entry["stop"]]
            elif entry["operation"] == "flatten":
                tensor = tensor.flatten()
            else:
                assert entry["operation"] == "identity"
            staged[entry["staged"]] = tensor
        migrated = convert_stft_state_dict(staged, fixture["config"])
        assert set(migrated) == set(converted)
        for key in converted:
            torch.testing.assert_close(migrated[key], converted[key], rtol=0, atol=0)
        converted = migrated
    codec = SQCodec(fixture["config"]).eval()
    codec.load_state_dict(converted, strict=True)
    audio = torch.tensor(fixture["audio"])
    probe = torch.tensor(fixture["probe"])
    with torch.inference_mode():
        actual = {
            "features": codec.encoder.spectral_features(
                codec.encoder.waveform_to_representation(audio)
            ),
            "latent": codec.encoder(audio, torch.tensor(fixture["lengths"])),
            "spectrum": codec.decoder.latent_to_representation(probe),
            "waveform": codec.decoder(probe),
        }
    for name, output in actual.items():
        torch.testing.assert_close(
            output,
            torch.tensor(fixture["expected"][name]),
            rtol=1e-5,
            atol=1e-6,
        )
    packed_key = "decoder.dec.up_layers.0.attention.mha.in_proj_weight"
    with pytest.raises(ValueError, match="packed Q/K/V shape"):
        convert_stft_state_dict(
            dict(state, **{packed_key: state[packed_key][:-1]}), fixture["config"]
        )
    gain_key = "encoder.encoder_frontend.gain.scale"
    with pytest.raises(ValueError, match="frequency gain shape"):
        convert_stft_state_dict(
            dict(state, **{gain_key: state[gain_key].flatten()}), fixture["config"]
        )


@pytest.mark.parametrize(
    "file_type", ["safetensors", "state_dict", "training_ema", "training_online"]
)
def test_conversion_cli_exports_only_selected_weights_and_preserves_source(tmp_path, file_type):
    config = small_config()
    state = SQCodec(config).state_dict()
    checkpoint = tmp_path / ("weights.safetensors" if file_type == "safetensors" else "weights.pt")
    config_path = tmp_path / "config.json"
    write_json(config_path, {"type": "vocoder", "model": config})
    output = tmp_path / "converted"
    arguments = ["--checkpoint", str(checkpoint), "--output", str(output)]
    if file_type == "safetensors":
        save_file(state, str(checkpoint))
    elif file_type == "state_dict":
        torch.save(
            {"state_dict": {f"codec.{k}": v for k, v in state.items()}, "epoch": 3}, checkpoint
        )
        arguments.extend(["--state-key", "state_dict", "--prefix", "codec."])
    else:
        selected = {key: value + 1 for key, value in state.items()}
        torch.save(
            {
                "format_version": 1,
                "stage": "vocoder",
                "model_config": config,
                "training": {
                    "generator": {"model": state, "optimizer": {"private": "not exported"}},
                    "ema": {f"ema_model.{key}": value for key, value in selected.items()},
                },
            },
            checkpoint,
        )
        if file_type == "training_online":
            arguments.append("--online")
        else:
            state = selected
    original_bytes = checkpoint.read_bytes()
    main(arguments)
    assert checkpoint.read_bytes() == original_bytes
    loaded = load_file(str(output / "model.safetensors"))
    SQCodec(config).load_state_dict(loaded, strict=True)
    for key in state:
        torch.testing.assert_close(loaded[key], state[key], rtol=0, atol=0)
    assert set(path.name for path in output.iterdir()) == {"config.json", "model.safetensors"}
    with pytest.raises(FileExistsError):
        main(arguments)


def test_local_legacy_weights_convert_load_and_export(tmp_path):
    """Optional integration with the private trained bundle; no old source is shipped."""
    directory = Path(__file__).resolve().parents[1] / "artifacts/pretrained/vocoder"
    if not (directory / "model.safetensors").exists():
        pytest.skip("Private vocoder weights are not installed")
    config = json.loads((directory / "config.json").read_text())["model"]
    state = load_file(str(directory / "model.safetensors"))
    if "encoder.encoder_frontend.gain.scale" not in state:
        pytest.skip("Installed vocoder is already migrated")
    converted = convert_stft_state_dict(state, config)
    assert sum(value.numel() for value in state.values()) == sum(
        value.numel() for value in converted.values()
    )
    mapping = json.loads(
        (Path(__file__).parent / "fixtures/stft_checkpoint_mapping.json").read_text()
    )
    assert set(state) == {entry["original"] for entry in mapping["entries"]}
    assert set(converted) == {entry["current"] for entry in mapping["entries"]}
    staged = {}
    for entry in mapping["entries"]:
        tensor = state[entry["original"]]
        assert list(tensor.shape) == entry["source_shape"]
        if entry["operation"] == "row_slice":
            tensor = tensor[entry["start"] : entry["stop"]]
        elif entry["operation"] == "reshape":
            tensor = tensor.reshape(entry["target_shape"])
        else:
            assert entry["operation"] == "identity"
        assert list(tensor.shape) == entry["target_shape"]
        torch.testing.assert_close(tensor, converted[entry["current"]], rtol=0, atol=0)
        staged[entry["staged"]] = tensor
    migrated_first = convert_stft_state_dict(staged, config)
    for key in converted:
        torch.testing.assert_close(migrated_first[key], converted[key], rtol=0, atol=0)
    entry = save_component(tmp_path / "base/vocoder", "vocoder", config, state)
    write_json(
        tmp_path / "base/bundle.json", {"format_version": 1, "components": {"vocoder": entry}}
    )
    codec = ModelBundle(tmp_path / "base").load("vocoder")
    codec.load_state_dict(converted, strict=True)
    with torch.inference_mode():
        assert torch.isfinite(codec(torch.randn(1, 1, 2560) * 0.1)).all()
    checkpoint = tmp_path / "training.pt"
    torch.save(
        {
            "format_version": 1,
            "stage": "vocoder",
            "model_config": config,
            "training": {"generator": {"model": state}},
        },
        checkpoint,
    )
    export_stage(SimpleNamespace(checkpoint=checkpoint, output=tmp_path / "export", online=True))
    exported = load_file(str(tmp_path / "export/model.safetensors"))
    assert set(exported) == set(converted)
