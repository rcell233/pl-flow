"""Convert legacy STFT checkpoints to the codec's current parameter layout.

Run ``python -m pl_flow.vocoder.convert_checkpoint --help`` for file conversion.
Only inference weights are exported; optimizer states are not transferable by
renaming because parameter ordering and attention parameter counts changed.
"""

import argparse
import json
import re
from collections.abc import Mapping
from pathlib import Path

import torch
from safetensors.torch import load_file


def _legacy_modules(config):
    """Describe the old tensor namespace, without importing the old implementation."""
    modules = {
        "encoder.encoder_frontend.conv_inp": "encoder.analysis.input_projection",
        "encoder.encoder_frontend.prenorm_2d_to_1d": "encoder.analysis.output_norm",
        "decoder.dec.conv_inp": "decoder.synthesis.temporal_input",
        "decoder.dec.conv_out_bottleneck": "decoder.synthesis.frequency_projection",
        "decoder.dec.norm_out": "decoder.synthesis.readout.norm",
        "decoder.dec.conv_out": "decoder.synthesis.readout.conv",
    }

    def residual(source, target):
        for old, new in {
            "conv1": "branch.conv_in",
            "conv2": "branch.conv_out",
            "norm1": "branch.norm_in",
            "norm2": "branch.norm_out",
            "res_conv": "shortcut",
            "attention.norm": "attention.norm",
            "attention.mha.out_proj": "attention.output",
            "attention.mha": "attention",
        }.items():
            modules[f"{source}.{old}"] = f"{target}.{new}"

    for side in ("encoder", "decoder"):
        depths = config[side].get("m2l_layers", (1, 1, 1, 1, 1))
        resolutions = range(len(depths)) if side == "encoder" else range(len(depths) - 1, -1, -1)
        source = (
            "encoder.encoder_frontend.down_layers" if side == "encoder" else "decoder.dec.up_layers"
        )
        target = "encoder.analysis.stages" if side == "encoder" else "decoder.synthesis.stages"
        flat_index = 0
        for stage in resolutions:
            for block in range(depths[stage]):
                residual(f"{source}.{flat_index}", f"{target}.{stage}.blocks.{block}")
                flat_index += 1
            has_resize = stage < len(depths) - 1 if side == "encoder" else stage > 0
            if has_resize:
                resize = "downsample" if side == "encoder" else "upsample"
                for part in ("conv", "norm"):
                    modules[f"{source}.{flat_index}.{part}"] = f"{target}.{stage}.{resize}.{part}"
                flat_index += 1

    for block in range(config["decoder"].get("m2l_num_bottleneck_layers", 4)):
        residual(
            f"decoder.dec.bottleneck_layers.{block}", f"decoder.synthesis.temporal_blocks.{block}"
        )
    return sorted(modules.items(), key=lambda item: len(item[0]), reverse=True)


def _current_name(name):
    """Translate parameter names from the staged checkpoint layout."""
    roots = {
        "encoder.analysis.frequency_gain": "encoder.frequency_scale",
        "encoder.analysis.input_projection": "encoder.spectral_input",
        "encoder.analysis.output_norm": "encoder.final_normalization",
        "encoder.enc_q.pre": "encoder.posterior.input",
        "encoder.enc_q.enc": "encoder.posterior.network",
        "encoder.enc_q.proj": "encoder.posterior.output",
        "decoder.synthesis.temporal_input": "decoder.latent_input",
        "decoder.synthesis.frequency_projection": "decoder.spectral_expansion",
        "decoder.synthesis.readout.norm": "decoder.final_normalization",
        "decoder.synthesis.readout.conv": "decoder.spectral_output",
    }
    for source, destination in roots.items():
        if name == source or name.startswith(source + "."):
            return destination + name[len(source) :]

    residual = re.fullmatch(
        r"(encoder|decoder)\.(?:analysis|synthesis)\.stages\.(\d+)\.blocks\.(\d+)\.(.+)", name
    )
    temporal = re.fullmatch(r"decoder\.synthesis\.temporal_blocks\.(\d+)\.(.+)", name)
    if residual or temporal:
        if residual:
            side, resolution, unit, role = residual.groups()
            prefix = f"{side}.resolutions.{resolution}.residuals.{unit}"
        else:
            unit, role = temporal.groups()
            prefix = f"decoder.temporal_residuals.{unit}"
        stem, parameter = role.rsplit(".", 1)
        roles = {
            "branch.norm_in": "norm_input",
            "branch.conv_in": "conv_input",
            "branch.norm_out": "norm_output",
            "branch.conv_out": "conv_output",
            "shortcut": "shortcut",
            "attention.norm": "frequency_attention.normalization",
            "attention.query": "frequency_attention.q",
            "attention.key": "frequency_attention.k",
            "attention.value": "frequency_attention.v",
            "attention.output": "frequency_attention.projection",
            # A transient packed-QKV address from the original data schema.
            "attention": "frequency_attention",
        }
        return f"{prefix}.{roles.get(stem, stem)}.{parameter}"

    transition = re.fullmatch(
        r"encoder\.analysis\.stages\.(\d+)\.downsample\.(norm|conv)\.(weight|bias)", name
    )
    if transition:
        resolution, role, parameter = transition.groups()
        role = {"norm": "normalization", "conv": "filter"}[role]
        return f"encoder.transitions.{resolution}.{role}.{parameter}"
    transition = re.fullmatch(
        r"decoder\.synthesis\.stages\.(\d+)\.upsample\.conv\.(weight|bias)", name
    )
    if transition:
        resolution, parameter = transition.groups()
        return f"decoder.transitions.{int(resolution) - 1}.filter.{parameter}"
    # Preserve unrecognized fields so strict validation reports them as unexpected.
    return name


def convert_stft_state_dict(state, config):
    """Convert full SQCodec weights and validate every tensor against ``config``.

    Already-converted states are accepted. Mixed old/new spectral namespaces,
    missing tensors, unexpected tensors and shape mismatches are rejected. The
    input mapping and its tensors are never modified in place.
    """
    from .model import SQCodec

    if (
        not isinstance(state, Mapping)
        or not state
        or any(
            not isinstance(key, str) or not isinstance(value, torch.Tensor)
            for key, value in state.items()
        )
    ):
        raise ValueError("Expected a nonempty tensor-only SQCodec state_dict")
    with torch.device("meta"):
        expected = SQCodec(config).state_dict()
    prefixes = {
        0: ("encoder.encoder_frontend.", "decoder.dec."),
        1: ("encoder.analysis.", "decoder.synthesis."),
        2: (
            "encoder.spectral_input.",
            "encoder.frequency_scale",
            "encoder.resolutions.",
            "encoder.transitions.",
            "encoder.final_normalization.",
            "encoder.posterior.",
            "decoder.latent_input.",
            "decoder.temporal_residuals.",
            "decoder.spectral_expansion.",
            "decoder.resolutions.",
            "decoder.transitions.",
            "decoder.final_normalization.",
            "decoder.spectral_output.",
        ),
    }
    schemas = {
        schema for schema, roots in prefixes.items() if any(key.startswith(roots) for key in state)
    }
    if len(schemas) > 1:
        raise ValueError("Mixed checkpoint parameter layouts")
    schema = next(iter(schemas), 2)
    legacy = schema == 0
    modules = _legacy_modules(config) if legacy else ()
    converted = {}

    def insert(name, tensor):
        if name in converted:
            raise ValueError(f"Multiple source tensors map to {name}")
        converted[name] = tensor

    for key, tensor in state.items():
        if key == "encoder.encoder_frontend.gain.scale":
            bins = expected["encoder.frequency_scale"].numel()
            if tuple(tensor.shape) != (1, 1, bins, 1):
                raise ValueError(f"Invalid legacy frequency gain shape: {tuple(tensor.shape)}")
            insert("encoder.frequency_scale", tensor.reshape(bins))
            continue
        name = key
        for source, target in modules:
            if key.startswith(source + "."):
                name = target + key[len(source) :]
                break
        if schema != 2:
            name = _current_name(name)
        if name.endswith(
            (".frequency_attention.in_proj_weight", ".frequency_attention.in_proj_bias")
        ):
            attention, suffix = name.rsplit(".in_proj_", 1)
            target_shape = expected.get(f"{attention}.q.{suffix}")
            if target_shape is None or tuple(tensor.shape) != (
                3 * target_shape.shape[0],
                *target_shape.shape[1:],
            ):
                raise ValueError(f"Invalid packed Q/K/V shape for {key}: {tuple(tensor.shape)}")
            for projection, part in zip(("q", "k", "v"), tensor.chunk(3, dim=0)):
                insert(f"{attention}.{projection}.{suffix}", part)
        else:
            insert(name, tensor)

    missing = sorted(expected.keys() - converted.keys())
    unexpected = sorted(converted.keys() - expected.keys())
    mismatched = [
        f"{key}: {tuple(converted[key].shape)} != {tuple(expected[key].shape)}"
        for key in expected.keys() & converted.keys()
        if converted[key].shape != expected[key].shape
    ]
    if missing or unexpected or mismatched:
        raise ValueError(
            "Incompatible SQCodec weights: "
            f"missing={missing}, unexpected={unexpected}, shape_mismatches={mismatched}"
        )
    return converted


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", type=Path, help="Architecture JSON or component config.json")
    parser.add_argument(
        "--output", required=True, type=Path, help="New release-component directory"
    )
    parser.add_argument("--state-key", help="Dotted path to weights in a PyTorch checkpoint")
    parser.add_argument(
        "--prefix", default="", help="Select and strip an explicit tensor key prefix"
    )
    parser.add_argument(
        "--online", action="store_true", help="Select online PL-Flow training weights"
    )
    args = parser.parse_args(argv)

    if args.checkpoint.suffix == ".safetensors":
        payload = load_file(str(args.checkpoint))
    else:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config_path = args.config
    is_training = (
        isinstance(payload, Mapping)
        and payload.get("format_version") == 1
        and payload.get("stage") == "vocoder"
    )
    if config_path is None and is_training:
        config = payload["model_config"]
    else:
        config_path = config_path or args.checkpoint.with_name("config.json")
        config = json.loads(config_path.read_text())
        if config.get("type") == "vocoder":
            config = config["model"]

    if args.online and (not is_training or args.state_key):
        raise ValueError("--online requires a PL-Flow vocoder checkpoint without --state-key")
    if args.state_key:
        state = payload
        for part in args.state_key.split("."):
            state = state[part]
    elif is_training:
        training = payload["training"]
        state = (
            training["generator"]["model"]
            if args.online
            else {
                key.removeprefix("ema_model."): value
                for key, value in training["ema"].items()
                if key.startswith("ema_model.")
            }
        )
    else:
        state = payload
    if args.prefix:
        state = {
            key.removeprefix(args.prefix): value
            for key, value in state.items()
            if key.startswith(args.prefix)
        }
    converted = convert_stft_state_dict(state, config)

    from pl_flow.checkpoint import save_component

    entry = save_component(args.output, "vocoder", config, converted)
    print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()
