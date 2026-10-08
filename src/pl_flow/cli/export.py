"""Export explicitly selected release weights, never optimizer or local training metadata."""

import torch

from pl_flow.checkpoint import save_component, tensor_fingerprint


def export_stage(args):
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if state.get("format_version") != 1:
        raise ValueError("Expected a PL-Flow training checkpoint")
    stage, training = state["stage"], state["training"]
    if stage == "vocoder":
        if args.online:
            weights = training["generator"]["model"]
        else:
            weights = {
                k.removeprefix("ema_model."): v
                for k, v in training["ema"].items()
                if k.startswith("ema_model.")
            }
    elif training["ema"] is not None and not args.online:
        weights = {
            k.removeprefix("module."): v
            for k, v in training["ema"]["model"].items()
            if k != "n_averaged"
        }
    else:
        weights = training["model"]
    if stage == "s1":
        weights = {k: v for k, v in weights.items() if not k.startswith("plbert.")}
    if stage == "vocoder":
        from pl_flow.vocoder.convert_checkpoint import convert_stft_state_dict

        weights = convert_stft_state_dict(weights, state["model_config"])
    interfaces = {}
    if stage == "s1":
        interfaces = state["interfaces"]
    elif stage == "s2":
        prefix = "conditioner.text_conditioner.prosody_down."
        interfaces["prosody_fingerprint"] = tensor_fingerprint(
            {k.removeprefix(prefix): v for k, v in weights.items() if k.startswith(prefix)}
        )
    entry = save_component(args.output, stage, state["model_config"], weights, interfaces)
    print(entry)
