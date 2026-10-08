"""Command-line entry points. No downloads or model loading happen on import."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="pl-flow")
    commands = parser.add_subparsers(dest="command", required=True)
    synth = commands.add_parser(
        "synthesize", help="Synthesize from text and a transcribed reference"
    )
    synth.add_argument("--models", required=True)
    synth.add_argument("--text", required=True)
    synth.add_argument("--reference-audio", required=True)
    synth.add_argument("--reference-text", required=True)
    synth.add_argument("--language", choices=["zh", "en", "ja"], default="zh")
    synth.add_argument("--reference-language", choices=["zh", "en", "ja"])
    synth.add_argument("--output", required=True)
    synth.add_argument("--device", default="cuda")
    synth.add_argument("--seed", type=int, default=1234)
    for stage in ("s1", "s2"):
        synth.add_argument(f"--{stage}-steps", type=int)
        synth.add_argument(f"--{stage}-cfg", type=float)
    from .seedtts import add_arguments

    add_arguments(commands.add_parser("seedtts", help="Generate the Seed-TTS-Eval TTS splits"))
    training = commands.add_parser("train", help="Train S1, S2, the aligner or the SQ codec")
    training.add_argument("--config", required=True)
    training.add_argument("--data", required=True)
    training.add_argument("--audio-root")
    training.add_argument("--models", help="Bundle containing frozen dependencies")
    initialization = training.add_mutually_exclusive_group()
    initialization.add_argument(
        "--init-weights", help="Warm-start from a release bundle; new optimizer/EMA"
    )
    initialization.add_argument("--resume", help="Restore a full training checkpoint")
    training.add_argument("--output", required=True)
    training.add_argument("--device", default="cuda")
    training.add_argument("--workers", type=int, default=4)
    training.add_argument("--max-steps", type=int)
    preprocess = commands.add_parser(
        "preprocess", help="Pack data, align phonemes or extract sidecars"
    )
    stages = preprocess.add_subparsers(dest="operation", required=True)
    for name in ("pack", "filter", "align", "features"):
        command = stages.add_parser(name)
        command.add_argument("--input", required=True)
        command.add_argument("--audio-root", required=True)
        if name != "features":
            command.add_argument("--output", required=True)
        if name in {"align", "features"}:
            command.add_argument("--models", required=True)
            command.add_argument("--device", default="cuda")
            command.add_argument("--seed", type=int, default=1234)
        if name == "features":
            command.add_argument("--max-frames", type=int, default=1000)
            command.add_argument("--verify-existing", action="store_true")
            command.add_argument(
                "--resume",
                action="store_true",
                help="Verify matching sidecars and create missing ones",
            )
        if name == "filter":
            command.add_argument("--min-seconds", type=float, default=0.5)
            command.add_argument("--max-seconds", type=float, default=20.0)
            command.add_argument("--max-tokens", type=int, default=512)
    codec = commands.add_parser(
        "codec", help="Encode, decode or reconstruct audio with the SQ codec"
    )
    codec.add_argument("operation", choices=["encode", "decode", "reconstruct"])
    codec.add_argument("--models", required=True)
    codec.add_argument("--input", required=True)
    codec.add_argument("--output", required=True)
    codec.add_argument("--device", default="cuda")
    export = commands.add_parser(
        "export", help="Export one trained stage as config.json + model.safetensors"
    )
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    export.add_argument(
        "--online", action="store_true", help="Export online rather than EMA weights"
    )
    bundle = commands.add_parser("bundle", help="Replace exported components in a new model bundle")
    bundle.add_argument("--base", required=True)
    bundle.add_argument("--component", action="append", required=True)
    bundle.add_argument("--output", required=True)
    verify = commands.add_parser(
        "verify", help="Verify all bundle hashes and strictly load components"
    )
    verify.add_argument("--models", required=True)
    args = parser.parse_args()
    if args.command == "seedtts":
        from .seedtts import run_seedtts

        run_seedtts(args)
    elif args.command == "train":
        from pl_flow.training.runner import train

        train(args)
    elif args.command == "preprocess":
        from pl_flow.data.preprocess import align_data, extract_features, filter_data, pack_data

        {
            "pack": pack_data,
            "filter": filter_data,
            "align": align_data,
            "features": extract_features,
        }[args.operation](args)
    elif args.command == "synthesize":
        import soundfile as sf
        import torch

        from pl_flow.inference.pipeline import Synthesizer

        if Path(args.output).exists():
            raise FileExistsError(args.output)
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision("high")
        settings = {
            key: getattr(args, key)
            for key in ("s1_steps", "s2_steps", "s1_cfg", "s2_cfg")
            if getattr(args, key) is not None
        }
        with Synthesizer(args.models, args.device, **settings) as model:
            wave = model.synthesize(
                args.text,
                args.reference_audio,
                args.reference_text,
                args.language,
                args.reference_language,
                args.seed,
            )
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        sf.write(args.output, wave.T.numpy(), model.sample_rate)
        print(f"Saved {wave.size(-1) / model.sample_rate:.2f} seconds")
    elif args.command == "codec":
        from .codec import run_codec

        run_codec(args)
    elif args.command == "bundle":
        from .bundle import assemble_bundle

        assemble_bundle(args)
    elif args.command == "export":
        from .export import export_stage

        export_stage(args)
    else:
        import torch

        from pl_flow.checkpoint import ModelBundle

        torch.set_num_threads(1)
        bundle = ModelBundle(args.models)
        for name in bundle.manifest["components"]:
            model = bundle.load(name)
            if name == "speaker":
                with model:
                    print(
                        json.dumps(
                            {"component": name, "execution": "subprocess", "embedding_dim": 256}
                        )
                    )
            else:
                print(
                    json.dumps(
                        {
                            "component": name,
                            "parameters": sum(p.numel() for p in model.parameters()),
                        }
                    )
                )
            del model


if __name__ == "__main__":
    main()
