# SPDX-License-Identifier: CC-BY-SA-3.0
"""Extract embeddings from 16 kHz audio, using files or a JSON-lines stream."""

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


def checkpoint_state(path):
    if path.suffix == ".safetensors":
        return load_file(str(path))
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError("Expected a Seed-TTS-Eval speaker checkpoint with a model state")
    position = "feature_extract.model.encoder.pos_conv.0."
    renamed = {
        position + "weight_g": position + "parametrizations.weight.original0",
        position + "weight_v": position + "parametrizations.weight.original1",
    }
    state = {}
    for key, tensor in payload["model"].items():
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("Expected a tensor-only speaker model state")
        # The training classifier is not part of the embedding network.
        if key == "loss_calculator.projection.weight":
            continue
        key = renamed.get(key, key)
        if key in state:
            raise ValueError(f"Duplicate speaker parameter: {key}")
        state[key] = tensor
    return state


def load_model(directory, device, weights_filename="model.safetensors"):
    from .model import SpeakerEncoder

    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    if config.get("format_version") != 1 or config.get("type") != "speaker":
        raise ValueError("Expected a speaker component directory")
    model = SpeakerEncoder(config["model"])
    model.load_state_dict(checkpoint_state(directory / weights_filename), strict=True)
    return model.to(device).eval()


def validate_audio(audio, lengths):
    if audio.ndim != 2 or min(audio.shape) <= 0 or audio.dtype.kind != "f":
        raise ValueError("Expected floating audio [batch, samples]")
    if (
        lengths.shape != (len(audio),)
        or lengths.dtype.kind not in "iu"
        or (lengths <= 0).any()
        or (lengths > audio.shape[1]).any()
    ):
        raise ValueError("Invalid audio lengths")
    if any(not np.isfinite(row[: int(size)]).all() for row, size in zip(audio, lengths)):
        raise ValueError("Audio must be finite")


def decode_request(request):
    if not isinstance(request, dict) or request.get("op") != "embed":
        raise ValueError("Expected an embed request")
    shape = request["shape"]
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(type(n) is not int or n <= 0 for n in shape)
    ):
        raise ValueError("Invalid audio shape")
    raw = base64.b64decode(request["audio"], validate=True)
    if len(raw) != shape[0] * shape[1] * 4:
        raise ValueError("Audio size does not match its shape")
    audio = np.frombuffer(raw, dtype="<f4").reshape(shape).copy()
    lengths = np.asarray(request["lengths"])
    validate_audio(audio, lengths)
    return audio, lengths.astype(np.int64)


def configure_precision(request):
    precision = request.get("precision", "highest")
    if precision not in ("highest", "high", "medium"):
        raise ValueError("Invalid matmul precision")
    flags = {
        "cudnn_tf32": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": False,
        "deterministic": False,
        "deterministic_warn_only": False,
    }
    for name, default in flags.items():
        flags[name] = request.get(name, default)
        if type(flags[name]) is not bool:
            raise ValueError(f"Expected a boolean for {name}")
    torch.set_float32_matmul_precision(precision)
    torch.backends.cudnn.allow_tf32 = flags["cudnn_tf32"]
    torch.backends.cudnn.benchmark = flags["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = flags["cudnn_deterministic"]
    torch.use_deterministic_algorithms(
        flags["deterministic"], warn_only=flags["deterministic_warn_only"]
    )


@torch.inference_mode()
def embed(model, audio, lengths, device):
    validate_audio(audio, lengths)
    output = (
        model(
            torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).to(device),
            torch.from_numpy(np.ascontiguousarray(lengths, dtype=np.int64)).to(device),
        )
        .float()
        .cpu()
        .numpy()
    )
    if output.shape != (len(audio), 256) or not np.isfinite(output).all():
        raise ValueError("Invalid speaker embedding")
    return output


def respond(message):
    print(json.dumps(message, separators=(",", ":")), flush=True)


def serve(model, device):
    respond({"ok": True, "ready": True, "embedding_dim": 256})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            audio, lengths = decode_request(request)
            configure_precision(request)
            output = embed(model, audio, lengths, device)
            respond(
                {
                    "ok": True,
                    "shape": list(output.shape),
                    "embedding": base64.b64encode(output.astype("<f4").tobytes()).decode("ascii"),
                }
            )
        except Exception as error:
            respond({"ok": False, "error": f"{type(error).__name__}: {error}"})


def read_audio(path, lengths_path=None):
    if Path(path).suffix.lower() == ".npy":
        audio = np.load(path, allow_pickle=False)
        if audio.ndim == 1:
            audio = audio[None]
    else:
        import soundfile as sf

        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        if rate != 16000:
            raise ValueError("Speaker audio must be sampled at 16000 Hz")
        audio = audio.mean(axis=1)[None]
    if audio.ndim != 2:
        raise ValueError("Expected audio [batch, samples]")
    lengths = (
        np.load(lengths_path, allow_pickle=False)
        if lengths_path
        else np.full(len(audio), audio.shape[1], dtype=np.int64)
    )
    validate_audio(audio, lengths)
    return audio, lengths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", required=True, help="Directory containing the speaker config.json"
    )
    parser.add_argument(
        "--weights", default="model.safetensors", help="Checkpoint filename within --model-dir"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--serve", action="store_true", help="Read JSON lines from stdin")
    mode.add_argument("--input", help="16 kHz audio file or float32 .npy waveform batch")
    parser.add_argument("--lengths", help="Optional integer .npy sample lengths for --input")
    parser.add_argument("--output", help="Write float32 [batch, 256] .npy embeddings")
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("--threads must be positive")
    if args.serve and (args.output or args.lengths):
        parser.error("--output and --lengths require --input")
    if args.input and not args.output:
        parser.error("--input requires --output")
    try:
        torch.set_num_threads(args.threads)
        if args.input:
            if Path(args.output).exists():
                raise FileExistsError(args.output)
            audio, lengths = read_audio(args.input, args.lengths)
        model = load_model(args.model_dir, args.device, args.weights)
        if args.serve:
            serve(model, args.device)
        else:
            output = embed(model, audio, lengths, args.device)
            with Path(args.output).open("xb") as stream:
                np.save(stream, output, allow_pickle=False)
    except Exception as error:
        if args.serve:
            respond({"ok": False, "error": f"{type(error).__name__}: {error}"})
        else:
            print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
