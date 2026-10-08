"""Portable packing, forced alignment and deterministic offline feature extraction."""

import math
from pathlib import Path

import numpy as np
import torch

from pl_flow.alignment.objective import align
from pl_flow.checkpoint import ModelBundle, tensor_fingerprint
from pl_flow.config import write_json
from pl_flow.models.conditioner import ScalarQuantize9

from .audio import fit_length, load_audio
from .manifest import Manifest, pack, write_rows


def pack_data(args):
    source = Path(args.input)
    if source.suffix in {".txt", ".lst"}:
        rows = [
            {"audio_path": line.strip()} for line in source.read_text().splitlines() if line.strip()
        ]
    else:
        rows = Manifest(source, args.audio_root).rows
    pack(rows, args.output, args.audio_root)


def aligned_rows(input_path, audio_root, models, device, seed):
    """Load models inside the generator, not in the Arrow fingerprint closure."""
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    bundle = ModelBundle(models)
    model = bundle.load("aligner", device).requires_grad_(False)
    hubert = bundle.load("hubert", device)
    source = Manifest(input_path, audio_root)
    for index, original in enumerate(source.rows):
        row = dict(original)
        ids = source.ids(row, mixed=True)
        wave = load_audio(source.audio_path(row), 16000).to(device)
        lengths = torch.tensor([wave.size(-1)], device=device)
        tokens = torch.tensor([ids], device=device)
        torch.manual_seed(seed + index)
        durations = align(model, hubert, wave, lengths, tokens, tokens.new_tensor([len(ids)]))
        row.update(phoneme_ids=ids, dur=durations[0].cpu().tolist(), duration=wave.size(-1) / 16000)
        yield row


def align_data(args):
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    write_rows(
        aligned_rows(args.input, args.audio_root, args.models, args.device, args.seed), args.output
    )


def filter_data(args):
    source = Manifest(args.input, args.audio_root)

    def selected():
        for row in source.rows:
            if not args.min_seconds <= float(row["duration"]) <= args.max_seconds:
                continue
            ids = source.ids(row)
            if len(ids) <= args.max_tokens:
                yield {**row, "phoneme_ids": ids}

    pack(selected(), args.output, args.audio_root)


def atomic_npy(path, value, verify_existing=False, resume=False):
    path = Path(path)
    if path.exists():
        if not (verify_existing or resume) or not np.array_equal(
            np.load(path, allow_pickle=False), value
        ):
            raise FileExistsError(
                f"Existing sidecar differs or overwrite was not authorized: {path}"
            )
        return
    if verify_existing:
        raise FileNotFoundError(f"Verification requires existing sidecar: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    temporary.replace(path)


@torch.inference_mode()
def extract_features(args):
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    bundle = ModelBundle(args.models)
    codec = bundle.load("vocoder", args.device).requires_grad_(False)
    s2 = bundle.load("s2", args.device).requires_grad_(False)
    hubert = bundle.load("hubert", args.device)
    with bundle.load("speaker", args.device) as speaker:
        data = Manifest(args.input, args.audio_root)
        for index, row in enumerate(data.rows):
            path, ids = data.audio_path(row), data.ids(row)
            wave32 = load_audio(path, 32000).to(args.device)
            frames = math.ceil(wave32.size(-1) / 640)
            if frames < 25 or frames > args.max_frames or len(ids) > 512:
                raise ValueError(
                    f"Row {index} is outside the training length budget; filter before extraction"
                )
            with torch.autocast(torch.device(args.device).type, enabled=False):
                latent = codec.encode(fit_length(wave32, frames * 640)[None])[0]
                wave16 = fit_length(load_audio(path, 16000).to(args.device), frames * 320)
                lengths = torch.tensor([frames * 320], device=args.device)
                frame_lengths = lengths.new_tensor([frames])
                embedding = speaker(wave16, lengths)[0]
                semantic = hubert.to_latent_frames(wave16, lengths, frame_lengths)
            durations = torch.tensor([row["dur"]], device=args.device)
            if (
                durations.size(1) != len(ids)
                or (durations < 0).any()
                or not 0 < int(durations.sum()) <= frames
            ):
                raise ValueError("Durations and phoneme IDs differ")
            with torch.autocast(
                torch.device(args.device).type,
                dtype=torch.bfloat16,
                enabled=torch.device(args.device).type == "cuda",
            ):
                pooled, _ = s2.conditioner.pool_ssl_to_phonemes(
                    semantic, durations, lengths.new_tensor([len(ids)]), frame_lengths
                )
                code = ScalarQuantize9.apply(
                    s2.conditioner.text_conditioner.prosody_down(pooled).tanh()
                )[0]
            if any(not torch.isfinite(value).all() for value in (latent, code, embedding)):
                raise FloatingPointError("Non-finite extracted feature")
            for suffix, value in [
                (".latent.npy", (latent.float() * 9).round().to(torch.int8)),
                (".code.npy", (code.float() * 9).round().to(torch.int8)),
                (".spk.npy", embedding.float()),
            ]:
                if not torch.isfinite(value).all():
                    raise FloatingPointError("Non-finite extracted feature")
                atomic_npy(
                    str(path) + suffix, value.cpu().numpy(), args.verify_existing, args.resume
                )
            print(f"features {index + 1}/{len(data)}", flush=True)
    write_json(
        Path(args.input).with_suffix(".features.json"),
        {
            "format_version": 1,
            "prosody_fingerprint": tensor_fingerprint(
                s2.conditioner.text_conditioner.prosody_down.state_dict()
            ),
            "components": bundle.manifest["components"],
            "rows": len(data),
            "precision": "bf16" if torch.device(args.device).type == "cuda" else "fp32",
            "latent_format": "int8[32,frames]/9",
            "code_format": "int8[16,tokens]/9",
            "speaker_format": "float32[256]",
            "latent_hz": 50,
        },
    )
