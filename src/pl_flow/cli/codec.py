"""Codec inference CLI using the same portable bundle and int8 latent format."""

import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from pl_flow.checkpoint import ModelBundle
from pl_flow.data.audio import fit_length, load_audio


@torch.inference_mode()
def run_codec(args):
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    model = ModelBundle(args.models).load("vocoder", args.device).requires_grad_(False)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.operation == "decode":
        codes = np.load(args.input, allow_pickle=False)
        if (
            codes.dtype != np.int8
            or codes.ndim != 2
            or codes.shape[0] != 32
            or not codes.size
            or (np.abs(codes.astype(np.int16)) > 9).any()
        ):
            raise ValueError("Expected int8 [32, frames] codes in [-9, 9]")
        latent = torch.from_numpy(codes).to(args.device).float()[None] / 9
        wave = model.decode(latent)[0]
    else:
        audio = load_audio(args.input, 32000).to(args.device)
        frames = math.ceil(audio.size(-1) / 640)
        latent = model.encode(fit_length(audio, frames * 640)[None])
        if not torch.isfinite(latent).all():
            raise FloatingPointError("Non-finite codec latent")
        if args.operation == "encode":
            with Path(args.output).open("wb") as stream:
                np.save(
                    stream,
                    (latent[0].float() * 9).round().to(torch.int8).cpu().numpy(),
                    allow_pickle=False,
                )
            return
        wave = model.decode(latent)[0, :, : audio.size(-1)]
    if not torch.isfinite(wave).all():
        raise FloatingPointError("Non-finite decoded waveform")
    sf.write(args.output, wave.float().cpu().T.numpy(), 32000)
