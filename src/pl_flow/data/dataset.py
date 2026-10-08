"""Stage-specific views over one manifest; no retry-to-random-row fallbacks."""

import random

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .audio import fit_length, load_audio
from .manifest import Manifest


def sidecar(path, suffix, shape, dtype):
    value = np.load(str(path) + suffix, allow_pickle=False)
    if (
        value.dtype != dtype
        or value.ndim != len(shape)
        or any(
            expected is not None and actual != expected
            for actual, expected in zip(value.shape, shape)
        )
        or not np.isfinite(value).all()
    ):
        raise ValueError(f"Invalid {suffix} sidecar for {path}")
    if dtype == np.int8 and (value.size == 0 or value.min() < -9 or value.max() > 9):
        raise ValueError("Scalar code values must lie in [-9, 9]")
    value = torch.from_numpy(value.copy())
    return value.float() / 9 if dtype == np.int8 else value


def select_prompt(durations, frames, config, rng):
    if rng.random() < config["dropout"]:
        return 0, 0
    boundaries = durations.cumsum(0)[:-1]
    upper = min(round(config["max_seconds"] * 50), int(frames * config["max_fraction"]), frames - 1)
    candidates = (
        torch.nonzero((boundaries >= round(config["min_seconds"] * 50)) & (boundaries <= upper))
        .flatten()
        .tolist()
    )
    if not candidates:
        return 0, 0
    token = rng.choice(candidates)
    return token + 1, int(boundaries[token])


class StageDataset(Dataset):
    def __init__(
        self,
        manifest,
        stage,
        *,
        audio_root=None,
        prosody_mode="frozen",
        prompt=None,
        seed=1234,
        sample_size=40960,
        max_frames=1000,
        max_tokens=512,
    ):
        if stage not in {"s1", "s2", "aligner", "vocoder"}:
            raise ValueError("Unknown training stage")
        self.manifest = Manifest(manifest, audio_root)
        self.stage, self.prosody_mode, self.prompt = stage, prosody_mode, prompt
        self.seed, self.sample_size = seed, sample_size
        self.max_frames, self.max_tokens = max_frames, max_tokens

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, key):
        # Epoch travels with each index: worker prefetch cannot alter resume RNG.
        epoch, index = key if isinstance(key, tuple) else (0, int(key))
        rng = random.Random((self.seed + epoch) * len(self) + index)
        row = self.manifest.rows[index]
        path = self.manifest.audio_path(row)
        if self.stage == "vocoder":
            wave = load_audio(path, 32000)
            # Match the original two-stage crop, then phase augmentation.
            crop_window = 2 * self.sample_size
            if wave.size(-1) > crop_window:
                offset = int(rng.random() * (wave.size(-1) - crop_window))
                wave = wave[..., offset : offset + crop_window]
            start = rng.randint(0, max(0, wave.size(-1) - self.sample_size))
            wave = fit_length(wave[..., start:], self.sample_size)
            return {"wave": -wave if rng.random() < 0.5 else wave}
        ids = torch.tensor(self.manifest.ids(row, mixed=self.stage == "aligner"), dtype=torch.long)
        if len(ids) > self.max_tokens and self.stage == "s1":
            raise ValueError("S1 sample exceeds max_tokens; filter before training")
        if self.stage == "aligner":
            return {"text": ids, "wave": load_audio(path, 16000)[0]}
        speaker = sidecar(path, ".spk.npy", (256,), np.float32)
        if self.stage == "s1":
            code = sidecar(path, ".code.npy", (16, len(ids)), np.int8)
            return {"ids": ids, "code": code, "speaker": speaker}
        durations = torch.tensor(row["dur"], dtype=torch.long)
        latent = sidecar(path, ".latent.npy", (32, None), np.int8)
        frames = latent.size(1)
        if (
            len(durations) != len(ids)
            or (durations < 0).any()
            or not 0 < int(durations.sum()) <= frames
        ):
            raise ValueError("Raw phoneme durations do not align with the acoustic latent")
        if not 25 <= frames <= self.max_frames:
            raise ValueError("S2 sample outside configured frame limits; filter before training")
        tokens, prefix = select_prompt(durations, frames, self.prompt, rng)
        item = {
            "text": ids,
            "durations": durations,
            "targets": latent,
            "speakers": speaker,
            "prompt_tokens": tokens,
            "prompt_lengths": prefix,
        }
        if self.prosody_mode == "frozen":
            item["codes"] = sidecar(path, ".code.npy", (16, len(ids)), np.int8)
        elif self.prosody_mode == "trainable":
            item["wave"] = fit_length(load_audio(path, 16000), frames * 320)[0]
        else:
            raise ValueError("Unknown prosody mode")
        return item


def collate(items):
    if set(items[0]) == {"wave"}:
        return {"wave": torch.stack([item["wave"] for item in items])}
    if "targets" in items[0]:
        items = sorted(items, key=lambda item: item["targets"].size(1), reverse=True)
    elif "text" in items[0]:
        items = sorted(items, key=lambda item: len(item["text"]), reverse=True)
    result = {}
    time_last = {"code", "codes", "targets"}
    sequences = {"ids", "text", "durations", "wave"}
    for key in items[0]:
        values = [item[key] for item in items]
        if key in time_last:
            result[key] = pad_sequence([value.T for value in values], batch_first=True).transpose(
                1, 2
            )
        elif key in sequences:
            result[key] = pad_sequence(values, batch_first=True)
        elif torch.is_tensor(values[0]):
            result[key] = torch.stack(values)
        else:
            result[key] = torch.tensor(values)
    for field, length_key in [
        ("ids", "lengths"),
        ("text", "text_lengths"),
        ("wave", "wave_lengths"),
        ("targets", "target_lengths"),
    ]:
        if field in result:
            axis = -1 if field == "targets" else 0
            result[length_key] = torch.tensor([item[field].size(axis) for item in items])
    return result
