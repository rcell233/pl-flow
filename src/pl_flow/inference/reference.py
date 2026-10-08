"""Reference-only audio alignment, prosody, speaker and codec features."""

import math

import torch
from torch.nn import functional as F

from pl_flow.alignment.objective import align
from pl_flow.data.audio import load_audio
from pl_flow.models.conditioner import LatentTTSConditioner, ScalarQuantize9


class ReferenceEncoder:
    def __init__(self, bundle, s2, codec, device):
        self.bundle, self.s2, self.codec = bundle, s2, codec
        self.device = torch.device(device)
        self.hubert = self.aligner = self.speaker = None

    def load_features(self):
        if self.hubert is None:
            self.hubert = self.bundle.load("hubert", self.device).requires_grad_(False)
        if self.aligner is None:
            self.aligner = self.bundle.load("aligner", self.device).requires_grad_(False)
        if self.speaker is None:
            self.speaker = self.bundle.load("speaker", self.device).requires_grad_(False)

    def close(self):
        if self.speaker is not None:
            self.speaker.close()

    @torch.inference_mode()
    def prepare(self, audio_path, ids):
        if len(ids) <= 2:
            raise ValueError("Reference transcript must contain phonemes")
        self.load_features()
        audio16 = load_audio(audio_path, 16000).to(self.device)
        audio32 = load_audio(audio_path, 32000).to(self.device)
        lengths16 = torch.tensor([audio16.size(-1)], device=self.device)
        text = torch.tensor([ids], device=self.device)
        text_lengths = torch.tensor([len(ids)], device=self.device)
        with torch.autocast(self.device.type, enabled=False):
            speaker = self.speaker(audio16, lengths16)[0]
            durations = align(self.aligner, self.hubert, audio16, lengths16, text, text_lengths)[0]
            frames = math.ceil(audio32.size(-1) / 640)
            padded = F.pad(audio32, (0, frames * 640 - audio32.size(-1)))
            latent = self.codec.encode(padded[None])[0, :, :frames]
            frame_lengths = text_lengths.new_tensor([frames])
            aligned = LatentTTSConditioner.fit_durations_to_targets(
                durations[None], text_lengths, frame_lengths
            )[0]
            fitted = F.pad(audio16, (0, max(0, frames * 320 - audio16.size(-1))))[:, : frames * 320]
            semantic = self.hubert.to_latent_frames(
                fitted, text_lengths.new_tensor([frames * 320]), frame_lengths
            )
        with torch.autocast(
            self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"
        ):
            pooled, _ = LatentTTSConditioner.pool_ssl_to_phonemes(
                semantic, durations[None], text_lengths, frame_lengths
            )
            code = ScalarQuantize9.apply(
                self.s2.conditioner.text_conditioner.prosody_down(pooled).tanh()
            )[0]
        cap = min(
            self.bundle.manifest["sampling"]["max_prompt_frames"],
            self.s2.conditioner.max_infer_frames - 1,
        )
        count = len(ids)
        if frames > cap:
            candidates = torch.nonzero(
                (aligned.cumsum(0) > 0) & (aligned.cumsum(0) <= cap)
            ).flatten()
            if not len(candidates):
                raise ValueError("Reference has no phoneme boundary within the prompt budget")
            count = int(candidates[-1]) + 1
            frames = int(aligned[:count].sum())
        result = {
            "ids": text[0, :count],
            "codes": (code[:, :count].float() * 9).round().to(torch.int8),
            "durations": aligned[:count],
            "pool_durations": durations[:count],
            "latent": latent[:, :frames].float(),
            "speaker": speaker.float(),
        }
        if any(not torch.isfinite(value).all() for value in result.values()):
            raise FloatingPointError("Non-finite reference features")
        return result


def pack_prompt_conditions(references, targets):
    if not targets or len(references) != len(targets):
        raise ValueError("One reference is required per target")
    device = targets[0].device
    prefix = torch.tensor([r["latent"].size(-1) for r in references], device=device)
    lengths = prefix + torch.tensor([len(t) for t in targets], device=device)
    condition = targets[0].new_zeros(len(targets), int(lengths.max()), targets[0].size(-1))
    prompt = targets[0].new_zeros(len(targets), 32, int(prefix.max()))
    for b, (reference, target) in enumerate(zip(references, targets)):
        n = int(prefix[b])
        if reference["condition"].shape != (n, target.size(-1)) or len(target) == 0:
            raise ValueError("Reference condition/latent mismatch or empty target")
        condition[b, :n] = reference["condition"]
        condition[b, n : int(lengths[b])] = target
        prompt[b, :, :n] = reference["latent"].to(device)
    return condition, lengths, prompt, prefix
