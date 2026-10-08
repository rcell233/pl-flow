"""Online two-stage synthesis, with length-safe phoneme segmentation."""

import hashlib

import torch

from pl_flow.checkpoint import ModelBundle
from pl_flow.data.manifest import LANGUAGES
from pl_flow.text.frontend import KokoroG2P

from .reference import ReferenceEncoder, pack_prompt_conditions


def stable_seed(seed, key):
    return (seed + int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "little")) % (2**31)


def split_ids(ids):
    inner = ids[1:-1]
    if len(inner) < 2:
        raise ValueError("Cannot split this over-budget phoneme sequence further")
    midpoint = len(inner) // 2
    lo, hi = max(1, len(inner) // 4), min(len(inner) - 1, 3 * len(inner) // 4)
    cuts = [i for i in range(lo, hi + 1) if inner[i - 1] in {1, 2, 3, 4, 5, 6, 9, 10}]
    if not cuts:
        cuts = [i for i in range(lo, hi + 1) if inner[i - 1] == 16]
    cut = min(cuts, key=lambda i: abs(i - midpoint)) if cuts else midpoint
    return [[0] + inner[:cut] + [0], [0] + inner[cut:] + [0]]


class Synthesizer:
    sample_rate = 32000

    def __init__(self, bundle, device="cuda", **sampling):
        self.bundle = bundle if isinstance(bundle, ModelBundle) else ModelBundle(bundle)
        if (
            self.bundle.manifest["s1_prosody_fingerprint"]
            != self.bundle.manifest["prosody_fingerprint"]
        ):
            raise ValueError(
                "S1 and S2 use different prosody spaces; regenerate codes and retrain S1"
            )
        self.device = torch.device(device)
        self.settings = {**self.bundle.manifest["sampling"], **sampling}
        self.s1 = self.bundle.load("s1", device).requires_grad_(False)
        self.s2 = self.bundle.load("s2", device).requires_grad_(False)
        self.codec = self.bundle.load("vocoder", device).requires_grad_(False)
        self.s2.conditioner.max_infer_frames = self.settings["max_frames"]
        self.references = ReferenceEncoder(self.bundle, self.s2, self.codec, device)
        self.g2p = None

    def close(self):
        """Release the speaker process; subsequent synthesis can restart it."""
        self.references.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def autocast(self):
        return torch.autocast(
            self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"
        )

    def phonemes(self, text, language):
        if self.g2p is None:
            self.g2p = KokoroG2P(lang_codes=["a", "z", "j"])
        ids = self.g2p(text, lang_code=LANGUAGES[language])[1]
        if len(ids) <= 2:
            raise ValueError("Text contains no supported phonemes")
        return ids

    @torch.inference_mode()
    def condition_reference(self, reference):
        ref = {k: v.to(self.device) for k, v in reference.items() if torch.is_tensor(v)}
        text = ref["ids"][None]
        lengths = text.new_tensor([text.size(1)])
        with self.autocast():
            out = self.s2.conditioner.text_conditioner.forward_from_prosody(
                text,
                lengths,
                ref["codes"].to(torch.bfloat16 if self.device.type == "cuda" else torch.float32)[
                    None
                ]
                / 9,
            )
            ref["condition"] = torch.repeat_interleave(
                out["token_condition"][0].T, ref["durations"], dim=0
            )
        ref["speaker"] = ref["speaker"].reshape(1, -1)
        return ref

    @torch.inference_mode()
    def reference(self, audio, transcript, language="zh", seed=1234):
        ids = self.phonemes(transcript, language)
        self.references.load_features()
        devices = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            # The aligner retains its trained token-corruption operation in eval.
            # Seed after lazy model construction so cold and warm calls agree.
            torch.manual_seed(seed)
            return self.condition_reference(self.references.prepare(audio, ids))

    def prompt_length(self, ref):
        return min(len(ref["ids"]), self.s1.prompt_config["max_tokens"])

    def sample_s1(self, ids, lengths, speakers, refs):
        prefix = lengths.new_tensor([self.prompt_length(r) for r in refs])
        prompt_ids = ids.new_zeros(len(refs), int(prefix.max()))
        prompt_codes = speakers.new_zeros(len(refs), 16, int(prefix.max()))
        for b, (ref, count) in enumerate(zip(refs, prefix.tolist())):
            prompt_ids[b, :count] = ref["ids"][:count]
            prompt_codes[b, :, :count] = ref["codes"][:, :count].float() / 9
        self.s1.cfm.estimator.setup_caches(2 * len(refs), self.s1.max_tokens + 2)
        return self.s1.sample(
            ids,
            lengths,
            speakers,
            self.settings["s1_steps"],
            self.settings["s1_cfg"],
            self.settings["temperature"],
            prompt_ids=prompt_ids,
            prompt_code=prompt_codes,
            prompt_lens=prefix,
        )

    @torch.inference_mode()
    def synthesize_prepared(self, sequences, references, keys, seed=1234, depth=0):
        if not sequences or not (len(sequences) == len(references) == len(keys)):
            raise ValueError("Sequences, references and keys must have the same nonzero length")
        if depth > 12:
            raise ValueError("Exceeded the segmentation depth limit")
        if any(
            len(ids) + self.prompt_length(ref) > self.s1.max_tokens
            for ids, ref in zip(sequences, references)
        ):
            if len(sequences) > 1:
                return [
                    self.synthesize_prepared([ids], [ref], [key], seed, depth)[0]
                    for ids, ref, key in zip(sequences, references, keys)
                ]
            return [
                self._split(ids, ref, key, seed, depth)
                for ids, ref, key in zip(sequences, references, keys)
            ]
        lengths = torch.tensor([len(ids) for ids in sequences], device=self.device)
        text = torch.zeros(len(sequences), int(lengths.max()), dtype=torch.long, device=self.device)
        for b, ids in enumerate(sequences):
            text[b, : len(ids)] = text.new_tensor(ids)
        speakers = torch.cat([r["speaker"] for r in references])
        devices = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(stable_seed(seed, "|".join(keys)))
            with self.autocast():
                codes, raw = self.sample_s1(text, lengths, speakers, references)
                if not torch.isfinite(raw).all():
                    raise FloatingPointError("Non-finite S1 sample")
                out = self.s2.conditioner.text_conditioner.forward_from_prosody(
                    text,
                    lengths,
                    codes.to(torch.bfloat16 if self.device.type == "cuda" else torch.float32) / 9,
                )
                durations = [
                    out["duration_logits"][b, : len(ids)].argmax(-1)
                    for b, ids in enumerate(sequences)
                ]
            if any(
                int(d.sum()) + r["latent"].size(-1) > self.settings["max_frames"]
                for d, r in zip(durations, references)
            ):
                if len(sequences) > 1:
                    return [
                        self.synthesize_prepared([ids], [ref], [key], seed, depth)[0]
                        for ids, ref, key in zip(sequences, references, keys)
                    ]
                return [self._split(sequences[0], references[0], keys[0], seed, depth)]
            if any(int(d.sum()) <= 0 for d in durations):
                raise ValueError("S2 predicted zero total duration")
            targets = [
                torch.repeat_interleave(out["token_condition"][b, :, : len(d)].T, d, dim=0)
                for b, d in enumerate(durations)
            ]
            condition, total, prompt, prefixes = pack_prompt_conditions(references, targets)
            self.s2.cfm.estimator.setup_caches(2 * len(sequences), self.settings["max_frames"] + 2)
            with self.autocast():
                latent = self.s2.cfm.inference(
                    condition,
                    total,
                    speakers,
                    None,
                    self.settings["s2_steps"],
                    self.settings["temperature"],
                    self.settings["s2_cfg"],
                    prompt=prompt,
                    prompt_lens=prefixes,
                )
                if not torch.isfinite(latent).all():
                    raise FloatingPointError("Non-finite S2 sample")
                latent = (latent * 9).round().clamp(-9, 9) / 9
        results = []
        for b, ids in enumerate(sequences):
            start, frames = int(prefixes[b]), len(targets[b])
            wave = (
                self.codec.decode(latent[b : b + 1, :, start : start + frames].float())[
                    0, :, : frames * 640
                ]
                .float()
                .cpu()
                .clamp(-1, 1)
            )
            if not torch.isfinite(wave).all() or wave.numel() == 0:
                raise FloatingPointError("Empty or non-finite synthesis")
            results.append(
                (
                    wave,
                    [
                        {
                            "phoneme_ids": ids,
                            "code": codes[b, :, : len(ids)].cpu(),
                            "durations": durations[b].cpu(),
                            "frames": frames,
                        }
                    ],
                )
            )
        return results

    def _split(self, ids, ref, key, seed, depth):
        parts = [
            self.synthesize_prepared([part], [ref], [f"{key}/{i}"], seed, depth + 1)[0]
            for i, part in enumerate(split_ids(ids))
        ]
        return torch.cat([wave for wave, _ in parts], -1), [
            chunk for _, chunks in parts for chunk in chunks
        ]

    def synthesize(
        self,
        text,
        reference_audio,
        reference_text,
        language="zh",
        reference_language=None,
        seed=1234,
        key="sample",
    ):
        torch.manual_seed(seed)
        ref = self.reference(reference_audio, reference_text, reference_language or language, seed)
        return self.synthesize_prepared([self.phonemes(text, language)], [ref], [key], seed)[0][0]
