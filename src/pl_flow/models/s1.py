"""Prompted phoneme-aligned prosody flow."""

import math

import torch
from torch import nn

from pl_flow.config import recursive_munch
from pl_flow.text.frontend import KOKORO_VOCAB

from .conditioner import ConvReluNorm, Encoder, sequence_mask
from .flow import CFM


class S1(nn.Module):
    def __init__(self, config, plbert):
        super().__init__()
        self.plbert = plbert
        self.plbert.requires_grad_(False).eval()
        if self.plbert.bert.config.vocab_size != 178 or self.plbert.vocab != KOKORO_VOCAB:
            raise ValueError("PLBert vocabulary must match the S2 frontend")
        self.embedding = nn.Embedding(178, 192)
        nn.init.normal_(self.embedding.weight, std=192 ** (-0.5))
        self.bert_projection = nn.Linear(768, 192)
        self.prenet = ConvReluNorm(192, 192, 192, kernel_size=5, n_layers=3, p_dropout=0.5)
        self.encoder = Encoder(192, 768, 2, 6, 3, 0.1)
        self.projection = nn.Conv1d(192, 32, 1)
        self.cfm = CFM(recursive_munch(config["model_params"]))
        self.max_tokens = int(config.get("max_tokens", 512))
        self.prompt_config = config.get("prompt")
        p = self.prompt_config
        if (
            not p
            or not 2 <= p["min_tokens"] <= p["max_tokens"] < self.max_tokens - 2
            or (not 0 < p["max_fraction"] < 1)
            or (not 0 <= p["dropout"] <= 1)
        ):
            raise ValueError("Invalid S1 phoneme prompt configuration")

    def train(self, mode=True):
        super().train(mode)
        self.plbert.eval()
        return self

    def condition(self, ids, lengths):
        mask = sequence_mask(lengths, ids.size(1))
        with torch.no_grad():
            bert = self.plbert(ids, attention_mask=mask.long())
        x = self.bert_projection(bert) + self.embedding(ids) * math.sqrt(192)
        x = x.transpose(1, 2)
        m = mask.unsqueeze(1).to(x.dtype)
        x = self.prenet(x * m, m)
        x = self.encoder(x, m)
        return (self.projection(x) * m).transpose(1, 2)

    def select_prompt_lengths(self, lengths, randomize=True):
        """Select token prefixes without durations; leave a phoneme plus EOS as target."""
        p = self.prompt_config
        upper = (lengths.float() * p["max_fraction"]).floor().long()
        upper = torch.minimum(upper.clamp_max(p["max_tokens"]), lengths - 2)
        if randomize:
            span = (upper - p["min_tokens"] + 1).clamp_min(1)
            selected = (torch.rand(lengths.shape, device=lengths.device) * span).long() + p[
                "min_tokens"
            ]
            selected = selected.masked_fill(
                torch.rand(lengths.shape, device=lengths.device) < p["dropout"], 0
            )
        else:
            selected = upper
        return selected.masked_fill(upper < p["min_tokens"], 0)

    def condition_segments(self, ids, lengths, prompt_lens):
        """Encode each reference/target independently, then join on the token axis."""
        if (
            lengths.shape != (ids.size(0),)
            or prompt_lens.shape != lengths.shape
            or ((prompt_lens < 0) | (prompt_lens >= lengths) | (lengths > ids.size(1))).any()
            or (ids.size(1) > self.max_tokens)
        ):
            raise ValueError("Invalid S1 prompt/target lengths or context budget")
        segments = []
        for b, (length, cut) in enumerate(zip(lengths.tolist(), prompt_lens.tolist())):
            if cut:
                segments.append((b, 0, cut))
            segments.append((b, cut, length))
        segment_lengths = lengths.new_tensor([end - start for (_, start, end) in segments])
        segment_ids = ids.new_zeros(len(segments), int(segment_lengths.max()))
        for s, (b, start, end) in enumerate(segments):
            segment_ids[s, : end - start] = ids[b, start:end]
        encoded = self.condition(segment_ids, segment_lengths)
        condition = encoded.new_zeros(ids.size(0), ids.size(1), encoded.size(-1))
        for s, (b, start, end) in enumerate(segments):
            condition[b, start:end] = encoded[s, : end - start]
        return condition

    def forward(self, ids, lengths, speaker, code, prompt_lens=None):
        if prompt_lens is None:
            prompt_lens = self.select_prompt_lengths(lengths, randomize=self.training)
        condition = self.condition_segments(ids, lengths, prompt_lens)
        return self.cfm(code, lengths, condition, speaker, prompt_lens=prompt_lens)[0]

    @torch.inference_mode()
    def sample(
        self,
        ids,
        lengths,
        speaker,
        steps=8,
        cfg=2.1,
        temperature=1.0,
        *,
        prompt_ids=None,
        prompt_code=None,
        prompt_lens=None,
    ):
        """Return only target codes. Prompt codes use the training scale [-1, 1]."""
        (batch, width) = ids.shape
        if lengths.shape != (batch,) or ((lengths <= 0) | (lengths > width)).any():
            raise ValueError("Invalid S1 target lengths")
        if all((value is None for value in (prompt_ids, prompt_code, prompt_lens))):
            prompt_ids = ids.new_zeros(batch, 0)
            prompt_code = speaker.new_zeros(batch, self.cfm.in_channels, 0)
            prompt_lens = lengths.new_zeros(batch)
        if (
            any((value is None for value in (prompt_ids, prompt_code, prompt_lens)))
            or prompt_ids.ndim != 2
            or prompt_ids.size(0) != batch
            or (prompt_code.shape != (batch, self.cfm.in_channels, prompt_ids.size(1)))
            or (not prompt_code.is_floating_point())
            or (prompt_lens.shape != lengths.shape)
            or (
                (prompt_lens < 0)
                | (prompt_lens > prompt_ids.size(1))
                | (prompt_lens > self.prompt_config.get("max_tokens", 512))
            ).any()
        ):
            raise ValueError("S1 prompt IDs, normalized codes and lengths must align")
        total_lengths = prompt_lens + lengths
        if (total_lengths > self.max_tokens).any():
            raise ValueError("S1 reference plus target exceeds the token context budget")
        packed_ids = ids.new_zeros(batch, int(total_lengths.max()))
        prompt = prompt_code.new_zeros(batch, self.cfm.in_channels, int(prompt_lens.max()))
        for b, (cut, length) in enumerate(zip(prompt_lens.tolist(), lengths.tolist())):
            packed_ids[b, :cut] = prompt_ids[b, :cut]
            packed_ids[b, cut : cut + length] = ids[b, :length]
            prompt[b, :, :cut] = prompt_code[b, :, :cut]
        condition = self.condition_segments(packed_ids, total_lengths, prompt_lens)
        continuous = self.cfm.inference(
            condition,
            total_lengths,
            speaker,
            None,
            steps,
            temperature=temperature,
            inference_cfg_rate=cfg,
            prompt=prompt,
            prompt_lens=prompt_lens,
        )
        target = continuous.new_zeros(batch, self.cfm.in_channels, width)
        for b, (cut, length) in enumerate(zip(prompt_lens.tolist(), lengths.tolist())):
            target[b, :, :length] = continuous[b, :, cut : cut + length]
        return ((target.clamp(-1, 1) * 9).round().to(torch.int8), target)
