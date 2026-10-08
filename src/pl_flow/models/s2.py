"""Acoustic flow and phoneme conditioner with explicit prosody trainability."""

from torch import nn

from pl_flow.config import recursive_munch

from .conditioner import LatentTTSConditioner, ScalarQuantize9, sequence_mask
from .flow import CFM


class S2(nn.Module):
    def __init__(self, config, prosody_mode="frozen"):
        super().__init__()
        self.cfm = CFM(recursive_munch(config["model_params"]))
        self.conditioner = LatentTTSConditioner(config["conditioner"])
        if prosody_mode not in {"frozen", "trainable"}:
            raise ValueError("prosody_mode must be frozen or trainable")
        self.prosody_mode = prosody_mode
        self.conditioner.text_conditioner.prosody_down.requires_grad_(prosody_mode == "trainable")
        self.duration_loss_weight = config["conditioner"].get("duration_loss_weight", 1.0)

    def prosody_codes(self, semantic, durations, text_lengths, target_lengths):
        pooled, _ = self.conditioner.pool_ssl_to_phonemes(
            semantic, durations, text_lengths, target_lengths
        )
        mask = sequence_mask(text_lengths, durations.size(1)).unsqueeze(1).to(pooled)
        projected = self.conditioner.text_conditioner.prosody_down(pooled) * mask
        return ScalarQuantize9.apply(projected.tanh()) * mask

    def forward(self, batch, semantic=None):
        if self.prosody_mode == "trainable":
            if semantic is None or "codes" in batch:
                raise ValueError(
                    "Trainable prosody requires online HuBERT features, not cached codes"
                )
            codes = self.prosody_codes(
                semantic, batch["durations"], batch["text_lengths"], batch["target_lengths"]
            )
        else:
            if semantic is not None:
                raise ValueError("Frozen prosody must use pre-extracted codes")
            codes = batch["codes"]
        out = self.conditioner.forward_segments(
            batch["text"],
            batch["text_lengths"],
            codes,
            batch["durations"],
            batch["target_lengths"],
            batch["prompt_tokens"],
        )
        flow, _ = self.cfm(
            batch["targets"],
            batch["target_lengths"],
            out["condition"],
            batch["speakers"],
            prompt_lens=batch["prompt_lengths"],
        )
        return {
            "loss": flow + self.duration_loss_weight * out["duration_loss"],
            "flow_loss": flow,
            "duration_loss": out["duration_loss"],
            "duration_accuracy": out["duration_accuracy"],
        }
