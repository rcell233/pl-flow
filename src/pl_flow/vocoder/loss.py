# Uses auraloss (Apache-2.0), with an adapted spectral convergence term.
# Modified for PL-Flow: per-example reduction and reconstruction normalization.
# See licenses/Apache-2.0.txt and THIRD_PARTY.md.
"""SQ-GAN losses: least-squares discriminator, normalized feature matching, MR-STFT."""

import torch
from torch import nn

from .discriminator import MultiScaleSTFTDiscriminator


class PerSampleSpectralConvergence(nn.Module):
    def forward(self, real_magnitude, reconstructed_magnitude):
        numerator = torch.norm(reconstructed_magnitude - real_magnitude, p="fro", dim=[-1, -2])
        denominator = torch.norm(reconstructed_magnitude, p="fro", dim=[-1, -2])
        return (numerator / denominator).unsqueeze(-1).unsqueeze(-1)


class VocoderLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        import auraloss

        self.spectral = auraloss.freq.MultiResolutionSTFTLoss(
            sample_rate=32000, **config["spectral"]["config"]
        )
        for loss in self.spectral.stft_losses:
            loss.spectralconv = PerSampleSpectralConvergence()
        self.weights = config

    def forward(self, discriminator, real, fake, generator=True):
        width = min(real.size(-1), fake.size(-1))
        real, fake = real[..., :width], fake[..., :width]
        real_logits, real_features = discriminator(real)
        fake_logits, fake_features = discriminator(fake)
        if not generator:
            loss = sum(
                ((r - 1) ** 2).mean() + (f**2).mean() for r, f in zip(real_logits, fake_logits)
            )
            return {"loss": loss, "discriminator_loss": loss}
        adversarial = sum(((f - 1) ** 2).mean() for f in fake_logits)
        # The released trainer used the real-feature magnitude without an epsilon.
        matching = sum(
            (r - f).abs().mean() / r.abs().mean()
            for real_layers, fake_layers in zip(real_features, fake_features)
            for r, f in zip(real_layers, fake_layers)
        )
        with torch.autocast(real.device.type, enabled=False):
            # Training normalizes convergence by the reconstructed magnitude.
            spectral = self.spectral(real.float(), fake.float())
        weights = self.weights["discriminator"]["weights"]
        loss = (
            weights["adversarial"] * adversarial
            + weights["feature_matching"] * matching
            + self.weights["spectral"]["weights"]["mrstft"] * spectral
        )
        return {
            "loss": loss,
            "adversarial": adversarial,
            "feature_matching": matching,
            "spectral": spectral,
        }


def build_discriminator(config):
    return MultiScaleSTFTDiscriminator(in_channels=1, **config["discriminator"]["config"])
