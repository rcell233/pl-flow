"""32 kHz mono, 32-channel, 50 Hz SQ-GAN audio codec."""

from torch import nn

from .stft import STFT2DDecoder, STFT2DEncoder


class SQCodec(nn.Module):
    sample_rate = 32000
    downsampling_ratio = 640
    latent_dim = 32

    def __init__(self, config):
        super().__init__()
        self.encoder = STFT2DEncoder(**config["encoder"])
        self.decoder = STFT2DDecoder(**config["decoder"])

    def encode(self, audio):
        return self.encoder(audio)

    def decode(self, latent):
        return self.decoder(latent)

    def forward(self, audio):
        return self.decode(self.encode(audio))
