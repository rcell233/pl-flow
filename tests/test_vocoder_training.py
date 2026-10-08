import copy

import torch
from torch import nn

from pl_flow.training import vocoder


class ToyLoss(nn.Module):
    def forward(self, discriminator, real, fake, generator):
        if generator:
            return {"loss": (fake - real).square().mean() + discriminator(fake).square().mean()}
        return {
            "loss": (discriminator(real) - 1).square().mean() + discriminator(fake).square().mean()
        }


def test_gan_ema_only_updates_with_generator_and_resumes(monkeypatch):
    monkeypatch.setattr(vocoder, "build_discriminator", lambda config: nn.Linear(2, 2))
    monkeypatch.setattr(vocoder, "VocoderLoss", lambda config: ToyLoss())
    settings = {
        "loss": {},
        "learning_rate": 1e-3,
        "discriminator_learning_rate": 1e-3,
        "precision": "fp32",
    }
    engine = vocoder.VocoderEngine(nn.Linear(2, 2), settings)
    data = [{"wave": torch.randn(3, 2)}]
    engine.train_step(data)
    assert int(engine.ema.step) == 1
    engine.train_step(data)
    assert engine.step == 2 and engine.generator.step == engine.adversary.step == 1
    assert int(engine.ema.step) == 1
    state = copy.deepcopy(engine.state_dict())
    engine.train_step(data)
    restored = vocoder.VocoderEngine(nn.Linear(2, 2), settings)
    restored.load_state_dict(state)
    restored.train_step(data)
    for name, value in engine.model.state_dict().items():
        torch.testing.assert_close(value, restored.model.state_dict()[name], atol=0, rtol=0)
    assert restored.step == 3 and int(restored.ema.step) == 2
