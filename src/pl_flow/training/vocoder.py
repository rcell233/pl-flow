"""Alternating generator/discriminator updates for the SQ codec."""

import torch

from pl_flow.vocoder.loss import VocoderLoss, build_discriminator

from .engine import Engine
from .state import restore_rng, rng_state


def inverse_scheduler(optimizer, inv_gamma=200000, power=0.5, warmup=0.999):
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: (1 + step / inv_gamma) ** (-power) * (1 - warmup ** (step + 1))
    )


class VocoderEngine:
    def __init__(self, codec, config):
        from ema_pytorch import EMA

        self.model, self.config = codec, config
        self.device = next(codec.parameters()).device
        self.discriminator = build_discriminator(config["loss"]).to(self.device)
        self.loss = VocoderLoss(config["loss"]).to(self.device)
        settings = {
            "precision": config["precision"],
            "gradient_clip": config.get("gradient_clip", 0),
            "ema_decay": None,
        }

        def engine(model, lr):
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, betas=(0.8, 0.99), weight_decay=0.001, fused=False
            )
            return Engine(model, optimizer, inverse_scheduler(optimizer), **settings)

        self.generator = engine(codec, config["learning_rate"])
        self.adversary = engine(self.discriminator, config["discriminator_learning_rate"])
        # Preserve the codec's original warmup schedule and pre-generator-update timing.
        self.ema = EMA(
            codec,
            beta=0.9999,
            power=0.75,
            update_every=1,
            update_after_step=1,
            include_online_model=False,
        )
        self.ema.ema_model.eval()
        self.step = self.epoch = self.offset = 0

    def train_step(self, batches):
        generator = self.step % 2 == 0
        if generator:
            self.discriminator.requires_grad_(False)

            def objective(model, batch):
                return self.loss(self.discriminator, batch["wave"], model(batch["wave"]), True)

            # Take the pre-update snapshot only if the optimizer actually steps.
            handle = self.generator.optimizer.register_step_pre_hook(
                lambda *args: self.ema.update()
            )
            try:
                result = self.generator.train_step(batches, objective)
            finally:
                handle.remove()
                self.discriminator.requires_grad_(True)
        else:

            def objective(model, batch):
                with torch.no_grad():
                    fake = self.model(batch["wave"])
                return self.loss(model, batch["wave"], fake, False)

            result = self.adversary.train_step(batches, objective)
        if result["updated"]:
            self.step += 1
        self.offset += len(batches)
        return {**result, "step": self.step, "generator_update": generator}

    def state_dict(self):
        return {
            "format_version": 1,
            "stage": "vocoder",
            "generator": self.generator.state_dict(),
            "adversary": self.adversary.state_dict(),
            "ema": self.ema.state_dict(),
            "loss": self.loss.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
            "offset": self.offset,
            "rng": rng_state(),
        }

    def load_state_dict(self, state):
        if state.get("format_version") != 1 or state.get("stage") != "vocoder":
            raise ValueError("Not a vocoder training checkpoint")
        self.generator.load_state_dict(state["generator"])
        self.adversary.load_state_dict(state["adversary"])
        self.ema.load_state_dict(state["ema"], strict=True)
        self.loss.load_state_dict(state["loss"], strict=True)
        self.step, self.epoch, self.offset = state["step"], state["epoch"], state["offset"]
        if self.step != self.generator.step + self.adversary.step:
            raise ValueError("GAN update counters disagree")
        restore_rng(state["rng"])
