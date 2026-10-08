"""Independent FP32 EMA, using PyTorch's official averaged-model primitive."""

import math

import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn


class ModelEMA:
    def __init__(self, model, decay=0.9999):
        if not math.isfinite(decay) or not 0 <= decay < 1:
            raise ValueError("EMA decay must be in [0, 1)")
        if any(p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError("Use FP32 master weights with autocast, not a half-cast model")
        self.decay = decay
        self.averaged = AveragedModel(
            model, multi_avg_fn=get_ema_multi_avg_fn(decay), use_buffers=False
        )
        self.averaged.requires_grad_(False).eval()
        self.averaged.update_parameters(
            model
        )  # Prime the snapshot before the first optimizer update.

    @property
    def updates(self):
        return int(self.averaged.n_averaged) - 1

    def update(self, model):
        self.averaged.update_parameters(model)

    def state_dict(self):
        return {"decay": self.decay, "use_buffers": False, "model": self.averaged.state_dict()}

    def load_state_dict(self, state, updates):
        if state["decay"] != self.decay or state["use_buffers"] is not False:
            raise ValueError("EMA decay/buffer policy changed on resume")
        if int(state["model"]["n_averaged"]) != updates + 1:
            raise ValueError("EMA and optimizer update counters disagree")
        self.averaged.load_state_dict(state["model"], strict=True)
