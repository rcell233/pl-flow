"""Native PyTorch optimizer-step engine: accumulation, AMP, EMA and exact resume."""

import torch

from .ema import ModelEMA
from .state import restore_rng, rng_state


class Engine:
    def __init__(
        self,
        model,
        optimizer,
        scheduler=None,
        *,
        ema_decay=0.9999,
        precision="bf16",
        gradient_clip=10.0,
    ):
        if any(group.get("fused") for group in optimizer.param_groups):
            raise ValueError("Fused optimizers are not supported by the successful-step hook")
        optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
        if optimized != {id(p) for p in model.parameters() if p.requires_grad}:
            raise ValueError("Optimizer must contain exactly the trainable model parameters")
        if precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("Precision must be fp32, bf16 or fp16")
        self.model, self.optimizer, self.scheduler = model, optimizer, scheduler
        self.precision, self.gradient_clip = precision, gradient_clip
        self.device = next(model.parameters()).device
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=precision == "fp16")
        self.step, self.epoch, self.offset = 0, 0, 0
        self.ema = ModelEMA(model, ema_decay) if ema_decay is not None else None
        self.optimizer.register_step_post_hook(self._successful_update)
        self._accumulating = False

    def _successful_update(self, optimizer, args, kwargs):
        self.step += 1
        if self.ema is not None:
            self.ema.update(self.model)

    def train_step(self, batches, objective):
        if not batches:
            raise ValueError("An optimizer update needs at least one microbatch")
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        self._accumulating = True
        previous_step = self.step
        totals = {}
        counts = [len(next(v for v in b.values() if torch.is_tensor(v))) for b in batches]
        count = sum(counts)
        try:
            for batch, size in zip(batches, counts):
                with torch.autocast(
                    self.device.type,
                    dtype=torch.bfloat16 if self.precision == "bf16" else torch.float16,
                    enabled=self.precision != "fp32",
                ):
                    values = objective(self.model, batch)
                    loss = values["loss"]
                if loss.ndim != 0 or not torch.isfinite(loss):
                    raise FloatingPointError(
                        "Non-finite or non-scalar training loss; no optimizer update"
                    )
                self.scaler.scale(loss * (size / count)).backward()
                for key, value in values.items():
                    if torch.is_tensor(value) and value.ndim == 0:
                        totals[key] = totals.get(key, 0.0) + float(value.detach()) * size / count
            self.scaler.unscale_(self.optimizer)
            parameters = [p for p in self.model.parameters() if p.requires_grad]
            gradient = torch.nn.utils.clip_grad_norm_(
                parameters,
                self.gradient_clip or float("inf"),
                error_if_nonfinite=not self.scaler.is_enabled(),
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            successful = self.step != previous_step
            if successful and self.scheduler is not None:
                self.scheduler.step()
            self.offset += len(batches)
            return {
                **totals,
                "grad_norm": float(gradient),
                "updated": successful,
                "step": self.step,
            }
        finally:
            self._accumulating = False
            self.optimizer.zero_grad(set_to_none=True)

    def state_dict(self):
        if self._accumulating:
            raise RuntimeError("Checkpointing during gradient accumulation is not supported")
        return {
            "format_version": 1,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "parameter_names": [n for n, p in self.model.named_parameters() if p.requires_grad],
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler": self.scaler.state_dict(),
            "precision": self.precision,
            "ema": self.ema.state_dict() if self.ema else None,
            "step": self.step,
            "epoch": self.epoch,
            "offset": self.offset,
            "rng": rng_state(),
        }

    def load_state_dict(self, state):
        names = [n for n, p in self.model.named_parameters() if p.requires_grad]
        if state.get("format_version") != 1 or state["parameter_names"] != names:
            raise ValueError("Training checkpoint schema or trainable parameters changed")
        if state["precision"] != self.precision or (state["ema"] is None) != (self.ema is None):
            raise ValueError("Precision or EMA policy changed on resume")
        if (state["scheduler"] is None) != (self.scheduler is None):
            raise ValueError("Scheduler policy changed on resume")
        self.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.step, self.epoch, self.offset = state["step"], state["epoch"], state["offset"]
        if self.ema is not None:
            self.ema.load_state_dict(state["ema"], self.step)
        restore_rng(state["rng"])
