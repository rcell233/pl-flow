import copy

import pytest
import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from pl_flow.training.engine import Engine
from pl_flow.training.state import atomic_save


def make_engine(state=None, precision="fp32"):
    model = nn.Sequential(weight_norm(nn.Linear(3, 4)), nn.Tanh(), nn.Linear(4, 2))
    if state is not None:
        model.load_state_dict(state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, fused=False)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.97)
    return Engine(model, optimizer, scheduler, precision=precision, ema_decay=0.9)


def objective(model, batch):
    return {"loss": (model(batch["x"]) - batch["y"]).square().mean()}


def batch(count=4):
    return {"x": torch.randn(count, 3), "y": torch.randn(count, 2)}


def test_accumulation_only_updates_ema_once():
    torch.manual_seed(3)
    a = make_engine()
    b = make_engine(a.model.state_dict())
    data = batch(6)
    a.train_step([data], objective)
    b.train_step(
        [{k: v[:2] for k, v in data.items()}, {k: v[2:] for k, v in data.items()}], objective
    )
    assert a.step == b.step == a.ema.updates == b.ema.updates == 1
    for key, value in a.model.state_dict().items():
        torch.testing.assert_close(value, b.model.state_dict()[key], atol=1e-7, rtol=1e-6)


def test_ema_initialization_independent_and_first_update():
    engine = make_engine()
    before = copy.deepcopy(engine.model.state_dict())
    engine.train_step([batch()], objective)
    for name, parameter in engine.model.named_parameters():
        shadow = engine.ema.averaged.module.get_parameter(name)
        assert parameter.requires_grad and not shadow.requires_grad
        assert parameter.data_ptr() != shadow.data_ptr()
        torch.testing.assert_close(shadow, before[name].lerp(parameter.detach(), 0.1))


def test_checkpoint_resume_exact(tmp_path):
    torch.manual_seed(12)
    a = make_engine()
    a.train_step([batch()], objective)
    a.epoch, a.offset = 4, 17
    path = tmp_path / "last.pt"
    atomic_save(path, a.state_dict())
    a.train_step([batch()], objective)
    b = make_engine()
    b.load_state_dict(torch.load(path, weights_only=True))
    assert (b.epoch, b.offset) == (4, 17)
    b.train_step([batch()], objective)
    assert b.step == a.step == 2
    for group in ("model",):
        for key, value in a.state_dict()[group].items():
            assert torch.equal(value, b.state_dict()[group][key])
    for key, value in a.ema.averaged.state_dict().items():
        assert torch.equal(value, b.ema.averaged.state_dict()[key])
    assert a.scheduler.get_last_lr() == b.scheduler.get_last_lr()


def test_nonfinite_loss_never_steps():
    engine = make_engine()
    before = copy.deepcopy(engine.model.state_dict())
    with pytest.raises(FloatingPointError):
        engine.train_step([batch()], lambda m, b: {"loss": m(b["x"]).mean() * float("nan")})
    assert engine.step == engine.ema.updates == 0
    for key, value in before.items():
        assert torch.equal(value, engine.model.state_dict()[key])


def test_amp_skipped_optimizer_does_not_advance_ema_or_scheduler():
    engine = make_engine(precision="fp16")
    lr = engine.scheduler.get_last_lr()

    class Overflow(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value.mean()

        @staticmethod
        def backward(ctx, grad):
            return grad.new_full((4, 2), float("inf"))

    result = engine.train_step([batch()], lambda m, b: {"loss": Overflow.apply(m(b["x"]))})
    assert not result["updated"]
    assert engine.step == engine.ema.updates == 0
    assert lr == engine.scheduler.get_last_lr()


def test_corrupt_ema_count_rejected():
    a, b = make_engine(), make_engine()
    state = copy.deepcopy(a.state_dict())
    state["ema"]["model"]["n_averaged"] += 1
    with pytest.raises(ValueError, match="counters"):
        b.load_state_dict(state)
