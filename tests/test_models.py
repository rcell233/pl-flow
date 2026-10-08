import pytest
import torch

from pl_flow.inference.pipeline import split_ids
from pl_flow.models.conditioner import RotaryPositionalEmbeddings
from pl_flow.models.s2 import S2


def config():
    return {
        "model_params": {
            "style_encoder": {"dim": 256},
            "DiT": {
                "in_channels": 32,
                "content_dim": 32,
                "hidden_dim": 32,
                "num_heads": 2,
                "depth": 3,
                "class_dropout_prob": 0.0,
            },
            "wavenet": {
                "hidden_dim": 32,
                "kernel_size": 3,
                "dilation_rate": 1,
                "num_layers": 2,
                "p_dropout": 0.0,
            },
        },
        "conditioner": {
            "n_vocab": 178,
            "hidden_channels": 32,
            "filter_channels": 64,
            "n_layers": 2,
            "p_dropout": 0.0,
        },
    }


def example():
    return {
        "text": torch.tensor([[0, 50, 83, 54, 0]]),
        "text_lengths": torch.tensor([5]),
        "durations": torch.tensor([[2, 4, 3, 4, 2]]),
        "targets": torch.randn(1, 32, 17),
        "target_lengths": torch.tensor([17]),
        "prompt_tokens": torch.tensor([2]),
        "prompt_lengths": torch.tensor([6]),
        "speakers": torch.randn(1, 256),
    }


def test_trainable_prosody_has_gradient_and_frozen_does_not():
    for mode in ("trainable", "frozen"):
        model = S2(config(), mode)
        model.cfm.estimator.setup_caches(2, 32)
        batch = example()
        semantic = torch.randn(1, 768, 17) if mode == "trainable" else None
        if mode == "frozen":
            batch["codes"] = torch.zeros(1, 16, 5)
        result = model(batch, semantic)
        result["loss"].backward()
        projection = model.conditioner.text_conditioner.prosody_down
        if mode == "trainable":
            assert projection.weight.grad is not None
            assert projection.weight.grad.abs().sum() > 0
        else:
            assert not projection.weight.requires_grad and projection.weight.grad is None


def test_trainable_prosody_rejects_stale_codes():
    model = S2(config(), "trainable")
    batch = example()
    batch["codes"] = torch.zeros(1, 16, 5)
    with pytest.raises(ValueError, match="not cached"):
        model(batch, torch.randn(1, 768, 17))


def test_position_cache_allows_inference_then_training():
    layer = RotaryPositionalEmbeddings(4)
    with torch.inference_mode():
        layer(torch.randn(1, 2, 8, 8))
    value = torch.randn(1, 2, 5, 8, requires_grad=True)
    layer(value).sum().backward()
    assert value.grad is not None


def test_split_preserves_phonemes():
    ids = [0, 50, 83, 54, 16, 50, 83, 3, 4, 5, 6, 0]
    parts = split_ids(ids)
    assert [0] + [token for part in parts for token in part[1:-1]] + [0] == ids


def test_eval_forward_returns_finite_loss():
    model = S2(config()).eval()
    model.cfm.estimator.setup_caches(2, 32)
    batch = example()
    batch["codes"] = torch.zeros(1, 16, 5)
    torch.manual_seed(4)
    output = model(batch)["loss"]
    assert torch.isfinite(output)
