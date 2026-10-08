"""Reproducible training state, atomically saved only between optimizer updates."""

import random
from pathlib import Path

import numpy as np
import torch


def rng_state():
    numpy = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python": random.getstate(),
        "numpy": (numpy[0], numpy[1].tolist(), *numpy[2:]),
    }


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("A CUDA training checkpoint cannot exactly resume on CPU")
        torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.array(numpy[1], dtype=np.uint32), *numpy[2:]))


def atomic_save(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)
