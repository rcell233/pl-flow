"""Portable configuration and strict JSON serialization."""

import json
from pathlib import Path
from types import SimpleNamespace

import yaml


def recursive_munch(value):
    """Attribute view used only at the neural-network construction boundary."""
    if isinstance(value, dict):
        return SimpleNamespace(**{key: recursive_munch(item) for key, item in value.items()})
    return value


def read_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream) if Path(path).suffix == ".json" else yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a mapping")
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)
