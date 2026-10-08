"""Hash-pinned release components and separately supplied speaker weights."""

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .config import write_json


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def tensor_fingerprint(state):
    value = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        tensor = tensor.detach().cpu().contiguous()
        header = {"name": key, "dtype": str(tensor.dtype), "shape": list(tensor.shape)}
        value.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode())
        value.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return value.hexdigest()


def save_component(directory, kind, config, state, interfaces=None):
    """Only tensors and an explicit architecture config enter a release artifact."""
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Refusing to overwrite component: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    tensors = {key: value.detach().cpu().contiguous().clone() for key, value in state.items()}
    if not tensors or any(not torch.isfinite(value).all() for value in tensors.values()):
        raise ValueError("Empty or non-finite model state")
    save_file(tensors, str(directory / "model.safetensors"), metadata={"format": "pt"})
    metadata = {"format_version": 1, "type": kind, "model": config}
    if interfaces:
        metadata["interfaces"] = interfaces
    write_json(directory / "config.json", metadata)
    return {
        "path": directory.name,
        "config_sha256": sha256(directory / "config.json"),
        "weights_sha256": sha256(directory / "model.safetensors"),
    }


class ModelBundle:
    def __init__(self, directory):
        self.root = Path(directory).resolve()
        self.manifest = json.loads((self.root / "bundle.json").read_text())
        if self.manifest.get("format_version") != 1:
            raise ValueError("Unsupported bundle format")
        self.verified = set()

    def component(self, name, *, require_weights=True):
        entry = self.manifest["components"][name]
        path = (self.root / entry["path"]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Component path escapes the model bundle")
        external = entry.get("external_weights")
        filename = "model.safetensors"
        if external is not None:
            if name != "speaker" or not isinstance(external, dict):
                raise ValueError("Only speaker components support external weights")
            filename = external.get("filename")
            if (
                not isinstance(filename, str)
                or filename in {"", ".", ".."}
                or Path(filename).name != filename
            ):
                raise ValueError("Invalid external weights filename")
        check_weights = require_weights or external is None
        if name not in self.verified:
            if sha256(path / "config.json") != entry["config_sha256"]:
                raise ValueError(f"Corrupted component: {name}/config.json")
            if check_weights:
                if external is not None and not (path / filename).is_file():
                    raise FileNotFoundError(
                        f"Download the speaker checkpoint from {external.get('url', 'its upstream source')} "
                        f"and save it as {path / filename}"
                    )
                if sha256(path / filename) != entry["weights_sha256"]:
                    raise ValueError(f"Corrupted component: {name}/{filename}")
                self.verified.add(name)
        config = json.loads((path / "config.json").read_text())
        if config.get("format_version") != 1 or config.get("type") != name:
            raise ValueError(f"Component type mismatch: {name}")
        return config["model"], path / filename

    def load(self, name, device="cpu", *, prosody_mode="frozen"):
        config, path = self.component(name)
        if name == "speaker":
            from .features.speaker import SpeakerEncoder

            return SpeakerEncoder(path.parent, device, weights_filename=path.name).start()

        from .alignment.model import ASRTransformerAligner
        from .features.hubert import HubertFeatures
        from .models.s1 import S1
        from .models.s2 import S2
        from .text.plbert import PLBert
        from .vocoder.model import SQCodec

        factories = {
            "aligner": ASRTransformerAligner,
            "hubert": HubertFeatures,
            "plbert": PLBert,
            "vocoder": SQCodec,
        }
        if name == "s1":
            model = S1(config, self.load("plbert"))
        elif name == "s2":
            model = S2(config, prosody_mode)
        elif name == "aligner":
            model = ASRTransformerAligner(**config)
        else:
            model = factories[name](config)
        state = load_file(str(path))
        if name == "vocoder":
            from .vocoder.convert_checkpoint import convert_stft_state_dict

            state = convert_stft_state_dict(state, config)
        if name == "s2":
            prefix = "conditioner.text_conditioner.prosody_down."
            projection = {
                k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)
            }
            if tensor_fingerprint(projection) != self.manifest["prosody_fingerprint"]:
                raise ValueError("S2 projection does not match the declared prosody space")
        if name == "s1":
            state.update({f"plbert.{k}": v for k, v in model.plbert.state_dict().items()})
        model.load_state_dict(state, strict=True)
        return model.to(device).eval()
