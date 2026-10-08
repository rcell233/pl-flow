"""Assemble exported components into a relocatable, hash-pinned model bundle."""

import copy
import json
import shutil
from pathlib import Path

from safetensors.torch import load_file

from pl_flow.checkpoint import ModelBundle, sha256, tensor_fingerprint
from pl_flow.config import write_json


def copy_notices(source, destination):
    """Keep model cards and licensing material with redistributed weights."""
    for path in Path(source).iterdir():
        name = path.name.upper()
        if path.is_dir() and name == "LICENSES":
            shutil.copytree(path, destination / path.name)
        elif path.is_file() and (
            name.startswith(("LICENSE", "COPYING", "NOTICE"))
            or name in {"README.MD", "THIRD_PARTY.MD"}
        ):
            shutil.copyfile(path, destination / path.name)


def assemble_bundle(args):
    base = ModelBundle(args.base)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    manifest = copy.deepcopy(base.manifest)
    sources = {}
    for name in manifest["components"]:
        _, weights = base.component(name, require_weights=False)
        sources[name] = weights.parent
    replaced = set()
    for directory in args.component:
        directory = Path(directory)
        config = json.loads((directory / "config.json").read_text())
        name = config["type"]
        if config.get("format_version") != 1 or name not in sources or name in replaced:
            raise ValueError("Unknown, duplicate or invalid component")
        sources[name] = directory
        replaced.add(name)
        if name == "s1":
            manifest["s1_prosody_fingerprint"] = config["interfaces"]["prosody_fingerprint"]
    if "s2" in replaced:
        prefix = "conditioner.text_conditioner.prosody_down."
        state = load_file(str(sources["s2"] / "model.safetensors"))
        manifest["prosody_fingerprint"] = tensor_fingerprint(
            {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}
        )
        del state
    output.mkdir(parents=True)
    for name, source in sources.items():
        destination = output / name
        destination.mkdir()
        external = name not in replaced and "external_weights" in manifest["components"][name]
        filenames = ("config.json",) if external else ("config.json", "model.safetensors")
        for filename in filenames:
            shutil.copyfile(source / filename, destination / filename)
        copy_notices(source, destination)
        if external:
            manifest["components"][name]["path"] = name
        else:
            manifest["components"][name] = {
                "path": name,
                "config_sha256": sha256(destination / "config.json"),
                "weights_sha256": sha256(destination / "model.safetensors"),
            }
    copy_notices(base.root, output)
    write_json(output / "bundle.json", manifest)
    if manifest["s1_prosody_fingerprint"] != manifest["prosody_fingerprint"]:
        print(
            "Bundle ready for feature extraction/S1 training, but S1 must be retrained for this prosody space before synthesis."
        )
    else:
        print("Bundle assembled. Run pl-flow verify before use.")
