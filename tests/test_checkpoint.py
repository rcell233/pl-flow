import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from pl_flow.checkpoint import ModelBundle, save_component, tensor_fingerprint
from pl_flow.cli.bundle import assemble_bundle
from pl_flow.config import write_json
from pl_flow.inference.pipeline import Synthesizer


def bundle(tmp_path):
    root = tmp_path / "base"
    projection = {"weight": torch.tensor([[1.0, 2.0]])}
    fingerprint = tensor_fingerprint(projection)
    components = {
        "s1": save_component(root / "s1", "s1", {}, {"weight": torch.ones(1)}),
        "s2": save_component(
            root / "s2",
            "s2",
            {},
            {
                "conditioner.text_conditioner.prosody_down.weight": projection["weight"],
            },
        ),
    }
    write_json(
        root / "bundle.json",
        {
            "format_version": 1,
            "components": components,
            "sampling": {},
            "prosody_fingerprint": fingerprint,
            "s1_prosody_fingerprint": fingerprint,
        },
    )
    return root


def test_component_hash_and_path_validation(tmp_path):
    root = bundle(tmp_path)
    model = ModelBundle(root)
    model.component("s1")
    (root / "s1/config.json").write_text("{}")
    with pytest.raises(ValueError, match="Corrupted"):
        ModelBundle(root).component("s1")
    model = ModelBundle(root)
    model.manifest["components"]["s1"]["path"] = "../outside"
    with pytest.raises(ValueError, match="escapes"):
        model.component("s1")


def test_changed_projection_rejects_old_s1_pairing(tmp_path):
    root = bundle(tmp_path)
    replacement = tmp_path / "s2"
    save_component(
        replacement,
        "s2",
        {},
        {
            "conditioner.text_conditioner.prosody_down.weight": torch.tensor([[3.0, 2.0]]),
        },
    )
    output = tmp_path / "updated"
    assemble_bundle(SimpleNamespace(base=root, component=[replacement], output=output))
    manifest = json.loads((output / "bundle.json").read_text())
    assert manifest["prosody_fingerprint"] != manifest["s1_prosody_fingerprint"]
    with pytest.raises(ValueError, match="different prosody"):
        Synthesizer(output, "cpu")


def test_export_does_not_overwrite_component(tmp_path):
    root = bundle(tmp_path)
    with pytest.raises(FileExistsError):
        save_component(root / "s1", "s1", {}, {"weight": torch.zeros(1)})


def test_bundle_keeps_notices_for_selected_components(tmp_path):
    root = bundle(tmp_path)
    (root / "LICENSE").write_text("Bundle license")
    (root / "README.md").write_text("Model card")
    (root / "THIRD_PARTY.md").write_text("Component attribution")
    (root / "licenses").mkdir()
    (root / "licenses/upstream.txt").write_text("Upstream license")
    (root / "s1/NOTICE.md").write_text("Unchanged component notice")
    (root / "s2/NOTICE.md").write_text("Replaced component notice")
    (root / "unrelated.txt").write_text("Do not copy unrelated files")

    replacement = tmp_path / "replacement"
    save_component(
        replacement,
        "s2",
        {},
        {"conditioner.text_conditioner.prosody_down.weight": torch.tensor([[1.0, 2.0]])},
    )
    (replacement / "NOTICE.md").write_text("Replacement component notice")
    (replacement / "LICENSE.txt").write_text("Replacement license")
    (replacement / "licenses").mkdir()
    (replacement / "licenses/dependency.txt").write_text("Replacement dependency license")
    output = tmp_path / "assembled"
    assemble_bundle(SimpleNamespace(base=root, component=[replacement], output=output))

    for name in ("LICENSE", "README.md", "THIRD_PARTY.md", "licenses/upstream.txt", "s1/NOTICE.md"):
        assert (output / name).read_bytes() == (root / name).read_bytes()
    for name in ("NOTICE.md", "LICENSE.txt", "licenses/dependency.txt"):
        assert (output / "s2" / name).read_bytes() == (replacement / name).read_bytes()
    assert not (output / "unrelated.txt").exists()
    ModelBundle(output).component("s2")


@pytest.mark.parametrize("downloaded", [False, True])
def test_bundle_excludes_external_speaker_weights(tmp_path, downloaded):
    root = bundle(tmp_path)
    entry = save_component(root / "speaker", "speaker", {}, {"dummy": torch.ones(1)})
    weights = b"user-supplied checkpoint"
    entry["weights_sha256"] = hashlib.sha256(weights).hexdigest()
    entry["external_weights"] = {
        "filename": "wavlm_large_finetune.pth",
        "url": "https://example.org/speaker",
    }
    (root / "speaker/NOTICE.md").write_text("Download from the upstream source")
    if downloaded:
        (root / "speaker/wavlm_large_finetune.pth").write_bytes(weights)
    manifest = json.loads((root / "bundle.json").read_text())
    manifest["components"]["speaker"] = entry
    write_json(root / "bundle.json", manifest)
    output = tmp_path / "release"
    assemble_bundle(SimpleNamespace(base=root, component=[], output=output))
    assert {path.name for path in (output / "speaker").iterdir()} == {"config.json", "NOTICE.md"}
    assembled = ModelBundle(output)
    assert assembled.manifest["components"]["speaker"] == entry
    assembled.component("speaker", require_weights=False)
    with pytest.raises(FileNotFoundError, match="https://example.org/speaker"):
        assembled.component("speaker")
    (output / "speaker/wavlm_large_finetune.pth").write_bytes(weights)
    assembled.component("speaker")
