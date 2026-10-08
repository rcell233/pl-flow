import base64
import gc
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import torch

from pl_flow.checkpoint import ModelBundle, save_component, sha256
from pl_flow.config import write_json
from pl_flow.features.speaker import SpeakerEncoder


@pytest.fixture
def fake_tool(tmp_path, monkeypatch):
    path = tmp_path / "speaker_tool.py"
    path.write_text("""
import base64, json, struct, sys, time
mode = sys.argv[1]
if mode == "startup_error":
    print("checkpoint diagnostics", file=sys.stderr, flush=True)
    print(json.dumps({"ok": False, "error": "Invalid checkpoint"}), flush=True)
    sys.exit(1)
print(json.dumps({"ok": True, "ready": True, "embedding_dim": 256}), flush=True)
if mode == "no_read":
    time.sleep(60)
for line in sys.stdin:
    request = json.loads(line)
    if mode == "exit":
        sys.exit(2)
    if mode == "timeout":
        time.sleep(60)
    batch, samples = request["shape"]
    data = struct.unpack("<" + "f" * (batch * samples), base64.b64decode(request["audio"]))
    values = []
    for i, length in enumerate(request["lengths"]):
        mean = sum(data[i*samples:i*samples+length]) / length
        values.extend([mean] * 256)
    if mode == "nonfinite":
        values[0] = float("nan")
    result = {"ok": True, "shape": [batch, 255 if mode == "bad_shape" else 256],
              "embedding": base64.b64encode(struct.pack("<" + "f"*len(values), *values)).decode()}
    print(json.dumps(result), flush=True)
""")
    settings = {"mode": "normal"}
    monkeypatch.setattr(
        SpeakerEncoder, "_command", lambda self: [sys.executable, "-u", str(path), settings["mode"]]
    )
    return settings


def test_speaker_process_reuse_lengths_and_close(tmp_path, fake_tool):
    waves = torch.tensor([[1.0, 3.0, 999.0], [2.0, 4.0, 6.0]], requires_grad=True)
    lengths = torch.tensor([2, 3])
    encoder = SpeakerEncoder(tmp_path)
    with encoder as active:
        process = active._worker.process
        active.train()
        assert not active.training
        assert not list(active.parameters()) and not active.state_dict()
        first = active(waves, lengths)
        second = active(waves, lengths)
        assert not first.requires_grad and first.dtype == torch.float32
        assert torch.equal(first, torch.tensor([2.0, 4.0])[:, None].expand(-1, 256))
        assert torch.equal(first, second)
        assert active._worker.process is process
    assert process.poll() is not None
    with encoder:
        assert encoder._worker.process.pid != process.pid
        assert torch.equal(encoder(waves, lengths), first)


def test_speaker_calls_are_serialized(tmp_path, fake_tool):
    with SpeakerEncoder(tmp_path) as encoder, ThreadPoolExecutor(max_workers=4) as pool:

        def embed(value):
            return encoder(torch.full((1, 20), float(value)), torch.tensor([20]))

        results = list(pool.map(embed, range(8)))
        assert all(
            torch.equal(result, torch.full((1, 256), float(i))) for i, result in enumerate(results)
        )


def test_speaker_finalizer_stops_worker(tmp_path, fake_tool):
    encoder = SpeakerEncoder(tmp_path).start()
    process = encoder._worker.process
    del encoder
    gc.collect()
    assert process.poll() is not None


def test_speaker_device_change_stops_worker(tmp_path, fake_tool):
    with SpeakerEncoder(tmp_path) as encoder:
        process = encoder._worker.process
        encoder.to("cpu")
        assert process.poll() is not None
        encoder(torch.ones(1, 4), torch.tensor([4]))
        assert encoder._worker.process.pid != process.pid


@pytest.mark.parametrize(
    "mode, error, message",
    [
        ("startup_error", RuntimeError, "checkpoint diagnostics"),
        ("exit", RuntimeError, "exited without a response"),
        ("timeout", TimeoutError, "did not respond"),
        ("no_read", TimeoutError, "did not respond"),
        ("bad_shape", ValueError, "shape"),
        ("nonfinite", ValueError, "Non-finite"),
    ],
)
def test_speaker_failure_stops_child_and_allows_retry(tmp_path, fake_tool, mode, error, message):
    fake_tool["mode"] = mode
    encoder = SpeakerEncoder(tmp_path, timeout=1)
    size = 100000 if mode == "no_read" else 8
    with pytest.raises(error, match=message):
        encoder(torch.ones(1, size), torch.tensor([size]))
    assert encoder._worker.process is None
    fake_tool["mode"] = "normal"
    with encoder:
        assert torch.equal(encoder(torch.ones(1, 8), torch.tensor([8])), torch.ones(1, 256))


@pytest.mark.parametrize(
    "waves, lengths",
    [
        (torch.empty(0, 8), torch.empty(0, dtype=torch.long)),
        (torch.ones(1, 8), torch.tensor([0])),
        (torch.ones(1, 8), torch.tensor([9])),
        (torch.ones(1, 8), torch.tensor([2.5])),
        (torch.ones(1, 8), torch.tensor([[8]])),
        (torch.ones(1, 8, dtype=torch.long), torch.tensor([8])),
        (torch.full((1, 8), float("nan")), torch.tensor([8])),
    ],
)
def test_speaker_rejects_invalid_input_before_startup(tmp_path, waves, lengths):
    encoder = SpeakerEncoder(tmp_path)
    with pytest.raises(ValueError):
        encoder(waves, lengths)
    assert encoder._worker.process is None


def test_model_bundle_delegates_weight_loading(tmp_path, fake_tool, monkeypatch):
    entry = save_component(tmp_path / "speaker", "speaker", {}, {"dummy": torch.ones(1)})
    write_json(tmp_path / "bundle.json", {"format_version": 1, "components": {"speaker": entry}})

    def reject_load(*args, **kwargs):
        pytest.fail("The parent must not load speaker tensors")

    monkeypatch.setattr("pl_flow.checkpoint.load_file", reject_load)
    with ModelBundle(tmp_path).load("speaker") as encoder:
        assert encoder._worker.process is not None
    (tmp_path / "speaker/model.safetensors").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Corrupted component"):
        ModelBundle(tmp_path).load("speaker")


def test_parent_imports_no_speaker_implementation():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "\n".join(
                [
                    "import sys",
                    "from pl_flow.features.speaker import SpeakerEncoder",
                    "from pl_flow.checkpoint import ModelBundle",
                    "assert not any(k.startswith('wavlm_ecapa') for k in sys.modules)",
                ]
            ),
        ],
        cwd=Path(__file__).resolve().parents[1] / "src",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_external_speaker_requires_download_and_checks_hash(tmp_path, fake_tool, monkeypatch):
    directory = tmp_path / "speaker"
    entry = save_component(directory, "speaker", {}, {"dummy": torch.ones(1)})
    weights = directory / "wavlm_large_finetune.pth"
    weights.write_bytes(b"downloaded checkpoint")
    entry["weights_sha256"] = sha256(weights)
    entry["external_weights"] = {"filename": weights.name, "url": "https://example.org/speaker"}
    write_json(tmp_path / "bundle.json", {"format_version": 1, "components": {"speaker": entry}})
    weights.unlink()
    bundle = ModelBundle(tmp_path)
    bundle.component("speaker", require_weights=False)
    with pytest.raises(FileNotFoundError, match="https://example.org/speaker"):
        bundle.load("speaker")

    def reject_load(*args, **kwargs):
        pytest.fail("The parent must not deserialize speaker tensors")

    monkeypatch.setattr("pl_flow.checkpoint.load_file", reject_load)
    monkeypatch.setattr(torch, "load", reject_load)
    weights.write_bytes(b"downloaded checkpoint")
    with ModelBundle(tmp_path).load("speaker") as encoder:
        assert encoder.weights_filename == weights.name
        assert encoder._worker.process is not None
    weights.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Corrupted component"):
        ModelBundle(tmp_path).load("speaker")
    bundle = ModelBundle(tmp_path)
    bundle.manifest["components"]["speaker"]["external_weights"]["filename"] = "../outside.pth"
    with pytest.raises(ValueError, match="Invalid external weights filename"):
        bundle.component("speaker", require_weights=False)


def test_standalone_loads_upstream_checkpoint_strictly(tmp_path, monkeypatch):
    from torch import nn
    from torch.nn.utils.parametrizations import weight_norm

    from wavlm_ecapa.__main__ import load_model

    def small_model(config):
        model = nn.Module()
        model.feature_extract = nn.Module()
        model.feature_extract.model = nn.Module()
        model.feature_extract.model.encoder = nn.Module()
        model.feature_extract.model.encoder.pos_conv = nn.Sequential(
            weight_norm(nn.Conv1d(2, 2, 3), dim=2)
        )
        return model

    monkeypatch.setattr("wavlm_ecapa.model.SpeakerEncoder", small_model)
    expected = small_model({}).state_dict()
    prefix = "feature_extract.model.encoder.pos_conv.0."
    state = {prefix + "bias": expected[prefix + "bias"]}
    state[prefix + "weight_g"] = expected[prefix + "parametrizations.weight.original0"]
    state[prefix + "weight_v"] = expected[prefix + "parametrizations.weight.original1"]
    state["loss_calculator.projection.weight"] = torch.randn(3, 2)
    write_json(tmp_path / "config.json", {"format_version": 1, "type": "speaker", "model": {}})
    weights = tmp_path / "wavlm_large_finetune.pth"
    torch.save({"model": state, "best_valid_eer": 0.0}, weights)
    loaded = load_model(tmp_path, "cpu", weights.name)
    assert all(torch.equal(tensor, loaded.state_dict()[key]) for key, tensor in expected.items())

    state["unexpected.weight"] = torch.ones(1)
    torch.save({"model": state}, weights)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        load_model(tmp_path, "cpu", weights.name)
    del state["unexpected.weight"]
    del state[prefix + "weight_g"]
    torch.save({"model": state}, weights)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_model(tmp_path, "cpu", weights.name)


def test_standalone_protocol_and_file_input(tmp_path):
    # Import the independent program only in its own tests, never in the client.
    from wavlm_ecapa.__main__ import decode_request, read_audio

    audio = np.array([[1.0, 2.0, np.nan]], dtype="<f4")
    request = {
        "op": "embed",
        "shape": [1, 3],
        "lengths": [2],
        "audio": base64.b64encode(audio.tobytes()).decode(),
    }
    decoded, lengths = decode_request(request)
    np.testing.assert_equal(decoded, audio)
    np.testing.assert_array_equal(lengths, [2])
    for replacement in ({"shape": [1, 4]}, {"lengths": [4]}, {"lengths": [1.5]}, {"lengths": [3]}):
        with pytest.raises(ValueError):
            decode_request({**request, **replacement})
    np.save(tmp_path / "audio.npy", audio)
    np.save(tmp_path / "lengths.npy", np.array([2]))
    loaded, sizes = read_audio(tmp_path / "audio.npy", tmp_path / "lengths.npy")
    np.testing.assert_equal(loaded, audio)
    np.testing.assert_equal(sizes, lengths)


def test_standalone_reports_startup_errors(tmp_path):
    (tmp_path / "config.json").write_text('{"format_version":1,"type":"wrong"}')
    result = subprocess.run(
        [sys.executable, "-m", "wavlm_ecapa", "--model-dir", str(tmp_path), "--serve"],
        cwd=Path(__file__).resolve().parents[1] / "src",
        input="",
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1
    message = json.loads(result.stdout)
    assert message["ok"] is False and "speaker component" in message["error"]


def test_synthesizer_context_closes_speaker_on_error(tmp_path, fake_tool):
    from pl_flow.inference.pipeline import Synthesizer
    from pl_flow.inference.reference import ReferenceEncoder

    model = Synthesizer.__new__(Synthesizer)
    model.references = ReferenceEncoder(None, None, None, "cpu")
    model.references.speaker = SpeakerEncoder(tmp_path).start()
    process = model.references.speaker._worker.process
    with pytest.raises(ValueError, match="synthesis error"), model:
        raise ValueError("synthesis error")
    assert process.poll() is not None
