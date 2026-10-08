import json

import numpy as np
import pytest

from pl_flow.data.dataset import StageDataset, collate
from pl_flow.data.manifest import Manifest, pack
from pl_flow.data.sampler import BatchSampler


@pytest.fixture
def data(tmp_path):
    row = {
        "audio_path": "missing.wav",
        "text": "test",
        "language": "en",
        "duration": 0.6,
        "phoneme_ids": [0, 50, 83, 0],
        "dur": [3, 8, 9, 8],
    }
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(row) + "\n")
    for suffix, value in [
        (".spk.npy", np.ones(256, np.float32)),
        (".code.npy", np.ones((16, 4), np.int8)),
        (".latent.npy", np.ones((32, 30), np.int8)),
    ]:
        np.save(tmp_path / ("missing.wav" + suffix), value)
    return path


def test_frozen_stages_never_read_waveform(data):
    prompt = {"min_seconds": 1, "max_seconds": 5, "max_fraction": 0.5, "dropout": 0.1}
    assert StageDataset(data, "s1")[0]["code"].shape == (16, 4)
    value = StageDataset(data, "s2", prompt=prompt)[0]
    assert "wave" not in value and "codes" in value
    packed = collate([value, value])
    assert packed["targets"].shape == (2, 32, 30)
    assert packed["target_lengths"].tolist() == [30, 30]


def test_trainable_requires_waveform(data):
    prompt = {"min_seconds": 1, "max_seconds": 5, "max_fraction": 0.5, "dropout": 0.1}
    with pytest.raises(Exception):
        StageDataset(data, "s2", prosody_mode="trainable", prompt=prompt)[0]


def test_portable_arrow_pack(data, tmp_path):
    rows = Manifest(data).rows
    pack(rows, tmp_path / "train.ds", tmp_path)
    manifest = Manifest(tmp_path / "train.ds", tmp_path)
    assert manifest.rows[0]["audio_path"] == "missing.wav"
    assert manifest.ids(manifest.rows[0]) == [0, 50, 83, 0]


def test_sampler_offset_exact_and_budget():
    lengths = [3, 4, 7, 5, 9, 3, 14]
    all_batches = list(BatchSampler(lengths, 3, 28, epoch=2))
    resumed = list(BatchSampler(lengths, 3, 28, epoch=2, offset=2))
    assert resumed == all_batches[2:]
    for batch in all_batches:
        assert len(batch) * max(lengths[index] for _, index in batch) <= 28
        assert all(epoch == 2 for epoch, _ in batch)


def test_code_range_validation(data):
    np.save(data.parent / "missing.wav.code.npy", np.full((16, 4), 10, np.int8))
    with pytest.raises(ValueError):
        StageDataset(data, "s1")[0]
