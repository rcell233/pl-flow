import argparse
import json

import numpy as np
import pytest
import soundfile as sf
import torch

from pl_flow.cli.seedtts import add_arguments, read_metadata, run_seedtts
from pl_flow.inference.pipeline import stable_seed


@pytest.fixture
def evaluation(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    (dataset / "en").mkdir(parents=True)
    reference = dataset / "en" / "reference.wav"
    sf.write(reference, np.zeros(1600), 16000)
    meta = dataset / "en" / "meta.lst"
    meta.write_text(
        "first|Prompt text.|reference.wav|First target.|missing-ground-truth.wav\n"
        "second|Prompt text.|reference.wav|Second target.\n"
        "third|Prompt text.|reference.wav|Third target.\n"
    )
    models = tmp_path / "models"
    models.mkdir()
    (models / "bundle.json").write_text(
        json.dumps({"format_version": 1, "sampling": {"s1_steps": 8, "s2_steps": 10}})
    )

    class Synthesizer:
        sample_rate = 32000
        batches = []
        prompts = []

        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def reference(self, audio, text, language, seed):
            self.prompts.append((audio, text, language))
            return text

        def phonemes(self, text, language):
            return text

        def synthesize_prepared(self, sequences, references, keys, seed):
            self.batches.append((sequences, references, keys))
            rng = torch.Generator().manual_seed(stable_seed(seed, "|".join(keys)))
            return [(torch.rand(1, 1600, generator=rng) * 0.5, []) for _ in keys]

    monkeypatch.setattr("pl_flow.inference.pipeline.Synthesizer", Synthesizer)
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(
        [
            "--models",
            str(models),
            "--dataset",
            str(dataset),
            "--output",
            str(tmp_path / "output"),
            "--split",
            "test-en",
            "--device",
            "cpu",
            "--batch-size",
            "2",
        ]
    )
    return args, Synthesizer, meta, tmp_path / "output" / "test-en"


def test_generation_and_partial_resume_preserve_batching(evaluation):
    args, model, meta, output = evaluation
    run_seedtts(args)
    assert len(model.prompts) == 1
    assert model.prompts[0][1:] == ("Prompt text.", "en")
    assert model.batches[0] == (
        ["First target.", "Second target."],
        ["Prompt text.", "Prompt text."],
        ["en_meta/first", "en_meta/second"],
    )
    assert len(read_metadata(output / "meta.lst")) == 3
    assert "missing-ground-truth" not in (output / "meta.lst").read_text()
    record = json.loads((output / "run.json").read_text())
    assert (record["completed"], record["status"]) == (3, "complete")
    audio = {p.name: p.read_bytes() for p in output.glob("*.wav")}
    assert len(audio) == 3
    (output / "second.wav").unlink()
    model.batches.clear()
    args.resume = True
    run_seedtts(args)
    assert len(model.batches) == 1
    assert model.batches[0][2] == ["en_meta/first", "en_meta/second"]
    assert {p.name: p.read_bytes() for p in output.glob("*.wav")} == audio


def test_resume_rejects_changed_inputs_or_settings(evaluation):
    args, _, meta, output = evaluation
    run_seedtts(args)
    with pytest.raises(FileExistsError):
        run_seedtts(args)
    args.resume = True
    args.seed += 1
    with pytest.raises(ValueError, match="inputs or settings differ"):
        run_seedtts(args)
    args.seed -= 1
    sf.write(meta.parent / "reference.wav", np.ones(1600) * 0.1, 16000)
    with pytest.raises(ValueError, match="inputs or settings differ"):
        run_seedtts(args)


def test_limit_is_recorded_and_exports_only_selected_rows(evaluation):
    args, _, _, output = evaluation
    args.limit = 1
    run_seedtts(args)
    record = json.loads((output / "run.json").read_text())
    assert (record["config"]["total"], record["config"]["selected"]) == (3, 1)
    assert len(read_metadata(output / "meta.lst")) == 1
    assert len(list(output.glob("*.wav"))) == 1


@pytest.mark.parametrize(
    "row",
    [
        "../escape|Prompt|reference.wav|Target",
        "bad|missing prompt and audio",
        "empty||reference.wav|Target",
        "same|Prompt|reference.wav|Target\nsame|Prompt|reference.wav|Target",
    ],
)
def test_invalid_metadata_fails_before_generation(evaluation, row):
    args, model, meta, _ = evaluation
    meta.write_text(row)
    with pytest.raises(ValueError):
        run_seedtts(args)
    assert not model.batches


@pytest.mark.parametrize("option", ["--batch-size", "--limit", "--s1-steps", "--s2-steps"])
def test_nonpositive_sizes_rejected(option):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--models", "m", "--dataset", "d", "--output", "o", option, "0"])
