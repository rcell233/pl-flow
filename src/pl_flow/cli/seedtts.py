"""Generate the three Seed-TTS-Eval TTS splits with the released model bundle."""

import argparse
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

SPLITS = {
    "test-en": ("en/meta.lst", "en"),
    "test-zh": ("zh/meta.lst", "zh"),
    "test-zh-hard": ("zh/hardcase.lst", "zh"),
}


@dataclass(frozen=True)
class Utterance:
    name: str
    reference_text: str
    reference_audio: Path
    text: str


def read_metadata(path):
    """Read prompt/text pairs; the optional ground-truth audio field is unused."""
    path = Path(path)
    items, names = [], set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split("|")]
        if len(fields) not in (4, 5) or not all(fields[:4]):
            raise ValueError(f"{path}:{number}: expected ID|prompt text|prompt audio|text[|GT]")
        name, reference_text, reference_audio, text = fields[:4]
        if name in {".", ".."} or Path(name).name != name or "\\" in name or name in names:
            raise ValueError(f"{path}:{number}: invalid or duplicate utterance ID: {name}")
        reference = Path(reference_audio)
        if not reference.is_absolute():
            reference = path.parent / reference
        if not reference.is_file():
            raise FileNotFoundError(reference)
        names.add(name)
        items.append(Utterance(name, reference_text, reference.resolve(), text))
    if not items:
        raise ValueError(f"Empty metadata: {path}")
    return items


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def add_arguments(parser):
    parser.add_argument("--models", required=True, help="PL-Flow model bundle")
    parser.add_argument("--dataset", required=True, help="Extracted seedtts_testset directory")
    parser.add_argument("--output", required=True, help="Output root; each split gets a directory")
    parser.add_argument("--split", nargs="+", choices=list(SPLITS), default=list(SPLITS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--limit", type=positive_int, help="Generate only the first N items per split"
    )
    parser.add_argument("--resume", action="store_true", help="Resume the same inputs and settings")
    for stage in ("s1", "s2"):
        parser.add_argument(f"--{stage}-steps", type=positive_int)
        parser.add_argument(f"--{stage}-cfg", type=float)


def valid_audio(path, sample_rate):
    import soundfile as sf

    try:
        info = sf.info(path)
        return info.frames > 0 and info.channels == 1 and info.samplerate == sample_rate
    except (OSError, RuntimeError):
        return False


def run_seedtts(args):
    import soundfile as sf
    import torch

    from pl_flow.checkpoint import ModelBundle, sha256
    from pl_flow.config import write_json
    from pl_flow.inference.pipeline import Synthesizer

    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    bundle = ModelBundle(args.models)
    overrides = {
        key: getattr(args, key)
        for key in ("s1_steps", "s2_steps", "s1_cfg", "s2_cfg")
        if getattr(args, key) is not None
    }
    settings = {**bundle.manifest["sampling"], **overrides}
    plans = []
    reference_hashes = {}
    for split in dict.fromkeys(args.split):
        relative, language = SPLITS[split]
        meta = Path(args.dataset) / relative
        all_items = read_metadata(meta)
        items = all_items[: args.limit] if args.limit is not None else all_items
        for item in items:
            if str(item.reference_audio) not in reference_hashes:
                reference_hashes[str(item.reference_audio)] = sha256(item.reference_audio)
        references = {
            str(item.reference_audio): reference_hashes[str(item.reference_audio)] for item in items
        }
        config = {
            "bundle_sha256": sha256(bundle.root / "bundle.json"),
            "meta_sha256": sha256(meta),
            "references_sha256": hashlib.sha256(
                json.dumps(references, sort_keys=True).encode()
            ).hexdigest(),
            "sampling": settings,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "device": str(torch.device(args.device)),
            "torch_version": str(torch.__version__),
            "split": split,
            "seed_namespace": f"{language}_{Path(relative).stem}",
            "total": len(all_items),
            "selected": len(items),
        }
        folder = Path(args.output) / split
        record = folder / "run.json"
        if folder.exists() and any(folder.iterdir()):
            if not args.resume:
                raise FileExistsError(f"{folder}: use --resume to continue the same run")
            if not record.is_file() or json.loads(record.read_text()).get("config") != config:
                raise ValueError(f"{folder}: inputs or settings differ; use a new output directory")
        plans.append((folder, language, items, config))

    with Synthesizer(bundle, args.device, **overrides) as model:

        @lru_cache(maxsize=64)
        def reference(audio, text, language):
            return model.reference(audio, text, language, args.seed)

        try:
            for folder, language, items, config in plans:
                folder.mkdir(parents=True, exist_ok=True)
                # Absolute prompt paths let the upstream scorer use this manifest anywhere.
                (folder / "meta.lst").write_text(
                    "".join(
                        f"{item.name}|{item.reference_text}|{item.reference_audio}|{item.text}\n"
                        for item in items
                    ),
                    encoding="utf-8",
                )
                record = {"config": config, "completed": 0, "status": "running"}
                write_json(folder / "run.json", record)
                for start in range(0, len(items), args.batch_size):
                    batch = items[start : start + args.batch_size]
                    paths = [folder / f"{item.name}.wav" for item in batch]
                    ready = [valid_audio(path, model.sample_rate) for path in paths]
                    if not (args.resume and all(ready)):
                        refs = [
                            reference(str(item.reference_audio), item.reference_text, language)
                            for item in batch
                        ]
                        sequences = [model.phonemes(item.text, language) for item in batch]
                        # Recompute an incomplete batch intact so resuming preserves its noise.
                        outputs = model.synthesize_prepared(
                            sequences,
                            refs,
                            [f"{config['seed_namespace']}/{item.name}" for item in batch],
                            args.seed,
                        )
                        if len(outputs) != len(batch):
                            raise RuntimeError("Synthesis returned an incomplete batch")
                        for path, (wave, _) in zip(paths, outputs):
                            if (
                                wave.ndim != 2
                                or wave.size(0) != 1
                                or wave.size(1) == 0
                                or not torch.isfinite(wave).all()
                            ):
                                raise ValueError(f"Invalid generated waveform: {path}")
                            temporary = path.with_suffix(".wav.tmp")
                            sf.write(
                                temporary, wave.T.cpu().numpy(), model.sample_rate, format="WAV"
                            )
                            temporary.replace(path)
                    record["completed"] = start + len(batch)
                    write_json(folder / "run.json", record)
                    print(f"{folder.name}: {record['completed']}/{len(items)}", flush=True)
                record["status"] = "complete"
                write_json(folder / "run.json", record)
        finally:
            reference.cache_clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    run_seedtts(parser.parse_args())


if __name__ == "__main__":
    main()
