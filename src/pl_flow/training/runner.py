"""A small common training runner; stage objectives stay independent of orchestration."""

import hashlib
import json
import random
import signal
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pl_flow.alignment.model import ASRTransformerAligner
from pl_flow.alignment.objective import alignment_objective
from pl_flow.checkpoint import ModelBundle, tensor_fingerprint
from pl_flow.config import read_config
from pl_flow.data.dataset import StageDataset, collate
from pl_flow.data.sampler import BatchSampler
from pl_flow.models.s1 import S1
from pl_flow.models.s2 import S2
from pl_flow.vocoder.model import SQCodec

from .engine import Engine
from .state import atomic_save


def train(args):
    config = read_config(args.config)
    stage, recipe = config["stage"], config["train"]
    if stage not in {"s1", "s2", "aligner", "vocoder"}:
        raise ValueError("Unknown training stage")
    output = Path(args.output)
    if (output / "last.pt").exists() and not args.resume:
        raise FileExistsError(
            "Training output already contains a checkpoint; select --resume or a new directory"
        )
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    seed = config.get("seed", 1234)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    dependencies = args.models or args.init_weights
    bundle = ModelBundle(dependencies) if dependencies else None
    model_config = read_config(Path(args.config).parent / config["model_config"])
    mode = config.get("prosody_mode", "frozen")
    if args.init_weights:
        initialization = ModelBundle(args.init_weights)
        if initialization.component(stage)[0] != model_config:
            raise ValueError(
                "Warm-start architecture differs from the selected model configuration"
            )
        model = initialization.load(stage, device, prosody_mode=mode)
    elif stage == "s1":
        if bundle is None:
            raise ValueError("S1 requires a bundle containing frozen PL-BERT")
        model = S1(model_config, bundle.load("plbert")).to(device)
    elif stage == "s2":
        model = S2(model_config, mode).to(device)
        if mode == "frozen":
            if bundle is None:
                raise ValueError(
                    "Frozen S2 requires a bundle supplying the trained prosody projection"
                )
            source = bundle.load("s2")
            model.conditioner.text_conditioner.prosody_down.load_state_dict(
                source.conditioner.text_conditioner.prosody_down.state_dict()
            )
            del source
    elif stage == "aligner":
        model = ASRTransformerAligner(**model_config).to(device)
    else:
        model = SQCodec(model_config).to(device)
    if stage in {"s1", "s2"}:
        model.cfm.estimator.setup_caches(2 * recipe["batch_size"], 1026)
    hubert = None
    if stage == "aligner" or (stage == "s2" and mode == "trainable"):
        if bundle is None:
            raise ValueError("Online audio features require a pretrained HuBERT component")
        hubert = bundle.load("hubert", device)
    dataset = StageDataset(
        args.data,
        stage,
        audio_root=args.audio_root,
        prosody_mode=mode,
        prompt=model_config.get("prompt"),
        seed=seed,
        sample_size=recipe.get("sample_size", 40960),
    )
    dataset_identity = getattr(dataset.manifest.rows, "_fingerprint", None)
    if dataset_identity is None:
        dataset_identity = hashlib.sha256(
            json.dumps(dataset.manifest.rows, sort_keys=True).encode()
        ).hexdigest()
    interfaces = {}
    if stage in {"s1", "s2"} and mode == "frozen":
        metadata_path = Path(args.data).with_suffix(".features.json")
        if not metadata_path.exists():
            raise ValueError(
                "Missing code-space metadata; run preprocess features --verify-existing for existing sidecars"
            )
        metadata = read_config(metadata_path)
        expected = (
            bundle.manifest["prosody_fingerprint"]
            if stage == "s1"
            else tensor_fingerprint(model.conditioner.text_conditioner.prosody_down.state_dict())
        )
        if metadata["prosody_fingerprint"] != expected:
            raise ValueError("Dataset code space differs from the selected frozen projection")
        interfaces["prosody_fingerprint"] = expected
    lengths = []
    for row in dataset.manifest.rows:
        if stage == "s1":
            lengths.append(
                len(row.get("phoneme_ids") or row.get("dur") or dataset.manifest.ids(row))
            )
        elif stage == "vocoder":
            lengths.append(recipe["sample_size"])
        else:
            lengths.append(max(1, round(float(row["duration"]) * 50)))
    if stage == "vocoder":
        from .vocoder import VocoderEngine

        engine = VocoderEngine(model, recipe)
    else:
        parameters = [p for p in model.parameters() if p.requires_grad]
        if stage == "aligner":
            optimizer = torch.optim.Adam(parameters, lr=recipe["learning_rate"], betas=(0.9, 0.98))
            scheduler = None
        else:
            optimizer = torch.optim.AdamW(
                parameters,
                lr=recipe["learning_rate"],
                betas=(0.9, 0.98),
                eps=1e-6,
                weight_decay=0.01,
                fused=False,
            )
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.999996)
        engine = Engine(
            model,
            optimizer,
            scheduler,
            ema_decay=recipe.get("ema_decay"),
            precision=recipe["precision"],
            gradient_clip=recipe.get("gradient_clip", 10),
        )
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        if state["config"] != config:
            raise ValueError("Resume requires the same training recipe; use --max-steps to extend")
        if state["dataset_identity"] != dataset_identity or state["model_config"] != model_config:
            raise ValueError("Dataset or architecture changed on resume")
        engine.load_state_dict(state["training"])

    def objective(model, batch):
        if stage == "s1":
            return {"loss": model(**batch)}
        with torch.autocast(device.type, enabled=False):
            if stage == "aligner":
                features, sizes = hubert.aligned(batch["wave"], batch["wave_lengths"])
            elif mode == "trainable":
                features = hubert.to_latent_frames(
                    batch["wave"], batch["wave_lengths"], batch["target_lengths"]
                )
            else:
                features = None
        if stage == "aligner":
            return alignment_objective(
                model,
                batch["text"],
                batch["text_lengths"],
                features,
                sizes,
                step=engine.step,
                mono_start_step=recipe["mono_start_step"],
            )
        return model(batch, features)

    output.mkdir(parents=True, exist_ok=True)
    maximum = args.max_steps or recipe["max_steps"]
    accumulation = recipe.get("accumulation_steps", 1)
    if accumulation < 1:
        raise ValueError("accumulation_steps must be positive")
    stopping = False

    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True
        print(
            "Stopping after this optimizer update; a resumable checkpoint will be saved.",
            flush=True,
        )

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, request_stop)

    def save():
        atomic_save(
            output / "last.pt",
            {
                "format_version": 1,
                "stage": stage,
                "config": config,
                "model_config": model_config,
                "dataset_identity": dataset_identity,
                "interfaces": interfaces,
                "training": engine.state_dict(),
            },
        )

    while engine.step < maximum:
        sampler = BatchSampler(
            lengths, recipe["batch_size"], recipe["token_budget"], seed, engine.epoch, engine.offset
        )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            multiprocessing_context="spawn" if args.workers else None,
            generator=torch.Generator().manual_seed(seed + engine.epoch),
        )
        iterator = iter(loader)
        while engine.step < maximum:
            batches = []
            for _ in range(accumulation):
                batch = next(iterator, None)
                if batch is None:
                    break
                batches.append(
                    {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                )
            if not batches:
                break
            metrics = (
                engine.train_step(batches)
                if stage == "vocoder"
                else engine.train_step(batches, objective)
            )
            if stopping:
                save()
                return
            if engine.step % recipe["log_interval"] == 0:
                print(json.dumps(metrics), flush=True)
            if metrics["updated"] and engine.step % recipe["checkpoint_interval"] == 0:
                save()
        if engine.step < maximum:
            engine.epoch += 1
            engine.offset = 0
    save()
