# WavLM/ECAPA speaker embedding tool

Extracts 256-dimensional WavLM/ECAPA embeddings from 16 kHz audio using a local
`config.json` and checkpoint. This program can run independently of
PL-Flow. Its dependencies are PyTorch, NumPy, safetensors and SoundFile, and its
licenses are described in [NOTICE.md](NOTICE.md).

PL-Flow starts and manages this program automatically during synthesis.
The commands below provide standalone embedding extraction.

Download [wavlm_large_finetune.pth](https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view)
from the model link in [Seed-TTS-Eval](https://github.com/BytedanceSpeech/seed-tts-eval#metrics)
and place it beside the speaker `config.json`.

```bash
wavlm-ecapa --model-dir /path/to/speaker --weights wavlm_large_finetune.pth --device cuda \
  --input reference.wav --output embedding.npy
# Equivalent module entry point:
python -m wavlm_ecapa --model-dir /path/to/speaker \
  --weights wavlm_large_finetune.pth \
  --input audio.npy --lengths lengths.npy --output embeddings.npy
```

`--weights` selects the checkpoint filename and defaults to `model.safetensors`.
PL-Flow selects the file declared in its bundle manifest.

Audio files must be sampled at 16000 Hz; multiple channels are averaged.
NumPy input is a floating array `[samples]` or `[batch, samples]`, with optional
integer sample counts `[batch]`. Only the valid prefix of each row is used.
Output is a float32 NumPy array `[batch, 256]`. Existing output files are not
overwritten. `--threads` controls CPU threads (default 1).

## Streaming interface

Use `--serve` instead of `--input` for persistent operation over stdin/stdout.
Each request and response is a UTF-8 JSON object on one line. Diagnostics go to
stderr. The program loads the checkpoint strictly before emitting:

```json
{"ok":true,"ready":true,"embedding_dim":256}
```

An embedding request has `op: "embed"`, `shape: [batch, samples]`, integer
`lengths: [n1, n2, ...]`, and `audio`, a base64 string of row-major, little-endian
float32 samples at 16 kHz. The successful response has `ok: true`,
`shape: [batch, 256]`, and `embedding`, encoded in the same byte format. Invalid
requests return `ok: false` and a readable `error` string. EOF ends the program;
a startup failure returns an error and exits with a nonzero status.

Optional execution settings are `precision` (`highest`, `high`, or `medium`;
default `highest`), `cudnn_tf32` (default true), `cudnn_benchmark`,
`cudnn_deterministic`, `deterministic`, and `deterministic_warn_only` (default
false). PL-Flow supplies its execution settings to the speaker process.
Inputs and embeddings are data only; the interface carries no Python objects,
model parameters, hidden activations, or gradients.
