# Third-party notices

The PL-Flow application (`src/pl_flow`) is licensed under [GPL-3.0-only](LICENSE).
Incorporated third-party code, the standalone speaker program and installed
dependencies retain their applicable licenses and notices.

## Code attribution

| Source | Use | Retained license / notice |
| --- | --- | --- |
| [Seed-VC](https://github.com/Plachtaa/seed-vc) | Flow/DiT, WaveNet and integration | [GPL-3.0](LICENSE) |
| [gpt-fast](https://github.com/meta-pytorch/gpt-fast) | Transformer, rotary embeddings and normalization | [BSD-3-Clause](licenses/gpt-fast-BSD-3-Clause.txt) |
| [VITS](https://github.com/jaywalnut310/vits) | Text/convolution blocks, masking and alignment foundations | [MIT](licenses/MIT-NOTICES.txt) |
| [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) | Text encoder and flow foundations | [MIT](licenses/MIT-NOTICES.txt) |
| [Stable Audio Tools](https://github.com/Stability-AI/stable-audio-tools) | SQ codec and GAN integration | [MIT](licenses/MIT-NOTICES.txt) |
| [EnCodec](https://github.com/facebookresearch/encodec) | Convolution padding and STFT discriminator | [MIT](licenses/MIT-NOTICES.txt) |
| [MQTTS](https://github.com/b04901014/MQTTS) | Alignment attention and positional bias | [MIT](licenses/MIT-NOTICES.txt) |
| [auraloss](https://github.com/csteinmetz1/auraloss) | MR-STFT loss and adapted spectral convergence term | [Apache-2.0](licenses/Apache-2.0.txt) |
| [Kokoro](https://github.com/hexgrad/kokoro) | Vocabulary and phoneme frontend integration | [Apache-2.0](licenses/Apache-2.0.txt) |
| [PL-BERT](https://github.com/yl4579/PL-BERT) | Phoneme language-model lineage | [MIT](licenses/MIT-NOTICES.txt) |

Adaptations include PL-Flow integration, dynamic alignment masks, parametrized
weight normalization and per-example spectral convergence. Original copyright
and permission notices are retained in the linked files.

## Standalone speaker program

`src/wavlm_ecapa` uses the
[UniSpeech speaker verification implementation](https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification)
in a separate process. UniSpeech declares CC BY-SA 3.0 for its repository.
WavLM and fairseq components retain MIT terms; s3prl components retain Apache-2.0.
Source attribution, modifications and license texts are listed in
[NOTICE.md](src/wavlm_ecapa/NOTICE.md).

## Architecture reference

The STFT codec in `src/pl_flow/vocoder/stft.py` is an independent implementation
under [GPL-3.0-only](LICENSE), with
[Music2Latent](https://github.com/SonyCSLParis/music2latent) as an architectural reference.

## Pretrained weights

Pretrained models are available from
[rcell233/pl-flow](https://huggingface.co/rcell233/pl-flow):

| Components | Origin | Weight license |
| --- | --- | --- |
| `s1`, `s2`, `aligner`, `vocoder` | Project-trained from random initialization | [GPL-3.0-only](https://huggingface.co/rcell233/pl-flow/blob/main/LICENSE) |
| `plbert` | Phoneme ALBERT extracted from [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) | Apache-2.0 |
| `hubert` | [TencentGameMate/chinese-hubert-base](https://huggingface.co/TencentGameMate/chinese-hubert-base) | MIT, as declared by the upstream model card |
| `speaker` | `wavlm_large_finetune.pth`, obtained through [Seed-TTS-Eval](https://github.com/BytedanceSpeech/seed-tts-eval#metrics) | External download; upstream terms apply |

Download the [speaker checkpoint](https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view)
from the link provided by Seed-TTS-Eval and follow the
[setup instructions](README.md#models-and-inference). Its upstream terms apply.

Redistributed weights must retain their applicable licenses, attribution and
modification notices. Rights to recordings and voices are separate from model licenses.
