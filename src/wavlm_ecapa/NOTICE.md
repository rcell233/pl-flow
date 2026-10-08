# Speaker tool licenses and attribution

This standalone speaker embedding program is distributed alongside PL-Flow
under the component licenses below.

- `model.py` and `ecapa.py` follow Microsoft's
  [UniSpeech speaker verification implementation](https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification),
  including its attribution to [lawlict/ECAPA-TDNN](https://github.com/lawlict/ECAPA-TDNN).
  The UniSpeech repository license, **CC BY-SA 3.0**, is reproduced in [LICENSE](LICENSE).
  Adaptations include local checkpoint loading, frozen feature extraction, length-aware
  batching, and PyTorch parametrized weight normalization. The command-line interface
  and other original files in this package are also provided under CC BY-SA 3.0.
- `wavlm/` retains the Microsoft WavLM copyright headers and **MIT** license in
  [wavlm/LICENSE](wavlm/LICENSE). The implementation originates from
  [Microsoft UniLM/WavLM](https://github.com/microsoft/unilm/tree/master/wavlm).
  [fairseq's MIT notice](licenses/fairseq-MIT.txt) and
  [s3prl's Apache-2.0 license](licenses/s3prl-Apache-2.0.txt) are retained for the
  incorporated layers and integration provenance.

PyTorch, NumPy, safetensors and SoundFile are installed dependencies.
The speaker checkpoint used by PL-Flow is `wavlm_large_finetune.pth`, obtained
through the [Seed-TTS-Eval model link](https://github.com/BytedanceSpeech/seed-tts-eval#metrics).
Download the [speaker checkpoint](https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view)
and place it beside the speaker `config.json`. The checkpoint is subject to its
upstream terms.
