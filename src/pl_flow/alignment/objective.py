"""Original sequence-to-sequence and monotonic-path alignment objectives."""

import numpy as np
import torch
from monotonic_align import mask_from_lens
from monotonic_align.core import maximum_path_c
from torch.nn import functional as F


def maximum_path(scores, mask):
    # The C dynamic-programming kernel overwrites its score array. Always copy:
    # on CPU, .numpy() would otherwise alias attention needed by the loss.
    values = np.array(scores.detach().float().cpu().numpy(), dtype=np.float32, order="C", copy=True)
    path = np.zeros(values.shape, dtype=np.int32)
    tokens = np.ascontiguousarray(mask.sum(1)[:, 0].cpu().numpy().astype(np.int32))
    frames = np.ascontiguousarray(mask.sum(2)[:, 0].cpu().numpy().astype(np.int32))
    if (tokens > frames).any() or (tokens <= 0).any():
        raise ValueError("Each aligned phoneme needs at least one HuBERT frame")
    maximum_path_c(path, values, tokens, frames)
    return torch.from_numpy(path).to(scores)


def monotonic_loss(attention, text_lengths, frame_lengths):
    attention = attention[:, 1:, :]
    text_mask = (
        torch.arange(attention.size(1), device=attention.device)[None] < text_lengths[:, None]
    )
    frame_mask = (
        torch.arange(attention.size(2), device=attention.device)[None] < frame_lengths[:, None]
    )
    attention = attention.masked_fill(~(text_mask[:, :, None] & frame_mask[:, None]), 0)
    with torch.no_grad():
        path = maximum_path(attention, mask_from_lens(attention, text_lengths, frame_lengths))
    return F.l1_loss(attention, path) * 10, path


def alignment_objective(
    model, text, text_lengths, features, feature_lengths, *, step=0, mono_start_step=100000
):
    _, logits, attention = model(
        features, src_key_padding_mask=model.length_to_mask(feature_lengths), text_input=text
    )
    mono, path = monotonic_loss(attention, text_lengths, feature_lengths)
    ce = sum(
        F.cross_entropy(pred[: int(n)], target[: int(n)], ignore_index=-1)
        for pred, target, n in zip(logits, text, text_lengths)
    ) / len(text)
    return {
        "loss": ce + mono * (step >= mono_start_step),
        "ce_loss": ce,
        "mono_loss": mono,
        "path": path,
    }


@torch.inference_mode()
def align(model, hubert, waves, lengths, ids, id_lengths):
    features, feature_lengths = hubert.aligned(waves, lengths)
    result = alignment_objective(model, ids, id_lengths, features, feature_lengths)
    return result["path"].sum(-1).long()
