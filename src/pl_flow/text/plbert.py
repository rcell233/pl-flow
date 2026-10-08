"""Frozen Kokoro PL-BERT; the unused TTS projection is intentionally omitted."""

from torch import nn
from transformers import AlbertConfig, AlbertModel

from .frontend import KOKORO_VOCAB


class PLBert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = AlbertModel(AlbertConfig(**config))
        self.vocab = KOKORO_VOCAB
        self.requires_grad_(False).eval()

    def train(self, mode=True):
        return super().train(False)

    def forward(self, ids, attention_mask=None):
        return self.bert(ids, attention_mask=attention_mask).last_hidden_state
