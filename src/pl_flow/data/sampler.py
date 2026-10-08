"""Deterministic token-budget batches with an explicit resumable cursor."""

import numpy as np
from torch.utils.data import Sampler


class BatchSampler(Sampler):
    def __init__(self, lengths, batch_size, token_budget, seed=1234, epoch=0, offset=0):
        self.lengths = np.asarray(lengths, dtype=np.int64)
        if len(self.lengths) == 0 or (self.lengths <= 0).any():
            raise ValueError("Sample lengths must be positive")
        if batch_size < 1 or token_budget < int(self.lengths.max()):
            raise ValueError("A sample exceeds the padded token budget")
        self.batch_size, self.token_budget = batch_size, token_budget
        self.seed, self.epoch, self.offset = seed, epoch, offset

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(len(self.lengths))
        batches = []
        for start in range(0, len(order), 4096):
            block = order[start : start + 4096]
            block = block[np.argsort(self.lengths[block], kind="stable")]
            batch = []
            for index in block:
                if batch and (
                    len(batch) >= self.batch_size
                    or (len(batch) + 1) * self.lengths[index] > self.token_budget
                ):
                    batches.append(batch)
                    batch = []
                batch.append(int(index))
            if batch:
                batches.append(batch)
        rng.shuffle(batches)
        for batch in batches[self.offset :]:
            yield [(self.epoch, index) for index in batch]
