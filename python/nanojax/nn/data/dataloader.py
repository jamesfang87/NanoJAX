from __future__ import annotations

import random

from ... import _tensor as nj
from .dataset import Dataset, Sample


def _collate(samples: list[Sample]) -> Sample:
    x = nj.Tensor([sample.x.tolist() for sample in samples])
    y = nj.Tensor([sample.y.tolist() for sample in samples])
    return Sample(x, y)


class DataLoader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("DataLoader: batch_size must be positive")

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return (len(self.indices) + self.batch_size - 1) // self.batch_size

    def reset(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def batch(self, batch_index: int) -> list[Sample]:
        if not 0 <= batch_index < len(self):
            raise IndexError("DataLoader: batch_index out of range")

        begin = batch_index * self.batch_size
        end = min(begin + self.batch_size, len(self.indices))
        return [self.dataset[self.indices[i]] for i in range(begin, end)]

    def __iter__(self):
        self.reset()
        for batch_index in range(len(self)):
            yield _collate(self.batch(batch_index))
