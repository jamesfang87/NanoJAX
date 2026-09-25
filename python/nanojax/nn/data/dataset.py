from __future__ import annotations

from typing import NamedTuple

from ... import _tensor as nj


class Sample(NamedTuple):
    x: nj.Tensor
    y: nj.Tensor


class Dataset:
    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Sample:
        raise NotImplementedError
