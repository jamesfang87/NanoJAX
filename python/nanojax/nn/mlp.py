from __future__ import annotations

import math
import random
from typing import NamedTuple, Optional

import nanojax as nj
from .module import Module


class MLPParams(NamedTuple):
    w: nj.Tensor
    b: Optional[nj.Tensor]


class MLP(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        seed: int | None = None,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

        rng = random.Random(seed) if seed is not None else random
        bound = 1.0 / math.sqrt(in_features)
        self.weight = nj.Tensor(
            [
                [rng.uniform(-bound, bound) for _ in range(out_features)]
                for _ in range(in_features)
            ]
        )
        self.bias = (
            nj.Tensor([rng.uniform(-bound, bound) for _ in range(out_features)])
            if use_bias
            else None
        )

    def forward(self, x: nj.Tensor) -> nj.Tensor:
        if self.bias is None:
            return x @ self.weight
        return x @ self.weight + self.bias

    @staticmethod
    def apply(params: MLPParams, x: nj.Tensor) -> nj.Tensor:
        if params.b is None:
            return x @ params.w
        return x @ params.w + params.b
