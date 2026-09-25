from __future__ import annotations

from typing import NamedTuple, Optional

from .... import _tensor as nj


class MLPParams(NamedTuple):
    w: nj.Tensor
    b: Optional[nj.Tensor]


class MLP:
    def __init__(
        self, in_features: int, out_features: int, use_bias: bool = True
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

    def init(self) -> MLPParams:
        bias = nj.zeros([self.out_features]) if self.use_bias else None
        return MLPParams(nj.zeros([self.in_features, self.out_features]), bias)

    @staticmethod
    def apply(params: MLPParams, x: nj.Tensor) -> nj.Tensor:
        if params.b is None:
            return x @ params.w
        return x @ params.w + params.b

    def input_features(self) -> int:
        return self.in_features

    def output_features(self) -> int:
        return self.out_features
