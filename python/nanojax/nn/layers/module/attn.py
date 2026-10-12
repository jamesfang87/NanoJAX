from __future__ import annotations


from ...._tensor import Tensor
from .mlp import MLP
from .module import Module


class MultiHeadAttention(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int,
        use_bias: bool = True,
        seed: int | None = None,
    ) -> None:
        if in_features % num_heads != 0:
            raise ValueError("in_features must be divisible by num_heads")

        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.head_dim = in_features // num_heads
        self.use_bias = use_bias

        seeds = [seed + i for i in range(4)] if seed is not None else [None] * 4
        self.q_proj = MLP(in_features, in_features, use_bias, seed=seeds[0])
        self.k_proj = MLP(in_features, in_features, use_bias, seed=seeds[1])
        self.v_proj = MLP(in_features, in_features, use_bias, seed=seeds[2])
        self.out_proj = MLP(in_features, out_features, use_bias, seed=seeds[3])

    def forward(self, x) -> Tensor:
        raise NotImplementedError

    def split_heads(self, x: Tensor, batch_size: int) -> Tensor:
        raise NotImplementedError

    def combine_heads(self, x: Tensor, batch_size: int) -> Tensor:
        raise NotImplementedError
