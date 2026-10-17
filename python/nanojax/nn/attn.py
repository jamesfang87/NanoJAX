from __future__ import annotations


import nanojax as nj
from .activations import softmax
from .mlp import MLP
from .module import Module

import math


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

    def forward(self, x) -> nj.Tensor:
        q = self.split_heads(self.q_proj(x))
        k = self.split_heads(self.k_proj(x))
        v = self.split_heads(self.v_proj(x))

        o, _ = MultiHeadAttention._scaled_dot_product(q, k, v)
        return self.out_proj(self.combine_heads(o))

    @staticmethod
    def _scaled_dot_product(q: nj.Tensor, k: nj.Tensor, v: nj.Tensor, mask=None):
        d_k = q.shape[-1]
        attn = (q @ MultiHeadAttention._transpose_last_two(k)) / math.sqrt(d_k)
        if mask is not None:
            attn = attn + mask

        attn = softmax(attn)
        return attn @ v, attn

    @staticmethod
    def _transpose_last_two(x: nj.Tensor) -> nj.Tensor:
        axes = list(range(x.ndim))
        axes[-2], axes[-1] = axes[-1], axes[-2]
        return x.permute(axes)

    def split_heads(self, x: nj.Tensor) -> nj.Tensor:
        shape = x.shape
        # x should be in (B, T, H, d)
        x = x.reshape([shape[0], shape[1], self.num_heads, self.head_dim])
        x = x.permute([0, 2, 1, 3])
        return x

    def combine_heads(self, x: nj.Tensor) -> nj.Tensor:
        b, h, t, d = x.shape
        x = x.permute([0, 2, 1, 3])
        x = x.reshape([b, t, h * d])
        return x
