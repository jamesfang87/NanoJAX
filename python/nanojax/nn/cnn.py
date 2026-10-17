from __future__ import annotations

import math
import random
from typing import NamedTuple

from .. import _tensor as nj
from .module import Module


class Conv2DParams(NamedTuple):
    kernel: nj.Tensor
    bias: nj.Tensor | None
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int = 1
    padding: int = 0
    dilation: int = 1
    use_bias: bool = True


class Conv2D(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        use_bias: bool = True,
        seed: int | None = None,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.use_bias = use_bias

        rng = random.Random(seed) if seed is not None else random
        bound = 1.0 / math.sqrt(in_channels * kernel_size * kernel_size)

        n = out_channels * in_channels * kernel_size * kernel_size
        flat = [rng.uniform(-bound, bound) for _ in range(n)]
        self.kernel = nj.Tensor(flat).reshape(
            [out_channels, in_channels, kernel_size, kernel_size]
        )
        self.bias = (
            nj.Tensor([rng.uniform(-bound, bound) for _ in range(out_channels)])
            if use_bias
            else None
        )

    def forward(self, x: nj.Tensor) -> nj.Tensor:
        y = nj.conv2d(
            x,
            self.kernel,
            self.out_channels,
            self.kernel_size,
            self.kernel_size,
            self.dilation,
            self.dilation,
            self.padding,
            self.padding,
            self.stride,
            self.stride,
        )
        if self.bias is not None:
            y = y + self.bias.reshape([1, self.out_channels, 1, 1])
        return y

    @staticmethod
    def apply(params: Conv2DParams, x: nj.Tensor) -> nj.Tensor:
        y = nj.conv2d(
            x,
            params.kernel,
            params.out_channels,
            params.kernel_size,
            params.kernel_size,
            params.dilation,
            params.dilation,
            params.padding,
            params.padding,
            params.stride,
            params.stride,
        )
        if params.use_bias and params.bias is not None:
            y = y + params.bias.reshape([1, params.out_channels, 1, 1])
        return y
