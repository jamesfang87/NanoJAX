from __future__ import annotations

import nanojax as nj
from .module import Module


class Flatten(Module):
    def forward(self, x: nj.Tensor) -> nj.Tensor:
        if x.ndim == 0:
            raise ValueError("Flatten: expected a tensor with a batch dimension")
        batch = x.shape[0]
        if batch == 0:
            raise ValueError("Flatten: cannot flatten an empty batch")
        return x.reshape([batch, x.numel() // batch])
