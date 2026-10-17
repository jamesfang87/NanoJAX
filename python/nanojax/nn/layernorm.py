import nanojax as nj
from nanojax._tensor import sqrt
from .module import Module


class LayerNorm(Module):
    def __init__(self, dim):
        self.weight = nj.ones([dim])
        self.bias = nj.zeros([dim])

    def forward(self, x: nj.Tensor, eps=1e-5) -> nj.Tensor:
        mu = x.mean(x.ndim - 1).unsqueeze(x.ndim - 1)
        var = ((x - mu) ** 2).mean(x.ndim - 1).unsqueeze(x.ndim - 1)
        scale = 1 / sqrt(var + eps)
        return ((x - mu) * scale) * self.weight + self.bias
