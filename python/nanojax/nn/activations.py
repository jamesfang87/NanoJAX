from __future__ import annotations

from .. import _tensor as nj
from .module import Module


def sigmoid(x: nj.Tensor) -> nj.Tensor:
    return nj.sigmoid(x)


def swish(x: nj.Tensor, beta: float = 1.0) -> nj.Tensor:
    return x * sigmoid(beta * x)


def tanh(x: nj.Tensor) -> nj.Tensor:
    return nj.tanh(x)


def relu(x: nj.Tensor) -> nj.Tensor:
    return nj.relu(x)


def leaky_relu(x: nj.Tensor, alpha: float = 0.01) -> nj.Tensor:
    return nj.relu(x) - alpha * nj.relu(-x)


def elu(x: nj.Tensor, alpha: float = 1.0) -> nj.Tensor:
    return nj.relu(x) + alpha * (nj.exp(-nj.relu(-x)) - 1.0)


def selu(x: nj.Tensor, lam: float = 1.0507, alpha: float = 1.6733) -> nj.Tensor:
    return lam * elu(x, alpha)


def gelu(x: nj.Tensor) -> nj.Tensor:
    scale = (2.0 / 3.141592653589793) ** 0.5
    return 0.5 * x * (1.0 + tanh(scale * (x + 0.044715 * x.pow(3.0))))


def softmax(logits: nj.Tensor, axis: int = -1) -> nj.Tensor:
    axis = axis % logits.ndim
    shifted = logits - logits.max().item()
    exps = nj.exp(shifted)
    return exps / exps.sum(axis).unsqueeze(axis)


class Sigmoid(Module):
    def forward(self, x: nj.Tensor) -> nj.Tensor:
        return sigmoid(x)


class Swish(Module):
    def __init__(self, beta: float = 1.0) -> None:
        self.beta = beta

    def forward(self, x: nj.Tensor) -> nj.Tensor:
        return swish(x, self.beta)


class Tanh(Module):
    def forward(self, x: nj.Tensor) -> nj.Tensor:
        return tanh(x)


class ReLU(Module):
    def forward(self, x: nj.Tensor) -> nj.Tensor:
        return relu(x)


class LeakyReLU(Module):
    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha = alpha

    def forward(self, x: nj.Tensor) -> nj.Tensor:
        return leaky_relu(x, self.alpha)


class ELU(Module):
    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def forward(self, x: nj.Tensor) -> nj.Tensor:
        return elu(x, self.alpha)


class SELU(Module):
    def __init__(self, lam: float = 1.0507, alpha: float = 1.6733) -> None:
        self.lam = lam
        self.alpha = alpha

    def forward(self, x: nj.Tensor) -> nj.Tensor:
        return selu(x, self.lam, self.alpha)


class GELU(Module):
    def forward(self, x: nj.Tensor) -> nj.Tensor:
        return gelu(x)


class Softmax(Module):
    def __init__(self, axis: int = -1) -> None:
        self.axis = axis

    def forward(self, x: nj.Tensor) -> nj.Tensor:
        return softmax(x, self.axis)
