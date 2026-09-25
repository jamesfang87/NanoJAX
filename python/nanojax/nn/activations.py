from __future__ import annotations

from .. import _tensor as nj


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
