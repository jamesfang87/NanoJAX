from __future__ import annotations

from .. import _tensor as nj


def mse_loss(y_pred: nj.Tensor, y_true: nj.Tensor) -> nj.Tensor:
    if y_pred.numel() != y_true.numel():
        raise ValueError("mse_loss: size mismatch between predictions and labels")
    diff = y_pred - y_true
    return (diff * diff).mean()


def cross_entropy(y_pred: nj.Tensor, y_true: nj.Tensor, eps: float = 1e-6) -> nj.Tensor:
    if y_pred.numel() != y_true.numel():
        raise ValueError("cross_entropy: size mismatch between predictions and labels")
    return -(y_true * nj.log(y_pred + eps)).sum()
