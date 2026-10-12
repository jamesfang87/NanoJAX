from __future__ import annotations

from ...transform import flatten
from .optim import Optimizer


class SGD(Optimizer):
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate

    def step(self, params, grads):
        for param, grad in zip(flatten(params), flatten(grads)):
            param -= self.learning_rate * grad
        return params
