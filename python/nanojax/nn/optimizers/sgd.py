from __future__ import annotations

from ... import _tensor as nj
from .optim import Optimizer


def _tree_map2(a, b, fn):
    if a is None:
        return None
    if isinstance(a, nj.Tensor):
        return fn(a, b)
    if isinstance(a, tuple):
        children = [_tree_map2(x, y, fn) for x, y in zip(a, b)]
        if type(a) is tuple:
            return tuple(children)
        return type(a)(*children)
    if isinstance(a, list):
        return [_tree_map2(x, y, fn) for x, y in zip(a, b)]
    raise TypeError(f"SGD: unsupported leaf of type {type(a).__name__}")


class SGD(Optimizer):
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate

    def step(self, params, grads):
        return _tree_map2(params, grads, lambda p, g: p - self.learning_rate * g)
