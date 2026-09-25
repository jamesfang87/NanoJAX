from __future__ import annotations


class Optimizer:
    def step(self, params, grads):
        raise NotImplementedError
