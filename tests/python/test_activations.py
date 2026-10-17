"""Tests for the activation modules in nanojax.nn.activations.

Run with:  python -m unittest tests.python.test_activations
or:        python tests/python/test_activations.py
"""

from __future__ import annotations

import unittest

import nanojax as nj
import nanojax.nn as nn
from nanojax.nn import activations as F


def make(shape, seed=0.0):
    """A deterministic tensor of the given shape."""
    n = 1
    for d in shape:
        n *= d
    vals = [((i * 7 + int(seed)) % 13) * 0.1 - 0.6 for i in range(n)]
    return nj.Tensor(vals).reshape(list(shape))


def assert_close(test, a, b, tol=1e-3):
    if isinstance(a, list):
        test.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            assert_close(test, x, y, tol)
    else:
        test.assertLessEqual(abs(a - b), tol, f"{a} vs {b}")


class TestActivations(unittest.TestCase):
    def test_matches_functional(self):
        x = make([2, 3, 4])
        cases = [
            (nn.Sigmoid(), F.sigmoid),
            (nn.Tanh(), F.tanh),
            (nn.ReLU(), F.relu),
            (nn.Swish(), F.swish),
            (nn.Swish(beta=0.5), lambda t: F.swish(t, 0.5)),
            (nn.LeakyReLU(), F.leaky_relu),
            (nn.LeakyReLU(alpha=0.2), lambda t: F.leaky_relu(t, 0.2)),
            (nn.ELU(), F.elu),
            (nn.ELU(alpha=0.5), lambda t: F.elu(t, 0.5)),
            (nn.SELU(), F.selu),
            (nn.GELU(), F.gelu),
            (nn.Softmax(), F.softmax),
        ]
        for module, ref in cases:
            assert_close(self, module(x).tolist(), ref(x).tolist())

    def test_are_modules_without_parameters(self):
        for module in (nn.Sigmoid(), nn.Tanh(), nn.ReLU(), nn.Swish(), nn.LeakyReLU(),
                       nn.ELU(), nn.SELU(), nn.GELU(), nn.Softmax()):
            self.assertIsInstance(module, nn.Module)
            self.assertEqual(module.named_parameters(), {})

    def test_softmax_axis(self):
        x = make([3, 2], seed=1)
        columns = nn.Softmax(axis=0)(x).tolist()
        for j in range(2):
            self.assertAlmostEqual(sum(columns[i][j] for i in range(3)), 1.0, places=4)

    def test_gradients_flow(self):
        x = make([2, 3, 4], seed=2)

        for module in (nn.Sigmoid(), nn.Tanh(), nn.ReLU(), nn.Swish(), nn.LeakyReLU(),
                       nn.ELU(), nn.SELU(), nn.GELU(), nn.Softmax()):
            _, grads = nj.value_and_grad(lambda t: (module(t) ** 2).sum())(x)
            self.assertEqual(grads.shape, x.shape)

    def test_used_in_a_network(self):
        class Net(nn.Module):
            def __init__(self):
                self.fc1 = nn.MLP(4, 5, seed=0)
                self.act = nn.ReLU()
                self.fc2 = nn.MLP(5, 2, seed=1)

            def forward(self, t):
                return self.fc2(self.act(self.fc1(t)))

        net = Net()
        value, grads = nj.value_and_grad(
            lambda m: (m(make([3, 4], seed=3)) ** 2).sum()
        )(net)
        self.assertTrue(value.item() > 0.0)
        self.assertEqual(sorted(grads.named_parameters()),
                         ["fc1.bias", "fc1.weight", "fc2.bias", "fc2.weight"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
