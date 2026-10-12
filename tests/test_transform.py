from __future__ import annotations

import glob
import os
import pathlib
import sys
import types
import unittest


def _ensure_nanojax_importable() -> None:
    """Make the in-tree package importable, including the compiled extension."""
    try:
        import nanojax  # noqa: F401

        return
    except ImportError:
        pass

    root = pathlib.Path(__file__).resolve().parents[1]
    src = root / "python"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    pkg = src / "nanojax"
    if not list(pkg.glob("_tensor*.so")) and not list(pkg.glob("_tensor*.pyd")):
        for so in glob.glob(str(root / "build" / "**" / "_tensor*.so"), recursive=True):
            os.symlink(so, pkg / os.path.basename(so))
            break


_ensure_nanojax_importable()

import nanojax as nj  # noqa: E402
from nanojax.nn.layers import MLP, MLPParams, Module  # noqa: E402
from nanojax.transform import flatten, grad, tree_map, value_and_grad  # noqa: E402


def scalar(value: float) -> nj.Tensor:
    return nj.Tensor([float(value)])


def vector(values) -> nj.Tensor:
    return nj.Tensor([float(v) for v in values])


class TreeMapTest(unittest.TestCase):
    def test_applies_to_tensor(self):
        x = scalar(2.0)
        self.assertIs(tree_map(lambda t: t, x), x)

    def test_list_structure_preserved(self):
        x, y = scalar(1.0), scalar(2.0)
        out = tree_map(lambda t: t * 2.0, [x, y])
        self.assertIsInstance(out, list)
        self.assertTrue(out[0].allclose(scalar(2.0), 1e-6))
        self.assertTrue(out[1].allclose(scalar(4.0), 1e-6))

    def test_tuple_structure_preserved(self):
        x, y = scalar(1.0), scalar(2.0)
        out = tree_map(lambda t: t * 2.0, (x, y))
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        self.assertTrue(out[0].allclose(scalar(2.0), 1e-6))

    def test_namedtuple_type_preserved(self):
        params = MLPParams(w=nj.ones([2, 2]), b=nj.ones([2]))
        out = tree_map(lambda t: t, params)
        self.assertIsInstance(out, MLPParams)
        self.assertTrue(out.w.allclose(params.w, 1e-6))

    def test_none_stays_none(self):
        self.assertIsNone(tree_map(lambda t: t, None))

    def test_module_rebuilt_without_mutating_original(self):
        model = MLP(2, 2, seed=0)
        before = model.weight.tolist()
        out = tree_map(lambda t: t * 0.0, model)
        self.assertIsInstance(out, Module)
        self.assertTrue(out.weight.allclose(nj.zeros([2, 2]), 1e-6))
        self.assertEqual(model.weight.tolist(), before)

    def test_unsupported_leaf_raises(self):
        with self.assertRaises(TypeError):
            tree_map(lambda t: t, 5)


class FlattenTest(unittest.TestCase):
    def test_is_lazy_generator(self):
        self.assertIsInstance(flatten(scalar(1.0)), types.GeneratorType)

    def test_left_to_right_order(self):
        x, y, z = scalar(1.0), scalar(2.0), scalar(3.0)
        self.assertEqual([id(t) for t in flatten([x, (y, z)])], [id(x), id(y), id(z)])

    def test_none_skipped(self):
        x, y = scalar(1.0), scalar(2.0)
        self.assertEqual([id(t) for t in flatten((x, None, y))], [id(x), id(y)])

    def test_module_order_matches_named_parameters(self):
        model = MLP(2, 2, seed=0)
        self.assertEqual(
            [id(t) for t in flatten(model)],
            [id(t) for t in model.named_parameters().values()],
        )

    def test_unsupported_leaf_raises(self):
        with self.assertRaises(TypeError):
            list(flatten(5))


class ValueAndGradTest(unittest.TestCase):
    def test_scalar_polynomial(self):
        def f(x):
            return (x * x * x).sum()

        value, gradient = value_and_grad(f)(vector([2.0]))
        self.assertAlmostEqual(value.item(), 8.0, places=5)
        self.assertTrue(gradient.allclose(vector([12.0]), 1e-4))

    def test_grad_returns_only_gradient(self):
        def f(x):
            return (x * x).sum()

        self.assertTrue(grad(f)(vector([3.0])).allclose(vector([6.0]), 1e-4))

    def test_higher_order(self):
        def f(x):
            return (x * x * x).sum()

        self.assertTrue(grad(grad(f))(vector([2.0])).allclose(vector([12.0]), 1e-4))

    def test_argnum_selects_argument(self):
        def f(a, b):
            return (a * a * b).sum()

        a = nj.Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = nj.ones([2, 2])
        self.assertTrue(grad(f, argnum=1)(a, b).allclose(a * a, 1e-4))

    def test_tuple_argnum_returns_tuple(self):
        def f(a, b):
            return (a * a * b).sum()

        a = nj.Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = nj.ones([2, 2])
        value, grads = value_and_grad(f, argnum=(0, 1))(a, b)
        self.assertIsInstance(grads, tuple)
        self.assertTrue(grads[0].allclose(a * b * 2.0, 1e-4))
        self.assertTrue(grads[1].allclose(a * a, 1e-4))

    def test_list_target_shaped_like_input(self):
        def f(params):
            x, y = params
            return (x * x + y * y).sum()

        x, y = vector([1.0, 2.0]), vector([3.0, 4.0])
        value, grads = value_and_grad(f)([x, y])
        self.assertIsInstance(grads, list)
        self.assertTrue(grads[0].allclose(x * 2.0, 1e-4))
        self.assertTrue(grads[1].allclose(y * 2.0, 1e-4))

    def test_tuple_target_shaped_like_input(self):
        def f(params):
            x, y = params
            return (x * x + y * y).sum()

        x, y = vector([1.0, 2.0]), vector([3.0, 4.0])
        value, grads = value_and_grad(f)((x, y))
        self.assertIsInstance(grads, tuple)

    def test_module_gradient_shaped_like_module(self):
        model = MLP(3, 2, seed=0)
        x = nj.Tensor([[0.1, 0.2, 0.3]])

        def loss(m):
            return (m(x) * m(x)).sum()

        before = model.weight.tolist()
        value, grads = value_and_grad(loss)(model)
        self.assertIsInstance(grads, MLP)
        self.assertEqual(set(grads.named_parameters()), set(model.named_parameters()))
        for name, param in model.named_parameters().items():
            self.assertEqual(grads.named_parameters()[name].shape, param.shape)
        self.assertEqual(model.weight.tolist(), before)

    def test_finite_difference_matches(self):
        def f(x):
            return (x * x * x).sum()

        x = vector([0.5, -1.5, 2.0])
        analytic = grad(f)(x).tolist()
        eps = 1e-3
        base = x.tolist()
        for i in range(len(base)):
            plus = list(base)
            minus = list(base)
            plus[i] += eps
            minus[i] -= eps
            numeric = (f(vector(plus)).item() - f(vector(minus)).item()) / (2 * eps)
            self.assertAlmostEqual(analytic[i], numeric, places=2)

    def test_non_scalar_output_raises(self):
        def f(x):
            return x * x

        with self.assertRaises(ValueError):
            value_and_grad(f)(vector([1.0, 2.0]))

    def test_empty_targets_raises(self):
        with self.assertRaises(ValueError):
            value_and_grad(lambda: scalar(1.0), argnum=())()

    def test_non_tensor_argument_raises(self):
        with self.assertRaises(TypeError):
            value_and_grad(lambda a: a)(5)


if __name__ == "__main__":
    unittest.main()
