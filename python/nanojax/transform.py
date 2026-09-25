from __future__ import annotations

from collections.abc import Callable

# This is some weird thing with nanobind where the module structure is weird
# maybe I just did it wrong...
from . import _tensor as nj

# Callable that must return a tensor
TensorCallable = Callable[..., nj.Tensor]


def flatten(tree) -> list[nj.Tensor]:
    """Collect the Tensors in a nested structure of lists and tuples.

    Args:
        tree: A Tensor, or a nested list/tuple of Tensors; None entries are skipped.

    Returns:
        list[Tensor]: the leaves in left-to-right order.
    """
    leaves: list[nj.Tensor] = []

    def walk(node) -> None:
        if node is None:
            return
        if isinstance(node, nj.Tensor):
            leaves.append(node)
        elif isinstance(node, (list, tuple)):
            for child in node:
                walk(child)
        else:
            raise TypeError(f"grad: unsupported leaf of type {type(node).__name__}")

    walk(tree)
    return leaves


def unflatten_like(template, values):
    """Rebuild a structure shaped like @p template from @p values.

    Args:
        template: The nested structure whose shape is mirrored.
        values: Iterable of leaves consumed in the same order as @ref flatten.

    Returns:
        A structure shaped like @p template, holding @p values.
    """
    it = iter(values)

    def build(node):
        if node is None:
            return None
        if isinstance(node, nj.Tensor):
            return next(it)
        if isinstance(node, tuple):
            children = [build(child) for child in node]
            if type(node) is tuple:
                return tuple(children)
            return type(node)(*children)
        if isinstance(node, list):
            return [build(child) for child in node]
        raise TypeError(f"grad: unsupported leaf of type {type(node).__name__}")

    return build(template)


def value_and_grad(
    fun: TensorCallable, argnum: int = 0
) -> Callable[..., tuple[nj.Tensor, nj.Tensor]]:
    """Differentiate @p fun with respect to argument @p argnum.

    Args:
        fun (Callable): the function to differentiate.
        argnum (int): the index of the argument to differentiate with respect to (default 0).

    Returns:
        Callable: a function returning the output and the gradient, each shaped
        like the differentiated argument.
    """

    def grad_fn(*args):
        target = args[argnum]
        leaves = flatten(target)
        if not leaves:
            raise ValueError("value_and_grad: no Tensor arguments to differentiate")

        with nj.Trace(leaves[0].dtype):
            attached = [nj.attach(leaf) for leaf in leaves]
            traced_target = unflatten_like(target, attached)

            args_as_list = list(args)
            args_as_list[argnum] = traced_target

            out = fun(*args_as_list)
            if out.numel() != 1:
                raise ValueError(
                    "`grad` requires the provided function to have a scalar output"
                )

            nj.backward(out)
            grads = [nj.adjoint(leaf) for leaf in attached]

        return nj.stop_gradient(out), unflatten_like(target, grads)

    return grad_fn


def grad(fun: TensorCallable, argnum: int = 0) -> TensorCallable:
    """Return the gradient of @p fun with respect to argument @p argnum.

    Args:
        fun (Callable): the function to differentiate.
        argnum (int): the index of the argument to differentiate with respect to (default 0).

    Returns:
        Callable: the gradient of the function.
    """
    value_and_grad_fun = value_and_grad(fun, argnum)

    def grad_fn(*args):
        return value_and_grad_fun(*args)[1]

    return grad_fn
