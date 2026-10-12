from __future__ import annotations

from collections.abc import Callable
from typing import Tuple

# This is some weird thing with nanobind where the module structure is weird
# maybe I just did it wrong...
from . import _tensor as nj

# Callable that must return a tensor
TensorCallable = Callable[..., nj.Tensor]


def tree_map(f, tree):
    if tree is None:
        return None
    if isinstance(tree, nj.Tensor):
        return f(tree)
    if isinstance(tree, tuple):
        children = [tree_map(f, child) for child in tree]
        if type(tree) is tuple:
            return tuple(children)
        return type(tree)(*children)
    if isinstance(tree, list):
        return [tree_map(f, child) for child in tree]
    if callable(getattr(tree, "named_parameters", None)):
        return tree.with_parameters(
            {
                name: tree_map(f, child)
                for name, child in tree.named_parameters().items()
            }
        )
    raise TypeError(f"grad: unsupported leaf of type {type(tree).__name__}")


def flatten(node):
    """Collect the Tensors in a nested structure of lists, tuples, and modules.

    Args:
        tree: A Tensor, or a nested structure of Tensors, lists, tuples, and
            objects exposing @c named_parameters; None entries are skipped.

    Returns:
        list[Tensor]: the leaves in left-to-right order.
    """
    if node is None:
        return
    if isinstance(node, nj.Tensor):
        yield node
    elif isinstance(node, (list, tuple)):
        for child in node:
            yield from flatten(child)
    elif callable(getattr(node, "named_parameters", None)):
        for child in node.named_parameters().values():
            yield from flatten(child)
    else:
        raise TypeError(f"grad: unsupported leaf of type {type(node).__name__}")


def value_and_grad(
    f: TensorCallable, argnum: int | Tuple[int] = 0
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
        # Ensure argnums is a tuple
        argnums = (argnum,) if isinstance(argnum, int) else argnum

        # Get the args which are to be differentiated w.r.t
        targets = tuple(args[i] for i in argnums)

        # Flatten the targets
        ref = next(flatten(targets), None)
        if ref is None:
            raise ValueError("value_and_grad: no Tensor arguments to differentiate")

        with nj.Trace(ref.dtype):
            # Attach each target to the Trace
            attached_targets = tree_map(nj.attach, targets)

            # Change the differentiated args to the targets
            args_as_list = list(args)
            for i, j in enumerate(argnums):
                args_as_list[j] = attached_targets[i]

            # Forward pass to generate gradients
            out = f(*args_as_list)
            if out.numel() != 1:
                raise ValueError(
                    "`grad` requires the provided function to have a scalar output"
                )

            # Extract gradients
            nj.backward(out)
            grads = tree_map(nj.adjoint, attached_targets)

        return nj.stop_gradient(out), (
            grads[0] if isinstance(argnum, int) else tuple(grads)
        )

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
