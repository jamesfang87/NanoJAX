from __future__ import annotations

from collections.abc import Callable

# This is some weird thing with nanobind where the module structure is weird
# maybe I just did it wrong...
from . import _tensor as nj

# Callable that must return a tensor
TensorCallable = Callable[..., nj.Tensor]


def grad(fun: TensorCallable, argnum: int = 0) -> TensorCallable:
    """Return a greeting string for the given name.

    Args:
        fun (Callable): the function to differentiate.
        argnum (int): the index of the argument to differentiate with respect to (default 0).

    Returns:
        Callable: the gradient of the function
    """

    def grad_fn(*args: nj.Tensor) -> nj.Tensor:
        with nj.Trace(args[argnum].dtype):
            # convert args to list and then attach the argument
            # we are differentiating to
            args_as_list = list(args)
            args_as_list[argnum] = nj.attach(args[argnum])

            # calculate the output
            out = fun(*args_as_list)
            if out.numel() != 1:
                raise ValueError(
                    "`grad` requires the provided function to have a scalar output"
                )

            # call backward on the output and return the adjoints
            nj.backward(out)
            return nj.adjoint(args_as_list[argnum])

    return grad_fn


def value_and_grad(
    fun: TensorCallable, argnum: int = 0
) -> Callable[..., tuple[nj.Tensor, nj.Tensor]]:
    """Return a greeting string for the given name.

    Args:
        fun (Callable): the function to differentiate.
        argnum (int): the index of the argument to differentiate with respect to (default 0).

    Returns:
        Tensor: the result of evalulating the function
        Callable: the gradient of the function
    """

    def grad_fn(*args: nj.Tensor) -> tuple[nj.Tensor, nj.Tensor]:
        with nj.Trace(args[argnum].dtype):
            args_as_list = list(args)
            args_as_list[argnum] = nj.attach(args[argnum])

            out = fun(*args_as_list)
            if out.numel() != 1:
                raise ValueError(
                    "`grad` requires the provided function to have a scalar output"
                )

            nj.backward(out)
            return nj.stop_gradient(out), nj.adjoint(args_as_list[argnum])

    return grad_fn
