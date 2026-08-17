# NanoJAX
NanoJAX is an autodiff (automatic differentiation) library for Python supporting calculations on the GPU using CUDA. It has a C++ core to support fast autodiff while maintaining the flexibility, simplicity, and development speed of Python. A C++ interface is also exposed (see main.cpp), which can be use as well. In fact, the Python interface leverages this C++ interface through bindings. In addition, there is also a small machine learning library featuring certain layers like MLPs, CNNs, and activation functions such as ReLU, Sigmoid, and more.

# Introduction
The syntax of NanoJAX is very similar compared to JAX:

A differentiable function in NanoJAX is written as an ordinary function taking and returning a `Scalar`. Take the cubic function $f(x) = x^3$ as a first example, implemented in C++ as follows.

```cpp
Scalar f(const Scalar &x) { return x * x * x; }
```

To get the derivative, we call `grad` on `f` which returns a callable derivative. This derivative is computed through reverse-mode automatic differentiation.

```cpp
auto df = grad(f);
Scalar g = df(3.0);
```

Here, it is important to note that the result of `grad(f)` is a new function `df` such that `df(x)` evaluates to $\frac{df}{dx}(x) = 3x^2$. Evaluating `to_double(g)` therefore yields $27.0$, since $3 \cdot 3.0^2 = 27.0$. Because `f` is built entirely from the overloaded `Scalar` operators, `grad` composes with any function assembled from these operators without modification, including the activation functions defined in `include/nn/layers/activation/activations.hpp`. Differentiating the ReLU activation, for instance, requires no more than `grad(relu)`.

Like in JAX, `grad()` can also easily be nested to calculate higher order derivatives:
```cpp
auto ddf = grad(grad(f));
Scalar g = df(3.0);
```

When `f` accepts more than one argument, `grad(f)` differentiates with respect to all of them at once and returns a `std::vector<Scalar>` of derivatives, one per input, in argument order, rather than a single `Scalar`. Separately, if `f` itself returns a `std::vector<Scalar>` rather than a single `Scalar`, the `argnums` parameter passed to `grad` selects which component of that output is treated as the scalar loss from which backpropagation proceeds.

# Work in Progress
Currently, NanoJAX is a work in progress. The Python bindings for the C++ core have not yet been implemented and neither has CUDA support.
