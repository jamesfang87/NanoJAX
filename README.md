# NanoJAX
NanoJAX is an autodiff (automatic differentiation) library for Python supporting hardware acceleration using both BLAS and CUDA. It uses a C++ core to support fast autodiff while maintaining the flexibility, simplicity, and development speed of Python. In addition, there is also a small machine learning library featuring certain layers like MLPs, and activation functions such as ReLU, Sigmoid, and more.

# Introduction
The syntax of NanoJAX is very similar compared to JAX:

A differentiable function in NanoJAX is written as an ordinary function taking and returning a `nanojax.Tensor`. Take the cubic function $f(x) = x^3$ as a first example, implemented in Python as follows.

```python
def f(x):
    return x * x * x;

```

To get the derivative, we call `grad()` on `f` which returns a callable derivative. This derivative is computed through reverse-mode automatic differentiation.

```python
df = nanojax.grad(f);
g = df(3.0);
```

Here, it is important to note that the result of `grad(f)` is a new function `df` such that `df(x)` evaluates to $\frac{df}{dx}(x) = 3x^2$. 
When `f` accepts more than one argument, `grad(f)` differentiates with respect to the argument named by the parameter `argnum`, which can be an integer or a tuple of integers. By default, `argnum = 0`.

When using `grad()`, there are a couple of restrictions on the function which can be inputted. The function must produce a scalar output (that is, it cannot have a rank greater than 0). The dtype of the Tensors being inputted must be floating point numbers.

Like in JAX, `grad()` can also easily be nested to calculate higher order derivatives:
```python
ddf = grad(grad(f));
g = df(3.0);
```


# Notes
Everything else is implemented except for CUDA support; I forgot to bring my system with CUDA to college, so hardware acceleration with CUDA will likely have to wait until the next time I go home, since I do not have access to a system with CUDA installed. I also forgot to implement strided views on Tensors, so CNNs will have to wait too. However, I will likely implement them with im2col rather an strided views.
