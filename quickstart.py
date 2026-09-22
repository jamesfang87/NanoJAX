import nanojax


def f(x):
    return x * x


df = nanojax.grad(f)  # f'(x)

x1 = nanojax.ones([])  # Scalar tensor holding 1.0
x2 = nanojax.full([], 2.0)  # Scalar tensor holding 2.0

print(df(x1))  # f'(x) = 2x; f'(1) = 2
print(df(x2))  # f'(x) = 2x; f'(2) = 4
