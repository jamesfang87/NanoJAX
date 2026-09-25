import nanojax


def f(x):
    return x * x


df = nanojax.grad(f)  # f'(x)
ddf = nanojax.grad(nanojax.grad(f))

x1 = nanojax.ones([])  # Scalar tensor holding 1.0

print(df(x1))  # f'(x) = 2x; f'(1) = 2
print(ddf(x1))  # f''(x) = 2; f'(1) = 2
