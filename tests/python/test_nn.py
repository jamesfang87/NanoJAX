"""Tests for the nn layer modules (Conv2D, Flatten).

Run with:  python -m unittest tests.python.test_nn
or:        python tests/python/test_nn.py
"""

from __future__ import annotations

import math
import unittest

import nanojax as nj
from nanojax.nn import Conv2D, Flatten, MultiHeadAttention


def make(shape, seed=0.0):
    """A deterministic tensor of the given shape (float32, like the modules)."""
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


def ref_conv2d(x, w, bias, stride, pad, dilation):
    """Naive NCHW convolution, independent of the library."""
    xt = x.tolist()
    wt = w.tolist()
    bt = None if bias is None else bias.tolist()
    N, Cin, H, W = x.shape
    Cout, _, KH, KW = w.shape
    OH = (H + 2 * pad - dilation * (KH - 1) - 1) // stride + 1
    OW = (W + 2 * pad - dilation * (KW - 1) - 1) // stride + 1
    out = [[[[0.0] * OW for _ in range(OH)] for _ in range(Cout)] for _ in range(N)]
    for n in range(N):
        for co in range(Cout):
            for oh in range(OH):
                for ow in range(OW):
                    acc = 0.0
                    for c in range(Cin):
                        for i in range(KH):
                            for j in range(KW):
                                ih = oh * stride + i * dilation - pad
                                iw = ow * stride + j * dilation - pad
                                if 0 <= ih < H and 0 <= iw < W:
                                    acc += xt[n][c][ih][iw] * wt[co][c][i][j]
                    if bt is not None:
                        acc += bt[co]
                    out[n][co][oh][ow] = acc
    return out


class TestConv2D(unittest.TestCase):
    def test_parameter_shapes(self):
        conv = Conv2D(3, 4, 5)
        self.assertEqual(conv.kernel.shape, [4, 3, 5, 5])  # (C_out, C_in, KH, KW)
        self.assertEqual(conv.bias.shape, [4])

    def test_no_bias(self):
        conv = Conv2D(2, 3, 3, use_bias=False)
        self.assertIsNone(conv.bias)

    def test_forward_shapes(self):
        cases = [
            # (in, out, k, stride, pad, dil, H, W, OH, OW)
            (3, 2, 3, 1, 0, 1, 8, 8, 6, 6),
            (3, 4, 3, 2, 1, 1, 8, 8, 4, 4),
            (2, 2, 3, 1, 0, 2, 9, 9, 5, 5),
            (1, 1, 5, 1, 2, 1, 7, 7, 7, 7),
        ]
        for cin, cout, k, s, p, d, H, W, OH, OW in cases:
            conv = Conv2D(cin, cout, k, stride=s, padding=p, dilation=d)
            x = make([2, cin, H, W])
            y = conv(x)
            self.assertEqual(y.shape, [2, cout, OH, OW])

    def test_matches_reference(self):
        cases = [
            dict(in_channels=3, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1),
            dict(in_channels=2, out_channels=3, kernel_size=3, stride=2, padding=1, dilation=1),
            dict(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=2),
        ]
        for kwargs in cases:
            conv = Conv2D(seed=1, **kwargs)
            x = make([2, kwargs["in_channels"], 8, 8])
            got = conv(x).tolist()
            want = ref_conv2d(
                x, conv.kernel, conv.bias,
                kwargs["stride"], kwargs["padding"], kwargs["dilation"],
            )
            assert_close(self, got, want)

    def test_matches_reference_no_bias(self):
        conv = Conv2D(2, 3, 3, stride=2, padding=1, use_bias=False, seed=2)
        x = make([2, 2, 7, 7])
        assert_close(self, conv(x).tolist(), ref_conv2d(x, conv.kernel, None, 2, 1, 1))

    def test_is_callable_module(self):
        conv = Conv2D(2, 2, 3)
        x = make([1, 2, 6, 6])
        self.assertEqual(conv(x).shape, [1, 2, 4, 4])
        params = conv.named_parameters()
        self.assertIn("kernel", params)
        self.assertIn("bias", params)

    def test_gradients_flow(self):
        conv = Conv2D(2, 3, 3, seed=1)
        x = make([2, 2, 6, 6], seed=3)

        def loss(module):
            return (module(x) ** 2).sum()

        value, grads = nj.value_and_grad(loss)(conv)
        self.assertTrue(value.item() > 0.0)
        self.assertEqual(grads.kernel.shape, conv.kernel.shape)
        self.assertEqual(grads.bias.shape, conv.bias.shape)

    def test_kernel_grad_matches_fd(self):
        conv = Conv2D(2, 2, 3, seed=1)
        x = make([1, 2, 5, 5], seed=4)

        def loss(module):
            return (module(x) ** 2).sum()

        _, grads = nj.value_and_grad(loss)(conv)
        analytic = grads.kernel.flatten().tolist()

        base = conv.kernel.flatten().tolist()
        h = 1e-3

        def forward_flat(vals):
            w = nj.Tensor(vals).reshape(conv.kernel.shape)
            y = nj.conv2d(
                x, w, conv.out_channels, conv.kernel_size, conv.kernel_size,
                conv.dilation, conv.dilation, conv.padding, conv.padding,
                conv.stride, conv.stride,
            )
            y = y + conv.bias.reshape([1, conv.out_channels, 1, 1])
            return (y ** 2).sum().item()

        for idx in (0, 5, len(base) - 1):
            plus = base[:]
            minus = base[:]
            plus[idx] += h
            minus[idx] -= h
            numeric = (forward_flat(plus) - forward_flat(minus)) / (2 * h)
            self.assertAlmostEqual(analytic[idx], numeric, places=2)


class TestFlatten(unittest.TestCase):
    def test_shape_preserves_batch(self):
        x = make([2, 3, 4, 4])
        self.assertEqual(Flatten()(x).shape, [2, 48])

    def test_values_row_major(self):
        x = make([2, 1, 2, 2])
        xt = x.tolist()
        self.assertEqual(
            Flatten()(x).tolist(),
            [[xt[0][0][0][0], xt[0][0][0][1], xt[0][0][1][0], xt[0][0][1][1]],
             [xt[1][0][0][0], xt[1][0][0][1], xt[1][0][1][0], xt[1][0][1][1]]],
        )

    def test_gradient(self):
        x = make([2, 3, 4, 4])

        def loss(a):
            return (Flatten()(a) ** 2).sum()

        _, g = nj.value_and_grad(loss)(x)
        self.assertEqual(g.shape, [2, 3, 4, 4])


def ref_mha(m, x):
    """Naive multi-head self-attention in pure Python, given the module's weights."""
    xt = x.tolist()
    B, T, _ = x.shape
    H, d = m.num_heads, m.head_dim

    def proj(xx, w, b):
        wt, bt = w.tolist(), b.tolist()
        return [[[sum(xx[n][t][c] * wt[c][o] for c in range(len(wt))) + bt[o]
                  for o in range(len(bt))] for t in range(T)] for n in range(B)]

    def to_heads(z):  # (B,T,C) -> (B,H,T,d)
        return [[[[z[n][t][h * d + di] for di in range(d)] for t in range(T)]
                 for h in range(H)] for n in range(B)]

    def softmax(row):
        mx = max(row)
        es = [math.exp(e - mx) for e in row]
        s = sum(es)
        return [e / s for e in es]

    q = to_heads(proj(xt, m.q_proj.weight, m.q_proj.bias))
    k = to_heads(proj(xt, m.k_proj.weight, m.k_proj.bias))
    v = to_heads(proj(xt, m.v_proj.weight, m.v_proj.bias))
    scale = math.sqrt(d)

    o = [[[[0.0] * d for _ in range(T)] for _ in range(H)] for _ in range(B)]
    for n in range(B):
        for h in range(H):
            for i in range(T):
                scores = [sum(q[n][h][i][di] * k[n][h][j][di] for di in range(d)) / scale
                          for j in range(T)]
                a = softmax(scores)
                for di in range(d):
                    o[n][h][i][di] = sum(a[j] * v[n][h][j][di] for j in range(T))

    combined = [[[o[n][h][t][di] for h in range(H) for di in range(d)] for t in range(T)]
                for n in range(B)]
    return proj(combined, m.out_proj.weight, m.out_proj.bias)


class TestMultiHeadAttention(unittest.TestCase):
    def test_output_shape(self):
        cases = [
            # (in, out, heads, B, T, expected out)
            (8, 8, 2, 2, 5, [2, 5, 8]),
            (8, 4, 2, 3, 7, [3, 7, 4]),
            (6, 6, 3, 1, 4, [1, 4, 6]),
        ]
        for cin, cout, heads, b, t, want in cases:
            mha = MultiHeadAttention(cin, cout, heads, seed=0)
            x = make([b, t, cin])
            self.assertEqual(mha(x).shape, want)

    def test_heads_roundtrip(self):
        mha = MultiHeadAttention(8, 8, 2, seed=0)
        x = make([2, 5, 8])
        self.assertEqual(mha.combine_heads(mha.split_heads(x)).shape, [2, 5, 8])

    def test_matches_reference(self):
        mha = MultiHeadAttention(4, 4, 2, seed=3)
        x = make([2, 3, 4], seed=1)
        assert_close(self, mha(x).tolist(), ref_mha(mha, x), tol=1e-4)

    def test_single_head_matches_reference(self):
        mha = MultiHeadAttention(6, 6, 1, seed=5)
        x = make([1, 4, 6], seed=2)
        assert_close(self, mha(x).tolist(), ref_mha(mha, x), tol=1e-4)

    def test_gradients_flow(self):
        mha = MultiHeadAttention(4, 4, 2, seed=0)
        x = make([2, 3, 4], seed=1)

        def loss(module):
            return (module(x) ** 2).sum()

        value, grads = nj.value_and_grad(loss)(mha)
        self.assertTrue(value.item() > 0.0)
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            self.assertEqual(getattr(grads, name).weight.shape,
                             getattr(mha, name).weight.shape)

    def test_attention_weights_are_convex(self):
        mha = MultiHeadAttention(4, 4, 2, seed=0)
        x = make([2, 3, 4], seed=1)
        q = mha.split_heads(mha.q_proj(x))
        k = mha.split_heads(mha.k_proj(x))
        v = mha.split_heads(mha.v_proj(x))
        _, attn = MultiHeadAttention._scaled_dot_product(q, k, v)
        # each row of attention over keys sums to 1
        rows = attn.tolist()  # (B,H,T,T)
        for n in range(2):
            for h in range(2):
                for i in range(3):
                    self.assertAlmostEqual(sum(rows[n][h][i]), 1.0, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
