// Forward and gradient checks for im2col, col2im, and conv2D.
//
// Forward ops are compared against naive reference implementations. The VJPs
// are checked two ways: the adjoint identity <im2col(x), g> == <x, col2im(g)>
// (which proves col2im is exactly im2col^T), and finite differences of a scalar
// loss through conv2D w.r.t. both the input image and the kernel.

#include "tensor/ops.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>

using namespace nanojax;

namespace {

int failures = 0;
int checks = 0;

void check(bool condition, const char* expression, const char* file, int line) {
    ++checks;
    if (!condition) {
        ++failures;
        std::cerr << file << ':' << line << ": CHECK failed: " << expression << '\n';
    }
}
#define CHECK(expr) check(static_cast<bool>(expr), #expr, __FILE__, __LINE__)

struct Config {
    size_t B, C_in, H, W, C_out, KH, KW, dh, dw, ph, pw, sh, sw;
};

size_t out_dim(size_t in, size_t k, size_t d, size_t p, size_t s) {
    return (in + 2 * p - d * (k - 1) - 1) / s + 1;
}

// Deterministic, transcendental-free values so finite differences stay clean
// under the project's -ffast-math.
double val(size_t i) { return 0.1 * static_cast<double>(i % 13) - 0.6; }

Tensor<double> filled(std::initializer_list<size_t> shape, size_t offset) {
    auto tensor = Tensor<double>::zeros(shape);
    for (size_t i = 0; i < tensor.numel(); ++i) {
        tensor.data()[i] = val(i + offset);
    }
    return tensor;
}

bool same_shape(const Tensor<double>& a, const Tensor<double>& b) {
    if (a.ndim() != b.ndim()) {
        return false;
    }
    for (size_t d = 0; d < a.ndim(); ++d) {
        if (a.shape()[d] != b.shape()[d]) {
            return false;
        }
    }
    return true;
}

bool allclose(const Tensor<double>& a, const Tensor<double>& b, double tol) {
    if (!same_shape(a, b)) {
        return false;
    }
    for (size_t i = 0; i < a.numel(); ++i) {
        if (std::abs(a.data()[i] - b.data()[i]) > tol) {
            return false;
        }
    }
    return true;
}

// Reference implementations (independent re-derivation of the index math).

Tensor<double> naive_im2col(const Tensor<double>& x, const Config& c) {
    const size_t OH = out_dim(c.H, c.KH, c.dh, c.ph, c.sh);
    const size_t OW = out_dim(c.W, c.KW, c.dw, c.pw, c.sw);
    const size_t K = c.C_in * c.KH * c.KW;
    auto cols = Tensor<double>::zeros({c.B, K, OH * OW});
    for (size_t b = 0; b < c.B; ++b) {
        for (size_t ch = 0; ch < c.C_in; ++ch) {
            for (size_t kh = 0; kh < c.KH; ++kh) {
                for (size_t kw = 0; kw < c.KW; ++kw) {
                    const size_t row = (ch * c.KH + kh) * c.KW + kw;
                    for (size_t oh = 0; oh < OH; ++oh) {
                        for (size_t ow = 0; ow < OW; ++ow) {
                            const long ih = (long)oh * c.sh + (long)kh * c.dh - (long)c.ph;
                            const long iw = (long)ow * c.sw + (long)kw * c.dw - (long)c.pw;
                            double v = 0.0;
                            if (ih >= 0 && ih < (long)c.H && iw >= 0 && iw < (long)c.W) {
                                v = x.data()[((b * c.C_in + ch) * c.H + (size_t)ih) * c.W +
                                             (size_t)iw];
                            }
                            cols.data()[(b * K + row) * (OH * OW) + oh * OW + ow] = v;
                        }
                    }
                }
            }
        }
    }
    return cols;
}

Tensor<double> naive_col2im(const Tensor<double>& g, const Config& c) {
    const size_t OH = out_dim(c.H, c.KH, c.dh, c.ph, c.sh);
    const size_t OW = out_dim(c.W, c.KW, c.dw, c.pw, c.sw);
    const size_t K = c.C_in * c.KH * c.KW;
    auto x = Tensor<double>::zeros({c.B, c.C_in, c.H, c.W});
    for (size_t b = 0; b < c.B; ++b) {
        for (size_t ch = 0; ch < c.C_in; ++ch) {
            for (size_t kh = 0; kh < c.KH; ++kh) {
                for (size_t kw = 0; kw < c.KW; ++kw) {
                    const size_t row = (ch * c.KH + kh) * c.KW + kw;
                    for (size_t oh = 0; oh < OH; ++oh) {
                        for (size_t ow = 0; ow < OW; ++ow) {
                            const long ih = (long)oh * c.sh + (long)kh * c.dh - (long)c.ph;
                            const long iw = (long)ow * c.sw + (long)kw * c.dw - (long)c.pw;
                            if (ih >= 0 && ih < (long)c.H && iw >= 0 && iw < (long)c.W) {
                                x.data()[((b * c.C_in + ch) * c.H + (size_t)ih) * c.W +
                                         (size_t)iw] +=
                                    g.data()[(b * K + row) * (OH * OW) + oh * OW + ow];
                            }
                        }
                    }
                }
            }
        }
    }
    return x;
}

Tensor<double> naive_conv(const Tensor<double>& x, const Tensor<double>& w, const Config& c) {
    const size_t OH = out_dim(c.H, c.KH, c.dh, c.ph, c.sh);
    const size_t OW = out_dim(c.W, c.KW, c.dw, c.pw, c.sw);
    auto y = Tensor<double>::zeros({c.B, c.C_out, OH, OW});
    for (size_t b = 0; b < c.B; ++b) {
        for (size_t co = 0; co < c.C_out; ++co) {
            for (size_t oh = 0; oh < OH; ++oh) {
                for (size_t ow = 0; ow < OW; ++ow) {
                    double acc = 0.0;
                    for (size_t ch = 0; ch < c.C_in; ++ch) {
                        for (size_t kh = 0; kh < c.KH; ++kh) {
                            for (size_t kw = 0; kw < c.KW; ++kw) {
                                const long ih = (long)oh * c.sh + (long)kh * c.dh - (long)c.ph;
                                const long iw = (long)ow * c.sw + (long)kw * c.dw - (long)c.pw;
                                if (ih >= 0 && ih < (long)c.H && iw >= 0 && iw < (long)c.W) {
                                    const double xv =
                                        x.data()[((b * c.C_in + ch) * c.H + (size_t)ih) * c.W +
                                                 (size_t)iw];
                                    const double wv =
                                        w.data()[((co * c.C_in + ch) * c.KH + kh) * c.KW + kw];
                                    acc += xv * wv;
                                }
                            }
                        }
                    }
                    y.data()[((b * c.C_out + co) * OH + oh) * OW + ow] = acc;
                }
            }
        }
    }
    return y;
}

Config conv_config() {
    // B C_in H W C_out KH KW dh dw ph pw sh sw
    return Config{2, 2, 5, 4, 3, 3, 2, 1, 1, 1, 0, 1, 2};
}

double loss(const Tensor<double>& x, const Tensor<double>& w, const Config& c) {
    auto y = conv2D(x, w, c.C_out, c.KH, c.KW, c.dh, c.dw, c.ph, c.pw, c.sh, c.sw);
    return sum(y * y).item();
}

struct TraceGuard {
    Trace<double>& tr;
    explicit TraceGuard(Trace<double>& t) : tr(t) { tr.push(); }
    ~TraceGuard() { tr.pop(); }
};

void test_forward(const std::string& name, const Config& c) {
    const size_t OH = out_dim(c.H, c.KH, c.dh, c.ph, c.sh);
    const size_t OW = out_dim(c.W, c.KW, c.dw, c.pw, c.sw);

    auto x = filled({c.B, c.C_in, c.H, c.W}, 1);
    auto w = filled({c.C_out, c.C_in, c.KH, c.KW}, 50);

    // im2col against the reference.
    auto cols = im2col(x, c.KH, c.KW, c.dh, c.dw, c.ph, c.pw, c.sh, c.sw);
    CHECK(cols.ndim() == 3);
    CHECK(cols.shape()[0] == c.B && cols.shape()[1] == c.C_in * c.KH * c.KW);
    CHECK(cols.shape()[2] == OH * OW);
    const bool im2col_ok = allclose(cols, naive_im2col(x, c), 1e-12);
    CHECK(im2col_ok);
    if (!im2col_ok) {
        std::cerr << "  (" << name << " im2col mismatch)\n";
    }

    // col2im against the reference.
    auto dx = col2im(cols, c.C_in, c.H, c.W, c.KH, c.KW, c.dh, c.dw, c.ph, c.pw, c.sh, c.sw);
    CHECK(dx.ndim() == 4 && dx.shape()[0] == c.B && dx.shape()[1] == c.C_in);
    const bool col2im_ok = allclose(dx, naive_col2im(cols, c), 1e-12);
    CHECK(col2im_ok);
    if (!col2im_ok) {
        std::cerr << "  (" << name << " col2im mismatch)\n";
    }

    // conv2D against the reference.
    auto y = conv2D(x, w, c.C_out, c.KH, c.KW, c.dh, c.dw, c.ph, c.pw, c.sh, c.sw);
    CHECK(y.ndim() == 4);
    CHECK(y.shape()[0] == c.B && y.shape()[1] == c.C_out && y.shape()[2] == OH &&
          y.shape()[3] == OW);
    const bool conv_ok = allclose(y, naive_conv(x, w, c), 1e-9);
    CHECK(conv_ok);
    if (!conv_ok) {
        std::cerr << "  (" << name << " conv2D mismatch)\n";
    }

    std::cout << "ok   " << name << '\n';
}

// <im2col(x), g> must equal <x, col2im(g)> for arbitrary g.
void test_adjoint(const std::string& name, const Config& c) {
    auto x = filled({c.B, c.C_in, c.H, c.W}, 1);
    auto cols = im2col(x, c.KH, c.KW, c.dh, c.dw, c.ph, c.pw, c.sh, c.sw);
    auto g = filled({c.B, cols.shape()[1], cols.shape()[2]}, 200);

    auto dx = col2im(g, c.C_in, c.H, c.W, c.KH, c.KW, c.dh, c.dw, c.ph, c.pw, c.sh, c.sw);

    double lhs = 0.0;
    double rhs = 0.0;
    for (size_t i = 0; i < cols.numel(); ++i) {
        lhs += cols.data()[i] * g.data()[i];
    }
    for (size_t i = 0; i < x.numel(); ++i) {
        rhs += x.data()[i] * dx.data()[i];
    }

    const bool ok = std::abs(lhs - rhs) < 1e-9;
    CHECK(ok);
    if (!ok) {
        std::cerr << "  (" << name << " adjoint mismatch: " << lhs << " vs " << rhs << ")\n";
    }
    std::cout << "ok   " << name << '\n';
}

// Central-difference check of dL/dx and dL/dw through conv2D.
void test_grad(const std::string& name, const Config& c) {
    auto x0 = filled({c.B, c.C_in, c.H, c.W}, 1);
    auto w0 = filled({c.C_out, c.C_in, c.KH, c.KW}, 50);

    Tensor<double> dx = Tensor<double>::zeros({});
    Tensor<double> dw = Tensor<double>::zeros({});
    try {
        Trace<double> tr;
        TraceGuard guard(tr);
        auto x = x0;
        auto w = w0;
        tr.add_node(x, nullptr);
        tr.add_node(w, nullptr);

        auto y = conv2D(x, w, c.C_out, c.KH, c.KW, c.dh, c.dw, c.ph, c.pw, c.sh, c.sw);
        tr.backward(sum(y * y));

        dx = tr.adjoints_[x.id()];
        dw = tr.adjoints_[w.id()];
    } catch (const std::exception& e) {
        ++failures;
        std::cerr << "FAIL " << name << ": threw " << e.what() << '\n';
        return;
    }

    const double h = 1e-4;
    const double tol = 1e-4;

    int bad = 0;
    for (size_t k = 0; k < x0.numel(); ++k) {
        auto plus = x0.clone();
        auto minus = x0.clone();
        plus.data()[k] += h;
        minus.data()[k] -= h;
        const double numeric = (loss(plus, w0, c) - loss(minus, w0, c)) / (2 * h);
        if (std::abs(dx.data()[k] - numeric) > tol) {
            if (bad < 3) {
                std::cerr << "  (" << name << " dx[" << k << "]: " << dx.data()[k] << " vs "
                          << numeric << ")\n";
            }
            ++bad;
        }
    }
    for (size_t k = 0; k < w0.numel(); ++k) {
        auto plus = w0.clone();
        auto minus = w0.clone();
        plus.data()[k] += h;
        minus.data()[k] -= h;
        const double numeric = (loss(x0, plus, c) - loss(x0, minus, c)) / (2 * h);
        if (std::abs(dw.data()[k] - numeric) > tol) {
            if (bad < 3) {
                std::cerr << "  (" << name << " dw[" << k << "]: " << dw.data()[k] << " vs "
                          << numeric << ")\n";
            }
            ++bad;
        }
    }

    CHECK(bad == 0);
    std::cout << (bad == 0 ? "ok   " : "FAIL ") << name << " grad\n";
}

void test_errors() {
    auto x = Tensor<double>::zeros({2, 3, 4, 5});
    auto g = Tensor<double>::zeros({2, 6, 6});

    // im2col/conv2D require rank-4 input; col2im requires rank-3.
    bool threw = false;
    try {
        (void)im2col(Tensor<double>::zeros({2, 3, 4}), 3, 3, 1, 1, 0, 0, 1, 1);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);

    threw = false;
    try {
        (void)col2im(Tensor<double>::zeros({2, 5}), 1, 2, 2, 3, 3, 1, 1, 0, 0, 1, 1);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);

    // Kernel larger than the padded input.
    threw = false;
    try {
        (void)im2col(x, 10, 10, 1, 1, 0, 0, 1, 1);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);

    (void)g;
}

} // namespace

int main() {
    const Config c = conv_config();
    const Config square{1, 1, 4, 4, 1, 2, 2, 1, 1, 0, 0, 1, 1};
    const Config dilated{1, 2, 6, 6, 2, 3, 3, 2, 2, 0, 0, 1, 1};
    const Config strided_padded{2, 3, 7, 5, 2, 3, 3, 1, 1, 1, 1, 2, 2};
    const Config one_by_one{1, 3, 4, 4, 2, 1, 1, 1, 1, 0, 0, 1, 1};

    for (const auto& [name, cfg] :
         std::initializer_list<std::pair<const char*, Config>>{
             {"forward base", c},
             {"forward square", square},
             {"forward dilated", dilated},
             {"forward strided_padded", strided_padded},
             {"forward 1x1", one_by_one}}) {
        test_forward(name, cfg);
    }

    test_adjoint("adjoint base", c);
    test_adjoint("adjoint dilated", dilated);
    test_adjoint("adjoint strided_padded", strided_padded);

    test_grad("base", c);
    test_grad("square", square);
    test_grad("dilated", dilated);
    test_grad("strided_padded", strided_padded);
    test_grad("1x1", one_by_one);

    test_errors();

    std::cout << (checks - failures) << '/' << checks << " checks passed\n";
    return failures == 0 ? 0 : 1;
}
