// Gradient checks for Tensor::slice (both overloads).
//
// First order: the adjoint of sum(slice(x) * w) is finite-difference checked.
// Second order: the diagonal of the Hessian of sum(slice(x)^2) is computed by
// nesting two traces and compared against the analytic value (2 at selected
// elements, 0 elsewhere). The second-order test is what proves the VJP is built
// from traced ops (scatter) rather than raw writes.

#include "tensor/ops.hpp"

#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <span>
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

struct TraceGuard {
    Trace<double>& tr;
    explicit TraceGuard(Trace<double>& t) : tr(t) { tr.push(); }
    ~TraceGuard() { tr.pop(); }
};

Tensor<double> filled(std::initializer_list<size_t> shape, double base, double step) {
    auto tensor = Tensor<double>::zeros(shape);
    for (size_t i = 0; i < tensor.numel(); ++i) {
        tensor.data()[i] = base + step * static_cast<double>(i % 7);
    }
    return tensor;
}

void fill_into(Tensor<double>& tensor, double base, double step) {
    for (size_t i = 0; i < tensor.numel(); ++i) {
        tensor.data()[i] = base + step * static_cast<double>(i % 5);
    }
}

// The coordinate of `flat` along `dim` in a dense row-major tensor of `shape`.
size_t coord_along(size_t flat, size_t dim, std::span<const size_t> shape) {
    size_t stride = 1;
    size_t coord = 0;
    for (size_t d = shape.size(); d-- > 0;) {
        const size_t c = (flat / stride) % shape[d];
        if (d == dim) {
            coord = c;
        }
        stride *= shape[d];
    }
    return coord;
}

void test_first_order(const std::string& name, std::initializer_list<size_t> shape, size_t dim,
                      size_t start, size_t end, size_t step) {
    auto x0 = filled(shape, -0.3, 0.1);
    auto s0 = x0.slice(dim, start, end, step);
    auto w = Tensor<double>::zeros(s0.shape());
    fill_into(w, -0.4, 0.2);

    Tensor<double> dx = Tensor<double>::zeros({});
    try {
        Trace<double> tr;
        TraceGuard guard(tr);
        auto x = x0;
        tr.add_node(x, nullptr);
        auto s = x.slice(dim, start, end, step);
        auto loss = sum(s * w);
        tr.backward(loss);
        dx = tr.adjoint(x);
    } catch (const std::exception& e) {
        ++failures;
        std::cerr << "FAIL " << name << ": threw " << e.what() << '\n';
        return;
    }

    const double h = 1e-4;
    auto loss_of = [&](const Tensor<double>& xx) {
        return sum(xx.slice(dim, start, end, step) * w).item();
    };

    bool ok = true;
    for (size_t k = 0; k < x0.numel(); ++k) {
        auto plus = x0.clone();
        auto minus = x0.clone();
        plus.data()[k] += h;
        minus.data()[k] -= h;
        const double numeric = (loss_of(plus) - loss_of(minus)) / (2 * h);
        if (std::abs(dx.data()[k] - numeric) > 1e-4) {
            if (ok) {
                std::cerr << "FAIL " << name << " first order:\n";
            }
            ok = false;
            std::cerr << "  dx[" << k << "] = " << dx.data()[k] << " want " << numeric << '\n';
        }
    }

    CHECK(ok);
    std::cout << (ok ? "ok   " : "FAIL ") << name << " first order\n";
}

void test_drop_first_order(const std::string& name, size_t index) {
    auto x0 = filled({4, 3}, -0.5, 0.3);
    auto s0 = x0.slice(index);
    auto w = Tensor<double>::zeros(s0.shape());
    fill_into(w, -0.2, 0.4);

    Tensor<double> dx = Tensor<double>::zeros({});
    try {
        Trace<double> tr;
        TraceGuard guard(tr);
        auto x = x0;
        tr.add_node(x, nullptr);
        auto s = x.slice(index);
        tr.backward(sum(s * w));
        dx = tr.adjoint(x);
    } catch (const std::exception& e) {
        ++failures;
        std::cerr << "FAIL " << name << ": threw " << e.what() << '\n';
        return;
    }

    // Analytic: only row `index` gets w; other rows zero.
    bool ok = true;
    for (size_t r = 0; r < 4; ++r) {
        for (size_t c = 0; c < 3; ++c) {
            const double want = (r == index) ? w.data()[c] : 0.0;
            const double got = dx.data()[r * 3 + c];
            if (std::abs(got - want) > 1e-12) {
                ok = false;
                std::cerr << "  dx[" << r << "," << c << "] = " << got << " want " << want << '\n';
            }
        }
    }
    CHECK(ok);
    std::cout << (ok ? "ok   " : "FAIL ") << name << " first order\n";
}

void test_second_order(const std::string& name, std::initializer_list<size_t> shape, size_t dim,
                       size_t start, size_t end, size_t step) {
    auto x0 = filled(shape, -0.3, 0.1);

    Tensor<double> ddx = Tensor<double>::zeros({});
    try {
        Trace<double> outer;
        outer.push();

        auto x = x0;
        outer.add_node(x, nullptr);

        Tensor<double> dx = Tensor<double>::zeros({});
        {
            Trace<double> inner;
            inner.push();
            inner.add_node(x, nullptr);
            auto s = x.slice(dim, start, end, step);
            inner.backward(sum(s * s));
            dx = inner.adjoint(x);
            inner.pop();
        }

        outer.backward(sum(dx));
        ddx = outer.adjoint(x);
        outer.pop();
    } catch (const std::exception& e) {
        ++failures;
        std::cerr << "FAIL " << name << ": threw " << e.what() << '\n';
        return;
    }

    // Analytic Hessian diagonal: d^2/dx_i^2 of sum_{selected} x^2 = 2 if selected.
    bool ok = true;
    const auto s = x0.shape();
    for (size_t k = 0; k < x0.numel(); ++k) {
        const size_t c = coord_along(k, dim, s);
        const bool selected = c >= start && c < end && ((c - start) % step == 0);
        const double want = selected ? 2.0 : 0.0;
        if (std::abs(ddx.data()[k] - want) > 1e-9) {
            if (ok) {
                std::cerr << "FAIL " << name << " second order:\n";
            }
            ok = false;
            std::cerr << "  ddx[" << k << "] = " << ddx.data()[k] << " want " << want << '\n';
        }
    }

    CHECK(ok);
    std::cout << (ok ? "ok   " : "FAIL ") << name << " second order\n";
}

} // namespace

int main() {
    test_first_order("1d step2", {6}, 0, 1, 6, 2);
    test_first_order("2d dim1 step2", {2, 5}, 1, 1, 5, 2);
    test_first_order("2d dim0 step1", {3, 4}, 0, 1, 3, 1);
    test_first_order("3d dim2 step1", {2, 3, 4}, 2, 0, 4, 1);
    test_first_order("3d dim1 step2", {2, 3, 4}, 1, 0, 3, 2);
    test_first_order("3d dim0 step2", {3, 4, 2}, 0, 1, 3, 2);

    test_drop_first_order("drop index 0", 0);
    test_drop_first_order("drop index 2", 2);

    test_second_order("2nd 1d step2", {6}, 0, 1, 6, 2);
    test_second_order("2nd 2d dim1 step2", {2, 5}, 1, 1, 5, 2);
    test_second_order("2nd 3d dim2 step2", {2, 3, 4}, 2, 1, 4, 2);
    test_second_order("2nd 3d dim1 step1", {2, 3, 4}, 1, 0, 3, 1);

    std::cout << (checks - failures) << '/' << checks << " checks passed\n";
    return failures == 0 ? 0 : 1;
}
