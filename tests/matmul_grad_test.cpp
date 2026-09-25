// Finite-difference checks for vjp::matmul: 2-D, batched, broadcast batch.
//
// TDD: the batched cases are expected to fail until vjp::matmul swaps only the
// last two axes and reduces the broadcast batch dimensions.

#include "tensor/ops.hpp"

#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <string>

using namespace nanojax;

namespace {

int failures = 0;

// Pushes a trace for the block's lifetime so operations record on it.
struct TraceGuard {
    Trace<double>& tr;
    explicit TraceGuard(Trace<double>& t) : tr(t) { tr.push(); }
    ~TraceGuard() { tr.pop(); }
};

Tensor<double> make(std::initializer_list<size_t> shape, double base) {
    auto tensor = Tensor<double>::value(shape, 0.0);
    for (size_t i = 0; i < tensor.numel(); ++i) {
        tensor.data()[i] = base + static_cast<double>(i);
    }
    return tensor;
}

Tensor<double> loss(const Tensor<double>& a, const Tensor<double>& b) {
    auto c = matmul(a, b);
    return sum(c * c);
}

// Compares one operand's adjoint against a central difference of loss().
void check(const std::string& name, const Tensor<double>& analytic, const Tensor<double>& a,
           const Tensor<double>& b, bool perturb_a) {
    const double h = 1e-3;
    const Tensor<double>& base = perturb_a ? a : b;
    const Tensor<double>& other = perturb_a ? b : a;

    for (size_t k = 0; k < base.numel(); ++k) {
        auto plus = base.clone();
        auto minus = base.clone();
        plus.data()[k] += h;
        minus.data()[k] -= h;

        const double lp = (perturb_a ? loss(plus, other) : loss(other, plus)).item();
        const double lm = (perturb_a ? loss(minus, other) : loss(other, minus)).item();
        const double numeric = (lp - lm) / (2 * h);
        const double got = analytic.data()[k];

        if (std::abs(got - numeric) > 1e-4) {
            ++failures;
            std::cerr << "FAIL " << name << "[" << k << "]: got " << got << " want " << numeric
                      << '\n';
            return;
        }
    }
}

void run(const std::string& name, std::initializer_list<size_t> a_shape,
         std::initializer_list<size_t> b_shape) {
    auto a0 = make(a_shape, 0.5);
    auto b0 = make(b_shape, 1.0);

    try {
        Trace<double> tr;
        TraceGuard guard(tr);
        auto a = a0;
        auto b = b0;
        tr.add_node(a, nullptr);
        tr.add_node(b, nullptr);

        auto c = matmul(a, b);
        auto l = sum(c * c);
        tr.backward(l);

        const int before = failures;
        check(name + " dA", tr.adjoints_[a.id()], a0, b0, true);
        check(name + " dB", tr.adjoints_[b.id()], a0, b0, false);
        if (failures == before) {
            std::cout << "ok   " << name << '\n';
        }
    } catch (const std::exception& e) {
        ++failures;
        std::cerr << "FAIL " << name << ": threw " << e.what() << '\n';
    }
}

} // namespace

int main() {
    run("matmul 2d", {2, 3}, {3, 2});
    run("matmul batched", {2, 2, 3}, {2, 3, 2});
    run("matmul broadcast batch", {2, 1, 2, 3}, {1, 3, 3, 2});

    // Rank-1 promotion: dot, matvec, vecmat, batched matvec.
    run("matvec", {2, 3}, {3});
    run("vecmat", {3}, {3, 2});
    run("dot", {3}, {3});
    run("batched matvec", {2, 2, 3}, {3});

    std::cout << (failures == 0 ? "PASS\n" : "FAIL\n");
    return failures == 0 ? 0 : 1;
}
