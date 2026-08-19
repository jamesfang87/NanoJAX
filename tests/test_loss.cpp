#include "include/ad/scalar.hpp"
#include "include/ad/trace.hpp"

#include "include/nn/layers/loss_fn/loss_fns.hpp"

#include "tests/test_utils.hpp"

#include <cmath>
#include <memory>
#include <variant>

using nanojax_test::expect_near;

namespace {

// RAII wrapper around trace_stack, mirroring the one in test_variable.cpp,
// so each test case gets its own isolated Trace.
struct TraceScope {
  TraceScope() { trace_stack.push(std::make_unique<Trace>()); }
  ~TraceScope() { trace_stack.pop(); }
};

} // namespace

int main() {
  // mse_loss: primal is the mean, not the sum, of the squared errors, and
  // the gradient w.r.t. each prediction is 2 * (y_pred[i] - y_true[i]) / n.
  {
    TraceScope scope;
    Trace &tr = trace();
    Variable p0 = tr.add_variable(Scalar{1.0}, nullptr);
    Variable p1 = tr.add_variable(Scalar{1.0}, nullptr);
    Variable t0 = tr.add_variable(Scalar{0.0}, nullptr);
    Variable t1 = tr.add_variable(Scalar{0.0}, nullptr);

    Vector y_pred = {Scalar{p0}, Scalar{p1}};
    Vector y_true = {Scalar{t0}, Scalar{t1}};

    Scalar out = mse_loss(y_pred, y_true);
    expect_near("mse_loss({1,1},{0,0})", to_double(out), 1.0); // mean, not 2.0

    tr.backward(std::get<Variable>(out));
    expect_near("d(mse)/dp0", to_double(tr.adjoints[p0.id]), 1.0); // 2*(1-0)/2
    expect_near("d(mse)/dp1", to_double(tr.adjoints[p1.id]), 1.0);
  }

  // cross_entropy: y_pred is a probability distribution (as if already
  // passed through softmax), y_true is one-hot at the true class. The loss
  // reduces to -log(y_pred[true_class]), and its gradient is nonzero only
  // at that class: d/dp_i = -y_true[i] / p_i.
  {
    TraceScope scope;
    Trace &tr = trace();
    Variable p0 = tr.add_variable(Scalar{0.2}, nullptr);
    Variable p1 = tr.add_variable(Scalar{0.5}, nullptr);
    Variable p2 = tr.add_variable(Scalar{0.3}, nullptr);
    Variable t0 = tr.add_variable(Scalar{0.0}, nullptr);
    Variable t1 = tr.add_variable(Scalar{1.0}, nullptr);
    Variable t2 = tr.add_variable(Scalar{0.0}, nullptr);

    Vector y_pred = {Scalar{p0}, Scalar{p1}, Scalar{p2}};
    Vector y_true = {Scalar{t0}, Scalar{t1}, Scalar{t2}};

    // Expected values are computed against kCrossEntropyEps rather than
    // against the idealized -log(p) / -1/p formulas, since cross_entropy
    // adds that epsilon to y_pred before taking the log for numerical
    // stability, which shifts both the loss and its gradient by a small,
    // deliberate amount.
    Scalar out = cross_entropy(y_pred, y_true);
    expect_near("cross_entropy at true class 1", to_double(out),
                -std::log(0.5 + kCrossEntropyEps));

    tr.backward(std::get<Variable>(out));
    expect_near("d(xent)/dp0 (wrong class)", to_double(tr.adjoints[p0.id]),
                0.0);
    expect_near("d(xent)/dp1 (true class)", to_double(tr.adjoints[p1.id]),
                -1.0 / (0.5 + kCrossEntropyEps));
    expect_near("d(xent)/dp2 (wrong class)", to_double(tr.adjoints[p2.id]),
                0.0);
  }

  return nanojax_test::report();
}
