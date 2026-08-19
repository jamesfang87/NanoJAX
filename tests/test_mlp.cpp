#include "include/ad/scalar.hpp"
#include "include/ad/trace.hpp"

#include "include/nn/layers/module/mlp.hpp"

#include "tests/test_utils.hpp"

#include <memory>
#include <string>
#include <variant>

using nanojax_test::expect_near;
using nanojax_test::expect_vector_near;

namespace {

// RAII wrapper around trace_stack, mirroring the one in test_variable.cpp
// and test_loss.cpp, so each test case gets its own isolated Trace.
struct TraceScope {
  TraceScope() { trace_stack.push(std::make_unique<Trace>()); }
  ~TraceScope() { trace_stack.pop(); }
};

// Fills an in_features x out_features layer's weight and bias with fixed,
// non-symmetric values (rather than the all-zero default MLP's constructor
// leaves them at) so forward()'s result actually depends on which entry
// went where; a transposed weight index would show up as a wrong number
// here instead of silently cancelling out.
void set_weights(MLP &layer, size_t in_features, size_t out_features) {
  double next = 1.0;
  for (size_t j = 0; j < in_features; ++j) {
    for (size_t i = 0; i < out_features; ++i) {
      layer.weight()(j, i) = Scalar{next};
      next += 1.0;
    }
  }
  for (size_t i = 0; i < out_features; ++i) {
    layer.bias()[i] = Scalar{i % 2 == 0 ? 0.5 : -0.5};
  }
}

} // namespace

int main() {
  // forward(), with bias: in_features=3, out_features=2, rectangular so a
  // swapped dimension can't hide behind a square weight matrix. Weight
  // layout follows set_weights above: W(0,0)=1, W(0,1)=2, W(1,0)=3,
  // W(1,1)=4, W(2,0)=5, W(2,1)=6, bias = {0.5, -0.5}.
  {
    MLP layer(3, 2);
    set_weights(layer, 3, 2);
    Vector x = {Scalar{1.0}, Scalar{2.0}, Scalar{3.0}};

    // y_i = sum_j x_j * W(j,i) + b_i
    // y0 = 1*1 + 2*3 + 3*5 + 0.5 = 22.5
    // y1 = 1*2 + 2*4 + 3*6 + -0.5 = 27.5
    Vector y = layer.forward(x);
    expect_vector_near("MLP forward (with bias)", y,
                       {Scalar{22.5}, Scalar{27.5}});
  }

  // Same weights and input, without bias: the +b0.5/-0.5 shift disappears.
  {
    MLP layer(3, 2, /*use_bias=*/false);
    set_weights(layer, 3, 2);
    Vector x = {Scalar{1.0}, Scalar{2.0}, Scalar{3.0}};

    Vector y = layer.forward(x);
    expect_vector_near("MLP forward (no bias)", y, {Scalar{22.0}, Scalar{28.0}});
  }

  // Gradient check: build the same 3x2 layer with every weight, bias, and
  // input entry as its own Variable, take loss = dot(forward(x), target)
  // for a fixed constant target, and check every partial against the
  // closed-form derivative of that linear loss.
  //
  //   y_i = sum_j x_j * W(j,i) + b_i
  //   loss = sum_i y_i * target_i
  //
  //   d(loss)/dW(j,i) = target_i * x_j
  //   d(loss)/db_i    = target_i
  //   d(loss)/dx_j    = sum_i target_i * W(j,i)
  {
    TraceScope scope;
    Trace &tr = trace();

    MLP layer(3, 2);
    Variable w[3][2];
    for (size_t j = 0; j < 3; ++j) {
      for (size_t i = 0; i < 2; ++i) {
        w[j][i] = tr.add_variable(Scalar{static_cast<double>(j * 2 + i + 1)},
                                  nullptr);
        layer.weight()(j, i) = Scalar{w[j][i]};
      }
    }
    Variable b0 = tr.add_variable(Scalar{0.5}, nullptr);
    Variable b1 = tr.add_variable(Scalar{-0.5}, nullptr);
    layer.bias()[0] = Scalar{b0};
    layer.bias()[1] = Scalar{b1};

    Variable x0 = tr.add_variable(Scalar{1.0}, nullptr);
    Variable x1 = tr.add_variable(Scalar{2.0}, nullptr);
    Variable x2 = tr.add_variable(Scalar{3.0}, nullptr);
    Vector x = {Scalar{x0}, Scalar{x1}, Scalar{x2}};

    Vector target = {Scalar{2.0}, Scalar{1.0}};

    Vector y = layer.forward(x);
    Scalar loss = dot(y, target);
    expect_near("MLP loss = dot(forward(x), target)", to_double(loss), 72.5);

    tr.backward(std::get<Variable>(loss));

    double target_vals[2] = {2.0, 1.0};
    double x_vals[3] = {1.0, 2.0, 3.0};
    for (size_t j = 0; j < 3; ++j) {
      for (size_t i = 0; i < 2; ++i) {
        expect_near("d(loss)/dW(" + std::to_string(j) + "," +
                        std::to_string(i) + ")",
                    to_double(tr.adjoints[w[j][i].id]),
                    target_vals[i] * x_vals[j]);
      }
    }
    expect_near("d(loss)/db0", to_double(tr.adjoints[b0.id]), 2.0);
    expect_near("d(loss)/db1", to_double(tr.adjoints[b1.id]), 1.0);

    double w_vals[3][2] = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
    expect_near("d(loss)/dx0", to_double(tr.adjoints[x0.id]),
               target_vals[0] * w_vals[0][0] + target_vals[1] * w_vals[0][1]);
    expect_near("d(loss)/dx1", to_double(tr.adjoints[x1.id]),
               target_vals[0] * w_vals[1][0] + target_vals[1] * w_vals[1][1]);
    expect_near("d(loss)/dx2", to_double(tr.adjoints[x2.id]),
               target_vals[0] * w_vals[2][0] + target_vals[1] * w_vals[2][1]);
  }

  return nanojax_test::report();
}
