#include "include/ad/grad.hpp"
#include "include/ad/scalar.hpp"
#include "include/ad/trace.hpp"

#include "include/nn/layers/module/mlp.hpp"

#include "tests/test_utils.hpp"

#include <memory>
#include <string>

using nanojax_test::expect_near;
using nanojax_test::expect_vector_near;

namespace {

struct TraceScope {
  TraceScope() { trace_stack.push(std::make_unique<Trace>()); }
  ~TraceScope() { trace_stack.pop(); }
};

// Fills an in_features x out_features Params with fixed, non-symmetric
// values
void set_weights(MLP::Params &params, size_t in_features, size_t out_features) {
  double next = 1.0;
  for (size_t j = 0; j < in_features; ++j) {
    for (size_t i = 0; i < out_features; ++i) {
      params.w(j, i) = Scalar{next};
      next += 1.0;
    }
  }
  for (size_t i = 0; i < params.b.size(); ++i) {
    params.b[i] = Scalar{i % 2 == 0 ? 0.5 : -0.5};
  }
}

// Flattens a 3-in, 2-out layer's apply() into a function of eleven plain
// scalar arguments (six weight entries, two bias entries, three input
// entries), which is the shape grad() requires
Vector apply_flat(const Scalar &w00, const Scalar &w01, const Scalar &w10,
                  const Scalar &w11, const Scalar &w20, const Scalar &w21,
                  const Scalar &b0, const Scalar &b1, const Scalar &x0,
                  const Scalar &x1, const Scalar &x2) {
  MLP::Params params{Matrix(3, 2), Vector{b0, b1}};
  params.w(0, 0) = w00;
  params.w(0, 1) = w01;
  params.w(1, 0) = w10;
  params.w(1, 1) = w11;
  params.w(2, 0) = w20;
  params.w(2, 1) = w21;
  Vector x = {x0, x1, x2};
  return MLP::apply(params, x);
}

} // namespace

int main() {
  // apply(), with bias: in_features=3, out_features=2, rectangular so a
  // swapped dimension can't hide behind a square weight matrix. Weight
  // layout follows set_weights above: W(0,0)=1, W(0,1)=2, W(1,0)=3,
  // W(1,1)=4, W(2,0)=5, W(2,1)=6, bias = {0.5, -0.5}.
  {
    MLP layer(3, 2);
    MLP::Params params = layer.init();
    set_weights(params, 3, 2);
    Vector x = {Scalar{1.0}, Scalar{2.0}, Scalar{3.0}};

    // y_i = sum_j x_j * W(j,i) + b_i
    // y0 = 1*1 + 2*3 + 3*5 + 0.5 = 22.5
    // y1 = 1*2 + 2*4 + 3*6 + -0.5 = 27.5
    Vector y = MLP::apply(params, x);
    expect_vector_near("MLP apply (with bias)", y,
                       {Scalar{22.5}, Scalar{27.5}});
  }

  // Same weights and input, without bias: the +b0.5/-0.5 shift disappears,
  // since init() leaves params.b empty when the layer was built with
  // use_bias=false, and apply() checks emptiness rather than taking a
  // separate flag.
  {
    MLP layer(3, 2, /*use_bias=*/false);
    MLP::Params params = layer.init();
    set_weights(params, 3, 2);
    Vector x = {Scalar{1.0}, Scalar{2.0}, Scalar{3.0}};

    Vector y = MLP::apply(params, x);
    expect_vector_near("MLP apply (no bias)", y, {Scalar{22.0}, Scalar{28.0}});
  }

  // Gradient check via the raw Trace API: build the same 3x2 layer's Params
  // with every weight, bias, and input entry as its own Variable, take
  // loss = dot(apply(w, x), target) for a fixed constant target, and check
  // every partial against the closed-form derivative of that linear loss.
  // Because apply() is static and reads no instance state, this works
  // without an MLP instance ever entering the trace; only params.w, params.b,
  // and x do.
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

    MLP::Params params{Matrix(3, 2), Vector(2)};
    Variable w[3][2];
    for (size_t j = 0; j < 3; ++j) {
      for (size_t i = 0; i < 2; ++i) {
        w[j][i] = tr.add_variable(Scalar{static_cast<double>(j * 2 + i + 1)},
                                  nullptr);
        params.w(j, i) = Scalar{w[j][i]};
      }
    }
    Variable b0 = tr.add_variable(Scalar{0.5}, nullptr);
    Variable b1 = tr.add_variable(Scalar{-0.5}, nullptr);
    params.b[0] = Scalar{b0};
    params.b[1] = Scalar{b1};

    Variable x0 = tr.add_variable(Scalar{1.0}, nullptr);
    Variable x1 = tr.add_variable(Scalar{2.0}, nullptr);
    Variable x2 = tr.add_variable(Scalar{3.0}, nullptr);
    Vector x = {Scalar{x0}, Scalar{x1}, Scalar{x2}};

    Vector target = {Scalar{2.0}, Scalar{1.0}};

    Vector y = MLP::apply(params, x);
    Scalar loss = dot(y, target);
    expect_near("MLP loss = dot(apply(w, x), target)", to_double(loss), 72.5);

    tr.backward(std::get<Variable>(loss));

    double target_vals[2] = {2.0, 1.0};
    double x_vals[3] = {1.0, 2.0, 3.0};
    for (size_t j = 0; j < 3; ++j) {
      for (size_t i = 0; i < 2; ++i) {
        expect_near(
            "d(loss)/dW(" + std::to_string(j) + "," + std::to_string(i) + ")",
            to_double(tr.adjoints[w[j][i].id]), target_vals[i] * x_vals[j]);
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

  // Compatibility with grad(): unlike jax.grad, where argnums selects which
  // of f's input arguments to differentiate with respect to, NanoJax's
  // grad(f, argnums) always differentiates with respect to every scalar
  // argument f is called with, and argnums instead selects which entry of
  // a Vector-valued f's output to use as the scalar to backpropagate from.
  // apply_flat returns the layer's two-entry output Vector directly, so
  // grad(apply_flat, argnums=i) computes row i of apply's Jacobian, i.e.
  // dy_i/d(w00, w01, w10, w11, w20, w21, b0, b1, x0, x1, x2), all in one
  // backward pass.
  {
    double w_vals[3][2] = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
    double b_vals[2] = {0.5, -0.5};
    double x_vals[3] = {1.0, 2.0, 3.0};

    auto call = [&](auto &g) {
      return g(w_vals[0][0], w_vals[0][1], w_vals[1][0], w_vals[1][1],
               w_vals[2][0], w_vals[2][1], b_vals[0], b_vals[1], x_vals[0],
               x_vals[1], x_vals[2]);
    };

    // Row 0: dy0/d(...).
    auto row0 = grad(apply_flat, /*argnums=*/0);
    Vector g0 = call(row0);
    expect_vector_near("grad(apply_flat, argnums=0)", g0,
                       {Scalar{x_vals[0]}, Scalar{0.0}, Scalar{x_vals[1]},
                        Scalar{0.0}, Scalar{x_vals[2]}, Scalar{0.0},
                        Scalar{1.0}, Scalar{0.0}, Scalar{w_vals[0][0]},
                        Scalar{w_vals[1][0]}, Scalar{w_vals[2][0]}});

    // Row 1: dy1/d(...).
    auto row1 = grad(apply_flat, /*argnums=*/1);
    Vector g1 = call(row1);
    expect_vector_near("grad(apply_flat, argnums=1)", g1,
                       {Scalar{0.0}, Scalar{x_vals[0]}, Scalar{0.0},
                        Scalar{x_vals[1]}, Scalar{0.0}, Scalar{x_vals[2]},
                        Scalar{0.0}, Scalar{1.0}, Scalar{w_vals[0][1]},
                        Scalar{w_vals[1][1]}, Scalar{w_vals[2][1]}});
  }

  return nanojax_test::report();
}
