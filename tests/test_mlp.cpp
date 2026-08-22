#include "include/ad/grad.hpp"
#include "include/ad/scalar.hpp"
#include "include/ad/trace.hpp"

#include "include/nn/layers/activation/activations.hpp"
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

// ReLU is only defined on a single Scalar in activations.hpp; this applies
// it elementwise so it can sit between two apply() calls.
Vector relu_vec(const Vector &v) {
  Vector out(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    out[i] = ReLU(v[i]);
  }
  return out;
}

// The 2-3-2 network from the ReLU-composition demo, taking both layers'
// Params and the input directly rather than a flattened scalar argument
// list. Because this returns a Vector, grad(two_layer_relu, argnums=k)
// selects y_k as the scalar to backpropagate from, giving row k of the
// whole network's Jacobian, shaped as (dy_k/dP1, dy_k/dP2, dy_k/dx), in
// one pass.
Vector two_layer_relu(const MLP::Params &p1, const MLP::Params &p2,
                      const Vector &x) {
  return MLP::apply(p2, relu_vec(MLP::apply(p1, x)));
}

// Same network, folded into a Scalar loss against a fixed target instead
// of returning y directly. grad(two_layer_relu_loss) then differentiates
// with respect to every one of its three arguments in a single backward
// pass, and since none of them is a single Scalar, each comes back shaped
// like itself: a Params of gradients for p1, a Params for p2, and a
// Vector for x.
Scalar two_layer_relu_loss(const MLP::Params &p1, const MLP::Params &p2,
                           const Vector &x) {
  Vector target = {Scalar{1.0}, Scalar{1.0}};
  return dot(two_layer_relu(p1, p2, x), target);
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

  // Compatibility with grad(), using MLP::Params directly: unlike jax.grad,
  // where argnums selects which of f's input arguments to differentiate
  // with respect to, NanoJax's grad(f, argnums) always differentiates with
  // respect to every argument f is called with, and argnums instead
  // selects which entry of a Vector-valued f's output to use as the scalar
  // to backpropagate from. MLP::apply itself, unmodified, is a valid f
  // here: grad(MLP::apply, argnums=i)(params, x) computes row i of apply's
  // Jacobian and hands it back as a (Params, Vector) pair shaped exactly
  // like (params, x) themselves, since grad() lifts and reads back
  // gradients through Params the same way it does a bare Vector.
  {
    MLP layer(3, 2);
    MLP::Params params = layer.init();
    set_weights(params, 3, 2);
    Vector x = {Scalar{1.0}, Scalar{2.0}, Scalar{3.0}};

    // Row 0: dy0/d(...).
    auto row0 = grad(MLP::apply, /*argnums=*/0);
    auto [dparams0, dx0] = row0(params, x);
    expect_vector_near("dy0/dW", dparams0.w.data,
                       {Scalar{1.0}, Scalar{0.0}, Scalar{2.0}, Scalar{0.0},
                        Scalar{3.0}, Scalar{0.0}});
    expect_vector_near("dy0/db", dparams0.b, {Scalar{1.0}, Scalar{0.0}});
    expect_vector_near("dy0/dx", dx0, {Scalar{1.0}, Scalar{3.0}, Scalar{5.0}});

    // Row 1: dy1/d(...).
    auto row1 = grad(MLP::apply, /*argnums=*/1);
    auto [dparams1, dx1] = row1(params, x);
    expect_vector_near("dy1/dW", dparams1.w.data,
                       {Scalar{0.0}, Scalar{1.0}, Scalar{0.0}, Scalar{2.0},
                        Scalar{0.0}, Scalar{3.0}});
    expect_vector_near("dy1/db", dparams1.b, {Scalar{0.0}, Scalar{1.0}});
    expect_vector_near("dy1/dx", dx1, {Scalar{2.0}, Scalar{4.0}, Scalar{6.0}});
  }

  // Two-layer MLP with a ReLU in between, via grad() and MLP::Params: a
  // 2-3-2 network where W1, b1 are chosen so that the hidden pre-activation
  // has one positive entry (unit 0) and two non-positive entries (units 1,
  // 2), exercising ReLU's gating of the backward pass rather than just its
  // forward branch.
  //
  //   h_pre = x * W1 + b1 = [3.5, -2.5, 0.0]
  //   h     = ReLU(h_pre) = [3.5, 0.0, 0.0]
  //   y     = h * W2 + b2 = [3.6, 1.65]
  //
  // two_layer_relu returns y itself rather than a loss and takes p1, p2,
  // and x directly, so grad(two_layer_relu, argnums=k)(p1, p2, x) gives row
  // k of the whole network's Jacobian as (dy_k/dP1, dy_k/dP2, dy_k/dx),
  // each shaped like its own argument. Since h1 = h2 = 0, every path
  // through hidden units 1 and 2 is zeroed by ReLU's backward closure,
  // leaving only the entries that route through unit 0 nonzero in both
  // rows.
  {
    MLP::Params p1{Matrix(2, 3), Vector{Scalar{0.5}, Scalar{-0.5}, Scalar{0.0}}};
    p1.w(0, 0) = Scalar{1.0};
    p1.w(0, 1) = Scalar{-1.0};
    p1.w(0, 2) = Scalar{0.5};
    p1.w(1, 0) = Scalar{-2.0};
    p1.w(1, 1) = Scalar{1.0};
    p1.w(1, 2) = Scalar{0.5};

    MLP::Params p2{Matrix(3, 2), Vector{Scalar{0.1}, Scalar{-0.1}}};
    p2.w(0, 0) = Scalar{1.0};
    p2.w(0, 1) = Scalar{0.5};
    p2.w(1, 0) = Scalar{-1.0};
    p2.w(1, 1) = Scalar{2.0};
    p2.w(2, 0) = Scalar{0.5};
    p2.w(2, 1) = Scalar{-0.5};

    Vector x = {Scalar{1.0}, Scalar{-1.0}};

    auto row0 = grad(two_layer_relu, /*argnums=*/0);
    auto [dp1_0, dp2_0, dx_0] = row0(p1, p2, x);
    expect_vector_near("dy0/dW1", dp1_0.w.data,
                       {Scalar{1.0}, Scalar{0.0}, Scalar{0.0}, Scalar{-1.0},
                        Scalar{0.0}, Scalar{0.0}});
    expect_vector_near("dy0/db1", dp1_0.b, {Scalar{1.0}, Scalar{0.0}, Scalar{0.0}});
    expect_vector_near("dy0/dW2", dp2_0.w.data,
                       {Scalar{3.5}, Scalar{0.0}, Scalar{0.0}, Scalar{0.0},
                        Scalar{0.0}, Scalar{0.0}});
    expect_vector_near("dy0/db2", dp2_0.b, {Scalar{1.0}, Scalar{0.0}});
    expect_vector_near("dy0/dx", dx_0, {Scalar{1.0}, Scalar{-2.0}});

    auto row1 = grad(two_layer_relu, /*argnums=*/1);
    auto [dp1_1, dp2_1, dx_1] = row1(p1, p2, x);
    expect_vector_near("dy1/dW1", dp1_1.w.data,
                       {Scalar{0.5}, Scalar{0.0}, Scalar{0.0}, Scalar{-0.5},
                        Scalar{0.0}, Scalar{0.0}});
    expect_vector_near("dy1/db1", dp1_1.b, {Scalar{0.5}, Scalar{0.0}, Scalar{0.0}});
    expect_vector_near("dy1/dW2", dp2_1.w.data,
                       {Scalar{0.0}, Scalar{3.5}, Scalar{0.0}, Scalar{0.0},
                        Scalar{0.0}, Scalar{0.0}});
    expect_vector_near("dy1/db2", dp2_1.b, {Scalar{0.0}, Scalar{1.0}});
    expect_vector_near("dy1/dx", dx_1, {Scalar{0.5}, Scalar{-1.0}});

    // Same 2-3-2 ReLU network, but reached through the full-gradient form
    // of grad() instead of a per-output Jacobian row: two_layer_relu_loss
    // folds the whole forward pass, including the ReLU and the dot with
    // target, into a Scalar loss of the same three arguments. Because
    // none of p1, p2, x is a single Scalar, grad()'s argnums selection
    // never comes into play, and the returned tuple's entries are shaped
    // like p1, p2, and x themselves; entrywise, they equal row0 + row1
    // above, since loss = dot(y, [1, 1]) = y0 + y1.
    auto dloss = grad(two_layer_relu_loss);
    auto [dp1, dp2, dx] = dloss(p1, p2, x);
    expect_vector_near("d(loss)/dW1", dp1.w.data,
                       {Scalar{1.5}, Scalar{0.0}, Scalar{0.0}, Scalar{-1.5},
                        Scalar{0.0}, Scalar{0.0}});
    expect_vector_near("d(loss)/db1", dp1.b, {Scalar{1.5}, Scalar{0.0}, Scalar{0.0}});
    expect_vector_near("d(loss)/dW2", dp2.w.data,
                       {Scalar{3.5}, Scalar{3.5}, Scalar{0.0}, Scalar{0.0},
                        Scalar{0.0}, Scalar{0.0}});
    expect_vector_near("d(loss)/db2", dp2.b, {Scalar{1.0}, Scalar{1.0}});
    expect_vector_near("d(loss)/dx", dx, {Scalar{1.5}, Scalar{-3.0}});
  }

  return nanojax_test::report();
}
