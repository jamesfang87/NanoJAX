#include "include/ad/grad.hpp"
#include "include/ad/scalar.hpp"

#include "include/nn/layers/loss_fn/loss_fns.hpp"
#include "include/nn/layers/module/mlp.hpp"
#include "include/nn/optimizers/sgd.hpp"

#include "tests/test_utils.hpp"

using nanojax_test::expect_near;
using nanojax_test::expect_true;
using nanojax_test::expect_vector_near;

namespace {

// A fixed 3-point linear regression problem, y = 2x + 1, folded into a
// single Scalar loss of just params so grad() returns the exact batch
// gradient in one backward pass, the same pattern two_layer_relu_loss used
// in test_mlp.cpp for a fixed target.
Scalar batch_loss(const MLP::Params &params) {
  double xs[3] = {1.0, 2.0, 3.0};
  double ys[3] = {3.0, 5.0, 7.0};

  Scalar total = 0.0;
  for (size_t i = 0; i < 3; ++i) {
    Vector x = {Scalar{xs[i]}};
    Vector y = {Scalar{ys[i]}};
    total += mse_loss(MLP::apply(params, x), y);
  }
  return total / 3.0;
}

} // namespace

int main() {
  // step() on a bare Vector: param -= learning_rate * grad, entrywise.
  {
    SGD sgd(0.5);
    Vector params = {Scalar{1.0}, Scalar{2.0}, Scalar{3.0}};
    Vector grads = {Scalar{0.1}, Scalar{0.2}, Scalar{0.3}};

    Vector next = sgd.step(params, grads);
    expect_vector_near("SGD::step (Vector)", next,
                       {Scalar{0.95}, Scalar{1.9}, Scalar{2.85}});
  }

  // step() on an MLP::Params, using the same 3-in/2-out layer and gradient
  // as test_mlp.cpp's "MLP loss = dot(apply(w, x), target)" check: W =
  // [[1,2],[3,4],[5,6]], b = [0.5,-0.5], x = [1,2,3], target = [2,1], with
  // d(loss)/dW = [[2,1],[4,2],[6,3]] and d(loss)/db = [2,1].
  {
    MLP layer(3, 2);
    MLP::Params params = layer.init();
    double next = 1.0;
    for (size_t j = 0; j < 3; ++j) {
      for (size_t i = 0; i < 2; ++i) {
        params.w(j, i) = Scalar{next};
        next += 1.0;
      }
    }
    params.b[0] = Scalar{0.5};
    params.b[1] = Scalar{-0.5};

    MLP::Params grads{Matrix(3, 2), Vector{Scalar{2.0}, Scalar{1.0}}};
    grads.w(0, 0) = Scalar{2.0};
    grads.w(0, 1) = Scalar{1.0};
    grads.w(1, 0) = Scalar{4.0};
    grads.w(1, 1) = Scalar{2.0};
    grads.w(2, 0) = Scalar{6.0};
    grads.w(2, 1) = Scalar{3.0};

    SGD sgd(0.1);
    MLP::Params next_params = sgd.step(params, grads);

    expect_vector_near("SGD::step (MLP::Params.w)", next_params.w.data,
                       {Scalar{0.8}, Scalar{1.9}, Scalar{2.6}, Scalar{3.8},
                        Scalar{4.4}, Scalar{5.7}});
    expect_vector_near("SGD::step (MLP::Params.b)", next_params.b,
                       {Scalar{0.3}, Scalar{-0.6}});
  }

  // End-to-end: repeatedly differentiate batch_loss with grad() and apply
  // SGD::step, and check that training actually reduces the loss rather
  // than just that a single step's arithmetic is correct.
  {
    MLP layer(1, 1);
    MLP::Params params = layer.init();
    SGD sgd(0.1);

    double initial_loss = to_double(batch_loss(params));
    for (int epoch = 0; epoch < 200; ++epoch) {
      MLP::Params g = grad(batch_loss)(params);
      params = sgd.step(params, g);
    }
    double final_loss = to_double(batch_loss(params));

    expect_true("SGD reduces batch_loss", final_loss < initial_loss);
    expect_near("SGD converges close to zero loss", final_loss, 0.0, 1e-3);
  }

  return nanojax_test::report();
}
