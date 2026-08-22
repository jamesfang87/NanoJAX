#pragma once

#include "ad/scalar.hpp"
#include "nn/optimizers/optim.hpp"

// Vanilla stochastic gradient descent: param -= learning_rate * grad,
class SGD : public Optimizer {
public:
  explicit SGD(double learning_rate) : learning_rate_(learning_rate) {}

  template <typename Params, typename Grads>
  Params step(const Params &params, const Grads &grads) const {
    return zip_scalars(params, grads, [this](const Scalar &p, const Scalar &g) {
      return p - Scalar{learning_rate_} * g;
    });
  }

private:
  double learning_rate_;
};
