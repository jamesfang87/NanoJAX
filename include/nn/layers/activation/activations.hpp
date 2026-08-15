#pragma once

#include "ad/scalar.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

inline Scalar sigmoid(const Scalar &x) { return 1.0 / (1.0 + exp(-x)); }

inline Scalar swish(const Scalar &x, Real beta = 1.0) {
  return x * sigmoid(beta * x);
}

inline Scalar tanh(const Scalar &x) { return 2.0 * sigmoid(2.0 * x) - 1.0; }

inline Scalar ReLU(const Scalar &x) { return (x > 0.0) ? x : 0.0; }

inline Scalar Leaky_ReLU(const Scalar &x, Real alpha = 0.01) {
  return (x > 0.0) ? x : alpha * x;
}

inline Scalar ELU(const Scalar &x, Real alpha = 1.0) {
  return (x > 0.0) ? x : alpha * (exp(x) - 1.0);
}

inline Scalar SELU(const Scalar &x, Real lambda = 1.0507, Real alpha = 1.6733) {
  return lambda * ELU(x, alpha);
}

inline Scalar GELU(const Scalar &x) {
  // approximation for Gaussian cumulative distribution function
  return 0.5 * x *
         (1.0 +
          tanh(sqrt(2.0 / std::numbers::pi) * (x + 0.044715 * pow(x, 3.0))));
}

inline std::vector<Scalar> softmax(const std::vector<Scalar> &logits) {
  // Subtracts the maximum logit before exponentiating, which shifts every
  // input by the constant max_logit and therefore leaves the output
  // distribution unchanged: exp(x_i - c) / sum_j exp(x_j - c) equals
  // exp(x_i) / sum_j exp(x_j) for any c. This keeps the largest exponent
  // argument at 0.0, which avoids the overflow that exponentiating raw
  // logits would otherwise risk.
  Scalar max_logit = *std::max_element(logits.begin(), logits.end());

  std::vector<Scalar> exps;
  exps.reserve(logits.size());
  for (const auto &logit : logits) {
    exps.push_back(exp(logit - max_logit));
  }

  Scalar sum = 0.0;
  for (const auto &e : exps) {
    sum = sum + e;
  }

  std::vector<Scalar> probs;
  probs.reserve(exps.size());
  for (const auto &e : exps) {
    probs.push_back(e / sum);
  }
  return probs;
}
