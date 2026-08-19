#pragma once

#include "include/ad/scalar.hpp"

#include <cassert>
#include <cstddef>

inline constexpr double kCrossEntropyEps = 1e-6;

/// y_pred must be after softmax (probability distribution)
/// y_true must be one-hot encoding where 1 is the correct label
inline Scalar mse_loss(const Vector &y_pred, const Vector &y_true) {
  assert(y_pred.size() == y_true.size() &&
         "Size mismatch for predictions and labels for Mean Square Error Loss");

  Scalar error = 0.0;
  for (size_t i = 0; i < y_pred.size(); ++i) {
    Scalar diff = y_pred[i] - y_true[i];
    error += diff * diff;
  }
  return error / static_cast<double>(y_pred.size());
}

/// y_pred must be after softmax (probability distribution)
/// y_true must be one-hot encoding where 1 is the correct label
inline Scalar cross_entropy(const Vector &y_pred, const Vector &y_true) {
  assert(y_pred.size() == y_true.size() &&
         "Size mismatch for predictions and labels for Cross Entropy Loss");

  Scalar error = 0.0;
  for (size_t i = 0; i < y_true.size(); ++i) {
    error += y_true[i] * log(y_pred[i] + kCrossEntropyEps);
  }
  return -error;
}
