#pragma once

#include "include/ad/scalar.hpp"

inline Scalar relu(const Scalar &x) {
  if (x > 0.0) {
    return x;
  }
  return 0.0 * x;
}

inline Scalar sigmoid(const Scalar &x) { return 1.0 / (1.0 + exp(-x)); }
