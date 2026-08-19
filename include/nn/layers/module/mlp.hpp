#pragma once

#include "ad/scalar.hpp"

#include <cstddef>

class MLP {
private:
  Matrix w;
  Vector b;
  bool use_bias;

public:
  MLP(size_t in_features, size_t out_features, bool use_bias = true)
      : w(Matrix(in_features, out_features)), b(Vector(out_features)),
        use_bias(use_bias) {}

  Vector forward(Vector x) {
    if (!use_bias) {
      return x * w;
    } else {
      return x * w + b;
    }
  }

  Matrix &weight() { return w; }
  const Matrix &weight() const { return w; }
  Vector &bias() { return b; }
  const Vector &bias() const { return b; }
};
