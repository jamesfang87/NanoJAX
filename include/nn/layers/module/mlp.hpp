#pragma once

#include "ad/scalar.hpp"

#include <cstddef>

class MLP {
private:
  size_t in_features;
  size_t out_features;
  bool use_bias;

public:
  // The full set of learnable arrays for one layer. b is left as an empty
  // Vector when the layer has no bias
  struct Params {
    Matrix w;
    Vector b;
  };

  MLP(size_t in_features, size_t out_features, bool use_bias = true)
      : in_features(in_features), out_features(out_features),
        use_bias(use_bias) {}

  // Builds a zero-initialized Params matching this layer's shape
  Params init() const {
    return Params{Matrix(in_features, out_features),
                  use_bias ? Vector(out_features) : Vector()};
  }

  static Vector apply(const Params &w, const Vector &x) {
    if (w.b.empty()) {
      return x * w.w;
    }
    return x * w.w + w.b;
  }

  size_t input_features() const { return in_features; }
  size_t output_features() const { return out_features; }
};
