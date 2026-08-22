#pragma once

#include "ad/scalar.hpp"

#include <cstddef>

class MLP {
private:
  size_t in_features;
  size_t out_features;
  bool use_bias;

public:
  // b is left as an empty
  //  Vector when the layer has no bias
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

// Lets grad() (ad/grad.hpp) read gradients through an MLP::Params
template <typename F>
MLP::Params map_scalars(const MLP::Params &p, const F &fn) {
  return MLP::Params{map_scalars(p.w, fn), map_scalars(p.b, fn)};
}

// Lets an Optimizer (nn/optimizers/optim.hpp) combine an MLP::Params with a
// same-shaped gradient
template <typename F>
MLP::Params zip_scalars(const MLP::Params &a, const MLP::Params &b,
                        const F &fn) {
  return MLP::Params{zip_scalars(a.w, b.w, fn), zip_scalars(a.b, b.b, fn)};
}
