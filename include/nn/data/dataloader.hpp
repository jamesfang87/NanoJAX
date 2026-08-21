#pragma once

#include "nn/data/dataset.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <numeric>
#include <random>
#include <vector>

class DataLoader {
public:
  DataLoader(const Dataset &dataset, size_t batch_size, bool shuffle = true)
      : dataset_(dataset), batch_size_(batch_size), shuffle_(shuffle),
        indices_(dataset.size()), rng_(std::random_device{}()) {
    assert(batch_size_ > 0 && "DataLoader: batch_size must be positive");
    std::iota(indices_.begin(), indices_.end(), 0);
  }

  void reset() {
    if (shuffle_) {
      std::shuffle(indices_.begin(), indices_.end(), rng_);
    }
  }

  size_t num_batches() const {
    return (indices_.size() + batch_size_ - 1) / batch_size_;
  }

  std::vector<Sample> batch(size_t batch_index) const {
    assert(batch_index < num_batches() &&
           "DataLoader: batch_index out of range");

    size_t begin = batch_index * batch_size_;
    size_t end = std::min(begin + batch_size_, indices_.size());

    std::vector<Sample> samples;
    samples.reserve(end - begin);
    for (size_t i = begin; i < end; ++i) {
      samples.push_back(dataset_[indices_[i]]);
    }
    return samples;
  }

private:
  const Dataset &dataset_;
  size_t batch_size_;
  bool shuffle_;
  std::vector<size_t> indices_;
  std::mt19937 rng_;
};
