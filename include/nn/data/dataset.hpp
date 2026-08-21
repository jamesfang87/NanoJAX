#pragma once

#include "ad/scalar.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct Sample {
  Vector x;
  Vector y;
};

class Dataset {
public:
  virtual ~Dataset() = default;
  virtual size_t size() const = 0;
  virtual Sample operator[](size_t index) const = 0;
};

class MNISTDataset : public Dataset {
public:
  static constexpr size_t kRows = 28;
  static constexpr size_t kCols = 28;
  static constexpr size_t kImageSize = kRows * kCols;
  static constexpr size_t kNumClasses = 10;

  MNISTDataset(const std::string &images_path, const std::string &labels_path);

  size_t size() const override;
  Sample operator[](size_t index) const override;

private:
  static constexpr uint32_t kImageMagic = 0x00000803;
  static constexpr uint32_t kLabelMagic = 0x00000801;

  std::vector<std::vector<uint8_t>> images_;
  std::vector<uint8_t> labels_;
};
