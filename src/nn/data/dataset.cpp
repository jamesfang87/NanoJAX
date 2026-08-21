#include "nn/data/dataset.hpp"

#include <cassert>
#include <cstdint>
#include <fstream>

namespace {

uint32_t read_u32_be(std::ifstream &stream) {
  unsigned char bytes[4];
  stream.read(reinterpret_cast<char *>(bytes), 4);
  return (static_cast<uint32_t>(bytes[0]) << 24) |
         (static_cast<uint32_t>(bytes[1]) << 16) |
         (static_cast<uint32_t>(bytes[2]) << 8) |
         static_cast<uint32_t>(bytes[3]);
}

} // namespace

MNISTDataset::MNISTDataset(const std::string &images_path,
                           const std::string &labels_path) {
  std::ifstream images_file(images_path, std::ios::binary);
  assert(images_file && "MNISTDataset: failed to open images file");
  uint32_t images_magic = read_u32_be(images_file);
  assert(images_magic == kImageMagic &&
         "MNISTDataset: invalid images file magic number");
  uint32_t num_images = read_u32_be(images_file);
  uint32_t rows = read_u32_be(images_file);
  uint32_t cols = read_u32_be(images_file);
  assert(rows == kRows && cols == kCols &&
         "MNISTDataset: unexpected image dimensions");

  std::ifstream labels_file(labels_path, std::ios::binary);
  assert(labels_file && "MNISTDataset: failed to open labels file");
  uint32_t labels_magic = read_u32_be(labels_file);
  assert(labels_magic == kLabelMagic &&
         "MNISTDataset: invalid labels file magic number");
  uint32_t num_labels = read_u32_be(labels_file);
  assert(num_images == num_labels &&
         "MNISTDataset: image/label count mismatch");

  images_.resize(num_images, std::vector<uint8_t>(kImageSize));
  for (uint32_t i = 0; i < num_images; ++i) {
    images_file.read(reinterpret_cast<char *>(images_[i].data()), kImageSize);
  }

  labels_.resize(num_labels);
  labels_file.read(reinterpret_cast<char *>(labels_.data()), num_labels);
}

size_t MNISTDataset::size() const { return images_.size(); }

Sample MNISTDataset::operator[](size_t index) const {
  assert(index < images_.size() && "MNISTDataset: index out of range");

  const auto &image = images_[index];
  Vector x(kImageSize);
  for (size_t i = 0; i < kImageSize; ++i) {
    x[i] = Scalar{static_cast<double>(image[i]) / 255.0};
  }

  Vector y(kNumClasses, Scalar{0.0});
  y[labels_[index]] = Scalar{1.0};

  return Sample{std::move(x), std::move(y)};
}
