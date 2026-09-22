#pragma once

#include <ostream>
#include <span>

#include "tensor/tensor.hpp"

namespace nanojax {

namespace detail {

/**
 * @brief Recursively writes a tensor's elements in row-major order.
 * @tparam dtype Element type.
 * @param os Output stream.
 * @param base Pointer to the first element of the sub-tensor.
 * @param shape The shape of the tensor.
 * @param stride The stride of each dimension.
 * @param dim Dimension currently being expanded.
 */
template <typename dtype>
void print(std::ostream& os, const dtype* base, std::span<const size_t> shape,
           std::span<const size_t> stride, size_t dim) {
    os << '[';
    if (dim + 1 == shape.size()) {
        for (size_t i = 0; i < shape[dim]; ++i) {
            if (i != 0) {
                os << ", ";
            }
            os << base[i * stride[dim]];
        }
    } else {
        for (size_t i = 0; i < shape[dim]; ++i) {
            if (i != 0) {
                os << ",\n";
            }
            print(os, base + i * stride[dim], shape, stride, dim + 1);
        }
    }
    os << ']';
}

} // namespace detail

/**
 * @brief Writes @p tensor as its shape followed by its nested values.
 *
 * The elements are read in logical order using the tensor's strides, so a
 * non-contiguous view prints the same values as its contiguous equivalent.
 *
 * @tparam dtype Element type.
 * @param os Output stream.
 * @param tensor Tensor to write.
 * @return Reference to @p os.
 */
template <typename dtype> std::ostream& operator<<(std::ostream& os, const Tensor<dtype>& tensor) {
    const auto shape = tensor.shape();

    os << "Tensor(shape=[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            os << ", ";
        }
        os << shape[i];
    }
    os << "])";

    if (shape.empty()) {
        os << ' ' << *tensor.data();
    } else {
        os << '\n';
        detail::print(os, tensor.data(), shape, tensor.strides(), 0);
    }
    return os;
}

} // namespace nanojax
