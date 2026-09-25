#pragma once

#include <ostream>
#include <span>

#include "tensor/tensor.hpp"

namespace nanojax {

namespace detail {

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
