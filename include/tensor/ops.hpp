#pragma once

#include "grad/vjp.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>

#if defined(NANOJAX_HAVE_CBLAS)
#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

#include "tensor/tensor.hpp"

namespace nanojax {

namespace detail {

/**
 * @brief Applies a binary element-wise operation with broadcasting.
 * Neither operand is copied and the result is dense.
 *
 * @tparam dtype Element type.
 * @tparam Op Callable mapping two @p dtype values to a @p dtype.
 * @param lhs Left-hand operand.
 * @param rhs Right-hand operand; must be broadcastable with @p lhs.
 * @param op Element-wise operation, applied as @c op(lhs, rhs).
 * @return A new contiguous tensor holding the result.
 * @throws std::logic_error If the shapes are not broadcastable.
 */
template <typename dtype, typename Op>
Tensor<dtype> binary_broadcast(const Tensor<dtype>& lhs, const Tensor<dtype>& rhs, Op op) {
    const size_t ndim = std::max(lhs.ndim(), rhs.ndim());
    boost::container::small_vector<size_t, 4> shape(ndim, 1);

    for (size_t i = 0; i < rhs.ndim(); ++i) {
        shape[ndim - rhs.ndim() + i] = rhs.shape()[i];
    }
    for (size_t i = 0; i < lhs.ndim(); ++i) {
        size_t& dim = shape[ndim - lhs.ndim() + i];
        dim = std::max(dim, lhs.shape()[i]);
    }

    auto lhs_view = lhs.detach().broadcast(shape);
    auto rhs_view = rhs.detach().broadcast(shape);
    const auto lhs_stride = lhs_view.strides();
    const auto rhs_stride = rhs_view.strides();

    auto result = Tensor<dtype>::uninitialized(shape);
    const auto dst_stride = result.strides();

    const dtype* lhs_data = lhs_view.data();
    const dtype* rhs_data = rhs_view.data();

    size_t lhs_offset = 0;
    size_t rhs_offset = 0;
    size_t dst_offset = 0;

    boost::container::small_vector<size_t, 4> idx(ndim, 0);
    for (size_t count = result.numel(); count > 0; --count) {
        result.data()[dst_offset] = op(lhs_data[lhs_offset], rhs_data[rhs_offset]);

        // Advance to the next index in row-major order.
        for (size_t dim = ndim; dim-- > 0;) {
            ++idx[dim];
            lhs_offset += lhs_stride[dim];
            rhs_offset += rhs_stride[dim];
            dst_offset += dst_stride[dim];

            if (idx[dim] < shape[dim]) {
                break;
            }

            // Reset this index and carry into the next dimension.
            idx[dim] = 0;
            lhs_offset -= lhs_stride[dim] * shape[dim];
            rhs_offset -= rhs_stride[dim] * shape[dim];
            dst_offset -= dst_stride[dim] * shape[dim];
        }
    }

    return result;
}

/**
 * @brief Computes the broadcast shape of @p lhs and @p rhs.
 *
 * Align the two shapes from the right, treating missing leading dimensions as
 * one. For each aligned pair the result dimension is
 *
 *     S[i] = max(lhs[i], rhs[i])   when lhs[i] == 1 or rhs[i] == 1
 *
 * and lhs[i] == rhs[i] otherwise.
 *
 * @param lhs First shape.
 * @param rhs Second shape.
 * @return The broadcast shape.
 * @throws std::invalid_argument If a pair of aligned dimensions are neither
 *         equal nor one.
 */
inline boost::container::small_vector<size_t, 4> broadcast_shapes(std::span<const size_t> lhs,
                                                                  std::span<const size_t> rhs) {
    const size_t ndim = std::max(lhs.size(), rhs.size());
    boost::container::small_vector<size_t, 4> shape(ndim, 1);

    for (size_t i = 0; i < rhs.size(); ++i) {
        shape[ndim - rhs.size() + i] = rhs[i];
    }
    for (size_t i = 0; i < lhs.size(); ++i) {
        const size_t dim = lhs[i];
        size_t& target = shape[ndim - lhs.size() + i];
        if (dim != 1 && target != 1 && dim != target) {
            throw std::invalid_argument{"shapes are not broadcastable"};
        }
        target = std::max(target, dim);
    }
    return shape;
}

/**
 * @brief Concatenates a batch shape with trailing dimensions.
 * @tparam Dims Dimension types, each convertible to @c size_t.
 * @param batch Leading dimensions.
 * @param dims Trailing dimensions.
 * @return A new shape holding @p batch followed by @p dims.
 */
template <typename... Dims>
boost::container::small_vector<size_t, 4> concat_shape(std::span<const size_t> batch,
                                                       Dims... dims) {
    boost::container::small_vector<size_t, 4> shape(batch.begin(), batch.end());
    shape.reserve(shape.size() + sizeof...(dims));
    (shape.push_back(static_cast<size_t>(dims)), ...);
    return shape;
}

#if defined(NANOJAX_HAVE_CBLAS)
template <typename dtype>
inline void unbatched_cblas_matmul(const dtype* pa, const dtype* pb, dtype* pc, size_t M, size_t K,
                                   size_t N) {
    if constexpr (std::is_same_v<dtype, float>) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(M),
                    static_cast<int>(N), static_cast<int>(K), 1.0f, pa, static_cast<int>(K), pb,
                    static_cast<int>(N), 0.0f, pc, static_cast<int>(N));
    } else if constexpr (std::is_same_v<dtype, double>) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(M),
                    static_cast<int>(N), static_cast<int>(K), 1.0, pa, static_cast<int>(K), pb,
                    static_cast<int>(N), 0.0, pc, static_cast<int>(N));
    }
}
#endif

template <typename dtype>
inline void unbatched_NO_cblas_matmul(const dtype* pa, const dtype* pb, dtype* pc, size_t M,
                                      size_t K, size_t N) {
    std::fill_n(pc, M * N, dtype{0});
    for (size_t i = 0; i < M; i++) {
        const dtype* a_row = pa + i * K;
        dtype* c_row = pc + i * N;
        for (size_t k = 0; k < K; ++k) {
            const dtype a_ik = a_row[k];
            const dtype* b_row = pb + k * N;
            for (size_t j = 0; j < N; ++j) {
                c_row[j] += a_ik * b_row[j];
            }
        }
    }
}

template <typename dtype>
inline void unbatched_matmul(const dtype* pa, const dtype* pb, dtype* pc, size_t M, size_t K,
                             size_t N) {
#if defined(NANOJAX_HAVE_CBLAS)
    if constexpr (std::is_same_v<dtype, float> || std::is_same_v<dtype, double>)
        unbatched_cblas_matmul(pa, pb, pc, M, K, N);
    else
        unbatched_NO_cblas_matmul(pa, pb, pc, M, K, N);
#else
    unbatched_NO_cblas_matmul(pa, pb, pc, M, K, N);
#endif
}

} // namespace detail

/**
 * @brief Adds two tensors element-wise with broadcasting.
 * @param lhs Left-hand operand.
 * @param rhs Right-hand operand; must be broadcastable with @p lhs.
 * @return A new contiguous tensor holding the element-wise sum.
 * @throws std::logic_error If the shapes are not broadcastable.
 */
template <typename dtype>
Tensor<dtype> operator+(const Tensor<dtype>& lhs, const Tensor<dtype>& rhs) {
    auto result = detail::binary_broadcast(lhs, rhs, [](dtype a, dtype b) { return a + b; });
    vjp::add(lhs, rhs, result);
    return result;
}

/**
 * @brief Subtracts two tensors element-wise with broadcasting.
 * @param lhs Left-hand operand.
 * @param rhs Right-hand operand; must be broadcastable with @p lhs.
 * @return A new contiguous tensor holding the element-wise difference.
 * @throws std::logic_error If the shapes are not broadcastable.
 */
template <typename dtype>
Tensor<dtype> operator-(const Tensor<dtype>& lhs, const Tensor<dtype>& rhs) {
    auto result = detail::binary_broadcast(lhs, rhs, [](dtype a, dtype b) { return a - b; });
    vjp::sub(lhs, rhs, result);
    return result;
}

/**
 * @brief Multiplies two tensors element-wise with broadcasting.
 * @param lhs Left-hand operand.
 * @param rhs Right-hand operand; must be broadcastable with @p lhs.
 * @return A new contiguous tensor holding the element-wise product.
 * @throws std::logic_error If the shapes are not broadcastable.
 */
template <typename dtype>
Tensor<dtype> operator*(const Tensor<dtype>& lhs, const Tensor<dtype>& rhs) {
    auto result = detail::binary_broadcast(lhs, rhs, [](dtype a, dtype b) { return a * b; });
    vjp::mul(lhs, rhs, result);
    return result;
}

/**
 * @brief Divides two tensors element-wise with broadcasting.
 * @param lhs Left-hand operand.
 * @param rhs Right-hand operand; must be broadcastable with @p lhs.
 * @return A new contiguous tensor holding the element-wise quotient.
 * @throws std::logic_error If the shapes are not broadcastable.
 */
template <typename dtype>
Tensor<dtype> operator/(const Tensor<dtype>& lhs, const Tensor<dtype>& rhs) {
    auto result = detail::binary_broadcast(lhs, rhs, [](dtype a, dtype b) { return a / b; });
    vjp::div(lhs, rhs, result);
    return result;
}

/**
 * @brief Negates every element of a tensor.
 * @param tensor The tensor to negate.
 * @return A new contiguous tensor with each element negated.
 */
template <typename dtype> Tensor<dtype> operator-(const Tensor<dtype>& tensor) {
    auto result = tensor.map([](dtype element) { return -element; });
    vjp::neg(tensor, result);
    return result;
}

/**
 * @brief Adds a scalar to every element of a tensor.
 * @param tensor The tensor.
 * @param scalar Value added to each element.
 * @return A new contiguous tensor with @p scalar added element-wise.
 */
template <typename dtype> Tensor<dtype> operator+(const Tensor<dtype>& tensor, dtype scalar) {
    auto result = tensor.map([scalar](dtype element) { return element + scalar; });
    vjp::scalar_add(tensor, result);
    return result;
}

/**
 * @brief Adds a scalar to every element of a tensor.
 * @param scalar Value added to each element.
 * @param tensor The tensor.
 * @return A new contiguous tensor with @p scalar added element-wise.
 */
template <typename dtype> Tensor<dtype> operator+(dtype scalar, const Tensor<dtype>& tensor) {
    return tensor + scalar;
}

/**
 * @brief Subtracts a scalar from every element of a tensor.
 * @param tensor The tensor.
 * @param scalar Value subtracted from each element.
 * @return A new contiguous tensor with @p scalar subtracted element-wise.
 */
template <typename dtype> Tensor<dtype> operator-(const Tensor<dtype>& tensor, dtype scalar) {
    auto result = tensor.map([scalar](dtype element) { return element - scalar; });
    vjp::lhs_scalar_sub(tensor, result);
    return result;
}

/**
 * @brief Subtracts every element of a tensor from a scalar.
 * @param scalar Value each element is subtracted from.
 * @param tensor The tensor.
 * @return A new contiguous tensor holding @p scalar minus each element.
 */
template <typename dtype> Tensor<dtype> operator-(dtype scalar, const Tensor<dtype>& tensor) {
    auto result = tensor.map([scalar](dtype element) { return scalar - element; });
    vjp::rhs_scalar_sub(tensor, result);
    return result;
}

/**
 * @brief Multiplies every element of a tensor by a scalar.
 * @param tensor The tensor.
 * @param scalar Value multiplied into each element.
 * @return A new contiguous tensor with each element scaled by @p scalar.
 */
template <typename dtype> Tensor<dtype> operator*(const Tensor<dtype>& tensor, dtype scalar) {
    auto result = tensor.map([scalar](dtype element) { return element * scalar; });
    vjp::scalar_mul(tensor, scalar, result);
    return result;
}

/**
 * @brief Multiplies every element of a tensor by a scalar.
 * @param scalar Value multiplied into each element.
 * @param tensor The tensor.
 * @return A new contiguous tensor with each element scaled by @p scalar.
 */
template <typename dtype> Tensor<dtype> operator*(dtype scalar, const Tensor<dtype>& tensor) {
    return tensor * scalar;
}

/**
 * @brief Divides every element of a tensor by a scalar.
 * @param tensor The tensor.
 * @param scalar Value each element is divided by.
 * @return A new contiguous tensor with each element divided by @p scalar.
 */
template <typename dtype> Tensor<dtype> operator/(const Tensor<dtype>& tensor, dtype scalar) {
    auto result = tensor.map([scalar](dtype element) { return element / scalar; });
    vjp::lhs_scalar_div(tensor, scalar, result);
    return result;
}

/**
 * @brief Divides a scalar by every element of a tensor.
 * @param scalar The numerator.
 * @param tensor The tensor.
 * @return A new contiguous tensor holding @p scalar divided by each element.
 */
template <typename dtype> Tensor<dtype> operator/(dtype scalar, const Tensor<dtype>& tensor) {
    auto result = tensor.map([scalar](dtype element) { return scalar / element; });
    vjp::rhs_scalar_div(scalar, tensor, result);
    return result;
}

/**
 * @brief Computes the matrix product of two tensors.
 *
 * Rank-0 operands are rejected.
 * A rank-1 @p lhs is treated as a @c (1,K) matrix and a rank-1 @p rhs as a
 * @c (K,1) matrix. If such promotions occur, any dimension added is removed from the
 * product.
 *
 * @param lhs Left-hand operand, shaped @c (M,K) or @c (K,).
 * @param rhs Right-hand operand, shaped @c (K,N) or @c (K,).
 * @return The product.
 * @throws std::invalid_argument If an operand is rank-0 or the inner dimensions
 *         do not agree.
 */
template <typename dtype> Tensor<dtype> matmul(const Tensor<dtype>& lhs, const Tensor<dtype>& rhs) {
    if (lhs.ndim() == 0 || rhs.ndim() == 0) {
        throw std::invalid_argument{"matmul requires operands of rank at least one"};
    }

    const bool promoted_lhs = lhs.ndim() == 1;
    const bool promoted_rhs = rhs.ndim() == 1;
    const Tensor<dtype> lhs_p = promoted_lhs ? lhs.unsqueeze(0) : lhs;
    const Tensor<dtype> rhs_p = promoted_rhs ? rhs.unsqueeze(rhs.ndim()) : rhs;

    const size_t M = lhs_p.shape()[lhs_p.ndim() - 2];
    const size_t K = lhs_p.shape()[lhs_p.ndim() - 1];
    const size_t N = rhs_p.shape()[rhs_p.ndim() - 1];
    if (rhs_p.shape()[rhs_p.ndim() - 2] != K) {
        throw std::invalid_argument{"Invalid dimensions of lhs and rhs for matmul"};
    }

    auto batch = detail::broadcast_shapes(lhs_p.shape().first(lhs_p.ndim() - 2),
                                          rhs_p.shape().first(rhs_p.ndim() - 2));
    const size_t batch_extent =
        std::accumulate(batch.begin(), batch.end(), size_t{1}, std::multiplies{});

    auto lhs_shape = detail::concat_shape(batch, M, K);
    auto rhs_shape = detail::concat_shape(batch, K, N);
    auto out_shape = detail::concat_shape(batch, M, N);

    auto a = lhs_p.contiguous().detach().broadcast(lhs_shape);
    auto b = rhs_p.contiguous().detach().broadcast(rhs_shape);
    auto c = Tensor<dtype>::uninitialized(out_shape);

    const size_t bndim = batch.size();
    const auto shape = a.shape();
    const auto a_stride = a.strides();
    const auto b_stride = b.strides();
    const auto c_stride = c.strides();
    const dtype* a_data = a.data();
    const dtype* b_data = b.data();
    dtype* c_data = c.data();

    size_t a_offset = 0;
    size_t b_offset = 0;
    size_t c_offset = 0;
    boost::container::small_vector<size_t, 4> idx(bndim, 0);
    for (size_t count = batch_extent; count > 0; --count) {
        detail::unbatched_matmul(a_data + a_offset, b_data + b_offset, c_data + c_offset, M, K, N);

        for (size_t dim = bndim; dim-- > 0;) {
            ++idx[dim];
            a_offset += a_stride[dim];
            b_offset += b_stride[dim];
            c_offset += c_stride[dim];

            const size_t extent = shape[dim];
            if (idx[dim] < extent) {
                break;
            }

            // Reset this index and carry into the next dimension.
            idx[dim] = 0;
            a_offset -= a_stride[dim] * extent;
            b_offset -= b_stride[dim] * extent;
            c_offset -= c_stride[dim] * extent;
        }
    }

    vjp::matmul(lhs_p, rhs_p, c);

    if (!promoted_lhs && !promoted_rhs) {
        return c;
    }

    boost::container::small_vector<size_t, 4> final_shape(out_shape.begin(), out_shape.end());
    if (promoted_lhs) {
        final_shape.erase(final_shape.end() - 2);
    }
    if (promoted_rhs) {
        final_shape.pop_back();
    }
    return c.reshape(final_shape);
}

// TODO: Speed these ops up

/**
 * @brief Applies the exponential function to every element.
 * @param tensor The tensor.
 * @return A new contiguous tensor holding element-wise @c exp.
 */
template <typename dtype> Tensor<dtype> exp(const Tensor<dtype>& tensor) {
    auto result = tensor.map([](dtype element) { return std::exp(element); });
    vjp::exp(tensor, result);
    return result;
}

/**
 * @brief Applies the natural logarithm to every element.
 * @param tensor The tensor.
 * @return A new contiguous tensor holding element-wise @c log.
 */
template <typename dtype> Tensor<dtype> log(const Tensor<dtype>& tensor) {
    auto result = tensor.map([](dtype element) { return std::log(element); });
    vjp::log(tensor, result);
    return result;
}

/**
 * @brief Applies the square root to every element.
 * @param tensor The tensor.
 * @return A new contiguous tensor holding element-wise @c sqrt.
 */
template <typename dtype> Tensor<dtype> sqrt(const Tensor<dtype>& tensor) {
    auto result = tensor.map([](dtype element) { return std::sqrt(element); });
    vjp::sqrt(tensor, result);
    return result;
}

/**
 * @brief Raises every element to a power.
 * @param tensor The tensor.
 * @param exponent Exponent each element is raised to.
 * @return A new contiguous tensor holding element-wise powers.
 */
template <typename dtype> Tensor<dtype> pow(const Tensor<dtype>& tensor, dtype exponent) {
    // TODO: special case integer exponents
    auto result = tensor.map([exponent](dtype element) { return std::pow(element, exponent); });
    vjp::pow(tensor, exponent, result);
    return result;
}

/**
 * @brief Takes the absolute value of every element.
 * @param tensor The tensor.
 * @return A new contiguous tensor holding element-wise absolute values.
 */
template <typename dtype> Tensor<dtype> abs(const Tensor<dtype>& tensor) {
    auto result = tensor.map([](dtype element) { return std::abs(element); });
    vjp::abs(tensor, result);
    return result;
}

/**
 * @brief Applies the rectified linear unit element-wise.
 * @param tensor The tensor.
 * @return A new contiguous tensor holding @c max(0, x) per element.
 */
template <typename dtype> Tensor<dtype> relu(const Tensor<dtype>& tensor) {
    auto result = tensor.map([](dtype element) { return element > dtype{0} ? element : dtype{0}; });
    vjp::relu(tensor, result);
    return result;
}

/**
 * @brief Applies the logistic sigmoid element-wise.
 * @param tensor The tensor.
 * @return A new contiguous tensor holding @c 1/(1+exp(-x)) per element.
 */
template <typename dtype> Tensor<dtype> sigmoid(const Tensor<dtype>& tensor) {
    auto result =
        tensor.map([](dtype element) { return dtype{1} / (dtype{1} + std::exp(-element)); });
    vjp::sigmoid(tensor, result);
    return result;
}

/**
 * @brief Applies the hyperbolic tangent element-wise.
 * @param tensor The tensor.
 * @return A new contiguous tensor holding element-wise @c tanh.
 */
template <typename dtype> Tensor<dtype> tanh(const Tensor<dtype>& tensor) {
    auto result = tensor.map([](dtype element) { return std::tanh(element); });
    vjp::tanh(tensor, result);
    return result;
}

/**
 * @brief Sums every element.
 * @param tensor The tensor.
 * @return A rank-0 tensor holding the total.
 */
template <typename dtype> Tensor<dtype> sum(const Tensor<dtype>& tensor) {
    Tensor<dtype> result = Tensor<dtype>::zeros({});
    const auto ndim = tensor.ndim();
    const auto shape = tensor.shape();
    const auto stride = tensor.strides();

    size_t src_offset = 0;
    boost::container::small_vector<size_t, 4> idx(ndim, 0);
    for (size_t count = tensor.numel(); count > 0; --count) {
        result.data()[0] += tensor.data()[src_offset];

        for (size_t dim = ndim; dim-- > 0;) {
            ++idx[dim];
            src_offset += stride[dim];
            if (idx[dim] < shape[dim]) {
                break;
            }
            idx[dim] = 0;
            src_offset -= stride[dim] * shape[dim];
        }
    }

    vjp::sum(tensor, result);
    return result;
}

/**
 * @brief Sums along one dimension.
 * @param tensor The tensor.
 * @param axis Dimension to reduce; must be less than the tensor's rank.
 * @return A tensor with @p axis removed.
 * @throws std::out_of_range If @p axis is not less than the tensor's rank.
 */
template <typename dtype> Tensor<dtype> sum(const Tensor<dtype>& tensor, size_t axis) {
    if (axis >= tensor.ndim()) {
        throw std::out_of_range{"Tensor::sum: axis is out of range"};
    }

    boost::container::small_vector<size_t, 4> out_shape;
    out_shape.reserve(tensor.ndim() - 1);
    for (size_t d = 0; d < tensor.ndim(); ++d) {
        if (d != axis) {
            out_shape.push_back(tensor.shape()[d]);
        }
    }

    auto result = Tensor<dtype>::zeros(out_shape);

    const auto out_stride = result.strides();
    boost::container::small_vector<size_t, 4> dst_stride(tensor.ndim(), 0);
    for (size_t d = 0; d < tensor.ndim(); ++d) {
        if (d < axis) {
            dst_stride[d] = out_stride[d];
        } else if (d > axis) {
            dst_stride[d] = out_stride[d - 1];
        }
    }

    const auto ndim = tensor.ndim();
    const auto shape = tensor.shape();
    const auto stride = tensor.strides();
    const dtype* src = tensor.data();
    dtype* dst = result.data();

    size_t src_offset = 0;
    size_t dst_offset = 0;
    boost::container::small_vector<size_t, 4> idx(ndim, 0);
    for (size_t count = tensor.numel(); count > 0; --count) {
        dst[dst_offset] += src[src_offset];

        for (size_t dim = ndim; dim-- > 0;) {
            ++idx[dim];
            src_offset += stride[dim];
            dst_offset += dst_stride[dim];
            if (idx[dim] < shape[dim]) {
                break;
            }
            idx[dim] = 0;
            src_offset -= stride[dim] * shape[dim];
            dst_offset -= dst_stride[dim] * shape[dim];
        }
    }

    vjp::sum(tensor, axis, result);
    return result;
}

/**
 * @brief Averages every element.
 * @param tensor The tensor.
 * @return A rank-0 tensor holding the mean.
 * @throws std::invalid_argument If @p tensor has no elements.
 */
template <typename dtype> Tensor<dtype> mean(const Tensor<dtype>& tensor) {
    if (tensor.numel() == 0) {
        throw std::invalid_argument{"mean requires a non-empty tensor"};
    }

    return sum(tensor) / static_cast<dtype>(tensor.numel());
}

/**
 * @brief Averages along one dimension.
 * @param tensor The tensor.
 * @param axis Dimension to reduce; must be less than the tensor's rank.
 * @return A tensor with @p axis removed.
 * @throws std::out_of_range If @p axis is not less than the tensor's rank.
 * @throws std::invalid_argument If the reduced dimension has no elements.
 */
template <typename dtype> Tensor<dtype> mean(const Tensor<dtype>& tensor, size_t axis) {
    if (axis >= tensor.ndim()) {
        throw std::out_of_range{"mean axis is out of range"};
    }
    const size_t extent = tensor.shape()[axis];
    if (extent == 0) {
        throw std::invalid_argument{"mean requires a non-empty reduction axis"};
    }

    return sum(tensor, axis) / static_cast<dtype>(extent);
}

/**
 * @brief Finds the largest element.
 * @param tensor The tensor.
 * @return A rank-0 tensor holding the maximum.
 * @throws std::invalid_argument If @p tensor has no elements.
 */
template <typename dtype> Tensor<dtype> max(const Tensor<dtype>& tensor) {
    if (tensor.numel() == 0) {
        throw std::invalid_argument{"max requires a non-empty tensor"};
    }

    Tensor<dtype> result = Tensor<dtype>::value({}, tensor.data()[0]);
    size_t flat_idx = 0;
    const auto ndim = tensor.ndim();
    const auto shape = tensor.shape();
    const auto stride = tensor.strides();

    size_t src_offset = 0;
    boost::container::small_vector<size_t, 4> idx(ndim, 0);
    for (size_t count = tensor.numel(); count > 0; --count) {
        if (tensor.data()[src_offset] > result.data()[0]) {
            result.data()[0] = tensor.data()[src_offset];
            flat_idx = tensor.numel() - count;
        }

        for (size_t dim = ndim; dim-- > 0;) {
            ++idx[dim];
            src_offset += stride[dim];
            if (idx[dim] < shape[dim]) {
                break;
            }
            idx[dim] = 0;
            src_offset -= stride[dim] * shape[dim];
        }
    }

    vjp::max(tensor, flat_idx, result);
    return result;
}

/**
 * @brief Finds the smallest element.
 * @param tensor The tensor.
 * @return A rank-0 tensor holding the minimum.
 * @throws std::invalid_argument If @p tensor has no elements.
 */
template <typename dtype> Tensor<dtype> min(const Tensor<dtype>& tensor) {
    if (tensor.numel() == 0) {
        throw std::invalid_argument{"min requires a non-empty tensor"};
    }

    Tensor<dtype> result = Tensor<dtype>::value({}, tensor.data()[0]);
    size_t flat_idx = 0;

    const auto ndim = tensor.ndim();
    const auto shape = tensor.shape();
    const auto stride = tensor.strides();

    size_t src_offset = 0;
    boost::container::small_vector<size_t, 4> idx(ndim, 0);
    for (size_t count = tensor.numel(); count > 0; --count) {
        if (tensor.data()[src_offset] < result.data()[0]) {
            result.data()[0] = tensor.data()[src_offset];
            flat_idx = tensor.numel() - count;
        }

        for (size_t dim = ndim; dim-- > 0;) {
            ++idx[dim];
            src_offset += stride[dim];
            if (idx[dim] < shape[dim]) {
                break;
            }
            idx[dim] = 0;
            src_offset -= stride[dim] * shape[dim];
        }
    }

    vjp::min(tensor, flat_idx, result);
    return result;
}

} // namespace nanojax
