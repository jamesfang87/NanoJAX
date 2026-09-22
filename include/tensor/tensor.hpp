#pragma once

#include <boost/container/small_vector.hpp>

#include <cmath>
#include <cstddef>
#include <format>
#include <functional>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <typeinfo>
#include <utility>

#include "storage.hpp"

namespace nanojax {

template <typename dtype> class Trace;

template <typename dtype>
    requires std::is_trivially_copyable_v<dtype>
class Tensor;

namespace vjp {
template <typename dtype> void reshape(const Tensor<dtype>& a, Tensor<dtype>& result);
template <typename dtype> void transpose(const Tensor<dtype>& a, Tensor<dtype>& result);
template <typename dtype>
void permute(const Tensor<dtype>& a, Tensor<dtype>& result, std::span<const size_t> axes);
template <typename dtype> void squeeze(const Tensor<dtype>& a, Tensor<dtype>& result);
template <typename dtype> void unsqueeze(const Tensor<dtype>& a, Tensor<dtype>& result);
template <typename dtype> void broadcast(const Tensor<dtype>& a, Tensor<dtype>& result);
} // namespace vjp

enum class Device {
    CUDA,
    CPU,
};

/**
 * @brief A multi-dimensional array of trivially copyable elements.
 *
 * @tparam dtype Element type. Must be trivially copyable.
 */
template <typename dtype>
    requires std::is_trivially_copyable_v<dtype>
class Tensor {
  private:
    std::shared_ptr<TensorStorage<dtype>> storage_;
    Device device_ = Device::CPU;

    size_t offset_ = 0;
    boost::container::small_vector<size_t, 4> shape_;
    boost::container::small_vector<size_t, 4> stride_;

    template <typename> friend class Trace;

    Trace<dtype>* trace_ = nullptr;
    size_t id_ = 0;

    /**
     * @brief Returns the number of elements in @p shape.
     * @param shape The shape of the tensor.
     * @return The number of elements.
     */
    static size_t extent(std::span<const size_t> shape) {
        return std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies{});
    }

    /**
     * @brief Sets the shape and derives dense row-major strides from it.
     * @param shape The shape assigned to the tensor
     */
    void assign_shape(std::span<const size_t> shape) {
        shape_.assign(shape.begin(), shape.end());
        stride_.resize(shape_.size());
        size_t stride = 1;
        for (size_t i = shape_.size(); i-- > 0;) {
            stride_[i] = stride;
            stride *= shape_[i];
        }
    }

    /**
     * @brief Builds a tensor over existing storage with a dense row-major layout.
     * @param storage Backing storage for the tensor.
     * @param shape The shape of the tensor.
     * @param offset Index of the tensor's first element within @p storage.
     * @param device Where @p storage lives.
     */
    explicit Tensor(std::shared_ptr<TensorStorage<dtype>> storage, std::span<const size_t> shape,
                    size_t offset = 0, Device device = Device::CPU)
        : storage_(std::move(storage)), device_(device), offset_(offset) {
        assign_shape(shape);
    }

    /**
     * @brief Builds a view over existing storage given a shape and strides.
     * @param storage Backing storage for the tensor.
     * @param shape The shape of the tensor.
     * @param stride Distance in elements between successive indices of each dimension.
     * @param offset Index of the view's first element within @p storage.
     * @param device Where @p storage lives.
     * @note The strides are taken as given, so the view need not be contiguous.
     */
    explicit Tensor(std::shared_ptr<TensorStorage<dtype>> storage, std::span<const size_t> shape,
                    std::span<const size_t> stride, size_t offset, Device device)
        : storage_(std::move(storage)), device_(device), offset_(offset),
          shape_(shape.begin(), shape.end()), stride_(stride.begin(), stride.end()) {}

    /**
     * @brief Copies the elements, in logical row-major order, into @p dst.
     *
     * Walks the tensor one element at a time and keeps a running source and
     * destination offset, so each step costs O(1) on average. The caller owns
     * @p dst and guarantees it is large enough to hold every addressed element.
     *
     * @param dst Base of the destination buffer.
     * @param dst_stride Distance in elements between successive indices of each
     *                   destination dimension; must hold @ref ndim entries.
     */
    void gather_to(dtype* dst, std::span<const size_t> dst_stride) const {
        const dtype* src = storage_->data() + offset_;
        size_t src_offset = 0;
        size_t dst_offset = 0;

        boost::container::small_vector<size_t, 4> idx(ndim(), 0);
        for (size_t count = numel(); count > 0; --count) {
            dst[dst_offset] = src[src_offset];

            // Advance to the next index in row-major order.
            for (size_t dim = ndim(); dim-- > 0;) {
                ++idx[dim];
                src_offset += stride_[dim];
                dst_offset += dst_stride[dim];

                if (idx[dim] < shape_[dim]) {
                    break;
                }

                // Reset this index and carry into the next dimension.
                idx[dim] = 0;
                src_offset -= stride_[dim] * shape_[dim];
                dst_offset -= dst_stride[dim] * shape_[dim];
            }
        }
    }

    /**
     * @brief Returns the storage span of this tensor's layout.
     *
     * This is the highest address the strides can reach, plus one: the buffer
     * size an owning copy needs in order to keep this tensor's layout. It equals
     * @ref numel only when the layout is dense, since gapped views occupy more
     * slots than they have elements.
     *
     * @return The address span, or 0 when the tensor has no elements.
     */
    size_t address_span() const {
        if (numel() == 0) {
            return 0;
        }

        size_t span = 1;
        for (size_t k = 0; k < ndim(); ++k) {
            span += (shape_[k] - 1) * stride_[k];
        }
        return span;
    }

    /**
     * @brief Applies a binary operation in place, broadcasting @p other to this shape.
     *
     * @p other is viewed at this tensor's shape, so its stretched and leading
     * dimensions have stride zero; both are walked together in one pass.
     * @p other must broadcast to this shape, not the other way around.
     *
     * @tparam Op Callable mapping two @p dtype values to a @p dtype.
     * @param other Right-hand operand; must broadcast to this tensor's shape.
     * @param op Operation applied as @c op(current, other).
     * @return Reference to this tensor.
     * @throws std::logic_error If @p other cannot broadcast to this shape.
     */
    template <typename Op> Tensor& apply_inplace(const Tensor& other, Op op) {
        if (is_tracked()) {
            throw std::logic_error{"in-place operation on a tracked Tensor"};
        }
        auto other_view = other.detach().broadcast(shape_);
        const auto other_stride = other_view.strides();
        const dtype* src = other_view.data();
        dtype* dst = data();

        size_t dst_offset = 0;
        size_t src_offset = 0;
        boost::container::small_vector<size_t, 4> idx(ndim(), 0);
        for (size_t count = numel(); count > 0; --count) {
            dst[dst_offset] = op(dst[dst_offset], src[src_offset]);

            // Advance to the next index in row-major order.
            for (size_t dim = ndim(); dim-- > 0;) {
                ++idx[dim];
                dst_offset += stride_[dim];
                src_offset += other_stride[dim];

                if (idx[dim] < shape_[dim]) {
                    break;
                }

                // Reset this index and carry into the next dimension.
                idx[dim] = 0;
                dst_offset -= stride_[dim] * shape_[dim];
                src_offset -= other_stride[dim] * shape_[dim];
            }
        }

        return *this;
    }

    /**
     * @brief Applies a unary operation to every element in place.
     *
     * The elements are visited in logical order through this tensor's strides,
     * so a non-contiguous tensor is handled correctly.
     *
     * @tparam Op Callable mapping a @p dtype to a @p dtype.
     * @param op Operation applied to each element.
     * @return Reference to this tensor.
     */
    template <typename Op> Tensor& map_inplace(Op op) {
        if (is_tracked()) {
            throw std::logic_error{"in-place operation on a tracked Tensor"};
        }
        dtype* dst = data();
        size_t dst_offset = 0;
        boost::container::small_vector<size_t, 4> idx(ndim(), 0);
        for (size_t count = numel(); count > 0; --count) {
            dst[dst_offset] = op(dst[dst_offset]);

            // Advance to the next index in row-major order.
            for (size_t dim = ndim(); dim-- > 0;) {
                ++idx[dim];
                dst_offset += stride_[dim];

                if (idx[dim] < shape_[dim]) {
                    break;
                }

                // Reset this index and carry into the next dimension.
                idx[dim] = 0;
                dst_offset -= stride_[dim] * shape_[dim];
            }
        }

        return *this;
    }

  public:
    /**
     * @brief Creates a tensor of the given shape filled with zeros.
     * @param shape The shape of the tensor.
     * @return A newly allocated CPU tensor.
     */
    static Tensor<dtype> zeros(std::initializer_list<size_t> shape);

    /**
     * @brief Creates a tensor of the given shape filled with zeros.
     * @param shape The shape of the tensor.
     * @return A newly allocated CPU tensor.
     */
    static Tensor<dtype> zeros(std::span<const size_t> shape);

    /**
     * @brief Creates a tensor of the given shape filled with ones.
     * @param shape The shape of the tensor.
     * @return A newly allocated CPU tensor.
     */
    static Tensor<dtype> ones(std::initializer_list<size_t> shape);

    /**
     * @brief Creates a tensor of the given shape filled with ones.
     * @param shape The shape of the tensor.
     * @return A newly allocated CPU tensor.
     */
    static Tensor<dtype> ones(std::span<const size_t> shape);

    /**
     * @brief Creates a tensor of the given shape with every element set to @p value.
     * @param shape The shape of the tensor.
     * @param value Fill value copied into every element.
     * @return A newly allocated CPU tensor.
     */
    static Tensor<dtype> value(std::initializer_list<size_t> shape, dtype value);

    /**
     * @brief Creates a tensor of the given shape with every element set to @p value.
     * @param shape The shape of the tensor.
     * @param value Fill value copied into every element.
     * @return A newly allocated CPU tensor.
     */
    static Tensor<dtype> value(std::span<const size_t> shape, dtype value);

    /**
     * @brief Creates a tensor of the given shape without initializing its elements.
     * @param shape The shape of the tensor.
     * @return A newly allocated CPU tensor whose elements are indeterminate.
     * @warning Every element must be assigned before it is read; reading an
     *          uninitialized element is undefined behavior. Safe only when the
     *          caller writes the whole tensor first.
     */
    static Tensor<dtype> uninitialized(std::initializer_list<size_t> shape);

    /**
     * @brief Creates a tensor of the given shape without initializing its elements.
     * @param shape The shape of the tensor.
     * @return A newly allocated CPU tensor whose elements are indeterminate.
     * @warning Every element must be assigned before it is read; reading an
     *          uninitialized element is undefined behavior. Safe only when the
     *          caller writes the whole tensor first.
     */
    static Tensor<dtype> uninitialized(std::span<const size_t> shape);

    static Tensor<dtype> values_like(const Tensor<dtype>& other, dtype value);
    static Tensor<dtype> zeros_like(const Tensor<dtype>& other);
    static Tensor<dtype> ones_like(const Tensor<dtype>& other);

    /**
     * @brief Returns the element type information.
     * @return Reference to the std::type_info for @c dtype.
     */
    const std::type_info& get_dtype() const { return typeid(dtype); }

    /**
     * @brief Returns the shape of the tensor.
     * @return A view over the tensor's shape, valid while the tensor is alive.
     */
    auto shape() const { return std::span<const size_t>(shape_); }

    /**
     * @brief Returns the strides of the tensor.
     * @return A view over the tensor's strides, valid while the tensor is alive.
     */
    auto strides() const { return std::span<const size_t>(stride_); }

    /**
     * @brief Returns the device the tensor is stored on.
     * @return The device holding the tensor's data.
     */
    auto device() const { return device_; }

    /**
     * @brief Returns the number of dimensions (rank).
     * @return The rank; 0 for a scalar, 1 for a vector, and so on.
     */
    auto ndim() const { return shape_.size(); }

    /**
     * @brief Returns the total number of elements.
     * @return The number of elements; 1 for a scalar.
     */
    auto numel() const { return extent(shape_); }

    /**
     * @brief True if the tensor has rank zero.
     * @return True if @ref ndim is zero.
     */
    auto empty() const { return shape_.empty(); }

    /**
     * @brief Returns the value of a single-element tensor.
     * @return The scalar element.
     * @throws std::logic_error If @ref numel is not 1.
     */
    auto item() const {
        if (numel() != 1) {
            throw std::logic_error{"Tensor::item requires a single-element tensor"};
        }
        return storage_->data()[offset_];
    }

    /**
     * @brief Accesses an element by multi-dimensional index.
     * @param self The tensor this is called on.
     * @param index One index per dimension.
     * @return Reference to the addressed element; const when the tensor is const.
     * @throws std::invalid_argument If @p index does not contain @ref ndim entries.
     * @throws std::out_of_range If an index is past the end of its dimension.
     */
    template <typename Self> auto& at(this Self&& self, std::initializer_list<size_t> index) {
        if (index.size() != self.ndim()) {
            throw std::invalid_argument{"Tensor::at requires one index per dimension"};
        }

        size_t addr = self.offset_;
        size_t axis = 0;
        for (size_t i : index) {
            if (i >= self.shape_[axis]) {
                throw std::out_of_range{"Tensor::at index is out of range"};
            }
            addr += i * self.stride_[axis];
            ++axis;
        }

        auto& element = self.storage_->data()[addr];
        if constexpr (std::is_const_v<std::remove_reference_t<Self>>) {
            return std::as_const(element);
        } else {
            if (self.is_tracked()) {
                throw std::logic_error{"mutable access to a tracked Tensor"};
            }
            return element;
        }
    }

    /**
     * @brief Reads an element by multi-dimensional index.
     *
     * Unlike @ref at, this is always const and may be called on a tracked
     * tensor, since reading cannot invalidate the forward values.
     *
     * @param index One index per dimension.
     * @return Const reference to the addressed element.
     * @throws std::invalid_argument If @p index does not contain @ref ndim entries.
     * @throws std::out_of_range If an index is past the end of its dimension.
     */
    const dtype& read(std::initializer_list<size_t> index) const {
        if (index.size() != ndim()) {
            throw std::invalid_argument{"Tensor::read requires one index per dimension"};
        }

        size_t addr = offset_;
        size_t axis = 0;
        for (size_t i : index) {
            if (i >= shape_[axis]) {
                throw std::out_of_range{"Tensor::read index is out of range"};
            }
            addr += i * stride_[axis];
            ++axis;
        }

        return storage_->data()[addr];
    }

    /**
     * @brief Returns a pointer to the tensor's first element.
     * @param self The tensor this is called on.
     * @return Pointer into the backing storage advanced by the tensor's offset; const
     *         when the tensor is const.
     * @note Valid only while the storage is alive. The elements are only laid out
     *       linearly when the tensor is contiguous.
     */
    template <typename Self> auto* data(this Self&& self) {
        dtype* first = self.storage_->data() + self.offset_;
        if constexpr (std::is_const_v<std::remove_reference_t<Self>>) {
            return static_cast<const dtype*>(first);
        } else {
            if (self.is_tracked()) {
                throw std::logic_error{"mutable access to a tracked Tensor"};
            }
            return first;
        }
    }

    /**
     * @brief Adds another tensor element-wise in place.
     * @param other Right-hand operand; must broadcast to this tensor's shape.
     * @return Reference to this tensor.
     * @throws std::logic_error If @p other cannot broadcast to this shape.
     */
    Tensor& operator+=(const Tensor& other) {
        return apply_inplace(other, [](dtype a, dtype b) { return a + b; });
    }

    /**
     * @brief Subtracts another tensor element-wise in place.
     * @param other Right-hand operand; must broadcast to this tensor's shape.
     * @return Reference to this tensor.
     * @throws std::logic_error If @p other cannot broadcast to this shape.
     */
    Tensor& operator-=(const Tensor& other) {
        return apply_inplace(other, [](dtype a, dtype b) { return a - b; });
    }

    /**
     * @brief Multiplies by another tensor element-wise in place.
     * @param other Right-hand operand; must broadcast to this tensor's shape.
     * @return Reference to this tensor.
     * @throws std::logic_error If @p other cannot broadcast to this shape.
     */
    Tensor& operator*=(const Tensor& other) {
        return apply_inplace(other, [](dtype a, dtype b) { return a * b; });
    }

    /**
     * @brief Divides by another tensor element-wise in place.
     * @param other Right-hand operand; must broadcast to this tensor's shape.
     * @return Reference to this tensor.
     * @throws std::logic_error If @p other cannot broadcast to this shape.
     */
    Tensor& operator/=(const Tensor& other) {
        return apply_inplace(other, [](dtype a, dtype b) { return a / b; });
    }

    /**
     * @brief Adds a scalar to every element in place.
     * @param scalar Value added to each element.
     * @return Reference to this tensor.
     */
    Tensor& operator+=(dtype scalar) {
        return map_inplace([scalar](dtype element) { return element + scalar; });
    }

    /**
     * @brief Subtracts a scalar from every element in place.
     * @param scalar Value subtracted from each element.
     * @return Reference to this tensor.
     */
    Tensor& operator-=(dtype scalar) {
        return map_inplace([scalar](dtype element) { return element - scalar; });
    }

    /**
     * @brief Multiplies every element by a scalar in place.
     * @param scalar Value multiplied into each element.
     * @return Reference to this tensor.
     */
    Tensor& operator*=(dtype scalar) {
        return map_inplace([scalar](dtype element) { return element * scalar; });
    }

    /**
     * @brief Divides every element by a scalar in place.
     * @param scalar Value each element is divided by.
     * @return Reference to this tensor.
     */
    Tensor& operator/=(dtype scalar) {
        return map_inplace([scalar](dtype element) { return element / scalar; });
    }

    /**
     * @brief True if this tensor is a node in a trace.
     * @return True if the tensor was produced by a tracked operation.
     */
    bool is_tracked() const { return trace_ != nullptr; }

    /**
     * @brief Returns the trace this tensor belongs to.
     * @return The owning trace, or nullptr when the tensor is untracked.
     */
    Trace<dtype>* trace() const { return trace_; }

    /**
     * @brief Returns this tensor's node index within its trace.
     * @return The node index; meaningless when the tensor is untracked.
     */
    size_t id() const { return id_; }

    /**
     * @brief Returns an untracked handle to this tensor's storage.
     *
     * The result shares storage with this tensor but carries no trace, so
     * operations applied to it are not recorded. Used to run internal,
     * non-differentiable views without polluting the tape.
     *
     * @return An untracked tensor viewing the same elements.
     * @note The handle aliases this tensor, so writing through it would change
     *       both; treat it as read-only when this tensor is tracked.
     */
    Tensor detach() const {
        Tensor result = *this;
        result.trace_ = nullptr;
        result.id_ = 0;
        return result;
    }

    /**
     * @brief True if the elements are laid out densely in memory.
     *
     * A contiguous tensor visits storage in logical order with no gaps or
     * repeats: its strides equal the dense row-major strides for its shape. The
     * stride of a size-one dimension is ignored, since it is never used to
     * advance.
     *
     * @return True if the strides describe a contiguous row-major layout.
     */
    bool is_contiguous() const {
        size_t expected = 1;
        for (size_t i = shape_.size(); i-- > 0;) {
            if (shape_[i] != 1 && stride_[i] != expected) {
                return false;
            }
            expected *= shape_[i];
        }
        return true;
    }

    /**
     * @brief Returns a view with a new shape but the same elements.
     * @param shape The new shape; it must hold the same number of elements.
     * @return A tensor viewing the same storage with the new shape.
     * @throws std::logic_error If @p shape does not contain @ref numel elements.
     * @note A non-contiguous tensor is materialised through @ref contiguous
     *       first, so the returned view always refers to dense storage.
     */
    Tensor reshape(std::span<const size_t> shape) const {
        if (extent(shape) != numel()) {
            throw std::logic_error{
                "New shape for Tensor::reshape must contain the same number of elements"};
        }

        auto cont = this->contiguous();
        Tensor result{cont.storage_, shape, cont.offset_, cont.device_};
        vjp::reshape(*this, result);
        return result;
    }

    /**
     * @brief Returns a view with a new shape but the same elements.
     * @param shape The new shape; it must hold the same number of elements.
     * @return A tensor viewing the same storage with the new shape.
     * @throws std::logic_error If @p shape does not contain @ref numel elements.
     */
    Tensor reshape(std::initializer_list<size_t> shape) const {
        return reshape(std::span<const size_t>(shape.begin(), shape.end()));
    }

    /**
     * @brief Collapses the tensor into one dimension.
     * @return A rank-1 view containing every element in row-major order.
     * @note A non-contiguous tensor is materialised through @ref contiguous
     *       first, so the returned view always refers to dense storage.
     */
    Tensor flatten() const { return reshape({numel()}); }

    /**
     * @brief Reverses the order of the dimensions.
     * @return A view with the dimensions permuted in reverse; generally non-contiguous.
     */
    Tensor transpose() const {
        boost::container::small_vector<size_t, 4> shape(shape_.rbegin(), shape_.rend());
        boost::container::small_vector<size_t, 4> stride(stride_.rbegin(), stride_.rend());
        Tensor result{storage_, shape, stride, offset_, device_};
        vjp::transpose(*this, result);
        return result;
    }

    /**
     * @brief Reorders the dimensions.
     * @param axes New order of the dimensions; must be a permutation of [0, ndim).
     * @return A view with the dimensions reordered as specified.
     * @throws std::invalid_argument If @p axes is not a permutation of [0, ndim).
     */
    Tensor permute(std::span<const size_t> axes) const {
        if (axes.size() != ndim()) {
            throw std::invalid_argument{"Tensor::permute requires one axis per dimension"};
        }

        boost::container::small_vector<char, 4> seen(ndim(), 0);
        boost::container::small_vector<size_t, 4> shape;
        boost::container::small_vector<size_t, 4> stride;
        shape.reserve(ndim());
        stride.reserve(ndim());
        for (size_t axis : axes) {
            if (axis >= ndim() || seen[axis]) {
                throw std::invalid_argument{
                    "Tensor::permute axes must be a permutation of [0, ndim)"};
            }
            seen[axis] = 1;
            shape.push_back(shape_[axis]);
            stride.push_back(stride_[axis]);
        }
        Tensor result{storage_, shape, stride, offset_, device_};
        vjp::permute(*this, result, axes);
        return result;
    }

    /**
     * @brief Reorders the dimensions.
     * @param axes New order of the dimensions; must be a permutation of [0, ndim).
     * @return A view with the dimensions reordered as specified.
     * @throws std::invalid_argument If @p axes is not a permutation of [0, ndim).
     */
    Tensor permute(std::initializer_list<size_t> axes) const {
        return permute(std::span<const size_t>(axes.begin(), axes.end()));
    }

    /**
     * @brief Removes dimensions of size one.
     * @return A view with all size-one dimensions dropped.
     */
    Tensor squeeze() const {
        boost::container::small_vector<size_t, 4> shape;
        boost::container::small_vector<size_t, 4> stride;
        shape.reserve(ndim());
        stride.reserve(ndim());
        for (size_t i = 0; i < ndim(); ++i) {
            if (shape_[i] == 1) {
                continue;
            }
            shape.push_back(shape_[i]);
            stride.push_back(stride_[i]);
        }
        Tensor result{storage_, shape, stride, offset_, device_};
        vjp::squeeze(*this, result);
        return result;
    }

    Tensor squeeze(size_t axis) const {
        if (axis >= ndim()) {
            throw std::out_of_range{"Tensor::squeeze axis is out of range"};
        }

        boost::container::small_vector<size_t, 4> shape;
        boost::container::small_vector<size_t, 4> stride;
        shape.reserve(ndim());
        stride.reserve(ndim());
        for (size_t i = 0; i < ndim(); ++i) {
            if (i == axis) {
                if (shape_[i] != 1) {
                    throw std::invalid_argument{"Tensor::squeeze axis does not have size one"};
                }
                continue;
            }
            shape.push_back(shape_[i]);
            stride.push_back(stride_[i]);
        }
        Tensor result{storage_, shape, stride, offset_, device_};
        vjp::squeeze(*this, result);
        return result;
    }

    /**
     * @brief Inserts a dimension of size one.
     * @param axis Position at which to insert the new dimension.
     * @return A view with one extra dimension.
     * @throws std::out_of_range If @p axis is greater than @ref ndim.
     */
    Tensor unsqueeze(size_t axis) const {
        if (axis > ndim()) {
            std::string msg =
                std::format("out of range `axis` for a Tensor of {} dimensions", ndim());
            throw std::out_of_range(msg);
        }

        boost::container::small_vector<size_t, 4> shape;
        boost::container::small_vector<size_t, 4> stride;
        shape.reserve(ndim() + 1);
        stride.reserve(ndim() + 1);

        for (size_t i = 0; i < axis; ++i) {
            shape.push_back(shape_[i]);
            stride.push_back(stride_[i]);
        }

        shape.push_back(1);
        // A size-one dimension never advances, so its stride is arbitrary.
        stride.push_back(1);

        for (size_t i = axis + 1; i <= ndim(); ++i) {
            shape.push_back(shape_[i - 1]);
            stride.push_back(stride_[i - 1]);
        }

        Tensor result{storage_, shape, stride, offset_, device_};
        vjp::unsqueeze(*this, result);
        return result;
    }

    /**
     * @brief Expands the tensor to a larger shape by repeating elements.
     * @param shape Target shape; each dimension must match this tensor's or be
     *              one. The target may have leading dimensions this tensor lacks.
     * @return A broadcasted view over the same storage.
     * @throws std::logic_error If the target has fewer dimensions than this
     *         tensor, or a dimension of size greater than one does not match.
     */
    Tensor broadcast(std::span<const size_t> shape) const {
        boost::container::small_vector<size_t, 4> target_shape(shape.begin(), shape.end());

        if (target_shape.size() < shape_.size()) {
            throw std::logic_error{
                "Tensor::broadcast target rank must be at least the tensor's rank"};
        }

        const size_t leading = target_shape.size() - shape_.size();
        for (size_t i = 0; i < shape_.size(); ++i) {
            const size_t target = target_shape[leading + i];
            if (shape_[i] != 1 && shape_[i] != target) {
                throw std::logic_error{
                    "Tensor::broadcast cannot stretch a dimension of size greater than one"};
            }
        }

        boost::container::small_vector<size_t, 4> stride;
        stride.reserve(target_shape.size());

        for (size_t i = 0; i < leading; i++) {
            stride.push_back(size_t{0});
        }

        for (size_t i = 0; i < shape_.size(); i++) {
            stride.push_back(shape_[i] == 1 ? 0 : stride_[i]);
        }

        Tensor result{storage_, target_shape, stride, offset_, device_};
        vjp::broadcast(*this, result);
        return result;
    }

    /**
     * @brief Expands the tensor to a larger shape by repeating elements.
     * @param shape Target shape; each dimension must match this tensor's or be
     *              one. The target may have leading dimensions this tensor lacks.
     * @return A broadcasted view over the same storage.
     * @throws std::logic_error If the target has fewer dimensions than this
     *         tensor, or a dimension of size greater than one does not match.
     */
    Tensor broadcast(std::initializer_list<size_t> shape) const {
        return broadcast(std::span<const size_t>(shape.begin(), shape.end()));
    }

    /**
     * @brief Returns a deep copy owning its own storage.
     *
     * Unlike the view operations, this always allocates and copies. The result
     * keeps this tensor's shape and strides, so the layout (including
     * non-contiguity) is preserved; the storage offset is normalized to zero.
     *
     * @return A new tensor holding an independent copy of the elements.
     * @note Writes to the result do not affect this tensor.
     */
    Tensor clone() const {
        auto storage = std::make_shared<TensorStorage<dtype>>(address_span(), dtype{});
        Tensor result{storage, shape_, stride_, 0, device_};
        gather_to(result.data(), stride_);
        result.trace_ = trace_;
        result.id_ = id_;
        return result;
    }

    /**
     * @brief Returns a tensor whose elements are stored densely in row-major order.
     *
     * When this tensor is already contiguous and @p always_copy is false, the
     * result shares this tensor's storage and nothing is copied. Otherwise new
     * storage is allocated and the elements are gathered in logical order.
     *
     * @param always_copy Force a fresh contiguous copy even when this tensor is
     *                    already contiguous; used when the caller intends to
     *                    mutate the result.
     * @return A contiguous tensor sharing this one's storage when possible, or a
     *         fresh contiguous copy otherwise.
     * @note The copying path breaks aliasing: writes to the result do not affect
     *       this tensor.
     */
    Tensor contiguous(bool always_copy = false) const {
        if (is_contiguous() && !always_copy) {
            return *this;
        }

        auto storage = std::make_shared<TensorStorage<dtype>>(numel(), dtype{});
        Tensor result{storage, shape_, 0, device_};
        gather_to(result.data(), result.stride_);
        result.trace_ = trace_;
        result.id_ = id_;
        return result;
    }

    /**
     * @brief Applies @p op to every element, returning a new tensor.
     *
     * The result is dense (row-major) regardless of this tensor's layout, and the
     * source is read in logical order, so the operation costs a single pass over
     * the elements. This tensor is not modified.
     *
     * @tparam Op Callable mapping a @p dtype to a @p dtype.
     * @param op Element-wise operation.
     * @return A new contiguous tensor holding @p op applied to each element.
     * @note The result is allocated uninitialized, but every element is written
     *       before it is read, so no indeterminate value escapes.
     */
    template <typename Op> Tensor<dtype> map(Op op) const {
        auto storage = std::make_shared<TensorStorage<dtype>>(numel());
        Tensor result{storage, shape_};

        const dtype* src = storage_->data() + offset_;
        dtype* dst = result.data();

        if (is_contiguous()) {
            for (size_t i = 0; i < numel(); ++i) {
                dst[i] = op(src[i]);
            }
            return result;
        }

        boost::container::small_vector<size_t, 4> idx(ndim(), 0);
        size_t src_offset = 0;
        for (size_t linear = 0; linear < numel(); ++linear) {
            dst[linear] = op(src[src_offset]);

            for (size_t dim = ndim(); dim-- > 0;) {
                ++idx[dim];
                src_offset += stride_[dim];
                if (idx[dim] < shape_[dim]) {
                    break;
                }
                idx[dim] = 0;
                src_offset -= stride_[dim] * shape_[dim];
            }
        }
        return result;
    }

    /**
     * @brief Compares two tensors element-wise within a tolerance.
     *
     * Both tensors are read in logical order through their own strides, so the
     * layouts need not match and the elements need not be dense. A tensor with no
     * elements compares equal to any other empty tensor.
     *
     * @param other Tensor to compare against; shapes must match.
     * @param tolerance Maximum allowed absolute difference per element.
     * @return True if every element agrees within @p tolerance.
     * @throws std::logic_error If @p other's shape does not match this tensor's.
     * @throws std::invalid_argument If @p tolerance is negative.
     */
    bool allclose(const Tensor& other, double tolerance) const {
        if (shape_ != other.shape_) {
            throw std::logic_error{"Tensor::allclose requires matching shapes"};
        }
        if (tolerance < 0.0) {
            throw std::invalid_argument{"Tensor::allclose requires a non-negative tolerance"};
        }

        const dtype* lhs = data();
        const dtype* rhs = other.data();

        size_t lhs_offset = 0;
        size_t rhs_offset = 0;
        boost::container::small_vector<size_t, 4> idx(ndim(), 0);
        for (size_t count = numel(); count > 0; --count) {
            const double diff =
                static_cast<double>(lhs[lhs_offset]) - static_cast<double>(rhs[rhs_offset]);
            if (std::abs(diff) > tolerance) {
                return false;
            }

            for (size_t dim = ndim(); dim-- > 0;) {
                ++idx[dim];
                lhs_offset += stride_[dim];
                rhs_offset += other.stride_[dim];

                if (idx[dim] < shape_[dim]) {
                    break;
                }

                idx[dim] = 0;
                lhs_offset -= stride_[dim] * shape_[dim];
                rhs_offset -= other.stride_[dim] * shape_[dim];
            }
        }

        return true;
    }
};

template <typename dtype>
    requires std::is_trivially_copyable_v<dtype>
Tensor<dtype> Tensor<dtype>::value(std::initializer_list<size_t> shape, dtype value) {
    return Tensor<dtype>::value(std::span<const size_t>{shape}, value);
}

template <typename dtype>
    requires std::is_trivially_copyable_v<dtype>
Tensor<dtype> Tensor<dtype>::value(std::span<const size_t> shape, dtype value) {
    auto storage = std::make_shared<TensorStorage<dtype>>(extent(shape), value);
    return Tensor{std::move(storage), shape};
}

template <typename dtype>
    requires std::is_trivially_copyable_v<dtype>
Tensor<dtype> Tensor<dtype>::zeros(std::initializer_list<size_t> shape) {
    return Tensor<dtype>::zeros(std::span<const size_t>{shape});
}

template <typename dtype>
    requires std::is_trivially_copyable_v<dtype>
Tensor<dtype> Tensor<dtype>::zeros(std::span<const size_t> shape) {
    return value(shape, dtype{});
}

template <typename dtype>
    requires std::is_trivially_copyable_v<dtype>
Tensor<dtype> Tensor<dtype>::ones(std::initializer_list<size_t> shape) {
    return Tensor<dtype>::ones(std::span<const size_t>{shape});
}

template <typename dtype>
    requires std::is_trivially_copyable_v<dtype>
Tensor<dtype> Tensor<dtype>::ones(std::span<const size_t> shape) {
    return value(shape, dtype{1});
}

template <typename dtype>
    requires std::is_trivially_copyable_v<dtype>
Tensor<dtype> Tensor<dtype>::uninitialized(std::initializer_list<size_t> shape) {
    return Tensor<dtype>::uninitialized(std::span<const size_t>{shape});
}

template <typename dtype>
    requires std::is_trivially_copyable_v<dtype>
Tensor<dtype> Tensor<dtype>::uninitialized(std::span<const size_t> shape) {
    auto storage = std::make_shared<TensorStorage<dtype>>(extent(shape));
    return Tensor{std::move(storage), shape};
}

template <typename dtype>
    requires std::is_trivially_copyable_v<dtype>
Tensor<dtype> Tensor<dtype>::values_like(const Tensor<dtype>& other, dtype value) {
    return Tensor<dtype>::value(other.shape(), value);
}

template <typename dtype>
    requires std::is_trivially_copyable_v<dtype>
Tensor<dtype> Tensor<dtype>::zeros_like(const Tensor<dtype>& other) {
    return Tensor<dtype>::zeros(other.shape());
}

template <typename dtype>
    requires std::is_trivially_copyable_v<dtype>
Tensor<dtype> Tensor<dtype>::ones_like(const Tensor<dtype>& other) {
    return Tensor<dtype>::ones(other.shape());
}

} // namespace nanojax
