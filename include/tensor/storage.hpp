#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

namespace nanojax {

/**
 * @brief Owns the contiguous data buffer for a Tensor.
 *
 * @tparam dtype Element type. Must be trivially copyable.
 */
template <typename dtype>
    requires std::is_trivially_copyable_v<dtype>
class Storage {
  public:
    /**
     * @brief Allocates room for @p size elements and fills them with @p value.
     * @param size Number of elements to allocate.
     * @param value Value copied into every element.
     */
    Storage(size_t size, dtype value) : data_(std::make_unique<dtype[]>(size)), size_(size) {
        std::fill_n(data_.get(), size, value);
    }

    /**
     * @brief Takes ownership of an existing element buffer.
     * @param data Buffer the storage owns; must hold @p size elements.
     * @param size Number of elements in @p data.
     */
    Storage(std::unique_ptr<dtype[]> data, size_t size) : data_(std::move(data)), size_(size) {}

    /**
     * @brief Allocates room for @p size elements without initializing them.
     *
     * Reading an element before it is assigned is undefined behavior, so only
     * use this when every element will be written before it is read.
     *
     * @param size Number of elements to allocate.
     */
    explicit Storage(size_t size) : data_(std::unique_ptr<dtype[]>(new dtype[size])), size_(size) {}

    /**
     * @brief Returns a pointer to the first element of the buffer.
     * @return Mutable pointer to the elements.
     */
    dtype* data() noexcept { return data_.get(); }

    /**
     * @brief Returns a pointer to the first element of the buffer.
     * @return Const pointer to the elements.
     */
    const dtype* data() const noexcept { return data_.get(); }

    /**
     * @brief Returns the number of elements in the buffer.
     * @return The allocated element count.
     */
    auto size() const { return size_; }

    /**
     * @brief Copies the buffer into a new independently owned storage.
     * @return A new storage holding a copy of this buffer's elements.
     */
    auto clone() const {
        auto data = std::make_unique<dtype[]>(size_);
        std::copy_n(data_.get(), size_, data.get());
        return Storage{std::move(data), size_};
    }

  private:
    std::unique_ptr<dtype[]> data_;
    size_t size_;
};

} // namespace nanojax
