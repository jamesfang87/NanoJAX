#pragma once

#include <cassert>
#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

#include "tensor/tensor.hpp"

namespace nanojax {

/**
 * @brief Trace to differentiate a forward pass using reverse-mode automatic differentiation
 *
 * @tparam dtype Element type. Must be trivially copyable.
 */
template <typename dtype> class Trace {
  public:
    std::vector<Tensor<dtype>> adjoints_;
    std::vector<std::function<void(size_t)>> backwards_;

    /**
     * @brief Pre-allocates room for @p n nodes.
     * @param n Number of nodes the pass is expected to create.
     */
    void reserve(size_t n) {
        adjoints_.reserve(n);
        backwards_.reserve(n);
    }

    /**
     * @brief Appends a node for @p result.
     * @param result Forward value the node represents.
     * @param backwards_fn Closure accumulated during the backward pass; may be
     *                     empty for a leaf.
     * @return The new node's id.
     */
    size_t add_node(Tensor<dtype>& result, std::function<void(size_t)> backwards_fn) {
        const size_t node = adjoints_.size();
        adjoints_.push_back(Tensor<dtype>::zeros_like(result));
        backwards_.push_back(std::move(backwards_fn));
        result.trace_ = this;
        result.id_ = node;
        return node;
    }

    /**
     * @brief Runs the reverse pass, seeding @p out with a cotangent of one.
     * @param out Scalar output to differentiate from; must belong to this trace.
     */
    void backward(const Tensor<dtype>& out) {
        assert(out.trace() == this && "backward: output belongs to a different trace");
        adjoints_[out.id_] = Tensor<dtype>::ones_like(out);

        for (size_t i = adjoints_.size(); i-- > 0;) {
            if (backwards_[i]) {
                backwards_[i](i);
            }
        }
    }
};

} // namespace nanojax
