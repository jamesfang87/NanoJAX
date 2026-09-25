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
    std::vector<std::function<void(Trace<dtype>*, size_t)>> backwards_;

    /**
     * @brief Returns the thread-local stack of active traces; the front is the
     *        outermost trace and the back is the top.
     */
    static std::vector<Trace<dtype>*>& stack() {
        static thread_local std::vector<Trace<dtype>*> traces;
        return traces;
    }

    /**
     * @brief Pushes this trace onto the stack.
     */
    void push() { stack().push_back(this); }

    /**
     * @brief Pops the top of the stack; this trace must be on top.
     */
    void pop() {
        assert(!stack().empty() && stack().back() == this &&
               "Trace::pop: not the top of the stack");
        stack().pop_back();
    }

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
    size_t add_node(Tensor<dtype>& result,
                    std::function<void(Trace<dtype>*, size_t)> backwards_fn) {
        const size_t node = adjoints_.size();
        adjoints_.push_back(Tensor<dtype>::zeros_like(result));
        backwards_.push_back(std::move(backwards_fn));
        result.set_tag(this, node);
        return node;
    }

    /**
     * @brief Attaches @p result on this trace
     * @param result Value to tag.
     */
    void attach(Tensor<dtype>& result) {
        if (!result.has_tag(this)) {
            add_node(result, nullptr);
        }
    }

    /**
     * @brief Returns the cotangent accumulated for node @p id.
     * @param id Node index returned by @ref add_node.
     */
    Tensor<dtype>& adjoint(size_t id) { return adjoints_[id]; }
    const Tensor<dtype>& adjoint(size_t id) const { return adjoints_[id]; }

    /**
     * @brief Returns the cotangent accumulated for @p node on this trace.
     * @param node Tensor tagged on this trace.
     */
    Tensor<dtype>& adjoint(const Tensor<dtype>& node) { return adjoints_[node.id(this)]; }
    const Tensor<dtype>& adjoint(const Tensor<dtype>& node) const {
        return adjoints_[node.id(this)];
    }

    /**
     * @brief Returns the backward closure registered for node @p id.
     * @param id Node index returned by @ref add_node.
     */
    std::function<void(Trace<dtype>*, size_t)>& backwards(size_t id) { return backwards_[id]; }
    const std::function<void(Trace<dtype>*, size_t)>& backwards(size_t id) const {
        return backwards_[id];
    }

    /**
     * @brief Returns the backward closure registered for @p node on this trace.
     * @param node Tensor tagged on this trace.
     */
    std::function<void(Trace<dtype>*, size_t)>& backwards(const Tensor<dtype>& node) {
        return backwards_[node.id(this)];
    }
    const std::function<void(Trace<dtype>*, size_t)>& backwards(const Tensor<dtype>& node) const {
        return backwards_[node.id(this)];
    }

    /**
     * @brief Runs the reverse pass, seeding @p out with a cotangent of one.
     * @param out Scalar output to differentiate from; must belong to this trace.
     */
    void backward(const Tensor<dtype>& out) {
        assert(out.trace() == this && "backward: output belongs to a different trace");
        assert(!stack().empty() && stack().back() == this && "backward: not the top of the stack");

        // Suspend this trace while sweeping so that cotangent operations record
        // on the enclosing trace(s) rather than on the tape being consumed.
        stack().pop_back();

        adjoint(out) = Tensor<dtype>::ones_like(out);

        for (size_t i = adjoints_.size(); i-- > 0;) {
            if (backwards(i)) {
                backwards(i)(this, i);
            }
        }

        stack().push_back(this);
    }
};

} // namespace nanojax
