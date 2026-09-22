#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>
#include <utility>
#include <utility>

#include "grad/trace.hpp"
#include "tensor/tensor.hpp"

namespace nanojax {
namespace vjp {

template <typename dtype>
Tensor<dtype> unbroadcast(const Tensor<dtype>& g, std::span<const size_t> target) {
    Tensor<dtype> out = Tensor<dtype>::zeros(target);
    out.broadcast(g.shape()) += g;
    return out;
}

/**
 * @brief Records @c a+b.
 * @param a Left operand.
 * @param b Right operand.
 * @param c Forward sum; tagged as a node when either operand is tracked.
 */
template <typename dtype>
void add(const Tensor<dtype>& a, const Tensor<dtype>& b, Tensor<dtype>& c) {
    if (!a.is_tracked() && !b.is_tracked()) {
        return;
    }
    assert((!a.is_tracked() || !b.is_tracked() || a.trace() == b.trace()) &&
           "vjp::add: operands belong to different traces");

    Trace<dtype>* tr = a.is_tracked() ? a.trace() : b.trace();
    tr->add_node(c, [tr, a, b](size_t self_id) {
        if (a.is_tracked()) {
            tr->adjoints_[a.id()] =
                tr->adjoints_[a.id()] + unbroadcast(tr->adjoints_[self_id], a.shape());
        }
        if (b.is_tracked()) {
            tr->adjoints_[b.id()] =
                tr->adjoints_[b.id()] + unbroadcast(tr->adjoints_[self_id], b.shape());
        }
    });
}

/**
 * @brief Records @c a-b.
 * @param a Left operand.
 * @param b Right operand.
 * @param c Forward difference; tagged as a node when either operand is tracked.
 */
template <typename dtype>
void sub(const Tensor<dtype>& a, const Tensor<dtype>& b, Tensor<dtype>& c) {
    if (!a.is_tracked() && !b.is_tracked()) {
        return;
    }
    assert((!a.is_tracked() || !b.is_tracked() || a.trace() == b.trace()) &&
           "vjp::sub: operands belong to different traces");

    Trace<dtype>* tr = a.is_tracked() ? a.trace() : b.trace();
    tr->add_node(c, [tr, a, b](size_t self_id) {
        if (a.is_tracked()) {
            tr->adjoints_[a.id()] =
                tr->adjoints_[a.id()] + unbroadcast(tr->adjoints_[self_id], a.shape());
        }
        if (b.is_tracked()) {
            tr->adjoints_[b.id()] =
                tr->adjoints_[b.id()] - unbroadcast(tr->adjoints_[self_id], b.shape());
        }
    });
}

/**
 * @brief Records @c a*b.
 * @param a Left operand.
 * @param b Right operand.
 * @param c Forward product; tagged as a node when either operand is tracked.
 */
template <typename dtype>
void mul(const Tensor<dtype>& a, const Tensor<dtype>& b, Tensor<dtype>& c) {
    if (!a.is_tracked() && !b.is_tracked()) {
        return;
    }
    assert((!a.is_tracked() || !b.is_tracked() || a.trace() == b.trace()) &&
           "vjp::mul: operands belong to different traces");

    Trace<dtype>* tr = a.is_tracked() ? a.trace() : b.trace();
    tr->add_node(c, [tr, a, b](size_t self_id) {
        if (a.is_tracked()) {
            tr->adjoints_[a.id()] =
                tr->adjoints_[a.id()] +
                unbroadcast(tr->adjoints_[self_id] * b.detach(), a.shape());
        }
        if (b.is_tracked()) {
            tr->adjoints_[b.id()] =
                tr->adjoints_[b.id()] +
                unbroadcast(tr->adjoints_[self_id] * a.detach(), b.shape());
        }
    });
}

/**
 * @brief Records @c a/b.
 * @param a Numerator.
 * @param b Denominator.
 * @param c Forward quotient; tagged as a node when either operand is tracked.
 */
template <typename dtype>
void div(const Tensor<dtype>& a, const Tensor<dtype>& b, Tensor<dtype>& c) {
    if (!a.is_tracked() && !b.is_tracked()) {
        return;
    }
    assert((!a.is_tracked() || !b.is_tracked() || a.trace() == b.trace()) &&
           "vjp::div: operands belong to different traces");

    Trace<dtype>* tr = a.is_tracked() ? a.trace() : b.trace();
    tr->add_node(c, [tr, a, b](size_t self_id) {
        const Tensor<dtype>& bval = b.detach();
        if (a.is_tracked()) {
            tr->adjoints_[a.id()] =
                tr->adjoints_[a.id()] + unbroadcast(tr->adjoints_[self_id] / bval, a.shape());
        }
        if (b.is_tracked()) {
            tr->adjoints_[b.id()] =
                tr->adjoints_[b.id()] -
                unbroadcast(tr->adjoints_[self_id] * a.detach() / (bval * bval), b.shape());
        }
    });
}

/**
 * @brief Records @c -a.
 * @param a Operand.
 * @param b Forward negation; tagged as a node when @p a is tracked.
 */
template <typename dtype> void neg(const Tensor<dtype>& a, Tensor<dtype>& b) {
    if (!a.is_tracked()) {
        return;
    }

    Trace<dtype>* tr = a.trace();
    tr->add_node(b, [tr, a](size_t self_id) {
        tr->adjoints_[a.id()] = tr->adjoints_[a.id()] - tr->adjoints_[self_id];
    });
}

template <typename dtype> void scalar_add(const Tensor<dtype>& a, Tensor<dtype>& b) {
    if (!a.is_tracked()) {
        return;
    }

    Trace<dtype>* tr = a.trace();
    tr->add_node(b, [tr, a](size_t self_id) {
        tr->adjoints_[a.id()] = tr->adjoints_[a.id()] + tr->adjoints_[self_id];
    });
}

template <typename dtype> void lhs_scalar_sub(const Tensor<dtype>& a, Tensor<dtype>& b) {
    if (!a.is_tracked()) {
        return;
    }

    Trace<dtype>* tr = a.trace();
    tr->add_node(b, [tr, a](size_t self_id) {
        tr->adjoints_[a.id()] = tr->adjoints_[a.id()] + tr->adjoints_[self_id];
    });
}

template <typename dtype> void rhs_scalar_sub(const Tensor<dtype>& a, Tensor<dtype>& b) {
    if (!a.is_tracked()) {
        return;
    }

    Trace<dtype>* tr = a.trace();
    tr->add_node(b, [tr, a](size_t self_id) {
        tr->adjoints_[a.id()] = tr->adjoints_[a.id()] - tr->adjoints_[self_id];
    });
}

template <typename dtype> void scalar_mul(const Tensor<dtype>& a, dtype b, Tensor<dtype>& c) {
    if (!a.is_tracked()) {
        return;
    }

    Trace<dtype>* tr = a.trace();
    tr->add_node(c, [tr, a, b](size_t self_id) {
        tr->adjoints_[a.id()] = tr->adjoints_[a.id()] + tr->adjoints_[self_id] * b;
    });
}

template <typename dtype> void lhs_scalar_div(const Tensor<dtype>& a, dtype b, Tensor<dtype>& c) {
    if (!a.is_tracked()) {
        return;
    }

    Trace<dtype>* tr = a.trace();
    tr->add_node(c, [tr, a, b](size_t self_id) {
        tr->adjoints_[a.id()] = tr->adjoints_[a.id()] + tr->adjoints_[self_id] / b;
    });
}

template <typename dtype> void rhs_scalar_div(dtype a, const Tensor<dtype>& b, Tensor<dtype>& c) {
    if (!b.is_tracked()) {
        return;
    }

    Trace<dtype>* tr = b.trace();
    tr->add_node(c, [tr, b, a](size_t self_id) {
        const Tensor<dtype>& bval = b.detach();
        tr->adjoints_[b.id()] = tr->adjoints_[b.id()] - tr->adjoints_[self_id] * a / (bval * bval);
    });
}

template <typename dtype> void exp(const Tensor<dtype>& a, Tensor<dtype>& c) {
    if (!a.is_tracked()) {
        return;
    }

    Trace<dtype>* tr = a.trace();
    tr->add_node(c, [tr, a, c](size_t self_id) {
        tr->adjoints_[a.id()] = tr->adjoints_[a.id()] + tr->adjoints_[self_id] * c;
    });
}

template <typename dtype> void log(const Tensor<dtype>& a, Tensor<dtype>& c) {
    if (!a.is_tracked()) {
        return;
    }

    Trace<dtype>* tr = a.trace();
    tr->add_node(c, [tr, a](size_t self_id) {
        tr->adjoints_[a.id()] = tr->adjoints_[a.id()] + tr->adjoints_[self_id] / a.detach();
    });
}

template <typename dtype> void sqrt(const Tensor<dtype>& a, Tensor<dtype>& c) {
    if (!a.is_tracked()) {
        return;
    }

    Trace<dtype>* tr = a.trace();
    tr->add_node(c, [tr, a, c](size_t self_id) {
        tr->adjoints_[a.id()] = tr->adjoints_[a.id()] + tr->adjoints_[self_id] / (c * dtype{2});
    });
}

template <typename dtype> void pow(const Tensor<dtype>& a, dtype exponent, Tensor<dtype>& c) {
    if (!a.is_tracked()) {
        return;
    }

    Trace<dtype>* tr = a.trace();
    tr->add_node(c, [tr, a, exponent](size_t self_id) {
        auto derivative = a.detach().map(
            [exponent](dtype element) { return std::pow(element, exponent - dtype{1}); });
        tr->adjoints_[a.id()] =
            tr->adjoints_[a.id()] + (tr->adjoints_[self_id] * exponent) * derivative;
    });
}

template <typename dtype> void abs(const Tensor<dtype>& a, Tensor<dtype>& c) {
    if (!a.is_tracked()) {
        return;
    }

    Trace<dtype>* tr = a.trace();
    tr->add_node(c, [tr, a](size_t self_id) {
        auto sign = a.detach().map([](dtype element) {
            return element > dtype{0} ? dtype{1} : (element < dtype{0} ? dtype{-1} : dtype{0});
        });
        tr->adjoints_[a.id()] = tr->adjoints_[a.id()] + tr->adjoints_[self_id] * sign;
    });
}

template <typename dtype> void relu(const Tensor<dtype>& a, Tensor<dtype>& c) {
    if (!a.is_tracked()) {
        return;
    }

    Trace<dtype>* tr = a.trace();
    tr->add_node(c, [tr, a](size_t self_id) {
        auto mask = a.detach().map(
            [](dtype element) { return element > dtype{0} ? dtype{1} : dtype{0}; });
        tr->adjoints_[a.id()] = tr->adjoints_[a.id()] + tr->adjoints_[self_id] * mask;
    });
}

template <typename dtype> void sigmoid(const Tensor<dtype>& a, Tensor<dtype>& c) {
    if (!a.is_tracked()) {
        return;
    }

    Trace<dtype>* tr = a.trace();
    tr->add_node(c, [tr, a, c](size_t self_id) {
        tr->adjoints_[a.id()] = tr->adjoints_[a.id()] + tr->adjoints_[self_id] * c * (dtype{1} - c);
    });
}

template <typename dtype> void tanh(const Tensor<dtype>& a, Tensor<dtype>& c) {
    if (!a.is_tracked()) {
        return;
    }

    Trace<dtype>* tr = a.trace();
    tr->add_node(c, [tr, a, c](size_t self_id) {
        tr->adjoints_[a.id()] = tr->adjoints_[a.id()] + tr->adjoints_[self_id] * (dtype{1} - c * c);
    });
}

template <typename dtype> void sum(const Tensor<dtype>& a, Tensor<dtype>& b) {
    if (!a.is_tracked()) {
        return;
    }

    Trace<dtype>* tr = a.trace();
    tr->add_node(b, [tr, a](size_t self_id) {
        tr->adjoints_[a.id()] += tr->adjoints_[self_id].broadcast(a.shape());
    });
}

template <typename dtype> void sum(const Tensor<dtype>& a, size_t axis, Tensor<dtype>& b) {
    if (!a.is_tracked()) {
        return;
    }

    Trace<dtype>* tr = a.trace();
    tr->add_node(b, [tr, a, axis](size_t self_id) {
        tr->adjoints_[a.id()] += tr->adjoints_[self_id].unsqueeze(axis).broadcast(a.shape());
    });
}
template <typename dtype> void max(const Tensor<dtype>& a, size_t flat_idx, Tensor<dtype>& b) {
    if (!a.is_tracked()) {
        return;
    }

    Trace<dtype>* tr = a.trace();
    tr->add_node(b, [tr, a, flat_idx](size_t self_id) {
        Tensor<dtype> onehot = Tensor<dtype>::zeros_like(a);
        onehot.data()[flat_idx] = dtype{1};

        tr->adjoints_[a.id()] += tr->adjoints_[self_id] * onehot;
    });
}

template <typename dtype> void min(const Tensor<dtype>& a, size_t flat_idx, Tensor<dtype>& b) {
    if (!a.is_tracked()) {
        return;
    }

    Trace<dtype>* tr = a.trace();
    tr->add_node(b, [tr, a, flat_idx](size_t self_id) {
        Tensor<dtype> onehot = Tensor<dtype>::zeros_like(a);
        onehot.data()[flat_idx] = dtype{1};

        tr->adjoints_[a.id()] += tr->adjoints_[self_id] * onehot;
    });
}

template <typename dtype> void reshape(const Tensor<dtype>& a, Tensor<dtype>& result) {
    if (!a.is_tracked()) {
        return;
    }

    Trace<dtype>* tr = a.trace();
    tr->add_node(result, [tr, a](size_t self_id) {
        tr->adjoints_[a.id()] += tr->adjoints_[self_id].reshape(a.shape());
    });
}

template <typename dtype> void transpose(const Tensor<dtype>& a, Tensor<dtype>& result) {
    if (!a.is_tracked()) {
        return;
    }

    Trace<dtype>* tr = a.trace();
    tr->add_node(result, [tr, a](size_t self_id) {
        tr->adjoints_[a.id()] += tr->adjoints_[self_id].transpose();
    });
}

template <typename dtype>
void permute(const Tensor<dtype>& a, Tensor<dtype>& result, std::span<const size_t> axes) {
    if (!a.is_tracked()) {
        return;
    }

    boost::container::small_vector<size_t, 4> inverse(axes.size());
    for (size_t i = 0; i < axes.size(); ++i) {
        inverse[axes[i]] = i;
    }

    Trace<dtype>* tr = a.trace();
    tr->add_node(result, [tr, a, inverse](size_t self_id) {
        tr->adjoints_[a.id()] += tr->adjoints_[self_id].permute(std::span<const size_t>(inverse));
    });
}

template <typename dtype> void squeeze(const Tensor<dtype>& a, Tensor<dtype>& result) {
    if (!a.is_tracked()) {
        return;
    }

    Trace<dtype>* tr = a.trace();
    tr->add_node(result, [tr, a](size_t self_id) {
        tr->adjoints_[a.id()] += tr->adjoints_[self_id].reshape(a.shape());
    });
}

template <typename dtype> void unsqueeze(const Tensor<dtype>& a, Tensor<dtype>& result) {
    if (!a.is_tracked()) {
        return;
    }

    Trace<dtype>* tr = a.trace();
    tr->add_node(result, [tr, a](size_t self_id) {
        tr->adjoints_[a.id()] += tr->adjoints_[self_id].reshape(a.shape());
    });
}

template <typename dtype> void broadcast(const Tensor<dtype>& a, Tensor<dtype>& result) {
    if (!a.is_tracked()) {
        return;
    }

    Trace<dtype>* tr = a.trace();
    tr->add_node(result, [tr, a](size_t self_id) {
        tr->adjoints_[a.id()] += vjp::unbroadcast(tr->adjoints_[self_id], a.shape());
    });
}

template <typename dtype>
Tensor<dtype> transpose_last_two(const Tensor<dtype>& tensor) {
    boost::container::small_vector<size_t, 4> axes;
    axes.reserve(tensor.ndim());
    for (size_t i = 0; i < tensor.ndim(); ++i) {
        axes.push_back(i);
    }
    if (tensor.ndim() >= 2) {
        std::swap(axes[tensor.ndim() - 2], axes[tensor.ndim() - 1]);
    }
    return tensor.permute(axes);
}

template <typename dtype>
Tensor<dtype> unbatch(const Tensor<dtype>& cot, std::span<const size_t> target) {
    boost::container::small_vector<size_t, 4> shape;
    for (size_t dim = 0; dim + 2 < target.size(); ++dim) {
        shape.push_back(target[dim]);
    }
    shape.push_back(cot.shape()[cot.ndim() - 2]);
    shape.push_back(cot.shape()[cot.ndim() - 1]);
    return unbroadcast(cot, shape);
}

template <typename dtype>
void matmul(const Tensor<dtype>& a, const Tensor<dtype>& b, Tensor<dtype>& c) {
    if (!a.is_tracked() && !b.is_tracked()) {
        return;
    }
    assert((!a.is_tracked() || !b.is_tracked() || a.trace() == b.trace()) &&
           "vjp::matmul: operands belong to different traces");

    Trace<dtype>* tr = a.is_tracked() ? a.trace() : b.trace();
    tr->add_node(c, [tr, a, b](size_t self_id) {
        if (a.is_tracked()) {
            tr->adjoints_[a.id()] += unbatch(
                matmul(tr->adjoints_[self_id], transpose_last_two(b.detach())), a.shape());
        }

        if (b.is_tracked()) {
            tr->adjoints_[b.id()] += unbatch(
                matmul(transpose_last_two(a.detach()), tr->adjoints_[self_id]), b.shape());
        }
    });
}

} // namespace vjp
} // namespace nanojax
