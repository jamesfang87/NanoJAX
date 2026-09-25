#pragma once

#include <cassert>
#include <cstddef>
#include <utility>

#include "grad/trace.hpp"
#include "tensor/tensor.hpp"

namespace nanojax {
namespace vjp {

template <typename dtype>
Tensor<dtype> unbroadcast(const Tensor<dtype>& g, std::span<const size_t> target) {
    Tensor<dtype> out = g;

    for (size_t i = out.ndim(); i > target.size(); --i) {
        out = sum(out, 0);
    }

    for (size_t d = 0; d < target.size(); ++d) {
        if (target[d] == 1 && out.shape()[d] != 1) {
            out = sum(out, d).unsqueeze(d);
        }
    }

    return out;
}

template <typename dtype> Tensor<dtype> track(const Tensor<dtype>& x) {
    Tensor<dtype> tagged = x;
    for (Trace<dtype>* t : Trace<dtype>::stack()) {
        t->attach(tagged);
    }
    return tagged;
}

template <typename dtype, typename Fn> void record(Tensor<dtype>& result, Fn fn) {
    for (Trace<dtype>* t : Trace<dtype>::stack()) {
        t->attach(result);
        t->backwards(result) = fn;
    }
}

template <typename dtype>
void add(const Tensor<dtype>& a, const Tensor<dtype>& b, Tensor<dtype>& c) {
    if (!a.is_tracked() && !b.is_tracked()) {
        return;
    }

    Tensor<dtype> a2 = track(a);
    Tensor<dtype> b2 = track(b);
    record(c, [a2, b2](Trace<dtype>* tr, size_t self_id) {
        Tensor<dtype> tmp1 = track(tr->adjoint(self_id));
        Tensor<dtype> tmp2 = track(tr->adjoint(a2));
        tr->adjoint(a2) = tmp2 + unbroadcast(tmp1, a2.shape());
        tmp2 = track(tr->adjoint(b2));
        tr->adjoint(b2) = tmp2 + unbroadcast(tmp1, b2.shape());
    });
}

template <typename dtype>
void sub(const Tensor<dtype>& a, const Tensor<dtype>& b, Tensor<dtype>& c) {
    if (!a.is_tracked() && !b.is_tracked()) {
        return;
    }

    Tensor<dtype> a2 = track(a);
    Tensor<dtype> b2 = track(b);
    record(c, [a2, b2](Trace<dtype>* tr, size_t self_id) {
        Tensor<dtype> tmp1 = track(tr->adjoint(self_id));
        Tensor<dtype> tmp2 = track(tr->adjoint(a2));
        tr->adjoint(a2) = tmp2 + unbroadcast(tmp1, a2.shape());
        tmp2 = track(tr->adjoint(b2));
        tr->adjoint(b2) = tmp2 - unbroadcast(tmp1, b2.shape());
    });
}

template <typename dtype>
void mul(const Tensor<dtype>& a, const Tensor<dtype>& b, Tensor<dtype>& c) {
    if (!a.is_tracked() && !b.is_tracked()) {
        return;
    }

    Tensor<dtype> a2 = track(a);
    Tensor<dtype> b2 = track(b);
    record(c, [a2, b2](Trace<dtype>* tr, size_t self_id) {
        Tensor<dtype> tmp1 = track(tr->adjoint(self_id));
        Tensor<dtype> tmp2 = track(tr->adjoint(a2));
        tr->adjoint(a2) = tmp2 + unbroadcast(tmp1 * b2, a2.shape());
        tmp2 = track(tr->adjoint(b2));
        tr->adjoint(b2) = tmp2 + unbroadcast(tmp1 * a2, b2.shape());
    });
}

template <typename dtype>
void div(const Tensor<dtype>& a, const Tensor<dtype>& b, Tensor<dtype>& c) {
    if (!a.is_tracked() && !b.is_tracked()) {
        return;
    }

    Tensor<dtype> a2 = track(a);
    Tensor<dtype> b2 = track(b);
    record(c, [a2, b2](Trace<dtype>* tr, size_t self_id) {
        Tensor<dtype> tmp1 = track(tr->adjoint(self_id));
        Tensor<dtype> tmp2 = track(tr->adjoint(a2));
        tr->adjoint(a2) = tmp2 + unbroadcast(tmp1 / b2, a2.shape());
        tmp2 = track(tr->adjoint(b2));
        tr->adjoint(b2) = tmp2 - unbroadcast(tmp1 * a2 / (b2 * b2), b2.shape());
    });
}

template <typename dtype> void neg(const Tensor<dtype>& a, Tensor<dtype>& b) {
    if (!a.is_tracked()) {
        return;
    }

    Tensor<dtype> a2 = track(a);
    record(b, [a2](Trace<dtype>* tr, size_t self_id) {
        Tensor<dtype> tmp1 = track(tr->adjoint(self_id));
        Tensor<dtype> tmp2 = track(tr->adjoint(a2));
        tr->adjoint(a2) = tmp2 - tmp1;
    });
}

template <typename dtype> void scalar_add(const Tensor<dtype>& a, Tensor<dtype>& b) {
    if (!a.is_tracked()) {
        return;
    }

    Tensor<dtype> a2 = track(a);
    record(b, [a2](Trace<dtype>* tr, size_t self_id) {
        Tensor<dtype> tmp1 = track(tr->adjoint(self_id));
        Tensor<dtype> tmp2 = track(tr->adjoint(a2));
        tr->adjoint(a2) = tmp2 + tmp1;
    });
}

template <typename dtype> void lhs_scalar_sub(const Tensor<dtype>& a, Tensor<dtype>& b) {
    if (!a.is_tracked()) {
        return;
    }

    Tensor<dtype> a2 = track(a);
    record(b, [a2](Trace<dtype>* tr, size_t self_id) {
        Tensor<dtype> tmp1 = track(tr->adjoint(self_id));
        Tensor<dtype> tmp2 = track(tr->adjoint(a2));
        tr->adjoint(a2) = tmp2 + tmp1;
    });
}

template <typename dtype> void rhs_scalar_sub(const Tensor<dtype>& a, Tensor<dtype>& b) {
    if (!a.is_tracked()) {
        return;
    }

    Tensor<dtype> a2 = track(a);
    record(b, [a2](Trace<dtype>* tr, size_t self_id) {
        Tensor<dtype> tmp1 = track(tr->adjoint(self_id));
        Tensor<dtype> tmp2 = track(tr->adjoint(a2));
        tr->adjoint(a2) = tmp2 - tmp1;
    });
}

template <typename dtype> void scalar_mul(const Tensor<dtype>& a, dtype b, Tensor<dtype>& c) {
    if (!a.is_tracked()) {
        return;
    }

    Tensor<dtype> a2 = track(a);
    record(c, [a2, b](Trace<dtype>* tr, size_t self_id) {
        Tensor<dtype> tmp1 = track(tr->adjoint(self_id));
        Tensor<dtype> tmp2 = track(tr->adjoint(a2));
        tr->adjoint(a2) = tmp2 + tmp1 * b;
    });
}

template <typename dtype> void lhs_scalar_div(const Tensor<dtype>& a, dtype b, Tensor<dtype>& c) {
    if (!a.is_tracked()) {
        return;
    }

    Tensor<dtype> a2 = track(a);
    record(c, [a2, b](Trace<dtype>* tr, size_t self_id) {
        Tensor<dtype> tmp1 = track(tr->adjoint(self_id));
        Tensor<dtype> tmp2 = track(tr->adjoint(a2));
        tr->adjoint(a2) = tmp2 + tmp1 / b;
    });
}

template <typename dtype> void rhs_scalar_div(dtype a, const Tensor<dtype>& b, Tensor<dtype>& c) {
    if (!b.is_tracked()) {
        return;
    }

    Tensor<dtype> b2 = track(b);
    record(c, [b2, a](Trace<dtype>* tr, size_t self_id) {
        Tensor<dtype> tmp1 = track(tr->adjoint(self_id));
        Tensor<dtype> tmp2 = track(tr->adjoint(b2));
        tr->adjoint(b2) = tmp2 - tmp1 * a / (b2 * b2);
    });
}

template <typename dtype> void exp(const Tensor<dtype>& a, Tensor<dtype>& c) {
    if (!a.is_tracked()) {
        return;
    }

    Tensor<dtype> a2 = track(a);
    record(c, [a2](Trace<dtype>* tr, size_t self_id) {
        Tensor<dtype> tmp1 = track(tr->adjoint(self_id));
        Tensor<dtype> tmp2 = track(tr->adjoint(a2));
        tr->adjoint(a2) = tmp2 + tmp1 * exp(a2);
    });
}

template <typename dtype> void log(const Tensor<dtype>& a, Tensor<dtype>& c) {
    if (!a.is_tracked()) {
        return;
    }

    Tensor<dtype> a2 = track(a);
    record(c, [a2](Trace<dtype>* tr, size_t self_id) {
        Tensor<dtype> tmp1 = track(tr->adjoint(self_id));
        Tensor<dtype> tmp2 = track(tr->adjoint(a2));
        tr->adjoint(a2) = tmp2 + tmp1 / a2;
    });
}

template <typename dtype> void sqrt(const Tensor<dtype>& a, Tensor<dtype>& c) {
    if (!a.is_tracked()) {
        return;
    }

    Tensor<dtype> a2 = track(a);
    record(c, [a2](Trace<dtype>* tr, size_t self_id) {
        Tensor<dtype> tmp1 = track(tr->adjoint(self_id));
        Tensor<dtype> tmp2 = track(tr->adjoint(a2));
        tr->adjoint(a2) = tmp2 + tmp1 / (sqrt(a2) * dtype{2});
    });
}

template <typename dtype> void pow(const Tensor<dtype>& a, dtype exponent, Tensor<dtype>& c) {
    if (!a.is_tracked()) {
        return;
    }

    Tensor<dtype> a2 = track(a);
    record(c, [a2, exponent](Trace<dtype>* tr, size_t self_id) {
        Tensor<dtype> tmp1 = track(tr->adjoint(self_id));
        Tensor<dtype> tmp2 = track(tr->adjoint(a2));
        tr->adjoint(a2) = tmp2 + (tmp1 * exponent) * pow(a2, exponent - dtype{1});
    });
}

template <typename dtype> void abs(const Tensor<dtype>& a, Tensor<dtype>& c) {
    if (!a.is_tracked()) {
        return;
    }

    Tensor<dtype> a2 = track(a);
    record(c, [a2](Trace<dtype>* tr, size_t self_id) {
        Tensor<dtype> tmp1 = track(tr->adjoint(self_id));
        Tensor<dtype> tmp2 = track(tr->adjoint(a2));
        auto sign = a2.detach().map([](dtype element) {
            return element > dtype{0} ? dtype{1} : (element < dtype{0} ? dtype{-1} : dtype{0});
        });
        tr->adjoint(a2) = tmp2 + tmp1 * sign;
    });
}

template <typename dtype> void relu(const Tensor<dtype>& a, Tensor<dtype>& c) {
    if (!a.is_tracked()) {
        return;
    }

    Tensor<dtype> a2 = track(a);
    record(c, [a2](Trace<dtype>* tr, size_t self_id) {
        Tensor<dtype> tmp1 = track(tr->adjoint(self_id));
        Tensor<dtype> tmp2 = track(tr->adjoint(a2));
        auto mask =
            a2.detach().map([](dtype element) { return element > dtype{0} ? dtype{1} : dtype{0}; });
        tr->adjoint(a2) = tmp2 + tmp1 * mask;
    });
}

template <typename dtype> void sigmoid(const Tensor<dtype>& a, Tensor<dtype>& c) {
    if (!a.is_tracked()) {
        return;
    }

    Tensor<dtype> a2 = track(a);
    record(c, [a2](Trace<dtype>* tr, size_t self_id) {
        Tensor<dtype> tmp1 = track(tr->adjoint(self_id));
        Tensor<dtype> tmp2 = track(tr->adjoint(a2));
        Tensor<dtype> y = sigmoid(a2);
        tr->adjoint(a2) = tmp2 + tmp1 * y * (dtype{1} - y);
    });
}

template <typename dtype> void tanh(const Tensor<dtype>& a, Tensor<dtype>& c) {
    if (!a.is_tracked()) {
        return;
    }

    Tensor<dtype> a2 = track(a);
    record(c, [a2](Trace<dtype>* tr, size_t self_id) {
        Tensor<dtype> tmp1 = track(tr->adjoint(self_id));
        Tensor<dtype> tmp2 = track(tr->adjoint(a2));
        Tensor<dtype> y = tanh(a2);
        tr->adjoint(a2) = tmp2 + tmp1 * (dtype{1} - y * y);
    });
}

template <typename dtype> void sum(const Tensor<dtype>& a, Tensor<dtype>& b) {
    if (!a.is_tracked()) {
        return;
    }

    Tensor<dtype> a2 = track(a);
    record(b, [a2](Trace<dtype>* tr, size_t self_id) {
        Tensor<dtype> tmp1 = track(tr->adjoint(self_id));
        Tensor<dtype> tmp2 = track(tr->adjoint(a2));
        tr->adjoint(a2) = tmp2 + tmp1.broadcast(a2.shape());
    });
}

template <typename dtype> void sum(const Tensor<dtype>& a, size_t axis, Tensor<dtype>& b) {
    if (!a.is_tracked()) {
        return;
    }

    Tensor<dtype> a2 = track(a);
    record(b, [a2, axis](Trace<dtype>* tr, size_t self_id) {
        Tensor<dtype> tmp1 = track(tr->adjoint(self_id));
        Tensor<dtype> tmp2 = track(tr->adjoint(a2));
        tr->adjoint(a2) = tmp2 + tmp1.unsqueeze(axis).broadcast(a2.shape());
    });
}

template <typename dtype> void max(const Tensor<dtype>& a, size_t flat_idx, Tensor<dtype>& b) {
    if (!a.is_tracked()) {
        return;
    }

    Tensor<dtype> a2 = track(a);
    record(b, [a2, flat_idx](Trace<dtype>* tr, size_t self_id) {
        Tensor<dtype> tmp1 = track(tr->adjoint(self_id));
        Tensor<dtype> tmp2 = track(tr->adjoint(a2));
        Tensor<dtype> onehot = Tensor<dtype>::zeros_like(a2);
        onehot.data()[flat_idx] = dtype{1};
        tr->adjoint(a2) = tmp2 + tmp1 * onehot;
    });
}

template <typename dtype> void min(const Tensor<dtype>& a, size_t flat_idx, Tensor<dtype>& b) {
    if (!a.is_tracked()) {
        return;
    }

    Tensor<dtype> a2 = track(a);
    record(b, [a2, flat_idx](Trace<dtype>* tr, size_t self_id) {
        Tensor<dtype> tmp1 = track(tr->adjoint(self_id));
        Tensor<dtype> tmp2 = track(tr->adjoint(a2));
        Tensor<dtype> onehot = Tensor<dtype>::zeros_like(a2);
        onehot.data()[flat_idx] = dtype{1};
        tr->adjoint(a2) = tmp2 + tmp1 * onehot;
    });
}

template <typename dtype> void reshape(const Tensor<dtype>& a, Tensor<dtype>& result) {
    if (!a.is_tracked()) {
        return;
    }

    Tensor<dtype> a2 = track(a);
    record(result, [a2](Trace<dtype>* tr, size_t self_id) {
        Tensor<dtype> tmp1 = track(tr->adjoint(self_id));
        Tensor<dtype> tmp2 = track(tr->adjoint(a2));
        tr->adjoint(a2) = tmp2 + tmp1.reshape(a2.shape());
    });
}

template <typename dtype> void transpose(const Tensor<dtype>& a, Tensor<dtype>& result) {
    if (!a.is_tracked()) {
        return;
    }

    Tensor<dtype> a2 = track(a);
    record(result, [a2](Trace<dtype>* tr, size_t self_id) {
        Tensor<dtype> tmp1 = track(tr->adjoint(self_id));
        Tensor<dtype> tmp2 = track(tr->adjoint(a2));
        tr->adjoint(a2) = tmp2 + tmp1.transpose();
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

    Tensor<dtype> a2 = track(a);
    record(result, [a2, inverse](Trace<dtype>* tr, size_t self_id) {
        Tensor<dtype> tmp1 = track(tr->adjoint(self_id));
        Tensor<dtype> tmp2 = track(tr->adjoint(a2));
        tr->adjoint(a2) = tmp2 + tmp1.permute(std::span<const size_t>(inverse));
    });
}

template <typename dtype> void squeeze(const Tensor<dtype>& a, Tensor<dtype>& result) {
    if (!a.is_tracked()) {
        return;
    }

    Tensor<dtype> a2 = track(a);
    record(result, [a2](Trace<dtype>* tr, size_t self_id) {
        Tensor<dtype> tmp1 = track(tr->adjoint(self_id));
        Tensor<dtype> tmp2 = track(tr->adjoint(a2));
        tr->adjoint(a2) = tmp2 + tmp1.reshape(a2.shape());
    });
}

template <typename dtype> void unsqueeze(const Tensor<dtype>& a, Tensor<dtype>& result) {
    if (!a.is_tracked()) {
        return;
    }

    Tensor<dtype> a2 = track(a);
    record(result, [a2](Trace<dtype>* tr, size_t self_id) {
        Tensor<dtype> tmp1 = track(tr->adjoint(self_id));
        Tensor<dtype> tmp2 = track(tr->adjoint(a2));
        tr->adjoint(a2) = tmp2 + tmp1.reshape(a2.shape());
    });
}

template <typename dtype> void broadcast(const Tensor<dtype>& a, Tensor<dtype>& result) {
    if (!a.is_tracked()) {
        return;
    }

    Tensor<dtype> a2 = track(a);
    record(result, [a2](Trace<dtype>* tr, size_t self_id) {
        Tensor<dtype> tmp1 = track(tr->adjoint(self_id));
        Tensor<dtype> tmp2 = track(tr->adjoint(a2));
        tr->adjoint(a2) = tmp2 + vjp::unbroadcast(tmp1, a2.shape());
    });
}

template <typename dtype> Tensor<dtype> transpose_last_two(const Tensor<dtype>& tensor) {
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

    Tensor<dtype> a2 = track(a);
    Tensor<dtype> b2 = track(b);
    record(c, [a2, b2](Trace<dtype>* tr, size_t self_id) {
        Tensor<dtype> tmp1 = track(tr->adjoint(self_id));
        Tensor<dtype> tmp2 = track(tr->adjoint(a2));
        tr->adjoint(a2) = tmp2 + unbatch(matmul(tmp1, transpose_last_two(b2.detach())), a2.shape());
        tmp2 = track(tr->adjoint(b2));
        tr->adjoint(b2) = tmp2 + unbatch(matmul(transpose_last_two(a2.detach()), tmp1), b2.shape());
    });
}

} // namespace vjp
} // namespace nanojax
