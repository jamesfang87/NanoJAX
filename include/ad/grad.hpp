#include "ad/scalar.hpp"
#include "ad/trace.hpp"

#include <cstddef>
#include <memory>
#include <tuple>
#include <type_traits>

namespace {
template <typename Out>
Variable ensure_scalar_out(Out &&out, Trace &tr, size_t argnum) {
  using T = std::decay_t<Out>;

  if constexpr (std::is_same_v<T, Variable>) {
    return out;
  } else if constexpr (std::is_same_v<T, Real> || std::is_same_v<T, Scalar>) {
    return lift(out, tr);
  } else if constexpr (std::is_same_v<T, Vector>) {
    assert(argnum < out.size() &&
           "grad: argnums is out of range for f's return vector");
    return ensure_scalar_out(out[argnum], tr, argnum);
  } else {
    static_assert(
        false, "grad() requires f to return a scalar value. this means that a "
               "multi-variable f cannot be directly differentiated again");
  }
}

} // namespace

template <typename F> auto grad(F &&f, size_t argnums = 0) {
  return [f = std::forward<F>(f), argnums,
          last_node_count = size_t{0}](auto... inputs) mutable {
    trace_stack.push(std::make_unique<Trace>());
    Trace &tr = trace();
    if (last_node_count > 0) {
      tr.reserve(last_node_count);
    }

    auto x = std::make_tuple(tr.add_variable(Scalar{inputs}, nullptr)...);
    auto out = std::apply(f, x);

    tr.backward(ensure_scalar_out(out, tr, argnums));
    last_node_count = tr.primals.size();
    if constexpr (sizeof...(inputs) == 1) {
      Scalar g =
          std::apply([&](const auto &x) { return tr.adjoints[x.id]; }, x);
      trace_stack.pop();
      return g;
    } else {
      auto grads = std::apply(
          [&](const auto &...v) { return Vector{tr.adjoints[v.id]...}; }, x);
      trace_stack.pop();
      return grads;
    }
  };
}
