#include "include/ad/scalar.hpp"
#include "include/ad/trace.hpp"

#include <memory>
#include <tuple>
#include <type_traits>

namespace {
template <typename Out> Variable ensure_scalar_out(Out &&out, Trace &tr) {
  using T = std::decay_t<Out>;

  if constexpr (std::is_same_v<T, Variable>) {
    return out;
  } else if constexpr (std::is_same_v<T, Real> || std::is_same_v<T, Scalar>) {
    return lift(out, tr);
  } else {
    static_assert(
        false, "grad() requires f to return a scalar value. this means that a "
               "multi-variable f cannot be directly differentiated again");
  }
}

} // namespace

template <typename F> auto grad(F &&f) {
  return [f = std::forward<F>(f)](auto... inputs) {
    trace_stack.push(std::make_unique<Trace>());
    Trace &tr = trace();

    auto x = std::make_tuple(tr.add_variable(Scalar{inputs}, nullptr)...);
    auto out = std::apply(f, x);

    // To allow for grad(grad(f)), f must be a single-variable function
    // and output a scalar value
    tr.backward(ensure_scalar_out(out, tr));
    if constexpr (sizeof...(inputs) == 1) {
      // this is so ensure_scalar_out works, since it wouldn't work if
      // std::vector<Scalar> was returned
      Scalar g =
          std::apply([&](const auto &x) { return tr.adjoints[x.id]; }, x);
      trace_stack.pop();
      return g;
    } else {
      auto grads = std::apply(
          [&](const auto &...v) {
            return std::vector<Scalar>{tr.adjoints[v.id]...};
          },
          x);
      trace_stack.pop();
      return grads;
    }
  };
}
