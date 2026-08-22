#include "ad/scalar.hpp"
#include "ad/trace.hpp"

#include <cstddef>
#include <memory>
#include <tuple>
#include <type_traits>

namespace {

template <typename T> struct is_tuple : std::false_type {};
template <typename... Ts>
struct is_tuple<std::tuple<Ts...>> : std::true_type {};
template <typename T>
inline constexpr bool is_tuple_v = is_tuple<std::decay_t<T>>::value;

template <typename Out>
Variable ensure_scalar_out(Out &&out, Trace &tr, size_t argnum);

template <typename Tuple, size_t I = 0>
Variable tuple_ensure_scalar_out(Tuple &tup, Trace &tr, size_t argnum) {
  if constexpr (I < std::tuple_size_v<std::decay_t<Tuple>>) {
    if (argnum == I) {
      return ensure_scalar_out(std::get<I>(tup), tr, argnum);
    }
    return tuple_ensure_scalar_out<Tuple, I + 1>(tup, tr, argnum);
  } else {
    assert(false && "grad: argnums is out of range for f's returned tuple");
    return Variable{};
  }
}

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
  } else if constexpr (is_tuple_v<T>) {
    return tuple_ensure_scalar_out(out, tr, argnum);
  } else {
    static_assert(
        false, "grad() requires f to return a scalar value, a Vector, or a "
               "tuple ensure_scalar_out can index into; a shape like "
               "MLP::Params cannot be the thing grad() differentiates again");
  }
}

template <typename T, typename F> auto map_leaves(const T &value, const F &fn) {
  if constexpr (std::is_constructible_v<Scalar, T>) {
    return fn(Scalar{value});
  } else {
    return map_scalars(value, fn);
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

    auto lift_one_input = [&tr](const auto &input) {
      return map_leaves(input, [&tr](const Scalar &s) {
        return Scalar{tr.add_variable(s, nullptr)};
      });
    };
    auto x = std::make_tuple(lift_one_input(inputs)...);

    auto out = std::apply(f, x);
    tr.backward(ensure_scalar_out(out, tr, argnums));
    last_node_count = tr.primals.size();

    auto read_one_output = [&tr](const auto &traced) {
      return map_leaves(traced, [&tr](const Scalar &s) {
        return tr.adjoints[std::get<Variable>(s).id];
      });
    };

    if constexpr (sizeof...(inputs) == 1) {
      auto g = std::apply(
          [&](const auto &...v) { return (read_one_output(v), ...); }, x);
      trace_stack.pop();
      return g;
    } else {
      auto grads = std::apply(
          [&](const auto &...v) {
            return std::make_tuple(read_one_output(v)...);
          },
          x);
      trace_stack.pop();
      return grads;
    }
  };
}
