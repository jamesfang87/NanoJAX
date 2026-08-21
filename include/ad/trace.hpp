#pragma once

#include "ad/scalar.hpp"

#include <cassert>
#include <cstddef>
#include <functional>
#include <memory>
#include <stack>
#include <unordered_map>
#include <vector>

class Trace {
public:
  Vector primals;
  Vector adjoints;
  std::vector<std::function<void(size_t)>> backwards;
  std::unordered_map<Real, Variable> constant_cache;

  // Pre-sizes primals, adjoints, and backwards to n entries
  // This is done by grad to avoid expensive heap reallocations
  void reserve(size_t n) {
    primals.reserve(n);
    adjoints.reserve(n);
    backwards.reserve(n);
  }

  Variable add_variable(Scalar value, std::function<void(size_t)> backwards) {
    auto id = primals.size();
    this->primals.push_back(value);
    this->adjoints.push_back(0.0);
    this->backwards.push_back(std::move(backwards));
    return Variable{this, id};
  }

  Variable add_const(Real value) {
    auto it = constant_cache.find(value);
    if (it != constant_cache.end()) {
      return it->second;
    }
    Variable node = add_variable(value, nullptr);
    constant_cache.emplace(value, node);
    return node;
  }

  void backward(Variable out) {
    assert(out.trace == this);
    adjoints[out.id] = 1.0;

    for (size_t i = primals.size(); i-- > 0;) {
      if (backwards[i]) {
        backwards[i](i);
      }
    }
  }
};

inline std::stack<std::unique_ptr<Trace>> trace_stack;
inline Trace &trace() {
  assert(!trace_stack.empty() && "trace() called outside of grad()");
  return *trace_stack.top();
}

namespace detail {
inline void scan_trace(Trace *&result, const Variable &v) {
  assert((result == nullptr || result == v.trace) &&
         "trace_of: operands belong to different traces");
  result = v.trace;
}
inline void scan_trace(Trace *&, const Real &) {}
inline void scan_trace(Trace *&result, const Scalar &s) {
  if (const Variable *v = std::get_if<Variable>(&s)) {
    scan_trace(result, *v);
  }
}
} // namespace detail

template <typename... Args> Trace &trace_of(const Args &...args) {
  Trace *result = nullptr;
  (detail::scan_trace(result, args), ...);
  assert(result != nullptr && "trace_of: called with no Variable operand");
  return *result;
}

inline Variable lift(const Scalar &s, Trace &t) {
  if (const Variable *v = std::get_if<Variable>(&s)) {
    return *v;
  }
  return t.add_const(std::get<Real>(s));
}
