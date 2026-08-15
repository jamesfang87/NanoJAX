#include "ad/scalar.hpp"
#include "ad/trace.hpp"

#include <cassert>
#include <cmath>
#include <variant>

namespace {
template <typename F> Real real_op(const Real &lhs, const Real &rhs, F &&op) {
  return std::visit([&](auto a, auto b) -> Real { return op(a, b); }, lhs, rhs);
}

// Single-operand counterpart of real_op, used by the unary Real functions
// below (sin, cos, sqrt, ...).
template <typename F> Real real_unary_op(const Real &x, F &&op) {
  return std::visit([&](auto v) -> Real { return op(v); }, x);
}
} // namespace

Real value_of(const Scalar &s) {
  if (std::holds_alternative<Real>(s)) {
    return std::get<Real>(s);
  }
  return value_of(std::get<Variable>(s));
}

Real value_of(const Variable &x) { return value_of(x.trace->primals[x.id]); }

double to_double(const Real &r) {
  return std::visit([](auto v) { return static_cast<double>(v); }, r);
}

double to_double(const Scalar &s) { return to_double(value_of(s)); }

float to_float(const Real &r) {
  return std::visit([](auto v) { return static_cast<float>(v); }, r);
}

float to_float(const Scalar &s) { return to_float(value_of(s)); }

std::ostream &operator<<(std::ostream &os, const Real &r) {
  std::visit([&os](auto v) { os << v; }, r);
  return os;
}

std::ostream &operator<<(std::ostream &os, const Scalar &s) {
  return os << value_of(s);
}

std::ostream &operator<<(std::ostream &os, const Variable &v) {
  return os << value_of(v);
}

Scalar operator+(const Scalar &lhs, const Scalar &rhs) {
  if (std::holds_alternative<Real>(lhs) && std::holds_alternative<Real>(rhs)) {
    return real_op(std::get<Real>(lhs), std::get<Real>(rhs),
                   [](auto a, auto b) { return a + b; });
  }

  Trace &tr = trace_of(lhs, rhs);
  return Scalar{lift(lhs, tr) + lift(rhs, tr)};
}

Scalar operator-(const Scalar &lhs, const Scalar &rhs) {
  if (std::holds_alternative<Real>(lhs) && std::holds_alternative<Real>(rhs)) {
    return real_op(std::get<Real>(lhs), std::get<Real>(rhs),
                   [](auto a, auto b) { return a - b; });
  }

  Trace &tr = trace_of(lhs, rhs);
  return Scalar{lift(lhs, tr) - lift(rhs, tr)};
}

Scalar operator*(const Scalar &lhs, const Scalar &rhs) {
  if (std::holds_alternative<Real>(lhs) && std::holds_alternative<Real>(rhs)) {
    return real_op(std::get<Real>(lhs), std::get<Real>(rhs),
                   [](auto a, auto b) { return a * b; });
  }

  Trace &tr = trace_of(lhs, rhs);
  return Scalar{lift(lhs, tr) * lift(rhs, tr)};
}

Scalar operator/(const Scalar &lhs, const Scalar &rhs) {
  if (std::holds_alternative<Real>(lhs) && std::holds_alternative<Real>(rhs)) {
    return real_op(std::get<Real>(lhs), std::get<Real>(rhs),
                   [](auto a, auto b) { return a / b; });
  }

  Trace &tr = trace_of(lhs, rhs);
  return Scalar{lift(lhs, tr) / lift(rhs, tr)};
}

Scalar operator-(const Scalar &rhs) {
  if (std::holds_alternative<Real>(rhs)) {
    return real_unary_op(std::get<Real>(rhs), [](auto v) { return -v; });
  }
  return Scalar{-std::get<Variable>(rhs)};
}

std::partial_ordering operator<=>(const Scalar &lhs, const Scalar &rhs) {
  return to_double(lhs) <=> to_double(rhs);
}

bool operator==(const Scalar &lhs, const Scalar &rhs) {
  return to_double(lhs) == to_double(rhs);
}

Scalar sin(const Scalar &x) {
  if (std::holds_alternative<Real>(x)) {
    return real_unary_op(std::get<Real>(x), [](auto v) { return std::sin(v); });
  }
  return Scalar{sin(std::get<Variable>(x))};
}

Scalar cos(const Scalar &x) {
  if (std::holds_alternative<Real>(x)) {
    return real_unary_op(std::get<Real>(x), [](auto v) { return std::cos(v); });
  }
  return Scalar{cos(std::get<Variable>(x))};
}

Scalar tan(const Scalar &x) {
  if (std::holds_alternative<Real>(x)) {
    return real_unary_op(std::get<Real>(x), [](auto v) { return std::tan(v); });
  }
  return Scalar{tan(std::get<Variable>(x))};
}

Scalar csc(const Scalar &x) {
  if (std::holds_alternative<Real>(x)) {
    return real_unary_op(std::get<Real>(x),
                         [](auto v) { return 1 / std::sin(v); });
  }
  return Scalar{csc(std::get<Variable>(x))};
}

Scalar sec(const Scalar &x) {
  if (std::holds_alternative<Real>(x)) {
    return real_unary_op(std::get<Real>(x),
                         [](auto v) { return 1 / std::cos(v); });
  }
  return Scalar{sec(std::get<Variable>(x))};
}

Scalar cot(const Scalar &x) {
  if (std::holds_alternative<Real>(x)) {
    return real_unary_op(std::get<Real>(x),
                         [](auto v) { return 1 / std::tan(v); });
  }
  return Scalar{cot(std::get<Variable>(x))};
}

Scalar pow(const Scalar &base, const Scalar &exponent) {
  if (std::holds_alternative<Real>(base) &&
      std::holds_alternative<Real>(exponent)) {
    return real_op(std::get<Real>(base), std::get<Real>(exponent),
                   [](auto b, auto e) { return std::pow(b, e); });
  }

  Trace &tr = trace_of(base, exponent);
  return Scalar{pow(lift(base, tr), lift(exponent, tr))};
}

Scalar log(const Scalar &x) {
  if (std::holds_alternative<Real>(x)) {
    return real_unary_op(std::get<Real>(x), [](auto v) { return std::log(v); });
  }
  return Scalar{log(std::get<Variable>(x))};
}

Scalar exp(const Scalar &x) {
  if (std::holds_alternative<Real>(x)) {
    return real_unary_op(std::get<Real>(x), [](auto v) { return std::exp(v); });
  }
  return Scalar{exp(std::get<Variable>(x))};
}

Scalar abs(const Scalar &x) {
  if (std::holds_alternative<Real>(x)) {
    return real_unary_op(std::get<Real>(x), [](auto v) { return std::abs(v); });
  }
  return Scalar{abs(std::get<Variable>(x))};
}

Scalar sqrt(const Scalar &x) {
  if (std::holds_alternative<Real>(x)) {
    return real_unary_op(std::get<Real>(x),
                         [](auto v) { return std::sqrt(v); });
  }
  return Scalar{sqrt(std::get<Variable>(x))};
}

// Each Variable operator below builds the result node's primal from the
// operands' stored primals (rather than raw float/double), and each
// backward closure accumulates into the operands' adjoints via the Scalar
// operators above (rather than raw arithmetic). Both primals and adjoints
// are Scalar, not Real, precisely so that a nested trace (grad(grad(f)))
// can carry an outer Variable through this same code path.

Variable operator+(const Variable &lhs, const Variable &rhs) {
  assert(lhs.trace == rhs.trace &&
         "operator+: operands belong to different traces");
  Trace &tr = *lhs.trace;
  Scalar value = tr.primals[lhs.id] + tr.primals[rhs.id];
  return tr.add_variable(value, [lhs, rhs](size_t self_id) {
    Trace &tr = *lhs.trace;
    tr.adjoints[lhs.id] = tr.adjoints[lhs.id] + tr.adjoints[self_id];
    tr.adjoints[rhs.id] = tr.adjoints[rhs.id] + tr.adjoints[self_id];
  });
}

Variable operator-(const Variable &lhs, const Variable &rhs) {
  assert(lhs.trace == rhs.trace &&
         "operator-: operands belong to different traces");
  Trace &tr = *lhs.trace;
  Scalar value = tr.primals[lhs.id] - tr.primals[rhs.id];
  return tr.add_variable(value, [lhs, rhs](size_t self_id) {
    Trace &tr = *lhs.trace;
    tr.adjoints[lhs.id] = tr.adjoints[lhs.id] + tr.adjoints[self_id];
    tr.adjoints[rhs.id] = tr.adjoints[rhs.id] - tr.adjoints[self_id];
  });
}

Variable operator*(const Variable &lhs, const Variable &rhs) {
  assert(lhs.trace == rhs.trace &&
         "operator*: operands belong to different traces");
  Trace &tr = *lhs.trace;
  Scalar value = tr.primals[lhs.id] * tr.primals[rhs.id];
  return tr.add_variable(value, [lhs, rhs](size_t self_id) {
    Trace &tr = *lhs.trace;
    tr.adjoints[lhs.id] =
        tr.adjoints[lhs.id] + tr.adjoints[self_id] * tr.primals[rhs.id];
    tr.adjoints[rhs.id] =
        tr.adjoints[rhs.id] + tr.adjoints[self_id] * tr.primals[lhs.id];
  });
}

Variable operator/(const Variable &lhs, const Variable &rhs) {
  assert(lhs.trace == rhs.trace &&
         "operator/: operands belong to different traces");
  Trace &tr = *lhs.trace;
  Scalar value = tr.primals[lhs.id] / tr.primals[rhs.id];
  return tr.add_variable(value, [lhs, rhs](size_t self_id) {
    Trace &tr = *lhs.trace;
    const Scalar &rval = tr.primals[rhs.id];
    tr.adjoints[lhs.id] = tr.adjoints[lhs.id] + tr.adjoints[self_id] / rval;
    tr.adjoints[rhs.id] = tr.adjoints[rhs.id] - tr.adjoints[self_id] *
                                                    tr.primals[lhs.id] /
                                                    (rval * rval);
  });
}

Variable operator-(const Variable &rhs) {
  Trace &tr = *rhs.trace;
  Scalar value = -tr.primals[rhs.id];
  return tr.add_variable(value, [rhs](size_t self_id) {
    Trace &tr = *rhs.trace;
    tr.adjoints[rhs.id] = tr.adjoints[rhs.id] - tr.adjoints[self_id];
  });
}

std::partial_ordering operator<=>(const Variable &lhs, const Variable &rhs) {
  return to_double(value_of(lhs)) <=> to_double(value_of(rhs));
}

bool operator==(const Variable &lhs, const Variable &rhs) {
  return to_double(value_of(lhs)) == to_double(value_of(rhs));
}

Variable sin(const Variable &x) {
  Trace &tr = *x.trace;
  Scalar value = sin(tr.primals[x.id]);
  return tr.add_variable(value, [x](size_t self_id) {
    Trace &tr = *x.trace;
    tr.adjoints[x.id] =
        tr.adjoints[x.id] + tr.adjoints[self_id] * cos(tr.primals[x.id]);
  });
}

Variable cos(const Variable &x) {
  Trace &tr = *x.trace;
  Scalar value = cos(tr.primals[x.id]);
  return tr.add_variable(value, [x](size_t self_id) {
    Trace &tr = *x.trace;
    tr.adjoints[x.id] =
        tr.adjoints[x.id] - tr.adjoints[self_id] * sin(tr.primals[x.id]);
  });
}

Variable tan(const Variable &x) {
  Trace &tr = *x.trace;
  Scalar value = tan(tr.primals[x.id]);
  return tr.add_variable(value, [x](size_t self_id) {
    Trace &tr = *x.trace;
    const Scalar &y = tr.primals[self_id];
    tr.adjoints[x.id] =
        tr.adjoints[x.id] + tr.adjoints[self_id] * (Scalar{1.0} + y * y);
  });
}

Variable csc(const Variable &x) {
  Trace &tr = *x.trace;
  Scalar value = csc(tr.primals[x.id]);
  return tr.add_variable(value, [x](size_t self_id) {
    Trace &tr = *x.trace;
    const Scalar &y = tr.primals[self_id];
    tr.adjoints[x.id] =
        tr.adjoints[x.id] - tr.adjoints[self_id] * y * cot(tr.primals[x.id]);
  });
}

Variable sec(const Variable &x) {
  Trace &tr = *x.trace;
  Scalar value = sec(tr.primals[x.id]);
  return tr.add_variable(value, [x](size_t self_id) {
    Trace &tr = *x.trace;
    const Scalar &y = tr.primals[self_id];
    tr.adjoints[x.id] =
        tr.adjoints[x.id] + tr.adjoints[self_id] * y * tan(tr.primals[x.id]);
  });
}

Variable cot(const Variable &x) {
  Trace &tr = *x.trace;
  Scalar value = cot(tr.primals[x.id]);
  return tr.add_variable(value, [x](size_t self_id) {
    Trace &tr = *x.trace;
    const Scalar &y = tr.primals[self_id];
    tr.adjoints[x.id] =
        tr.adjoints[x.id] - tr.adjoints[self_id] * (Scalar{1.0} + y * y);
  });
}

Variable pow(const Variable &base, const Variable &exponent) {
  assert(base.trace == exponent.trace &&
         "pow: operands belong to different traces");
  Trace &tr = *base.trace;
  Scalar value = pow(tr.primals[base.id], tr.primals[exponent.id]);
  return tr.add_variable(value, [base, exponent](size_t self_id) {
    Trace &tr = *base.trace;
    const Scalar &base_val = tr.primals[base.id];
    const Scalar &exp_val = tr.primals[exponent.id];
    const Scalar &y = tr.primals[self_id];
    tr.adjoints[base.id] =
        tr.adjoints[base.id] +
        tr.adjoints[self_id] * exp_val * pow(base_val, exp_val - Scalar{1.0});
    tr.adjoints[exponent.id] =
        tr.adjoints[exponent.id] + tr.adjoints[self_id] * y * log(base_val);
  });
}

Variable log(const Variable &x) {
  Trace &tr = *x.trace;
  Scalar value = log(tr.primals[x.id]);
  return tr.add_variable(value, [x](size_t self_id) {
    Trace &tr = *x.trace;
    tr.adjoints[x.id] =
        tr.adjoints[x.id] + tr.adjoints[self_id] / tr.primals[x.id];
  });
}

Variable exp(const Variable &x) {
  Trace &tr = *x.trace;
  Scalar value = exp(tr.primals[x.id]);
  return tr.add_variable(value, [x](size_t self_id) {
    Trace &tr = *x.trace;
    tr.adjoints[x.id] =
        tr.adjoints[x.id] + tr.adjoints[self_id] * tr.primals[self_id];
  });
}

Variable abs(const Variable &x) {
  Trace &tr = *x.trace;
  Scalar value = abs(tr.primals[x.id]);
  return tr.add_variable(value, [x](size_t self_id) {
    Trace &tr = *x.trace;
    double v = to_double(tr.primals[x.id]);
    double sign = v > 0.0 ? 1.0 : (v < 0.0 ? -1.0 : 0.0);
    tr.adjoints[x.id] = tr.adjoints[x.id] + tr.adjoints[self_id] * Scalar{sign};
  });
}

Variable sqrt(const Variable &x) {
  Trace &tr = *x.trace;
  Scalar value = sqrt(tr.primals[x.id]);
  return tr.add_variable(value, [x](size_t self_id) {
    Trace &tr = *x.trace;
    const Scalar &y = tr.primals[self_id];
    tr.adjoints[x.id] =
        tr.adjoints[x.id] + tr.adjoints[self_id] / (Scalar{2.0} * y);
  });
}
