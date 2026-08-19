#include "ad/scalar.hpp"
#include "ad/trace.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <variant>

namespace {
template <typename F> Real real_op(const Real &lhs, const Real &rhs, F &&op) {
  if (const double *a = std::get_if<double>(&lhs)) {
    if (const double *b = std::get_if<double>(&rhs)) {
      return Real{op(*a, *b)};
    }
    return Real{op(*a, *std::get_if<float>(&rhs))};
  }
  const float *a = std::get_if<float>(&lhs);
  if (const double *b = std::get_if<double>(&rhs)) {
    return Real{op(*a, *b)};
  }
  return Real{op(*a, *std::get_if<float>(&rhs))};
}

template <typename F> Real real_unary_op(const Real &x, F &&op) {
  if (const double *v = std::get_if<double>(&x)) {
    return Real{op(*v)};
  }
  return Real{op(*std::get_if<float>(&x))};
}
} // namespace

Real value_of(const Scalar &s) {
  if (const Real *r = std::get_if<Real>(&s)) {
    return *r;
  }
  return value_of(std::get<Variable>(s));
}

Real value_of(const Variable &x) { return value_of(x.trace->primals[x.id]); }

double to_double(const Real &r) {
  if (const double *d = std::get_if<double>(&r)) {
    return *d;
  }
  return static_cast<double>(*std::get_if<float>(&r));
}

double to_double(const Scalar &s) { return to_double(value_of(s)); }

float to_float(const Real &r) {
  if (const float *f = std::get_if<float>(&r)) {
    return *f;
  }
  return static_cast<float>(*std::get_if<double>(&r));
}

float to_float(const Scalar &s) { return to_float(value_of(s)); }

std::ostream &operator<<(std::ostream &os, const Real &r) {
  if (const double *d = std::get_if<double>(&r)) {
    return os << *d;
  }
  return os << *std::get_if<float>(&r);
}

std::ostream &operator<<(std::ostream &os, const Scalar &s) {
  return os << value_of(s);
}

std::ostream &operator<<(std::ostream &os, const Variable &v) {
  return os << value_of(v);
}

Scalar operator+(const Scalar &lhs, const Scalar &rhs) {
  if (const Real *l = std::get_if<Real>(&lhs)) {
    if (const Real *r = std::get_if<Real>(&rhs)) {
      return real_op(*l, *r, [](auto a, auto b) { return a + b; });
    }
  }

  Trace &tr = trace_of(lhs, rhs);
  return Scalar{lift(lhs, tr) + lift(rhs, tr)};
}

Scalar operator-(const Scalar &lhs, const Scalar &rhs) {
  if (const Real *l = std::get_if<Real>(&lhs)) {
    if (const Real *r = std::get_if<Real>(&rhs)) {
      return real_op(*l, *r, [](auto a, auto b) { return a - b; });
    }
  }

  Trace &tr = trace_of(lhs, rhs);
  return Scalar{lift(lhs, tr) - lift(rhs, tr)};
}

Scalar operator*(const Scalar &lhs, const Scalar &rhs) {
  if (const Real *l = std::get_if<Real>(&lhs)) {
    if (const Real *r = std::get_if<Real>(&rhs)) {
      return real_op(*l, *r, [](auto a, auto b) { return a * b; });
    }
  }

  Trace &tr = trace_of(lhs, rhs);
  return Scalar{lift(lhs, tr) * lift(rhs, tr)};
}

Scalar operator/(const Scalar &lhs, const Scalar &rhs) {
  if (const Real *l = std::get_if<Real>(&lhs)) {
    if (const Real *r = std::get_if<Real>(&rhs)) {
      return real_op(*l, *r, [](auto a, auto b) { return a / b; });
    }
  }

  Trace &tr = trace_of(lhs, rhs);
  return Scalar{lift(lhs, tr) / lift(rhs, tr)};
}

Scalar operator-(const Scalar &rhs) {
  if (const Real *r = std::get_if<Real>(&rhs)) {
    return real_unary_op(*r, [](auto v) { return -v; });
  }
  return Scalar{-std::get<Variable>(rhs)};
}

Scalar &operator+=(Scalar &lhs, const Scalar &rhs) {
  lhs = lhs + rhs;
  return lhs;
}

Scalar &operator-=(Scalar &lhs, const Scalar &rhs) {
  lhs = lhs - rhs;
  return lhs;
}

Scalar &operator*=(Scalar &lhs, const Scalar &rhs) {
  lhs = lhs * rhs;
  return lhs;
}

Scalar &operator/=(Scalar &lhs, const Scalar &rhs) {
  lhs = lhs / rhs;
  return lhs;
}

std::partial_ordering operator<=>(const Scalar &lhs, const Scalar &rhs) {
  return to_double(lhs) <=> to_double(rhs);
}

bool operator==(const Scalar &lhs, const Scalar &rhs) {
  return to_double(lhs) == to_double(rhs);
}

Scalar sin(const Scalar &x) {
  if (const Real *r = std::get_if<Real>(&x)) {
    return real_unary_op(*r, [](auto v) { return std::sin(v); });
  }
  return Scalar{sin(std::get<Variable>(x))};
}

Scalar cos(const Scalar &x) {
  if (const Real *r = std::get_if<Real>(&x)) {
    return real_unary_op(*r, [](auto v) { return std::cos(v); });
  }
  return Scalar{cos(std::get<Variable>(x))};
}

Scalar tan(const Scalar &x) {
  if (const Real *r = std::get_if<Real>(&x)) {
    return real_unary_op(*r, [](auto v) { return std::tan(v); });
  }
  return Scalar{tan(std::get<Variable>(x))};
}

Scalar csc(const Scalar &x) {
  if (const Real *r = std::get_if<Real>(&x)) {
    return real_unary_op(*r, [](auto v) { return 1 / std::sin(v); });
  }
  return Scalar{csc(std::get<Variable>(x))};
}

Scalar sec(const Scalar &x) {
  if (const Real *r = std::get_if<Real>(&x)) {
    return real_unary_op(*r, [](auto v) { return 1 / std::cos(v); });
  }
  return Scalar{sec(std::get<Variable>(x))};
}

Scalar cot(const Scalar &x) {
  if (const Real *r = std::get_if<Real>(&x)) {
    return real_unary_op(*r, [](auto v) { return 1 / std::tan(v); });
  }
  return Scalar{cot(std::get<Variable>(x))};
}

Scalar pow(const Scalar &base, const Scalar &exponent) {
  if (const Real *b = std::get_if<Real>(&base)) {
    if (const Real *e = std::get_if<Real>(&exponent)) {
      return real_op(*b, *e, [](auto bb, auto ee) { return std::pow(bb, ee); });
    }
  }

  Trace &tr = trace_of(base, exponent);
  return Scalar{pow(lift(base, tr), lift(exponent, tr))};
}

Scalar log(const Scalar &x) {
  if (const Real *r = std::get_if<Real>(&x)) {
    return real_unary_op(*r, [](auto v) { return std::log(v); });
  }
  return Scalar{log(std::get<Variable>(x))};
}

Scalar exp(const Scalar &x) {
  if (const Real *r = std::get_if<Real>(&x)) {
    return real_unary_op(*r, [](auto v) { return std::exp(v); });
  }
  return Scalar{exp(std::get<Variable>(x))};
}

Scalar abs(const Scalar &x) {
  if (const Real *r = std::get_if<Real>(&x)) {
    return real_unary_op(*r, [](auto v) { return std::abs(v); });
  }
  return Scalar{abs(std::get<Variable>(x))};
}

Scalar sqrt(const Scalar &x) {
  if (const Real *r = std::get_if<Real>(&x)) {
    return real_unary_op(*r, [](auto v) { return std::sqrt(v); });
  }
  return Scalar{sqrt(std::get<Variable>(x))};
}

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

Variable &operator+=(Variable &lhs, const Variable &rhs) {
  lhs = lhs + rhs;
  return lhs;
}

Variable &operator-=(Variable &lhs, const Variable &rhs) {
  lhs = lhs - rhs;
  return lhs;
}

Variable &operator*=(Variable &lhs, const Variable &rhs) {
  lhs = lhs * rhs;
  return lhs;
}

Variable &operator/=(Variable &lhs, const Variable &rhs) {
  lhs = lhs / rhs;
  return lhs;
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

Vector operator+(const Vector &lhs, const Vector &rhs) {
  assert(lhs.size() == rhs.size() &&
         "Vector addition expects dimension of lhs and rhs to match");

  Vector out;
  out.reserve(lhs.size());
  for (size_t i = 0; i < lhs.size(); ++i) {
    out.push_back(lhs[i] + rhs[i]);
  }
  return out;
}

Vector operator-(const Vector &lhs, const Vector &rhs) {
  assert(lhs.size() == rhs.size() &&
         "Vector subtraction expects dimension of lhs and rhs to match");

  Vector out;
  out.reserve(lhs.size());
  for (size_t i = 0; i < lhs.size(); ++i) {
    out.push_back(lhs[i] - rhs[i]);
  }
  return out;
}

Scalar dot(const Vector &lhs, const Vector &rhs) {
  assert(lhs.size() == rhs.size() &&
         "Dot product expects dimension of lhs and rhs to match");

  Scalar out{0.0};
  for (size_t i = 0; i < lhs.size(); ++i) {
    out += lhs[i] * rhs[i];
  }
  return out;
}

Matrix operator*(const Matrix &lhs, const Matrix &rhs) {
  assert(lhs.cols == rhs.rows &&
         "Matrix operator*: lhs.cols must equal rhs.rows");
  Matrix result(lhs.rows, rhs.cols);
  for (size_t i = 0; i < lhs.rows; ++i) {
    for (size_t k = 0; k < lhs.cols; ++k) {
      const Scalar &a_ik = lhs(i, k);
      for (size_t j = 0; j < rhs.cols; ++j) {
        result(i, j) += a_ik * rhs(k, j);
      }
    }
  }
  return result;
}

Vector operator*(const Matrix &lhs, const Vector &rhs) {
  assert(lhs.cols == rhs.size() &&
         "Matrix-vector product: lhs.cols must equal rhs.size()");
  Vector result(lhs.rows);
  for (size_t i = 0; i < lhs.rows; ++i) {
    for (size_t j = 0; j < lhs.cols; ++j) {
      result[i] += lhs(i, j) * rhs[j];
    }
  }
  return result;
}

Vector operator*(const Vector &lhs, const Matrix &rhs) {
  assert(rhs.rows == lhs.size() &&
         "Vector-matrix product: lhs.size() must equal rhs.rows");
  Vector result(rhs.cols);
  for (size_t j = 0; j < rhs.rows; ++j) {
    const Scalar &x_j = lhs[j];
    for (size_t i = 0; i < rhs.cols; ++i) {
      result[i] += x_j * rhs(j, i);
    }
  }
  return result;
}
