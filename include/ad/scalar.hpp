#pragma once

#include <cstddef>
#include <format>
#include <ostream>
#include <variant>

using Real = std::variant<float, double>;
using Scalar = std::variant<Real, struct Variable>;

struct Variable {
  class Trace *trace;
  size_t id;
};

Real value_of(const Scalar &s);
Real value_of(const Variable &x);

double to_double(const Real &r);
double to_double(const Scalar &s);
float to_float(const Real &r);
float to_float(const Scalar &s);

// Streams the value held by r/s/v (the held float or double for Real, and
// the traced primal for Scalar and Variable), rather than a variant's index
// or an object's address.
std::ostream &operator<<(std::ostream &os, const Real &r);
std::ostream &operator<<(std::ostream &os, const Scalar &s);
std::ostream &operator<<(std::ostream &os, const Variable &v);

Scalar operator-(const Scalar &rhs);
Scalar operator+(const Scalar &lhs, const Scalar &rhs);
Scalar operator-(const Scalar &lhs, const Scalar &rhs);
Scalar operator*(const Scalar &lhs, const Scalar &rhs);
Scalar operator/(const Scalar &lhs, const Scalar &rhs);

Scalar operator-(const Scalar &rhs);
Scalar operator+(const Scalar &lhs, const Scalar &rhs);
Scalar operator-(const Scalar &lhs, const Scalar &rhs);
Scalar operator*(const Scalar &lhs, const Scalar &rhs);
Scalar operator/(const Scalar &lhs, const Scalar &rhs);

std::partial_ordering operator<=>(const Scalar &lhs, const Scalar &rhs);
bool operator==(const Scalar &lhs, const Scalar &rhs);

Scalar sin(const Scalar &x);
Scalar cos(const Scalar &x);
Scalar tan(const Scalar &x);
Scalar csc(const Scalar &x);
Scalar sec(const Scalar &x);
Scalar cot(const Scalar &x);

Scalar pow(const Scalar &base, const Scalar &exponent);
Scalar log(const Scalar &x);
Scalar exp(const Scalar &x);
Scalar abs(const Scalar &x);
Scalar sqrt(const Scalar &x);

Variable operator-(const Variable &rhs);
Variable operator+(const Variable &lhs, const Variable &rhs);
Variable operator-(const Variable &lhs, const Variable &rhs);
Variable operator*(const Variable &lhs, const Variable &rhs);
Variable operator/(const Variable &lhs, const Variable &rhs);

std::partial_ordering operator<=>(const Variable &lhs, const Variable &rhs);
bool operator==(const Variable &lhs, const Variable &rhs);

Variable sin(const Variable &x);
Variable cos(const Variable &x);
Variable tan(const Variable &x);
Variable csc(const Variable &x);
Variable sec(const Variable &x);
Variable cot(const Variable &x);

Variable pow(const Variable &base, const Variable &exponent);
Variable log(const Variable &x);
Variable exp(const Variable &x);
Variable abs(const Variable &x);
Variable sqrt(const Variable &x);

// Each formatter below inherits its parsing (precision, width, fill, ...)
// from std::formatter<double>, and only overrides format() to first reduce
// r/s/v to the double it holds. This is what makes std::format and
// std::println accept a Real, a Scalar, or a Variable directly, formatting
// each the same way to_double(...) would render it.
template <> struct std::formatter<Real> : std::formatter<double> {
  auto format(const Real &r, std::format_context &ctx) const {
    return std::formatter<double>::format(to_double(r), ctx);
  }
};

template <> struct std::formatter<Scalar> : std::formatter<double> {
  auto format(const Scalar &s, std::format_context &ctx) const {
    return std::formatter<double>::format(to_double(s), ctx);
  }
};

template <> struct std::formatter<Variable> : std::formatter<double> {
  auto format(const Variable &v, std::format_context &ctx) const {
    return std::formatter<double>::format(to_double(value_of(v)), ctx);
  }
};
