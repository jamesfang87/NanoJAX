#pragma once

#include <compare>
#include <cstddef>
#include <format>
#include <ostream>
#include <utility>
#include <variant>
#include <vector>

using Real = std::variant<float, double>;
using Scalar = std::variant<Real, struct Variable>;

struct Variable {
  class Trace *trace;
  size_t id;
};

struct Vector : std::vector<Scalar> {
  Vector() = default;
  Vector(std::initializer_list<Scalar> init) : std::vector<Scalar>(init) {}
  Vector(size_t n, Scalar fill = Scalar{0.0})
      : std::vector<Scalar>(n, std::move(fill)) {}
};

struct Matrix {
  Vector data;
  size_t rows = 0;
  size_t cols = 0;

  Matrix() = default;
  Matrix(size_t rows, size_t cols, Scalar fill = Scalar{0.0})
      : data(rows * cols, std::move(fill)), rows(rows), cols(cols) {}

  Scalar &operator()(size_t i, size_t j) { return data[i * cols + j]; }
  const Scalar &operator()(size_t i, size_t j) const {
    return data[i * cols + j];
  }
};

Real value_of(const Scalar &s);
Real value_of(const Variable &x);

double to_double(const Real &r);
double to_double(const Scalar &s);
float to_float(const Real &r);
float to_float(const Scalar &s);

std::ostream &operator<<(std::ostream &os, const Real &r);
std::ostream &operator<<(std::ostream &os, const Scalar &s);
std::ostream &operator<<(std::ostream &os, const Variable &v);

Scalar operator-(const Scalar &rhs);
Scalar operator+(const Scalar &lhs, const Scalar &rhs);
Scalar operator-(const Scalar &lhs, const Scalar &rhs);
Scalar operator*(const Scalar &lhs, const Scalar &rhs);
Scalar operator/(const Scalar &lhs, const Scalar &rhs);

Scalar &operator+=(Scalar &lhs, const Scalar &rhs);
Scalar &operator-=(Scalar &lhs, const Scalar &rhs);
Scalar &operator*=(Scalar &lhs, const Scalar &rhs);
Scalar &operator/=(Scalar &lhs, const Scalar &rhs);

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

Variable operator+(const Variable &lhs, const Real &rhs);
Variable operator+(const Real &lhs, const Variable &rhs);
Variable operator-(const Variable &lhs, const Real &rhs);
Variable operator-(const Real &lhs, const Variable &rhs);
Variable operator*(const Variable &lhs, const Real &rhs);
Variable operator*(const Real &lhs, const Variable &rhs);
Variable operator/(const Variable &lhs, const Real &rhs);
Variable operator/(const Real &lhs, const Variable &rhs);
Variable pow(const Variable &base, const Real &exponent);
Variable pow(const Real &base, const Variable &exponent);

Variable &operator+=(Variable &lhs, const Variable &rhs);
Variable &operator-=(Variable &lhs, const Variable &rhs);
Variable &operator*=(Variable &lhs, const Variable &rhs);
Variable &operator/=(Variable &lhs, const Variable &rhs);

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

Vector operator+(const Vector &lhs, const Vector &rhs);
Vector operator-(const Vector &lhs, const Vector &rhs);
Scalar dot(const Vector &lhs, const Vector &rhs);
Matrix operator*(const Matrix &lhs, const Matrix &rhs);
Vector operator*(const Matrix &lhs, const Vector &rhs);
Vector operator*(const Vector &lhs, const Matrix &rhs);

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
