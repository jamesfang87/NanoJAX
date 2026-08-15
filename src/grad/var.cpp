#include "include/grad.hpp"

#include <cmath>
#include <compare>

double Var::value() const { return tape.primals[id]; }

Var operator+(const Var &lhs, const Var &rhs) {
  double out = lhs.value() + rhs.value();
  return tape.add_node(out, std::vector<Var>{lhs, rhs}, [=](size_t self_id) {
    // grad_in += grad_out * d_out/d_in
    double output_adjoint = tape.adjoints[self_id];

    tape.adjoints[lhs.id] += output_adjoint;
    tape.adjoints[rhs.id] += output_adjoint;
  });
}

Var operator-(const Var &lhs, const Var &rhs) {
  double out = lhs.value() - rhs.value();
  return tape.add_node(out, std::vector<Var>{lhs, rhs}, [=](size_t self_id) {
    // grad_in += grad_out * d_out/d_in
    double output_adjoint = tape.adjoints[self_id];

    tape.adjoints[lhs.id] += output_adjoint;
    tape.adjoints[rhs.id] -= output_adjoint;
  });
}

Var operator*(const Var &lhs, const Var &rhs) {
  double out = lhs.value() * rhs.value();
  return tape.add_node(out, std::vector<Var>{lhs, rhs}, [=](size_t self_id) {
    // grad_in += grad_out * d_out/d_in
    double output_adjoint = tape.adjoints[self_id];

    tape.adjoints[lhs.id] += output_adjoint * rhs.value();
    tape.adjoints[rhs.id] += output_adjoint * lhs.value();
  });
}

Var operator/(const Var &lhs, const Var &rhs) {
  double out = lhs.value() / rhs.value();
  return tape.add_node(out, std::vector<Var>{lhs, rhs}, [=](size_t self_id) {
    // grad_in += grad_out * d_out/d_in
    double output_adjoint = tape.adjoints[self_id];

    tape.adjoints[lhs.id] += output_adjoint * (1.0 / rhs.value());
    tape.adjoints[rhs.id] +=
        output_adjoint * (-lhs.value() / (rhs.value() * rhs.value()));
  });
}

std::partial_ordering operator<=>(const Var &lhs, const Var &rhs) {
  return lhs.value() <=> rhs.value();
}

std::partial_ordering operator<=>(const Var &lhs, double rhs) {
  return lhs.value() <=> rhs;
}
std::partial_ordering operator<=>(double lhs, const Var &rhs) {
  return lhs <=> rhs.value();
}
bool operator==(const Var &lhs, double rhs) { return lhs.value() == rhs; }
bool operator==(double lhs, const Var &rhs) { return lhs == rhs.value(); }

Var pow(const Var &base, double exponent) {
  double out = std::pow(base.value(), exponent);
  return tape.add_node(out, std::vector<Var>{base}, [=](size_t self_id) {
    double output_adjoint = tape.adjoints[self_id];
    tape.adjoints[base.id] +=
        output_adjoint * exponent * std::pow(base.value(), exponent - 1);
  });
}

Var pow(double base, const Var &exponent) {
  double out = std::pow(base, exponent.value());
  return tape.add_node(out, std::vector<Var>{exponent}, [=](size_t self_id) {
    double output_adjoint = tape.adjoints[self_id];
    tape.adjoints[exponent.id] += output_adjoint * out * std::log(base);
  });
}

Var pow(const Var &base, const Var &exponent) {
  double out = std::pow(base.value(), exponent.value());
  return tape.add_node(
      out, std::vector<Var>{base, exponent}, [=](size_t self_id) {
        double output_adjoint = tape.adjoints[self_id];

        tape.adjoints[base.id] += output_adjoint * exponent.value() *
                                  std::pow(base.value(), exponent.value() - 1);
        tape.adjoints[exponent.id] +=
            output_adjoint * out * std::log(base.value());
      });
}

Var log(const Var &x) {
  double out = std::log(x.value());
  return tape.add_node(out, std::vector<Var>{x}, [=](size_t self_id) {
    double output_adjoint = tape.adjoints[self_id];
    tape.adjoints[x.id] += output_adjoint * (1.0 / x.value());
  });
}

Var exp(const Var &x) {
  double out = std::exp(x.value());
  return tape.add_node(out, std::vector<Var>{x}, [=](size_t self_id) {
    double output_adjoint = tape.adjoints[self_id];
    tape.adjoints[x.id] += output_adjoint * out;
  });
}

Var abs(const Var &x) {
  double out = std::abs(x.value());
  return tape.add_node(out, std::vector<Var>{x}, [=](size_t self_id) {
    double output_adjoint = tape.adjoints[self_id];
    double sign = x.value() > 0.0 ? 1.0 : (x.value() < 0.0 ? -1.0 : 0.0);
    tape.adjoints[x.id] += output_adjoint * sign;
  });
}

Var sqrt(const Var &x) { return pow(x, 0.5); }

Var sin(const Var &x) {
  double out = std::sin(x.value());
  return tape.add_node(out, std::vector<Var>{x}, [=](size_t self_id) {
    double output_adjoint = tape.adjoints[self_id];
    tape.adjoints[x.id] += output_adjoint * std::cos(x.value());
  });
}

Var cos(const Var &x) {
  double out = std::cos(x.value());
  return tape.add_node(out, std::vector<Var>{x}, [=](size_t self_id) {
    double output_adjoint = tape.adjoints[self_id];
    tape.adjoints[x.id] += output_adjoint * -std::sin(x.value());
  });
}

Var tan(const Var &x) {
  double out = std::tan(x.value());
  return tape.add_node(out, std::vector<Var>{x}, [=](size_t self_id) {
    double output_adjoint = tape.adjoints[self_id];
    tape.adjoints[x.id] +=
        output_adjoint * (1.0 / (std::cos(x.value()) * std::cos(x.value())));
  });
}

Var csc(const Var &x) {
  double out = 1.0 / std::sin(x.value());
  return tape.add_node(out, std::vector<Var>{x}, [=](size_t self_id) {
    double output_adjoint = tape.adjoints[self_id];
    tape.adjoints[x.id] += output_adjoint * 1.0 / std::tan(x.value()) * -out;
  });
}

Var sec(const Var &x) {
  double out = 1.0 / std::cos(x.value());
  return tape.add_node(out, std::vector<Var>{x}, [=](size_t self_id) {
    double output_adjoint = tape.adjoints[self_id];
    tape.adjoints[x.id] += output_adjoint * std::tan(x.value()) * out;
  });
}

Var cot(const Var &x) {
  double out = 1.0 / std::tan(x.value());
  return tape.add_node(out, std::vector<Var>{x}, [=](size_t self_id) {
    double output_adjoint = tape.adjoints[self_id];
    tape.adjoints[x.id] +=
        output_adjoint * -(1.0 / (std::sin(x.value()) * std::sin(x.value())));
  });
}
