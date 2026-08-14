#include "include/grad.hpp"

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
