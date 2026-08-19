#include "include/ad/scalar.hpp"
#include "include/ad/trace.hpp"
#include "tests/test_utils.hpp"

#include <cmath>
#include <memory>
#include <string>

using nanojax_test::expect_near;
using nanojax_test::expect_true;

namespace {

// RAII wrapper around trace_stack, mirroring what grad() does around a
// single call to f, so each test case gets its own isolated Trace.
struct TraceScope {
  TraceScope() { trace_stack.push(std::make_unique<Trace>()); }
  ~TraceScope() { trace_stack.pop(); }
};

} // namespace

int main() {
  // Binary arithmetic: primal and both partials.
  {
    TraceScope scope;
    Trace &tr = trace();
    Variable x = tr.add_variable(Scalar{2.0}, nullptr);
    Variable y = tr.add_variable(Scalar{3.0}, nullptr);

    Variable out = x + y;
    expect_near("Variable x+y primal", to_double(value_of(out)), 5.0);
    tr.backward(out);
    expect_near("d(x+y)/dx", to_double(tr.adjoints[x.id]), 1.0);
    expect_near("d(x+y)/dy", to_double(tr.adjoints[y.id]), 1.0);
  }
  {
    TraceScope scope;
    Trace &tr = trace();
    Variable x = tr.add_variable(Scalar{2.0}, nullptr);
    Variable y = tr.add_variable(Scalar{3.0}, nullptr);

    Variable out = x - y;
    expect_near("Variable x-y primal", to_double(value_of(out)), -1.0);
    tr.backward(out);
    expect_near("d(x-y)/dx", to_double(tr.adjoints[x.id]), 1.0);
    expect_near("d(x-y)/dy", to_double(tr.adjoints[y.id]), -1.0);
  }
  {
    TraceScope scope;
    Trace &tr = trace();
    Variable x = tr.add_variable(Scalar{2.0}, nullptr);
    Variable y = tr.add_variable(Scalar{3.0}, nullptr);

    Variable out = x * y;
    expect_near("Variable x*y primal", to_double(value_of(out)), 6.0);
    tr.backward(out);
    expect_near("d(x*y)/dx", to_double(tr.adjoints[x.id]), 3.0); // = y
    expect_near("d(x*y)/dy", to_double(tr.adjoints[y.id]), 2.0); // = x
  }
  {
    TraceScope scope;
    Trace &tr = trace();
    Variable x = tr.add_variable(Scalar{6.0}, nullptr);
    Variable y = tr.add_variable(Scalar{3.0}, nullptr);

    Variable out = x / y;
    expect_near("Variable x/y primal", to_double(value_of(out)), 2.0);
    tr.backward(out);
    expect_near("d(x/y)/dx", to_double(tr.adjoints[x.id]), 1.0 / 3.0);
    expect_near("d(x/y)/dy", to_double(tr.adjoints[y.id]), -6.0 / 9.0);
  }
  {
    TraceScope scope;
    Trace &tr = trace();
    Variable x = tr.add_variable(Scalar{2.0}, nullptr);

    Variable out = -x;
    expect_near("Variable -x primal", to_double(value_of(out)), -2.0);
    tr.backward(out);
    expect_near("d(-x)/dx", to_double(tr.adjoints[x.id]), -1.0);
  }

  // Compound assignment rebinds the local Variable to the new tape node
  // produced by lhs OP rhs; it must not disturb the node the operand
  // previously referred to, so gradients still have to reach the original
  // leaves correctly after the rebind.
  {
    TraceScope scope;
    Trace &tr = trace();
    Variable x = tr.add_variable(Scalar{2.0}, nullptr);
    Variable y = tr.add_variable(Scalar{3.0}, nullptr);

    Variable out = x;
    out += y;
    expect_near("Variable x+=y primal", to_double(value_of(out)), 5.0);
    tr.backward(out);
    expect_near("d(x+=y)/dx", to_double(tr.adjoints[x.id]), 1.0);
    expect_near("d(x+=y)/dy", to_double(tr.adjoints[y.id]), 1.0);
  }
  {
    TraceScope scope;
    Trace &tr = trace();
    Variable x = tr.add_variable(Scalar{2.0}, nullptr);
    Variable y = tr.add_variable(Scalar{3.0}, nullptr);

    Variable out = x;
    out *= y;
    expect_near("Variable x*=y primal", to_double(value_of(out)), 6.0);
    tr.backward(out);
    expect_near("d(x*=y)/dx", to_double(tr.adjoints[x.id]), 3.0); // = y
    expect_near("d(x*=y)/dy", to_double(tr.adjoints[y.id]), 2.0); // = x
  }

  // Comparisons compare by value, not by trace identity.
  {
    TraceScope scope;
    Trace &tr = trace();
    Variable x = tr.add_variable(Scalar{2.0}, nullptr);
    Variable y = tr.add_variable(Scalar{3.0}, nullptr);
    Variable x2 = tr.add_variable(Scalar{2.0}, nullptr);

    expect_true("Variable x == x2", x == x2);
    expect_true("Variable x != y", !(x == y));
    expect_true("Variable x < y", x < y);
    expect_true("Variable y > x", y > x);
  }

  // Unary math functions: primal plus the backward derivative w.r.t. x.
  auto check_unary = [](const char *name, double x0,
                        Variable (*fn)(const Variable &), double want_primal,
                        double want_deriv) {
    TraceScope scope;
    Trace &tr = trace();
    Variable x = tr.add_variable(Scalar{x0}, nullptr);
    Variable out = fn(x);
    expect_near(std::string(name) + " primal", to_double(value_of(out)),
                want_primal);
    tr.backward(out);
    expect_near(std::string("d/dx ") + name, to_double(tr.adjoints[x.id]),
                want_deriv);
  };

  double x0 = 0.6;
  check_unary("sin(x)", x0, sin, std::sin(x0), std::cos(x0));
  check_unary("cos(x)", x0, cos, std::cos(x0), -std::sin(x0));
  check_unary("tan(x)", x0, tan, std::tan(x0),
              1.0 / (std::cos(x0) * std::cos(x0)));
  check_unary("csc(x)", x0, csc, 1.0 / std::sin(x0),
              -1.0 / (std::sin(x0) * std::tan(x0)));
  check_unary("sec(x)", x0, sec, 1.0 / std::cos(x0),
              std::tan(x0) / std::cos(x0));
  check_unary("cot(x)", x0, cot, 1.0 / std::tan(x0),
              -1.0 / (std::sin(x0) * std::sin(x0)));
  check_unary("log(x)", 2.5, log, std::log(2.5), 1.0 / 2.5);
  check_unary("exp(x)", x0, exp, std::exp(x0), std::exp(x0));
  check_unary("abs(x) at +2", 2.0, abs, 2.0, 1.0);
  check_unary("abs(x) at -2", -2.0, abs, 2.0, -1.0);
  check_unary("sqrt(x)", 4.0, sqrt, 2.0, 1.0 / (2 * std::sqrt(4.0)));

  // pow(base, exponent): both partials, base^exponent = 2^3 = 8.
  {
    TraceScope scope;
    Trace &tr = trace();
    Variable base = tr.add_variable(Scalar{2.0}, nullptr);
    Variable exponent = tr.add_variable(Scalar{3.0}, nullptr);

    Variable out = pow(base, exponent);
    expect_near("pow(base, exponent) primal", to_double(value_of(out)), 8.0);
    tr.backward(out);
    expect_near("d(base^exp)/dbase", to_double(tr.adjoints[base.id]),
                3 * std::pow(2.0, 2.0)); // exponent * base^(exponent-1)
    expect_near("d(base^exp)/dexponent", to_double(tr.adjoints[exponent.id]),
                8.0 * std::log(2.0)); // base^exponent * log(base)
  }

  return nanojax_test::report();
}
