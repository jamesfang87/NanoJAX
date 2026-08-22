// Finite-difference gradient checks.
//
// Each case below compares the analytic derivative returned by grad()
// against a central-difference estimate of the same forward function, so
// that a check exercises exactly the function under test rather than a
// hand-rolled reference formula. This gives an independent correctness
// signal for every differentiable primitive and activation function, on top
// of the closed-form checks in test_grad.cpp.

#include "include/ad/grad.hpp"
#include "include/ad/scalar.hpp"

#include "include/nn/layers/activation/activations.hpp"

#include "tests/test_utils.hpp"

#include <functional>
#include <string>
#include <utility>
#include <vector>

using nanojax_test::expect_near;

namespace {

constexpr double kStep = 1e-5;

// Central-difference approximation of d/dx f(x).
double central_diff(const std::function<Scalar(const Scalar &)> &f, double x) {
  return (to_double(f(Scalar{x + kStep})) - to_double(f(Scalar{x - kStep}))) /
         (2.0 * kStep);
}

// Central-difference approximation of the two partial derivatives of a
// two-argument function f(a, b), one argument perturbed at a time.
std::pair<double, double>
central_diff2(const std::function<Scalar(const Scalar &, const Scalar &)> &f,
              double a, double b) {
  double da = (to_double(f(Scalar{a + kStep}, Scalar{b})) -
               to_double(f(Scalar{a - kStep}, Scalar{b}))) /
              (2.0 * kStep);
  double db = (to_double(f(Scalar{a}, Scalar{b + kStep})) -
               to_double(f(Scalar{a}, Scalar{b - kStep}))) /
              (2.0 * kStep);
  return {da, db};
}

struct UnaryCase {
  std::string name;
  std::function<Scalar(const Scalar &)> f;
  double x;
};

} // namespace

int main() {
  // Unary primitives and activation functions. x is chosen away from each
  // function's singularities and kinks (tan/sec/csc/cot away from 0 and
  // pi/2, log/sqrt away from 0, abs/ReLU/Leaky_ReLU/ELU/SELU checked on
  // both sides of their kink at 0 but never at 0 itself).
  std::vector<UnaryCase> cases = {
      {"d/dx (-x) at x=2", [](const Scalar &x) { return -x; }, 2.0},
      {"d/dx sin(x) at x=0.6", [](const Scalar &x) { return sin(x); }, 0.6},
      {"d/dx cos(x) at x=0.6", [](const Scalar &x) { return cos(x); }, 0.6},
      {"d/dx tan(x) at x=0.6", [](const Scalar &x) { return tan(x); }, 0.6},
      {"d/dx csc(x) at x=0.6", [](const Scalar &x) { return csc(x); }, 0.6},
      {"d/dx sec(x) at x=0.6", [](const Scalar &x) { return sec(x); }, 0.6},
      {"d/dx cot(x) at x=0.6", [](const Scalar &x) { return cot(x); }, 0.6},
      {"d/dx log(x) at x=2.5", [](const Scalar &x) { return log(x); }, 2.5},
      {"d/dx exp(x) at x=0.6", [](const Scalar &x) { return exp(x); }, 0.6},
      {"d/dx sqrt(x) at x=4", [](const Scalar &x) { return sqrt(x); }, 4.0},
      {"d/dx abs(x) at x=2", [](const Scalar &x) { return abs(x); }, 2.0},
      {"d/dx abs(x) at x=-3", [](const Scalar &x) { return abs(x); }, -3.0},
      {"d/dx sigmoid(x) at x=0.5", [](const Scalar &x) { return sigmoid(x); },
       0.5},
      {"d/dx swish(x) at x=0.5", [](const Scalar &x) { return swish(x); }, 0.5},
      {"d/dx tanh(x) at x=0.5", [](const Scalar &x) { return tanh(x); }, 0.5},
      {"d/dx ReLU(x) at x=2", [](const Scalar &x) { return ReLU(x); }, 2.0},
      {"d/dx ReLU(x) at x=-2", [](const Scalar &x) { return ReLU(x); }, -2.0},
      {"d/dx Leaky_ReLU(x) at x=2",
       [](const Scalar &x) { return Leaky_ReLU(x); }, 2.0},
      {"d/dx Leaky_ReLU(x) at x=-2",
       [](const Scalar &x) { return Leaky_ReLU(x); }, -2.0},
      {"d/dx ELU(x) at x=2", [](const Scalar &x) { return ELU(x); }, 2.0},
      {"d/dx ELU(x) at x=-2", [](const Scalar &x) { return ELU(x); }, -2.0},
      {"d/dx SELU(x) at x=2", [](const Scalar &x) { return SELU(x); }, 2.0},
      {"d/dx SELU(x) at x=-2", [](const Scalar &x) { return SELU(x); }, -2.0},
      {"d/dx GELU(x) at x=1", [](const Scalar &x) { return GELU(x); }, 1.0},
      {"d/dx GELU(x) at x=-1", [](const Scalar &x) { return GELU(x); }, -1.0},
  };

  for (const auto &c : cases) {
    double analytic = to_double(grad(c.f)(c.x));
    double numeric = central_diff(c.f, c.x);
    expect_near(c.name, analytic, numeric);
  }

  // Binary primitives: grad() returns a tuple of the two partials for a
  // two-argument function, so each entry is checked against the
  // corresponding partial derivative.
  {
    std::vector<std::pair<
        std::string, std::function<Scalar(const Scalar &, const Scalar &)>>>
        binary_cases = {
            {"a+b", [](const Scalar &a, const Scalar &b) { return a + b; }},
            {"a-b", [](const Scalar &a, const Scalar &b) { return a - b; }},
            {"a*b", [](const Scalar &a, const Scalar &b) { return a * b; }},
            {"a/b", [](const Scalar &a, const Scalar &b) { return a / b; }},
            {"pow(a,b)",
             [](const Scalar &a, const Scalar &b) { return pow(a, b); }},
        };

    double a0 = 2.0, b0 = 3.0;
    for (const auto &[name, f] : binary_cases) {
      auto [dfa, dfb] = grad(f)(a0, b0);
      auto [da, db] = central_diff2(f, a0, b0);
      expect_near("d/da " + name, to_double(dfa), da);
      expect_near("d/db " + name, to_double(dfb), db);
    }
  }

  // Second derivative: numerically differentiate the first-derivative
  // function d/dx sin(x) itself, and compare against the analytic second
  // derivative returned by grad(grad(f)). This checks that nested grad()
  // composes correctly, not just that a single derivative is correct.
  {
    auto f = [](const Scalar &x) { return sin(x); };
    auto d1 = grad(f);
    double x = 0.6;
    double numeric2 =
        (to_double(d1(x + kStep)) - to_double(d1(x - kStep))) / (2.0 * kStep);
    double analytic2 = to_double(grad(grad(f))(x));
    expect_near("d2/dx2 sin(x) at x=0.6", analytic2, numeric2);
  }

  return nanojax_test::report();
}
