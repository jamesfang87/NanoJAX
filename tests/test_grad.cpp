#include "include/ad/grad.hpp"
#include "include/ad/scalar.hpp"

#include "include/nn/layers/activation/activations.hpp"

#include "tests/test_utils.hpp"

#include <tuple>
#include <vector>

using nanojax_test::expect_near;

namespace {
Scalar f(const Scalar &x, const Scalar &y) { return x * x * y + y * y * y; }
} // namespace

int main() {
  double x0 = 2.0, y0 = 3.0;

  // Single-argument grad: returns a plain Scalar.
  {
    auto df = grad([](const Scalar &x) { return x * x * x; });
    expect_near("d/dx x^3 at x=3", to_double(df(3.0)), 27.0);
  }

  // Single-argument grad(grad(f)): second derivative of x^3 is 6x.
  {
    auto d2 = grad(grad([](const Scalar &x) { return x * x * x; }));
    expect_near("d2/dx2 x^3 at x=3", to_double(d2(3.0)), 18.0);
  }

  // relu: 0 on the negative side, 1 on the positive side.
  {
    auto drelu = grad(ReLU);
    expect_near("drelu/dx at x=-2", to_double(drelu(-2.0)), 0.0);
    expect_near("drelu/dx at x=2", to_double(drelu(2.0)), 1.0);
  }

  // Multi-argument grad(f): arity > 1 returns a tuple of one gradient per
  // argument, regardless of argnums.
  {
    auto g = grad(f);
    auto [dfdx, dfdy] = g(x0, y0);
    expect_near("df/dx", to_double(dfdx), 2 * x0 * y0);
    expect_near("df/dy", to_double(dfdy), x0 * x0 + 3 * y0 * y0);
  }

  // grad(grad(f), argnums=0) selects df/dx as the scalar to differentiate
  // again, then (arity still 2) returns its full gradient tuple: Hessian
  // row 0.
  {
    auto row0 = grad(grad(f), 0);
    auto [d2fdx2, d2fdxdy] = row0(x0, y0);
    expect_near("d2f/dx2 ", to_double(d2fdx2), 2 * y0);
    expect_near("d2f/dxdy", to_double(d2fdxdy), 2 * x0);
  }

  // grad(grad(f), argnums=1) selects df/dy instead: Hessian row 1.
  {
    auto row1 = grad(grad(f), 1);
    auto [d2fdydx, d2fdy2] = row1(x0, y0);
    expect_near("d2f/dydx", to_double(d2fdydx), 2 * x0);
    expect_near("d2f/dy2 ", to_double(d2fdy2), 6 * y0);
  }

  // Symmetry of mixed partials.
  {
    double dxdy = to_double(std::get<1>(grad(grad(f), 0)(x0, y0)));
    double dydx = to_double(std::get<0>(grad(grad(f), 1)(x0, y0)));
    expect_near("d2f/dxdy == d2f/dydx", dxdy, dydx);
  }

  return nanojax_test::report();
}
