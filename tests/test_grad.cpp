#include "include/ad/grad.hpp"
#include "include/ad/scalar.hpp"

#include "include/nn/layers/activation/activations.hpp"

#include "tests/test_utils.hpp"

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

  // Multi-argument grad(f): arity > 1 returns the full gradient vector,
  // regardless of argnums.
  {
    auto g = grad(f);
    Vector gv = g(x0, y0);
    expect_near("df/dx", to_double(gv[0]), 2 * x0 * y0);
    expect_near("df/dy", to_double(gv[1]), x0 * x0 + 3 * y0 * y0);
  }

  // grad(grad(f), argnums=0) selects df/dx as the scalar to differentiate
  // again, then (arity still 2) returns its full gradient: Hessian row 0.
  {
    auto row0 = grad(grad(f), 0);
    Vector h0 = row0(x0, y0);
    expect_near("d2f/dx2  (row0[0])", to_double(h0[0]), 2 * y0);
    expect_near("d2f/dxdy (row0[1])", to_double(h0[1]), 2 * x0);
  }

  // grad(grad(f), argnums=1) selects df/dy instead: Hessian row 1.
  {
    auto row1 = grad(grad(f), 1);
    Vector h1 = row1(x0, y0);
    expect_near("d2f/dydx (row1[0])", to_double(h1[0]), 2 * x0);
    expect_near("d2f/dy2  (row1[1])", to_double(h1[1]), 6 * y0);
  }

  // Symmetry of mixed partials.
  {
    double dxdy = to_double(grad(grad(f), 0)(x0, y0)[1]);
    double dydx = to_double(grad(grad(f), 1)(x0, y0)[0]);
    expect_near("d2f/dxdy == d2f/dydx", dxdy, dydx);
  }

  return nanojax_test::report();
}
