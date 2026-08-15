#include "include/ad/scalar.hpp"
#include "tests/test_utils.hpp"

#include <cmath>

using nanojax_test::expect_near;
using nanojax_test::expect_true;

int main() {
  Scalar a{2.0};
  Scalar b{3.0};

  expect_near("2 + 3", to_double(a + b), 5.0);
  expect_near("2 - 3", to_double(a - b), -1.0);
  expect_near("2 * 3", to_double(a * b), 6.0);
  expect_near("2 / 3", to_double(a / b), 2.0 / 3.0);
  expect_near("-2", to_double(-a), -2.0);

  expect_true("2 == 2", Scalar{2.0} == Scalar{2.0});
  expect_true("2 != 3", !(a == b));
  expect_true("2 < 3", a < b);
  expect_true("3 > 2", b > a);
  expect_true("2 <= 2", a <= Scalar{2.0});
  expect_true("3 >= 3", b >= Scalar{3.0});
  expect_true("!(3 < 2)", !(b < a));

  double x = 0.6;
  Scalar sx{x};
  expect_near("sin(x)", to_double(sin(sx)), std::sin(x));
  expect_near("cos(x)", to_double(cos(sx)), std::cos(x));
  expect_near("tan(x)", to_double(tan(sx)), std::tan(x));
  expect_near("csc(x)", to_double(csc(sx)), 1.0 / std::sin(x));
  expect_near("sec(x)", to_double(sec(sx)), 1.0 / std::cos(x));
  expect_near("cot(x)", to_double(cot(sx)), 1.0 / std::tan(x));

  expect_near("pow(2, 10)", to_double(pow(Scalar{2.0}, Scalar{10.0})), 1024.0);
  expect_near("log(2.5)", to_double(log(Scalar{2.5})), std::log(2.5));
  expect_near("exp(x)", to_double(exp(sx)), std::exp(x));
  expect_near("abs(2)", to_double(abs(Scalar{2.0})), 2.0);
  expect_near("abs(-2)", to_double(abs(Scalar{-2.0})), 2.0);
  expect_near("sqrt(4)", to_double(sqrt(Scalar{4.0})), 2.0);

  // float/double mixing within Real, since Scalar wraps variant<float,
  // double> rather than a single fixed width.
  Scalar f32{2.0f};
  Scalar f64{3.0};
  expect_near("float + double", to_double(f32 + f64), 5.0);
  expect_near("float * float", to_double(Scalar{2.0f} * Scalar{4.0f}), 8.0);

  return nanojax_test::report();
}
