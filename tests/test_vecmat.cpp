// Vector and Matrix arithmetic, checked against hand-computed values on
// plain Real entries (no Trace needed here, the same way test_scalar.cpp
// checks Scalar arithmetic without one). Gradient flow through these same
// operators is exercised end to end in test_mlp.cpp instead, since MLP's
// forward pass already composes Vector*Matrix and Vector addition under
// Variables.

#include "include/ad/scalar.hpp"
#include "tests/test_utils.hpp"

using nanojax_test::expect_near;
using nanojax_test::expect_true;
using nanojax_test::expect_vector_near;

int main() {
  Vector a = {Scalar{1.0}, Scalar{2.0}, Scalar{3.0}};
  Vector b = {Scalar{4.0}, Scalar{5.0}, Scalar{6.0}};

  expect_vector_near("a+b", a + b, {Scalar{5.0}, Scalar{7.0}, Scalar{9.0}});
  expect_vector_near("(a+b)-b", (a + b) - b, a);
  expect_near("dot(a,b)", to_double(dot(a, b)), 32.0); // 1*4+2*5+3*6

  // A is 2x3, B is 3x2, so both A*B (2x2) and A*x / x*A for a
  // correctly-sized vector are all rectangular, not square: a bug that
  // only shows up off the diagonal (a transposed index, a swapped
  // dimension) will not hide behind a square matrix here.
  Matrix A(2, 3);
  A(0, 0) = Scalar{1.0};
  A(0, 1) = Scalar{2.0};
  A(0, 2) = Scalar{3.0};
  A(1, 0) = Scalar{4.0};
  A(1, 1) = Scalar{5.0};
  A(1, 2) = Scalar{6.0};

  Matrix B(3, 2);
  B(0, 0) = Scalar{7.0};
  B(0, 1) = Scalar{8.0};
  B(1, 0) = Scalar{9.0};
  B(1, 1) = Scalar{10.0};
  B(2, 0) = Scalar{11.0};
  B(2, 1) = Scalar{12.0};

  Matrix C = A * B;
  expect_true("A*B rows", C.rows == 2);
  expect_true("A*B cols", C.cols == 2);
  expect_near("(A*B)(0,0)", to_double(C(0, 0)), 58.0);  // 1*7+2*9+3*11
  expect_near("(A*B)(0,1)", to_double(C(0, 1)), 64.0);  // 1*8+2*10+3*12
  expect_near("(A*B)(1,0)", to_double(C(1, 0)), 139.0); // 4*7+5*9+6*11
  expect_near("(A*B)(1,1)", to_double(C(1, 1)), 154.0); // 4*8+5*10+6*12

  Vector x = {Scalar{1.0}, Scalar{0.0}, Scalar{1.0}}; // size 3, matches A.cols
  expect_vector_near("A*x", A * x, {Scalar{4.0}, Scalar{10.0}});

  Vector y = {Scalar{1.0}, Scalar{1.0}}; // size 2, matches A.rows
  expect_vector_near("y*A", y * A, {Scalar{5.0}, Scalar{7.0}, Scalar{9.0}});

  return nanojax_test::report();
}
