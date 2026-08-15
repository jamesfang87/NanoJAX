#include "include/ad/grad.hpp"
#include "include/ad/scalar.hpp"

#include "include/nn/layers/activation/activations.hpp"

#include <iostream>
#include <ostream>

Scalar f(const Scalar &x) { return x * x * x; }

int main() {
  auto df = grad(f);
  Scalar g = df(3.0);
  std::cout << "df/dx at x = 3.0: " << to_double(g) << std::endl;

  auto drelu = grad(relu);
  std::cout << "drelu/dx at x = -2.0: " << to_double(drelu(-2.0)) << std::endl;
  std::cout << "drelu/dx at x = 2.0: " << to_double(drelu(2.0)) << std::endl;
}
