#include "ad/grad.hpp"
#include "ad/scalar.hpp"

#include "nn/layers/activation/activations.hpp"

#include <iostream>
#include <ostream>

Scalar f(const Scalar &x) { return pow(x, 3.0); }

int main() {
  auto df = grad(f);
  Scalar g = df(3.0);
  std::cout << "df/dx at x = 3.0: " << g << std::endl;

  auto drelu = grad(ReLU);
  std::cout << "drelu/dx at x = -2.0: " << drelu(-2.0) << std::endl;
  std::cout << "drelu/dx at x = 2.0: " << drelu(2.0) << std::endl;
}
