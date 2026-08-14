#include "include/grad.hpp"

#include <iostream>

int main() {
  auto f = [](const Var &x, const Var &y) { return x * x * x; };

  auto df = grad(f);
  auto gradient = df(2.0, 1.0);

  std::cout << "df/dx at x=2.0, y=1.0: " << gradient[0] << std::endl;
  std::cout << "df/dy at x=2.0, y=1.0: " << gradient[1] << std::endl;
}
