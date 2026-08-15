#include "include/grad.hpp"

#include <iostream>

int main() {
  auto f = [](const Var &x) { return pow(x, 3); };

  auto relu = [](const Var &x) {
    if (x > 0.0) {
      return x;
    } else {
      return tape.add_constant(0.0);
    }
  };

  auto df = grad(f);
  auto gradient1 = df(2.0);
  auto gradient2 = df(-2.0);

  std::cout << "df/dx at x=2.0: " << gradient1[0] << std::endl;
  std::cout << "df/dy at x=-2.0: " << gradient2[0] << std::endl;
}
