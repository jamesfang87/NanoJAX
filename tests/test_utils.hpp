#pragma once

#include "include/ad/scalar.hpp"

#include <cmath>
#include <iostream>
#include <string>

namespace nanojax_test {

inline int failures = 0;

inline void expect_near(const std::string &name, double got, double want,
                        double tol = 1e-6) {
  if (std::abs(got - want) > tol) {
    std::cout << "FAIL " << name << ": got " << got << ", want " << want
              << std::endl;
    ++failures;
  } else {
    std::cout << "ok   " << name << ": " << got << std::endl;
  }
}

inline void expect_true(const std::string &name, bool cond) {
  if (!cond) {
    std::cout << "FAIL " << name << std::endl;
    ++failures;
  } else {
    std::cout << "ok   " << name << std::endl;
  }
}

// Compares a Vector's entries against expected values elementwise, by their
// double value rather than by structural equality, so a Vector of Variables
// (the result of a forward pass under a Trace) can be checked against a
// Vector of plain Real constants without extra conversion at the call site.
inline void expect_vector_near(const std::string &name, const Vector &got,
                               const Vector &want, double tol = 1e-6) {
  expect_true(name + " size", got.size() == want.size());
  for (size_t i = 0; i < want.size() && i < got.size(); ++i) {
    expect_near(name + "[" + std::to_string(i) + "]", to_double(got[i]),
                to_double(want[i]), tol);
  }
}

inline int report() {
  std::cout << (failures == 0 ? "ALL PASS" : "SOME FAILED") << std::endl;
  return failures;
}

} // namespace nanojax_test
