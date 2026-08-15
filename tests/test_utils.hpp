#pragma once

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

inline int report() {
  std::cout << (failures == 0 ? "ALL PASS" : "SOME FAILED") << std::endl;
  return failures;
}

} // namespace nanojax_test
