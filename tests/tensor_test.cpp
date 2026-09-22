// tests/tensor_test.cpp

#include "tensor/io.hpp"
#include "tensor/ops.hpp"
#include "tensor/tensor.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>

using namespace nanojax;

namespace {

int failures = 0;
int checks = 0;

void check(bool condition, const char* expression, const char* file, int line) {
    ++checks;
    if (!condition) {
        ++failures;
        std::cerr << file << ':' << line << ": CHECK failed: " << expression << '\n';
    }
}

#define CHECK(expr) check(static_cast<bool>(expr), #expr, __FILE__, __LINE__)

template <typename Exception, typename Callable>
void check_throws(const Callable& callable, const char* description) {
    ++checks;
    try {
        callable();
        ++failures;
        std::cerr << "expected an exception from " << description << '\n';
    } catch (const Exception&) {
    } catch (...) {
        ++failures;
        std::cerr << "wrong exception type from " << description << '\n';
    }
}

Tensor<float> sample_2x3() {
    auto tensor = Tensor<float>::value({2, 3}, 0.0f);
    tensor.at({0, 0}) = 1.0f;
    tensor.at({0, 1}) = 2.0f;
    tensor.at({0, 2}) = 3.0f;
    tensor.at({1, 0}) = 4.0f;
    tensor.at({1, 1}) = 5.0f;
    tensor.at({1, 2}) = 6.0f;
    return tensor;
}

void test_factories_and_metadata() {
    auto a = Tensor<float>::value({2, 3}, 4.0f);
    CHECK(a.ndim() == 2);
    CHECK(a.shape()[0] == 2);
    CHECK(a.shape()[1] == 3);
    CHECK(a.numel() == 6);
    CHECK(!a.empty());
    CHECK(a.device() == Device::CPU);
    CHECK(a.is_contiguous());
    CHECK(a.get_dtype() == typeid(float));

    auto z = Tensor<int>::zeros({2, 2});
    CHECK(z.at({0, 0}) == 0 && z.at({1, 1}) == 0);

    auto o = Tensor<int>::ones({3});
    CHECK(o.at({0}) == 1 && o.at({2}) == 1);

    auto scalar = Tensor<double>::value({}, 7.5);
    CHECK(scalar.ndim() == 0);
    CHECK(scalar.empty());
    CHECK(scalar.numel() == 1);
    CHECK(scalar.is_contiguous());
    CHECK(scalar.item() == 7.5);
}

void test_item_and_at() {
    auto a = sample_2x3();
    CHECK(a.at({0, 0}) == 1.0f);
    CHECK(a.at({1, 2}) == 6.0f);

    a.at({1, 2}) = 60.0f;
    CHECK(a.at({1, 2}) == 60.0f);

    const auto& ca = a;
    CHECK(ca.at({0, 1}) == 2.0f);
    static_assert(std::is_same_v<decltype(a.at({0, 0})), float&>);
    static_assert(std::is_same_v<decltype(ca.at({0, 0})), const float&>);

    CHECK(Tensor<float>::value({1}, 3.0f).item() == 3.0f);

    check_throws<std::logic_error>([&] { (void)a.item(); }, "item on a multi-element tensor");
    check_throws<std::out_of_range>([&] { (void)a.at({2, 0}); }, "at row out of range");
    check_throws<std::out_of_range>([&] { (void)a.at({0, 3}); }, "at column out of range");
    check_throws<std::invalid_argument>([&] { (void)a.at({1}); }, "at with too few indices");
    check_throws<std::invalid_argument>([&] { (void)a.at({0, 0, 0}); }, "at with too many indices");
}

void test_data_pointer() {
    auto a = sample_2x3();
    CHECK(a.data()[0] == 1.0f);
    CHECK(a.data()[5] == 6.0f);

    const auto& ca = a;
    static_assert(std::is_same_v<decltype(a.data()), float*>);
    static_assert(std::is_same_v<decltype(ca.data()), const float*>);
    CHECK(ca.data()[3] == 4.0f);
}

void test_reshape_and_flatten() {
    auto a = sample_2x3();

    auto r = a.reshape({3, 2});
    CHECK(r.ndim() == 2 && r.shape()[0] == 3 && r.shape()[1] == 2);
    CHECK(r.at({1, 0}) == 3.0f);
    CHECK(r.at({2, 1}) == 6.0f);
    CHECK(r.is_contiguous());

    auto f = a.flatten();
    CHECK(f.ndim() == 1 && f.shape()[0] == 6);
    CHECK(f.at({0}) == 1.0f && f.at({5}) == 6.0f);

    check_throws<std::logic_error>([&] { (void)a.reshape({4}); },
                                   "reshape with a wrong element count");

    auto t = a.transpose();
    auto tr = t.reshape({6});
    CHECK(tr.is_contiguous());
    CHECK(tr.at({0}) == 1.0f && tr.at({1}) == 4.0f && tr.at({5}) == 6.0f);

    auto tf = t.flatten();
    CHECK(tf.is_contiguous());
    CHECK(tf.at({1}) == 4.0f && tf.at({5}) == 6.0f);
}

void test_contiguity() {
    CHECK(Tensor<float>::value({2, 3}, 0.0f).is_contiguous());
    CHECK(Tensor<float>::value({6}, 0.0f).is_contiguous());
    CHECK(Tensor<float>::value({}, 0.0f).is_contiguous());
    CHECK(Tensor<float>::value({1, 3, 1}, 0.0f).is_contiguous());
    CHECK(Tensor<float>::value({1, 3, 1}, 0.0f).squeeze().is_contiguous());

    auto a = sample_2x3();
    CHECK(!a.transpose().is_contiguous());
    CHECK(a.transpose().transpose().is_contiguous());
    CHECK(Tensor<float>::value({1, 4}, 0.0f).transpose().is_contiguous());
    CHECK(a.permute({0, 1}).is_contiguous());
    CHECK(!a.permute({1, 0}).is_contiguous());
}

void test_transpose_and_permute() {
    auto a = sample_2x3();
    auto t = a.transpose();
    CHECK(t.ndim() == 2 && t.shape()[0] == 3 && t.shape()[1] == 2);
    CHECK(t.at({0, 0}) == 1.0f && t.at({0, 1}) == 4.0f);
    CHECK(t.at({1, 0}) == 2.0f && t.at({1, 1}) == 5.0f);
    CHECK(t.at({2, 0}) == 3.0f && t.at({2, 1}) == 6.0f);

    t.at({2, 1}) = 60.0f;
    CHECK(a.at({1, 2}) == 60.0f);

    auto b = Tensor<float>::value({2, 3, 4}, 0.0f);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                b.at({i, j, k}) = static_cast<float>(i * 100 + j * 10 + k);
            }
        }
    }

    auto p = b.permute({1, 0, 2});
    CHECK(p.ndim() == 3 && p.shape()[0] == 3 && p.shape()[1] == 2 && p.shape()[2] == 4);
    CHECK(p.at({0, 0, 0}) == b.at({0, 0, 0}));
    CHECK(p.at({2, 1, 3}) == b.at({1, 2, 3}));

    auto reversed = b.permute({2, 1, 0});
    auto transposed = b.transpose();
    CHECK(reversed.shape()[0] == transposed.shape()[0]);
    CHECK(reversed.shape()[1] == transposed.shape()[1]);
    CHECK(reversed.shape()[2] == transposed.shape()[2]);
    CHECK(reversed.at({1, 2, 0}) == transposed.at({1, 2, 0}));

    check_throws<std::invalid_argument>([&] { (void)a.permute({0}); }, "permute with too few axes");
    check_throws<std::invalid_argument>([&] { (void)a.permute({0, 0}); },
                                        "permute with duplicate axes");
    check_throws<std::invalid_argument>([&] { (void)a.permute({0, 2}); },
                                        "permute with an out-of-range axis");
}

void test_squeeze() {
    auto a = Tensor<float>::value({1, 3, 1}, 0.0f);
    a.at({0, 0, 0}) = 1.0f;
    a.at({0, 1, 0}) = 2.0f;
    a.at({0, 2, 0}) = 3.0f;

    auto s = a.squeeze();
    CHECK(s.ndim() == 1 && s.shape()[0] == 3);
    CHECK(s.at({0}) == 1.0f && s.at({2}) == 3.0f);
    CHECK(s.is_contiguous());

    auto b = Tensor<float>::value({2, 1, 3}, 5.0f);
    auto s2 = b.squeeze();
    CHECK(s2.ndim() == 2 && s2.shape()[0] == 2 && s2.shape()[1] == 3);
    CHECK(s2.at({1, 2}) == 5.0f);

    auto scalar = Tensor<float>::value({1, 1}, 9.0f).squeeze();
    CHECK(scalar.ndim() == 0 && scalar.numel() == 1 && scalar.item() == 9.0f);

    // Single-axis squeeze drops only the requested size-1 dimension.
    auto c = Tensor<float>::value({2, 1, 3}, 0.0f);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t k = 0; k < 3; ++k) {
            c.at({i, 0, k}) = static_cast<float>(i * 3 + k + 1);
        }
    }
    auto s3 = c.squeeze(1);
    CHECK(s3.ndim() == 2 && s3.shape()[0] == 2 && s3.shape()[1] == 3);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t k = 0; k < 3; ++k) {
            CHECK(s3.at({i, k}) == c.at({i, 0, k}));
        }
    }
    CHECK(s3.is_contiguous());

    auto s4 = a.squeeze(0);
    CHECK(s4.ndim() == 2 && s4.shape()[0] == 3 && s4.shape()[1] == 1);
    CHECK(s4.at({0, 0}) == 1.0f && s4.at({2, 0}) == 3.0f);

    check_throws<std::invalid_argument>([&] { (void)b.squeeze(0); },
                                        "squeeze on an axis that is not size one");
    check_throws<std::out_of_range>([&] { (void)b.squeeze(3); },
                                    "squeeze with an out-of-range axis");
}

void test_unsqueeze() {
    auto a = sample_2x3();

    auto middle = a.unsqueeze(1);
    CHECK(middle.ndim() == 3);
    CHECK(middle.shape()[0] == 2 && middle.shape()[1] == 1 && middle.shape()[2] == 3);
    CHECK(middle.numel() == 6);
    CHECK(middle.is_contiguous());
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            CHECK(middle.at({i, 0, j}) == a.at({i, j}));
        }
    }

    middle.at({1, 0, 2}) = 99.0f;
    CHECK(a.at({1, 2}) == 99.0f);

    auto front = a.unsqueeze(0);
    CHECK(front.shape()[0] == 1 && front.shape()[1] == 2 && front.shape()[2] == 3);
    CHECK(front.at({0, 1, 2}) == 99.0f);

    auto back = a.unsqueeze(2);
    CHECK(back.shape()[0] == 2 && back.shape()[1] == 3 && back.shape()[2] == 1);
    CHECK(back.at({1, 2, 0}) == 99.0f);

    check_throws<std::out_of_range>([&] { (void)a.unsqueeze(3); },
                                    "unsqueeze with an out-of-range axis");
}

void test_broadcast_to() {
    auto a = sample_2x3();

    auto same = a.broadcast({2, 3});
    CHECK(same.shape()[0] == 2 && same.shape()[1] == 3);
    CHECK(same.strides()[0] == 3 && same.strides()[1] == 1);
    CHECK(same.numel() == 6);
    CHECK(same.data() == a.data());
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            CHECK(same.at({i, j}) == a.at({i, j}));
        }
    }

    auto row = Tensor<float>::value({1, 3}, 0.0f);
    row.at({0, 0}) = 1.0f;
    row.at({0, 1}) = 2.0f;
    row.at({0, 2}) = 3.0f;
    auto rows = row.broadcast({2, 3});
    CHECK(rows.shape()[0] == 2 && rows.shape()[1] == 3);
    CHECK(rows.strides()[0] == 0 && rows.strides()[1] == 1);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            CHECK(rows.at({i, j}) == row.at({0, j}));
        }
    }

    rows.at({1, 2}) = 30.0f;
    CHECK(row.at({0, 2}) == 30.0f);

    auto vec = Tensor<float>::value({3}, 0.0f);
    vec.at({0}) = 4.0f;
    vec.at({1}) = 5.0f;
    vec.at({2}) = 6.0f;
    auto grid = vec.broadcast({2, 3});
    CHECK(grid.shape()[0] == 2 && grid.shape()[1] == 3);
    CHECK(grid.strides()[0] == 0 && grid.strides()[1] == 1);
    CHECK(grid.at({0, 0}) == 4.0f && grid.at({1, 1}) == 5.0f && grid.at({1, 2}) == 6.0f);

    auto t = a.transpose();           // (3, 2), strides (1, 3), non-contiguous
    auto tb = t.broadcast({2, 3, 2}); // a leading dimension is added
    CHECK(tb.shape()[0] == 2 && tb.shape()[1] == 3 && tb.shape()[2] == 2);
    CHECK(tb.strides()[0] == 0 && tb.strides()[1] == 1 && tb.strides()[2] == 3);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 2; ++k) {
                CHECK(tb.at({i, j, k}) == a.at({k, j}));
            }
        }
    }

    check_throws<std::logic_error>([&] { (void)a.broadcast({3}); },
                                   "broadcast_to with a lower rank");
    check_throws<std::logic_error>([&] { (void)a.broadcast({4, 3}); },
                                   "broadcast_to with a mismatched dimension");
    check_throws<std::logic_error>([&] { (void)vec.broadcast({1}); },
                                   "broadcast_to that would shrink a dimension");
}

void test_clone() {
    auto a = sample_2x3();
    auto c = a.clone();

    CHECK(c.shape()[0] == 2 && c.shape()[1] == 3);
    CHECK(c.numel() == 6);
    CHECK(c.data() != a.data());
    CHECK(c.is_contiguous());
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            CHECK(c.at({i, j}) == a.at({i, j}));
        }
    }

    c.at({0, 0}) = 111.0f;
    CHECK(a.at({0, 0}) == 1.0f);

    auto t = a.transpose();
    auto tc = t.clone();
    CHECK(tc.shape()[0] == 3 && tc.shape()[1] == 2);
    CHECK(!tc.is_contiguous());
    CHECK(tc.data() != t.data());
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            CHECK(tc.at({i, j}) == t.at({i, j}));
        }
    }
}

void test_contiguous() {
    auto a = sample_2x3();
    CHECK(a.contiguous().data() == a.data());
    CHECK(a.contiguous(false).data() == a.data());
    CHECK(a.contiguous(true).data() != a.data());

    auto t = a.transpose();
    CHECK(!t.is_contiguous());
    CHECK(t.contiguous(false).data() != t.data());
    CHECK(t.contiguous(false).is_contiguous());

    auto c = t.contiguous();
    CHECK(c.shape()[0] == 3 && c.shape()[1] == 2);
    CHECK(c.is_contiguous());
    CHECK(c.numel() == 6);
    CHECK(c.data() != t.data());
    CHECK(c.at({0, 0}) == 1.0f && c.at({0, 1}) == 4.0f);
    CHECK(c.at({1, 0}) == 2.0f && c.at({1, 1}) == 5.0f);
    CHECK(c.at({2, 0}) == 3.0f && c.at({2, 1}) == 6.0f);

    c.at({2, 1}) = 60.0f;
    CHECK(a.at({1, 2}) == 6.0f);

    auto b = Tensor<float>::value({2, 3, 4}, 0.0f);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                b.at({i, j, k}) = static_cast<float>(i * 100 + j * 10 + k);
            }
        }
    }
    auto p = b.permute({1, 0, 2}).contiguous();
    CHECK(p.is_contiguous());
    CHECK(p.shape()[0] == 3 && p.shape()[1] == 2 && p.shape()[2] == 4);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                CHECK(p.at({i, j, k}) == b.at({j, i, k}));
            }
        }
    }

    auto scalar = Tensor<double>::value({}, 2.5);
    CHECK(scalar.contiguous().at({}) == 2.5);

    auto sq = Tensor<float>::value({1, 3, 1}, 0.0f);
    sq.at({0, 1, 0}) = 7.0f;
    CHECK(sq.contiguous().is_contiguous());
    CHECK(sq.contiguous().at({0, 1, 0}) == 7.0f);
}

void test_scalar_arithmetic() {
    auto a = sample_2x3();
    auto plus = a + 2.0f;
    auto rplus = 2.0f + a;
    auto minus = a - 2.0f;
    auto rminus = 2.0f - a;
    auto times = a * 2.0f;
    auto rtimes = 2.0f * a;
    auto divide = a / 2.0f;
    auto rdivide = 6.0f / a;

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            const float x = a.at({i, j});
            CHECK(plus.at({i, j}) == x + 2.0f);
            CHECK(rplus.at({i, j}) == 2.0f + x);
            CHECK(minus.at({i, j}) == x - 2.0f);
            CHECK(rminus.at({i, j}) == 2.0f - x);
            CHECK(times.at({i, j}) == x * 2.0f);
            CHECK(rtimes.at({i, j}) == 2.0f * x);
            CHECK(divide.at({i, j}) == x / 2.0f);
            CHECK(rdivide.at({i, j}) == 6.0f / x);
        }
    }
    CHECK(plus.data() != a.data());
    CHECK(rminus.data() != a.data());
    CHECK(a.at({0, 0}) == 1.0f);

    auto t = a.transpose();
    auto c = t - 2.0f;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            CHECK(c.at({i, j}) == a.at({j, i}) - 2.0f);
        }
    }
    CHECK(a.at({1, 2}) == 6.0f);

    auto scalar = Tensor<float>::value({}, 4.0f);
    CHECK((scalar + 1.0f).item() == 5.0f);
    CHECK((1.0f - scalar).item() == -3.0f);
    CHECK((8.0f / scalar).item() == 2.0f);
    CHECK(scalar.item() == 4.0f);
}

void test_tensor_arithmetic() {
    auto a = sample_2x3();

    auto sum = a + a;
    CHECK(sum.shape()[0] == 2 && sum.shape()[1] == 3);
    CHECK(sum.is_contiguous());
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            CHECK(sum.at({i, j}) == a.at({i, j}) + a.at({i, j}));
        }
    }
    CHECK(a.at({0, 0}) == 1.0f);

    auto row = Tensor<float>::value({1, 3}, 0.0f);
    row.at({0, 0}) = 10.0f;
    row.at({0, 1}) = 20.0f;
    row.at({0, 2}) = 30.0f;
    auto rows = row + a;
    CHECK(rows.shape()[0] == 2 && rows.shape()[1] == 3);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            CHECK(rows.at({i, j}) == row.at({0, j}) + a.at({i, j}));
        }
    }

    auto vec = Tensor<float>::value({3}, 0.0f);
    vec.at({0}) = 100.0f;
    vec.at({1}) = 200.0f;
    vec.at({2}) = 300.0f;
    auto grid = a + vec;
    CHECK(grid.shape()[0] == 2 && grid.shape()[1] == 3);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            CHECK(grid.at({i, j}) == a.at({i, j}) + vec.at({j}));
        }
    }

    auto t = a.transpose();
    auto mixed = t + t;
    CHECK(mixed.shape()[0] == 3 && mixed.shape()[1] == 2);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            CHECK(mixed.at({i, j}) == a.at({j, i}) + a.at({j, i}));
        }
    }

    auto diff = row - a;
    auto prod = row * a;
    auto quot = row / a;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            const float r = row.at({0, j});
            const float x = a.at({i, j});
            CHECK(diff.at({i, j}) == r - x);
            CHECK(prod.at({i, j}) == r * x);
            CHECK(quot.at({i, j}) == r / x);
        }
    }

    auto col = Tensor<float>::value({3, 1}, 0.0f);
    auto r4 = Tensor<float>::value({1, 4}, 0.0f);
    for (size_t i = 0; i < 3; ++i) {
        col.at({i, 0}) = static_cast<float>(i + 1);
    }
    for (size_t j = 0; j < 4; ++j) {
        r4.at({0, j}) = static_cast<float>(j + 1);
    }

    auto outer_mul = col * r4;
    auto outer_sub = col - r4;
    CHECK(outer_mul.shape()[0] == 3 && outer_mul.shape()[1] == 4);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            CHECK(outer_mul.at({i, j}) == col.at({i, 0}) * r4.at({0, j}));
            CHECK(outer_sub.at({i, j}) == col.at({i, 0}) - r4.at({0, j}));
        }
    }

    auto scalar = Tensor<float>::value({}, 10.0f);
    auto plus_scalar = a + scalar;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            CHECK(plus_scalar.at({i, j}) == a.at({i, j}) + 10.0f);
        }
    }

    auto x = Tensor<float>::value({2, 1, 3}, 0.0f);
    auto y = Tensor<float>::value({1, 4, 1}, 0.0f);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t k = 0; k < 3; ++k) {
            x.at({i, 0, k}) = static_cast<float>(i * 3 + k + 1);
        }
    }
    for (size_t j = 0; j < 4; ++j) {
        y.at({0, j, 0}) = static_cast<float>(j + 1) * 100.0f;
    }
    auto cube = x + y;
    CHECK(cube.shape()[0] == 2 && cube.shape()[1] == 4 && cube.shape()[2] == 3);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            for (size_t k = 0; k < 3; ++k) {
                CHECK(cube.at({i, j, k}) == x.at({i, 0, k}) + y.at({0, j, 0}));
            }
        }
    }

    auto m_minus_v = a - vec;
    auto v_minus_m = vec - a;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            CHECK(m_minus_v.at({i, j}) == a.at({i, j}) - vec.at({j}));
            CHECK(v_minus_m.at({i, j}) == vec.at({j}) - a.at({i, j}));
        }
    }

    auto empty = Tensor<float>::zeros({0, 3});
    auto empty_sum = empty + empty;
    CHECK(empty_sum.numel() == 0);
    CHECK(empty_sum.shape()[0] == 0 && empty_sum.shape()[1] == 3);

    auto bad = Tensor<float>::value({4, 3}, 0.0f);
    check_throws<std::logic_error>([&] { (void)(a + bad); }, "operator+ with incompatible shapes");
    check_throws<std::logic_error>([&] { (void)(a / bad); }, "operator/ with incompatible shapes");
}

void test_inplace_arithmetic() {
    auto a = sample_2x3();
    auto* base = a.data();

    a += a;
    CHECK(a.data() == base);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            CHECK(a.at({i, j}) == 2.0f * static_cast<float>(i * 3 + j + 1));
        }
    }

    a -= a;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            CHECK(a.at({i, j}) == 0.0f);
        }
    }

    auto m = sample_2x3();
    auto row = Tensor<float>::value({1, 3}, 10.0f);
    m += row;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            CHECK(m.at({i, j}) == static_cast<float>(i * 3 + j + 1) + 10.0f);
        }
    }
    CHECK(row.at({0, 0}) == 10.0f);

    auto vec = Tensor<float>::value({3}, 0.0f);
    vec.at({0}) = 2.0f;
    vec.at({1}) = 3.0f;
    vec.at({2}) = 5.0f;
    m *= vec;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            CHECK(m.at({i, j}) == (static_cast<float>(i * 3 + j + 1) + 10.0f) * vec.at({j}));
        }
    }
    CHECK(vec.at({2}) == 5.0f);

    auto scalar = Tensor<float>::value({}, 2.0f);
    m /= scalar;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            CHECK(m.at({i, j}) == (static_cast<float>(i * 3 + j + 1) + 10.0f) * vec.at({j}) / 2.0f);
        }
    }

    auto b = sample_2x3();
    auto tb = b.transpose();
    tb += tb;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            CHECK(b.at({j, i}) == 2.0f * static_cast<float>(j * 3 + i + 1));
        }
    }

    auto short_vec = Tensor<float>::value({3}, 0.0f);
    auto bad = Tensor<float>::value({4, 3}, 0.0f);
    check_throws<std::logic_error>([&] { a += bad; }, "operator+= with incompatible shapes");
    check_throws<std::logic_error>([&] { short_vec += b; },
                                   "operator+= with a higher-rank operand");

    auto c = sample_2x3();
    c /= 2.0f;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            CHECK(c.at({i, j}) == static_cast<float>(i * 3 + j + 1) / 2.0f);
        }
    }

    auto d = sample_2x3();
    auto td = d.transpose();
    td /= 2.0f;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            CHECK(d.at({j, i}) == static_cast<float>(j * 3 + i + 1) / 2.0f);
        }
    }

    auto e = sample_2x3();
    e += 1.0f;
    e -= 2.0f;
    e *= 3.0f;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            CHECK(e.at({i, j}) == (static_cast<float>(i * 3 + j + 1) + 1.0f - 2.0f) * 3.0f);
        }
    }

    auto f = sample_2x3();
    auto tf = f.transpose();
    tf += 5.0f;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            CHECK(f.at({j, i}) == static_cast<float>(j * 3 + i + 1) + 5.0f);
        }
    }
}

void test_elementwise_math() {
    auto a = sample_2x3();
    auto e = exp(a);
    auto l = log(e);
    auto s = sqrt(a);
    auto p = pow(a, 2.0f);
    auto ab = abs(a - 4.0f);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            const float x = a.at({i, j});
            CHECK(std::abs(e.at({i, j}) - std::exp(x)) < 1e-5f);
            CHECK(std::abs(l.at({i, j}) - x) < 1e-5f);
            CHECK(std::abs(s.at({i, j}) - std::sqrt(x)) < 1e-5f);
            CHECK(std::abs(p.at({i, j}) - x * x) < 1e-5f);
            CHECK(ab.at({i, j}) == std::abs(x - 4.0f));
        }
    }
    CHECK(a.at({0, 0}) == 1.0f);
    CHECK(e.data() != a.data());

    auto t = a.transpose();
    auto et = exp(t);
    CHECK(std::abs(et.at({0, 1}) - std::exp(a.at({1, 0}))) < 1e-5f);
    CHECK(std::abs(et.at({2, 0}) - std::exp(a.at({0, 2}))) < 1e-5f);
}

void test_reductions() {
    auto a = sample_2x3(); // 1 .. 6
    auto total = sum(a);
    CHECK(total.ndim() == 0);
    CHECK(total.numel() == 1);
    CHECK(total.item() == 21.0f);

    CHECK(sum(a.transpose()).item() == 21.0f);
    CHECK(sum(Tensor<float>::value({}, 4.0f)).item() == 4.0f);
    CHECK(sum(Tensor<float>::value({2, 2}, 3.0f)).item() == 12.0f);

    auto cube = Tensor<float>::value({2, 2, 2}, 0.0f);
    float n = 1.0f;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            for (size_t k = 0; k < 2; ++k) {
                cube.at({i, j, k}) = n++;
            }
        }
    }
    CHECK(sum(cube).item() == 36.0f);

    auto neg = a - 4.0f; // -3 .. 2
    CHECK(sum(neg).item() == -3.0f);

    auto empty = Tensor<float>::zeros({0, 3});
    CHECK(sum(empty).item() == 0.0f);
    check_throws<std::invalid_argument>([&] { (void)max(empty); }, "max of an empty tensor");
    check_throws<std::invalid_argument>([&] { (void)min(empty); }, "min of an empty tensor");
    check_throws<std::invalid_argument>([&] { (void)mean(empty); }, "mean of an empty tensor");

    CHECK(max(a).item() == 6.0f);
    CHECK(min(a).item() == 1.0f);
    CHECK(mean(a).item() == 3.5f);
    CHECK(max(a.transpose()).item() == 6.0f);
    CHECK(min(neg).item() == -3.0f);
    CHECK(max(neg).item() == 2.0f);

    auto col_sums = sum(a, 0);
    CHECK(col_sums.ndim() == 1 && col_sums.shape()[0] == 3);
    CHECK(col_sums.at({0}) == 5.0f && col_sums.at({1}) == 7.0f && col_sums.at({2}) == 9.0f);

    auto row_sums = sum(a, 1);
    CHECK(row_sums.ndim() == 1 && row_sums.shape()[0] == 2);
    CHECK(row_sums.at({0}) == 6.0f && row_sums.at({1}) == 15.0f);

    auto t = a.transpose(); // (3, 2), non-contiguous
    CHECK(sum(t, 1).at({0}) == 5.0f && sum(t, 1).at({2}) == 9.0f);
    CHECK(sum(t, 0).at({0}) == 6.0f && sum(t, 0).at({1}) == 15.0f);

    auto mid = sum(cube, 1);
    CHECK(mid.shape()[0] == 2 && mid.shape()[1] == 2);
    CHECK(mid.at({0, 0}) == 4.0f && mid.at({0, 1}) == 6.0f);
    CHECK(mid.at({1, 0}) == 12.0f && mid.at({1, 1}) == 14.0f);

    auto last = sum(cube, 2);
    CHECK(last.shape()[0] == 2 && last.shape()[1] == 2);
    CHECK(last.at({0, 0}) == 3.0f && last.at({0, 1}) == 7.0f);
    CHECK(last.at({1, 0}) == 11.0f && last.at({1, 1}) == 15.0f);

    auto vec_sum = sum(Tensor<float>::value({3}, 2.0f), 0);
    CHECK(vec_sum.ndim() == 0 && vec_sum.item() == 6.0f);

    check_throws<std::out_of_range>([&] { (void)sum(a, 2); }, "sum with an out-of-range axis");

    auto col_means = mean(a, 0);
    CHECK(col_means.shape()[0] == 3);
    CHECK(col_means.at({0}) == 2.5f && col_means.at({1}) == 3.5f && col_means.at({2}) == 4.5f);

    auto row_means = mean(a, 1);
    CHECK(row_means.shape()[0] == 2);
    CHECK(row_means.at({0}) == 2.0f && row_means.at({1}) == 5.0f);

    CHECK(mean(t, 0).at({1}) == 5.0f);
    CHECK(mean(cube, 1).at({1, 0}) == 6.0f);

    check_throws<std::out_of_range>([&] { (void)mean(a, 2); }, "mean with an out-of-range axis");
    check_throws<std::invalid_argument>([&] { (void)mean(Tensor<float>::zeros({2, 0, 3}), 1); },
                                        "mean over an empty axis");
}

void test_allclose() {
    auto a = sample_2x3();

    CHECK(a.allclose(a, 0.0));
    CHECK(a.allclose(a.clone(), 0.0));

    auto close = a + 1e-4f;
    CHECK(a.allclose(close, 1e-3));
    CHECK(!a.allclose(close, 1e-6));

    auto other = sample_2x3();
    other.at({1, 2}) = 7.0f;
    CHECK(!a.allclose(other, 0.5));
    CHECK(a.allclose(other, 1.0));

    auto t = a.transpose();
    CHECK(a.allclose(t.transpose(), 0.0));
    CHECK(a.allclose(t.transpose().contiguous(), 0.0));

    CHECK(Tensor<float>::zeros({0, 3}).allclose(Tensor<float>::zeros({0, 3}), 0.0));

    auto wrong = Tensor<float>::zeros({3, 2});
    check_throws<std::logic_error>([&] { (void)a.allclose(wrong, 0.0); },
                                   "allclose with mismatched shapes");
    check_throws<std::invalid_argument>([&] { (void)a.allclose(a, -1.0); },
                                        "allclose with a negative tolerance");
}

void test_activations() {
    auto neg = sample_2x3() - 4.0f; // -3 .. 2

    auto r = relu(neg);
    auto s = sigmoid(neg);
    auto t = tanh(neg);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            const float x = neg.at({i, j});
            CHECK(r.at({i, j}) == (x > 0.0f ? x : 0.0f));
            CHECK(std::abs(s.at({i, j}) - 1.0f / (1.0f + std::exp(-x))) < 1e-6f);
            CHECK(std::abs(t.at({i, j}) - std::tanh(x)) < 1e-6f);
            CHECK(s.at({i, j}) > 0.0f && s.at({i, j}) < 1.0f);
            CHECK(t.at({i, j}) > -1.0f && t.at({i, j}) < 1.0f);
        }
    }
    CHECK(std::abs(s.at({1, 0}) - 0.5f) < 1e-6f); // neg(1,0) == 0
    CHECK(neg.at({0, 0}) == -3.0f);               // source untouched

    auto from_t = tanh(neg.transpose());
    CHECK(std::abs(from_t.at({0, 1}) - std::tanh(neg.at({1, 0}))) < 1e-6f);
}

void test_printing() {
    auto a = sample_2x3();
    std::ostringstream os;
    os << a;
    const std::string s = os.str();
    CHECK(s.find("shape=[2, 3]") != std::string::npos);
    CHECK(s.find("[1, 2, 3]") != std::string::npos);
    CHECK(s.find("[4, 5, 6]") != std::string::npos);

    std::ostringstream os_t;
    os_t << a.transpose();
    CHECK(os_t.str().find("[1, 4]") != std::string::npos);
    CHECK(os_t.str().find("[3, 6]") != std::string::npos);

    auto scalar = Tensor<int>::value({}, 7);
    std::ostringstream os_s;
    os_s << scalar;
    CHECK(os_s.str().find("shape=[]") != std::string::npos);
    CHECK(os_s.str().find('7') != std::string::npos);
}

void test_span_factories() {
    std::array<size_t, 2> extents{2, 3};
    std::span<const size_t> shape{extents};

    auto z = Tensor<int>::zeros(shape);
    auto o = Tensor<int>::ones(shape);
    auto v = Tensor<float>::value(shape, 2.5f);

    CHECK(z.ndim() == 2 && z.shape()[0] == 2 && z.shape()[1] == 3);
    CHECK(z.at({1, 2}) == 0);
    CHECK(o.at({1, 2}) == 1);
    CHECK(v.at({1, 2}) == 2.5f);
}

void test_matmul() {
    auto a = Tensor<float>::value({2, 3}, 0.0f);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            a.at({i, j}) = static_cast<float>(i * 3 + j + 1);
        }
    }
    auto b = Tensor<float>::value({3, 2}, 0.0f);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            b.at({i, j}) = static_cast<float>(i * 2 + j + 7);
        }
    }

    auto c = matmul(a, b);
    CHECK(c.shape()[0] == 2 && c.shape()[1] == 2);
    CHECK(c.at({0, 0}) == 58.0f && c.at({0, 1}) == 64.0f);
    CHECK(c.at({1, 0}) == 139.0f && c.at({1, 1}) == 154.0f);

    auto id = Tensor<float>::value({2, 2}, 0.0f);
    id.at({0, 0}) = 1.0f;
    id.at({1, 1}) = 1.0f;
    auto ia = matmul(id, a);
    CHECK(ia.shape()[0] == 2 && ia.shape()[1] == 3);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            CHECK(ia.at({i, j}) == a.at({i, j}));
        }
    }

    auto bt = b.transpose(); // (2, 3), non-contiguous
    auto c2 = matmul(bt, b);
    CHECK(c2.shape()[0] == 2 && c2.shape()[1] == 2);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            float expected = 0.0f;
            for (size_t k = 0; k < 3; ++k) {
                expected += bt.at({i, k}) * b.at({k, j});
            }
            CHECK(c2.at({i, j}) == expected);
        }
    }

    auto d1 = Tensor<double>::value({1, 2}, 0.0);
    auto d2 = Tensor<double>::value({2, 1}, 0.0);
    d1.at({0, 0}) = 2.0;
    d1.at({0, 1}) = 3.0;
    d2.at({0, 0}) = 4.0;
    d2.at({1, 0}) = 5.0;
    CHECK(matmul(d1, d2).at({0, 0}) == 23.0);

    auto iA = Tensor<int>::value({2, 3}, 0);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            iA.at({static_cast<size_t>(i), static_cast<size_t>(j)}) = i * 3 + j + 1;
        }
    }
    auto iB = Tensor<int>::value({3, 2}, 0);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            iB.at({static_cast<size_t>(i), static_cast<size_t>(j)}) = i * 2 + j + 7;
        }
    }
    auto iC = matmul(iA, iB);
    CHECK(iC.at({0, 0}) == 58 && iC.at({0, 1}) == 64);
    CHECK(iC.at({1, 0}) == 139 && iC.at({1, 1}) == 154);

    auto iC2 = matmul(iB.transpose(), iB); // non-contiguous integer lhs
    CHECK(iC2.at({0, 0}) == 251 && iC2.at({1, 1}) == 308);

    auto k0 = matmul(Tensor<int>::value({2, 0}, 0), Tensor<int>::value({0, 2}, 0));
    CHECK(k0.shape()[0] == 2 && k0.shape()[1] == 2);
    CHECK(k0.at({0, 0}) == 0 && k0.at({1, 1}) == 0);

    auto wrong = Tensor<float>::value({2, 4}, 0.0f);
    check_throws<std::invalid_argument>([&] { (void)matmul(a, wrong); },
                                        "matmul with mismatched inner dimensions");
    // Rank-1 promotion: matvec, vecmat, dot.
    auto vec = Tensor<float>::value({3}, 0.0f);
    vec.at({0}) = 1.0f;
    vec.at({1}) = 2.0f;
    vec.at({2}) = 3.0f;

    auto matvec = matmul(a, vec);
    CHECK(matvec.ndim() == 1 && matvec.shape()[0] == 2);
    CHECK(matvec.at({0}) == 14.0f && matvec.at({1}) == 32.0f);

    auto vecmat = matmul(vec, b);
    CHECK(vecmat.ndim() == 1 && vecmat.shape()[0] == 2);
    CHECK(vecmat.at({0}) == 58.0f && vecmat.at({1}) == 64.0f);

    auto dot = matmul(vec, vec);
    CHECK(dot.ndim() == 0 && dot.item() == 14.0f);

    check_throws<std::invalid_argument>([&] { (void)matmul(a, Tensor<float>::value({}, 1.0f)); },
                                        "matmul with a rank-0 operand");

    // Batched matmul, each batch distinct
    auto batched_a = Tensor<float>::value({2, 2, 3}, 0.0f);
    auto batched_b = Tensor<float>::value({2, 3, 2}, 0.0f);
    for (size_t p = 0; p < 2; ++p) {
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                batched_a.at({p, i, j}) = a.at({i, j}) + static_cast<float>(p) * 100.0f;
            }
        }
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                batched_b.at({p, i, j}) = b.at({i, j}) + static_cast<float>(p) * 100.0f;
            }
        }
    }
    auto batched_c = matmul(batched_a, batched_b);
    CHECK(batched_c.shape()[0] == 2 && batched_c.shape()[1] == 2 && batched_c.shape()[2] == 2);
    for (size_t p = 0; p < 2; ++p) {
        auto a_p = Tensor<float>::value({2, 3}, 0.0f);
        auto b_p = Tensor<float>::value({3, 2}, 0.0f);
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                a_p.at({i, j}) = batched_a.at({p, i, j});
            }
        }
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                b_p.at({i, j}) = batched_b.at({p, i, j});
            }
        }
        auto c_p = matmul(a_p, b_p);
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                CHECK(batched_c.at({p, i, j}) == c_p.at({i, j}));
            }
        }
    }

    // Broadcast batch dims: (2,1,2,3) @ (1,3,3,2) -> (2,3,2,2)
    auto x = Tensor<float>::value({2, 1, 2, 3}, 0.0f);
    auto y = Tensor<float>::value({1, 3, 3, 2}, 0.0f);
    float xv = 1.0f;
    for (size_t p = 0; p < 2; ++p) {
        for (size_t i = 0; i < 2; ++i) {
            for (size_t k = 0; k < 3; ++k) {
                x.at({p, 0, i, k}) = xv++;
            }
        }
    }
    float yv = 1.0f;
    for (size_t q = 0; q < 3; ++q) {
        for (size_t k = 0; k < 3; ++k) {
            for (size_t j = 0; j < 2; ++j) {
                y.at({0, q, k, j}) = yv++;
            }
        }
    }
    auto z = matmul(x, y);
    CHECK(z.shape()[0] == 2 && z.shape()[1] == 3 && z.shape()[2] == 2 && z.shape()[3] == 2);
    for (size_t p = 0; p < 2; ++p) {
        for (size_t q = 0; q < 3; ++q) {
            auto x_pq = Tensor<float>::value({2, 3}, 0.0f);
            auto y_pq = Tensor<float>::value({3, 2}, 0.0f);
            for (size_t i = 0; i < 2; ++i) {
                for (size_t k = 0; k < 3; ++k) {
                    x_pq.at({i, k}) = x.at({p, 0, i, k});
                }
            }
            for (size_t k = 0; k < 3; ++k) {
                for (size_t j = 0; j < 2; ++j) {
                    y_pq.at({k, j}) = y.at({0, q, k, j});
                }
            }
            auto z_pq = matmul(x_pq, y_pq);
            for (size_t i = 0; i < 2; ++i) {
                for (size_t j = 0; j < 2; ++j) {
                    CHECK(z.at({p, q, i, j}) == z_pq.at({i, j}));
                }
            }
        }
    }

    // Batched integer matmul, exercising the no-BLAS path
    auto batched_i = Tensor<int>::value({2, 2, 2}, 0);
    int iv = 1;
    for (size_t p = 0; p < 2; ++p) {
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                batched_i.at({p, i, j}) = iv++;
            }
        }
    }
    auto batched_ic = matmul(batched_i, batched_i);
    CHECK(batched_ic.shape()[0] == 2 && batched_ic.shape()[1] == 2 && batched_ic.shape()[2] == 2);
    for (size_t p = 0; p < 2; ++p) {
        auto i_p = Tensor<int>::value({2, 2}, 0);
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                i_p.at({i, j}) = batched_i.at({p, i, j});
            }
        }
        auto ic_p = matmul(i_p, i_p);
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                CHECK(batched_ic.at({p, i, j}) == ic_p.at({i, j}));
            }
        }
    }

    auto bad_batch = Tensor<float>::value({4, 3, 2}, 0.0f);
    check_throws<std::invalid_argument>([&] { (void)matmul(batched_b, bad_batch); },
                                        "matmul with incompatible batch dimensions");
    auto bad_inner = Tensor<float>::value({2, 4, 2}, 0.0f);
    check_throws<std::invalid_argument>([&] { (void)matmul(batched_a, bad_inner); },
                                        "batched matmul with mismatched inner dimensions");
}

} // namespace

int main() {
    test_factories_and_metadata();
    test_item_and_at();
    test_data_pointer();
    test_reshape_and_flatten();
    test_contiguity();
    test_transpose_and_permute();
    test_squeeze();
    test_unsqueeze();
    test_broadcast_to();
    test_clone();
    test_contiguous();
    test_scalar_arithmetic();
    test_tensor_arithmetic();
    test_inplace_arithmetic();
    test_elementwise_math();
    test_reductions();
    test_allclose();
    test_activations();
    test_printing();
    test_span_factories();
    test_matmul();

    std::cout << (checks - failures) << '/' << checks << " checks passed\n";
    return failures == 0 ? 0 : 1;
}
