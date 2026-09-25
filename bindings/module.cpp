#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "tensor/io.hpp"
#include "tensor/ops.hpp"

namespace nb = nanobind;

using namespace nanojax;

namespace {

using Shape = std::span<const size_t>;

// A single Python-facing tensor that type-erases the C++ dtype.
struct PyTensor {
    using Variant =
        std::variant<Tensor<float>, Tensor<double>, Tensor<std::int32_t>, Tensor<std::int64_t>>;

    Variant value;
};

template <typename> struct element;
template <typename T> struct element<Tensor<T>> {
    using type = T;
};
template <typename V> using element_t = typename element<std::decay_t<V>>::type;

constexpr const char* dtype_name(float) { return "float32"; }
constexpr const char* dtype_name(double) { return "float64"; }
constexpr const char* dtype_name(std::int32_t) { return "int32"; }
constexpr const char* dtype_name(std::int64_t) { return "int64"; }

size_t offset_of(const auto& tensor, const std::vector<size_t>& index) {
    if (index.size() != tensor.ndim()) {
        throw std::invalid_argument{"expected one index per dimension"};
    }

    size_t offset = 0;
    for (size_t dim = 0; dim < index.size(); ++dim) {
        if (index[dim] >= tensor.shape()[dim]) {
            throw std::out_of_range{"index is out of range"};
        }
        offset += index[dim] * tensor.strides()[dim];
    }
    return offset;
}

template <typename T>
void append_nested(nb::list& out, const T* base, Shape shape, Shape stride, size_t dim) {
    for (size_t i = 0; i < shape[dim]; ++i) {
        if (dim + 1 == shape.size()) {
            out.append(base[i * stride[dim]]);
        } else {
            nb::list sub;
            append_nested(sub, base + i * stride[dim], shape, stride, dim + 1);
            out.append(std::move(sub));
        }
    }
}

template <typename T> nb::object to_list(const Tensor<T>& tensor) {
    const auto shape = tensor.shape();
    if (shape.empty()) {
        return nb::cast(tensor.item());
    }
    nb::list root;
    append_nested(root, tensor.data(), shape, tensor.strides(), 0);
    return root;
}

template <typename T>
void flatten_into(nb::handle node, std::vector<size_t>& shape, size_t depth, std::vector<T>& out) {
    const bool is_list = nb::isinstance<nb::list>(node);
    const bool is_tuple = nb::isinstance<nb::tuple>(node);

    if (is_list || is_tuple) {
        const size_t size = nb::len(node);
        if (depth == shape.size()) {
            shape.push_back(size);
        } else if (shape[depth] != size) {
            throw std::invalid_argument{"ragged nested sequences are not supported"};
        }

        for (size_t i = 0; i < size; ++i) {
            nb::object item = is_list ? nb::object(nb::cast<nb::list>(node)[i])
                                      : nb::object(nb::cast<nb::tuple>(node)[i]);
            flatten_into<T>(item, shape, depth + 1, out);
        }
    } else {
        out.push_back(nb::cast<T>(node));
    }
}

template <typename T> Tensor<T> tensor_from(nb::handle data) {
    std::vector<size_t> shape;
    std::vector<T> flat;
    flatten_into<T>(data, shape, 0, flat);

    size_t numel = 1;
    for (size_t extent : shape) {
        numel *= extent;
    }
    if (flat.size() != numel) {
        throw std::invalid_argument{"ragged nested sequences are not supported"};
    }

    auto tensor = Tensor<T>::uninitialized(Shape(shape));
    std::copy(flat.begin(), flat.end(), tensor.data());
    return tensor;
}

const char* infer_dtype(nb::handle node) {
    if (nb::isinstance<nb::list>(node) || nb::isinstance<nb::tuple>(node)) {
        if (nb::len(node) == 0) {
            return "float32";
        }
        return infer_dtype(nb::isinstance<nb::list>(node)
                               ? nb::object(nb::cast<nb::list>(node)[0])
                               : nb::object(nb::cast<nb::tuple>(node)[0]));
    }
    if (nb::isinstance<nb::int_>(node)) {
        return "int64";
    }
    return "float32";
}

template <typename T> struct Tag {
    using type = T;
};

template <typename F> auto dispatch_dtype(const std::string& dtype, F&& fn) {
    if (dtype == "float32" || dtype == "f32") {
        return fn(Tag<float>{});
    }
    if (dtype == "float64" || dtype == "f64") {
        return fn(Tag<double>{});
    }
    if (dtype == "int32" || dtype == "i32") {
        return fn(Tag<std::int32_t>{});
    }
    if (dtype == "int64" || dtype == "i64") {
        return fn(Tag<std::int64_t>{});
    }
    throw std::invalid_argument{"unknown dtype: " + dtype};
}

std::string dtype_of(const PyTensor& tensor) {
    return std::visit(
        [](const auto& t) { return std::string(dtype_name(element_t<decltype(t)>{})); },
        tensor.value);
}

template <typename F> PyTensor visit_unary(const PyTensor& tensor, F fn) {
    return std::visit([&](const auto& x) -> PyTensor { return PyTensor{fn(x)}; }, tensor.value);
}

template <typename F> PyTensor visit_binary(const PyTensor& a, const PyTensor& b, F fn) {
    return std::visit(
        [&](const auto& x, const auto& y) -> PyTensor {
            using X = std::decay_t<decltype(x)>;
            using Y = std::decay_t<decltype(y)>;
            if constexpr (!std::is_same_v<X, Y>) {
                throw std::invalid_argument{"operands must have the same dtype"};
            } else {
                return PyTensor{fn(x, y)};
            }
        },
        a.value, b.value);
}

template <typename F> PyTensor visit_scalar(const PyTensor& tensor, nb::handle scalar, F fn) {
    return std::visit(
        [&](const auto& x) -> PyTensor {
            using T = element_t<decltype(x)>;
            return PyTensor{fn(x, nb::cast<T>(scalar))};
        },
        tensor.value);
}

template <typename F> PyTensor visit_axis(const PyTensor& tensor, size_t axis, F fn) {
    return std::visit([&](const auto& x) -> PyTensor { return PyTensor{fn(x, axis)}; },
                      tensor.value);
}

enum class DType { F32, F64, I32, I64 };

DType dtype_id(const std::string& name) {
    if (name == "float32") {
        return DType::F32;
    }
    if (name == "float64") {
        return DType::F64;
    }
    if (name == "int32") {
        return DType::I32;
    }
    return DType::I64;
}

DType promote(DType a, DType b) {
    if (a == b) {
        return a;
    }
    const bool a_float = a == DType::F32 || a == DType::F64;
    const bool b_float = b == DType::F32 || b == DType::F64;
    if (a_float && b_float) {
        return DType::F64;
    }
    if (!a_float && !b_float) {
        return DType::I64;
    }
    return DType::F64;
}

// Copies @p src into a new dense Tensor<U>, converting every element.
template <typename U, typename T> Tensor<U> cast_tensor(const Tensor<T>& src) {
    auto contiguous = src.contiguous();
    auto out = Tensor<U>::uninitialized(contiguous.shape());

    const T* in = contiguous.data();
    U* dst = out.data();
    for (size_t i = 0; i < contiguous.numel(); ++i) {
        dst[i] = static_cast<U>(in[i]);
    }
    return out;
}

template <typename U> PyTensor cast_to(const PyTensor& tensor) {
    return std::visit(
        [](const auto& x) -> PyTensor {
            using T = element_t<decltype(x)>;
            if constexpr (std::is_same_v<T, U>) {
                return PyTensor{x};
            } else {
                return PyTensor{cast_tensor<U>(x)};
            }
        },
        tensor.value);
}

template <typename F> PyTensor visit_binary_promote(const PyTensor& a, const PyTensor& b, F fn) {
    switch (promote(dtype_id(dtype_of(a)), dtype_id(dtype_of(b)))) {
    case DType::F32:
        return visit_binary(cast_to<float>(a), cast_to<float>(b), fn);
    case DType::F64:
        return visit_binary(cast_to<double>(a), cast_to<double>(b), fn);
    case DType::I32:
        return visit_binary(cast_to<std::int32_t>(a), cast_to<std::int32_t>(b), fn);
    default:
        return visit_binary(cast_to<std::int64_t>(a), cast_to<std::int64_t>(b), fn);
    }
}

template <typename F> auto visit_binary_promote_value(const PyTensor& a, const PyTensor& b, F fn) {
    switch (promote(dtype_id(dtype_of(a)), dtype_id(dtype_of(b)))) {
    case DType::F32: {
        auto x = cast_to<float>(a);
        auto y = cast_to<float>(b);
        return fn(std::get<Tensor<float>>(x.value), std::get<Tensor<float>>(y.value));
    }
    case DType::F64: {
        auto x = cast_to<double>(a);
        auto y = cast_to<double>(b);
        return fn(std::get<Tensor<double>>(x.value), std::get<Tensor<double>>(y.value));
    }
    case DType::I32: {
        auto x = cast_to<std::int32_t>(a);
        auto y = cast_to<std::int32_t>(b);
        return fn(std::get<Tensor<std::int32_t>>(x.value), std::get<Tensor<std::int32_t>>(y.value));
    }
    default: {
        auto x = cast_to<std::int64_t>(a);
        auto y = cast_to<std::int64_t>(b);
        return fn(std::get<Tensor<std::int64_t>>(x.value), std::get<Tensor<std::int64_t>>(y.value));
    }
    }
}

template <typename Op> void apply_inplace(PyTensor& a, const PyTensor& b, Op op) {
    std::visit(
        [&](auto& x, const auto& y) {
            using X = element_t<decltype(x)>;
            using Y = element_t<decltype(y)>;
            if constexpr (std::is_same_v<X, Y>) {
                op(x, y);
            } else {
                auto y_cast = cast_tensor<X>(y);
                op(x, y_cast);
            }
        },
        a.value, b.value);
}

PyTensor make_tensor(nb::handle data, const std::string& dtype) {
    return dispatch_dtype(dtype, [&](auto tag) -> PyTensor {
        using T = typename decltype(tag)::type;
        return PyTensor{tensor_from<T>(data)};
    });
}

PyTensor make_zeros(const std::vector<size_t>& shape, const std::string& dtype) {
    return dispatch_dtype(dtype, [&](auto tag) -> PyTensor {
        using T = typename decltype(tag)::type;
        return PyTensor{Tensor<T>::zeros(Shape(shape))};
    });
}

PyTensor make_ones(const std::vector<size_t>& shape, const std::string& dtype) {
    return dispatch_dtype(dtype, [&](auto tag) -> PyTensor {
        using T = typename decltype(tag)::type;
        return PyTensor{Tensor<T>::ones(Shape(shape))};
    });
}

PyTensor make_full(const std::vector<size_t>& shape, nb::handle value, const std::string& dtype) {
    return dispatch_dtype(dtype, [&](auto tag) -> PyTensor {
        using T = typename decltype(tag)::type;
        return PyTensor{Tensor<T>::value(Shape(shape), nb::cast<T>(value))};
    });
}

std::string to_repr(const PyTensor& tensor) {
    return std::visit(
        [](const auto& t) {
            std::ostringstream os;
            os << t;
            return os.str();
        },
        tensor.value);
}

PyTensor op_exp(const PyTensor& a) {
    return visit_unary(a, [](const auto& x) { return exp(x); });
}
PyTensor op_log(const PyTensor& a) {
    return visit_unary(a, [](const auto& x) { return log(x); });
}
PyTensor op_sqrt(const PyTensor& a) {
    return visit_unary(a, [](const auto& x) { return sqrt(x); });
}
PyTensor op_abs(const PyTensor& a) {
    return visit_unary(a, [](const auto& x) { return abs(x); });
}
PyTensor op_relu(const PyTensor& a) {
    return visit_unary(a, [](const auto& x) { return relu(x); });
}
PyTensor op_sigmoid(const PyTensor& a) {
    return visit_unary(a, [](const auto& x) { return sigmoid(x); });
}
PyTensor op_tanh(const PyTensor& a) {
    return visit_unary(a, [](const auto& x) { return tanh(x); });
}
PyTensor op_pow(const PyTensor& a, nb::handle exponent) {
    return visit_scalar(a, exponent, [](const auto& x, const auto& v) { return pow(x, v); });
}

struct PyTrace {
    using Variant = std::variant<std::shared_ptr<Trace<float>>, std::shared_ptr<Trace<double>>>;
    Variant value;
};

PyTrace make_trace(const std::string& dtype) {
    if (dtype == "float32" || dtype == "f32") {
        return PyTrace{std::make_shared<Trace<float>>()};
    }
    if (dtype == "float64" || dtype == "f64") {
        return PyTrace{std::make_shared<Trace<double>>()};
    }
    throw std::invalid_argument{"autodiff needs a floating dtype: float32 or float64"};
}

thread_local std::vector<PyTrace> trace_stack;

PyTrace& current_trace() {
    if (trace_stack.empty()) {
        throw std::runtime_error{"no active Trace; wrap the call in 'with Trace(dtype):'"};
    }
    return trace_stack.back();
}

PyTensor trace_attach(const PyTensor& tensor) {
    PyTrace& trace = current_trace();
    return std::visit(
        [&](const auto& x) -> PyTensor {
            using T = element_t<decltype(x)>;
            if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
                auto* tr = std::get_if<std::shared_ptr<Trace<T>>>(&trace.value);
                if (tr == nullptr) {
                    throw std::invalid_argument{"tensor dtype does not match the trace"};
                }
                Tensor<T> leaf = x;
                (*tr)->add_node(leaf, nullptr);
                return PyTensor{std::move(leaf)};
            } else {
                throw std::invalid_argument{"autodiff needs a floating tensor"};
            }
        },
        tensor.value);
}

void trace_backward(const PyTensor& output) {
    PyTrace& trace = current_trace();
    std::visit(
        [&](const auto& x) {
            using T = element_t<decltype(x)>;
            if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
                auto* tr = std::get_if<std::shared_ptr<Trace<T>>>(&trace.value);
                if (tr == nullptr || !x.is_tracked() || x.trace() != tr->get()) {
                    throw std::invalid_argument{
                        "backward needs an output tracked on the current trace"};
                }
                (*tr)->backward(x);
            } else {
                throw std::invalid_argument{"autodiff needs a floating tensor"};
            }
        },
        output.value);
}

PyTensor trace_adjoint(const PyTensor& tensor) {
    PyTrace& trace = current_trace();
    return std::visit(
        [&](const auto& x) -> PyTensor {
            using T = element_t<decltype(x)>;
            if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
                const auto* tr = std::get_if<std::shared_ptr<Trace<T>>>(&trace.value);
                if (tr == nullptr || !x.is_tracked() || x.trace() != tr->get()) {
                    throw std::invalid_argument{
                        "adjoint needs a tensor lifted on the current trace"};
                }
                return PyTensor{(*tr)->adjoint(x)};
            } else {
                throw std::invalid_argument{"autodiff needs a floating tensor"};
            }
        },
        tensor.value);
}

} // namespace

NB_MODULE(_tensor, t) {
    t.doc() = "A tensor and autodiff library";

    nb::class_<PyTensor>(t, "Tensor")
        .def(
            "__init__",
            [](PyTensor* self, nb::handle data, nb::object dtype) {
                const std::string name =
                    dtype.is_none() ? infer_dtype(data) : nb::cast<std::string>(dtype);
                new (self) PyTensor{make_tensor(data, name)};
            },
            nb::arg("data"), nb::arg("dtype") = nb::none())
        .def_prop_ro("dtype", &dtype_of)
        .def_prop_ro("shape",
                     [](const PyTensor& t) {
                         return std::visit(
                             [](const auto& x) {
                                 return std::vector<size_t>(x.shape().begin(), x.shape().end());
                             },
                             t.value);
                     })
        .def_prop_ro("strides",
                     [](const PyTensor& t) {
                         return std::visit(
                             [](const auto& x) {
                                 return std::vector<size_t>(x.strides().begin(), x.strides().end());
                             },
                             t.value);
                     })
        .def_prop_ro("ndim",
                     [](const PyTensor& t) {
                         return std::visit([](const auto& x) { return x.ndim(); }, t.value);
                     })
        .def_prop_ro("is_contiguous",
                     [](const PyTensor& t) {
                         return std::visit([](const auto& x) { return x.is_contiguous(); },
                                           t.value);
                     })
        .def_prop_ro("empty",
                     [](const PyTensor& t) {
                         return std::visit([](const auto& x) { return x.empty(); }, t.value);
                     })
        .def_prop_ro("device",
                     [](const PyTensor& t) {
                         return std::visit(
                             [](const auto& x) {
                                 return x.device() == Device::CPU ? std::string("cpu")
                                                                  : std::string("cuda");
                             },
                             t.value);
                     })
        .def_prop_ro("is_tracked",
                     [](const PyTensor& t) {
                         return std::visit([](const auto& x) { return x.is_tracked(); }, t.value);
                     })
        .def("detach",
             [](const PyTensor& a) {
                 return visit_unary(a, [](const auto& x) { return x.detach(); });
             })
        .def("numel",
             [](const PyTensor& t) {
                 return std::visit([](const auto& x) { return x.numel(); }, t.value);
             })
        .def("item",
             [](const PyTensor& t) {
                 return std::visit([](const auto& x) -> nb::object { return nb::cast(x.item()); },
                                   t.value);
             })
        .def("__len__",
             [](const PyTensor& t) {
                 if (std::visit([](const auto& x) { return x.ndim(); }, t.value) == 0) {
                     throw std::invalid_argument{"len() of a 0-d tensor"};
                 }
                 return std::visit([](const auto& x) { return x.shape()[0]; }, t.value);
             })
        .def("__repr__", &to_repr)
        .def("tolist",
             [](const PyTensor& t) {
                 return std::visit([](const auto& x) -> nb::object { return to_list(x); }, t.value);
             })
        .def("__getitem__",
             [](const PyTensor& t, size_t i) {
                 return std::visit(
                     [&](const auto& x) -> nb::object {
                         return nb::cast(x.data()[offset_of(x, {i})]);
                     },
                     t.value);
             })
        .def("__getitem__",
             [](const PyTensor& t, std::vector<size_t> index) {
                 return std::visit(
                     [&](const auto& x) -> nb::object {
                         return nb::cast(x.data()[offset_of(x, index)]);
                     },
                     t.value);
             })
        .def("__setitem__",
             [](PyTensor& t, size_t i, nb::handle value) {
                 std::visit(
                     [&](auto& x) {
                         x.data()[offset_of(x, {i})] = nb::cast<element_t<decltype(x)>>(value);
                     },
                     t.value);
             })
        .def("__setitem__",
             [](PyTensor& t, std::vector<size_t> index, nb::handle value) {
                 std::visit(
                     [&](auto& x) {
                         x.data()[offset_of(x, index)] = nb::cast<element_t<decltype(x)>>(value);
                     },
                     t.value);
             })
        .def("__neg__",
             [](const PyTensor& a) { return visit_unary(a, [](const auto& x) { return -x; }); })
        .def("__add__",
             [](const PyTensor& a, const PyTensor& b) {
                 return visit_binary_promote(a, b,
                                             [](const auto& x, const auto& y) { return x + y; });
             })
        .def("__sub__",
             [](const PyTensor& a, const PyTensor& b) {
                 return visit_binary_promote(a, b,
                                             [](const auto& x, const auto& y) { return x - y; });
             })
        .def("__mul__",
             [](const PyTensor& a, const PyTensor& b) {
                 return visit_binary_promote(a, b,
                                             [](const auto& x, const auto& y) { return x * y; });
             })
        .def("__truediv__",
             [](const PyTensor& a, const PyTensor& b) {
                 return visit_binary_promote(a, b,
                                             [](const auto& x, const auto& y) { return x / y; });
             })
        .def(
            "__matmul__",
            [](const PyTensor& a, const PyTensor& b) {
                return visit_binary_promote(
                    a, b, [](const auto& x, const auto& y) { return matmul(x, y); });
            },
            nb::call_guard<nb::gil_scoped_release>())
        .def(
            "__add__",
            [](const PyTensor& a, nb::handle s) {
                return visit_scalar(a, s, [](const auto& x, const auto& v) { return x + v; });
            },
            nb::is_operator())
        .def(
            "__sub__",
            [](const PyTensor& a, nb::handle s) {
                return visit_scalar(a, s, [](const auto& x, const auto& v) { return x - v; });
            },
            nb::is_operator())
        .def(
            "__mul__",
            [](const PyTensor& a, nb::handle s) {
                return visit_scalar(a, s, [](const auto& x, const auto& v) { return x * v; });
            },
            nb::is_operator())
        .def(
            "__truediv__",
            [](const PyTensor& a, nb::handle s) {
                return visit_scalar(a, s, [](const auto& x, const auto& v) { return x / v; });
            },
            nb::is_operator())
        .def(
            "__radd__",
            [](const PyTensor& a, nb::handle s) {
                return visit_scalar(a, s, [](const auto& x, const auto& v) { return v + x; });
            },
            nb::is_operator())
        .def(
            "__rsub__",
            [](const PyTensor& a, nb::handle s) {
                return visit_scalar(a, s, [](const auto& x, const auto& v) { return v - x; });
            },
            nb::is_operator())
        .def(
            "__rmul__",
            [](const PyTensor& a, nb::handle s) {
                return visit_scalar(a, s, [](const auto& x, const auto& v) { return v * x; });
            },
            nb::is_operator())
        .def(
            "__rtruediv__",
            [](const PyTensor& a, nb::handle s) {
                return visit_scalar(a, s, [](const auto& x, const auto& v) { return v / x; });
            },
            nb::is_operator())
        .def("__iadd__",
             [](nb::object self, const PyTensor& b) -> nb::object {
                 PyTensor& a = nb::cast<PyTensor&>(self);
                 apply_inplace(a, b, [](auto& x, const auto& y) { x += y; });
                 return self;
             })
        .def("__isub__",
             [](nb::object self, const PyTensor& b) -> nb::object {
                 PyTensor& a = nb::cast<PyTensor&>(self);
                 apply_inplace(a, b, [](auto& x, const auto& y) { x -= y; });
                 return self;
             })
        .def("__imul__",
             [](nb::object self, const PyTensor& b) -> nb::object {
                 PyTensor& a = nb::cast<PyTensor&>(self);
                 apply_inplace(a, b, [](auto& x, const auto& y) { x *= y; });
                 return self;
             })
        .def("__itruediv__",
             [](nb::object self, const PyTensor& b) -> nb::object {
                 PyTensor& a = nb::cast<PyTensor&>(self);
                 apply_inplace(a, b, [](auto& x, const auto& y) { x /= y; });
                 return self;
             })
        .def("__iadd__",
             [](nb::object self, nb::handle s) -> nb::object {
                 PyTensor& a = nb::cast<PyTensor&>(self);
                 std::visit(
                     [&](auto& x) {
                         using T = element_t<decltype(x)>;
                         x += nb::cast<T>(s);
                     },
                     a.value);
                 return self;
             })
        .def("__isub__",
             [](nb::object self, nb::handle s) -> nb::object {
                 PyTensor& a = nb::cast<PyTensor&>(self);
                 std::visit(
                     [&](auto& x) {
                         using T = element_t<decltype(x)>;
                         x -= nb::cast<T>(s);
                     },
                     a.value);
                 return self;
             })
        .def("__imul__",
             [](nb::object self, nb::handle s) -> nb::object {
                 PyTensor& a = nb::cast<PyTensor&>(self);
                 std::visit(
                     [&](auto& x) {
                         using T = element_t<decltype(x)>;
                         x *= nb::cast<T>(s);
                     },
                     a.value);
                 return self;
             })
        .def("__itruediv__",
             [](nb::object self, nb::handle s) -> nb::object {
                 PyTensor& a = nb::cast<PyTensor&>(self);
                 std::visit(
                     [&](auto& x) {
                         using T = element_t<decltype(x)>;
                         x /= nb::cast<T>(s);
                     },
                     a.value);
                 return self;
             })
        .def("sum",
             [](const PyTensor& a) { return visit_unary(a, [](const auto& x) { return sum(x); }); })
        .def("sum",
             [](const PyTensor& a, size_t axis) {
                 return visit_axis(a, axis, [](const auto& x, size_t ax) { return sum(x, ax); });
             })
        .def(
            "mean",
            [](const PyTensor& a) { return visit_unary(a, [](const auto& x) { return mean(x); }); })
        .def("mean",
             [](const PyTensor& a, size_t axis) {
                 return visit_axis(a, axis, [](const auto& x, size_t ax) { return mean(x, ax); });
             })
        .def("max",
             [](const PyTensor& a) { return visit_unary(a, [](const auto& x) { return max(x); }); })
        .def("min",
             [](const PyTensor& a) { return visit_unary(a, [](const auto& x) { return min(x); }); })
        .def("transpose",
             [](const PyTensor& a) {
                 return visit_unary(a, [](const auto& x) { return x.transpose(); });
             })
        .def("clone",
             [](const PyTensor& a) {
                 return visit_unary(a, [](const auto& x) { return x.clone(); });
             })
        .def(
            "contiguous",
            [](const PyTensor& a, bool always_copy) {
                return visit_unary(a, [&](const auto& x) { return x.contiguous(always_copy); });
            },
            nb::arg("always_copy") = false)
        .def("broadcast_to",
             [](const PyTensor& a, std::vector<size_t> shape) {
                 return visit_unary(a, [&](const auto& x) { return x.broadcast(Shape(shape)); });
             })
        .def("reshape",
             [](const PyTensor& a, std::vector<size_t> shape) {
                 return visit_unary(a, [&](const auto& x) { return x.reshape(Shape(shape)); });
             })
        .def("flatten",
             [](const PyTensor& a) {
                 return visit_unary(a, [](const auto& x) { return x.flatten(); });
             })
        .def("permute",
             [](const PyTensor& a, std::vector<size_t> axes) {
                 return visit_unary(a, [&](const auto& x) { return x.permute(Shape(axes)); });
             })
        .def("squeeze",
             [](const PyTensor& a) {
                 return visit_unary(a, [](const auto& x) { return x.squeeze(); });
             })
        .def("unsqueeze",
             [](const PyTensor& a, size_t axis) {
                 return visit_unary(a, [&](const auto& x) { return x.unsqueeze(axis); });
             })
        .def("exp", &op_exp)
        .def("log", &op_log)
        .def("sqrt", &op_sqrt)
        .def("abs", &op_abs)
        .def("relu", &op_relu)
        .def("sigmoid", &op_sigmoid)
        .def("tanh", &op_tanh)
        .def("pow", &op_pow)
        .def("__abs__", &op_abs)
        .def("__pow__", &op_pow)
        .def(
            "allclose",
            [](const PyTensor& a, const PyTensor& b, double tolerance) {
                return visit_binary_promote_value(
                    a, b, [&](const auto& x, const auto& y) { return x.allclose(y, tolerance); });
            },
            nb::arg("other"), nb::arg("tolerance") = 1e-9);

    t.def(
        "zeros",
        [](std::vector<size_t> shape, nb::object dtype) {
            return make_zeros(shape, dtype.is_none() ? "float32" : nb::cast<std::string>(dtype));
        },
        nb::arg("shape"), nb::arg("dtype") = nb::none());

    t.def(
        "ones",
        [](std::vector<size_t> shape, nb::object dtype) {
            return make_ones(shape, dtype.is_none() ? "float32" : nb::cast<std::string>(dtype));
        },
        nb::arg("shape"), nb::arg("dtype") = nb::none());

    t.def(
        "full",
        [](std::vector<size_t> shape, nb::handle value, nb::object dtype) {
            const std::string name = dtype.is_none()
                                         ? (nb::isinstance<nb::int_>(value) ? "int64" : "float32")
                                         : nb::cast<std::string>(dtype);
            return make_full(shape, value, name);
        },
        nb::arg("shape"), nb::arg("value"), nb::arg("dtype") = nb::none());

    t.def(
        "uninitialized",
        [](std::vector<size_t> shape, nb::object dtype) {
            const std::string name = dtype.is_none() ? "float32" : nb::cast<std::string>(dtype);
            return dispatch_dtype(name, [&](auto tag) -> PyTensor {
                using T = typename decltype(tag)::type;
                return PyTensor{Tensor<T>::uninitialized(Shape(shape))};
            });
        },
        nb::arg("shape"), nb::arg("dtype") = nb::none());

    t.def("exp", &op_exp);
    t.def("log", &op_log);
    t.def("sqrt", &op_sqrt);
    t.def("abs", &op_abs);
    t.def("relu", &op_relu);
    t.def("sigmoid", &op_sigmoid);
    t.def("tanh", &op_tanh);

    t.def(
        "matmul",
        [](const PyTensor& a, const PyTensor& b) {
            return visit_binary_promote(a, b,
                                        [](const auto& x, const auto& y) { return matmul(x, y); });
        },
        nb::call_guard<nb::gil_scoped_release>());

    nb::class_<PyTrace>(t, "Trace")
        .def(
            "__init__",
            [](PyTrace* self, const std::string& dtype) { new (self) PyTrace{make_trace(dtype)}; },
            nb::arg("dtype") = "float32")
        .def("__enter__",
             [](PyTrace& self) -> PyTrace& {
                 std::visit([](auto& trace) { trace->push(); }, self.value);
                 trace_stack.push_back(self);
                 return self;
             })
        .def("__exit__", [](PyTrace& self, nb::args) {
            std::visit([](auto& trace) { trace->pop(); }, self.value);
            trace_stack.pop_back();
            return false;
        });

    t.def("attach", &trace_attach, nb::arg("tensor"));
    t.def("backward", &trace_backward, nb::arg("output"));
    t.def("adjoint", &trace_adjoint, nb::arg("tensor"));

    t.def("stop_gradient", [](const PyTensor& a) {
        return visit_unary(a, [](const auto& x) { return x.detach(); });
    });
}
