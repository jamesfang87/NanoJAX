// clone() and contiguous() are shape- and value-preserving, so they alias the
// source's tape node instead of recording an identity node of their own.

#include "tensor/ops.hpp"

#include <cstddef>
#include <iostream>

using namespace nanojax;

namespace {

int failures = 0;

void expect(bool condition, const char* what) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL " << what << '\n';
    }
}

Tensor<double> make() {
    auto tensor = Tensor<double>::value({2, 3}, 0.0);
    for (size_t i = 0; i < tensor.numel(); ++i) {
        tensor.data()[i] = static_cast<double>(i + 1);
    }
    return tensor;
}

bool all_ones(const Tensor<double>& t) {
    for (size_t i = 0; i < t.shape()[0]; ++i) {
        for (size_t j = 0; j < t.shape()[1]; ++j) {
            if (t.read({i, j}) != 1.0) {
                return false;
            }
        }
    }
    return true;
}

} // namespace

int main() {
    // clone() on a tracked tensor is a tracked identity that adds no node.
    {
        Trace<double> tr;
        auto x = make();
        tr.add_node(x, nullptr);
        auto y = x.clone();
        expect(y.is_tracked(), "clone is tracked");
        expect(tr.adjoints_.size() == 1, "clone adds no node");
        tr.backward(y);
        expect(all_ones(tr.adjoints_[x.id()]), "clone backprops identity");
    }

    // contiguous(true) on a tracked tensor is the same.
    {
        Trace<double> tr;
        auto x = make();
        tr.add_node(x, nullptr);
        auto y = x.contiguous(true);
        expect(y.is_tracked(), "contiguous is tracked");
        expect(tr.adjoints_.size() == 1, "contiguous adds no node");
        tr.backward(y);
        expect(all_ones(tr.adjoints_[x.id()]), "contiguous backprops identity");
    }

    // contiguous() after a shape-changing view aliases that view's node.
    {
        Trace<double> tr;
        auto x = make();
        tr.add_node(x, nullptr);
        auto y = x.transpose().contiguous();
        expect(tr.adjoints_.size() == 2, "only the transpose node is added");
        expect(y.is_tracked(), "contiguous of a view is tracked");
        tr.backward(y);
        expect(all_ones(tr.adjoints_[x.id()]), "transpose.contiguous backprops identity");
    }

    // Untracked inputs stay untracked.
    {
        auto x = make();
        expect(!x.clone().is_tracked(), "untracked clone stays untracked");
        expect(!x.contiguous(true).is_tracked(), "untracked contiguous stays untracked");
    }

    std::cout << (failures == 0 ? "PASS\n" : "FAIL\n");
    return failures == 0 ? 0 : 1;
}
