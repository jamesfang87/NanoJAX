// squeeze(axis) / unsqueeze(axis) are shape-only views, so their VJPs are the
// inverse reshape: a tracked squeeze backprops ones to every input element.

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

bool all_ones(const Tensor<double>& t) {
    for (size_t i = 0; i < t.numel(); ++i) {
        if (t.data()[i] != 1.0) {
            return false;
        }
    }
    return true;
}

} // namespace

int main() {
    // squeeze(axis) on a tracked tensor backprops identity.
    {
        Trace<double> tr;
        auto x = Tensor<double>::value({2, 1, 3}, 4.0);
        tr.add_node(x, nullptr);
        auto y = x.squeeze(1);
        expect(y.shape()[0] == 2 && y.shape()[1] == 3, "squeeze(1) shape");
        expect(tr.adjoints_.size() == 2, "squeeze adds one node");
        tr.backward(y);
        expect(tr.adjoints_[x.id()].ndim() == 3, "adjoint keeps the squeezed axis");
        expect(all_ones(tr.adjoints_[x.id()]), "squeeze(1) backprops identity");
    }

    // unsqueeze(axis) on a tracked tensor backprops identity.
    {
        Trace<double> tr;
        auto x = Tensor<double>::value({2, 3}, 4.0);
        tr.add_node(x, nullptr);
        auto y = x.unsqueeze(1);
        expect(y.ndim() == 3 && y.shape()[1] == 1, "unsqueeze(1) shape");
        tr.backward(y);
        expect(all_ones(tr.adjoints_[x.id()]), "unsqueeze(1) backprops identity");
    }

    std::cout << (failures == 0 ? "PASS\n" : "FAIL\n");
    return failures == 0 ? 0 : 1;
}
