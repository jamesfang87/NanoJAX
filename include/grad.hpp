#include <cstddef>
#include <functional>
#include <tuple>
#include <vector>

class Tape;
struct Var {
  const size_t id;

  double value() const;
};

Var operator+(const Var &lhs, const Var &rhs);
Var operator-(const Var &lhs, const Var &rhs);
Var operator*(const Var &lhs, const Var &rhs);
Var operator/(const Var &lhs, const Var &rhs);

class Tape {
public:
  std::vector<double> primals;
  std::vector<double> adjoints;
  std::vector<std::vector<Var>> parents;
  std::vector<std::function<void(int)>> backwards;

public:
  /// Args:
  ///   value: the value of the new node
  ///   parents: the node's parents: the operands used to calculate `value`
  ///   back: the backward func to calculate gradients for the op that create
  ///       value
  Var add_node(double value, std::vector<Var> parents,
               std::function<void(int)> back) {
    size_t id = primals.size();
    this->primals.push_back(value);
    this->adjoints.push_back(0.0);
    this->parents.push_back(parents);
    this->backwards.push_back(back);

    return Var{id};
  }

  void backward(Var out) {
    // adjoint of out beings as 1
    adjoints[out.id] = 1;

    for (size_t i = primals.size(); i-- > 0;) {
      if (backwards[i]) {
        backwards[i](i);
      }
    }
  }

  void reset_grads() { std::fill(adjoints.begin(), adjoints.end(), 0.0); }
};

// Singleton global tape
inline Tape tape;

template <typename F> auto grad(F &&f) {
  return [f = std::forward<F>(f)](auto... ins) {
    tape.reset_grads();

    // create the inputs (leaf nodes)
    auto vars = std::make_tuple(tape.add_node(ins, {}, nullptr)...);

    // run the function on the input and run the backward pass
    auto out = std::apply(f, vars);
    tape.backward(out);

    return std::apply(
        [](const auto &...v) {
          return std::vector<double>{tape.adjoints[v.id]...};
        },
        vars);
  };
}
