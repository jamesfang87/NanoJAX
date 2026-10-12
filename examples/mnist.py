from __future__ import annotations

import time

import nanojax as nj
from nanojax.nn.activations import relu, softmax
from nanojax.nn.data import DataLoader
from nanojax.nn.data.examples import MNISTDataset
from nanojax.nn.layers.module import MLP, Module
from nanojax.nn.loss import cross_entropy
from nanojax.nn.optimizers import SGD


class Net(Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        self.fc1 = MLP(in_features, hidden_features, seed=0)
        self.fc2 = MLP(hidden_features, out_features, seed=1)

    def forward(self, x: nj.Tensor) -> nj.Tensor:
        return self.fc2(relu(self.fc1(x)))


def make_loss(x: nj.Tensor, y: nj.Tensor):
    def loss(model: Module) -> nj.Tensor:
        return cross_entropy(softmax(model(x)), y) / x.shape[0]

    return loss


def argmax_rows(tensor: nj.Tensor) -> list[int]:
    return [row.index(max(row)) for row in tensor.tolist()]


def accuracy(model: Module, loader: DataLoader) -> float:
    correct = 0
    total = 0
    for x, y in loader:
        logits = model(x)
        predictions = argmax_rows(logits)
        labels = argmax_rows(y)
        correct += sum(p == t for p, t in zip(predictions, labels))
        total += len(predictions)
    return correct / total


def main() -> None:
    epochs = 3
    batch_size = 128
    learning_rate = 0.1

    train_set = MNISTDataset(root="data", train=True)
    test_set = MNISTDataset(root="data", train=False)
    train_loader = DataLoader(train_set, batch_size, shuffle=True, seed=0)
    test_loader = DataLoader(test_set, 256, shuffle=False)

    model = Net(28 * 28, 128, 10)
    optimizer = SGD(learning_rate)

    for epoch in range(1, epochs + 1):
        start = time.time()
        running_loss = 0.0
        batches = 0

        for x, y in train_loader:
            loss_value, grads = nj.value_and_grad(make_loss(x, y))(model)
            model = optimizer.step(model, grads)

            running_loss += loss_value.item()
            batches += 1

        elapsed = time.time() - start
        train_accuracy = accuracy(model, train_loader)
        test_accuracy = accuracy(model, test_loader)
        print(
            f"epoch {epoch}: loss={running_loss / batches:.4f} "
            f"train_acc={train_accuracy:.4f} test_acc={test_accuracy:.4f} "
            f"({elapsed:.1f}s)"
        )


if __name__ == "__main__":
    main()
