from __future__ import annotations

import random
import time

import nanojax as nj
from nanojax.nn.activations import relu, softmax
from nanojax.nn.data import DataLoader
from nanojax.nn.data.examples import MNISTDataset
from nanojax.nn.layers.module import MLP, MLPParams
from nanojax.nn.loss import cross_entropy
from nanojax.nn.optimizers import SGD


def init_layer(in_features: int, out_features: int, rng: random.Random) -> MLPParams:
    limit = (6.0 / (in_features + out_features)) ** 0.5
    weight = nj.Tensor(
        [
            [rng.uniform(-limit, limit) for _ in range(out_features)]
            for _ in range(in_features)
        ]
    )
    return MLPParams(weight, nj.zeros([out_features]))


def make_loss(x: nj.Tensor, y: nj.Tensor):
    def loss(model) -> nj.Tensor:
        layer1, layer2 = model
        hidden = relu(MLP.apply(layer1, x))
        logits = MLP.apply(layer2, hidden)
        return cross_entropy(softmax(logits), y) / x.shape[0]

    return loss


def argmax_rows(tensor: nj.Tensor) -> list[int]:
    return [row.index(max(row)) for row in tensor.tolist()]


def accuracy(model, loader: DataLoader) -> float:
    layer1, layer2 = model
    correct = 0
    total = 0
    for x, y in loader:
        logits = MLP.apply(layer2, relu(MLP.apply(layer1, x)))
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

    rng = random.Random(0)
    model = (init_layer(28 * 28, 128, rng), init_layer(128, 10, rng))
    optimizer = SGD(learning_rate)

    for epoch in range(1, epochs + 1):
        start = time.time()
        running_loss = 0.0
        batches = 0

        for x, y in train_loader:
            loss_value, grads = nj.value_and_grad(make_loss(x, y))(model)
            model = tuple(
                optimizer.step(params, grad) for params, grad in zip(model, grads)
            )

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
