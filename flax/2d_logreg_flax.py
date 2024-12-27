from typing import Iterator

import jax.numpy as jnp
from flax import nnx
import optax
from jaxtyping import Array, Float, Int, jaxtyped
from beartype import beartype
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm


def typed(fn):
    return jaxtyped(fn, typechecker=beartype)


class LogReg(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.rngs = rngs
        self.linear = nnx.Linear(
            in_features=2, out_features=1, use_bias=True, rngs=self.rngs
        )

    @typed
    @nnx.jit()
    def __call__(self, x: Float[Array, "N 2"]) -> Float[Array, "N 1"]:
        x = self.linear(x)
        return x


@typed
@nnx.jit()
def accuracy(logits: Float[Array, "N 1"], y: Int[Array, "N 1"]) -> Float[Array, ""]:
    yh = (nnx.sigmoid(logits) > 0.5).astype(jnp.int32)
    return (yh == y).mean()


@typed
def loss_fn(
    model: LogReg, batch: Float[Array, "N 2"], y: Int[Array, "N 1"]
) -> tuple[Float[Array, ""], Float[Array, "N 1"]]:
    logits = model(batch)
    loss = optax.sigmoid_binary_cross_entropy(logits, y).mean()
    return loss, logits


@typed
@nnx.jit()
def train_step(
    model: LogReg,
    optimizer: nnx.Optimizer,
    metric: nnx.MultiMetric,
    batch: Float[Array, "N 2"],
    y: Int[Array, "N 1"],
) -> None:
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch, y)
    metric.update(loss=loss, accuracy=accuracy(logits, y))
    optimizer.update(grads)


@typed
@nnx.jit()
def eval_step(
    model: LogReg,
    metric: nnx.MultiMetric,
    batch: Float[Array, "N 2"],
    y: Int[Array, "N 1"],
) -> None:
    loss, logits = loss_fn(model, batch, y)
    metric.update(loss=loss, accuracy=accuracy(logits, y))


def dataloader(
    X: Float[Array, "N 2"], y: Int[Array, "N"], batch_size: int = 64
) -> Iterator[tuple[Float[Array, "B 2"], Int[Array, "B"]]]:
    for i in range(0, len(X), batch_size):
        yield X[i : i + batch_size], y[i : i + batch_size]


def train(
    epochs: int = 1000, batch_size: int = 1, learning_rate: float = 0.01
) -> dict[str, list[float]]:
    rngs = nnx.Rngs(0)
    X, y = make_blobs(
        n_samples=1000, n_features=2, centers=2, cluster_std=2, random_state=42
    )
    X = jnp.array(X).astype(jnp.float32)
    y = jnp.array(y).astype(jnp.int32)[:, None]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
    X_test = (X_test - X_test.mean(axis=0)) / X_train.std(axis=0)

    model = LogReg(rngs=rngs)
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate=learning_rate))
    metric = nnx.MultiMetric(
        accuracy=nnx.metrics.Average("accuracy"), loss=nnx.metrics.Average("loss")
    )

    metrics_history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    pbar = tqdm(range(epochs))
    for epoch in pbar:
        try:
            metric.reset()
            for X_batch, y_batch in dataloader(X_train, y_train, batch_size):
                train_step(model, optimizer, metric, X_batch, y_batch)
            train_metrics = metric.compute()

            metric.reset()
            for X_batch, y_batch in dataloader(X_test, y_test, batch_size):
                eval_step(model, metric, X_batch, y_batch)
            test_metrics = metric.compute()

            metrics_history["train_loss"].append(float(train_metrics["loss"]))
            metrics_history["train_accuracy"].append(float(train_metrics["accuracy"]))
            metrics_history["test_loss"].append(float(test_metrics["loss"]))
            metrics_history["test_accuracy"].append(float(test_metrics["accuracy"]))

            pbar.set_description(
                f"Epoch {epoch} - "
                f"Train Accuracy: {train_metrics['accuracy']:.4f} - "
                f"Test Accuracy: {test_metrics['accuracy']:.4f}"
            )
        except KeyboardInterrupt:
            print("Training interrupted by user.")
            break
    return metrics_history


def plot_history(history: dict[str, list[float]]) -> None:
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["test_loss"], label="Test")
    plt.title("Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history["train_accuracy"], label="Train")
    plt.plot(history["test_accuracy"], label="Test")
    plt.title("Accuracy History")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    history = train(epochs=100, batch_size=64, learning_rate=1e-3)

    plot_history(history)
