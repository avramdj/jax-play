from dataclasses import dataclass, field
from typing import Any, Iterator, cast

import jax
import jax.numpy as jnp
import lovely_jax  # type: ignore
import optax  # type: ignore
from beartype import beartype
from datasets import load_dataset  # type: ignore
from jaxtyping import Array, Float, Int, jaxtyped
from tqdm import tqdm  # type: ignore
from flax import nnx


def typed(fn):
    return jaxtyped(fn, typechecker=beartype)


@dataclass
class VITConfig:
    rngs: nnx.Rngs = field(default_factory=lambda: nnx.Rngs(0))
    in_feature_shape: tuple[int, int, int] = (32, 32, 3)
    out_features: int = 10
    patch_size: int = 4
    num_layers: int = 8
    num_heads: int = 8
    embed_dim: int = 256


class Residual(nnx.Module):
    def __init__(self, module: nnx.Module, config: VITConfig):
        self.norm = nnx.LayerNorm(
            num_features=config.embed_dim,
            rngs=config.rngs,
        )
        self.module = module

    @typed
    @nnx.jit
    def __call__(self, x: Float[Array, "batch ..."]) -> Float[Array, "batch ..."]:  # type: ignore
        x = self.norm(x)
        return x + self.module(x)  # type: ignore


class Patchify(nnx.Module):
    def __init__(self, config: VITConfig):
        self.config = config
        self.conv = nnx.Conv(
            in_features=config.in_feature_shape[2],
            out_features=config.embed_dim,
            kernel_size=(config.patch_size, config.patch_size),
            strides=(config.patch_size, config.patch_size),
            rngs=config.rngs,
        )

    @typed
    @nnx.jit
    def __call__(
        self, x: Float[Array, "batch h w ch"]
    ) -> Float[Array, "batch patches emb"]:
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1, self.config.embed_dim)
        cls_token = jax.nn.initializers.truncated_normal(stddev=0.02)(
            jax.random.key(0),
            dtype=jnp.float32,
            shape=(x.shape[0], 1, self.config.embed_dim),
        )
        x = jnp.concatenate([cls_token, x], axis=1)
        return x


@typed
@nnx.jit
def apply_rope(
    q: Float[Array, "batch n d"],
    k: Float[Array, "batch n d"],
) -> tuple[Float[Array, "batch n d"], Float[Array, "batch n d"]]:
    return q, k


class AttnBlock(nnx.Module):
    def __init__(self, config: VITConfig):
        self.config = config
        self.qkv = nnx.Linear(
            in_features=config.embed_dim,
            out_features=config.embed_dim * 3,
            rngs=config.rngs,
        )

    @typed
    @nnx.jit
    def __call__(
        self, x: Float[Array, "batch patches emb"]
    ) -> Float[Array, "batch patches emb"]:
        qkv = self.qkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q, k = apply_rope(q, k)
        a = nnx.dot_product_attention(q, k, v)
        a = a.reshape(a.shape[0], -1, self.config.embed_dim)
        return a


class MLP(nnx.Module):
    def __init__(self, config: VITConfig):
        self.config = config
        self.norm = nnx.LayerNorm(
            num_features=config.embed_dim,
            rngs=config.rngs,
        )
        self.linear1 = nnx.Linear(
            in_features=config.embed_dim,
            out_features=config.embed_dim * 4,
            rngs=config.rngs,
        )
        self.linear2 = nnx.Linear(
            in_features=config.embed_dim * 4,
            out_features=config.embed_dim,
            rngs=config.rngs,
        )

    @typed
    @nnx.jit
    def __call__(
        self, x: Float[Array, "batch patches emb"]
    ) -> Float[Array, "batch patches emb"]:
        x = self.norm(x)
        x = self.linear1(x)
        x = jax.nn.gelu(x)
        x = self.linear2(x)
        return x


class EncoderBlock(nnx.Module):
    def __init__(self, config: VITConfig):
        self.config = config
        self.mha = Residual(AttnBlock(config=config), config=config)
        self.mlp = Residual(MLP(config=config), config=config)

    @typed
    @nnx.jit
    def __call__(
        self, x: Float[Array, "batch patches emb"]
    ) -> Float[Array, "batch patches emb"]:
        x = self.mha(x)
        x = self.mlp(x)
        return x


class Encoder(nnx.Module):
    def __init__(self, config: VITConfig):
        self.config = config
        self.layers = nnx.Sequential(
            *[EncoderBlock(config=config) for _ in range(config.num_layers)]
        )

    @typed
    @nnx.jit
    def __call__(
        self, x: Float[Array, "batch patches emb"]
    ) -> Float[Array, "batch patches emb"]:
        return self.layers(x)


class VIT(nnx.Module):
    def __init__(self, config: VITConfig):
        self.config = config
        self.patchify = Patchify(config=config)
        self.encoder = Encoder(config=config)

    @typed
    @nnx.jit
    def __call__(
        self, x: Float[Array, "batch h w ch"]
    ) -> Float[Array, "batch patches emb"]:
        x = self.patchify(x)
        x = self.encoder(x)
        return x


class VITClassifier(nnx.Module):
    def __init__(self, config: VITConfig, num_classes: int):
        self.config = config
        self.vit = VIT(config=config)
        self.linear_probe = nnx.Linear(
            in_features=config.embed_dim,
            out_features=num_classes,
            rngs=config.rngs,
        )

    @typed
    @nnx.jit
    def __call__(self, x: Float[Array, "batch h w ch"]) -> Float[Array, "batch c"]:
        x = self.vit(x)
        x = x[:, 0, :]
        x = self.linear_probe(x)
        return x


def dataloader(
    X: Float[Array, "n h w ch"], y: Int[Array, "n c"], batch_size: int = 64
) -> Iterator[tuple[Float[Array, "batch h w ch"], Int[Array, "batch c"]]]:
    for i in range(0, len(X), batch_size):
        yield X[i : i + batch_size], y[i : i + batch_size]


def prepare_dataset(train_size: int | None = None, val_size: int | None = None):
    dataset = load_dataset("cifar10")
    train_size = train_size or len(dataset["train"])
    val_size = val_size or len(dataset["test"])

    X_train = jnp.array([dataset["train"][i]["img"] for i in range(train_size)])
    y_train = jnp.array([dataset["train"][i]["label"] for i in range(train_size)])
    y_train = jax.nn.one_hot(y_train, num_classes=10).astype(jnp.int32)

    X_val = jnp.array([dataset["test"][i]["img"] for i in range(val_size)])
    y_val = jnp.array([dataset["test"][i]["label"] for i in range(val_size)])
    y_val = jax.nn.one_hot(y_val, num_classes=10).astype(jnp.int32)

    idx2cls = {i: cls for i, cls in enumerate(dataset["train"].features["label"].names)}
    cls2idx = {cls: i for i, cls in idx2cls.items()}
    num_classes = len(idx2cls)

    X_train = (X_train / 255.0).astype(jnp.float32)
    X_val = (X_val / 255.0).astype(jnp.float32)

    return X_train, y_train, X_val, y_val, num_classes


def loss_fn(
    model: VITClassifier, X: Float[Array, "batch h w ch"], y: Int[Array, "batch c"]
) -> tuple[Float[Array, ""], Float[Array, "batch c"]]:
    logits = model(X)
    loss = optax.softmax_cross_entropy(logits, y).mean()
    return loss, logits


def accuracy(
    logits: Float[Array, "batch c"], y: Int[Array, "batch c"]
) -> Float[Array, ""]:
    return (logits.argmax(axis=-1) == y.argmax(axis=-1)).mean()


@typed
@nnx.jit
def train_step(
    model: VITClassifier,
    X: Float[Array, "batch h w ch"],
    y: Int[Array, "batch c"],
    optimizer: nnx.Optimizer,
    metrics: nnx.Metric,
) -> None:
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, X, y)
    acc = accuracy(logits, y)
    metrics.update(loss=loss, accuracy=acc)
    optimizer.update(grads)


@typed
@nnx.jit
def val_step(
    model: VITClassifier,
    X: Float[Array, "batch h w ch"],
    y: Int[Array, "batch c"],
    metrics: nnx.Metric,
) -> None:
    (loss, logits), _ = nnx.value_and_grad(loss_fn, has_aux=True)(model, X, y)
    acc = accuracy(logits, y)
    metrics.update(loss=loss, accuracy=acc)


def train(
    model: VITClassifier,
    optimizer: nnx.Optimizer,
    X_train: Float[Array, "n h w ch"],
    y_train: Int[Array, "n c"],
    X_val: Float[Array, "n h w ch"],
    y_val: Int[Array, "n c"],
    epochs: int = 100,
) -> None:
    metrics = nnx.metrics.MultiMetric(
        accuracy=nnx.metrics.Average("accuracy"),
        loss=nnx.metrics.Average("loss"),
    )
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        metrics.reset()
        for X, y in dataloader(X_train, y_train, batch_size=64):
            train_step(model, X, y, optimizer, metrics)
        m = cast(dict[str, Any], metrics.compute())
        train_loss = float(m["loss"])
        train_acc = float(m["accuracy"])

        metrics.reset()
        for X, y in dataloader(X_val, y_val, batch_size=64):
            val_step(model, X, y, metrics)
        m = cast(dict[str, Any], metrics.compute())
        val_loss = float(m["loss"])
        val_acc = float(m["accuracy"])

        pbar.set_description(
            f"Epoch {epoch} train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} "
            f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}"
        )


if __name__ == "__main__":
    lovely_jax.monkey_patch()

    config = VITConfig()
    model = VITClassifier(config=config, num_classes=10)
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate=1e-3))
    X_train, y_train, X_val, y_val, num_classes = prepare_dataset()

    train(model, optimizer, X_train, y_train, X_val, y_val, epochs=100)
