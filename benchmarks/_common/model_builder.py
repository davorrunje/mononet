"""Embedding-composition model builder over the four mononet flavors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from benchmarks._common.bundle import (
    DatasetBundle,
    free_columns,
    mono_columns,
    mono_signs,
)

if TYPE_CHECKING:
    from benchmarks._common.config import BenchmarkConfig


def build_model(cfg: BenchmarkConfig, bundle: DatasetBundle) -> Any:
    """Dispatch to the backend-specific model builder.

    :param cfg: Benchmark configuration.
    :param bundle: Dataset bundle with monotonicity declarations.
    :returns: Backend-native callable model.
    :raises ValueError: If ``cfg.backend`` is not recognised.
    """
    if cfg.backend == "torch":
        return _build_torch(cfg, bundle)
    if cfg.backend == "jax":
        return _build_jax(cfg, bundle)
    if cfg.backend == "keras":
        return _build_keras(cfg, bundle)
    raise ValueError(cfg.backend)


def _build_torch_stack(
    cfg: BenchmarkConfig,
    stack_in: int,
) -> Any:
    """Build the monotone layer stack for the torch backend.

    :param cfg: Benchmark configuration.
    :param stack_in: Number of input features to the stack.
    :returns: ``nn.Sequential`` of monotone layers, and the output width.
    """
    from torch import nn

    from mononet.torch import MonoLinear, MonoResidual

    mono_layers: list[nn.Module] = []
    prev = stack_in
    if cfg.residual:
        mono_layers.append(
            MonoLinear(
                prev,
                cfg.width,
                mode=cfg.mode,
                activation=cfg.activation,
            )
        )
        prev = cfg.width
        for _ in range(cfg.depth):
            mono_layers.append(
                MonoResidual(
                    prev,
                    cfg.width,
                    mode=cfg.mode,
                    activation=cfg.activation,
                )
            )
            prev = cfg.width
    else:
        for _ in range(cfg.depth):
            mono_layers.append(
                MonoLinear(
                    prev,
                    cfg.width,
                    mode=cfg.mode,
                    activation=cfg.activation,
                )
            )
            prev = cfg.width
    return nn.Sequential(*mono_layers), prev


def _build_torch(cfg: BenchmarkConfig, bundle: DatasetBundle) -> Any:
    """Build a torch embedding-composition monotonic model.

    :param cfg: Benchmark configuration.
    :param bundle: Dataset bundle.
    :returns: ``nn.Module`` in float64 mode.
    """
    import torch
    from torch import nn

    from mononet.core.types import MonotonicityMask
    from mononet.torch import MonoInput, MonoLinear

    mono_cols = list(mono_columns(bundle))
    free_cols = list(free_columns(bundle))
    signs = mono_signs(bundle)
    binary = bundle.task == "binary_classification"

    # unconstrained branch: build embed dims
    free_layers: list[nn.Module] = []
    in_f = len(free_cols)
    embed_out = 0
    if free_cols:
        for h in cfg.embed_hidden:
            free_layers += [nn.Linear(in_f, h), nn.ELU()]
            if cfg.dropout:
                free_layers.append(nn.Dropout(cfg.dropout))
            in_f = h
        embed_out = in_f

    stack_in = len(mono_cols) + embed_out
    mono_stack, stack_out = _build_torch_stack(cfg, stack_in)

    class Model(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.mono_cols = torch.tensor(mono_cols, dtype=torch.long)
            self.free_cols = torch.tensor(free_cols, dtype=torch.long)
            self.mono_input = (
                MonoInput(MonotonicityMask(signs)) if mono_cols else None
            )
            self.free_mlp = nn.Sequential(*free_layers) if free_layers else None
            self.mono_stack = mono_stack
            self.head = MonoLinear(stack_out, 1, mode=cfg.mode)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Apply embedding-composition forward pass.

            :param x: Input tensor of shape ``(batch, features)``.
            :returns: Output tensor of shape ``(batch, 1)``.
            """
            parts: list[torch.Tensor] = []
            if self.mono_input is not None:
                parts.append(
                    self.mono_input(
                        x.index_select(1, self.mono_cols.to(x.device))
                    )
                )
            if self.free_mlp is not None:
                parts.append(
                    self.free_mlp(
                        x.index_select(1, self.free_cols.to(x.device))
                    )
                )
            z = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
            y = self.head(self.mono_stack(z))
            return torch.sigmoid(y) if binary else y

    return Model().double()


def _build_jax_embed(
    cfg: BenchmarkConfig,
    free_cols: list[int],
    rngs: Any,
) -> tuple[list[Any], list[bool], int]:
    """Build the unconstrained embedding layers for the jax backend.

    :param cfg: Benchmark configuration.
    :param free_cols: List of non-monotone column indices.
    :param rngs: Flax NNX RNG container.
    :returns: Tuple of (layer list, is_linear flags, embed_out width).
    """
    from flax import nnx

    raw_embed: list[Any] = []
    embed_is_linear: list[bool] = []
    embed_out = 0
    if free_cols:
        in_f = len(free_cols)
        for h in cfg.embed_hidden:
            raw_embed.append(nnx.Linear(in_f, h, rngs=rngs))
            embed_is_linear.append(True)
            if cfg.dropout:
                raw_embed.append(nnx.Dropout(cfg.dropout, rngs=rngs))
                embed_is_linear.append(False)
            in_f = h
        embed_out = in_f
    return raw_embed, embed_is_linear, embed_out


def _build_jax_stack(
    cfg: BenchmarkConfig,
    stack_in: int,
    rngs: Any,
) -> tuple[list[Any], int]:
    """Build the monotone layer stack for the jax backend.

    :param cfg: Benchmark configuration.
    :param stack_in: Number of input features to the stack.
    :param rngs: Flax NNX RNG container.
    :returns: Tuple of (layer list, output width).
    """
    from mononet.jax import MonoLinear, MonoResidual

    raw_mono: list[Any] = []
    prev = stack_in
    if cfg.residual:
        raw_mono.append(
            MonoLinear(
                prev,
                cfg.width,
                mode=cfg.mode,
                activation=cfg.activation,
                convex_fraction=cfg.convex_fraction,
                rngs=rngs,
            )
        )
        prev = cfg.width
        for _ in range(cfg.depth):
            raw_mono.append(
                MonoResidual(
                    prev,
                    cfg.width,
                    mode=cfg.mode,
                    activation=cfg.activation,
                    rngs=rngs,
                )
            )
            prev = cfg.width
    else:
        for _ in range(cfg.depth):
            raw_mono.append(
                MonoLinear(
                    prev,
                    cfg.width,
                    mode=cfg.mode,
                    activation=cfg.activation,
                    convex_fraction=cfg.convex_fraction,
                    rngs=rngs,
                )
            )
            prev = cfg.width
    return raw_mono, prev


def _build_jax(cfg: BenchmarkConfig, bundle: DatasetBundle) -> Any:
    """Build a JAX (Flax NNX) embedding-composition monotonic model.

    :param cfg: Benchmark configuration.
    :param bundle: Dataset bundle.
    :returns: Flax NNX ``Module`` callable returning ``(N, 1)`` output.
    """
    import jax
    import jax.numpy as jnp
    from flax import nnx

    from mononet.core.types import MonotonicityMask
    from mononet.jax import MonoInput, MonoLinear

    mono_cols = list(mono_columns(bundle))
    free_cols = list(free_columns(bundle))
    signs = mono_signs(bundle)
    binary = bundle.task == "binary_classification"

    mono_cols_arr = np.array(mono_cols, dtype=np.int32)
    free_cols_arr = np.array(free_cols, dtype=np.int32)

    rngs = nnx.Rngs(0)
    raw_embed, embed_is_linear, embed_out = _build_jax_embed(cfg, free_cols, rngs)
    stack_in = len(mono_cols) + embed_out
    raw_mono, prev = _build_jax_stack(cfg, stack_in, rngs)

    head = MonoLinear(prev, 1, mode=cfg.mode, rngs=rngs)
    mono_input_layer = MonoInput(MonotonicityMask(signs)) if mono_cols else None

    # Capture in locals for closure (not stored on module to avoid pytree issues)
    _mono_cols = mono_cols_arr
    _free_cols = free_cols_arr
    _binary = binary
    _embed_is_linear = embed_is_linear

    class JaxModel(nnx.Module):
        """Embedding-composition monotonic model (Flax NNX)."""

        def __init__(self) -> None:
            """Initialise model components."""
            self.mono_input = mono_input_layer
            # Use nnx.List so Flax tracks contained modules as data
            self.embed_seq: Any = nnx.List(raw_embed)  # type: ignore[attr-defined]
            self.mono_seq: Any = nnx.List(raw_mono)  # type: ignore[attr-defined]
            self.head = head

        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
            """Apply embedding-composition forward pass.

            :param x: Input array of shape ``(batch, features)``.
            :returns: Output array of shape ``(batch, 1)``.
            """
            parts: list[jnp.ndarray] = []
            if self.mono_input is not None:
                parts.append(self.mono_input(x[:, _mono_cols]))
            if len(self.embed_seq) > 0:
                emb = x[:, _free_cols]
                for i, layer in enumerate(self.embed_seq):
                    emb = (
                        jax.nn.elu(layer(emb)) if _embed_is_linear[i] else layer(emb)
                    )
                parts.append(emb)
            z = jnp.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
            for layer in self.mono_seq:
                z = layer(z)
            y = self.head(z)
            return jax.nn.sigmoid(y) if _binary else y

    return JaxModel()


def _build_keras(cfg: BenchmarkConfig, bundle: DatasetBundle) -> Any:
    """Build a Keras embedding-composition monotonic model.

    :param cfg: Benchmark configuration.
    :param bundle: Dataset bundle.
    :returns: ``keras.Model`` callable returning ``(N, 1)`` output.
    """
    import keras

    from mononet.core.types import MonotonicityMask
    from mononet.keras import MonoDense, MonoInput, MonoResidual

    mono_cols = list(mono_columns(bundle))
    free_cols = list(free_columns(bundle))
    signs = mono_signs(bundle)
    binary = bundle.task == "binary_classification"

    n_features = len(bundle.feature_names)
    # JAX does not support float64 by default (requires JAX_ENABLE_X64);
    # use float32 when the JAX backend is active to avoid truncation warnings.
    _keras_dtype = (
        "float32" if keras.backend.backend() == "jax" else "float64"
    )
    inputs = keras.Input(shape=(n_features,), dtype=_keras_dtype)

    # column selection via Lambda
    parts: list[Any] = []

    if mono_cols:
        mono_x = keras.layers.Lambda(
            lambda x: keras.ops.take(x, mono_cols, axis=1)
        )(inputs)
        mono_x = MonoInput(MonotonicityMask(signs))(mono_x)
        parts.append(mono_x)

    if free_cols:
        emb = keras.layers.Lambda(
            lambda x: keras.ops.take(x, free_cols, axis=1)
        )(inputs)
        for h in cfg.embed_hidden:
            emb = keras.layers.Dense(h, activation="elu")(emb)
            if cfg.dropout:
                emb = keras.layers.Dropout(cfg.dropout)(emb)
        parts.append(emb)

    z = keras.layers.Concatenate()(parts) if len(parts) > 1 else parts[0]

    # monotone stack
    if cfg.residual:
        z = MonoDense(
            cfg.width,
            mode=cfg.mode,
            activation=cfg.activation,
            convex_fraction=cfg.convex_fraction,
        )(z)
        for _ in range(cfg.depth):
            z = MonoResidual(
                cfg.width,
                mode=cfg.mode,
                activation=cfg.activation,
            )(z)
    else:
        for _ in range(cfg.depth):
            z = MonoDense(
                cfg.width,
                mode=cfg.mode,
                activation=cfg.activation,
                convex_fraction=cfg.convex_fraction,
            )(z)

    y = MonoDense(1, mode=cfg.mode)(z)
    if binary:
        y = keras.layers.Activation("sigmoid")(y)

    return keras.Model(inputs=inputs, outputs=y)
