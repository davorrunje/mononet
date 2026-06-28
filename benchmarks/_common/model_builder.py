"""Embedding-composition model builder over the four mononet flavors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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


def _build_jax(cfg: BenchmarkConfig, bundle: DatasetBundle) -> Any:  # Task 6
    """JAX model builder stub (implemented in Task 6).

    :param cfg: Benchmark configuration.
    :param bundle: Dataset bundle.
    :raises NotImplementedError: Always — JAX builder added in Task 6.
    """
    raise NotImplementedError("jax builder added in Task 6")


def _build_keras(cfg: BenchmarkConfig, bundle: DatasetBundle) -> Any:  # Task 6
    """Keras model builder stub (implemented in Task 6).

    :param cfg: Benchmark configuration.
    :param bundle: Dataset bundle.
    :raises NotImplementedError: Always — Keras builder added in Task 6.
    """
    raise NotImplementedError("keras builder added in Task 6")
