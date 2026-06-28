r"""CLI entry point: train and evaluate a mononet benchmark run.

Usage::

    python -m benchmarks.run \
        --dataset auto \
        --backend torch \
        --mode switch \
        [--residual] \
        [--epochs N] \
        [--seeds 0 1 2 ...] \
        [--data-dir DIR] \
        [--out results.jsonl]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

from benchmarks._common.config import BenchmarkConfig, OptimizerSpec
from benchmarks._common.results import ResultRow, aggregate, write_jsonl
from benchmarks._common.runner import run
from benchmarks.datasets.download import default_dest
from benchmarks.datasets.registry import load

# ---------------------------------------------------------------------------
# Sensible defaults for a single CLI invocation
# ---------------------------------------------------------------------------

_DEFAULT_DEPTH = 2
_DEFAULT_WIDTH = 64
_DEFAULT_ACTIVATION = "elu"
_DEFAULT_CONVEX_FRACTION = 0.5
_DEFAULT_EMBED_HIDDEN: tuple[int, ...] = (64,)
_DEFAULT_DROPOUT = 0.0
_DEFAULT_LR = 1e-3
_DEFAULT_WEIGHT_DECAY = 0.0
_DEFAULT_BATCH_SIZE = 256
_DEFAULT_EPOCHS = 50
_DEFAULT_SEEDS = (0, 1, 2, 3, 4)


def _build_config(args: argparse.Namespace, task: str) -> BenchmarkConfig:
    """Construct a :class:`~benchmarks._common.config.BenchmarkConfig` from CLI args.

    Task 8 will introduce ``config_io`` to load per-dataset TOML defaults.
    Until then, sensible defaults are hardcoded here with CLI overrides applied.

    :param args: Parsed CLI namespace.
    :param task: Dataset task string (e.g. ``"regression"`` or
        ``"binary_classification"``), used to select appropriate metrics.
    :returns: Fully specified benchmark configuration.
    """
    # Seam: per-dataset TOML defaults (benchmarks._common.config_io) can be
    # layered in here; for now the CLI builds the config from flags + defaults.
    seeds: tuple[int, ...] = tuple(args.seeds) if args.seeds else _DEFAULT_SEEDS
    epochs: int = args.epochs if args.epochs is not None else _DEFAULT_EPOCHS
    backend: Literal["torch", "jax", "keras"] = args.backend
    mode: Literal["switch", "absolute"] = args.mode

    metrics: tuple[Literal["accuracy", "rmse", "mse"], ...] = (
        ("accuracy",) if task == "binary_classification" else ("mse", "rmse")
    )

    return BenchmarkConfig(
        dataset=args.dataset,
        backend=backend,
        mode=mode,
        residual=args.residual,
        depth=_DEFAULT_DEPTH,
        width=_DEFAULT_WIDTH,
        activation=_DEFAULT_ACTIVATION,
        convex_fraction=_DEFAULT_CONVEX_FRACTION,
        embed_hidden=_DEFAULT_EMBED_HIDDEN,
        dropout=_DEFAULT_DROPOUT,
        optimizer=OptimizerSpec("adam", _DEFAULT_LR, _DEFAULT_WEIGHT_DECAY),
        lr_decay=None,
        batch_size=_DEFAULT_BATCH_SIZE,
        epochs=epochs,
        early_stopping=None,
        seeds=seeds,
        metrics=metrics,
    )


def _print_aggregate(rows: list[ResultRow], metric: str) -> None:
    """Print a human-readable aggregate summary.

    :param rows: Completed result rows.
    :param metric: Primary metric to aggregate.
    """
    agg = aggregate(rows, metric=metric, lower_is_better=True, top_k=len(rows))
    print(  # noqa: T201
        f"{metric}: mean={agg.mean:.6f}  std={agg.std:.6f}  "
        f"n={agg.n_seeds}  selected={agg.n_selected}"
    )


def main() -> None:
    """Parse CLI arguments, load data, run training, write JSONL, print aggregate."""
    ap = argparse.ArgumentParser(
        description="Train and evaluate a mononet model on a benchmark dataset."
    )
    ap.add_argument(
        "--dataset",
        required=True,
        help="Dataset name (e.g. 'auto', 'blog', 'compas', 'heart', 'loan').",
    )
    ap.add_argument(
        "--backend",
        required=True,
        choices=["torch", "jax", "keras"],
        help="ML backend to use.",
    )
    ap.add_argument(
        "--mode",
        required=True,
        choices=["switch", "absolute"],
        help="Monotonicity enforcement mode.",
    )
    ap.add_argument(
        "--residual",
        action="store_true",
        default=False,
        help="Use residual connections.",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=f"Training epochs (default: {_DEFAULT_EPOCHS}).",
    )
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=f"Random seeds (default: {list(_DEFAULT_SEEDS)}).",
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing dataset CSVs (default: ~/.cache/mononet/datasets).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="JSONL output file (default: results/<dataset>_<backend>_<mode>.jsonl).",
    )
    args = ap.parse_args()

    data_dir: Path = args.data_dir or default_dest()
    bundle = load(args.dataset, data_dir=data_dir)
    cfg = _build_config(args, bundle.task)

    rows = run(cfg, bundle)

    # Determine output path
    out_path: Path = args.out or Path("results") / (
        f"{cfg.dataset}_{cfg.backend}_{cfg.mode}.jsonl"
    )
    write_jsonl(rows, out_path)
    print(f"Wrote {len(rows)} rows to {out_path}")  # noqa: T201

    # Print aggregate for each metric
    for metric in cfg.metrics:
        _print_aggregate(rows, metric)


if __name__ == "__main__":
    main()
