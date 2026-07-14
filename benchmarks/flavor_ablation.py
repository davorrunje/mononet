r"""Fixed-architecture flavor ablation grid (mixed / alternate / split).

A standalone grid sweep — **not** the Optuna flavor-search path
(:mod:`benchmarks.search`). Each cell ``(dataset, mode, alt_init, activation,
depth)`` trains at a *fixed* architecture (width / LR / seeds held constant so
the flavor effect is isolated, not confounded with per-cell tuning) over N
seeds via :func:`benchmarks._common.runner.run`, and aggregates the primary
metric + dispersion (IQR) + convergence (epochs-to-best) + divergence-rate. A
collapse pre-check records whether the model is dead at init.

Topology is **plain** for the focused run (residual + alternate is out of
scope — see the spec). See
``docs/superpowers/specs/2026-07-14-flavor-ablation-benchmark-design.md``.

Run one dataset::

    uv run --group bench python -m benchmarks.flavor_ablation \\
        --dataset heart --backend torch \\
        --out-dir benchmarks/results/flavor-ablation

LR mini-sweep (fixes depth 8, sweeps LR)::

    uv run --group bench python -m benchmarks.flavor_ablation \\
        --dataset heart --backend torch --lr-sweep \\
        --out-dir benchmarks/results/flavor-ablation
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from benchmarks._common.config import (
    BenchmarkConfig,
    EarlyStoppingSpec,
    OptimizerSpec,
)
from benchmarks._common.results import ResultRow, interquartile_mean
from benchmarks._common.runner import run

if TYPE_CHECKING:
    from benchmarks._common.bundle import DatasetBundle

_ACTS: tuple[str, ...] = ("relu", "elu", "softplus", "selu")
_DEPTHS: tuple[int, ...] = (4, 8, 16)
_FLAVORS: tuple[tuple[str, str | None], ...] = (
    ("mixed", None),
    ("split", None),
    ("alternate", "composition"),
    ("alternate", "legacy"),
)
_FOCUSED_DATASETS: tuple[str, ...] = (
    "heart",
    "auto",
    "synth_lattice_clow",
    "synth_lattice_cmid",
    "synth_lattice_chigh",
)
_LR_GRID: tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
_LR_SWEEP_DEPTH = 8

# Fixed architecture — held constant across cells so the flavor is the only
# moving part.
_WIDTH = 32
_BASE_LR = 1e-3
_EMBED_HIDDEN: tuple[int, ...] = (32,)
_BATCH = 256
_EPOCHS = 300
_PATIENCE = 30
_N_SEEDS = 5


@dataclass(frozen=True, slots=True)
class Cell:
    """One grid cell of the ablation (plain topology).

    :param mode: Construction mode (``"mixed"`` / ``"split"`` / ``"alternate"``).
    :param alt_init: Init arm for ``alternate`` (``"composition"`` / ``"legacy"``);
        ``None`` for the other modes.
    :param activation: Activation name.
    :param depth: Number of stack layers.
    """

    mode: str
    alt_init: str | None
    activation: str
    depth: int


def ablation_cells(focused: bool = True) -> list[Cell]:
    """Enumerate the plain-topology grid cells.

    :param focused: Reserved for the (currently identical) focused-vs-full split;
        the focused run and the full run share the same per-dataset cell grid,
        differing only in the dataset list.
    :returns: The list of :class:`Cell` cells (flavor x activation x depth).
    """
    return [
        Cell(mode, alt_init, act, depth)
        for (mode, alt_init) in _FLAVORS
        for act in _ACTS
        for depth in _DEPTHS
    ]


def _load_bundle(dataset: str) -> DatasetBundle:
    """Load a dataset bundle from the default dataset cache.

    Generator-backed synthetic datasets ignore ``data_dir``; real datasets
    (heart, auto) read their train/test CSVs from :func:`default_dest` — where
    :mod:`benchmarks.datasets.download` places the Zenodo files. (Respects
    ``$MONONET_DATA_DIR`` via ``default_dest``.)

    :param dataset: Dataset key.
    :returns: The loaded :class:`DatasetBundle`.
    :raises FileNotFoundError: If a real dataset's CSVs are absent (run
        ``python -m benchmarks.datasets.download``).
    """
    from benchmarks.datasets.download import default_dest
    from benchmarks.datasets.registry import load

    return load(dataset, data_dir=default_dest())


def _subsample(bundle: DatasetBundle, n: int) -> DatasetBundle:
    """Return a copy of ``bundle`` with train/test truncated to ``n`` rows (smoke).

    :param bundle: Source bundle.
    :param n: Row cap for each split.
    :returns: A truncated copy.
    """
    return dataclasses.replace(
        bundle,
        X_train=bundle.X_train[:n],
        y_train=bundle.y_train[:n],
        X_test=bundle.X_test[: max(1, n // 2)],
        y_test=bundle.y_test[: max(1, n // 2)],
    )


def _cell_config(
    dataset: str, backend: str, cell: Cell, lr: float, *, binary: bool, smoke: bool
) -> BenchmarkConfig:
    """Build the fixed-architecture :class:`BenchmarkConfig` for one cell.

    :param dataset: Dataset key.
    :param backend: Backend name.
    :param cell: The grid cell.
    :param lr: Learning rate for this run.
    :param binary: Whether the task is binary classification.
    :param smoke: If True, use 1 seed / 2 epochs / no early stopping (fast test).
    :returns: The configuration.
    """
    return BenchmarkConfig(
        dataset=dataset,
        backend=backend,  # type: ignore[arg-type]
        mode=cell.mode,  # type: ignore[arg-type]
        residual=False,
        depth=cell.depth,
        width=_WIDTH,
        activation=cell.activation,  # type: ignore[arg-type]
        convex_fraction=0.5,
        embed_hidden=_EMBED_HIDDEN,
        dropout=0.0,
        optimizer=OptimizerSpec(name="adam", lr=lr),
        lr_decay=None,
        batch_size=_BATCH,
        epochs=2 if smoke else _EPOCHS,
        early_stopping=None
        if smoke
        else EarlyStoppingSpec(monitor="val", patience=_PATIENCE),
        seeds=(0,) if smoke else tuple(range(_N_SEEDS)),
        metrics=("roc_auc", "accuracy") if binary else ("mse", "rmse"),
        alt_init=cell.alt_init,  # type: ignore[arg-type]
    )


def _forward_np(model: Any, x: np.ndarray, backend: str) -> np.ndarray:  # type: ignore[type-arg]
    """Run a backend-native forward pass and return a 1-D NumPy array.

    :param model: Backend-native model from :func:`build_model`.
    :param x: Input batch (float32 NumPy).
    :param backend: Backend name.
    :returns: Flat float64 predictions.
    """
    if backend == "torch":
        import torch

        with torch.no_grad():
            out = model(torch.tensor(x, dtype=torch.float32)).cpu().numpy()
    elif backend == "jax":
        import jax.numpy as jnp

        out = np.asarray(model(jnp.asarray(x, dtype=jnp.float32)))
    elif backend == "keras":
        out = np.asarray(model.predict(x, verbose=0))
    else:
        raise ValueError(f"Unknown backend: {backend!r}")
    return np.asarray(out, dtype=np.float64).ravel()


def _collapse_precheck(cfg: BenchmarkConfig, bundle: DatasetBundle) -> bool:
    """Whether the model is dead at init (non-finite or ~constant output).

    Builds the model (seed 0), forwards a small real batch, and checks for
    non-finite outputs or near-zero output variance — the init-time collapse
    the composition-aware init is meant to prevent.

    :param cfg: The cell configuration.
    :param bundle: The dataset bundle.
    :returns: ``True`` if the model appears collapsed at init.
    """
    from benchmarks._common.model_builder import build_model
    from benchmarks._common.seeds import seed_everything

    seed_everything(cfg.backend, 0)
    model = build_model(cfg, bundle, seed=0)
    out = _forward_np(model, bundle.X_train[:64].astype("float32"), cfg.backend)
    if not np.all(np.isfinite(out)):
        return True
    return bool(np.var(out) < 1e-12)


def _aggregate(rows: list[ResultRow], primary: str) -> dict[str, Any]:
    """Aggregate per-seed rows into the cell record's statistics.

    :param rows: Per-seed result rows for one cell.
    :param primary: Primary metric key (``"roc_auc"`` or ``"mse"``).
    :returns: Dict of ``metric_iqm`` / ``metric_iqr`` / ``epochs_median`` /
        ``divergence_rate`` / ``n_seeds``.
    """
    vals = np.array([r.scores[primary] for r in rows], dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size:
        iqm = interquartile_mean(finite)
        iqr = float(np.subtract(*np.percentile(finite, [75, 25])))
    else:
        iqm = float("nan")
        iqr = float("nan")
    epochs = np.array([r.epochs_run for r in rows], dtype=float)
    return {
        "metric_iqm": iqm,
        "metric_iqr": iqr,
        "epochs_median": float(np.median(epochs)),
        "divergence_rate": float(np.mean([r.diverged for r in rows])),
        "n_seeds": len(rows),
    }


def run_cell(
    dataset: str,
    backend: str,
    cell: Cell,
    lr: float,
    bundle: DatasetBundle,
    *,
    smoke: bool,
) -> dict[str, Any]:
    """Train + aggregate one grid cell.

    :param dataset: Dataset key.
    :param backend: Backend name.
    :param cell: The grid cell.
    :param lr: Learning rate.
    :param bundle: The dataset bundle.
    :param smoke: Fast-test mode.
    :returns: The aggregated cell record (JSON-serialisable).
    """
    binary = bundle.task == "binary_classification"
    primary = "roc_auc" if binary else "mse"
    cfg = _cell_config(dataset, backend, cell, lr, binary=binary, smoke=smoke)
    collapsed = _collapse_precheck(cfg, bundle)
    rows = run(cfg, bundle)
    rec: dict[str, Any] = {
        "dataset": dataset,
        "backend": backend,
        "topology": "plain",
        "mode": cell.mode,
        "alt_init": cell.alt_init,
        "activation": cell.activation,
        "depth": cell.depth,
        "lr": lr,
        "primary": primary,
        "collapsed": collapsed,
    }
    rec.update(_aggregate(rows, primary))
    return rec


def run_dataset_ablation(
    dataset: str,
    backend: str,
    *,
    lr_sweep: bool,
    out_dir: Path | str,
    smoke: bool = False,
) -> Path:
    """Run the full focused grid for one dataset and write the records JSON.

    :param dataset: Dataset key.
    :param backend: Backend name.
    :param lr_sweep: If True, fix depth 8 and sweep the LR grid; else base LR only.
    :param out_dir: Output directory for the ``<dataset>[-lrsweep].json`` file.
    :param smoke: Fast-test mode (subsampled data, 1 seed, 2 epochs).
    :returns: The path to the written records JSON.
    """
    bundle = _load_bundle(dataset)
    if smoke:
        bundle = _subsample(bundle, 128)
    cells = ablation_cells(focused=True)
    lrs = _LR_GRID if lr_sweep else (_BASE_LR,)
    if lr_sweep:
        cells = [c for c in cells if c.depth == _LR_SWEEP_DEPTH]
    recs = [
        run_cell(dataset, backend, c, lr, bundle, smoke=smoke)
        for c in cells
        for lr in lrs
    ]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "-lrsweep" if lr_sweep else ""
    path = out_dir / f"{dataset}{suffix}.json"
    path.write_text(json.dumps(recs, indent=2) + "\n")
    return path


def main() -> None:
    """CLI entry point for a single-dataset ablation run."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=_FOCUSED_DATASETS)
    ap.add_argument("--backend", default="torch", choices=("torch", "jax", "keras"))
    ap.add_argument("--lr-sweep", action="store_true")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "flavor-ablation",
    )
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    path = run_dataset_ablation(
        args.dataset,
        args.backend,
        lr_sweep=args.lr_sweep,
        out_dir=args.out_dir,
        smoke=args.smoke,
    )
    print(f"wrote {path}")  # noqa: T201


if __name__ == "__main__":
    main()
