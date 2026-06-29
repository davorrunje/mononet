"""Typer CLI to run the Phase-2a flavor search.

Invoke: `uv run python -m benchmarks.search [options]`
(or `tools/mononet-benchmark-search`). Repo-only; not a packaged console script.
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 — Typer reads Path annotation at runtime
from typing import Any

import typer

from benchmarks._common.search import _ALL_FLAVORS, flavor_name, run_dataset

app = typer.Typer(add_completion=False, help="Run the Phase-2a HP-search flavor study.")

_ALL_DATASETS = ["auto", "heart", "compas", "loan", "blog"]
_SMOKE: dict[str, Any] = {
    "datasets": ["auto", "heart"],
    "n_trials": 5,
    "epochs": 5,
    "final_seeds": 2,
    "final_top_k": 2,
}


def _parse_flavors(spec: str | None) -> tuple[tuple[str, bool], ...]:
    """Parse a comma-separated flavor spec into ``(mode, residual)`` pairs.

    :param spec: Comma-separated flavor names like ``switch-plain,absolute-residual``,
        or ``None``/empty string to return all flavors.
    :returns: Tuple of ``(mode, residual)`` pairs.
    """
    if not spec:
        return _ALL_FLAVORS
    out: list[tuple[str, bool]] = []
    for name in spec.split(","):
        mode, _, kind = name.partition("-")
        out.append((mode, kind == "residual"))
    return tuple(out)


@app.command()
def main(
    datasets: str = typer.Option(",".join(_ALL_DATASETS), "--datasets"),
    flavors: str = typer.Option(
        "", "--flavors", help="e.g. switch-plain,absolute-residual"
    ),
    backend: str = typer.Option("torch", "--backend"),
    n_trials: int | None = typer.Option(None, "--n-trials"),
    epochs: int = typer.Option(50, "--epochs"),
    n_jobs: int = typer.Option(1, "--n-jobs"),
    final_seeds: int | None = typer.Option(None, "--final-seeds"),
    final_top_k: int | None = typer.Option(None, "--final-top-k"),
    out_dir: Path | None = typer.Option(None, "--out-dir"),  # noqa: B008
    storage_dir: Path | None = typer.Option(None, "--storage-dir"),  # noqa: B008
    smoke: bool = typer.Option(False, "--smoke", help="tiny preset for validation"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="print the plan, run nothing"
    ),
) -> None:
    """Run search + final_eval for each (dataset, flavor) and write result JSON."""
    ds: list[str] = (
        _SMOKE["datasets"] if smoke else [d for d in datasets.split(",") if d]
    )
    nt: int | None = _SMOKE["n_trials"] if smoke else n_trials
    ep: int = _SMOKE["epochs"] if smoke else epochs
    fseeds: int | None = _SMOKE["final_seeds"] if smoke else final_seeds
    ftopk: int | None = _SMOKE["final_top_k"] if smoke else final_top_k
    flavs = _parse_flavors(flavors)
    flav_names = [flavor_name(m, r) for m, r in flavs]

    if dry_run:
        typer.echo(
            f"would run datasets={ds} flavors={flav_names} backend={backend} "
            f"n_trials={nt} epochs={ep} n_jobs={n_jobs}"
        )
        raise typer.Exit(0)

    for dataset in ds:
        paths = run_dataset(
            dataset,
            backend=backend,
            flavors=flavs,
            n_trials=nt,
            epochs=ep,
            n_jobs=n_jobs,
            final_seeds=range(fseeds) if fseeds is not None else None,
            final_top_k=ftopk,
            out_dir=out_dir,
            storage_dir=storage_dir,
        )
        typer.echo(f"{dataset}: wrote {len(paths)} result files")


if __name__ == "__main__":
    app()
