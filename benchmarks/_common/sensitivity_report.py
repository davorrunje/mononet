"""Reconstruct HP-search sensitivity curves from committed Optuna storage.

Curve A (best-so-far search objective) is read directly from storage. Curve B
(test metric of the running incumbent) re-evaluates only the best-so-far
changepoints via `benchmarks._common.search.final_eval` — a bounded re-eval,
never a re-run of the search. See
`docs/superpowers/specs/2026-07-15-hp-search-sensitivity-curves-design.md`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import optuna

from benchmarks._common.search import final_eval

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


def best_so_far(values: list[float], lower: bool) -> list[float]:
    """Cumulative best of a per-trial objective sequence.

    :param values: Per-trial objective values, in trial order.
    :param lower: Whether a lower objective is better (running `min`, else `max`).
    :returns: The running-best sequence, same length as ``values``.
    """
    out: list[float] = []
    best: float | None = None
    for v in values:
        best = v if best is None else (min(best, v) if lower else max(best, v))
        out.append(best)
    return out


def saturation_trial(traj: list[float], lower: bool, p: float = 0.99) -> int:
    """Smallest 1-based trial count reaching fraction ``p`` of the eventual gain.

    With ``G = |traj[-1] - traj[0]|`` the total improvement, returns the first
    ``t`` (1-based) where ``|traj[-1] - traj[t-1]| <= (1 - p) * G``. A flat
    trajectory (``G == 0``) returns ``1``.

    :param traj: A best-so-far trajectory (monotone), as from :func:`best_so_far`.
    :param lower: Whether lower is better (accepted for a symmetric call site and
        future signed variants; the gap magnitude is direction-agnostic).
    :param p: Fraction of the eventual gain to reach (default 0.99).
    :returns: The 1-based saturation trial count ``t*`` (0 for an empty input).
    """
    if not traj:
        return 0
    final = traj[-1]
    gain = abs(final - traj[0])
    if gain == 0.0:
        return 1
    tol = (1.0 - p) * gain
    for i, b in enumerate(traj):
        if abs(final - b) <= tol:
            return i + 1
    return len(traj)


def load_study(db_path: Path, study_name: str) -> optuna.Study:
    """Load an Optuna study from a committed sqlite storage file (read-only use).

    :param db_path: Path to the ``{dataset}-{flavor}.db`` sqlite file.
    :param study_name: The study name it was created under (``{dataset}-{flavor}``).
    :returns: The loaded study.
    """
    return optuna.load_study(study_name=study_name, storage=f"sqlite:///{db_path}")


def completed_values(study: Any, lower: bool) -> list[float]:
    """Objective values of COMPLETE trials, in trial order.

    :param study: A loaded study (or a duck-typed stand-in with ``.trials``).
    :param lower: Whether lower is better (accepted for call-site symmetry).
    :returns: The per-trial objective values for completed trials only.
    """
    return [
        float(t.value)
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]


def incumbent_changepoints(study: Any, lower: bool) -> list[tuple[int, dict[str, Any]]]:
    """`(1-based trial index, params)` at each best-so-far improvement.

    Iterates completed trials in order; emits a changepoint whenever the running
    best strictly improves (including the first completed trial). The index
    counts completed trials only, matching :func:`completed_values`.

    :param study: A loaded study (or duck-typed stand-in with ``.trials``).
    :param lower: Whether a lower objective is better.
    :returns: The improving trials as ``(index, params)`` pairs, in order.
    """
    out: list[tuple[int, dict[str, Any]]] = []
    best: float | None = None
    idx = 0
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE or t.value is None:
            continue
        idx += 1
        v = float(t.value)
        improved = best is None or (v < best if lower else v > best)
        if improved:
            best = v
            out.append((idx, dict(t.params)))
    return out


def incumbent_test_curve(
    study: Any,
    bundle: Any,
    *,
    mode: str,
    residual: bool,
    backend: str,
    lower: bool,
    n_trials: int,
    seeds: Iterable[int],
    embed_layers: int = 2,
) -> tuple[list[float], int]:
    """Test metric of the running incumbent per trial (step-held), plus re-eval count.

    For each best-so-far changepoint, uses the stored ``test_metric`` user-attr
    when present (future ``log_test_trajectory`` runs), otherwise re-evaluates
    that incumbent's params once via :func:`final_eval` (bounded re-eval — one
    call per distinct incumbent, never per trial). The returned curve holds each
    incumbent's value until the next changepoint.

    :param study: Loaded study (or duck-typed stand-in with ``.trials``).
    :param bundle: The dataset bundle to re-evaluate on.
    :param mode: Flavor mode (``split``/``mixed``/``alternate``).
    :param residual: Whether the flavor is residual.
    :param backend: Backend name passed to :func:`final_eval`.
    :param lower: Whether lower objective is better.
    :param n_trials: Total completed-trial count (curve length).
    :param seeds: Final-eval seeds (match the base run's per-dataset count).
    :param embed_layers: Non-monotone embedding depth (base run used 2).
    :returns: ``(curve, n_incumbents_reevaluated)``.
    """
    seeds = list(seeds)
    cps = incumbent_changepoints(study, lower=lower)
    completed = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    n_eval = 0
    values_at_cp: list[tuple[int, float]] = []
    for idx, params in cps:
        stored = completed[idx - 1].user_attrs.get("test_metric")
        if stored is not None:
            metric_val = float(stored)
        else:
            agg, _ = final_eval(
                bundle,
                params,
                mode=mode,
                residual=residual,
                backend=backend,
                seeds=seeds,
                embed_layers=embed_layers,
            )
            metric_val = float(agg.metric)
            n_eval += 1
        values_at_cp.append((idx, metric_val))
    curve: list[float] = []
    cur = values_at_cp[0][1] if values_at_cp else float("nan")
    j = 0
    for t_idx in range(1, n_trials + 1):
        while j < len(values_at_cp) and values_at_cp[j][0] == t_idx:
            cur = values_at_cp[j][1]
            j += 1
        curve.append(cur)
    return curve, n_eval
