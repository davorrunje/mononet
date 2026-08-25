# SPDX-License-Identifier: Apache-2.0
"""Optuna hyperparameter search with an equal budget across methods.

The headline claim ("hard beats soft") is only credible if every method is tuned
equally hard. So all methods share the **identical search space and trial
budget**; the soft baseline additionally searches its penalty weight (it must be
tuned, not fixed). The objective is the L2 error of the run against ground truth.
Best configs are frozen to ``configs/`` for the sweep to consume.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import optuna

from applications.pinn.experiments.config import RunConfig
from applications.pinn.experiments.run import run_one
from applications.pinn.models.protocol import ModelConfig

if TYPE_CHECKING:
    from applications.pinn.models.protocol import Backend, Method

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _config_from_trial(trial: optuna.Trial, base: RunConfig) -> RunConfig:
    """Build a RunConfig from a trial's suggested hyperparameters.

    Shared space across methods: ``lr``, ``width``, ``residual_weight`` (the knob
    implicated in shock-smearing / residual divergence), and a data-term weight
    that depends on the tier (``ic_weight`` forward, ``data_weight`` inverse). The
    ``soft`` method additionally searches its penalty weight.
    """
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    width = trial.suggest_categorical("width", [16, 32, 64])
    residual_weight = trial.suggest_float("residual_weight", 1e-2, 1e1, log=True)
    soft_penalty = (
        trial.suggest_float("soft_penalty", 1e-2, 1e1, log=True)
        if base.method == "soft"
        else base.soft_penalty
    )
    if base.tier == "inverse":
        data_weight = trial.suggest_float("data_weight", 1.0, 1e2, log=True)
        return replace(
            base,
            lr=lr,
            residual_weight=residual_weight,
            data_weight=data_weight,
            soft_penalty=soft_penalty,
            model=replace(base.model, width=width),
        )
    ic_weight = trial.suggest_float("ic_weight", 1.0, 100.0, log=True)
    return replace(
        base,
        lr=lr,
        residual_weight=residual_weight,
        ic_weight=ic_weight,
        soft_penalty=soft_penalty,
        model=replace(base.model, width=width),
    )


def search(
    problem: str,
    method: Method,
    backend: Backend,
    *,
    n_trials: int,
    template: RunConfig | None = None,
    seed: int = 0,
) -> dict[str, float]:
    """Tune hyperparameters for one ``(problem, method, backend)`` cell.

    :param problem: Problem registry key.
    :param method: Method to tune (identical space across methods; soft adds its
        penalty weight).
    :param backend: Backend to tune on.
    :param n_trials: Trial budget (identical across methods).
    :param template: Base config (steps, point counts, eval grid). Defaults to a
        moderate config for the given problem/method/backend.
    :param seed: Sampler seed.
    :returns: The best hyperparameters found.
    """
    base = template or RunConfig(problem=problem, method=method, backend=backend)
    base = replace(base, problem=problem, method=method, backend=backend)

    def objective(trial: optuna.Trial) -> float:
        cfg = _config_from_trial(trial, base)
        return float(run_one(cfg)["l2"])

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials)
    best: dict[str, float] = dict(study.best_params)
    return best


def freeze(
    problem: str,
    method: Method,
    backend: Backend,
    best: dict[str, float],
    *,
    configs_dir: str | Path,
) -> Path:
    """Write a tuned config to ``configs/<problem>_<method>_<backend>.json``.

    :param problem: Problem key.
    :param method: Method.
    :param backend: Backend.
    :param best: Best hyperparameters from :func:`search`.
    :param configs_dir: Directory to write into.
    :returns: The path written.
    """
    payload = {
        "problem": problem,
        "method": method,
        "backend": backend,
        "best_params": best,
    }
    path = Path(configs_dir) / f"{problem}_{method}_{backend}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def tuned_config(
    problem: str, method: Method, backend: Backend, path: Path
) -> RunConfig:
    """Reconstruct a RunConfig from a frozen tuned-config file.

    :param problem: Problem key.
    :param method: Method.
    :param backend: Backend.
    :param path: Path to the frozen JSON.
    :returns: A RunConfig with the tuned hyperparameters applied.
    """
    best = json.loads(Path(path).read_text())["best_params"]
    base = RunConfig(problem=problem, method=method, backend=backend)
    return replace(
        base,
        lr=float(best.get("lr", base.lr)),
        ic_weight=float(best.get("ic_weight", base.ic_weight)),
        soft_penalty=float(best.get("soft_penalty", base.soft_penalty)),
        model=ModelConfig(width=int(best.get("width", base.model.width))),
    )
