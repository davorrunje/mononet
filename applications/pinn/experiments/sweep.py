# SPDX-License-Identifier: Apache-2.0
"""Run the full experiment matrix and collect metrics.

Enumerates ``problem x method x backend x seed`` (consuming tuned configs when
available) and returns one metrics artifact per cell. The heavy execution is a
``slow``, RUNBOOK-documented step; this module just composes ``run_one``.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import product
from typing import TYPE_CHECKING, Any

from applications.pinn.experiments.config import RunConfig
from applications.pinn.experiments.run import run_one

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from applications.pinn.models.protocol import Backend, Method


def build_matrix(
    problems: Sequence[str],
    methods: Sequence[Method],
    backends: Sequence[Backend],
    seeds: Sequence[int],
    *,
    template: RunConfig | None = None,
) -> list[RunConfig]:
    """Enumerate the experiment matrix as a list of RunConfigs.

    :param problems: Problem keys.
    :param methods: Methods.
    :param backends: Backends.
    :param seeds: Seeds.
    :param template: Base config for shared settings (steps, grids, ...).
    :returns: One RunConfig per cell.
    """
    base = template or RunConfig(
        problem=problems[0], method=methods[0], backend=backends[0]
    )
    return [
        replace(base, problem=p, method=m, backend=b, seed=s)
        for p, m, b, s in product(problems, methods, backends, seeds)
    ]


def run_matrix(configs: Iterable[RunConfig]) -> list[dict[str, Any]]:
    """Run every config and collect its metrics artifact.

    :param configs: RunConfigs to execute.
    :returns: One artifact dict per config.
    """
    return [run_one(cfg) for cfg in configs]
