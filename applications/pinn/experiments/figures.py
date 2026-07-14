# SPDX-License-Identifier: Apache-2.0
r"""Generate the paper figures from committed result artifacts + fresh reconstructions.

Two figure families:

1. **Crossover curves** (cheap, from the sweep JSONs): per-method L1/L2 error and
   admissibility violation vs observation noise, at a fixed observation count.
2. **Reconstruction profiles** (trains the tuned configs): a fixed-time spatial
   slice at a stress operating point, showing the hard-monotone field as a clean
   monotone ramp while the unconstrained/soft baselines oscillate across the shock,
   with the sparse noisy observations overlaid.

Each figure is written to ``applications/pinn/paper/figures/`` in **both** vector
PDF (for LaTeX ``\includegraphics``) and PNG (for the markdown/Sphinx preview).
Reconstructions need a JAX backend (fast on ``gpu-jax``; fine on CPU for the six
single-seed trainings).

Example::

    uv run python -m applications.pinn.experiments.figures
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from applications.pinn.core import exact, metrics, plotting
from applications.pinn.core.admissibility import violation
from applications.pinn.core.problems import get
from applications.pinn.core.problems.traffic_real import _DEFAULT_NPZ
from applications.pinn.experiments import baselines
from applications.pinn.experiments.config import RunConfig
from applications.pinn.experiments.run import predict_field

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from applications.pinn.models.protocol import Method

# Embed real (Type-42) fonts rather than Type-3 outlines -- required by most
# publishers, and keeps text selectable/searchable in the vector PDF.
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"

_FIG_DIR = Path("applications/pinn/paper/figures")
_RESULTS = Path("applications/pinn/results")
# Vector PDF for LaTeX \includegraphics; PNG for the markdown/Sphinx preview.
_FORMATS = ("pdf", "png")
# Baselines shown in the paper (weight_clip is a diagnostic, omitted from figures).
_METHODS: tuple[Method, ...] = ("hard_monotone", "vanilla", "soft")
_LABELS = {
    "hard_monotone": "hard-monotone (ours)",
    "vanilla": "vanilla PINN",
    "soft": "soft-penalty",
}
# Stress operating point for the reconstruction money-shot.
_STRESS_N_OBS = 80
_STRESS_NOISE = 0.20

# Real-NGSIM (detector) reconstruction: matches RunConfig's detector-mode defaults
# (results/real-ngsim.json does not itself record these, since run_panel's output
# dict only carries tuned per-method params, not the observation-mode knobs).
_NGSIM_N_DETECTORS = 8
_NGSIM_N_HOLDOUT_DETECTORS = 4


def _save(fig: Figure, stem: Path) -> None:
    """Write ``fig`` as vector PDF (LaTeX) and PNG (preview) under ``stem``."""
    for ext in _FORMATS:
        out = stem.with_suffix(f".{ext}")
        # dpi only affects the raster PNG; the PDF is vector regardless.
        fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {stem}.{{{','.join(_FORMATS)}}}", flush=True)


def _crossover(sweep_path: Path, key: str, ylabel: str, title: str, stem: Path) -> None:
    """Build a per-method ``key``-vs-noise curve at ``_STRESS_N_OBS`` observations."""
    data = json.loads(sweep_path.read_text())
    noise = sorted({float(c["noise"]) for c in data["cells"]})
    series = {}
    for method in _METHODS:
        vals = {
            float(c["noise"]): float(c[key])
            for c in data["cells"]
            if c["method"] == method and int(c["n_obs"]) == _STRESS_N_OBS
        }
        series[_LABELS[method]] = np.array([vals[n] for n in noise])
    fig = plotting.metric_vs_noise(np.array(noise), series, ylabel=ylabel, title=title)
    _save(fig, stem)


def _reconstruction(
    problem: str, tuned_path: Path, pretty: str, stem: Path, steps: int
) -> None:
    """Train each method at the stress point and plot a fixed-time spatial slice."""
    tuned = json.loads(tuned_path.read_text())
    series: dict[str, np.ndarray[Any, Any]] = {}
    x_values = t_values = None
    obs_slice: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]] | None = None
    for method in _METHODS:
        best = tuned["methods"][method]["best_params"]
        cfg = RunConfig(
            problem=problem,
            method=method,
            backend="jax",
            tier="inverse",
            steps=steps,
            grad_clip=1.0,
            seed=0,
            n_obs=_STRESS_N_OBS,
            noise_std=_STRESS_NOISE,
            lr=float(best["lr"]),
            residual_weight=float(best["residual_weight"]),
            data_weight=float(best.get("data_weight", 1.0)),
            soft_penalty=float(best.get("soft_penalty", 0.0)),
        )
        cfg = replace(cfg, model=replace(cfg.model, width=int(best["width"])))
        r = predict_field(cfg)
        x_values, t_values = r.x_values, r.t_values
        # Fixed time slice near the end, where the shock is fully developed.
        ti = int(0.85 * (len(t_values) - 1))
        if "true" not in series:
            series["true"] = r.ref[ti]
        series[_LABELS[method]] = r.pred[ti]
        if obs_slice is None and r.obs is not None:
            oc, ov = r.obs
            band = 0.12 * float(t_values[-1] - t_values[0])
            near = np.abs(oc[:, 1] - float(t_values[ti])) <= band
            obs_slice = (oc[near, 0], ov[near])
    assert x_values is not None
    assert t_values is not None
    ti = int(0.85 * (len(t_values) - 1))
    fig = plotting.reconstruction_profile(
        x_values,
        series,
        obs=obs_slice,
        title=(
            f"{pretty}: inverse reconstruction at t={float(t_values[ti]):.2f}\n"
            f"(n_obs={_STRESS_N_OBS}, noise={_STRESS_NOISE})"
        ),
    )
    _save(fig, stem)


def _ngsim_baseline_fields(
    problem: object,
    x_values: np.ndarray[Any, Any],
    t_values: np.ndarray[Any, Any],
    obs: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]],
) -> dict[str, np.ndarray[Any, Any]]:
    """Reconstruct the classical ASM and RBF-smoother baselines from ``obs``.

    Both are evaluated on the same detectors as the PINN methods, so the
    comparison in the reconstruction/metric-bar figures is apples-to-apples.

    :param problem: The ``ngsim_wave`` problem instance (calibrated Greenshields
        FD parameters).
    :param x_values: Spatial evaluation axis.
    :param t_values: Temporal evaluation axis.
    :param obs: Detector ``(coords, values)`` shared with the PINN methods.
    :returns: Mapping of ``"ASM"`` / ``"RBF smoother"`` to the reconstructed
        field, shape ``(len(t_values), len(x_values))``.
    """
    oc, ov = obs
    v_free = float(problem.v_max)  # type: ignore[attr-defined]
    rho_max = float(problem.rho_max)  # type: ignore[attr-defined]
    # Greenshields congested characteristic speed at jam density: -v_free.
    v_cong = float(exact.greenshields_flux_prime(np.asarray(rho_max), v_free, rho_max))
    x_extent = float(x_values[-1] - x_values[0])
    t_extent = float(t_values[-1] - t_values[0])
    asm = baselines.adaptive_smoothing(
        oc,
        ov,
        x_values,
        t_values,
        v_free=v_free,
        v_cong=v_cong,
        sigma=0.15 * x_extent,
        tau=0.15 * t_extent,
    )

    bounds = (
        float(x_values[0]),
        float(x_values[-1]),
        float(t_values[0]),
        float(t_values[-1]),
    )
    ocn = baselines._norm(oc, bounds)
    lam = baselines._best_lambda(ocn, ov, seed=0)
    grid_x, grid_t = np.meshgrid(x_values, t_values)
    grid = baselines._norm(np.column_stack([grid_x.ravel(), grid_t.ravel()]), bounds)
    rbf = baselines._fit_predict(ocn, ov, grid, lam).reshape(
        len(t_values), len(x_values)
    )
    return {"ASM": asm, "RBF smoother": rbf}


def _field_metrics(
    pred: np.ndarray[Any, Any],
    ref: np.ndarray[Any, Any],
    x_values: np.ndarray[Any, Any],
    t_values: np.ndarray[Any, Any],
    sign_x: int,
    holdout: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]],
) -> dict[str, float]:
    """Score a whole-field baseline reconstruction like ``run_one`` scores a PINN.

    :param pred: Baseline field, shape ``(len(t_values), len(x_values))``.
    :param ref: Ground-truth field on the same grid.
    :param x_values: Spatial evaluation axis.
    :param t_values: Temporal evaluation axis.
    :param sign_x: Admissible sign of ``du/dx``.
    :param holdout: Held-out detector ``(coords, values)`` for RMSE.
    :returns: ``{"l1", "l2", "admissibility_violation", "held_out_rmse"}``.
    """
    dx = float(x_values[1] - x_values[0])
    viol = max(violation(pred[i], axis=0, sign=sign_x) for i in range(len(t_values)))
    hc, hv = holdout
    interp = RegularGridInterpolator(
        (t_values, x_values), pred, bounds_error=False, fill_value=None
    )
    hpred = np.asarray(interp(np.column_stack([hc[:, 1], hc[:, 0]])))
    hrmse = float(np.sqrt(np.mean((hpred - hv) ** 2)))
    return {
        "l1": metrics.l1(pred, ref, dx=dx),
        "l2": metrics.l2(pred, ref, dx=dx),
        "admissibility_violation": viol,
        "held_out_rmse": hrmse,
    }


def _real(tuned_path: Path, steps: int) -> None:
    """Real-NGSIM figures: reconstruction slice, window heatmap, metric bars.

    Trains hard/vanilla/soft on detector observations of the real NGSIM window
    at the tuned params from ``real-ngsim.json`` (mirrors :func:`_reconstruction`),
    reconstructs the same detectors with the classical ASM and RBF-smoother
    baselines (:func:`_ngsim_baseline_fields`), and plots a fixed-time
    reconstruction slice, a window heatmap annotated with the field's
    monotonicity defect, and a grouped metric-bar comparison across all five
    methods. No-ops (prints a skip line) unless both the committed ``.npz`` and
    ``tuned_path`` exist, so this is inert in CI.

    :param tuned_path: Path to the real-NGSIM headline JSON (Task 7 output).
    :param steps: Optimisation steps for each PINN training.
    """
    npz_path = Path(_DEFAULT_NPZ)
    if not npz_path.exists() or not tuned_path.exists():
        print(f"skip real: {npz_path} or {tuned_path} missing", flush=True)
        return

    tuned = json.loads(tuned_path.read_text())
    problem = get("ngsim_wave")()

    fields: dict[str, np.ndarray[Any, Any]] = {}
    bar_data: dict[str, dict[str, float]] = {}
    x_values = t_values = ref = None
    obs: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]] | None = None
    holdout: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]] | None = None
    sign_x = 0
    for method in _METHODS:
        best = tuned["methods"][method]["best_params"]
        cfg = RunConfig(
            problem="ngsim_wave",
            method=method,
            backend="jax",
            tier="inverse",
            steps=steps,
            grad_clip=1.0,
            seed=0,
            observations="detectors",
            n_detectors=_NGSIM_N_DETECTORS,
            n_holdout_detectors=_NGSIM_N_HOLDOUT_DETECTORS,
            lr=float(best["lr"]),
            residual_weight=float(best["residual_weight"]),
            data_weight=float(best.get("data_weight", 1.0)),
            soft_penalty=float(best.get("soft_penalty", 0.0)),
        )
        cfg = replace(cfg, model=replace(cfg.model, width=int(best["width"])))
        r = predict_field(cfg)
        x_values, t_values, ref, sign_x = r.x_values, r.t_values, r.ref, r.sign_x
        assert r.obs is not None
        obs = r.obs
        assert r.holdout is not None
        hc, hv, _hpred = r.holdout
        holdout = (hc, hv)
        fields[_LABELS[method]] = r.pred
        agg = tuned["methods"][method]["agg"]
        bar_data[_LABELS[method]] = {
            key: float(agg[key]["iqm"])
            for key in ("l1", "l2", "admissibility_violation", "held_out_rmse")
        }
    assert x_values is not None
    assert t_values is not None
    assert ref is not None
    assert obs is not None
    assert holdout is not None

    baseline_fields = _ngsim_baseline_fields(problem, x_values, t_values, obs)
    fields.update(baseline_fields)
    for label, field in baseline_fields.items():
        bar_data[label] = _field_metrics(
            field, ref, x_values, t_values, sign_x, holdout
        )

    # Reconstruction slice at mid-window (real data has no single "shock" point).
    ti = len(t_values) // 2
    oc, ov = obs
    band = 0.12 * float(t_values[-1] - t_values[0])
    near = np.abs(oc[:, 1] - float(t_values[ti])) <= band
    series = {"true": ref[ti], **{label: field[ti] for label, field in fields.items()}}
    fig_recon = plotting.reconstruction_profile(
        x_values,
        series,
        obs=(oc[near, 0], ov[near]),
        title=f"NGSIM real-data reconstruction at t={float(t_values[ti]):.2f}",
        ylabel="density",
    )
    _save(fig_recon, _FIG_DIR / "reconstruction-ngsim")

    defect = float(problem.monotonicity_defect)  # type: ignore[attr-defined]
    fig_heatmap = plotting.field_heatmap(
        ref,
        x_values,
        t_values,
        title=f"NGSIM real-data window (monotonicity defect={defect:.4f})",
    )
    _save(fig_heatmap, _FIG_DIR / "window-ngsim")

    labels = list(bar_data.keys())
    groups = {
        "L1": [bar_data[m]["l1"] for m in labels],
        "L2": [bar_data[m]["l2"] for m in labels],
        "held-out RMSE": [bar_data[m]["held_out_rmse"] for m in labels],
        "admissibility violation": [
            bar_data[m]["admissibility_violation"] for m in labels
        ],
    }
    fig_bars = plotting.metric_bars(
        labels, groups, ylabel="error", title="NGSIM real-data comparison"
    )
    _save(fig_bars, _FIG_DIR / "metrics-ngsim")


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument(
        "--no-reconstructions",
        action="store_true",
        help="Only rebuild crossover curves from JSONs (no training).",
    )
    p.add_argument(
        "--real",
        action="store_true",
        help=(
            "Also generate the real-NGSIM figures (reconstruction/window/"
            "metric-bars). No-ops unless the committed .npz and "
            "results/real-ngsim.json are both present."
        ),
    )
    args = p.parse_args()
    _FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Crossover curves (from committed sweeps) -- LWR and Burgers, L1 and L2.
    panels = [
        ("inverse-sweep.json", "LWR (traffic, concave flux)", "lwr"),
        ("inverse-burgers-sweep.json", "Burgers (convex flux)", "burgers"),
    ]
    for fname, pretty, tag in panels:
        path = _RESULTS / fname
        if not path.exists():
            print(f"skip crossover: {path} missing", flush=True)
            continue
        _crossover(
            path,
            "l1",
            "L1 error (whole field)",
            f"{pretty}: whole-field L1 vs noise (n_obs={_STRESS_N_OBS})",
            _FIG_DIR / f"crossover-{tag}-l1",
        )
        _crossover(
            path,
            "l2",
            "L2 error (front-weighted)",
            f"{pretty}: front-weighted L2 vs noise (n_obs={_STRESS_N_OBS})",
            _FIG_DIR / f"crossover-{tag}-l2",
        )
        _crossover(
            path,
            "admissibility_violation",
            "admissibility violation",
            f"{pretty}: admissibility violation vs noise (n_obs={_STRESS_N_OBS})",
            _FIG_DIR / f"admissibility-{tag}",
        )

    if args.real:
        _real(_RESULTS / "real-ngsim.json", args.steps)

    if args.no_reconstructions:
        return
    recon = [
        ("lwr_riemann", "inverse-headline.json", "LWR (traffic)", "lwr"),
        ("burgers_riemann", "inverse-burgers-headline.json", "Burgers", "burgers"),
    ]
    for problem, tuned, pretty, tag in recon:
        tpath = _RESULTS / tuned
        if not tpath.exists():
            print(f"skip reconstruction: {tpath} missing", flush=True)
            continue
        _reconstruction(
            problem, tpath, pretty, _FIG_DIR / f"reconstruction-{tag}", args.steps
        )


if __name__ == "__main__":
    main()
