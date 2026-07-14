# Alternate base result — tuned shallow bake-off Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the Optuna search pipeline so a per-flavor HP search at ≤4 layers on the 5 paper datasets answers whether a tuned `alternate` beats the best of `mixed`/`split`, and render the result as the README-format table plus a Concepts write-up.

**Architecture:** Approach A — extend `suggest_config`, `search.py` (`search`/`final_eval`/`run_dataset`), the `benchmarks.search` CLI, and `make_tables` **backward-compatibly** (new behavior gated behind `search_activation`/`max_depth`/`embed_layers` keywords whose defaults reproduce today's behavior, so paper-reproduction and Stage-2 are untouched). Reuse the resumable Optuna-storage `search()` and the multi-GPU launcher. No new parallel modules.

**Tech Stack:** Python 3.11+, uv, pytest, Optuna, PyTorch (run backend), matplotlib; Typer CLI.

## Global Constraints

- Spec: [`docs/superpowers/specs/2026-07-14-alternate-base-result-design.md`](../specs/2026-07-14-alternate-base-result-design.md). Read it first.
- Branch `feat/alternate-base-result` (already off `main`, which has #110 merged). Commit `git commit --no-gpg-sign`. Never commit to `main`.
- Python 3.11+, line length 88 (ruff). Strict mypy (`uv run mypy`). MyST field-list docstrings (`:param:`/`:returns:`/`:raises:`) on public functions.
- **Backward compatibility is mandatory:** every change to `suggest_config`/`search`/`final_eval`/`run_dataset`/`_parse_flavors` must default to today's behavior. Existing tests in `tests/benchmarks/test_search*.py` must pass unchanged.
- **Layer counting:** `layers = depth + 1` (the read-out head counts; `Dense` embedding does not). "≤4 layers" ⇒ `depth ∈ [1, 3]` via `max_depth=3`. `make_tables._layers` already returns `depth + 1` for plain — do not change it.
- **Flavors this run:** `split-plain`, `mixed-plain`, `alternate-plain` (plain only). `alternate` is composition-init, plain-only; residual+alternate is unimplemented and must raise.
- Torch-only benchmark tests must start with `pytest.importorskip("torch")` (jax/keras CI installs no torch). Optuna tests use `optuna = pytest.importorskip("optuna")`.
- The published wheel ships layers only — all this code lives under `benchmarks/` and `docs/`, never `mononet/`.

## File structure

| File | Responsibility | Change |
|---|---|---|
| `benchmarks/_common/search_spaces.py` | per-trial `BenchmarkConfig` sampler | add `alternate` mode, `search_activation`, `max_depth`, `embed_layers` |
| `benchmarks/_common/search.py` | `search`/`final_eval`/`run_dataset` | thread the 3 new knobs; `final_eval` reads activation + sets `alt_init` + 2-layer embed |
| `benchmarks/search.py` | Typer CLI + `_parse_flavors` | accept `alternate-plain`; add `--search-activation`/`--max-depth`/`--embed-layers` |
| `benchmarks/stage2_launch.py` | multi-GPU fan-out | forward the new flags + `--flavors` to the subprocess |
| `benchmarks/_common/make_tables.py` | README-format table | add `alternate` rows, best-of-others bold, bootstrap verdict, param `_load(root)` |
| `docs/concepts/non-monotone-embedding.md` | Concepts page | new — the spec's write-up |
| `docs/benchmarks/alternate-base-result.md` | results page | new — the tuned-shallow table + verdict |
| `README.md` | headline table | replace the Benchmark-results table after the run |
| `tests/benchmarks/test_search_spaces.py`, `test_search.py`, `test_search_cli.py`, `test_make_tables.py` | tests | extend / create |

---

### Task 1: `suggest_config` — `alternate` mode, activation search, `max_depth`, `embed_layers`

**Files:**
- Modify: `benchmarks/_common/search_spaces.py` (signature ~21-33; body ~63-90)
- Test: `tests/benchmarks/test_search_spaces.py`

**Interfaces:**
- Produces: `suggest_config(trial, *, dataset, backend, mode, residual, epochs, metric, n_train, deep=False, search_activation=False, max_depth=4, embed_layers=1) -> BenchmarkConfig`. For `mode="alternate"`: `cfg.alt_init="composition"`, `convex_fraction` not sampled (stays `0.5`). `embed_hidden = (width,) * embed_layers`. `activation` searched over `["relu","elu","softplus","selu"]` iff `search_activation`, else `"elu"`. `depth = suggest_int("depth", 1, max_depth)` when not `deep`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/benchmarks/test_search_spaces.py` (uses the existing `study.ask()` idiom):
```python
def test_alternate_sets_composition_init_and_no_convex_fraction() -> None:
    study = optuna.create_study()
    trial = study.ask()
    cfg = suggest_config(
        trial, dataset="syn", backend="torch", mode="alternate",
        residual=False, epochs=3, metric="mse", n_train=10_000,
    )
    assert cfg.mode == "alternate"
    assert cfg.alt_init == "composition"
    assert cfg.convex_fraction == 0.5           # not a search dim for alternate
    assert "convex_fraction" not in trial.params

def test_search_activation_samples_one_of_four() -> None:
    study = optuna.create_study()
    trial = study.ask()
    cfg = suggest_config(
        trial, dataset="syn", backend="torch", mode="mixed", residual=False,
        epochs=3, metric="mse", n_train=10_000, search_activation=True,
    )
    assert cfg.activation in ("relu", "elu", "softplus", "selu")
    assert "activation" in trial.params

def test_default_activation_is_elu_and_alt_init_none() -> None:
    study = optuna.create_study()
    trial = study.ask()
    cfg = suggest_config(
        trial, dataset="syn", backend="torch", mode="mixed", residual=False,
        epochs=3, metric="mse", n_train=10_000,
    )
    assert cfg.activation == "elu"
    assert cfg.alt_init is None
    assert "activation" not in trial.params

def test_embed_layers_controls_embedding_depth() -> None:
    study = optuna.create_study()
    trial = study.ask()
    cfg = suggest_config(
        trial, dataset="syn", backend="torch", mode="mixed", residual=False,
        epochs=3, metric="mse", n_train=10_000, embed_layers=2,
    )
    assert cfg.embed_hidden == (cfg.width, cfg.width)

def test_max_depth_caps_depth() -> None:
    for _ in range(25):
        study = optuna.create_study()
        trial = study.ask()
        cfg = suggest_config(
            trial, dataset="syn", backend="torch", mode="mixed", residual=False,
            epochs=3, metric="mse", n_train=10_000, max_depth=3,
        )
        assert 1 <= cfg.depth <= 3
```

- [ ] **Step 2: Run to verify they fail**

Run: `MONONET_TEST_BACKEND=torch uv run pytest tests/benchmarks/test_search_spaces.py -q`
Expected: the 5 new tests FAIL (unexpected keyword `search_activation`/`max_depth`/`embed_layers`; `alt_init`/activation assertions).

- [ ] **Step 3: Widen the `mode` Literal + signature**

In `benchmarks/_common/search_spaces.py`, change the signature:
```python
def suggest_config(
    trial: optuna.Trial,
    *,
    dataset: str,
    backend: Literal["torch", "jax", "keras"],
    mode: Literal["split", "mixed", "alternate"],
    residual: bool,
    epochs: int,
    metric: Literal["accuracy", "rmse", "mse", "roc_auc"],
    n_train: int,
    deep: bool = False,
    search_activation: bool = False,
    max_depth: int = 4,
    embed_layers: int = 1,
) -> BenchmarkConfig:
```
Update the docstring: note `activation` is searched only when `search_activation`; `mode="alternate"` uses composition init and does not search `convex_fraction`; `max_depth` bounds the shallow `depth` band; `embed_layers` sets the number of non-monotone `Dense` layers.

- [ ] **Step 4: Sample the new dimensions**

Replace the `depth`, `convex_fraction`, and the return block. The depth line:
```python
    if deep:
        depth = trial.suggest_categorical("depth", [6, 10, 16])
    else:
        depth = trial.suggest_int("depth", 1, max_depth)
```
The activation + convex_fraction + alt_init (place just before building `metrics`):
```python
    activation = (
        trial.suggest_categorical("activation", ["relu", "elu", "softplus", "selu"])
        if search_activation
        else "elu"
    )
    convex_fraction = (
        trial.suggest_float("convex_fraction", 0.0, 1.0) if mode == "mixed" else 0.5
    )
    alt_init = "composition" if mode == "alternate" else None
    embed_hidden = tuple(int(width) for _ in range(embed_layers))
```
In the returned `BenchmarkConfig(...)`, set `activation=activation`, `embed_hidden=embed_hidden`, and add `alt_init=alt_init`. (Remove the old `activation="elu"` and `embed_hidden=(int(width),)` lines.)

- [ ] **Step 5: Run the tests green**

Run: `MONONET_TEST_BACKEND=torch uv run pytest tests/benchmarks/test_search_spaces.py -q`
Expected: PASS (new + existing). Existing `test_mixed_*`/`test_split_*` still pass (defaults: elu, `max_depth=4`, `embed_layers=1`).

- [ ] **Step 6: Lint + commit**

```bash
uv run ruff format benchmarks/_common/search_spaces.py tests/benchmarks/test_search_spaces.py
uv run ruff check benchmarks/_common/search_spaces.py tests/benchmarks/test_search_spaces.py
uv run mypy benchmarks/_common/search_spaces.py
git add benchmarks/_common/search_spaces.py tests/benchmarks/test_search_spaces.py
git commit --no-gpg-sign -m "feat(bench): suggest_config supports alternate, activation search, max_depth, embed_layers"
```

---

### Task 2: `search.py` — thread the knobs; `final_eval` activation/alt_init/embed

**Files:**
- Modify: `benchmarks/_common/search.py` (`search` 81-96 + objective's `suggest_config` call 111-123; `final_eval` 153-~205; `run_dataset` 325-417)
- Test: `tests/benchmarks/test_search.py`

**Interfaces:**
- Consumes: `suggest_config(..., search_activation, max_depth, embed_layers)` from Task 1.
- Produces: `search(..., search_activation=False, max_depth=4, embed_layers=1)`; `final_eval(..., embed_layers=1)` reads `best_params["activation"]` (default `"elu"`), sets `alt_init="composition"` for `mode="alternate"`, builds `embed_hidden=(width,)*embed_layers`; `run_dataset(..., search_activation=False, max_depth=4, embed_layers=1)` threads all three into both `search` and `final_eval`.

- [ ] **Step 1: Write the failing test**

Append to `tests/benchmarks/test_search.py` (it already `pytest.importorskip("torch")`s and builds a tiny bundle — reuse its bundle helper; if none, add the one below):
```python
def _tiny_reg_bundle():
    import numpy as np
    from benchmarks._common.bundle import DatasetBundle
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, (80, 3)).astype("float32")
    y = x.sum(1).astype("float32")
    return DatasetBundle(
        name="t", task="regression", X_train=x, y_train=y, X_test=x, y_test=y,
        mono_increasing=(0, 1, 2), mono_decreasing=(), feature_names=("a", "b", "c"),
        metadata={},
    )

def test_final_eval_honors_activation_and_alt_init() -> None:
    from benchmarks._common.search import final_eval
    b = _tiny_reg_bundle()
    params = {"width": 8, "depth": 2, "dropout": 0.0, "lr": 1e-2,
              "weight_decay": 0.0, "lr_decay": 1.0, "batch_size": 32,
              "activation": "relu"}
    agg, rows = final_eval(
        b, params, mode="alternate", residual=False, backend="torch",
        seeds=range(2), epochs=2, embed_layers=2,
    )
    assert len(rows) == 2
    assert all("mse" in r.scores for r in rows)

def test_search_alternate_shallow_runs() -> None:
    from benchmarks._common.search import search
    b = _tiny_reg_bundle()
    res = search(
        b, mode="alternate", residual=False, backend="torch", n_trials=2,
        epochs=2, n_splits=2, search_seeds=1, search_activation=True, max_depth=3,
        embed_layers=2,
    )
    assert res.flavor == "alternate-plain"
    assert 1 <= res.best_params["depth"] <= 3
    assert res.best_params["activation"] in ("relu", "elu", "softplus", "selu")
```

- [ ] **Step 2: Run to verify it fails**

Run: `MONONET_TORCH_DEVICE=cpu MONONET_TEST_BACKEND=torch uv run pytest tests/benchmarks/test_search.py -q -k "alternate or activation"`
Expected: FAIL (unexpected keyword `search_activation`/`embed_layers`; `final_eval` ignores `activation`).

- [ ] **Step 3: Extend `search()`**

Add to `search`'s keyword-only params (after `metric: str | None = None,`):
```python
    search_activation: bool = False,
    max_depth: int = 4,
    embed_layers: int = 1,
```
In the `objective`, pass them into `suggest_config(...)` (add after `deep=deep,`):
```python
            search_activation=search_activation,
            max_depth=max_depth,
            embed_layers=embed_layers,
```

- [ ] **Step 4: Extend `final_eval()`**

Add `embed_layers: int = 1` to `final_eval`'s keyword-only params. Then in its `BenchmarkConfig(...)` construction: replace `activation="elu"` with
```python
        activation=str(best_params.get("activation", "elu")),
```
replace `embed_hidden=(width,)` with
```python
        embed_hidden=tuple(width for _ in range(embed_layers)),
```
and add
```python
        alt_init="composition" if mode == "alternate" else None,
```
to the `BenchmarkConfig(...)` call (alongside `mode=mode`).

- [ ] **Step 5: Extend `run_dataset()`**

Add `search_activation: bool = False`, `max_depth: int = 4`, `embed_layers: int = 1` to `run_dataset`'s keyword-only params. Pass them into the `search(...)` call (add after `search_seeds=search_seeds,`):
```python
            search_activation=search_activation,
            max_depth=max_depth,
            embed_layers=embed_layers,
```
and pass `embed_layers=embed_layers` into the `final_eval(...)` call (after `epochs=epochs,`).

- [ ] **Step 6: Run green + full search suite**

```bash
MONONET_TORCH_DEVICE=cpu MONONET_TEST_BACKEND=torch uv run pytest tests/benchmarks/test_search.py -q
```
Expected: PASS (new + existing).

- [ ] **Step 7: Lint + commit**

```bash
uv run ruff format benchmarks/_common/search.py tests/benchmarks/test_search.py
uv run ruff check benchmarks/_common/search.py tests/benchmarks/test_search.py
uv run mypy benchmarks/_common/search.py
git add benchmarks/_common/search.py tests/benchmarks/test_search.py
git commit --no-gpg-sign -m "feat(bench): thread activation/max_depth/embed_layers through search + final_eval alt_init"
```

---

### Task 3: CLI — `_parse_flavors` accepts `alternate-plain`; new flags; launcher passthrough

**Files:**
- Modify: `benchmarks/search.py` (`_parse_flavors` 45-64; `main` 67-130 + the `run_dataset` call)
- Modify: `benchmarks/stage2_launch.py` (`_run_dataset` command 54-68; `run_parallel`/`main` to forward flags)
- Test: `tests/benchmarks/test_search_cli.py`

**Interfaces:**
- Produces: `_parse_flavors("alternate-plain") == (("alternate", False, False),)`; `_parse_flavors("alternate-residual")` raises `typer.BadParameter`. CLI gains `--search-activation` (bool), `--max-depth` (int, default 4), `--embed-layers` (int, default 1), threaded into `run_dataset`. `stage2_launch` forwards `--flavors`, `--search-activation`, `--max-depth`, `--embed-layers`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/benchmarks/test_search_cli.py`:
```python
def test_parse_flavors_accepts_alternate_plain() -> None:
    from benchmarks.search import _parse_flavors
    assert _parse_flavors("alternate-plain") == (("alternate", False, False),)

def test_parse_flavors_rejects_alternate_residual() -> None:
    import typer, pytest
    from benchmarks.search import _parse_flavors
    with pytest.raises(typer.BadParameter):
        _parse_flavors("alternate-residual")
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/benchmarks/test_search_cli.py -q -k alternate`
Expected: FAIL (`bad flavor: alternate-plain` — `alternate` not in `valid_modes`).

- [ ] **Step 3: Update `_parse_flavors`**

In `benchmarks/search.py`, replace the body's mode/kind validation:
```python
    valid_modes = {"split", "mixed", "alternate"}
    valid_kinds = {"plain", "residual", "deep"}
    out: list[tuple[str, bool, bool]] = []
    for name in spec.split(","):
        mode, _, kind = name.partition("-")
        if mode not in valid_modes or kind not in valid_kinds:
            raise typer.BadParameter(f"bad flavor: {name}")
        if mode == "alternate" and kind != "plain":
            raise typer.BadParameter(
                f"alternate supports only plain topology: {name}"
            )
        deep = kind == "deep"
        out.append((mode, kind == "residual" or deep, deep))
    return tuple(out)
```

- [ ] **Step 4: Add the CLI flags + thread them**

In `main`, add options (after `search_seeds`):
```python
    search_activation: bool = typer.Option(
        False, "--search-activation", help="search activation over relu/elu/softplus/selu"
    ),
    max_depth: int = typer.Option(4, "--max-depth", help="upper bound of the shallow depth band"),
    embed_layers: int = typer.Option(1, "--embed-layers", help="Dense layers in the non-monotone embedding"),
```
Find the `run_dataset(...)` call in `main` and pass:
```python
            search_activation=search_activation,
            max_depth=max_depth,
            embed_layers=embed_layers,
```

- [ ] **Step 5: Forward flags in `stage2_launch`**

In `benchmarks/stage2_launch.py`, give `_run_dataset` the extra args and append them to `cmd`. Change its signature to accept `extra: list[str]` and append `cmd += extra` before `subprocess.run`. In `run_parallel`/`main`, build `extra` from new options `--flavors`, `--search-activation`, `--max-depth`, `--embed-layers` and thread it through the `ThreadPoolExecutor` task. (Mirror the existing `storage_dir` passthrough pattern.) Add a `test_stage2_launch` assertion if a launch test exists; otherwise a `--dry-run`-style unit test is out of scope here — a manual `--help` check suffices.

- [ ] **Step 6: Run green + commit**

```bash
uv run pytest tests/benchmarks/test_search_cli.py -q
uv run ruff format benchmarks/search.py benchmarks/stage2_launch.py tests/benchmarks/test_search_cli.py
uv run ruff check benchmarks/search.py benchmarks/stage2_launch.py tests/benchmarks/test_search_cli.py
uv run mypy benchmarks/search.py benchmarks/stage2_launch.py
git add benchmarks/search.py benchmarks/stage2_launch.py tests/benchmarks/test_search_cli.py
git commit --no-gpg-sign -m "feat(bench): CLI accepts alternate-plain + search-activation/max-depth/embed-layers; launcher forwards them"
```

---

### Task 4: `make_tables` — `alternate` rows, best-of-others bold, bootstrap verdict

**Files:**
- Modify: `benchmarks/_common/make_tables.py` (`_load` 59-64; `render` 79-140; `main` ~141)
- Test: `tests/benchmarks/test_make_tables.py` (new)

**Interfaces:**
- Consumes: per-flavor result JSONs (`{dataset}-{flavor}.json`) with keys `dataset`, `flavor`, `test_values`, `test_metric`, `n_collapse`, `n_seeds`, `best_params.depth`, `cv_best` (as written by `run_dataset`).
- Produces: `_load(root: Path | None = None)`; `render(root: Path | None = None) -> str` that includes an `alternate | plain` row per dataset, bolds the best across `{split,mixed,alternate}-plain`, and appends a **Verdict** section with the bootstrap IQM delta `alternate − best-of-{split,mixed}`.

- [ ] **Step 1: Write the failing test**

Create `tests/benchmarks/test_make_tables.py`:
```python
from __future__ import annotations

import json
from pathlib import Path

from benchmarks._common.make_tables import render_verdict


def _rec(dataset, flavor, values, depth=2):
    return {
        "dataset": dataset, "flavor": flavor, "test_metric": "roc_auc",
        "test_values": values, "n_collapse": 0, "n_seeds": len(values),
        "best_params": {"depth": depth}, "cv_best": sum(values) / len(values),
    }


def test_render_verdict_reports_alternate_win(tmp_path: Path) -> None:
    d = {
        "split-plain": _rec("heart", "split-plain", [0.70, 0.71, 0.69]),
        "mixed-plain": _rec("heart", "mixed-plain", [0.72, 0.73, 0.71]),
        "alternate-plain": _rec("heart", "alternate-plain", [0.80, 0.81, 0.79]),
    }
    line = render_verdict("heart", d, lower=False)
    assert "alternate" in line
    assert "heart" in line
    # alternate clearly beats best-of-others -> verdict says beats/helps
    assert ("beats" in line.lower()) or ("helps" in line.lower())
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/benchmarks/test_make_tables.py -q`
Expected: FAIL (`render_verdict` does not exist).

- [ ] **Step 3: Parametrize `_load`, add the `alternate` row + `render_verdict`**

In `benchmarks/_common/make_tables.py`:

Make `_load` accept a root:
```python
def _load(root: Path | None = None) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    root = root or (Path(__file__).resolve().parents[1] / "results" / "phase2")
    for f in sorted(root.glob("*.json")):
        r = json.loads(f.read_text())
        out.setdefault(r["dataset"], {})[r["flavor"]] = r
    return out
```

Add `("alternate", "plain", d.get("alternate-plain"))` to the `entries` list in `render` (after the `mixed`/`residual` entries), so the bold selection (`scored`/`best`) automatically considers it.

Add the verdict helper (reuses the shared bootstrap):
```python
def render_verdict(ds: str, d: dict[str, dict[str, Any]], lower: bool) -> str:
    """One-line bootstrap verdict: alternate vs best-of-{split,mixed} plain.

    :param ds: Dataset name.
    :param d: Flavor -> record map for this dataset.
    :param lower: Whether lower metric is better.
    :returns: A Markdown table row ``| ds | Δ | 95% CI | verdict |``.
    """
    import numpy as np

    from benchmarks._common.results import bootstrap_delta

    alt = d.get("alternate-plain")
    others = [d.get("split-plain"), d.get("mixed-plain")]
    others = [o for o in others if o]
    if alt is None or not others:
        return f"| {ds} | — | — | *pending* |"
    best_other = (min if lower else max)(others, key=lambda o: _stats(o, ds)[3])
    av = np.asarray(alt["test_values"], np.float64)
    bv = np.asarray(best_other["test_values"], np.float64)
    if ds == "blog":  # values stored as MSE; table reports RMSE
        av, bv = np.sqrt(av), np.sqrt(bv)
    point, lo, hi = bootstrap_delta(av, bv, lower_is_better=lower)
    if lo > 0:
        verdict = "alternate **beats** best-of-others"
    elif hi < 0:
        verdict = "alternate loses"
    else:
        verdict = "matches (CI straddles 0)"
    return (
        f"| {ds} | {point:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {verdict} "
        f"(vs {best_other['flavor']}) |"
    )
```
Note `bootstrap_delta`'s convention: positive Δ always means the first arg (alternate) is better, sign-normalized by `lower_is_better` — so `lo > 0` ⇒ alternate beats.

In `render`, after the main-results table loop, append the verdict section:
```python
    out += ["", "### Verdict — alternate vs best-of-others", ""]
    out.append("| dataset | Δ (alt − best-other) | 95% CI | verdict |")
    out.append("|---|--:|:--|:--|")
    wins = 0
    for ds in _ORDER:
        d = rows.get(ds, {})
        if not d:
            continue
        lower = _DISP[ds][0] in ("MSE", "RMSE")
        line = render_verdict(ds, d, lower)
        out.append(line)
        if "beats" in line:
            wins += 1
    out.append("")
    out.append(f"**alternate beats best-of-others on {wins} of {len(_ORDER)} datasets.**")
```

- [ ] **Step 4: Run green**

Run: `uv run pytest tests/benchmarks/test_make_tables.py -q`
Expected: PASS.

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff format benchmarks/_common/make_tables.py tests/benchmarks/test_make_tables.py
uv run ruff check benchmarks/_common/make_tables.py tests/benchmarks/test_make_tables.py
uv run mypy benchmarks/_common/make_tables.py
git add benchmarks/_common/make_tables.py tests/benchmarks/test_make_tables.py
git commit --no-gpg-sign -m "feat(bench): make_tables renders alternate rows + bootstrap verdict; _load takes a root"
```

---

### Task 5: Concepts write-up + benchmarks results page + docs wiring

**Files:**
- Create: `docs/concepts/non-monotone-embedding.md`
- Create: `docs/benchmarks/alternate-base-result.md`
- Modify: `docs/concepts/index.md` and `docs/benchmarks/index.md` (toctrees)

**Interfaces:** none (docs only). No tests; gated by the docs build.

- [ ] **Step 1: Write the Concepts page**

Create `docs/concepts/non-monotone-embedding.md` from the spec's "Concepts: non-monotone feature embedding" section verbatim (the problem, the embedding-composition construction, and the two-`Dense`-layer UAP argument — the `|W|`-constrained first `MonoLinear` cannot be the free output projection, Prop 3.2). Start with an H1 title `# Non-monotone feature embedding`.

- [ ] **Step 2: Write the results page skeleton**

Create `docs/benchmarks/alternate-base-result.md` with an H1 `# Alternate base result (tuned, ≤4 layers)`, a paragraph stating the protocol (per-flavor Optuna HP search, activation searched, `depth ∈ [1,3]` ⇒ ≤4 layers, plain-only, 5 paper datasets, verdict = bootstrap CI on `alternate − best-of-{split,mixed}`), and a placeholder line `<!-- TABLE -->` where Task 6 pastes the generated table + verdict. (This placeholder is filled in Task 6, before any docs build is required to pass, so it is not a plan-level placeholder.)

- [ ] **Step 3: Wire the toctrees**

Add `non-monotone-embedding` to the `{toctree}` in `docs/concepts/index.md` and `alternate-base-result` to the `{toctree}` in `docs/benchmarks/index.md` (match the existing entry style — bare filename, no extension). Add a one-line bullet describing each page next to the other section bullets.

- [ ] **Step 4: Docs build + commit**

```bash
./tools/build-docs.sh
git add docs/concepts/non-monotone-embedding.md docs/benchmarks/alternate-base-result.md docs/concepts/index.md docs/benchmarks/index.md
git commit --no-gpg-sign -m "docs: non-monotone-embedding concept + alternate-base-result page (skeleton)"
```
Expected: docs build succeeds (`-W`, no warnings).

---

### Task 6 (GPU): smoke on `heart`, full run, regenerate table, update README

**Files:**
- Create: `benchmarks/results/alternate-base/*.json` (committed run outputs) + `benchmarks/results/alternate-base/studies/*.db` (Optuna storage — add to `.gitignore` if large; commit the JSONs only)
- Modify: `docs/benchmarks/alternate-base-result.md` (paste table), `README.md` (replace Benchmark-results table)

- [ ] **Step 1: Environment**

On a `gpu-torch` devcontainer from this branch: `uv sync --extra torch-gpu --group bench`. Confirm `MONONET_TORCH_DEVICE=cuda:0 uv run --no-sync python -c "from mononet.torch import MonoLinear; MonoLinear(4,8,mode='alternate')"` works, and that the five datasets are present (`uv run --no-sync python -m benchmarks.datasets.download`).

- [ ] **Step 2: Smoke on `heart` (all three flavors)**

```bash
uv run --group bench python -m benchmarks.search \
  --datasets heart --flavors split-plain,mixed-plain,alternate-plain \
  --search-activation --max-depth 3 --embed-layers 2 \
  --out-dir benchmarks/results/alternate-base \
  --storage-dir benchmarks/results/alternate-base/studies --smoke
```
Read the three `heart-*.json`; sanity-check that metrics are finite, `best_params` has an `activation` key, and `depth ∈ [1,3]`. Fix any surprises before scaling.

- [ ] **Step 3: Full run, both GPUs, resumable**

```bash
uv run --group bench python -m benchmarks.stage2_launch \
  --datasets heart,auto,compas,loan,blog --devices cuda:0,cuda:1 \
  --flavors split-plain,mixed-plain,alternate-plain \
  --search-activation --max-depth 3 --embed-layers 2 \
  --out-dir benchmarks/results/alternate-base \
  --storage-dir benchmarks/results/alternate-base/studies
```
Monitor to terminal (per-flavor JSON appears on study completion; a kill resumes from the Optuna DB). heart's smoke JSONs are overwritten by the full-seed run.

- [ ] **Step 4: Regenerate the table + verdict**

```bash
uv run --group bench python -c "from benchmarks._common.make_tables import render, _load; import pathlib; print(render(root=pathlib.Path('benchmarks/results/alternate-base')))"
```
Paste the output over the `<!-- TABLE -->` placeholder in `docs/benchmarks/alternate-base-result.md`, and replace the README "Benchmark results" table (and its `mixed wins 4 of 5` prose) with the new table + the "alternate beats best-of-others on N of 5" line. Update the README intro sentence to mention the third construction (`alternate`).

- [ ] **Step 5: Commit results + docs, docs build**

```bash
git add benchmarks/results/alternate-base/*.json docs/benchmarks/alternate-base-result.md README.md
./tools/build-docs.sh
git commit --no-gpg-sign -m "results(bench): tuned shallow alternate base result + README/table update"
```

- [ ] **Step 6: Full-repo green, push, open the PR**

```bash
MONONET_TEST_BACKEND=torch uv run pytest tests/benchmarks -q
uv run ruff check && uv run ruff format --check && uv run mypy
git push -u origin feat/alternate-base-result
```
Open the PR against `main` (title `feat(bench): tuned shallow alternate base result`), body summarizing the verdict (N of 5), the layer-counting/theorem framing, and linking the spec + this plan.

---

## Notes

- If a dataset's study errors mid-run, its `.db` under `studies/` lets the same command resume; delete the `.db` to force a fresh search.
- `loan`/`blog` are large; their `_BUDGET` already uses a single split and fewer seeds. Smoke `heart` first (Step 2) to catch wiring bugs before paying for the large datasets.
- `make_tables._layers` returns `depth + 1` for plain — this is the theorem-consistent layer count; do not change it.
