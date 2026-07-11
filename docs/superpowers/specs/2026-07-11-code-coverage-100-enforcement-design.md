# Design: 100% code-coverage enforcement and gap closure

**Date:** 2026-07-11
**Status:** Approved (brainstorming) — pending implementation plan
**Scope:** CI + test suite. No changes to shipped library behavior.

## Problem

Code coverage is well below 100% and is not checked in CI. A `CODECOV_TOKEN`
secret has been added so coverage can be reported and gated. We want:

- Coverage measured and enforced at **100%**.
- Any new PR blocked from merging unless it keeps coverage at 100% (both the
  project total and the PR's own diff).
- New tests written to close the current gap.

## Baseline (measured 2026-07-11, all three backends combined)

`mononet/` package: **89%** — 55 uncovered statements + 27 partially-covered
branches. `docs/examples/`: already **100%**. The uncovered lines are almost
entirely genuinely-testable paths, not unreachable code:

| Target | Uncovered lines | Nature |
|---|---|---|
| `core/numerics.py` | 4–29 (whole module, 0%) | untested helper module |
| `core/config.py` | 37, 98, 100 | invalid-mode / invalid-units `raise ValueError` paths |
| `core/reference.py` | 43–45, 75, 117 | unknown activation / unknown gate / invalid mode |
| `core/init.py` | 87 | one bisection branch (`else: hi = mid`) |
| `torch/_kernels.py` | 40, 69, 104 | error paths + activation variants |
| `torch/layers.py` | 50, 52, 185, 224 | optional-argument branches |
| `jax/_kernels.py` | 25, 27, 30, 56, 91, 124–126 | activation variants + error paths |
| `jax/layers.py` | 214, 253 | optional-argument branches |
| `keras/_kernels.py` | 25, 27, 30, 56, 93, 129–131 | activation variants + error paths |
| `keras/layers.py` | 267–278, 301 | optional-argument branches |

Two structural facts constrain the design:

1. **Each CI matrix `test` job runs only one backend** (`torch` *or* `jax` *or*
   `keras`, selected by `MONONET_TEST_BACKEND`). No single job can cover 100% of
   `mononet/`, since the torch job cannot execute the jax/keras layers. Coverage
   must be **merged across backends** to reach a meaningful total.
2. **`tests/examples/` is not executed by the CI `test` job** (the matrix runs
   `tests/core`, `tests/${backend}`, `tests/equivalence`, `tests/benchmarks`,
   and `tests/test_top_level_imports.py`). The example tests currently run
   nowhere in CI.

## Decisions

- **Scope of 100%: `mononet/` + `docs/examples/`.** `benchmarks/` is dropped
  from coverage measurement entirely — it is large, repo-only, not shipped in
  the wheel, and out of scope for this effort.
- **Enforcement: Codecov *plus* a hard local CI gate.** A deterministic
  `--cov-fail-under=100` step is the real blocker (independent of Codecov
  uptime); Codecov `project`/`patch` statuses add per-PR annotation of exactly
  which new lines are uncovered.
- **Merge model: one dedicated combined-coverage job.** A single ubuntu job
  installs all CPU backends and runs the whole suite in one process, appending
  across backends into one coverage data file — the single source of truth for
  the gate. The existing matrix stays for cross-OS/Python correctness, with
  coverage disabled there.

## Design

### 1. Coverage configuration (`pyproject.toml`)

- Change `addopts`: drop `--cov=benchmarks`, keep `--cov=mononet
  --cov=docs/examples --cov-append --cov-branch --cov-report=term-missing`.
- Do **not** put `--cov-fail-under` in `addopts` — a local partial `pytest` run
  must not spuriously fail. The gate lives only in the combined CI job command.
- Add a `[tool.coverage.report]` section with a **conservative** `exclude_also`
  list — only genuinely non-runtime lines:

  ```toml
  [tool.coverage.report]
  exclude_also = [
      "if TYPE_CHECKING:",
      "\\.\\.\\.",                       # overload / stub bodies
      "if __name__ == .__main__.:",
      "pragma: no cover",
  ]
  ```

  Notably **not** `raise ...` — the `raise ValueError` error paths are exactly
  what we want tested.

- **`# pragma: no cover` policy:** allowed only for provably unreachable
  defensive code, each occurrence justified in the PR. Expectation: near-zero
  uses; the current gap needs none.

### 2. CI architecture (`.github/workflows/build.yml`)

New **`coverage`** job (ubuntu, Python 3.13), installs `.[all-cpu]` +
`--group=dev`, and runs the whole suite in one process, appending across
backends so every backend's kernel/layer branches are hit:

```bash
uv run coverage erase
KERAS_BACKEND=jax MONONET_TEST_BACKEND=torch pytest \
  tests/core tests/torch tests/equivalence tests/examples \
  tests/test_top_level_imports.py
KERAS_BACKEND=jax MONONET_TEST_BACKEND=jax   pytest tests/jax   tests/equivalence
KERAS_BACKEND=jax MONONET_TEST_BACKEND=keras pytest tests/keras tests/equivalence
uv run coverage report --show-missing --fail-under=100   # hard gate
uv run coverage xml                                       # for Codecov
```

- `tests/equivalence` runs once per `MONONET_TEST_BACKEND` so each backend's
  kernel branches are exercised (the equivalence suite `importorskip`s the
  inactive backends).
- `tests/examples` is included here, closing the "examples run nowhere" gap.
- The existing matrix **`test`** job keeps its per-backend cross-OS/Python
  invocation but adds `--no-cov` (no coverage overhead, no gating there).
  Benchmarks still run there for correctness — just not measured.
- Add `coverage` to the **`check`** job's `needs`. The `--cov-fail-under=100`
  failure then propagates through the existing `alls-green` `check` gate, so the
  **hard gate requires no new branch-protection wiring** — it rides whatever
  already makes `check` a required status.

### 3. Codecov integration

- Add a `codecov/codecov-action@v5` step to the `coverage` job:
  `token: ${{ secrets.CODECOV_TOKEN }}`, `files: coverage.xml`,
  `fail_ci_if_error: true`.
- Add **`codecov.yml`** at the repo root:

  ```yaml
  coverage:
    status:
      project:
        default: { target: 100%, threshold: 0% }
      patch:
        default: { target: 100% }        # "new PR needs 100%" on the diff
  comment:
    layout: "condensed_header, diff, files"
  ignore:
    - "benchmarks/**"
    - "tests/**"
    - "docs/**"
  ```

  Note the intentional divergence: Codecov's `project`/`patch` statuses gate
  **`mononet/` only** (`docs/**` is ignored to avoid Codecov reporting on
  unmeasured docs source), while the hard local gate covers **both `mononet/`
  and `docs/examples/`**. `docs/examples/` is already 100% and is protected by
  the hard gate regardless.

- **Manual action item (repo settings, not code):** to make the Codecov
  `project`/`patch` statuses *required* for merge, add them to branch-protection
  required checks. The hard gate already blocks merges without this; the Codecov
  requirement is the per-PR-diff UX layer. Can be done via `gh api` during
  implementation or left to the maintainer.

### 4. Closing the gap (new tests)

All real tests, written TDD-style (add test → confirm it covers the line).
Because backend-specific files are only covered under their own
`MONONET_TEST_BACKEND`, per-backend tests live in `tests/{torch,jax,keras}/` (or
`tests/equivalence/`) so the combined job's three-pass run picks them up.

| Target | New tests |
|---|---|
| `core/numerics.py` | `test_numerics.py`: `default_atol`/`default_rtol` for float32, float64, and a non-64 dtype (float16 → float32 branch) |
| `core/config.py` | `pytest.raises` for `MonoConfig(mode=bad)`, `MonoResidualConfig(units<=0)`, `MonoResidualConfig(mode=bad)` |
| `core/reference.py` | `pytest.raises` on unknown activation, unknown gate token, invalid mode |
| `core/init.py` | input that drives the upper-half bisection branch (line 87) |
| `{torch,jax,keras}/_kernels.py` | per-backend `pytest.raises` for bad activation/gate/mode; forward passes exercising each activation name (elu/selu/softplus/identity) |
| `{torch,jax,keras}/layers.py` | targeted layer tests for the optional-argument branches (e.g. `bias=False`, skip/residual paths, repr) — exact lines pinned during the plan phase |

### 5. Sequencing

The order keeps `main` green throughout and makes "flip to enforcing" a small,
reviewable change:

1. Land coverage **config + CI + `codecov.yml`** with the hard gate
   **temporarily disabled** (job `continue-on-error` or no `--fail-under`), so
   the branch is not red while tests are being written.
2. Add tests category-by-category until the combined three-pass run reports
   **100%** locally.
3. **Flip the gate on** (`--fail-under=100`, remove `continue-on-error`) and wire
   `coverage` into `check.needs`.
4. (Optional) add the Codecov `project`/`patch` statuses to branch protection.

## Out of scope

- Coverage of `benchmarks/` (dropped by decision above).
- Any change to shipped library behavior — this is test/CI only.
- Coverage on GPU devcontainer flavors; the combined job is CPU-only, which is
  sufficient since kernels are backend-native and dtype-parametrized on CPU.

## Verification

- Locally: the three-pass combined invocation followed by
  `uv run coverage report --fail-under=100` exits 0.
- In CI: the `coverage` job passes and its status flows into `check`; Codecov
  reports 100% project and 100% patch on the implementing PR.
