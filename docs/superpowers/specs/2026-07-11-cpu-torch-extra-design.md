# CPU-only torch extra for the default devcontainer and CI

**Date:** 2026-07-11
**Status:** Design — awaiting review
**Author:** Davor Runje (with Claude)

## Problem

Building the `default` (CPU) devcontainer installs the full CUDA stack —
`torch 2.12.0` plus ~17 `nvidia-*-cu13` wheels and `triton` (~3 GB). None of it
is usable on a CPU box.

Root cause, confirmed end to end:

1. `.devcontainer/default/setup.sh` exports `MONONET_EXTRAS="all"`, and
   `install_dependencies.sh` runs `uv sync --extra all`.
2. `all = ["mononet[torch,jax,keras]"]`; the `torch` extra is `torch>=2.4`.
3. On linux-x86_64 the **default PyPI `torch` wheel is the CUDA build** and
   hard-depends on the `nvidia-*` packages. `jax` (CPU) contributes nothing
   here — every nvidia package comes from torch.

The `default` flavor exists to run the cross-backend equivalence suite
(`tests/equivalence/`), which needs torch, jax, and keras importable at once, so
`--extra all` (all three backends) is correct. The CUDA wheels are an unrelated
side effect of torch's packaging that this flavor never opted out of.

## Constraint that shapes the whole design

There is **no way to express "CPU torch" in standard extras metadata that pip
honors.** Plain `torch` on PyPI *is* the CUDA wheel; the CPU wheel lives at a
separate index (`https://download.pytorch.org/whl/cpu`). Redirecting to that
index is a **uv-only** mechanism (`[tool.uv.sources]` + `[tool.uv.index]`) that
never enters wheel metadata. Therefore:

- The CPU/GPU split can only be delivered under **uv** (`uv sync` / `uv lock`),
  which is exactly how the devcontainers and — after this change — the ubuntu CI
  jobs install. Plain-pip users always get torch's default wheel.
- A blanket, marker-only source redirect (`marker = "sys_platform == 'linux'"`)
  is rejected: it would also catch the `torch-gpu` extra and break the GPU
  devcontainers. The redirect must be **conditioned on an extra**, and uv
  requires that extra to be declared in `[project.optional-dependencies]`.

torch vs jax asymmetry (why this is torch-only):

- **torch:** the CUDA build ships under the *same* name+version as every other
  platform. `torch>=2.4` resolves on linux (→CUDA), macOS (→CPU), Windows
  (→CPU). One spec, installs everywhere.
- **jax:** `jax[cuda12]` is an *extra* pulling `jax-cuda12-plugin` /
  `jax-cuda12-pjrt`, whose wheels are **linux-only**. The CPU `jax` extra is
  already CPU and installs everywhere; jax needs no change.

## Non-goals

- Changing the published semantics of `torch` / `jax` / `keras` / `all`. After
  this change `pip install mononet[all]` (and `[torch]`) is byte-for-byte
  identical to today.
- Adding an `all-gpu` extra. A symmetric "all backends on GPU" extra would have
  to be linux-only (because of `jax[cuda12]`), and nobody uses one — the GPU
  devcontainers are deliberately single-backend. YAGNI; clean linux-gated
  follow-up if ever wanted.
- Touching the three GPU devcontainer flavors. Verified: each syncs exactly one
  extra (`torch-gpu` / `jax-gpu` / `keras-gpu`); `uv sync --extra X` installs
  base + that extra + default groups only, no cross-backend bleed.
  (`keras-gpu` including `jax[cuda12]` is intentional — Keras 3 runs on the JAX
  backend in that container.)

## Design

### 1. `pyproject.toml` — additive extras + uv config

```toml
[project.optional-dependencies]
torch       = ["torch>=2.4"]                    # unchanged — PyPI default (CUDA on linux)
torch-cpu   = ["torch>=2.4"]                    # NEW — same spec, uv redirects to cpu index
jax         = ["jax>=0.4.30", "flax>=0.10"]     # unchanged (already CPU)
keras       = ["keras>=3.5", "jax>=0.4.30"]     # unchanged
all         = ["mononet[torch,jax,keras]"]      # unchanged
all-cpu     = ["mononet[torch-cpu,jax,keras]"]  # NEW
# torch-gpu / jax-gpu / keras-gpu unchanged

[[tool.uv.index]]
name = "pytorch-cpu"
url  = "https://download.pytorch.org/whl/cpu"
explicit = true                                 # only used when a source points here

[tool.uv.sources]
torch = [{ index = "pytorch-cpu", extra = "torch-cpu" }]

[tool.uv]
# torch-cpu (cpu index) cannot co-resolve with torch / torch-gpu (both PyPI).
# torch and torch-gpu share the PyPI source and do not clash, but a single
# 3-way mutually-exclusive group is harmless (nobody combines them) and keeps
# the lockfile resolvable.
conflicts = [
  [{ extra = "torch-cpu" }, { extra = "torch" }, { extra = "torch-gpu" }],
]
```

Properties:

- `pip install mononet[all]` / `mononet[torch]` unchanged (CUDA on linux).
- `mononet[torch-cpu]` / `mononet[all-cpu]` deliver CPU torch **only under uv**;
  a plain-pip user gets the default wheel. Documented, not hidden.
- `conflicts` is what lets one `uv.lock` hold both torch variants (two indexes
  for the same package) without uv erroring at lock time.
- The existing `override-dependencies` (click pin) and other `[tool.uv]` keys are
  preserved.

### 2. Devcontainer

- `.devcontainer/default/setup.sh`: `MONONET_EXTRAS="all"` → `"all-cpu"`.
- `.devcontainer/shared/install_dependencies.sh`: default fallback
  `${MONONET_EXTRAS:-all}` → `${MONONET_EXTRAS:-all-cpu}`, so a bare CPU sync is
  lean by default.
- Comment in `default/setup.sh` updated to note CPU torch via the cpu index.

### 3. CI — ubuntu jobs only

macOS/Windows torch is already CPU-only from PyPI, so only ubuntu jobs change.

- `build.yml`: `static-analysis`, `pre-commit`, `docs-smoke` install
  `-e ".[all]"` → `-e ".[all-cpu]"`.
- `docs.yml`: deploy job `uv sync --group docs --extra all` →
  `--extra all-cpu`.
- `build.yml` `test` matrix: the **ubuntu** torch cell installs `-e ".[torch]"`
  → `-e ".[torch-cpu]"`; macOS/Windows torch cells stay `-e ".[torch]"`.
  Implement via a matrix `torch_extra` variable (or per-include override), not an
  inline ternary, to keep the install step readable.

### 4. Documentation

- **README.md** `## Install`: keep the four one-liners; add a short "CPU vs GPU /
  uv vs pip" note and a link to the new installation page.
- **New page `docs/installation.md`**, added to the root toctree in
  `docs/index.md` ahead of `guides/index`. The inline `## Install` block in
  `index.md` collapses to a one-line pointer to the page. Contents:
  - Extras table: `torch`, `jax`, `keras`, `all`, `torch-gpu`, `jax-gpu`,
    `keras-gpu`, `torch-cpu`, `all-cpu` — what each pulls, per platform.
  - The torch-vs-jax CUDA-packaging asymmetry (why `all` isn't symmetric; why
    there is no `all-gpu`).
  - The **uv-vs-pip caveat**, stated plainly: `torch-cpu` / `all-cpu` strip CUDA
    only under uv; a plain-pip user gets the default (CUDA-on-linux) wheel. Show
    the manual escape hatch:
    `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
  - Which devcontainer flavor maps to which extra.
- MyST field-list docstring conventions do not apply (prose docs), but follow the
  existing docs style.

## Risks and verification

**Primary risk — does `uv pip install` honor `[tool.uv.sources]`?**
`uv sync` (lockfile-based, used by devcontainers) definitely respects sources and
indexes. The CI jobs use `uv pip install --system -e ".[...]"`, the pip-compat
surface, whose respect for project-declared sources is version-dependent.

Verification (in the implementation plan, before declaring done):

1. In the default devcontainer: `uv sync --extra all-cpu`, then
   `uv pip list | grep -i nvidia` returns nothing and `torch` is the `+cpu`
   build.
2. In an ubuntu-like environment: `uv pip install --system -e ".[torch-cpu]"`
   and confirm no `nvidia-*` wheels land.
3. `uv lock` succeeds with the `conflicts` declaration; committed `uv.lock`
   regenerated.
4. GPU flavors unaffected: `uv sync --extra torch-gpu` still resolves the CUDA
   build.

**Fallback if `uv pip install` ignores the source:** either add
`--index-url` / `--extra-index-url` to the affected CI install steps, or switch
those jobs from `uv pip install --system` to `uv sync` + `uv run`. No guessing —
the fallback is chosen based on step 2's observed result.

**Secondary — cpu index platform coverage:** the cpu index carries linux and
Windows wheels; macOS/Windows CI torch cells keep the plain `torch` extra, so
`torch-cpu` is only ever resolved on linux. No macOS wheel dependency introduced.

## Files touched

- `pyproject.toml` — extras, `[tool.uv.index]`, `[tool.uv.sources]`,
  `[tool.uv] conflicts`; regenerate `uv.lock`.
- `.devcontainer/default/setup.sh`, `.devcontainer/shared/install_dependencies.sh`.
- `.github/workflows/build.yml`, `.github/workflows/docs.yml`.
- `README.md`, `docs/index.md`, `docs/installation.md` (new).

## Success criteria

- Fresh `default` devcontainer build installs **zero** `nvidia-*` packages;
  `torch` is the CPU build; all three backends import; `tests/equivalence` runs.
- `pip install mononet[all]` unchanged for end users (CUDA on linux, installs on
  macOS/Windows).
- ubuntu CI no longer downloads the CUDA torch stack.
- `docs/installation.md` documents the extras matrix and the uv-vs-pip caveat;
  README links to it.
- `uv lock` / `uv sync` / `uv run mypy` / test suite green.
