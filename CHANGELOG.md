# Changelog

All notable changes to mononet will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Public package skeleton with `mononet.core`, `mononet.torch`,
  `mononet.jax`, `mononet.keras` layers implementing the constrained
  monotonic construction.
- `MonotonicityMask`, `ActivationSpec`, `InitSpec`, `MonoConfig`
  framework-agnostic value objects in `mononet.core`.
- NumPy reference function signatures pinned by tests.
- Cross-backend equivalence test directory (`tests/equivalence/`)
  ready for the future harness.
- Four devcontainer flavors: `default` (CPU) + `gpu-torch`, `gpu-jax`,
  `gpu-keras` (CUDA 12.4 base, Python 3.13).
- CI matrix: 3 Python versions × 3 backends on Ubuntu + Python 3.13 on
  macOS and Windows.
- PyPI trusted publishing (OIDC) workflow.
- Sphinx (myst-nb) site rewrite with guides, concepts, benchmarks, and
  about sections; benchmark notebooks committed with their outputs.
- `NOTICE.md` with patent reservation + commercial-license contact.
- `tools/execute-benchmarks.sh` for manual notebook re-runs before
  releases.

### Changed
- **BREAKING:** the default `mode` is now `"absolute"` (was `"switch"`) for
  `MonoLinear` / `MonoDense` / `MonoResidual` / `MonoConfig` /
  `MonoResidualConfig`. `absolute` uses the static init (no `init` needed) and
  is the paper's base `|W|` construction — pass `mode="switch"` explicitly to
  keep the previous behaviour.
- **BREAKING:** `MonoLinear` / `MonoDense` now default `activation` to `None`,
  meaning `"identity"` — a linear monotone map, matching `torch.nn.Linear` and
  `keras.layers.Dense(activation=None)` (`MonoConfig` keeps a concrete
  `ActivationSpec("identity")` default). Layers that relied on the implicit
  ReLU are now linear — pass `activation="relu"` explicitly to restore the
  previous behavior.
- **BREAKING:** `MonoResidual` and `MonoResidualConfig` now require an explicit
  `activation` when the default `F` is built (a custom `F` must not also pass
  `activation`), preventing a silently-linear residual branch.
- Layer `activation` parameters are now typed `ActivationSpec | ActivationName`
  (`ActivationName = Literal["relu","elu","selu","softplus","identity"]`)
  instead of accepting an arbitrary `str`, so unknown names are rejected at
  type-check time.
- Relicensed from PolyForm Noncommercial License 1.0.0 to the **Apache
  License 2.0**, following AIRT Technologies Ltd.'s decision to discontinue
  patent-related activities. Apache-2.0's section 3 grants the patent
  rights needed to use the code. Effective from the first PyPI release.
- Switched LICENSE from proprietary (cookiecutter default) to
  **PolyForm Noncommercial License 1.0.0** (assignee: AIRT Technologies
  Ltd.).
- Python support range broadened from 3.13-only to 3.11–3.13.
- Removed runtime `pydantic` dependency; configs use stdlib
  `dataclasses`.

### Removed
- 1Password integration in devcontainer initialization.
- Legacy private PyPI index registry (`synthpop-pkgs`) and matching
  `UV_INDEX_SYNTHPOP_PKGS_*` workflow secrets.
- Linear workflow files (`.linear.toml`, `LINEAR_GUIDE.md`, the
  `linear-cli` Claude skill).
- Codecov configuration and CI upload step.
- Second cookiecutter devcontainer flavor (`partner`).
- `HelloWorld` placeholder in `mononet/__init__.py`.

## [0.1.0] - 2026-04-13

### Added
- Initial release of mononet
- Basic package structure with Python 3.11+ support
- Development toolchain with uv, pytest, and pre-commit hooks
- Comprehensive linting and static analysis setup
- Documentation framework with Sphinx
- CI/CD pipeline with GitHub Actions
- Automated version management and publishing

[Unreleased]: https://github.com/davorrunje/mononet/commits/main
[0.1.0]: https://github.com/davorrunje/mononet/commits/main
