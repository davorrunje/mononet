# Changelog

All notable changes to mononet will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Public package skeleton with `mononet.core`, `mononet.torch`,
  `mononet.jax`, `mononet.keras` (stub layers raising
  `NotImplementedError` — algorithm implementation in follow-up plan).
- `MonotonicityMask`, `ActivationSpec`, `InitSpec`, `MonoLinearConfig`
  framework-agnostic value objects in `mononet.core`.
- NumPy reference function signatures pinned by tests.
- Cross-backend equivalence test directory (`tests/equivalence/`)
  ready for the future harness.
- Four devcontainer flavors: `default` (CPU) + `gpu-torch`, `gpu-jax`,
  `gpu-keras` (CUDA 12.4 base, Python 3.13).
- CI matrix: 3 Python versions × 3 backends on Ubuntu + Python 3.13 on
  macOS and Windows.
- PyPI trusted publishing (OIDC) workflow.
- MkDocs site rewrite with guides, concepts, benchmarks, and about
  sections; `mike` versioning; `mkdocs-jupyter` for benchmark notebooks
  (execute: false — outputs committed).
- `NOTICE.md` with patent reservation + commercial-license contact.
- `tools/execute-benchmarks.sh` for manual notebook re-runs before
  releases.

### Changed
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
- Documentation framework with MkDocs Material
- CI/CD pipeline with GitHub Actions
- Automated version management and publishing

[Unreleased]: https://github.com/davorrunje/mononet/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/davorrunje/mononet/releases/tag/v0.1.0
