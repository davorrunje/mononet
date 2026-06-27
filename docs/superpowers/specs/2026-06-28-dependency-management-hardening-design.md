# Dependency Management Hardening — Design

**Date:** 2026-06-28
**Status:** Approved
**Scope:** Make Dependabot updates reviewable (grouped, no leaks), stop it from
bumping the deliberate backend version floors, and pin the `docs`/`bench`/`dev`
dependency groups for reproducibility.

## 1. Problem

The repo has many open individual Dependabot PRs (#28, #32-#36, #38-#41),
which are tedious to review one by one. Root causes:

- **Groups are over-constrained.** The existing `dependabot.yml` groups
  (`lint`/`docs`/`dev`/`production`) pair a `dependency-type` with *narrow*
  name patterns. Dependencies matching no listed pattern fall through to their
  own PR:
  - The `bench` deps (`scikit-learn`, `pandas`, `matplotlib`) have **no group**
    (scikit-learn PR #28).
  - Transitive dev dependencies (`cryptography`, `tornado`, `starlette`,
    `python-multipart`, `msgpack`, `pydantic-settings`) match no pattern → one
    PR each.
- **Backend floors get bumped.** `torch` PR #38 bumps a `>=` floor that is a
  deliberate minimum, not a pin to latest.
- The open PRs are routine version updates (no `security` label), so the leak
  is structural, not security-driven.

Separately, the `docs` and `bench` dependency groups use `>=` (unpinned), and
`ipython`/`ipykernel` have no version at all — undesirable for reproducible
dev/docs/bench environments.

## 2. Decisions

| Topic | Decision |
|---|---|
| Backend specifiers | Keep `>=` floors in `pyproject.toml`. Control via Dependabot `ignore`, not by editing specifiers. |
| Backend Dependabot policy | **Ignore entirely** — `torch`, `jax`, `flax`, `keras` (bare names; covers `[cuda12]`/`-gpu` extras, which share the package name). |
| `docs`/`bench` specifiers | Switch all `>=` → `==`, pinned to current `uv.lock`-resolved versions. |
| `dev` specifiers | Pin the two omitted (`ipython`, `ipykernel`); leave `hypothesis>=6.115` as-is (scope limited to docs/bench + the two omitted). `lint` already `==`. |
| Grouping | Keep semantic groups; add a `bench` group; add a `tooling` catch-all (development) so stray transitive dev deps batch; keep `production` as the production catch-all. |
| Security updates | Group via `applies-to: security-updates` mirrors — two catch-alls (`development-security`, `production-security`). |
| Open PRs | After the config PR merges, close the ~10 open individual Dependabot PRs; Dependabot regenerates grouped batches on its next run. |

## 3. Changes

### 3.1 `pyproject.toml`

Backends unchanged (`torch>=2.4`, `jax>=0.4.30`, `flax>=0.10`, `keras>=3.5`,
and the `-gpu` extras).

`[dependency-groups].docs` — pin all to resolved versions:

```toml
docs = [
    "sphinx==9.0.4",
    "pydata-sphinx-theme==0.18.0",
    "sphinx-autodoc2==0.5.0",
    "myst-nb==1.4.0",
    "sphinx-copybutton==0.5.2",
    "sphinx-design==0.7.0",
    "sphinx-togglebutton==0.4.5",
    "sphinx-autobuild==2025.8.25",
    "sphinx-polyversion==2.0.0",
    "linkify-it-py==2.1.0",
    "cairosvg==2.9.0",
]
```

`[dependency-groups].bench`:

```toml
bench = [
    "scikit-learn==1.8.0",
    "pandas==3.0.3",
    "matplotlib==3.10.9",
]
```

`[dependency-groups].dev` — pin the two omitted (other entries unchanged):

```toml
    "ipython==9.13.0",
    "ipykernel==7.2.0",
```

`hypothesis>=6.115` stays. `tool.uv.override-dependencies` (`click>=8.2.1`) is
untouched (a deliberate conflict override, not a dependency declaration).

All pins equal the current `uv.lock` resolution, so `uv lock` /
`uv sync --locked` stay consistent (verified by `uv lock --check`).

### 3.2 `.github/dependabot.yml`

The `github-actions` ecosystem block is unchanged. The `uv` ecosystem block
gains `ignore` and a redesigned `groups`:

```yaml
  - package-ecosystem: uv
    directory: "/"
    schedule:
      interval: "daily"
      time: "00:00"
    cooldown:
      default-days: 3
    ignore:
      - dependency-name: "torch"
      - dependency-name: "jax"
      - dependency-name: "flax"
      - dependency-name: "keras"
    groups:
      lint:
        applies-to: version-updates
        dependency-type: "development"
        patterns:
          - "ruff*"
          - "mypy*"
          - "bandit*"
          - "semgrep*"
          - "codespell*"
          - "detect-secrets*"
          - "pre-commit*"
          - "types-*"
      docs:
        applies-to: version-updates
        dependency-type: "development"
        patterns:
          - "sphinx*"
          - "pydata-sphinx-theme*"
          - "myst-nb*"
          - "linkify-it-py*"
          - "cairosvg*"
      dev:
        applies-to: version-updates
        dependency-type: "development"
        patterns:
          - "pytest*"
          - "hypothesis*"
          - "ipython*"
          - "ipykernel*"
          - "nest-asyncio*"
      bench:
        applies-to: version-updates
        dependency-type: "development"
        patterns:
          - "scikit-learn*"
          - "pandas*"
          - "matplotlib*"
      tooling:
        applies-to: version-updates
        dependency-type: "development"
        patterns:
          - "*"
      production:
        applies-to: version-updates
        dependency-type: "production"
        patterns:
          - "*"
      development-security:
        applies-to: security-updates
        dependency-type: "development"
        patterns:
          - "*"
      production-security:
        applies-to: security-updates
        dependency-type: "production"
        patterns:
          - "*"
```

**Group precedence:** Dependabot assigns each dependency to the first group
whose criteria it matches, so the narrow groups (`lint`, `docs`, `dev`,
`bench`) must precede the `tooling` catch-all. The catch-all collects every
remaining development dependency (including transitive ones) into a single PR.
`production` collects runtime deps (`numpy`, `typing-extensions`; backends are
ignored).

### 3.3 Operational follow-up (post-merge, manual)

After this change merges to `main`, close the open individual Dependabot PRs
(#28, #32, #33, #34, #35, #36, #38, #39, #40, #41 — whichever remain). The
torch PR (#38) is dropped permanently by the new `ignore`. Dependabot recreates
the rest as grouped batches on its next scheduled run.

## 4. Acceptance

- `pyproject.toml`: `docs` and `bench` use only `==`; `ipython`/`ipykernel`
  carry `==` pins; backends still `>=`. `uv lock --check` passes (pins match
  the lockfile) and `uv sync` succeeds.
- `.github/dependabot.yml` is valid YAML; the `uv` block has the four `ignore`
  entries and the eight groups above; `github-actions` block unchanged.
- The test suite, ruff, and the strict docs build remain green (pinning must
  not change resolved versions, so nothing should break).
- (Post-merge, not part of the code change) the next Dependabot run produces
  grouped PRs and no torch PR.

## 5. Non-goals

- Not resolving the 21 open security advisories — grouping only makes the
  resulting PRs reviewable; they are merged separately.
- Not changing backend version floors or the `click` override.
- Not pinning `hypothesis` or any `production`/backend dependency.
- No source-code changes.
