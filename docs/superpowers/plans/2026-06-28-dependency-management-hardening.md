# Dependency Management Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pin the `docs`/`bench`/`dev` dependency groups and redesign `.github/dependabot.yml` so Dependabot updates are grouped, leak-free, and never bump the backend version floors.

**Architecture:** Config/metadata only — no source-code changes. Two reviewable units: (1) `pyproject.toml` specifier pins; (2) `.github/dependabot.yml` groups + ignores. Verification is `uv lock` consistency plus suite/ruff/docs staying green and YAML-validity assertions.

**Tech Stack:** `uv` (lock/sync), Dependabot v2 config (groups, `applies-to`, `ignore`), ruff, pytest, Sphinx.

## Global Constraints

- Backends stay `>=` floors in `pyproject.toml`: `torch>=2.4`, `jax>=0.4.30`, `flax>=0.10`, `keras>=3.5`, and the `-gpu` extras (`jax[cuda12]>=0.4.30`). Do NOT edit these.
- All pins must equal the current `uv.lock`-resolved versions (listed in Task 1), so resolution does not change.
- Dependabot `ignore` (uv ecosystem): bare names `torch`, `jax`, `flax`, `keras` (all update types).
- Dependabot version-update groups (in this order, first-match-wins): `lint`, `docs`, `dev`, `bench`, `tooling` (dev catch-all), `production` (prod catch-all) — each `applies-to: version-updates`.
- Dependabot security groups: `development-security`, `production-security` — each `applies-to: security-updates`, `patterns: ["*"]`.
- `github-actions` ecosystem block: unchanged. `tool.uv.override-dependencies` (`click>=8.2.1`): unchanged. `hypothesis>=6.115`: unchanged.
- No source-code changes. Branch: `chore/dependency-management` (already created, holds the spec). Never commit to `main`. Commits signed (Secretive SSH); end messages with the `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` trailer.
- All commands run from repo root `/Users/davor/Projects/PhD/mononet`.

---

### Task 1: Pin `docs`/`bench`/`dev` specifiers in `pyproject.toml`

**Files:**
- Modify: `pyproject.toml` (`[dependency-groups]` — `dev`, `docs`, `bench`)
- Possibly modify: `uv.lock` (regenerated; resolved versions must not change)

**Interfaces:**
- Consumes: nothing.
- Produces: pinned dev/docs/bench groups that Dependabot will bump as reviewed `==`→`==` edits.

- [ ] **Step 1: Pin the two omitted `dev` deps**

In `pyproject.toml`, the `dev` group's first two lines are:

```toml
    "ipython",
    "ipykernel",
```

Change them to:

```toml
    "ipython==9.13.0",
    "ipykernel==7.2.0",
```

Leave the rest of the `dev` group unchanged (including `hypothesis>=6.115`).

- [ ] **Step 2: Pin the `docs` group**

Replace the entire `docs = [ ... ]` list with:

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

- [ ] **Step 3: Pin the `bench` group**

Replace the entire `bench = [ ... ]` list with:

```toml
bench = [
    "scikit-learn==1.8.0",
    "pandas==3.0.3",
    "matplotlib==3.10.9",
]
```

- [ ] **Step 4: Re-lock and confirm resolution is unchanged**

Run:

```bash
uv lock
git diff --stat uv.lock
uv lock --check
```

Expected: `uv lock` succeeds; `uv.lock` either is unchanged or shows only
the workspace package's own constraint annotations updating (NO third-party
package version changed — the pins equal the already-resolved versions);
`uv lock --check` exits 0 (lockfile up-to-date). If `uv lock` upgrades or
downgrades any third-party package version, STOP — a pin does not match the
lockfile; report it.

- [ ] **Step 5: Confirm the environment, lint, tests, and docs stay green**

Run:

```bash
uv sync --extra all
uv run ruff check --exit-non-zero-on-fix
MONONET_TEST_BACKEND=torch uv run pytest -q tests/core tests/torch tests/equivalence tests/test_top_level_imports.py
./tools/build-docs.sh
```

Expected: `uv sync` succeeds; ruff "All checks passed!"; pytest `70 passed`;
docs build exits 0. (Pinning to the already-resolved versions must not change
anything that breaks these.)

- [ ] **Step 6: Commit**

Stage `pyproject.toml` and `uv.lock` (if it changed):

```bash
git add pyproject.toml uv.lock
git commit -m "$(cat <<'EOF'
build: pin docs/bench groups and ipython/ipykernel to resolved versions

Switch docs and bench dependency-groups from >= to == at their current
uv.lock-resolved versions, and pin the previously-unversioned ipython and
ipykernel. Backend floors (torch/jax/flax/keras) deliberately unchanged.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Redesign `.github/dependabot.yml`

**Files:**
- Modify: `.github/dependabot.yml` (the `uv` ecosystem entry: add `ignore`, replace `groups`)

**Interfaces:**
- Consumes: nothing.
- Produces: grouped, leak-free Dependabot config with backend ignores.

- [ ] **Step 1: Replace the `uv` ecosystem block**

In `.github/dependabot.yml`, leave the `version: 2` line and the entire
`github-actions` ecosystem entry unchanged. Replace the existing `uv`
ecosystem entry (the `- package-ecosystem: uv` block, from that line through
the end of its `groups:`) with:

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

- [ ] **Step 2: Verify YAML validity and structure**

Run:

```bash
uv run python - <<'PY'
import yaml
d = yaml.safe_load(open(".github/dependabot.yml"))
assert d["version"] == 2, d.get("version")
updates = d["updates"]
ga = [u for u in updates if u["package-ecosystem"] == "github-actions"]
uv = [u for u in updates if u["package-ecosystem"] == "uv"]
assert len(ga) == 1 and len(uv) == 1, (len(ga), len(uv))
uv = uv[0]
ignored = sorted(i["dependency-name"] for i in uv["ignore"])
assert ignored == ["flax", "jax", "keras", "torch"], ignored
groups = uv["groups"]
order = list(groups.keys())
assert order == ["lint","docs","dev","bench","tooling","production",
                 "development-security","production-security"], order
# version-update groups
for g in ["lint","docs","dev","bench","tooling","production"]:
    assert groups[g]["applies-to"] == "version-updates", (g, groups[g])
# security groups are dev/prod catch-alls
for g in ["development-security","production-security"]:
    assert groups[g]["applies-to"] == "security-updates", (g, groups[g])
    assert groups[g]["patterns"] == ["*"], (g, groups[g])
assert groups["development-security"]["dependency-type"] == "development"
assert groups["production-security"]["dependency-type"] == "production"
# catch-alls
assert groups["tooling"]["patterns"] == ["*"] and groups["tooling"]["dependency-type"] == "development"
assert groups["production"]["patterns"] == ["*"] and groups["production"]["dependency-type"] == "production"
# docs gained cairosvg
assert "cairosvg*" in groups["docs"]["patterns"], groups["docs"]["patterns"]
print("dependabot.yml OK")
PY
```

Expected output: `dependabot.yml OK`.

- [ ] **Step 3: Confirm the github-actions block was untouched**

Run:

```bash
uv run python - <<'PY'
import yaml
d = yaml.safe_load(open(".github/dependabot.yml"))
ga = [u for u in d["updates"] if u["package-ecosystem"] == "github-actions"][0]
assert ga["directory"] == "/", ga
assert ga["schedule"]["interval"] == "weekly", ga
assert list(ga["groups"].keys()) == ["github-actions"], ga["groups"]
assert ga["groups"]["github-actions"]["patterns"] == ["*"], ga["groups"]
print("github-actions block intact")
PY
```

Expected output: `github-actions block intact`.

- [ ] **Step 4: Commit**

```bash
git add .github/dependabot.yml
git commit -m "$(cat <<'EOF'
ci: group Dependabot updates and ignore backend version floors

Adds a bench group, a development catch-all (tooling) so transitive dev deps
batch into one PR, a production catch-all, and security-update group mirrors.
Ignores torch/jax/flax/keras so their deliberate >= floors are not bumped.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Notes for the executor

- This is a config/metadata change — no application logic. The "tests" are
  `uv lock` consistency, the existing suite/ruff/docs staying green, and the
  YAML-validity assertions.
- **Post-merge operational follow-up (NOT a code task):** after this merges to
  `main`, close the ~10 open individual Dependabot PRs (#28, #32-#36, #38-#41);
  Dependabot regenerates them as grouped batches on its next run, and drops the
  torch PR permanently via the new `ignore`. The controller handles this when
  finishing the branch — do not attempt it as part of these tasks.
- `./tools/get-version.sh` uses PCRE grep and fails on macOS; irrelevant here.
- Do not commit `dist/` (git-ignored).
