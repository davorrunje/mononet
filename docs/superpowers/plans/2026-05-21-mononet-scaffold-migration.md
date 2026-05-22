# mononet Scaffold Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the cookiecutter-scaffolded repo into a clean public open-source multi-framework package skeleton ready for algorithm implementation in a follow-up plan.

**Architecture:** Eight phases executed in order. (A) Strip the cookiecutter's company-specific bits. (B) Replace top-level docs and the license. (C) Restructure `pyproject.toml`. (D) Scaffold the `mononet/{core,torch,jax,keras}` packages with contract tests. (E) Rewrite the four devcontainer flavors. (F) Rewrite CI workflows with a per-backend matrix and OIDC publishing. (G) Update the MkDocs site. (H) Add release tooling and run final verification. After every phase the repo must build, lint, and test green so each phase boundary is a safe commit point.

**Tech Stack:** Python 3.11–3.13, uv, hatchling, pytest, ruff, mypy, stdlib `dataclasses`, MkDocs Material + `mike` + `mkdocstrings` + `mkdocs-jupyter`, PyTorch ≥2.4, JAX ≥0.4.30 + Flax NNX, Keras 3, GitHub Actions, devcontainers (CPU base + CUDA 12 base).

---

## Reference

The full design is at [`docs/superpowers/specs/2026-05-21-mononet-package-design.md`](../specs/2026-05-21-mononet-package-design.md). This plan implements §9 of that spec (the migration plan) with bite-sized TDD steps. Algorithm implementation is intentionally deferred to a future plan.

Throughout this plan, the legacy company-specific PyPI index from the cookiecutter is referred to as "the legacy private index" — concretely it is the registry named `synthpop-pkgs` configured in `pyproject.toml` and the matching `UV_INDEX_SYNTHPOP_PKGS_USERNAME` / `UV_INDEX_SYNTHPOP_PKGS_PASSWORD` environment variables in workflows.

## Conventions used in this plan

- **TDD where there is logic.** For Python code with behavior (the lazy-import scheme in `mononet/__init__.py`, the contract tests for each backend stub, the `MonotonicityMask` validation, etc.), a failing test is written first.
- **Contract tests for scaffolds.** Where the implementation is a `NotImplementedError` stub, the test asserts the public API exists, has the correct signature, and that calling the stub raises the expected error. This pins down the API surface so the follow-up algorithm plan cannot accidentally rename or reshape it.
- **Verification commands for configs.** For YAML/TOML/Dockerfile changes, the verification is a real command (`uv sync`, `mkdocs build --strict`, `python -c "import mononet"`, `actionlint`, etc.) with the expected exit code or output snippet.
- **One commit per task.** Each task ends with a single `git commit`. Phase boundaries do not need a separate commit.
- **Commit messages** use Conventional Commits (`feat:`, `chore:`, `docs:`, `ci:`, `build:`, `refactor:`, `test:`).
- **All commands run from repo root** (`/workspaces/mononet`) unless stated otherwise.

---

## Phase A — Strip cookiecutter's company-specific bits

This phase makes no functional changes to the package — it only removes files and config that point at the previous owner. After Phase A the repo should still install (`uv sync`) and tests (the trivial cookiecutter `HelloWorld` test) should still pass.

### Task A.1: Delete obsolete files

**Files:**
- Delete: `.linear.toml`
- Delete: `LINEAR_GUIDE.md`
- Delete: `.claude/skills/linear-cli/` (entire directory)
- Delete: `codecov.yml`
- Delete: `.devcontainer/partner/` (entire directory)
- Delete: `.devcontainer/default/initialize_devcontainer.sh`
- Delete: `.devcontainer/default/devcontainer.env`
- Delete: `.devcontainer/default/devcontainer.env.tmp` (if present)
- Delete: `.devcontainer/default/post-start.sh`
- Delete: `.devcontainer/default/setup_env_vars.sh`

- [ ] **Step 1: Confirm each path exists before deleting.**

Run:
```bash
ls -1 .linear.toml LINEAR_GUIDE.md codecov.yml \
      .devcontainer/default/initialize_devcontainer.sh \
      .devcontainer/default/devcontainer.env \
      .devcontainer/default/post-start.sh \
      .devcontainer/default/setup_env_vars.sh
ls -d .claude/skills/linear-cli .devcontainer/partner
```
Expected: every path listed, no `No such file or directory` errors. (`devcontainer.env.tmp` may or may not exist — that's fine.)

- [ ] **Step 2: Delete the files and directories.**

Run:
```bash
git rm -r .linear.toml LINEAR_GUIDE.md codecov.yml \
          .claude/skills/linear-cli .devcontainer/partner \
          .devcontainer/default/initialize_devcontainer.sh \
          .devcontainer/default/devcontainer.env \
          .devcontainer/default/post-start.sh \
          .devcontainer/default/setup_env_vars.sh
rm -f .devcontainer/default/devcontainer.env.tmp
```

- [ ] **Step 3: Verify nothing in the working tree still references the deleted paths.**

Run:
```bash
git grep -lE 'LINEAR_GUIDE|linear-cli|codecov\.yml|\.devcontainer/partner|initialize_devcontainer|setup_env_vars' || echo "no references"
```
Expected: `no references`. (References inside this plan or the spec under `docs/superpowers/` are fine; they describe history.)

- [ ] **Step 4: Commit.**

```bash
git commit -m "chore: remove cookiecutter company-specific files

Delete Linear workflow files, Codecov config, the legacy second
devcontainer flavor, and the 1Password initialization scripts from
the default devcontainer. No functional change to the package."
```

### Task A.2: Strip 1Password & secrets from default devcontainer

**Files:**
- Modify: `.devcontainer/default/devcontainer.json`
- Modify: `.devcontainer/default/docker-compose.yml`

- [ ] **Step 1: Edit `.devcontainer/default/devcontainer.json`.**

Replace the entire file with:
```json
{
    "name": "python-3.13",
    "dockerComposeFile": [
        "./docker-compose.yml"
    ],
    "service": "python-3.13-mononet",
    "shutdownAction": "stopCompose",
    "workspaceFolder": "/workspaces/mononet",
    "remoteEnv": {},
    "containerEnv": {
        "CLAUDE_CONFIG_DIR": "/root/.claude"
    },
    "postCreateCommand": "bash .devcontainer/shared/post-create.sh",
    "remoteUser": "root",
    "features": {
        "ghcr.io/devcontainers/features/common-utils:2": {
            "installZsh": true,
            "installOhMyZsh": true,
            "configureZshAsDefaultShell": true,
            "username": "vscode",
            "userUid": "1000",
            "userGid": "1000"
        },
        "ghcr.io/devcontainers/features/git:1": {},
        "ghcr.io/devcontainers-extra/features/apt-packages:1": {
            "packages": "gnupg git-lfs"
        },
        "ghcr.io/devcontainers/features/docker-in-docker:2": {
            "moby": false
        },
        "ghcr.io/devcontainers/features/node:1": {
            "version": "22"
        },
        "ghcr.io/devcontainers/features/github-cli:1": {}
    },
    "updateContentCommand": "bash .devcontainer/default/setup.sh",
    "mounts": [
        "source=claude-code-config-${devcontainerId},target=/root/.claude,type=volume"
    ],
    "customizations": {
        "vscode": {
            "settings": {
                "python.linting.enabled": true,
                "python.testing.pytestEnabled": true,
                "editor.formatOnSave": true,
                "editor.codeActionsOnSave": {
                    "source.organizeImports": "always"
                },
                "[python]": {
                    "editor.defaultFormatter": "ms-python.vscode-pylance"
                },
                "editor.rulers": [88],
                "terminal.integrated.defaultProfile.linux": "zsh",
                "terminal.integrated.profiles.linux": {
                    "zsh": { "path": "/bin/zsh" }
                }
            },
            "extensions": [
                "ms-python.python",
                "ms-toolsai.jupyter",
                "ms-toolsai.vscode-jupyter-cell-tags",
                "ms-toolsai.jupyter-keymap",
                "ms-toolsai.jupyter-renderers",
                "ms-toolsai.vscode-jupyter-slideshow",
                "ms-python.vscode-pylance",
                "anthropic.claude-code"
            ]
        }
    }
}
```

Changes vs the existing file: removed `initializeCommand`, `postStartCommand`, the `secrets` block, the `ms-playwright.playwright` extension (not needed), updated the ruler from 80 to 88 (matches `ruff` line length), bumped service name and `name` field to `python-3.13`, added `postCreateCommand` pointing to a shared script that Task E.1 will create, kept `updateContentCommand` pointing at a per-flavor `setup.sh` that Task E.2 will create.

- [ ] **Step 2: Edit `.devcontainer/default/docker-compose.yml`.**

Replace the entire file with:
```yaml
version: '3'
name: mononet-devcontainer

services:
  python-3.13-mononet:  # nosemgrep
    image: mcr.microsoft.com/devcontainers/python:3.13
    container_name: mononet-${USER}-python-3.13
    pull_policy: always
    volumes:
      - ../../:/workspaces/mononet:cached
    command: sleep infinity
    networks:
      - mononet-network

networks:
  mononet-network:
    name: mononet-${USER}-network
```

Changes: bumped image to `python:3.13`, renamed service to `python-3.13-mononet`, removed the `env_file` block (the 1Password-generated `devcontainer.env.tmp` no longer exists).

- [ ] **Step 3: Verify the devcontainer.json is valid JSON.**

Run:
```bash
python -m json.tool .devcontainer/default/devcontainer.json > /dev/null && echo "valid JSON"
```
Expected: `valid JSON`.

- [ ] **Step 4: Verify nothing in the working tree references the removed shell scripts.**

Run:
```bash
git grep -lE 'initialize_devcontainer|setup_env_vars|post-start\.sh|devcontainer\.env\.tmp' || echo "no references"
```
Expected: `no references`.

- [ ] **Step 5: Commit.**

```bash
git commit -am "chore(devcontainer): remove 1Password integration from default flavor

Strip initializeCommand, postStartCommand, and the secrets block.
Bump base image and service name to Python 3.13. Point postCreateCommand
at .devcontainer/shared/post-create.sh (created in Task E.1)."
```

### Task A.3: Strip legacy private-index env vars from existing workflows

**Files:**
- Modify: `.github/workflows/build.yml`
- Modify: `.github/workflows/publish.yml`
- Modify: `.github/workflows/bump-version.yml`

These workflows will be rewritten entirely in Phase F. Strip the legacy env vars now so the intermediate state is clean and so any incidental workflow run between phases doesn't fail on missing secrets.

- [ ] **Step 1: In each of the three YAML files, delete the top-level `env:` block.**

The block looks like:
```yaml
env:
  UV_INDEX_SYNTHPOP_PKGS_USERNAME: ${{ secrets.UV_INDEX_SYNTHPOP_PKGS_USERNAME }}
  UV_INDEX_SYNTHPOP_PKGS_PASSWORD: ${{ secrets.UV_INDEX_SYNTHPOP_PKGS_PASSWORD }}
  UV_INDEX_URL: https://${{ secrets.UV_INDEX_SYNTHPOP_PKGS_USERNAME }}:${{ secrets.UV_INDEX_SYNTHPOP_PKGS_PASSWORD }}@dirac.synthpop.ai/synthpop/pkgs/+simple/
```

Delete the four-line block (header + three env vars). In `build.yml` keep the existing `permissions:`, `concurrency:`, etc. In `publish.yml` also remove the references to the legacy publish URL inside the `Publish to private PyPI` step (the entire workflow will be rewritten in Task F.2, but this strip avoids a half-broken state).

- [ ] **Step 2: In `build.yml` also delete the Codecov upload step.**

Remove these lines from the `test` job:
```yaml
      - name: Upload coverage reports to Codecov
        uses: codecov/codecov-action@v6
        with:
          token: ${{ secrets.CODECOV_TOKEN }}
          files: ./coverage.xml
          flags: unittests
          name: codecov-umbrella
          fail_ci_if_error: true
          verbose: true
          functionalities: gcov,search,upload
```

- [ ] **Step 3: Verify no legacy env-var references remain in workflows.**

Run:
```bash
git grep -l 'UV_INDEX_SYNTHPOP_PKGS\|CODECOV_TOKEN\|dirac\.synthpop\.ai' .github/ || echo "no references"
```
Expected: `no references`.

- [ ] **Step 4: Verify workflow YAML still parses.**

Run:
```bash
python -c "
import yaml, pathlib
for p in pathlib.Path('.github/workflows').glob('*.yml'):
    yaml.safe_load(p.read_text())
    print(f'{p}: ok')
"
```
Expected: each workflow file prints `: ok`.

- [ ] **Step 5: Commit.**

```bash
git commit -am "ci: remove legacy private-index env vars and Codecov upload

These workflows will be fully rewritten in Phase F. Strip the legacy
references now so the intermediate repo state is clean."
```

### Task A.4: Clean codespell exclude that references a legacy file

**Files:**
- Modify: `.pre-commit-config.yaml`

- [ ] **Step 1: Edit the codespell exclude block.**

Find:
```yaml
        exclude: |
            (?x)^(
                uv.lock|
                synthpop/utils/standardize.py
            )$
```

Replace with:
```yaml
        exclude: |
            (?x)^(
                uv.lock
            )$
```

- [ ] **Step 2: Verify pre-commit config still parses.**

Run:
```bash
uv run pre-commit validate-config
```
Expected: exit code 0, no error output.

- [ ] **Step 3: Commit.**

```bash
git commit -am "chore: remove stale codespell exclude

The excluded path referenced a file from the cookiecutter source repo
that does not exist here."
```

### Task A.5: Update mkdocs.yml metadata to drop company references

**Files:**
- Modify: `docs/mkdocs.yml`

The full mkdocs.yml rewrite happens in Phase G; this small targeted edit removes the stale company reference so any intermediate `mkdocs build` invocation doesn't look weird.

- [ ] **Step 1: Edit `docs/mkdocs.yml`.**

Find:
```yaml
copyright: '&copy; 2025 <a href="https://davorrunje.github.io/" target="_blank" rel="noopener">Davor Runje</a>'
```
Change `2025` to `2026`.

Find:
```yaml
  social_image: https://opengraph.githubassets.com/1671805243.560327/synthpop-inc/mononet
```
Replace with:
```yaml
  social_image: https://opengraph.githubassets.com/1671805243.560327/davorrunje/mononet
```

- [ ] **Step 2: Verify the file still parses as YAML.**

Run:
```bash
python -c "import yaml; yaml.safe_load(open('docs/mkdocs.yml'))"
```
Expected: no output, exit code 0. (PyYAML may warn about `!!python/name:` tags; that is normal — use `yaml.unsafe_load` if needed:
```bash
python -c "import yaml; yaml.unsafe_load(open('docs/mkdocs.yml'))"
```)

- [ ] **Step 3: Commit.**

```bash
git commit -am "docs(mkdocs): point social image at the public repo"
```

---

## Phase B — License & top-level docs

After Phase B the repo's public-facing surface (license, README, NOTICE, CONTRIBUTING) reflects the new ownership and project goals. No code changes yet.

### Task B.1: Replace LICENSE with PolyForm Noncommercial 1.0.0

**Files:**
- Modify (full overwrite): `LICENSE`

- [ ] **Step 1: Replace `LICENSE` with the verbatim PolyForm Noncommercial 1.0.0 text.**

The canonical text is at <https://polyformproject.org/licenses/noncommercial/1.0.0/>. Copy it verbatim (no changes inside the license text — paraphrasing breaks SPDX recognition). Append a one-line `Licensor` annotation at the very top (above the license title), like this:

```text
Licensor: AIRT Technologies Ltd., Zagreb, Croatia.

# PolyForm Noncommercial License 1.0.0

<https://polyformproject.org/licenses/noncommercial/1.0.0>

## Acceptance

In order to get any license under these terms, you must agree
to them as both strict obligations and conditions to all your
licenses.

## Copyright License

The licensor grants you a copyright license for the
software to do everything you might do with the software
that would otherwise infringe the licensor's copyright
in it for any permitted purpose. However, you may
only distribute the software according to [Distribution
License](#distribution-license) and make changes or new works
based on the software according to [Changes and New Works
License](#changes-and-new-works-license).

## Distribution License

The licensor grants you an additional copyright license
to distribute copies of the software. Your license
to distribute covers distributing the software with
changes and new works permitted by [Changes and New Works
License](#changes-and-new-works-license).

## Changes and New Works License

The licensor grants you the rights to copy, modify, and
distribute changes and new works based on the software
for any permitted purpose.

## Patent License

The licensor grants you a patent license for the software that
covers patent claims the licensor can license, or becomes able
to license, that you would infringe by using the software in
the form provided by the licensor, before changes or new works
based on the software.

## Noncommercial Purposes

Any noncommercial purpose is a permitted purpose.

## Personal Uses

Personal use for research, experiment, and testing for
the benefit of public knowledge, personal study, private
entertainment, hobby projects, amateur pursuits, or religious
observance, without any anticipated commercial application,
is use for a permitted purpose.

## Noncommercial Organizations

Use by any charitable organization, educational institution,
public research organization, public safety or health
organization, environmental protection organization,
or government institution is use for a permitted purpose
regardless of the source of funding or obligations resulting
from the funding.

## Fair Use

You may have "fair use" rights for the software under the
law. These terms do not limit them.

## No Other Rights

These terms do not allow you to sublicense or transfer any of
your licenses to anyone else, or prevent the licensor from
granting licenses to anyone else.  These terms do not imply
any other licenses.

## Patent Defense

If you make any written claim that the software infringes or
contributes to infringement of any patent, your patent license
for the software granted under these terms ends immediately. If
your company makes such a claim, your patent license ends
immediately for work on behalf of your company.

## Violations

The first time you are notified in writing that you have
violated any of these terms, or done anything with the software
not covered by your licenses, your licenses can nonetheless
continue if you come into full compliance with these terms,
and take practical steps to correct past violations, within
32 days of receiving notice.  Otherwise, all your licenses
end immediately.

## No Liability

***As far as the law allows, the software comes as is, without
any warranty or condition, and the licensor will not be liable
to you for any damages arising out of these terms or the use
or nature of the software, under any kind of legal claim.***

## Definitions

The **licensor** is the individual or entity offering these
terms, and the **software** is the software the licensor makes
available under these terms.

**You** refers to the individual or entity agreeing to these
terms.

**Your company** is any legal entity, sole proprietorship,
or other kind of organization that you work for, plus all
organizations that have control over, are under the control of,
or are under common control with that organization.  **Control**
means ownership of substantially all the assets of an entity,
or the power to direct its management and policies by vote,
contract, or otherwise.  Control can be direct or indirect.

**Your licenses** are all the licenses granted to you for the
software under these terms.

**Use** means anything you do with the software requiring one
of your licenses.
```

> Note: this is the canonical PolyForm Noncommercial 1.0.0 text. Confirm against the source URL above; do not modify any wording inside the license itself.

- [ ] **Step 2: Confirm SPDX identifier matches.**

Run:
```bash
head -1 LICENSE
grep -c "PolyForm Noncommercial License 1.0.0" LICENSE
```
Expected: first command prints `Licensor: AIRT Technologies Ltd., Zagreb, Croatia.`; second command prints `1`.

- [ ] **Step 3: Commit.**

```bash
git commit -am "feat(license): switch to PolyForm Noncommercial 1.0.0

Replace the proprietary cookiecutter LICENSE with PolyForm
Noncommercial 1.0.0 (verbatim). Licensor: AIRT Technologies Ltd."
```

### Task B.2: Create NOTICE.md (patent + paper + commercial contact)

**Files:**
- Create: `NOTICE.md`

- [ ] **Step 1: Create `NOTICE.md`.**

```markdown
# NOTICE

`mononet` is licensed under the PolyForm Noncommercial License 1.0.0
(see `LICENSE`).

## Patent

This software implements technology covered by **U.S. Patent No.
11,551,063** ("Implementing monotonic constrained neural networks",
assignee: AIRT Technologies Ltd.). See
<https://patents.justia.com/patent/11551063>.

The PolyForm Noncommercial License covers your use of this *source code*
for noncommercial purposes. It does **not** grant any rights under the
patent, whether for commercial or noncommercial use. Practicing the
patented method (by any means, in any framework) requires a separate
patent license.

## Reference paper

Runje, D., Shankaranarayana, S. M. (2023). *Constrained Monotonic Neural
Networks.* ICML 2023. <https://arxiv.org/abs/2205.11775>

If you use `mononet` in academic work, please cite this paper. See
[`docs/docs/about/citation.md`](docs/docs/about/citation.md) for the BibTeX
entry.

## Commercial licensing

For commercial use of the code and/or a license to U.S. Patent
11,551,063, contact: **licensing@airt.ai** *(placeholder — confirm the
exact contact address before the first PyPI release)*.

## Trademarks

PyTorch is a trademark of the Linux Foundation. JAX is a trademark of
Google LLC. Keras is a trademark of Google LLC. Use of these trademark
names here indicates compatibility, not endorsement.
```

- [ ] **Step 2: Add to git and commit.**

```bash
git add NOTICE.md
git commit -m "docs: add NOTICE with patent reservation and commercial contact

References US Patent 11,551,063 (assignee: AIRT Technologies Ltd.) and
the reference paper (arxiv:2205.11775). Notes the commercial-licensing
contact placeholder."
```

### Task B.3: Rewrite README.md

**Files:**
- Modify (full overwrite): `README.md`

- [ ] **Step 1: Replace `README.md`.**

```markdown
# mononet — Unconstrained Monotonic Neural Networks

[![PyPI version](https://img.shields.io/pypi/v/mononet)](https://pypi.org/project/mononet/)
[![Python versions](https://img.shields.io/pypi/pyversions/mononet)](https://pypi.org/project/mononet/)
[![Docs](https://img.shields.io/badge/docs-mononet-blue)](https://davorrunje.github.io/mononet/)
[![Build](https://github.com/davorrunje/mononet/actions/workflows/build.yml/badge.svg)](https://github.com/davorrunje/mononet/actions/workflows/build.yml)

Reference implementation of the unconstrained monotonic neural network
construction from:

> Runje, D., Shankaranarayana, S. M. (2023). *Constrained Monotonic
> Neural Networks.* ICML 2023. <https://arxiv.org/abs/2205.11775>

First-class support for **PyTorch**, **JAX** (Flax NNX), and **Keras 3**.

## Install

    pip install "mononet[torch]"      # PyTorch
    pip install "mononet[jax]"        # JAX + Flax NNX
    pip install "mononet[keras]"      # Keras 3
    pip install "mononet[all]"        # all three

## Quick start

A 60-second tour will appear here once the algorithm implementation lands.
Each backend exposes the same composed model (`MonoMLP`) and the
framework-idiomatic layer name (`MonoLinear` for PyTorch and JAX,
`MonoDense` for Keras).

```python
# PyTorch
from mononet.torch import MonoMLP

# JAX
from mononet.jax import MonoMLP

# Keras 3
from mononet.keras import MonoMLP
```

## License & patent

Code: PolyForm Noncommercial 1.0.0. Patent: US 11,551,063 reserved
(assignee: AIRT Technologies Ltd.). Commercial users contact
**licensing@airt.ai**. See [`NOTICE.md`](NOTICE.md) for full details.

## Documentation

Full docs at <https://davorrunje.github.io/mononet/>. Source for guides
and benchmarks lives in [`docs/docs/`](docs/docs/).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the development workflow:
devcontainer choice, `uv sync`, pre-commit, per-backend test commands.

## Citation

If you use `mononet` in academic work, please cite the paper. BibTeX is
in [`docs/docs/about/citation.md`](docs/docs/about/citation.md).
```

- [ ] **Step 2: Verify no legacy references remain in README.**

Run:
```bash
grep -E '1password|synthpop|Linear|dirac' README.md && echo "PROBLEM" || echo "clean"
```
Expected: `clean`.

- [ ] **Step 3: Commit.**

```bash
git commit -am "docs: rewrite README for public open-source release

Drop the 1Password walkthrough and internal-tooling references. Add a
public install matrix, patent + license summary, and links to docs and
the reference paper."
```

### Task B.4: Rewrite CONTRIBUTING.md

**Files:**
- Modify (full overwrite): `CONTRIBUTING.md`

- [ ] **Step 1: Replace `CONTRIBUTING.md`.**

```markdown
# Contributing to mononet

Thank you for your interest in contributing to mononet! This guide
covers the development workflow.

## License & patent reminder

`mononet` is distributed under the PolyForm Noncommercial License 1.0.0,
and the underlying technique is covered by U.S. Patent 11,551,063. By
contributing, you confirm that your contribution is your own work and
that you license it under the same terms. See [`NOTICE.md`](NOTICE.md)
for the full statement. For commercial use questions, contact
**licensing@airt.ai**.

## Development environments

The repo ships four devcontainer flavors. Pick the one matching your
hardware:

| Flavor          | When to use                                                      |
|-----------------|------------------------------------------------------------------|
| `default`       | CPU work: writing code, running unit tests, building docs.       |
| `gpu-torch`     | GPU benchmarks against the paper's PyTorch baseline.             |
| `gpu-jax`       | GPU work with JAX (Flax NNX).                                    |
| `gpu-keras`     | GPU work with Keras 3 (backed by JAX with CUDA 12 by default).   |

In VS Code, `Ctrl/Cmd+Shift+P` → `Dev Containers: Reopen in Container`,
then pick the flavor by name.

Outside devcontainers, you need Python ≥3.11, [uv](https://docs.astral.sh/uv/),
and git.

## Setup

```bash
git clone https://github.com/davorrunje/mononet.git
cd mononet
uv sync                            # install runtime + dev + docs + lint
uv run pre-commit install          # install git hooks
```

## Running tests

```bash
uv run pytest                      # full suite (skips backends not installed)
uv run pytest tests/core           # framework-agnostic tests only
uv run pytest tests/torch          # PyTorch-only tests
uv run pytest tests/jax            # JAX-only tests
uv run pytest tests/keras          # Keras-only tests
uv run pytest tests/equivalence    # cross-backend numerical equivalence
```

Set the active backend with `MONONET_TEST_BACKEND={torch|jax|keras}` when
running the equivalence suite to mirror what a single CI matrix cell
does.

## Lint, format, static analysis

```bash
uv run ruff check --exit-non-zero-on-fix    # lint
uv run ruff format                           # format
uv run mypy                                  # strict type check
uv run bandit -c pyproject.toml -r mononet   # security scan
uv run semgrep scan --config auto --error    # semgrep
uv run pre-commit run --all-files            # everything pre-commit runs
```

## Building docs

```bash
./tools/build-docs.sh              # one-shot build
./tools/serve-docs.sh              # live preview
```

Benchmark notebooks under `docs/docs/benchmarks/` are committed with
their outputs and are **not** re-executed during a docs build. To
re-execute them before a release, see "Release process" below.

## Release process

1. Open a `gpu-*` devcontainer.
2. Run `./tools/execute-benchmarks.sh` to re-execute the benchmark
   notebooks against the GPU.
3. `git diff docs/docs/benchmarks/` — sanity-check the new outputs.
4. Commit the notebook updates.
5. Trigger the `Bump Version` workflow on GitHub Actions, then merge
   the resulting version PR.
6. Tag the merge commit `vX.Y.Z` and push. The `Publish` workflow ships
   the wheel to PyPI via trusted publishing; the `Docs` workflow
   deploys versioned docs with `mike`.

## Commit messages

We use [Conventional Commits](https://www.conventionalcommits.org/):
`feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `test:`, `ci:`, `build:`.

## Pull requests

See [`PULL_REQUEST_GUIDE.md`](PULL_REQUEST_GUIDE.md) for repo-specific
PR conventions. New issues go to the project's GitHub Issues tab.

## Coding conventions

- Python 3.11+, line length 88 (ruff).
- Google-style docstrings on all public functions and classes.
- Strict mypy throughout. Type hints on every function and method.
- Stdlib `dataclasses` for simple value objects; avoid adding new
  runtime dependencies without discussion.
- Tests use `pytest`. Per-backend tests live under `tests/<backend>/`
  and use `pytest.importorskip("<framework>")` so they skip cleanly
  when the backend is not installed.

## Reporting security issues

See [`SECURITY.md`](SECURITY.md).
```

- [ ] **Step 2: Commit.**

```bash
git commit -am "docs: rewrite CONTRIBUTING for the new multi-backend layout

Replace the cookiecutter's generic guide with the actual flow: four
devcontainer flavors, per-backend test commands, release process
including the manual benchmark re-execution step."
```

### Task B.5: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Remove Linear and 1Password references from `CLAUDE.md`.**

Find the "Pull Request & Linear Workflow" section near the bottom:
```markdown
## Pull Request & Linear Workflow

- **PRs**: see [PULL_REQUEST_GUIDE.md](PULL_REQUEST_GUIDE.md)
- **Linear issues**: see [LINEAR_GUIDE.md](LINEAR_GUIDE.md) or use the `linear-cli` skill in `.claude/skills/linear-cli/`
```

Replace with:
```markdown
## Pull Request Workflow

- **PRs**: see [PULL_REQUEST_GUIDE.md](PULL_REQUEST_GUIDE.md)
- **Issues**: tracked in this repository's GitHub Issues tab.
```

Find the brief "Repository:" line:
```markdown
Repository: `https://github.com/davorrunje/mononet`
```
Leave as-is — already correct.

- [ ] **Step 2: Verify no Linear references remain.**

Run:
```bash
grep -i 'linear' CLAUDE.md && echo "PROBLEM" || echo "clean"
```
Expected: `clean`.

- [ ] **Step 3: Commit.**

```bash
git commit -am "docs(CLAUDE.md): drop Linear workflow references

Linear is no longer part of the public project workflow. Issues live
on GitHub."
```

---

## Phase C — pyproject.toml restructure

After Phase C the package metadata reflects the public release shape, dependencies are correct, and `uv sync` succeeds. No package code exists yet — that's Phase D.

### Task C.1: Update `[project]` metadata and dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Replace the `[project]` block (lines starting at `[project]` and ending before `[project.optional-dependencies]`).**

```toml
[project]
name = "mononet"
version = "0.0.0"
description = "Unconstrained monotonic neural networks for PyTorch, JAX, and Keras"
authors = [{ name = "Davor Runje", email = "davor.runje@fer.hr" }]
maintainers = [{ name = "Davor Runje", email = "davor.runje@fer.hr" }]
requires-python = ">=3.11,<3.14"
readme = "README.md"
license = "LicenseRef-PolyForm-Noncommercial-1.0.0"
license-files = ["LICENSE", "NOTICE.md"]
keywords = ["monotonic", "neural-network", "pytorch", "jax", "keras", "deep-learning"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Typing :: Typed",
]
dependencies = [
    "numpy>=1.26",
    "typing-extensions>=4.12; python_version<'3.12'",
]
```

Changes vs the current file: set `requires-python` to `>=3.11,<3.14`, added `maintainers`, replaced `license` value with the SPDX LicenseRef for PolyForm, added `license-files`, added `keywords` + `classifiers`, replaced `dependencies` (dropped `pydantic`, added `numpy` + conditional `typing-extensions`).

- [ ] **Step 2: Add a `[project.urls]` block immediately after `[project]`.**

```toml
[project.urls]
Homepage      = "https://github.com/davorrunje/mononet"
Documentation = "https://davorrunje.github.io/mononet"
Repository    = "https://github.com/davorrunje/mononet"
Issues        = "https://github.com/davorrunje/mononet/issues"
Changelog     = "https://github.com/davorrunje/mononet/blob/main/CHANGELOG.md"
Paper         = "https://arxiv.org/abs/2205.11775"
Patent        = "https://patents.justia.com/patent/11551063"
```

- [ ] **Step 3: Validate that `pyproject.toml` parses.**

Run:
```bash
python -c "import tomllib, pathlib; tomllib.loads(pathlib.Path('pyproject.toml').read_text())"
```
Expected: no output, exit code 0.

- [ ] **Step 4: Commit.**

```bash
git commit -am "build(pyproject): update project metadata and dependencies

Switch license to PolyForm Noncommercial 1.0.0, set requires-python
to >=3.11,<3.14, drop pydantic from runtime deps (configs use stdlib
dataclasses), add numpy. Add classifiers and project URLs."
```

### Task C.2: Add `[project.optional-dependencies]` for all backends

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Replace the empty `[project.optional-dependencies]` block with:**

```toml
[project.optional-dependencies]
# CPU backends
torch       = ["torch>=2.4"]
jax         = ["jax>=0.4.30", "flax>=0.10"]
keras       = ["keras>=3.5", "jax>=0.4.30"]
all         = ["mononet[torch,jax,keras]"]

# GPU backends — devcontainer-level scripts configure the framework's
# CUDA wheel index when needed.
torch-gpu   = ["torch>=2.4"]
jax-gpu     = ["jax[cuda12]>=0.4.30", "flax>=0.10"]
keras-gpu   = ["keras>=3.5", "jax[cuda12]>=0.4.30"]
```

- [ ] **Step 2: Validate.**

Run:
```bash
python -c "import tomllib, pathlib; d = tomllib.loads(pathlib.Path('pyproject.toml').read_text()); assert 'all' in d['project']['optional-dependencies']"
```
Expected: no output.

- [ ] **Step 3: Commit.**

```bash
git commit -am "build(pyproject): add per-backend optional extras

Define torch / jax / keras CPU extras, an 'all' meta-extra, and
matching torch-gpu / jax-gpu / keras-gpu extras for devcontainer use."
```

### Task C.3: Restructure `[dependency-groups]`

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Replace the `[dependency-groups]` block.**

```toml
[dependency-groups]
dev = [
    "ipython",
    "ipykernel",
    "mypy==1.20.1",
    "pytest==9.0.3",
    "pytest-asyncio==1.3.0",
    "pytest-cov==7.1.0",
    "nest-asyncio==1.6.0",
    "hypothesis>=6.115",
]
docs = [
    "mkdocs-material==9.7.6",
    "mkdocstrings==1.0.3",
    "mkdocstrings-python==2.0.3",
    "mkdocs-literate-nav==0.6.3",
    "mkdocs-glightbox==0.5.2",
    "mkdocs-jupyter>=0.25",
    "mdx-include==1.4.2",
    "typer==0.24.1",
    "mkdocs-git-revision-date-localized-plugin==1.5.1",
    "mkdocs-minify-plugin==0.8.0",
    "mike==2.1.4",
]
lint = [
    "pre-commit==4.5.1",
    "mypy==1.20.1",
    "types-pyyaml==6.0.12.20260408",
    "types-setuptools==82.0.0.20260408",
    "ruff==0.15.10",
    "bandit==1.9.4",
    "semgrep==1.159.0",
    "codespell==2.4.2",
    "detect-secrets==1.5.0",
]
bench = [
    "scikit-learn>=1.5",
    "pandas>=2.2",
    "matplotlib>=3.9",
    # airtai/monotonic-nn is NOT listed here; installed via --no-deps in
    # tools/execute-benchmarks.sh to avoid its outdated typing-extensions pin.
]
```

Changes: renamed `devdocs` → `docs`, added `mkdocs-jupyter`, added `hypothesis` to `dev`, added a new `bench` group.

- [ ] **Step 2: Update `[tool.uv]` block to drop the legacy index and rename the default group.**

Find:
```toml
[tool.uv]
default-groups = [
    "dev",
    "devdocs",
    "lint",
]
override-dependencies = [
    "click>=8.2.1",
]

[[tool.uv.index]]
name = "synthpop-pkgs"
url = "https://none/synthpop/pkgs/+simple/"
```

Replace with:
```toml
[tool.uv]
default-groups = ["dev", "docs", "lint"]
```

- [ ] **Step 3: Validate.**

Run:
```bash
python -c "
import tomllib, pathlib
d = tomllib.loads(pathlib.Path('pyproject.toml').read_text())
assert 'docs' in d['dependency-groups']
assert 'devdocs' not in d['dependency-groups']
assert 'bench' in d['dependency-groups']
assert 'synthpop-pkgs' not in str(d)
print('ok')
"
```
Expected: `ok`.

- [ ] **Step 4: Commit.**

```bash
git commit -am "build(pyproject): restructure dev dependency groups

Rename devdocs -> docs, add mkdocs-jupyter (for benchmark notebook
rendering), add a 'bench' group for the manual benchmark re-execution
step, add hypothesis to 'dev'. Drop the legacy private package
index and the no-longer-needed click override-dependency."
```

### Task C.4: Remove pydantic-mypy plugin and config block

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: In the `[tool.mypy]` block, remove the `plugins` line.**

Find:
```toml
plugins = ["pydantic.mypy"]
```
Delete it.

- [ ] **Step 2: Remove the entire `[tool.pydantic-mypy]` block.**

Find and delete:
```toml
[tool.pydantic-mypy]
init_forbid_extra = true
init_typed = true
warn_required_dynamic_aliases = true
```

- [ ] **Step 3: Validate.**

Run:
```bash
python -c "
import tomllib, pathlib
d = tomllib.loads(pathlib.Path('pyproject.toml').read_text())
assert 'pydantic-mypy' not in d.get('tool', {})
assert 'plugins' not in d['tool']['mypy']
print('ok')
"
```
Expected: `ok`.

- [ ] **Step 4: Commit.**

```bash
git commit -am "build(pyproject): drop pydantic-mypy plugin

pydantic is no longer a runtime dependency. Config classes use stdlib
dataclasses (see Phase D)."
```

### Task C.5: Update `[tool.pytest.ini_options]`

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Replace the `addopts` line.**

Find:
```toml
addopts = '--cov=mononet --cov-append --cov-branch --cov-report=xml'
```

Replace with:
```toml
addopts = '--cov=mononet --cov-append --cov-branch --cov-report=term-missing'
```

Rationale: no Codecov upload anymore, so XML output is dead weight. `term-missing` gives the developer immediate feedback on uncovered lines locally.

- [ ] **Step 2: Validate.**

Run:
```bash
python -c "
import tomllib, pathlib
d = tomllib.loads(pathlib.Path('pyproject.toml').read_text())
assert 'term-missing' in d['tool']['pytest']['ini_options']['addopts']
print('ok')
"
```
Expected: `ok`.

- [ ] **Step 3: Commit.**

```bash
git commit -am "build(pyproject): switch coverage report to term-missing

Codecov upload was removed; XML report is no longer needed. The terminal
'missing' report gives developers immediate feedback on uncovered lines."
```

### Task C.6: Run `uv sync` to confirm the new pyproject installs cleanly

**Files:**
- Modify (implicitly): `uv.lock`

- [ ] **Step 1: Run `uv sync` with all extras + all dev groups.**

```bash
uv sync --all-extras
```
Expected: completes without errors, prints a list of installed packages, no mention of `synthpop-pkgs`.

- [ ] **Step 2: Verify the package itself imports.**

```bash
uv run python -c "import mononet; print(mononet.__version__)"
```
Expected: prints `0.0.0` (the `HelloWorld` placeholder still exists at this point and that is OK — it will be removed in Task D.1).

- [ ] **Step 3: Verify `pydantic` is no longer installed at the package level.**

```bash
uv pip list | grep -i pydantic && echo "PROBLEM" || echo "clean"
```
Expected: `clean`.

> If `pydantic` shows up here, it means something transitively pulls it — investigate; do not proceed.

- [ ] **Step 4: Commit the updated lockfile.**

```bash
git add uv.lock
git commit -m "build(uv.lock): regenerate after pyproject restructure

Drops pydantic and the legacy private index; adds numpy and the new
extras/groups (torch/jax/keras, hypothesis, mkdocs-jupyter, bench)."
```

---

## Phase D — Package scaffolding

After Phase D the `mononet` package has the public API surface (lazy-imported per backend) with `NotImplementedError` stubs that pin down the signature, and the test suite verifies that surface. The CI per-backend matrix will have something to run from Phase F onward.

### Task D.1: Replace `mononet/__init__.py` with the lazy public surface

**Files:**
- Modify: `mononet/__init__.py`
- Create: `tests/test_top_level_imports.py`

- [ ] **Step 1: Write the failing test first.**

Create `tests/test_top_level_imports.py`:
```python
"""Smoke tests for the top-level package import surface."""
from __future__ import annotations


def test_import_mononet_succeeds_without_any_backend() -> None:
    """`import mononet` must succeed even with no backend installed.

    The package must not eagerly import torch/jax/keras at module load.
    """
    import mononet

    assert isinstance(mononet.__version__, str)
    assert mononet.__version__ != ""


def test_no_backend_modules_imported_at_top_level() -> None:
    """Verify torch/jax/keras are not pulled in by `import mononet`."""
    import sys

    # Drop any previously imported mononet sub-modules so the test is
    # deterministic regardless of test order.
    for name in list(sys.modules):
        if name == "mononet" or name.startswith("mononet."):
            del sys.modules[name]

    import mononet  # noqa: F401

    assert "mononet.torch" not in sys.modules
    assert "mononet.jax" not in sys.modules
    assert "mononet.keras" not in sys.modules


def test_public_re_exports_core_symbols() -> None:
    """Importing core symbols from `mononet` (top-level) must work."""
    from mononet import MonotonicityMask  # noqa: F401
```

- [ ] **Step 2: Run the test; confirm it fails.**

```bash
uv run pytest tests/test_top_level_imports.py -v
```
Expected: FAIL — likely `ImportError` on `MonotonicityMask` (it does not exist yet) and possibly other issues.

- [ ] **Step 3: Replace `mononet/__init__.py`.**

```python
"""mononet — Unconstrained monotonic neural networks.

Multi-backend support for PyTorch, JAX (Flax NNX), and Keras 3.
See https://arxiv.org/abs/2205.11775 for the reference paper.

Backends are imported lazily: `import mononet` does **not** import
torch / jax / keras. Use `from mononet.torch import ...` (or the
equivalent for jax/keras) to access backend layers.
"""

from importlib.metadata import version

from mononet.core.types import MonotonicityMask

__version__ = version("mononet")

__all__ = ["MonotonicityMask", "__version__"]
```

- [ ] **Step 4: Run the test; expect it to still fail because `mononet.core.types` does not exist yet.**

```bash
uv run pytest tests/test_top_level_imports.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'mononet.core'`.

Leave the failing test in place; Task D.2 implements `core` and turns it green.

- [ ] **Step 5: Commit.**

```bash
git add mononet/__init__.py tests/test_top_level_imports.py
git commit -m "feat(mononet): replace HelloWorld placeholder with lazy public surface

The top-level package re-exports framework-agnostic symbols from
mononet.core. Backend modules (mononet.torch / .jax / .keras) are
imported lazily by the consumer, never at top-level."
```

### Task D.2: Create `mononet/core/` with stub types, config, reference, numerics

**Files:**
- Create: `mononet/core/__init__.py`
- Create: `mononet/core/types.py`
- Create: `mononet/core/config.py`
- Create: `mononet/core/reference.py`
- Create: `mononet/core/numerics.py`
- Create: `tests/core/__init__.py`
- Create: `tests/core/test_types.py`
- Create: `tests/core/test_config.py`
- Create: `tests/core/test_reference_signatures.py`

- [ ] **Step 1: Write failing test for `MonotonicityMask` in `tests/core/test_types.py`.**

```python
"""Unit tests for mononet.core.types."""
from __future__ import annotations

import numpy as np
import pytest

from mononet.core.types import ActivationSpec, MonotonicityMask


class TestMonotonicityMask:
    def test_accepts_valid_values(self) -> None:
        mask = MonotonicityMask(np.array([1, 0, -1, 0, 1], dtype=np.int8))
        assert mask.shape == (5,)

    def test_rejects_out_of_range_values(self) -> None:
        with pytest.raises(ValueError, match="must be in"):
            MonotonicityMask(np.array([2, 0, -1], dtype=np.int8))

    def test_rejects_non_1d_input(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            MonotonicityMask(np.zeros((2, 3), dtype=np.int8))

    def test_is_hashable_and_frozen(self) -> None:
        mask = MonotonicityMask(np.array([1, 0, -1], dtype=np.int8))
        # frozen dataclass: setting attributes must raise
        with pytest.raises(Exception):  # FrozenInstanceError
            mask.values = np.array([0, 0, 0], dtype=np.int8)  # type: ignore[misc]


class TestActivationSpec:
    @pytest.mark.parametrize("name", ["relu", "tanh", "sigmoid", "elu"])
    def test_accepts_known_activations(self, name: str) -> None:
        spec = ActivationSpec(name=name)
        assert spec.name == name

    def test_rejects_unknown_activation(self) -> None:
        with pytest.raises(ValueError, match="unknown activation"):
            ActivationSpec(name="frobnicate")
```

- [ ] **Step 2: Run the test; expect failure.**

```bash
uv run pytest tests/core/test_types.py -v
```
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Create `mononet/core/__init__.py`.**

```python
"""Framework-agnostic primitives shared by all backends."""

from mononet.core.types import ActivationSpec, InitSpec, MonotonicityMask

__all__ = ["ActivationSpec", "InitSpec", "MonotonicityMask"]
```

- [ ] **Step 4: Create `mononet/core/types.py`.**

```python
"""Shared types used by all mononet backends.

These dataclasses are deliberately simple value objects — no Pydantic.
Validation runs in `__post_init__`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt

_KNOWN_ACTIVATIONS: frozenset[str] = frozenset({"relu", "tanh", "sigmoid", "elu"})

ActivationName = Literal["relu", "tanh", "sigmoid", "elu"]


@dataclass(frozen=True, slots=True)
class MonotonicityMask:
    """Per-input-feature monotonicity specification.

    Each entry in `values` is one of `{-1, 0, +1}`:
    - `+1`: output should be monotonically non-decreasing in this input.
    - `-1`: output should be monotonically non-increasing in this input.
    -  `0`: no monotonicity constraint on this input.
    """

    values: npt.NDArray[np.int8]

    def __post_init__(self) -> None:
        arr = np.asarray(self.values, dtype=np.int8)
        if arr.ndim != 1:
            raise ValueError(
                f"MonotonicityMask must be 1-D; got shape {arr.shape}"
            )
        if not np.isin(arr, (-1, 0, 1)).all():
            raise ValueError(
                "MonotonicityMask values must be in {-1, 0, +1}; "
                f"got unique values {np.unique(arr).tolist()}"
            )
        # frozen dataclass — assign through object.__setattr__
        object.__setattr__(self, "values", arr)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    def __len__(self) -> int:
        return int(self.values.shape[0])


@dataclass(frozen=True, slots=True)
class ActivationSpec:
    """Backend-agnostic activation specification.

    Backends resolve `name` to their own activation function.
    """

    name: ActivationName

    def __post_init__(self) -> None:
        if self.name not in _KNOWN_ACTIVATIONS:
            raise ValueError(
                f"unknown activation {self.name!r}; "
                f"known: {sorted(_KNOWN_ACTIVATIONS)}"
            )


@dataclass(frozen=True, slots=True)
class InitSpec:
    """Weight initialization specification.

    Backends resolve `scheme` to their own initializer.
    """

    scheme: Literal["glorot_uniform", "he_normal", "lecun_normal"] = "glorot_uniform"
    seed: int | None = None
```

- [ ] **Step 5: Run the test; expect it to pass.**

```bash
uv run pytest tests/core/test_types.py -v
```
Expected: PASS — 7 tests pass.

- [ ] **Step 6: Write failing test for `MonoLinearConfig` in `tests/core/test_config.py`.**

```python
"""Unit tests for mononet.core.config."""
from __future__ import annotations

import json

import numpy as np
import pytest

from mononet.core.config import MonoLinearConfig
from mononet.core.types import ActivationSpec, InitSpec, MonotonicityMask


def _mask(n: int) -> MonotonicityMask:
    return MonotonicityMask(np.zeros(n, dtype=np.int8))


class TestMonoLinearConfig:
    def test_constructs_with_valid_args(self) -> None:
        cfg = MonoLinearConfig(
            in_features=8,
            out_features=4,
            monotonicity=_mask(8),
            activation=ActivationSpec(name="relu"),
            init=InitSpec(),
        )
        assert cfg.in_features == 8
        assert cfg.out_features == 4

    def test_rejects_non_positive_in_features(self) -> None:
        with pytest.raises(ValueError, match="in_features must be positive"):
            MonoLinearConfig(
                in_features=0,
                out_features=4,
                monotonicity=_mask(8),
                activation=ActivationSpec(name="relu"),
                init=InitSpec(),
            )

    def test_rejects_non_positive_out_features(self) -> None:
        with pytest.raises(ValueError, match="out_features must be positive"):
            MonoLinearConfig(
                in_features=8,
                out_features=-1,
                monotonicity=_mask(8),
                activation=ActivationSpec(name="relu"),
                init=InitSpec(),
            )

    def test_rejects_mismatched_mask_length(self) -> None:
        with pytest.raises(ValueError, match="mask length"):
            MonoLinearConfig(
                in_features=8,
                out_features=4,
                monotonicity=_mask(7),
                activation=ActivationSpec(name="relu"),
                init=InitSpec(),
            )

    def test_round_trips_through_json(self) -> None:
        cfg = MonoLinearConfig(
            in_features=8,
            out_features=4,
            monotonicity=MonotonicityMask(np.array([1, 1, 0, 0, -1, -1, 0, 0], dtype=np.int8)),
            activation=ActivationSpec(name="tanh"),
            init=InitSpec(scheme="he_normal", seed=42),
        )
        payload = cfg.to_json()
        # Confirm it is valid JSON.
        d = json.loads(payload)
        assert d["in_features"] == 8
        round_tripped = MonoLinearConfig.from_json(payload)
        assert round_tripped.in_features == cfg.in_features
        assert round_tripped.out_features == cfg.out_features
        assert round_tripped.activation.name == "tanh"
        assert round_tripped.init.seed == 42
        np.testing.assert_array_equal(
            round_tripped.monotonicity.values, cfg.monotonicity.values
        )
```

- [ ] **Step 7: Run the test; expect failure (config module doesn't exist).**

```bash
uv run pytest tests/core/test_config.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'mononet.core.config'`.

- [ ] **Step 8: Create `mononet/core/config.py`.**

```python
"""Backend-agnostic configuration objects.

Plain dataclasses with `__post_init__` validation. Round-trip to JSON for
benchmark reproducibility.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mononet.core.types import ActivationSpec, InitSpec, MonotonicityMask


@dataclass(frozen=True, slots=True)
class MonoLinearConfig:
    """Hyperparameters for a single monotonic linear layer."""

    in_features: int
    out_features: int
    monotonicity: MonotonicityMask
    activation: ActivationSpec
    init: InitSpec

    def __post_init__(self) -> None:
        if self.in_features <= 0:
            raise ValueError(
                f"in_features must be positive; got {self.in_features}"
            )
        if self.out_features <= 0:
            raise ValueError(
                f"out_features must be positive; got {self.out_features}"
            )
        if len(self.monotonicity) != self.in_features:
            raise ValueError(
                f"mask length ({len(self.monotonicity)}) "
                f"must equal in_features ({self.in_features})"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain-Python dict suitable for JSON encoding."""
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "monotonicity": self.monotonicity.values.tolist(),
            "activation": {"name": self.activation.name},
            "init": {"scheme": self.init.scheme, "seed": self.init.seed},
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MonoLinearConfig":
        return cls(
            in_features=int(data["in_features"]),
            out_features=int(data["out_features"]),
            monotonicity=MonotonicityMask(
                np.asarray(data["monotonicity"], dtype=np.int8)
            ),
            activation=ActivationSpec(name=data["activation"]["name"]),
            init=InitSpec(
                scheme=data["init"]["scheme"], seed=data["init"]["seed"]
            ),
        )

    @classmethod
    def from_json(cls, payload: str) -> "MonoLinearConfig":
        return cls.from_dict(json.loads(payload))
```

- [ ] **Step 9: Run the config tests; expect them to pass.**

```bash
uv run pytest tests/core/test_config.py -v
```
Expected: PASS — 5 tests pass.

- [ ] **Step 10: Create `mononet/core/reference.py` with stub functions and signatures pinned.**

```python
"""NumPy reference implementations of the monotonic primitives.

These are the **arithmetic ground truth**: every backend kernel is
asserted equivalent to these functions within a fixed tolerance
(see tests/equivalence/). Real implementations land in a follow-up plan;
this module currently raises NotImplementedError but locks down the
function signatures.

Reference paper: https://arxiv.org/abs/2205.11775
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from mononet.core.types import ActivationSpec, MonotonicityMask


def monotonic_dense(
    x: npt.NDArray[np.floating],
    weights: npt.NDArray[np.floating],
    bias: npt.NDArray[np.floating],
    mask: MonotonicityMask,
    activation: ActivationSpec,
) -> npt.NDArray[np.floating]:
    """Single-layer monotonic transformation (NumPy reference).

    Args:
        x: Input array of shape (batch, in_features).
        weights: Unconstrained weights of shape (in_features, out_features).
        bias: Bias vector of shape (out_features,).
        mask: Per-input monotonicity mask.
        activation: Activation specification.

    Returns:
        Output array of shape (batch, out_features).
    """
    raise NotImplementedError(
        "monotonic_dense reference implementation lands in the follow-up plan."
    )


def monotonic_mlp(
    x: npt.NDArray[np.floating],
    weights: list[npt.NDArray[np.floating]],
    biases: list[npt.NDArray[np.floating]],
    mask: MonotonicityMask,
    activation: ActivationSpec,
) -> npt.NDArray[np.floating]:
    """Multi-layer monotonic MLP (NumPy reference).

    Args:
        x: Input array of shape (batch, in_features).
        weights: Per-layer weight arrays.
        biases: Per-layer bias vectors.
        mask: Monotonicity mask applied to the first layer.
        activation: Activation used between hidden layers.

    Returns:
        Output array of shape (batch, weights[-1].shape[1]).
    """
    raise NotImplementedError(
        "monotonic_mlp reference implementation lands in the follow-up plan."
    )
```

- [ ] **Step 11: Write the signature-pin test.**

Create `tests/core/test_reference_signatures.py`:
```python
"""Tests that pin the public signature of the NumPy reference."""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from mononet.core import reference
from mononet.core.types import ActivationSpec, MonotonicityMask


def test_monotonic_dense_signature() -> None:
    sig = inspect.signature(reference.monotonic_dense)
    assert list(sig.parameters) == [
        "x",
        "weights",
        "bias",
        "mask",
        "activation",
    ]


def test_monotonic_mlp_signature() -> None:
    sig = inspect.signature(reference.monotonic_mlp)
    assert list(sig.parameters) == [
        "x",
        "weights",
        "biases",
        "mask",
        "activation",
    ]


def test_monotonic_dense_raises_not_implemented() -> None:
    x = np.zeros((1, 2), dtype=np.float32)
    w = np.zeros((2, 1), dtype=np.float32)
    b = np.zeros((1,), dtype=np.float32)
    with pytest.raises(NotImplementedError):
        reference.monotonic_dense(
            x, w, b,
            MonotonicityMask(np.zeros(2, dtype=np.int8)),
            ActivationSpec(name="relu"),
        )
```

- [ ] **Step 12: Run the reference tests; expect them to pass.**

```bash
uv run pytest tests/core/test_reference_signatures.py -v
```
Expected: PASS — 3 tests pass.

- [ ] **Step 13: Create `mononet/core/numerics.py` with tolerance constants.**

```python
"""Numerical tolerances and dtype helpers shared by all backends."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

# Default tolerances used by the cross-backend equivalence harness.
ATOL_FLOAT32 = 1e-5
RTOL_FLOAT32 = 1e-5
ATOL_FLOAT64 = 1e-9
RTOL_FLOAT64 = 1e-9


def default_atol(dtype: npt.DTypeLike) -> float:
    """Return the default absolute tolerance for a given floating dtype."""
    d = np.dtype(dtype)
    if d == np.float64:
        return ATOL_FLOAT64
    return ATOL_FLOAT32


def default_rtol(dtype: npt.DTypeLike) -> float:
    """Return the default relative tolerance for a given floating dtype."""
    d = np.dtype(dtype)
    if d == np.float64:
        return RTOL_FLOAT64
    return RTOL_FLOAT32
```

- [ ] **Step 14: Re-run the top-level import test now that `mononet.core.types` exists.**

```bash
uv run pytest tests/test_top_level_imports.py tests/core -v
```
Expected: PASS — all top-level + core tests green.

- [ ] **Step 15: Create empty `tests/core/__init__.py`.**

```python
```

(Empty file — makes the directory an explicit package which keeps editor tooling happy.)

- [ ] **Step 16: Commit.**

```bash
git add mononet/core tests/core
git commit -m "feat(core): add framework-agnostic types, config, and reference stubs

- MonotonicityMask, ActivationSpec, InitSpec dataclasses with __post_init__
  validation.
- MonoLinearConfig with JSON round-trip.
- NumPy reference function stubs (monotonic_dense, monotonic_mlp) with
  signatures pinned by tests. Implementations raise NotImplementedError
  and will land in the follow-up algorithm plan.
- Tolerance helpers in numerics.py for the future equivalence harness."
```

### Task D.3: Create `mononet/torch/` with stub layers

**Files:**
- Create: `mononet/torch/__init__.py`
- Create: `mononet/torch/_kernels.py`
- Create: `mononet/torch/layers.py`
- Create: `mononet/torch/models.py`
- Create: `tests/torch/__init__.py`
- Create: `tests/torch/test_public_api.py`

- [ ] **Step 1: Write the contract test first.**

Create `tests/torch/test_public_api.py`:
```python
"""Contract test for the mononet.torch public API surface."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_mono_linear_exists_and_is_nn_module() -> None:
    from mononet.torch import MonoLinear
    assert issubclass(MonoLinear, torch.nn.Module)


def test_mono_mlp_exists_and_is_nn_module() -> None:
    from mononet.torch import MonoMLP
    assert issubclass(MonoMLP, torch.nn.Module)


def test_instantiating_mono_linear_raises_not_implemented() -> None:
    import numpy as np
    from mononet.core.types import ActivationSpec, MonotonicityMask
    from mononet.torch import MonoLinear

    with pytest.raises(NotImplementedError):
        MonoLinear(
            in_features=4,
            out_features=2,
            monotonicity=MonotonicityMask(np.zeros(4, dtype=np.int8)),
            activation=ActivationSpec(name="relu"),
        )


def test_no_unexpected_top_level_exports() -> None:
    import mononet.torch as t
    expected = {"MonoLinear", "MonoMLP"}
    actual = {name for name in t.__all__}
    assert actual == expected, f"got: {actual}"
```

- [ ] **Step 2: Run the test; expect failure.**

```bash
uv run pytest tests/torch/test_public_api.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'mononet.torch'`.

- [ ] **Step 3: Create `mononet/torch/__init__.py`.**

```python
"""PyTorch backend for mononet.

Imports `torch` eagerly — only loaded when the user explicitly
imports `mononet.torch`.
"""
from mononet.torch.layers import MonoLinear
from mononet.torch.models import MonoMLP

__all__ = ["MonoLinear", "MonoMLP"]
```

- [ ] **Step 4: Create `mononet/torch/_kernels.py`.**

```python
"""Private PyTorch kernels for monotonic primitives.

Stateless functions that take tensors and return tensors. Wrapper
classes in layers.py / models.py instantiate parameters and delegate
here. Real implementations land in the follow-up algorithm plan.
"""
from __future__ import annotations

import torch

from mononet.core.types import ActivationSpec, MonotonicityMask


def monotonic_dense(
    x: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor,
    mask: MonotonicityMask,
    activation: ActivationSpec,
) -> torch.Tensor:
    """PyTorch kernel for the monotonic dense transformation."""
    raise NotImplementedError(
        "monotonic_dense PyTorch kernel lands in the follow-up plan."
    )
```

- [ ] **Step 5: Create `mononet/torch/layers.py`.**

```python
"""PyTorch idiomatic layer wrappers around mononet kernels."""
from __future__ import annotations

import torch
from torch import nn

from mononet.core.types import ActivationSpec, MonotonicityMask


class MonoLinear(nn.Module):
    """Monotonic analogue of `torch.nn.Linear`.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        monotonicity: Per-input-feature monotonicity mask.
        activation: Activation specification (resolved by the kernel).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        monotonicity: MonotonicityMask,
        activation: ActivationSpec,
    ) -> None:
        super().__init__()
        raise NotImplementedError(
            "MonoLinear PyTorch wrapper lands in the follow-up plan."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError
```

- [ ] **Step 6: Create `mononet/torch/models.py`.**

```python
"""PyTorch monotonic-model compositions."""
from __future__ import annotations

import torch
from torch import nn

from mononet.core.types import ActivationSpec, MonotonicityMask


class MonoMLP(nn.Module):
    """Multi-layer monotonic MLP, PyTorch backend."""

    def __init__(
        self,
        in_features: int,
        hidden_features: list[int],
        out_features: int,
        monotonicity: MonotonicityMask,
        activation: ActivationSpec,
    ) -> None:
        super().__init__()
        raise NotImplementedError(
            "MonoMLP PyTorch composition lands in the follow-up plan."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError
```

- [ ] **Step 7: Create `tests/torch/__init__.py`.**

```python
```

(Empty.)

- [ ] **Step 8: Run the contract test; expect it to pass.**

```bash
uv run pytest tests/torch/test_public_api.py -v
```
Expected: PASS — 4 tests pass.

- [ ] **Step 9: Commit.**

```bash
git add mononet/torch tests/torch
git commit -m "feat(torch): scaffold PyTorch backend with NotImplementedError stubs

Pin the public API (MonoLinear, MonoMLP) and the private kernel signature.
Contract tests assert the classes exist, subclass nn.Module, and that
instantiation currently raises NotImplementedError. Real implementations
land in the follow-up algorithm plan."
```

### Task D.4: Create `mononet/jax/` with stub layers

**Files:**
- Create: `mononet/jax/__init__.py`
- Create: `mononet/jax/_kernels.py`
- Create: `mononet/jax/layers.py`
- Create: `mononet/jax/models.py`
- Create: `tests/jax/__init__.py`
- Create: `tests/jax/test_public_api.py`

- [ ] **Step 1: Write the contract test.**

Create `tests/jax/test_public_api.py`:
```python
"""Contract test for the mononet.jax public API surface."""
from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
nnx = pytest.importorskip("flax.nnx")


def test_mono_linear_exists_and_is_nnx_module() -> None:
    from mononet.jax import MonoLinear
    assert issubclass(MonoLinear, nnx.Module)


def test_mono_mlp_exists_and_is_nnx_module() -> None:
    from mononet.jax import MonoMLP
    assert issubclass(MonoMLP, nnx.Module)


def test_instantiating_mono_linear_raises_not_implemented() -> None:
    import numpy as np
    from mononet.core.types import ActivationSpec, MonotonicityMask
    from mononet.jax import MonoLinear

    with pytest.raises(NotImplementedError):
        MonoLinear(
            in_features=4,
            out_features=2,
            monotonicity=MonotonicityMask(np.zeros(4, dtype=np.int8)),
            activation=ActivationSpec(name="relu"),
            rngs=nnx.Rngs(0),
        )


def test_no_unexpected_top_level_exports() -> None:
    import mononet.jax as j
    expected = {"MonoLinear", "MonoMLP"}
    assert set(j.__all__) == expected
```

- [ ] **Step 2: Run the test; expect failure.**

```bash
uv run pytest tests/jax/test_public_api.py -v
```
Expected: FAIL.

- [ ] **Step 3: Create `mononet/jax/__init__.py`.**

```python
"""JAX backend (Flax NNX) for mononet."""
from mononet.jax.layers import MonoLinear
from mononet.jax.models import MonoMLP

__all__ = ["MonoLinear", "MonoMLP"]
```

- [ ] **Step 4: Create `mononet/jax/_kernels.py`.**

```python
"""Private JAX kernels for monotonic primitives."""
from __future__ import annotations

import jax.numpy as jnp

from mononet.core.types import ActivationSpec, MonotonicityMask


def monotonic_dense(
    x: jnp.ndarray,
    weights: jnp.ndarray,
    bias: jnp.ndarray,
    mask: MonotonicityMask,
    activation: ActivationSpec,
) -> jnp.ndarray:
    """JAX kernel for the monotonic dense transformation."""
    raise NotImplementedError(
        "monotonic_dense JAX kernel lands in the follow-up plan."
    )
```

- [ ] **Step 5: Create `mononet/jax/layers.py`.**

```python
"""JAX (Flax NNX) idiomatic layer wrappers."""
from __future__ import annotations

import jax.numpy as jnp
from flax import nnx

from mononet.core.types import ActivationSpec, MonotonicityMask


class MonoLinear(nnx.Module):
    """Monotonic analogue of `flax.nnx.Linear`."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        monotonicity: MonotonicityMask,
        activation: ActivationSpec,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        raise NotImplementedError(
            "MonoLinear JAX wrapper lands in the follow-up plan."
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:  # pragma: no cover
        raise NotImplementedError
```

- [ ] **Step 6: Create `mononet/jax/models.py`.**

```python
"""JAX monotonic-model compositions."""
from __future__ import annotations

import jax.numpy as jnp
from flax import nnx

from mononet.core.types import ActivationSpec, MonotonicityMask


class MonoMLP(nnx.Module):
    """Multi-layer monotonic MLP, JAX backend."""

    def __init__(
        self,
        in_features: int,
        hidden_features: list[int],
        out_features: int,
        monotonicity: MonotonicityMask,
        activation: ActivationSpec,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        raise NotImplementedError(
            "MonoMLP JAX composition lands in the follow-up plan."
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:  # pragma: no cover
        raise NotImplementedError
```

- [ ] **Step 7: Create empty `tests/jax/__init__.py`.**

```python
```

- [ ] **Step 8: Run the contract test; expect it to pass.**

```bash
uv run pytest tests/jax/test_public_api.py -v
```
Expected: PASS.

- [ ] **Step 9: Commit.**

```bash
git add mononet/jax tests/jax
git commit -m "feat(jax): scaffold JAX (Flax NNX) backend with NotImplementedError stubs

Pin the public API (MonoLinear, MonoMLP) as nnx.Module subclasses and
the private kernel signature. Contract tests assert the classes exist
and that instantiation currently raises NotImplementedError."
```

### Task D.5: Create `mononet/keras/` with stub layers

**Files:**
- Create: `mononet/keras/__init__.py`
- Create: `mononet/keras/_kernels.py`
- Create: `mononet/keras/layers.py`
- Create: `mononet/keras/models.py`
- Create: `tests/keras/__init__.py`
- Create: `tests/keras/test_public_api.py`

- [ ] **Step 1: Write the contract test.**

Create `tests/keras/test_public_api.py`:
```python
"""Contract test for the mononet.keras public API surface."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("KERAS_BACKEND", "jax")
keras = pytest.importorskip("keras")


def test_mono_dense_exists_and_is_keras_layer() -> None:
    from mononet.keras import MonoDense
    assert issubclass(MonoDense, keras.layers.Layer)


def test_mono_mlp_exists_and_is_keras_model() -> None:
    from mononet.keras import MonoMLP
    assert issubclass(MonoMLP, keras.Model)


def test_instantiating_mono_dense_raises_not_implemented() -> None:
    import numpy as np
    from mononet.core.types import ActivationSpec, MonotonicityMask
    from mononet.keras import MonoDense

    with pytest.raises(NotImplementedError):
        MonoDense(
            units=4,
            monotonicity=MonotonicityMask(np.zeros(8, dtype=np.int8)),
            activation=ActivationSpec(name="relu"),
        )


def test_no_unexpected_top_level_exports() -> None:
    import mononet.keras as k
    expected = {"MonoDense", "MonoMLP"}
    assert set(k.__all__) == expected
```

- [ ] **Step 2: Run the test; expect failure.**

```bash
uv run pytest tests/keras/test_public_api.py -v
```
Expected: FAIL.

- [ ] **Step 3: Create `mononet/keras/__init__.py`.**

```python
"""Keras 3 backend for mononet.

Uses `keras.ops`, so the same code runs whether the user has Keras set
to a JAX, TensorFlow, or PyTorch backend.
"""
from mononet.keras.layers import MonoDense
from mononet.keras.models import MonoMLP

__all__ = ["MonoDense", "MonoMLP"]
```

- [ ] **Step 4: Create `mononet/keras/_kernels.py`.**

```python
"""Private Keras (keras.ops) kernels for monotonic primitives."""
from __future__ import annotations

from typing import Any

import keras

from mononet.core.types import ActivationSpec, MonotonicityMask


def monotonic_dense(
    x: Any,
    weights: Any,
    bias: Any,
    mask: MonotonicityMask,
    activation: ActivationSpec,
) -> Any:
    """Keras kernel for the monotonic dense transformation.

    Uses keras.ops, so it works with the JAX, TensorFlow, or PyTorch
    Keras backend.
    """
    raise NotImplementedError(
        "monotonic_dense Keras kernel lands in the follow-up plan."
    )
```

- [ ] **Step 5: Create `mononet/keras/layers.py`.**

```python
"""Keras 3 idiomatic layer wrappers."""
from __future__ import annotations

from typing import Any

import keras

from mononet.core.types import ActivationSpec, MonotonicityMask


class MonoDense(keras.layers.Layer):
    """Monotonic analogue of `keras.layers.Dense`."""

    def __init__(
        self,
        units: int,
        monotonicity: MonotonicityMask,
        activation: ActivationSpec,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        raise NotImplementedError(
            "MonoDense Keras wrapper lands in the follow-up plan."
        )

    def call(self, inputs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError
```

- [ ] **Step 6: Create `mononet/keras/models.py`.**

```python
"""Keras monotonic-model compositions."""
from __future__ import annotations

from typing import Any

import keras

from mononet.core.types import ActivationSpec, MonotonicityMask


class MonoMLP(keras.Model):
    """Multi-layer monotonic MLP, Keras backend."""

    def __init__(
        self,
        in_features: int,
        hidden_features: list[int],
        out_features: int,
        monotonicity: MonotonicityMask,
        activation: ActivationSpec,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        raise NotImplementedError(
            "MonoMLP Keras composition lands in the follow-up plan."
        )

    def call(self, inputs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError
```

- [ ] **Step 7: Create empty `tests/keras/__init__.py`.**

```python
```

- [ ] **Step 8: Run the contract test; expect it to pass.**

```bash
KERAS_BACKEND=jax uv run pytest tests/keras/test_public_api.py -v
```
Expected: PASS.

- [ ] **Step 9: Commit.**

```bash
git add mononet/keras tests/keras
git commit -m "feat(keras): scaffold Keras 3 backend with NotImplementedError stubs

MonoDense (matching keras.layers.Dense naming) and MonoMLP. Uses keras.ops
under the hood so it works regardless of the configured Keras backend.
Contract tests assert the API surface and stub behavior."
```

### Task D.6: Add PEP 561 typed marker

**Files:**
- Create: `mononet/py.typed`

- [ ] **Step 1: Create the empty marker file.**

```bash
touch mononet/py.typed
```

- [ ] **Step 2: Ensure hatch ships it.**

Open `pyproject.toml` and confirm the `[tool.hatch.build.targets.wheel]` block reads:
```toml
[tool.hatch.build.targets.wheel]
include = ["mononet"]
```
This already covers `mononet/py.typed` because the whole `mononet/` package directory is included. No change needed.

- [ ] **Step 3: Commit.**

```bash
git add mononet/py.typed
git commit -m "feat(typing): add PEP 561 py.typed marker"
```

### Task D.7: Reorganize `tests/` and add an equivalence-suite placeholder

**Files:**
- Create: `tests/__init__.py` (if missing)
- Create: `tests/equivalence/__init__.py`
- Create: `tests/equivalence/test_placeholder.py`
- Create: `tests/equivalence/cases/.gitkeep`
- Delete (if present): any leftover cookiecutter `tests/test_*.py` from the HelloWorld placeholder.

- [ ] **Step 1: Check for leftover HelloWorld tests.**

Run:
```bash
git grep -l "HelloWorld" tests/ || echo "none"
```
Expected: `none`. (If any file matches, delete it: `git rm <file>`.)

- [ ] **Step 2: Ensure `tests/__init__.py` exists.**

```bash
test -f tests/__init__.py || touch tests/__init__.py
```

- [ ] **Step 3: Create the equivalence-suite directory and placeholder test.**

```bash
mkdir -p tests/equivalence/cases
touch tests/equivalence/cases/.gitkeep
```

Create `tests/equivalence/__init__.py` (empty):
```python
```

Create `tests/equivalence/test_placeholder.py`:
```python
"""Placeholder for the cross-backend equivalence harness.

The real harness lands in the follow-up algorithm plan. This file exists
so the per-backend CI matrix has a `tests/equivalence` directory to point
at from day one.
"""
from __future__ import annotations


def test_equivalence_directory_exists() -> None:
    """Smoke test: ensures the equivalence test module is importable."""
    assert True
```

- [ ] **Step 4: Verify the full suite collects and runs.**

```bash
uv run pytest -v
```
Expected: PASS — every test in `tests/test_top_level_imports.py`, `tests/core/`, `tests/torch/`, `tests/jax/`, `tests/keras/`, and `tests/equivalence/` succeeds. (Backend-specific tests run because Phase C installed all extras; in CI single-backend cells, `pytest.importorskip` will gate them.)

- [ ] **Step 5: Commit.**

```bash
git add tests
git commit -m "test: reorganize tests into per-backend + equivalence layout

Mirrors the spec's test directory structure: tests/{core, torch, jax,
keras, equivalence}. tests/equivalence holds a placeholder test today;
the real cross-backend harness lands in the follow-up algorithm plan."
```

### Task D.8: Full local verification of Phase D

**Files:** none.

- [ ] **Step 1: Run the full test suite, lint, and type-check.**

```bash
uv run pytest -v
uv run ruff check --exit-non-zero-on-fix
uv run ruff format --check
uv run mypy
```
Expected: every command exits 0.

- [ ] **Step 2: Confirm `pip install -e ".[all]"` would work from a fresh environment.**

```bash
uv run python -c "
import mononet
import mononet.core
import mononet.torch
import mononet.jax
import mononet.keras
print('all backends import cleanly')
"
```
Expected: `all backends import cleanly`.

- [ ] **Step 3: No commit; this is just a checkpoint.**

---

## Phase E — Devcontainers

After Phase E the four devcontainer flavors are in place. Validation here is limited (we cannot fully build a CUDA container in CI's CPU environment) — JSON/YAML/Dockerfile syntax checks plus a `docker build` smoke test on the default flavor when available.

### Task E.1: Create shared scripts

**Files:**
- Create: `.devcontainer/shared/post-create.sh`
- Create: `.devcontainer/shared/install_uv.sh`
- Modify (keep): `.devcontainer/shared/install_common_tools.sh` (already exists from cookiecutter — review and update)
- Modify (keep): `.devcontainer/shared/install_dependencies.sh` (already exists — replace with backend-agnostic version)
- Modify (keep): `.devcontainer/shared/setup_path.sh` (already exists — review)

- [ ] **Step 1: Review what `shared/` currently contains.**

```bash
ls -1 .devcontainer/shared/
cat .devcontainer/shared/install_common_tools.sh
cat .devcontainer/shared/install_dependencies.sh
cat .devcontainer/shared/setup_path.sh
```
Note what's already there. The cookiecutter `install_common_tools.sh` and `setup_path.sh` are generic; keep them. `install_dependencies.sh` may reference the legacy private index — replace it.

- [ ] **Step 2: Create `.devcontainer/shared/post-create.sh`.**

```bash
#!/usr/bin/env bash
# Run once after the container is created. Installs the project (uv sync)
# and pre-commit hooks. The per-flavor setup.sh is responsible for
# choosing which extras to install.
set -euo pipefail

cd /workspaces/mononet

# Sync the project's lockfile, including any flavor-specific extras the
# per-flavor setup.sh placed in $MONONET_EXTRAS.
EXTRAS="${MONONET_EXTRAS:-all}"
echo ">>> uv sync --extra ${EXTRAS}"
uv sync --extra "${EXTRAS}"

echo ">>> installing pre-commit hooks"
uv run pre-commit install --install-hooks

echo ">>> done"
```

Make executable:
```bash
chmod +x .devcontainer/shared/post-create.sh
```

- [ ] **Step 3: Create `.devcontainer/shared/install_uv.sh`.**

```bash
#!/usr/bin/env bash
# Install uv on a base image that does not already have it.
# Used by GPU flavors built from nvidia/cuda images.
set -euo pipefail

if command -v uv >/dev/null 2>&1; then
  echo "uv already installed: $(uv --version)"
  exit 0
fi

curl -LsSf https://astral.sh/uv/install.sh | sh

# Add ~/.local/bin (the default uv install location) to the shell's PATH
# for interactive sessions launched after the install.
echo 'export PATH="$HOME/.local/bin:$PATH"' >> /root/.bashrc
echo 'export PATH="$HOME/.local/bin:$PATH"' >> /root/.zshrc 2>/dev/null || true

echo "uv installed: $($HOME/.local/bin/uv --version)"
```

Make executable:
```bash
chmod +x .devcontainer/shared/install_uv.sh
```

- [ ] **Step 4: Replace `.devcontainer/shared/install_dependencies.sh`.**

```bash
#!/usr/bin/env bash
# Backend-agnostic dependency install. Per-flavor setup.sh exports
# MONONET_EXTRAS before calling this (defaults to 'all' for CPU flavor).
set -euo pipefail

cd /workspaces/mononet

EXTRAS="${MONONET_EXTRAS:-all}"
echo ">>> installing mononet[${EXTRAS}] with dev + docs + lint groups"
uv pip install --system -e ".[${EXTRAS}]" --group=dev --group=docs --group=lint
```

Make executable:
```bash
chmod +x .devcontainer/shared/install_dependencies.sh
```

- [ ] **Step 5: Verify all shared scripts are syntactically valid bash.**

```bash
for f in .devcontainer/shared/*.sh; do
  bash -n "$f" && echo "$f: ok"
done
```
Expected: each script prints `: ok`.

- [ ] **Step 6: Commit.**

```bash
git add .devcontainer/shared
git commit -m "feat(devcontainer): shared post-create and uv install scripts

post-create.sh runs uv sync + pre-commit install for any flavor.
install_uv.sh installs uv on CUDA base images. install_dependencies.sh
respects the MONONET_EXTRAS env var set by each per-flavor setup.sh."
```

### Task E.2: Finalize the `default` (CPU) devcontainer setup script

**Files:**
- Modify: `.devcontainer/default/setup.sh`

Task A.2 already updated `devcontainer.json` and `docker-compose.yml`. Now wire `setup.sh` to the shared install script.

- [ ] **Step 1: Replace `.devcontainer/default/setup.sh`.**

```bash
#!/usr/bin/env bash
# Default (CPU) devcontainer: install all backends + dev dependencies.
set -euo pipefail

export MONONET_EXTRAS="all"
bash .devcontainer/shared/install_dependencies.sh
```

Make executable:
```bash
chmod +x .devcontainer/default/setup.sh
```

- [ ] **Step 2: Verify the script parses.**

```bash
bash -n .devcontainer/default/setup.sh && echo ok
```
Expected: `ok`.

- [ ] **Step 3: Commit.**

```bash
git add .devcontainer/default/setup.sh
git commit -m "feat(devcontainer): wire default flavor to shared install script

The default flavor installs mononet[all] + dev + docs + lint."
```

### Task E.3: Create `gpu-torch` devcontainer

**Files:**
- Create: `.devcontainer/gpu-torch/Dockerfile`
- Create: `.devcontainer/gpu-torch/devcontainer.json`
- Create: `.devcontainer/gpu-torch/docker-compose.yml`
- Create: `.devcontainer/gpu-torch/setup.sh`

- [ ] **Step 1: Create the Dockerfile.**

`.devcontainer/gpu-torch/Dockerfile`:
```dockerfile
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin:${PATH}"

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates curl git git-lfs gnupg zsh sudo \
 && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python 3.13 via uv (the base image does not ship a Python).
RUN /root/.local/bin/uv python install 3.13

WORKDIR /workspaces/mononet

CMD ["sleep", "infinity"]
```

- [ ] **Step 2: Create the docker-compose.yml.**

`.devcontainer/gpu-torch/docker-compose.yml`:
```yaml
version: '3'
name: mononet-devcontainer-gpu-torch

services:
  python-3.13-mononet-gpu-torch:  # nosemgrep
    build:
      context: .
      dockerfile: Dockerfile
    container_name: mononet-${USER}-python-3.13-gpu-torch
    volumes:
      - ../../:/workspaces/mononet:cached
    command: sleep infinity
    networks:
      - mononet-network-gpu-torch
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

networks:
  mononet-network-gpu-torch:
    name: mononet-${USER}-network-gpu-torch
```

- [ ] **Step 3: Create the devcontainer.json.**

`.devcontainer/gpu-torch/devcontainer.json`:
```json
{
    "name": "python-3.13 — GPU (PyTorch)",
    "dockerComposeFile": ["./docker-compose.yml"],
    "service": "python-3.13-mononet-gpu-torch",
    "shutdownAction": "stopCompose",
    "workspaceFolder": "/workspaces/mononet",
    "remoteEnv": {},
    "containerEnv": {
        "CLAUDE_CONFIG_DIR": "/root/.claude",
        "PATH": "/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    },
    "runArgs": ["--gpus=all"],
    "postCreateCommand": "bash .devcontainer/shared/post-create.sh",
    "remoteUser": "root",
    "features": {
        "ghcr.io/devcontainers/features/git:1": {},
        "ghcr.io/devcontainers/features/github-cli:1": {}
    },
    "updateContentCommand": "bash .devcontainer/gpu-torch/setup.sh",
    "mounts": [
        "source=claude-code-config-${devcontainerId},target=/root/.claude,type=volume"
    ],
    "customizations": {
        "vscode": {
            "settings": {
                "python.linting.enabled": true,
                "python.testing.pytestEnabled": true,
                "editor.formatOnSave": true,
                "editor.rulers": [88]
            },
            "extensions": [
                "ms-python.python",
                "ms-toolsai.jupyter",
                "ms-python.vscode-pylance",
                "anthropic.claude-code"
            ]
        }
    }
}
```

- [ ] **Step 4: Create the setup.sh.**

`.devcontainer/gpu-torch/setup.sh`:
```bash
#!/usr/bin/env bash
# GPU (PyTorch) devcontainer: install torch-gpu extra + dev/docs/lint.
set -euo pipefail

export MONONET_EXTRAS="torch-gpu"
bash .devcontainer/shared/install_dependencies.sh
```

Make executable:
```bash
chmod +x .devcontainer/gpu-torch/setup.sh
```

- [ ] **Step 5: Validate JSON and YAML.**

```bash
python -m json.tool .devcontainer/gpu-torch/devcontainer.json > /dev/null && echo "json ok"
python -c "import yaml; yaml.safe_load(open('.devcontainer/gpu-torch/docker-compose.yml'))" && echo "compose ok"
bash -n .devcontainer/gpu-torch/setup.sh && echo "bash ok"
```
Expected: all three print `ok`.

- [ ] **Step 6: Commit.**

```bash
git add .devcontainer/gpu-torch
git commit -m "feat(devcontainer): add gpu-torch flavor

CUDA 12.4 base, Python 3.13 via uv, --gpus=all runtime, installs
mononet[torch-gpu] + dev/docs/lint."
```

### Task E.4: Create `gpu-jax` devcontainer

**Files:**
- Create: `.devcontainer/gpu-jax/Dockerfile`
- Create: `.devcontainer/gpu-jax/devcontainer.json`
- Create: `.devcontainer/gpu-jax/docker-compose.yml`
- Create: `.devcontainer/gpu-jax/setup.sh`

- [ ] **Step 1: Copy and adapt the gpu-torch files.**

Use the gpu-torch templates verbatim, replacing `gpu-torch` with `gpu-jax` and `torch-gpu` with `jax-gpu`. Specifically:

`.devcontainer/gpu-jax/Dockerfile`: identical to gpu-torch's Dockerfile.

`.devcontainer/gpu-jax/docker-compose.yml`: replace all occurrences of `gpu-torch` with `gpu-jax`:
```yaml
version: '3'
name: mononet-devcontainer-gpu-jax

services:
  python-3.13-mononet-gpu-jax:  # nosemgrep
    build:
      context: .
      dockerfile: Dockerfile
    container_name: mononet-${USER}-python-3.13-gpu-jax
    volumes:
      - ../../:/workspaces/mononet:cached
    command: sleep infinity
    networks:
      - mononet-network-gpu-jax
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

networks:
  mononet-network-gpu-jax:
    name: mononet-${USER}-network-gpu-jax
```

`.devcontainer/gpu-jax/devcontainer.json`: same as gpu-torch's, with `name: "python-3.13 — GPU (JAX)"`, `service: "python-3.13-mononet-gpu-jax"`, and `updateContentCommand: "bash .devcontainer/gpu-jax/setup.sh"`.

`.devcontainer/gpu-jax/setup.sh`:
```bash
#!/usr/bin/env bash
# GPU (JAX) devcontainer: install jax-gpu extra + dev/docs/lint.
set -euo pipefail

export MONONET_EXTRAS="jax-gpu"
bash .devcontainer/shared/install_dependencies.sh
```

Make executable:
```bash
chmod +x .devcontainer/gpu-jax/setup.sh
```

- [ ] **Step 2: Validate.**

```bash
python -m json.tool .devcontainer/gpu-jax/devcontainer.json > /dev/null && echo "json ok"
python -c "import yaml; yaml.safe_load(open('.devcontainer/gpu-jax/docker-compose.yml'))" && echo "compose ok"
bash -n .devcontainer/gpu-jax/setup.sh && echo "bash ok"
```
Expected: all three print `ok`.

- [ ] **Step 3: Commit.**

```bash
git add .devcontainer/gpu-jax
git commit -m "feat(devcontainer): add gpu-jax flavor

CUDA 12.4 base, Python 3.13, installs mononet[jax-gpu] which pulls
jax[cuda12] + flax."
```

### Task E.5: Create `gpu-keras` devcontainer

**Files:**
- Create: `.devcontainer/gpu-keras/Dockerfile`
- Create: `.devcontainer/gpu-keras/devcontainer.json`
- Create: `.devcontainer/gpu-keras/docker-compose.yml`
- Create: `.devcontainer/gpu-keras/setup.sh`

- [ ] **Step 1: Create the files mirroring gpu-jax with `gpu-keras` substituted.**

`.devcontainer/gpu-keras/Dockerfile`: identical to gpu-torch/gpu-jax Dockerfile.

`.devcontainer/gpu-keras/docker-compose.yml`: substitute `gpu-keras` throughout (same pattern as Task E.4).

`.devcontainer/gpu-keras/devcontainer.json`:
- `name`: `"python-3.13 — GPU (Keras)"`
- `service`: `"python-3.13-mononet-gpu-keras"`
- `updateContentCommand`: `"bash .devcontainer/gpu-keras/setup.sh"`
- `containerEnv` must include `KERAS_BACKEND`:
  ```json
  "containerEnv": {
      "CLAUDE_CONFIG_DIR": "/root/.claude",
      "PATH": "/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
      "KERAS_BACKEND": "jax"
  }
  ```

`.devcontainer/gpu-keras/setup.sh`:
```bash
#!/usr/bin/env bash
# GPU (Keras) devcontainer: keras + jax-cuda12 by default.
# To use the PyTorch backend instead:
#   1. Re-run with MONONET_KERAS_BACKEND=torch
#   2. Set the container env var KERAS_BACKEND=torch
set -euo pipefail

export MONONET_EXTRAS="keras-gpu"
bash .devcontainer/shared/install_dependencies.sh
```

Make executable:
```bash
chmod +x .devcontainer/gpu-keras/setup.sh
```

- [ ] **Step 2: Validate.**

```bash
python -m json.tool .devcontainer/gpu-keras/devcontainer.json > /dev/null && echo "json ok"
python -c "import yaml; yaml.safe_load(open('.devcontainer/gpu-keras/docker-compose.yml'))" && echo "compose ok"
bash -n .devcontainer/gpu-keras/setup.sh && echo "bash ok"
```
Expected: all three print `ok`.

- [ ] **Step 3: Commit.**

```bash
git add .devcontainer/gpu-keras
git commit -m "feat(devcontainer): add gpu-keras flavor

CUDA 12.4 base, Python 3.13, installs mononet[keras-gpu] (keras+jax-cuda12),
sets KERAS_BACKEND=jax. Setup script notes how to switch to the PyTorch
backend if preferred."
```

---

## Phase F — CI workflows

After Phase F the CI matrix runs per backend on 3 Python versions × Ubuntu plus macOS/Windows for 3.13, docs deploy via `mike`, and PyPI publishing uses OIDC trusted publishing. The previous private-index workflow shape is gone.

### Task F.1: Rewrite `build.yml` with per-backend matrix

**Files:**
- Modify (full overwrite): `.github/workflows/build.yml`

- [ ] **Step 1: Replace `.github/workflows/build.yml`.**

```yaml
name: Build

on:
  push:
    branches: [main]
  pull_request:
  merge_group:

permissions:
  contents: read

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}-${{ github.head_ref }}
  cancel-in-progress: true

jobs:

  static-analysis:
    runs-on: ubuntu-latest
    permissions:
      contents: read
    steps:
      - uses: actions/checkout@v6
      - uses: actions/setup-python@v6
        with:
          python-version: "3.13"
      - uses: astral-sh/setup-uv@v7
      - name: Install dependencies
        run: uv pip install --system -e ".[all]" --group=dev --group=lint
      - name: Run static analysis
        run: ./tools/static-analysis.sh

  pre-commit:
    runs-on: ubuntu-latest
    permissions:
      contents: read
    steps:
      - uses: actions/checkout@v6
      - uses: actions/setup-python@v6
        with:
          python-version: "3.13"
      - uses: astral-sh/setup-uv@v7
      - name: Install dependencies
        run: uv pip install --system -e ".[all]" --group=dev --group=lint
      - uses: actions/cache@v5
        with:
          path: ~/.cache/pre-commit
          key: pre-commit|${{ hashFiles('.pre-commit-config.yaml') }}
      - uses: pre-commit/action@v3.0.1
        with:
          extra_args: --hook-stage manual --all-files

  docs-smoke:
    runs-on: ubuntu-latest
    permissions:
      contents: read
    steps:
      - uses: actions/checkout@v6
      - uses: actions/setup-python@v6
        with:
          python-version: "3.13"
      - uses: astral-sh/setup-uv@v7
      - name: Install dependencies
        run: uv pip install --system -e ".[all]" --group=docs
      - name: Build docs (strict)
        run: cd docs && mkdocs build --strict

  test:
    runs-on: ${{ matrix.os }}
    permissions:
      contents: read
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest]
        python-version: ["3.11", "3.12", "3.13"]
        backend: [torch, jax, keras]
        include:
          - { os: macos-latest,   python-version: "3.13", backend: torch }
          - { os: macos-latest,   python-version: "3.13", backend: jax }
          - { os: macos-latest,   python-version: "3.13", backend: keras }
          - { os: windows-latest, python-version: "3.13", backend: torch }
          - { os: windows-latest, python-version: "3.13", backend: jax }
          - { os: windows-latest, python-version: "3.13", backend: keras }
    steps:
      - uses: actions/checkout@v6
      - uses: actions/setup-python@v6
        with:
          python-version: ${{ matrix.python-version }}
      - uses: astral-sh/setup-uv@v7
      - name: Install dependencies
        run: uv pip install --system -e ".[${{ matrix.backend }}]" --group=dev
      - name: Run tests
        env:
          MONONET_TEST_BACKEND: ${{ matrix.backend }}
          KERAS_BACKEND: jax
        run: pytest tests/core "tests/${{ matrix.backend }}" tests/equivalence tests/test_top_level_imports.py -v

  check:
    if: always()
    runs-on: ubuntu-latest
    permissions: {}
    needs: [static-analysis, pre-commit, docs-smoke, test]
    steps:
      - name: Decide whether the needed jobs succeeded or failed
        uses: re-actors/alls-green@release/v1 # nosemgrep
        with:
          jobs: ${{ toJSON(needs) }}
```

- [ ] **Step 2: Validate workflow YAML parses.**

```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/build.yml'))" && echo ok
```
Expected: `ok`.

- [ ] **Step 3: Confirm no references to the legacy private index remain in any workflow.**

```bash
git grep -lE 'UV_INDEX_SYNTHPOP_PKGS|dirac\.synthpop\.ai|CODECOV_TOKEN' .github/ || echo "clean"
```
Expected: `clean`.

- [ ] **Step 4: Commit.**

```bash
git commit -am "ci: rewrite build.yml with per-backend matrix

18-cell matrix (3 Python versions × 3 backends on Ubuntu, plus 3 backends
on macOS-latest and Windows-latest for Python 3.13). Each cell installs
only the backend extra it tests, mirroring how users install. Adds a
docs-smoke job for PR-time mkdocs --strict validation."
```

### Task F.2: Rewrite `publish.yml` for OIDC trusted publishing

**Files:**
- Modify (full overwrite): `.github/workflows/publish.yml`

- [ ] **Step 1: Replace `.github/workflows/publish.yml`.**

```yaml
name: Publish

on:
  push:
    tags:
      - 'v*.*.*'
  workflow_dispatch:

permissions:
  id-token: write
  contents: read

jobs:
  publish:
    runs-on: ubuntu-latest
    environment:
      name: pypi
      url: https://pypi.org/p/mononet
    steps:
      - name: Checkout code
        uses: actions/checkout@v6

      - name: Set up Python
        uses: actions/setup-python@v6
        with:
          python-version: "3.13"

      - name: Install uv
        uses: astral-sh/setup-uv@v7

      - name: Get version from pyproject.toml
        id: version
        run: |
          VERSION=$(./tools/get-version.sh)
          TAG=${GITHUB_REF#refs/tags/}
          echo "package_version=$VERSION" >> $GITHUB_OUTPUT
          echo "tag=$TAG" >> $GITHUB_OUTPUT
          echo "Publishing version: $VERSION (tag: $TAG)"

      - name: Build package
        run: uv build

      - name: Verify build artifacts
        run: |
          ls -lh dist/
          test -f "dist/mononet-${{ steps.version.outputs.package_version }}-py3-none-any.whl"
          test -f "dist/mononet-${{ steps.version.outputs.package_version }}.tar.gz"

      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
```

- [ ] **Step 2: Validate.**

```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/publish.yml'))" && echo ok
```
Expected: `ok`.

- [ ] **Step 3: Add a note to release process about the one-time PyPI setup.**

Append the following to `CONTRIBUTING.md` (just above the "Commit messages" section), if not already present:
```markdown
### One-time PyPI setup (maintainer only)

Before the first release, register the project at <https://pypi.org/manage/projects/>, then under Settings → Publishing add a "trusted publisher" for:

- Owner: `davorrunje`
- Repository: `mononet`
- Workflow filename: `publish.yml`
- Environment: `pypi`

After that, every tag-pushed release publishes via OIDC with no API tokens.
```

- [ ] **Step 4: Commit.**

```bash
git add .github/workflows/publish.yml CONTRIBUTING.md
git commit -m "ci: switch publish.yml to PyPI trusted publishing (OIDC)

No more username/password secrets. The workflow declares the 'pypi'
environment; the project must be registered on PyPI with this repo +
workflow + environment configured as a trusted publisher. Document the
one-time setup in CONTRIBUTING.md."
```

### Task F.3: Update `docs.yml` for `mike` versioning, no notebook execution

**Files:**
- Create or modify: `.github/workflows/docs.yml`

Inspect current state:
```bash
ls .github/workflows/
```
If `docs.yml` exists (it was `docs-build-deploy.yml` in the cookiecutter — check), update it. Otherwise create new.

- [ ] **Step 1: Replace (or create) `.github/workflows/docs.yml`.**

```yaml
name: Docs

on:
  push:
    branches: [main]
    tags: ['v*.*.*']
  workflow_dispatch:

permissions:
  contents: write

concurrency:
  group: docs-${{ github.ref }}
  cancel-in-progress: true

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
        with:
          fetch-depth: 0

      - uses: actions/setup-python@v6
        with:
          python-version: "3.13"

      - uses: astral-sh/setup-uv@v7

      - name: Install dependencies
        run: uv pip install --system -e ".[all]" --group=docs

      - name: Configure git for mike
        run: |
          git config user.name  "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"

      - name: Deploy latest (main)
        if: github.ref == 'refs/heads/main'
        run: |
          cd docs
          mike deploy --push --update-aliases dev latest

      - name: Deploy versioned (tag)
        if: startsWith(github.ref, 'refs/tags/')
        run: |
          cd docs
          VERSION=${GITHUB_REF#refs/tags/v}
          mike deploy --push --update-aliases "$VERSION" stable
          mike set-default --push stable
```

- [ ] **Step 2: Delete the old `docs-build-deploy.yml` if it still exists.**

```bash
test -f .github/workflows/docs-build-deploy.yml && git rm .github/workflows/docs-build-deploy.yml || echo "already absent"
```

- [ ] **Step 3: Validate the new YAML.**

```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/docs.yml'))" && echo ok
```
Expected: `ok`.

- [ ] **Step 4: Commit.**

```bash
git add .github/workflows/docs.yml
git commit -am "ci(docs): replace docs deploy with mike-based versioning

Pushes to main publish the 'dev' alias and update 'latest'. Tag pushes
publish the version (e.g. '0.1.0') and update the 'stable' alias.
Notebooks are not executed during this workflow (see release process)."
```

### Task F.4: Strip legacy env vars from `bump-version.yml`

**Files:**
- Modify: `.github/workflows/bump-version.yml`

Phase A already removed the `env:` block. This task is a final pass to ensure the file is clean and the workflow still does what it should.

- [ ] **Step 1: Open `.github/workflows/bump-version.yml` and confirm no `UV_INDEX_SYNTHPOP_PKGS_*` references remain.**

```bash
grep -E 'UV_INDEX_SYNTHPOP_PKGS|dirac\.synthpop\.ai' .github/workflows/bump-version.yml && echo "PROBLEM" || echo "clean"
```
Expected: `clean`.

- [ ] **Step 2: Confirm the rest of the workflow is intact.**

```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/bump-version.yml'))" && echo ok
```
Expected: `ok`.

- [ ] **Step 3: Update the install command to use the `dev` group (no `devdocs` reference).**

Find:
```yaml
      - name: Install dependencies
        run: uv pip install --system -e "." --group={dev,lint}
```
Replace with:
```yaml
      - name: Install dependencies
        run: uv pip install --system -e "." --group=dev --group=lint
```

(The brace-expansion form is bash-only; the explicit `--group=X --group=Y` is portable.)

- [ ] **Step 4: Commit (only if Step 3 changed anything).**

```bash
git commit -am "ci(bump-version): clean up install command to use explicit groups"
```

### Task F.5: Rewrite `dependabot.yml`

**Files:**
- Modify (full overwrite): `.github/dependabot.yml`

- [ ] **Step 1: Replace `.github/dependabot.yml`.**

```yaml
version: 2

updates:
  - package-ecosystem: github-actions
    directory: /
    schedule:
      interval: weekly
    cooldown:
      default-days: 3
    groups:
      github-actions:
        patterns:
          - "*"

  - package-ecosystem: uv
    directory: "/"
    schedule:
      interval: "daily"
      time: "00:00"
    cooldown:
      default-days: 3
    groups:
      lint:
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
        dependency-type: "development"
        patterns:
          - "mkdocs*"
          - "mike*"
          - "mdx-include*"
          - "typer*"
      dev:
        dependency-type: "development"
        patterns:
          - "pytest*"
          - "hypothesis*"
          - "ipython*"
          - "ipykernel*"
          - "nest-asyncio*"
      production:
        dependency-type: "production"
        patterns:
          - "*"
```

- [ ] **Step 2: Validate.**

```bash
python -c "import yaml; yaml.safe_load(open('.github/dependabot.yml'))" && echo ok
```
Expected: `ok`.

- [ ] **Step 3: Commit.**

```bash
git commit -am "ci(dependabot): drop legacy private registry and group updates

Drop the legacy private package index registry. Group dev/lint/docs/
production updates separately so the maintainer can review each lane."
```

---

## Phase G — Docs site

After Phase G the MkDocs site builds with the new navigation, the placeholder content is in place, and the API reference is auto-generated by the existing `create_api_docs.py` hook.

### Task G.1: Update `mkdocs.yml` plugins and site URL

**Files:**
- Modify: `docs/mkdocs.yml`

- [ ] **Step 1: Update `site_name` and `site_description`.**

Find:
```yaml
site_name: Davor Runje-mononet
site_description: Implementation of unconstrained neural networks
```

Replace with:
```yaml
site_name: mononet
site_description: Unconstrained monotonic neural networks for PyTorch, JAX, and Keras
```

- [ ] **Step 2: Add the `mkdocs-jupyter` plugin to the `plugins:` block.**

Find the `plugins:` list (starts with `- search:` and includes existing plugins). Add this entry just after `- mike:` (or at the end of the list):

```yaml
  - mkdocs-jupyter:
      execute: false
      include_source: true
      ignore_h1_titles: true
```

- [ ] **Step 3: Validate that mkdocs picks up the change.**

```bash
cd docs && uv run mkdocs build --strict
cd ..
```
Expected: build succeeds. Some warnings about missing pages (`guides/...`, `concepts/...`, `benchmarks/...`, `about/...`) are expected at this point — they'll be created in subsequent tasks of Phase G. If `--strict` fails on those, temporarily run without `--strict` for this step; it must succeed with `--strict` only at the end of Phase G.

- [ ] **Step 4: Commit.**

```bash
git add docs/mkdocs.yml
git commit -m "docs(mkdocs): update site metadata and add mkdocs-jupyter plugin

site_name is now plain 'mononet'. Add mkdocs-jupyter for rendering the
benchmark notebooks under docs/docs/benchmarks/ (execute: false — outputs
are committed with the notebooks)."
```

### Task G.2: Update `navigation_template.txt`

**Files:**
- Modify: `docs/docs/navigation_template.txt`

The cookiecutter uses `literate-nav` with this template + the auto-generated API tree. Extend the template to include the new top-level sections from spec §7.

- [ ] **Step 1: Replace `docs/docs/navigation_template.txt`.**

```text
---
search:
  exclude: true
---
- [Home](index.md)
- Getting started:
    - [PyTorch](guides/pytorch.md)
    - [JAX](guides/jax.md)
    - [Keras](guides/keras.md)
- Concepts:
    - [Monotonicity](concepts/monotonicity.md)
    - [Layer reference](concepts/layers.md)
- Benchmarks:
    - [Overview](benchmarks/index.md)
- Reference - Code API
{public_api}
{api}
- About:
    - [License & patent](about/license.md)
    - [Changelog](about/changelog.md)
    - [Citation](about/citation.md)
- [Contributing](contributing.md)
```

- [ ] **Step 2: Commit (after the next tasks create the referenced pages — defer the commit to G.8 below, where the docs build is verified end-to-end).**

Note: The remaining tasks of Phase G (G.3–G.7) create the new markdown stubs. G.8 then runs `mkdocs build --strict` to verify the whole site builds, and commits the navigation + stubs together.

### Task G.3: Replace `docs/docs/index.md`

**Files:**
- Modify (full overwrite): `docs/docs/index.md`

- [ ] **Step 1: Replace `docs/docs/index.md`.**

```markdown
---
hide:
  - navigation
search:
  exclude: false
---

# mononet

**Unconstrained monotonic neural networks**, with first-class support for
**PyTorch**, **JAX** (Flax NNX), and **Keras 3**.

Reference implementation of:

> Runje, D., Shankaranarayana, S. M. (2023). *Constrained Monotonic
> Neural Networks.* ICML 2023.
> [arXiv:2205.11775](https://arxiv.org/abs/2205.11775)

## Install

    pip install "mononet[torch]"      # PyTorch
    pip install "mononet[jax]"        # JAX + Flax NNX
    pip install "mononet[keras]"      # Keras 3
    pip install "mononet[all]"        # all three

## Where to go next

- [PyTorch guide](guides/pytorch.md)
- [JAX guide](guides/jax.md)
- [Keras guide](guides/keras.md)
- [Concepts: monotonicity](concepts/monotonicity.md)
- [Benchmarks (reproducing the paper)](benchmarks/index.md)

## License & patent

Code: PolyForm Noncommercial 1.0.0. Patent: US 11,551,063 reserved
(assignee: AIRT Technologies Ltd.). Commercial users contact
**licensing@airt.ai**. See [License & patent](about/license.md).
```

### Task G.4: Create `guides/` stubs

**Files:**
- Create: `docs/docs/guides/pytorch.md`
- Create: `docs/docs/guides/jax.md`
- Create: `docs/docs/guides/keras.md`

- [ ] **Step 1: Create `docs/docs/guides/pytorch.md`.**

```markdown
# PyTorch guide

`mononet.torch` exposes `MonoLinear` and `MonoMLP`, both subclasses of
`torch.nn.Module`. They drop into any existing training loop (plain
PyTorch, PyTorch Lightning, etc.).

## Install

    pip install "mononet[torch]"

## Public API

- [`MonoLinear`](../api/mononet/torch/MonoLinear.md) — monotonic
  analogue of `torch.nn.Linear`.
- [`MonoMLP`](../api/mononet/torch/MonoMLP.md) — multi-layer composition.

A worked example lands once the algorithm implementation is in.

## See also

- [Concepts: monotonicity](../concepts/monotonicity.md)
- [Layer reference](../concepts/layers.md)
- [Benchmarks](../benchmarks/index.md)
```

- [ ] **Step 2: Create `docs/docs/guides/jax.md`.**

```markdown
# JAX guide

`mononet.jax` uses **Flax NNX** — the new object-oriented Flax API. Layers
are `flax.nnx.Module` subclasses, fully compatible with `jax.jit` and
`jax.grad`.

## Install

    pip install "mononet[jax]"

## Public API

- [`MonoLinear`](../api/mononet/jax/MonoLinear.md)
- [`MonoMLP`](../api/mononet/jax/MonoMLP.md)

A worked example lands once the algorithm implementation is in.

## See also

- [Concepts: monotonicity](../concepts/monotonicity.md)
- [Benchmarks](../benchmarks/index.md)
```

- [ ] **Step 3: Create `docs/docs/guides/keras.md`.**

```markdown
# Keras 3 guide

`mononet.keras` uses `keras.ops`, so the same code runs whether Keras is
configured to use JAX, TensorFlow, or PyTorch under the hood. The
GPU devcontainer ships with `KERAS_BACKEND=jax`.

## Install

    pip install "mononet[keras]"

## Public API

- [`MonoDense`](../api/mononet/keras/MonoDense.md) — monotonic analogue
  of `keras.layers.Dense`.
- [`MonoMLP`](../api/mononet/keras/MonoMLP.md)

A worked example lands once the algorithm implementation is in.

## See also

- [Concepts: monotonicity](../concepts/monotonicity.md)
- [Benchmarks](../benchmarks/index.md)
```

### Task G.5: Create `concepts/` stubs

**Files:**
- Create: `docs/docs/concepts/monotonicity.md`
- Create: `docs/docs/concepts/layers.md`

- [ ] **Step 1: Create `docs/docs/concepts/monotonicity.md`.**

```markdown
# Monotonicity

A function `f(x_1, ..., x_n)` is **monotonically non-decreasing** in input
`x_i` if `x_i ≤ x_i'` (holding others fixed) implies `f(...) ≤ f(...)`.
Symmetrically, it is **non-increasing** if `x_i ≤ x_i'` implies
`f(...) ≥ f(...)`. A `0` entry in the monotonicity mask means no
constraint on that input.

mononet implements the construction from Runje &
Shankaranarayana (2023) which yields **provably monotonic** networks
without constraining the underlying weights — see the paper at
<https://arxiv.org/abs/2205.11775> for the proof.

## API

See [`MonotonicityMask`](../api/mononet/core/types/MonotonicityMask.md) for
the type used to declare per-input monotonicity in code.
```

- [ ] **Step 2: Create `docs/docs/concepts/layers.md`.**

```markdown
# Layer reference

Each backend mirrors its host framework's vocabulary for the analogous
unconstrained layer:

| Concept                  | PyTorch                  | JAX (Flax NNX)             | Keras 3                       |
|--------------------------|--------------------------|----------------------------|-------------------------------|
| Single monotonic layer   | `MonoLinear`             | `MonoLinear`               | `MonoDense`                   |
| Composed MLP             | `MonoMLP`                | `MonoMLP`                  | `MonoMLP`                     |

PyTorch and Flax NNX both call the standard analog `Linear`, so the
monotonic version is `MonoLinear` in those backends. Keras calls it
`Dense`, so the monotonic version is `MonoDense`.

The composed MLP shares the name `MonoMLP` across all three backends
since "MLP" is universal.

Pure-function NumPy reference implementations under
`mononet.core.reference` (`monotonic_dense`, `monotonic_mlp`) provide the
arithmetic ground truth used by the cross-backend equivalence tests.
```

### Task G.6: Create `benchmarks/` stubs

**Files:**
- Create: `docs/docs/benchmarks/index.md`
- Create: `docs/docs/benchmarks/00-overview.ipynb`

- [ ] **Step 1: Create `docs/docs/benchmarks/index.md`.**

```markdown
# Benchmarks — reproducing the paper

These notebooks reproduce experiments from
[Runje & Shankaranarayana (2023)](https://arxiv.org/abs/2205.11775) using
`mononet`. They are committed with their outputs and re-executed manually
before each release — see
[`CONTRIBUTING.md`](../../../CONTRIBUTING.md#release-process).

Each notebook also benchmarks against `airtai/monotonic-nn`
(the paper's original PyTorch reference) installed via the `bench`
dependency group.

## Notebooks

- [Overview](00-overview.ipynb) — placeholder; full set of benchmarks
  lands in the follow-up algorithm plan.
```

- [ ] **Step 2: Create a minimal placeholder notebook.**

Use the following exact content for `docs/docs/benchmarks/00-overview.ipynb`. (This is valid Jupyter JSON; the engineer should write it byte-for-byte.)

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Benchmarks overview (placeholder)\n",
    "\n",
    "The full benchmark notebooks reproducing the paper's tables land in the follow-up algorithm plan."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 3: Sanity-check the notebook is valid JSON.**

```bash
python -m json.tool docs/docs/benchmarks/00-overview.ipynb > /dev/null && echo "valid notebook"
```
Expected: `valid notebook`.

### Task G.7: Create `about/` stubs

**Files:**
- Create: `docs/docs/about/license.md`
- Create: `docs/docs/about/changelog.md`
- Create: `docs/docs/about/citation.md`

- [ ] **Step 1: Create `docs/docs/about/license.md`.**

```markdown
# License & patent

## Code license

`mononet` is licensed under the **PolyForm Noncommercial License 1.0.0**.
See the [full text](https://polyformproject.org/licenses/noncommercial/1.0.0/)
or the `LICENSE` file in the repository.

Noncommercial purposes — research, education, hobby projects, public-sector
work — are permitted with attribution. Commercial use of the code requires
a separate license.

## Patent

The technique implemented by `mononet` is covered by **U.S. Patent No.
11,551,063** ("Implementing monotonic constrained neural networks").
Assignee: **AIRT Technologies Ltd.**, Zagreb, Croatia. See
<https://patents.justia.com/patent/11551063>.

The PolyForm Noncommercial code license does **not** grant any rights
under this patent. Practicing the patented method — in any framework, by
any means — requires a separate patent license, regardless of whether the
use is commercial.

## Commercial licensing

For commercial use of the code and/or a license to U.S. Patent 11,551,063,
contact **licensing@airt.ai**.
```

- [ ] **Step 2: Create `docs/docs/about/changelog.md`.**

```markdown
# Changelog

The repository's `CHANGELOG.md` is authoritative — this page mirrors it.

{!../../CHANGELOG.md!}
```

(The `{!path!}` syntax is the `mdx-include` plugin already configured in `mkdocs.yml`.)

- [ ] **Step 3: Create `docs/docs/about/citation.md`.**

```markdown
# Citation

If you use `mononet` in academic work, please cite the reference paper:

```bibtex
@inproceedings{runje2023constrained,
  title     = {Constrained Monotonic Neural Networks},
  author    = {Runje, Davor and Shankaranarayana, Sharath M.},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {202},
  year      = {2023},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v202/runje23a.html},
  eprint    = {2205.11775},
  archivePrefix = {arXiv}
}
```

> Note: confirm the exact BibTeX entry against the PMLR proceedings page
> before the first PyPI release — venue, volume, and URL fields are
> sensitive to typos.
```

### Task G.8: Verify the docs site builds and commit Phase G changes

**Files:** none new; this is a verification + commit task.

- [ ] **Step 1: Run a strict docs build.**

```bash
cd docs
uv run mkdocs build --strict
cd ..
```
Expected: build succeeds with no warnings.

> If you see warnings about missing files like `api/mononet/...`, run the `create_api_docs.py` hook explicitly first:
> ```bash
> cd docs && uv run python -c "from create_api_docs import create_api_docs; create_api_docs()"
> ```
> The hook walks the `mononet` package and generates API stub pages.

- [ ] **Step 2: Optional — preview the site locally.**

```bash
./tools/serve-docs.sh
```
Open the URL it prints, click through the new nav, confirm all pages render. `Ctrl-C` to stop.

- [ ] **Step 3: Commit all Phase G changes in one commit.**

```bash
git add docs/
git commit -m "docs: rewrite site navigation and add public content stubs

- New top-level sections: Getting started (PyTorch/JAX/Keras), Concepts
  (monotonicity, layers), Benchmarks, About (license, changelog, citation).
- mkdocs-jupyter plugin wired in (execute: false; outputs committed
  with notebooks).
- Index page rewritten as a short public pitch with install matrix.
- API reference continues to be auto-generated by create_api_docs.py."
```

---

## Phase H — Release tooling & final verification

### Task H.1: Create `tools/execute-benchmarks.sh`

**Files:**
- Create: `tools/execute-benchmarks.sh`

- [ ] **Step 1: Create the script.**

```bash
#!/usr/bin/env bash
# Re-execute all benchmark notebooks in place.
# Run from a gpu-* devcontainer so the timings reflect real hardware.
# Intended to be run manually before tagging a release; see
# CONTRIBUTING.md "Release process".
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

# Ensure the bench group is installed.
uv sync --group bench

# The paper's original PyTorch implementation is installed without its
# own dependency metadata, because its old typing-extensions pin
# conflicts with modern Python+TF. It is only imported for direct
# numerical comparison inside the notebooks; its own deps are not
# needed at runtime.
echo ">>> installing airtai/monotonic-nn (no-deps) for paper-baseline comparison"
uv pip install --no-deps "git+https://github.com/airtai/monotonic-nn"

echo ">>> executing notebooks under docs/docs/benchmarks/"
uv run jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=14400 \
  docs/docs/benchmarks/*.ipynb

echo
echo ">>> done. Review diffs:"
echo "    git diff docs/docs/benchmarks/"
echo
echo ">>> then commit and tag a release."
```

Make executable:
```bash
chmod +x tools/execute-benchmarks.sh
```

- [ ] **Step 2: Verify the script parses and runs `--help` cleanly when given a non-existent target (it should error gracefully if there are no notebooks beyond the placeholder).**

```bash
bash -n tools/execute-benchmarks.sh && echo ok
```
Expected: `ok`.

- [ ] **Step 3: Commit.**

```bash
git add tools/execute-benchmarks.sh
git commit -m "feat(tools): add execute-benchmarks.sh for manual notebook re-runs

Manual step in the release process: run inside a GPU devcontainer, review
diffs in committed notebook outputs, then tag a release."
```

### Task H.2: Verify the existing `tools/*` scripts still work

**Files:** none modified (verification only).

- [ ] **Step 1: Test each script briefly.**

```bash
./tools/build-docs.sh
./tools/lint.sh
./tools/static-analysis.sh
./tools/get-version.sh
```
Expected:
- `build-docs.sh` builds the site successfully.
- `lint.sh` runs ruff check + ruff format with no changes needed (Phase D code is already formatted).
- `static-analysis.sh` runs mypy, bandit, semgrep without errors.
- `get-version.sh` prints `0.0.0`.

> If any script references the legacy private index or 1Password, fix in-place and commit.

- [ ] **Step 2: No commit required if no script needed changes.**

### Task H.3: Final full local verification

**Files:** none.

- [ ] **Step 1: Run the entire pre-commit suite.**

```bash
uv run pre-commit run --all-files
```
Expected: every hook passes.

- [ ] **Step 2: Run the full test suite.**

```bash
uv run pytest -v
```
Expected: every test passes (with each backend running its own tests since all extras are installed in the local dev environment).

- [ ] **Step 3: Build docs strict.**

```bash
cd docs && uv run mkdocs build --strict && cd ..
```
Expected: build succeeds with no warnings.

- [ ] **Step 4: Confirm the wheel builds.**

```bash
uv build
ls -lh dist/
```
Expected: `dist/mononet-0.0.0-py3-none-any.whl` and `dist/mononet-0.0.0.tar.gz` exist.

- [ ] **Step 5: Inspect the wheel contents to confirm `LICENSE`, `NOTICE.md`, and `py.typed` are present.**

```bash
uv run python -m zipfile -l dist/mononet-0.0.0-py3-none-any.whl | grep -E '(LICENSE|NOTICE|py\.typed)'
```
Expected: lists `LICENSE`, `NOTICE.md`, and `mononet/py.typed`.

- [ ] **Step 6: Clean up build artifacts (do not commit `dist/`).**

```bash
rm -rf dist/
```

- [ ] **Step 7: No commit required — this is a checkpoint.**

### Task H.4: Update `CHANGELOG.md` with the migration entry

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add an `[Unreleased]` entry at the top of `CHANGELOG.md`.**

```markdown
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
```

- [ ] **Step 2: Commit.**

```bash
git commit -am "docs(changelog): record scaffold-migration changes"
```

---

## Self-review against the spec

Before declaring the plan ready, verify each spec section maps to a task.

| Spec section                                  | Implemented by tasks |
|-----------------------------------------------|----------------------|
| §1 Goals & non-goals                          | implicit throughout  |
| §2 Package layout (top-level dirs + files)    | Phase A (deletions), Phase D (mononet/), Phase G (docs/), Phase E (.devcontainer/), Phase F (.github/), Tasks B.1–B.5 (top-level docs), Task H.1 (tools/) |
| §2 Public import surface                      | Task D.1 (lazy `__init__.py`) + each backend's `__init__.py` in D.3/D.4/D.5 |
| §2 Naming convention table                    | Tasks D.3 (MonoLinear/MonoMLP), D.4 (same), D.5 (MonoDense/MonoMLP) |
| §2 Files removed from cookiecutter            | Tasks A.1–A.5, D.1 (HelloWorld removed) |
| §3 Per-backend `_kernels.py` / `layers.py` split | Tasks D.3, D.4, D.5  |
| §3 Stdlib dataclass configs                   | Task D.2 (config.py + tests)  |
| §3 NumPy reference                            | Task D.2 (reference.py + signature tests) |
| §3 Equivalence-testing harness                | Task D.7 (placeholder directory) — full harness deferred to algorithm plan, per spec §9 "intentionally not in this migration" |
| §3 JAX = Flax NNX                             | Task D.4 (uses nnx.Module) |
| §3 Keras = keras.ops                          | Task D.5 (uses keras module, `KERAS_BACKEND=jax`) |
| §4 Four devcontainer flavors                  | Tasks E.1 (shared), A.2+E.2 (default), E.3, E.4, E.5 |
| §4 NVIDIA CUDA base, uv-installed Python      | Tasks E.3, E.4, E.5 (Dockerfile) |
| §5 [project] + [project.urls]                 | Tasks C.1                    |
| §5 [project.optional-dependencies]            | Task C.2                     |
| §5 [dependency-groups]                        | Task C.3                     |
| §5 Drop pydantic.mypy                         | Task C.4                     |
| §5 Update pytest addopts                      | Task C.5                     |
| §5 Drop legacy index                          | Task C.3                     |
| §6 build.yml per-backend matrix               | Task F.1                     |
| §6 docs.yml mike versioning                   | Task F.3                     |
| §6 publish.yml OIDC                           | Task F.2                     |
| §6 bump-version.yml strip                     | Task A.3 + Task F.4          |
| §6 dependabot.yml                             | Task F.5                     |
| §6 docs-smoke job (in build.yml)              | Task F.1                     |
| §7 LICENSE PolyForm                           | Task B.1                     |
| §7 NOTICE.md                                  | Task B.2                     |
| §7 README rewrite                             | Task B.3                     |
| §7 CONTRIBUTING rewrite                       | Task B.4                     |
| §7 CHANGELOG entry                            | Task H.4                     |
| §7 Docs site nav + plugins                    | Tasks G.1, G.2               |
| §7 Docs pages (index/guides/concepts/benchmarks/about) | Tasks G.3–G.7      |
| §8 Manual notebook execution                  | Task H.1 (`execute-benchmarks.sh`) + CONTRIBUTING.md update in Task B.4 |
| §9 Group A strip                              | Tasks A.1–A.5                |
| §9 Group B scaffold                           | Tasks D.1–D.7                |
| §9 Group C pyproject                          | Tasks C.1–C.6                |
| §9 Group D devcontainers                      | Tasks E.1–E.5                |
| §9 Group E workflows                          | Tasks F.1–F.5                |
| §9 Group F docs                               | Tasks G.1–G.8                |
| §9 Group G release tooling                    | Tasks H.1, H.2               |
| §9 "Intentionally not in this migration"      | acknowledged in plan introduction and in stub `NotImplementedError` messages |
| §10 Open items (licensing@airt.ai, BibTeX, etc.) | flagged as placeholders in NOTICE.md (Task B.2), citation.md (Task G.7), and README.md (Task B.3) — confirmation is part of the *release* step, not this migration |

All spec sections are covered.

## What this plan deliberately does NOT do

- Implement the monotonic algorithm itself (`monotonic_dense` / `monotonic_mlp` / each backend's kernels). Every layer raises `NotImplementedError` after this plan completes. The algorithm implementation is the subject of the next plan.
- Run real benchmarks. `docs/docs/benchmarks/00-overview.ipynb` is a one-cell placeholder.
- Publish anything to PyPI. The first tag-push is gated on the open items in spec §10 (commercial-license email, AIRT legal address wording, BibTeX confirmation, PyPI project registration).
