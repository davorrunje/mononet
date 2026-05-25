# Lean devcontainer flavor design

**Date:** 2026-05-25
**Author:** Davor Runje
**Status:** Draft (brainstorming output); pending user review.
**Parent spec:** [`2026-05-21-mononet-package-design.md`](2026-05-21-mononet-package-design.md) (devcontainer flavors)
**Related:** [`2026-05-22-E-lean-proofs-design.md`](2026-05-22-E-lean-proofs-design.md) (the proofs the new flavor exists to support)

## 1. Goals & non-goals

### Goals

- Add a fifth devcontainer flavor `proofs` whose single purpose is interactive review of the Lean 4 / mathlib4 formalization under `proofs/`.
- Make the VS Code `Lean 4` extension Just Work the moment the container starts — toolchain on PATH, mathlib build cache warm, Lean Infoview functional, no manual setup steps.
- Keep commit hygiene parity with other flavors: commits made from the `proofs` container pass the same pre-commit hooks as commits from `default`.
- Provide an extraction point in `.devcontainer/shared/` for the Lean install steps so any other flavor can opt in later with a one-line addition to its own `setup.sh`.

### Non-goals

- No GPU support for the `proofs` flavor. Lean is CPU-bound; theorem proving does not benefit from CUDA.
- No alternate Lean version pinning at the devcontainer level. The Lean version is read from `proofs/lean-toolchain` (currently `leanprover/lean4:v4.15.0`); bumping the version is a single edit to that file.
- No installation of Python ML extras (`torch`, `jax`, `keras`). The `proofs` flavor deliberately omits them; users running benchmarks already have GPU flavors.
- No `doc-gen4` HTML build at devcontainer create time. The script under `proofs/tools/doc-gen.sh` exists for that — running it on every create is wasteful (~10 min) and rarely needed.
- No mathlib cache *refresh* on rebuild beyond the initial fetch. If `proofs/lake-manifest.json` changes, the user manually re-runs `lake exe cache get` or rebuilds the devcontainer.
- No changes to the existing four flavors (`default`, `gpu-torch`, `gpu-jax`, `gpu-keras`). The shared install script is purely additive.

## 2. File layout

### New files

```
.devcontainer/proofs/
├── devcontainer.json     # VS Code picker entry: "python-3.13 — proofs (Lean)"
├── docker-compose.yml    # service: python-3.13-mononet-proofs
└── setup.sh              # calls shared install_common_tools + install_lean, then uv sync lint/docs

.devcontainer/shared/
└── install_lean.sh       # elan install + Lean toolchain via lean-toolchain pin + mathlib cache fetch + smoke build
```

### Files NOT modified

- `.devcontainer/default/` and `.devcontainer/gpu-*/` are untouched. The shared install scripts they depend on (`install_common_tools.sh`, `install_dependencies.sh`, `post-create.sh`, `host-init.sh`, `setup_path.sh`, `setup_json_env_vars.sh`) gain no new responsibilities.

## 3. `.devcontainer/proofs/devcontainer.json`

Mirrors the structure of `.devcontainer/default/devcontainer.json` with these specific deviations:

- **`name`**: `"python-3.13 — proofs (Lean)"` — clear in the VS Code "Reopen in Container" picker.
- **`service`**: `python-3.13-mononet-proofs` (must be unique across flavors; matches `docker-compose.yml`).
- **`features`**: drop `apt-packages`, `docker-in-docker`, `node`. Keep `common-utils`, `git`, `github-cli`. The omitted features are cost the `proofs` flavor doesn't need.
- **`extensions`**: minimal set — drop all Python/Jupyter extensions, keep only:
  ```json
  "extensions": [
      "leanprover.lean4",
      "anthropic.claude-code"
  ]
  ```
- **`settings`**: drop Python-specific settings (linting, testing, pylance, [python] block). Keep `editor.rulers`, `terminal.integrated.*`. Add a Lean-friendly default:
  ```json
  "editor.unicodeHighlight.allowedLocales": { "_os": true, "_vscode": true },
  "lean4.serverEnv": { "PATH": "/root/.elan/bin:/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" }
  ```
  (The `lean4.serverEnv` PATH ensures the Lean extension's spawned `lake serve` finds `elan`-managed binaries.)
- **`updateContentCommand`**: `bash .devcontainer/proofs/setup.sh`.
- **`postCreateCommand`**: `bash .devcontainer/shared/post-create.sh` (same as other flavors — installs pre-commit).
- **`mounts`**: identical to default.
- **`containerEnv`**: identical to default plus an explicit `PATH` line so the container sees `/root/.elan/bin` even from non-interactive shells:
  ```json
  "containerEnv": {
      "CLAUDE_CONFIG_DIR": "/root/.claude",
      "PATH": "/root/.elan/bin:/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  }
  ```

## 4. `.devcontainer/proofs/docker-compose.yml`

A single service definition matching the structure of `default/docker-compose.yml`. Uses the same `mcr.microsoft.com/devcontainers/python:3.13` base image so disk usage stays bounded by the layers already pulled for other flavors. The service name `python-3.13-mononet-proofs` is unique.

The compose file is ~15 lines, exactly mirroring `default/docker-compose.yml` apart from the service name.

## 5. `.devcontainer/proofs/setup.sh`

```bash
#!/usr/bin/env bash
# Proofs (Lean-only) devcontainer: install Lean toolchain + mathlib cache,
# and the lint/docs groups so pre-commit hooks can run on commits from
# this container.
set -euo pipefail

cd /workspaces/mononet

bash .devcontainer/shared/install_common_tools.sh
bash .devcontainer/shared/install_lean.sh

# No --extra flag: skip torch/jax/keras. Only the dev-time groups
# pre-commit needs to run cleanly.
echo ">>> uv sync --only-group lint --only-group docs"
uv sync --only-group lint --only-group docs
```

Three lines of substantive logic. The Lean-specific install lives entirely in `shared/install_lean.sh` so other flavors can adopt it.

## 6. `.devcontainer/shared/install_lean.sh`

```bash
#!/usr/bin/env bash
# Install elan (Lean version manager), let it install the toolchain
# pinned in proofs/lean-toolchain, fetch the mathlib4 build cache, and
# run a smoke build to catch toolchain breakage early.
#
# Idempotent: each step checks before reinstalling.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo -e "\033[36m=== Installing Lean toolchain ===\033[0m"

# 1. Install elan (the Lean version manager) only when missing.
if command -v elan >/dev/null 2>&1 || [ -x "$HOME/.elan/bin/elan" ]; then
  echo "elan already installed: $($HOME/.elan/bin/elan --version 2>&1 | head -1)"
else
  echo -e "\033[32mInstalling elan...\033[0m"
  # --default-toolchain none: don't install a global toolchain;
  # elan reads proofs/lean-toolchain when invoked from that directory.
  curl -fsSL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
    | sh -s -- -y --default-toolchain none

  # Add ~/.elan/bin to interactive shells (only on first install).
  echo 'export PATH="$HOME/.elan/bin:$PATH"' >> /root/.bashrc
  echo 'export PATH="$HOME/.elan/bin:$PATH"' >> /root/.zshrc 2>/dev/null || true
fi
export PATH="$HOME/.elan/bin:$PATH"
echo "elan: $(elan --version)"

# 2. Trigger elan to install the toolchain pinned in proofs/lean-toolchain.
#    Running any lake command from proofs/ does this lazily; we force it
#    here so the create-time error is clear if the pinned version is gone.
cd "$REPO_ROOT/proofs"
echo "lean: $(lean --version)"

# 3. Fetch the mathlib4 build cache (skip if .lake/build/lib already populated).
if [ -d ".lake/build/lib/Mathlib" ] && [ "$(find .lake/build/lib/Mathlib -name '*.olean' | head -1)" ]; then
  echo "mathlib cache already populated; skipping lake exe cache get"
else
  echo -e "\033[32mFetching mathlib4 build cache...\033[0m"
  lake exe cache get
fi

# 4. Smoke build: confirm everything wired up. Mononet.Basic has no
#    dependencies on the more substantive proof modules, so it builds
#    quickly and validates elan / mathlib / our own files all line up.
echo -e "\033[32mSmoke build: lake build Mononet.Basic\033[0m"
lake build Mononet.Basic

echo -e "\033[32m✓ Lean toolchain ready\033[0m"
```

The script is ~40 lines, with each step explained inline and the failure modes localized:

- elan install failure → network or curl-availability issue (fails at step 1).
- Toolchain version unavailable → step 2 surfaces a clear elan error before any mathlib download.
- Mathlib cache fetch failure → step 3 fails; user can re-run manually.
- Smoke build failure → step 4 surfaces a real Lean error before VS Code opens.

## 7. `.devcontainer/shared/post-create.sh` (unchanged)

The existing `post-create.sh` runs `uv run pre-commit install --install-hooks`. The `proofs` flavor reuses it as-is: by the time `post-create.sh` runs, `setup.sh` has already executed `uv sync --only-group lint --only-group docs`, so `uv run pre-commit install` finds its dependencies.

## 8. VS Code picker entries

After this lands, the "Reopen in Container" picker shows:

- `python-3.13` (default, CPU)
- `python-3.13 — GPU (PyTorch)`
- `python-3.13 — GPU (JAX)`
- `python-3.13 — GPU (Keras)`
- `python-3.13 — proofs (Lean)`

The "Lean" suffix is the disambiguator. The container is still Python-3.13-based; the name reflects what you'd use it for, not what's installed under the hood.

## 9. Expected create-time profile

| Phase | First create | Rebuild (no cache invalidation) |
|---|---|---|
| Pull base image `python:3.13` | 30–90 s | 0 s |
| Devcontainer features (`common-utils`, `git`, `github-cli`) | 30–60 s | 0 s (cached) |
| `install_common_tools.sh` (uv, gh, claude) | 30–45 s | <5 s (all idempotent) |
| `install_lean.sh` step 1 (elan install) | ~5 s | <1 s (skip) |
| `install_lean.sh` step 2 (toolchain install via elan) | ~60 s first time | <1 s (cached in `~/.elan/`) |
| `install_lean.sh` step 3 (mathlib cache fetch) | 60–120 s (5826 files, ~600 MB) | <5 s (skip if `.lake/build` populated) |
| `install_lean.sh` step 4 (smoke build) | <10 s | <5 s |
| `uv sync --only-group lint --only-group docs` | 60–90 s | <5 s |
| `post-create.sh` (pre-commit install) | 10–20 s | <5 s |
| **Total first create** | **~5–7 min** | **~10–20 s** |

## 10. Documentation updates

- [`CONTRIBUTING.md`](../../../CONTRIBUTING.md) — the devcontainer flavors table gets a new row:
  ```
  | `proofs`        | Reviewing the Lean 4 / mathlib4 formalization under `proofs/`.   |
  ```
  No changes to the existing four rows.
- [`CLAUDE.md`](../../../CLAUDE.md) — the "Devcontainer flavors" table mirrors the same addition.

Both edits are mechanical one-row inserts.

## 11. Open items

- **Should `install_lean.sh` cache the elan installer locally?** Currently it pulls from `raw.githubusercontent.com` every time the script runs (idempotent, but a network dependency). Caching to `/var/cache/elan-init.sh` would be safer for offline rebuilds but adds complexity. Defer until network flakiness becomes a problem.
- **Should pre-commit run the Sphinx docs hook for commits from this flavor?** The `docs` uv group is installed so it *can*. Running it adds ~2 s per commit. Currently the answer is yes (matches other flavors). Reconsider if Lean-heavy commits trigger unnecessary doc rebuilds.
- **Should we expose a `MONONET_LEAN_VERSION` override?** No — the version is in `proofs/lean-toolchain`, and overriding it would create a split-brain state. Skip.

## 12. What is intentionally NOT in this spec

- **No `doc-gen4` build at create time.** The CI workflow handles it; `proofs/tools/doc-gen.sh` is available for local builds. Doing it in `install_lean.sh` would add ~10 min to first create — not worth it.
- **No automatic re-fetch of mathlib cache when the manifest changes.** Detecting the manifest change would require diffing across container rebuilds, which is more complexity than it's worth. The user re-runs `lake exe cache get` manually after pulling new commits that touched the manifest.
- **No headless Lean server warmup at create time.** The Lean 4 VS Code extension warms its own server on first file open. Pre-warming a server we don't have a UI connection to would burn memory for no benefit.
- **No Lean-specific git hooks.** The Lean files are validated by `lake build` in CI; pre-commit hooks would either be (a) full `lake build` (too slow for commits) or (b) `grep`-based sorry checks (better as CI). The existing `.github/workflows/lean.yml` is the right home.
