# Lean Devcontainer Flavor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land a fifth devcontainer flavor `proofs` whose single purpose is interactive review of the Lean 4 / mathlib4 formalization, with toolchain on PATH, mathlib cache warm, the `leanprover.lean4` VS Code extension installed, and pre-commit hooks working on commits from the container.

**Architecture:** A new `.devcontainer/proofs/` directory (mirroring the structure of `.devcontainer/default/`) plus one new shared script `.devcontainer/shared/install_lean.sh` that any flavor can opt into. No changes to the existing four flavors. The proofs flavor uses the same `mcr.microsoft.com/devcontainers/python:3.13` base image as `default` (Python is needed for pre-commit hooks); ML extras (`torch`, `jax`, `keras`) are deliberately omitted.

**Tech Stack:** elan (Lean version manager) → lake/lean toolchain pinned via `proofs/lean-toolchain` → mathlib4 build cache via `lake exe cache get`; Docker Compose; VS Code Dev Containers; `leanprover.lean4` extension.

**Spec:** [docs/superpowers/specs/2026-05-25-lean-devcontainer-design.md](../specs/2026-05-25-lean-devcontainer-design.md). Read §3 (devcontainer.json), §5 (proofs/setup.sh), §6 (install_lean.sh), §9 (expected timings) before starting Task 1.

**Branch:** Per the repo convention, this plan executes on `feat/lean-in-devcontainer` (already created off `origin/main`). All work commits to that branch; the final PR ships it.

**Dependency on PR #18:** The smoke build step in `install_lean.sh` (Step 4 of the spec's §6 script) invokes `lake build Mononet.Basic`, which requires `proofs/` to be present. PR #18 (the Lean proofs implementation, branch `feat/lean-proofs`) must be merged into `main` before this plan's Task 4 verification can succeed end-to-end. The script itself guards the smoke-build step with `[ -d "$REPO_ROOT/proofs" ]` so it remains usable on a branch where `proofs/` does not yet exist — but **the canonical PR-merge order is #18 first, then this PR**.

---

## File map

| Path | Created / Modified | Responsibility |
|---|---|---|
| `.devcontainer/shared/install_lean.sh` | Create | elan install + Lean toolchain install + mathlib cache fetch + smoke build; idempotent, reusable by any flavor |
| `.devcontainer/proofs/docker-compose.yml` | Create | Service definition for the `proofs` flavor; mirrors `default/docker-compose.yml` apart from the service name |
| `.devcontainer/proofs/setup.sh` | Create | Calls `install_common_tools.sh` + `install_lean.sh`, then `uv sync` with `lint`+`docs` groups (no ML extras) |
| `.devcontainer/proofs/devcontainer.json` | Create | VS Code picker entry, Lean extension, server PATH, minimal features set |
| `CONTRIBUTING.md` | Modify | Add one row to the devcontainer flavors table |
| `CLAUDE.md` | Modify | Add one row to the devcontainer flavors table |

---

## Pre-flight

- [ ] **Step 1: Confirm branch state**

Run:
```bash
git fetch origin
git rev-parse --abbrev-ref HEAD
git rev-list --left-right --count origin/main...HEAD
```
Expected: branch name is `feat/lean-in-devcontainer` (or you create it with `git checkout -b feat/lean-in-devcontainer origin/main`), and the rev-list count shows you are 0 or 1 commits ahead of `origin/main` (the existing spec commit on this branch is fine).

- [ ] **Step 2: Confirm reference files are accessible**

Run:
```bash
ls .devcontainer/default/devcontainer.json .devcontainer/default/docker-compose.yml .devcontainer/default/setup.sh .devcontainer/shared/install_common_tools.sh .devcontainer/shared/install_dependencies.sh
```
Expected: all five files exist. You will read from them as templates.

- [ ] **Step 3: Confirm PR #18's content is or is not on disk**

Run:
```bash
test -d proofs && echo "proofs/ on disk (PR #18 merged or staged)" || echo "proofs/ absent — install_lean.sh will skip smoke build until PR #18 merges"
```

Either outcome is OK. The script gracefully handles both states. The expected canonical flow is "PR #18 merges first, then this plan's PR opens against a `main` that has `proofs/` on it." Note the actual state in your status report.

---

## Task 1 — Shared install_lean.sh

**Files:**
- Create: `.devcontainer/shared/install_lean.sh`

This script is the heart of the feature. It installs `elan` (the Lean version manager), lets `elan` resolve the toolchain version from `proofs/lean-toolchain` lazily, fetches the mathlib4 build cache so VS Code's first session is fast, and runs a smoke build to surface toolchain problems before the user opens an editor.

The script is idempotent at every step — VS Code's `updateContentCommand` runs it on every devcontainer rebuild, so re-running on an already-set-up container must be a no-op-ish (<5 s total).

- [ ] **Step 1: Write `.devcontainer/shared/install_lean.sh`**

Exact contents:

```bash
#!/usr/bin/env bash
# Install elan (Lean version manager), let it install the toolchain
# pinned in proofs/lean-toolchain, fetch the mathlib4 build cache, and
# run a smoke build to catch toolchain breakage early.
#
# Idempotent: each step checks before reinstalling. Designed to be
# called from any devcontainer flavor's setup.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo -e "\033[36m=== Installing Lean toolchain ===\033[0m"

# 1. Install elan only when missing.
if command -v elan >/dev/null 2>&1 || [ -x "$HOME/.elan/bin/elan" ]; then
  echo "elan already installed: $($HOME/.elan/bin/elan --version 2>&1 | head -1)"
else
  echo -e "\033[32mInstalling elan...\033[0m"
  # --default-toolchain none keeps elan from installing a global toolchain;
  # elan reads proofs/lean-toolchain when invoked from that directory.
  curl -fsSL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
    | sh -s -- -y --default-toolchain none

  # Add ~/.elan/bin to interactive shells. Only on first install to
  # avoid duplicates in the rc files.
  echo 'export PATH="$HOME/.elan/bin:$PATH"' >> /root/.bashrc
  echo 'export PATH="$HOME/.elan/bin:$PATH"' >> /root/.zshrc 2>/dev/null || true
fi
export PATH="$HOME/.elan/bin:$PATH"
echo "elan: $(elan --version)"

# 2. The toolchain install + smoke build only make sense if proofs/ is
#    on disk (this branch may run before PR #18 lands).
if [ ! -d "$REPO_ROOT/proofs" ]; then
  echo -e "\033[1;33mproofs/ directory not present; skipping toolchain install and smoke build.\033[0m"
  echo -e "\033[1;33mThis is expected when this script runs before the Lean proofs PR has merged.\033[0m"
  echo -e "\033[32m✓ elan installed; Lean toolchain install deferred\033[0m"
  exit 0
fi

# 3. Trigger elan to install the toolchain pinned in proofs/lean-toolchain.
#    Any lake command from proofs/ does this lazily; we force it here
#    so the create-time error is clear if the pinned version is gone.
cd "$REPO_ROOT/proofs"
echo "lean: $(lean --version)"

# 4. Fetch the mathlib4 build cache (skip if .lake/build/lib is populated).
if [ -d ".lake/build/lib/Mathlib" ] && [ -n "$(find .lake/build/lib/Mathlib -name '*.olean' -print -quit)" ]; then
  echo "mathlib cache already populated; skipping lake exe cache get"
else
  echo -e "\033[32mFetching mathlib4 build cache...\033[0m"
  lake exe cache get
fi

# 5. Smoke build: confirm everything wired up. Mononet.Basic has no
#    dependencies on the more substantive proof modules, so it builds
#    quickly and validates elan / mathlib / our own files all line up.
echo -e "\033[32mSmoke build: lake build Mononet.Basic\033[0m"
lake build Mononet.Basic

echo -e "\033[32m✓ Lean toolchain ready\033[0m"
```

- [ ] **Step 2: Make the script executable**

Run:
```bash
chmod +x .devcontainer/shared/install_lean.sh
ls -l .devcontainer/shared/install_lean.sh
```
Expected: permission string includes `x` for owner (`-rwxr-xr-x`).

- [ ] **Step 3: Validate shell syntax**

Run:
```bash
bash -n .devcontainer/shared/install_lean.sh && echo "syntax OK"
```
Expected: prints `syntax OK` (`bash -n` checks syntax without executing).

- [ ] **Step 4: Smoke-run the script in the current container (idempotency check)**

Run:
```bash
bash .devcontainer/shared/install_lean.sh
```

Expected output covers two scenarios:

(a) If `proofs/` is on disk and `~/.elan/bin/elan` already exists (this is the typical state in the orchestrator's current container after Sub-project E work):
- Prints `elan already installed: elan X.Y.Z`
- Prints `lean: Lean (version Y.Y.Y, ...)`
- Prints `mathlib cache already populated; skipping lake exe cache get`
- Prints `Smoke build: lake build Mononet.Basic`
- `lake build Mononet.Basic` completes in <5 s
- Final line: `✓ Lean toolchain ready`

(b) If `proofs/` is NOT on disk (PR #18 not yet merged):
- Prints `elan already installed: ...` (or installs elan)
- Prints `proofs/ directory not present; skipping toolchain install and smoke build.`
- Final line: `✓ elan installed; Lean toolchain install deferred`

Either outcome is success. If you see any non-yellow `error:` lines or a non-zero exit, that's a real failure — investigate before continuing.

- [ ] **Step 5: Commit**

```bash
git add .devcontainer/shared/install_lean.sh
git commit -m "$(cat <<'EOF'
feat(devcontainer): add shared install_lean.sh

Installs elan (Lean version manager), lets elan resolve the toolchain
from proofs/lean-toolchain lazily, fetches the mathlib4 build cache,
and runs a smoke build (lake build Mononet.Basic) to surface toolchain
breakage before the user opens an editor.

Idempotent at every step. Guards the toolchain steps with a
proofs/-exists check so the script remains usable on branches where
the Lean proofs have not yet landed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Expected: commit lands; pre-commit hooks pass.

---

## Task 2 — Proofs flavor docker-compose + setup.sh

**Files:**
- Create: `.devcontainer/proofs/docker-compose.yml`
- Create: `.devcontainer/proofs/setup.sh`

These two together with `devcontainer.json` (Task 3) form the flavor. We land docker-compose.yml and setup.sh first; without `devcontainer.json` the VS Code picker won't list the flavor yet, so the intermediate state is invisible rather than broken.

- [ ] **Step 1: Read the reference docker-compose.yml**

Run:
```bash
cat .devcontainer/default/docker-compose.yml
```
Expected: a short YAML file (~15 lines) defining a single service named `python-3.13-mononet`. Note the structure (`services:` key, the service's `image:` or `build:` field, `volumes:`, `working_dir:`, `command:`).

- [ ] **Step 2: Write `.devcontainer/proofs/docker-compose.yml`**

The compose file mirrors `default/docker-compose.yml` exactly, with one change: the service name becomes `python-3.13-mononet-proofs` (to be unique across flavors). The image, volumes, working_dir, command — all identical.

Concretely, here is the exact content to write (the implementer should copy `default/docker-compose.yml` and edit only the service-name line). Read the actual `default/docker-compose.yml` to get the current verbatim content, then create the new file with this one change:

- Service key changes from `python-3.13-mononet:` to `python-3.13-mononet-proofs:`
- Everything else (image, volumes, working_dir, command) is byte-identical

If `default/docker-compose.yml` is structured as:

```yaml
services:
  python-3.13-mononet:
    image: mcr.microsoft.com/devcontainers/python:3.13
    volumes:
      - ../../:/workspaces/mononet:cached
    working_dir: /workspaces/mononet
    command: sleep infinity
```

then `.devcontainer/proofs/docker-compose.yml` is the same with `python-3.13-mononet` → `python-3.13-mononet-proofs`. Use the actual current content from `default/docker-compose.yml` as the source of truth; do not hardcode the structure shown above if the actual file differs.

- [ ] **Step 3: Validate the YAML syntax**

Run:
```bash
python3 -c "import yaml; yaml.safe_load(open('.devcontainer/proofs/docker-compose.yml'))" && echo "YAML OK"
```
Expected: prints `YAML OK`. If `python3-yaml` isn't installed, `uv run python -c "..."` works equivalently.

- [ ] **Step 4: Write `.devcontainer/proofs/setup.sh`**

Exact contents:

```bash
#!/usr/bin/env bash
# Proofs (Lean) devcontainer: install Lean toolchain + mathlib cache,
# plus the lint/docs uv groups so pre-commit hooks run on commits made
# from this container.
#
# Deliberately NOT installing any of mononet's ML extras (torch, jax,
# keras) — the proofs flavor is for Lean review only.
set -euo pipefail

cd /workspaces/mononet

bash .devcontainer/shared/install_common_tools.sh
bash .devcontainer/shared/install_lean.sh

# No --extra flag: skip torch/jax/keras. Only the dev-time groups
# pre-commit needs to run cleanly.
echo ">>> uv sync --only-group lint --only-group docs"
uv sync --only-group lint --only-group docs
```

- [ ] **Step 5: Make setup.sh executable**

Run:
```bash
chmod +x .devcontainer/proofs/setup.sh
ls -l .devcontainer/proofs/setup.sh
```
Expected: `-rwxr-xr-x ... .devcontainer/proofs/setup.sh`.

- [ ] **Step 6: Validate shell syntax**

Run:
```bash
bash -n .devcontainer/proofs/setup.sh && echo "syntax OK"
```
Expected: `syntax OK`.

- [ ] **Step 7: Commit**

```bash
git add .devcontainer/proofs/docker-compose.yml .devcontainer/proofs/setup.sh
git commit -m "$(cat <<'EOF'
feat(devcontainer): scaffold proofs flavor (docker-compose + setup.sh)

Adds .devcontainer/proofs/docker-compose.yml (service named
python-3.13-mononet-proofs, mirrors the default flavor's compose file
otherwise) and setup.sh that calls install_common_tools.sh and the new
install_lean.sh, then uv sync --only-group lint --only-group docs.

No ML extras (torch, jax, keras) — this flavor is for Lean proof
review only. Pre-commit parity is preserved by installing lint+docs
groups so commits from this container pass the same hooks as commits
from the default flavor.

Devcontainer.json lands in a follow-up commit; without it, the VS Code
picker will not list this flavor yet, so the intermediate state is
invisible rather than broken.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3 — Proofs flavor devcontainer.json

**Files:**
- Create: `.devcontainer/proofs/devcontainer.json`

Once this file lands, the VS Code "Reopen in Container" picker shows the new flavor.

- [ ] **Step 1: Read the reference devcontainer.json**

Run:
```bash
cat .devcontainer/default/devcontainer.json
```
Expected: a JSON file (~70 lines) with `name`, `dockerComposeFile`, `service`, `workspaceFolder`, `containerEnv`, `postCreateCommand`, `remoteUser`, `features`, `updateContentCommand`, `initializeCommand`, `mounts`, and `customizations.vscode.extensions` (Python + Jupyter + Claude Code).

- [ ] **Step 2: Write `.devcontainer/proofs/devcontainer.json`**

Use the following structure. The key differences from `default/devcontainer.json` are:

1. `name` changes to `"python-3.13 — proofs (Lean)"`
2. `service` changes to `"python-3.13-mononet-proofs"`
3. `updateContentCommand` changes to `"bash .devcontainer/proofs/setup.sh"`
4. `containerEnv.PATH` is added explicitly (so even non-interactive shells see `/root/.elan/bin`)
5. The `features` array drops `apt-packages`, `docker-in-docker`, and `node` — they're not needed for Lean review
6. The `customizations.vscode.extensions` array is replaced with the minimal set: just the Lean extension and Claude Code
7. The `customizations.vscode.settings` drops Python-specific entries and adds two Lean-friendly ones

Exact file content:

```json
{
    "name": "python-3.13 — proofs (Lean)",
    "dockerComposeFile": [
        "./docker-compose.yml"
    ],
    "service": "python-3.13-mononet-proofs",
    "shutdownAction": "stopCompose",
    "workspaceFolder": "/workspaces/mononet",
    "remoteEnv": {},
    "containerEnv": {
        "CLAUDE_CONFIG_DIR": "/root/.claude",
        "PATH": "/root/.elan/bin:/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
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
        "ghcr.io/devcontainers/features/github-cli:1": {}
    },
    "updateContentCommand": "bash .devcontainer/proofs/setup.sh",
    "initializeCommand": "bash .devcontainer/shared/host-init.sh",
    "mounts": [
        "source=claude-code-config-${devcontainerId},target=/root/.claude,type=volume",
        "source=${localEnv:HOME}/.config/mononet-devcontainer,target=/var/run/devcontainer-host-secrets,type=bind,readonly"
    ],
    "customizations": {
        "vscode": {
            "settings": {
                "editor.formatOnSave": true,
                "editor.rulers": [88],
                "terminal.integrated.defaultProfile.linux": "zsh",
                "terminal.integrated.profiles.linux": {
                    "zsh": { "path": "/bin/zsh" }
                },
                "lean4.serverEnv": {
                    "PATH": "/root/.elan/bin:/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
                }
            },
            "extensions": [
                "leanprover.lean4",
                "anthropic.claude-code"
            ]
        }
    }
}
```

- [ ] **Step 3: Validate the JSON syntax**

Run:
```bash
python3 -c "import json; json.load(open('.devcontainer/proofs/devcontainer.json'))" && echo "JSON OK"
```
Expected: prints `JSON OK`. If parsing fails, fix the JSON (most common cause: trailing comma).

- [ ] **Step 4: Sanity-check schema-critical keys**

Run:
```bash
python3 -c "
import json
d = json.load(open('.devcontainer/proofs/devcontainer.json'))
assert d['name'] == 'python-3.13 — proofs (Lean)'
assert d['service'] == 'python-3.13-mononet-proofs'
assert d['updateContentCommand'] == 'bash .devcontainer/proofs/setup.sh'
assert 'leanprover.lean4' in d['customizations']['vscode']['extensions']
assert 'anthropic.claude-code' in d['customizations']['vscode']['extensions']
assert '/root/.elan/bin' in d['containerEnv']['PATH']
print('schema OK')
"
```
Expected: prints `schema OK`.

- [ ] **Step 5: Commit**

```bash
git add .devcontainer/proofs/devcontainer.json
git commit -m "$(cat <<'EOF'
feat(devcontainer): add proofs/devcontainer.json (Lean flavor entry)

Lands the VS Code picker entry "python-3.13 — proofs (Lean)" with the
leanprover.lean4 + anthropic.claude-code extensions. Drops Python /
Jupyter extensions from the default flavor (not needed for Lean
review). Adds /root/.elan/bin to both containerEnv.PATH and the
lean4.serverEnv.PATH so the Lean extension's spawned lake/lean
processes find the elan-managed binaries.

Devcontainer features are minimal: common-utils, git, github-cli.
Dropped: apt-packages, docker-in-docker, node (not needed for Lean).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4 — CLAUDE.md and CONTRIBUTING.md row additions

**Files:**
- Modify: `CLAUDE.md`
- Modify: `CONTRIBUTING.md`

Both files have a devcontainer-flavors table. We add a single row for the new `proofs` flavor.

- [ ] **Step 1: Read the existing flavors table in CLAUDE.md**

Run:
```bash
grep -n -A 7 "Devcontainer flavors" CLAUDE.md
```

You should see a Markdown table with 4 rows (`default`, `gpu-torch`, `gpu-jax`, `gpu-keras`). Note the exact column widths and formatting.

- [ ] **Step 2: Edit CLAUDE.md to add the new row**

Use the `Edit` tool. Find the existing row for `gpu-keras` and add a new row immediately after it:

```markdown
| `proofs` | Reviewing the Lean 4 / mathlib4 formalization under `proofs/` (CPU, no ML extras) |
```

The exact `old_string` for the Edit call should be the `gpu-keras` row plus one line of trailing context so the match is unique. The `new_string` should be the `gpu-keras` row, the new `proofs` row, then the same trailing context line.

- [ ] **Step 3: Confirm the table renders cleanly**

Run:
```bash
grep -n -A 10 "Devcontainer flavors" CLAUDE.md
```
Expected: the table now has 5 rows, with `proofs` immediately after `gpu-keras`.

- [ ] **Step 4: Read the existing flavors table in CONTRIBUTING.md**

Run:
```bash
grep -n -A 10 "ships four devcontainer flavors\|devcontainer flavor" CONTRIBUTING.md | head -30
```

The table format is the same as CLAUDE.md but the surrounding prose differs. Identify the table.

- [ ] **Step 5: Edit CONTRIBUTING.md — update the count and add the new row**

Two edits:

(a) Change `four devcontainer flavors` to `five devcontainer flavors` in the table's introduction.

(b) Add the new `proofs` row after the `gpu-keras` row, with the same description as in CLAUDE.md:

```markdown
| `proofs`        | Reviewing the Lean 4 / mathlib4 formalization under `proofs/` (CPU, no ML extras). |
```

Note: match the column widths and trailing punctuation of the existing rows in CONTRIBUTING.md (which may differ slightly from CLAUDE.md's table).

- [ ] **Step 6: Confirm both edits**

Run:
```bash
grep -n "proofs" CLAUDE.md | head -5
grep -n "proofs\|five devcontainer flavors" CONTRIBUTING.md | head -5
```
Expected: both files mention `proofs` in the table area; CONTRIBUTING.md mentions "five devcontainer flavors".

- [ ] **Step 7: Run pre-commit on the modified files**

Run:
```bash
uv run pre-commit run --files CLAUDE.md CONTRIBUTING.md
```
Expected: all hooks pass. Codespell sometimes flags markdown table content — if it does, the fix is to add a legit word to `.codespell-whitelist.txt` or rephrase.

- [ ] **Step 8: Commit**

```bash
git add CLAUDE.md CONTRIBUTING.md
git commit -m "$(cat <<'EOF'
docs: document the proofs devcontainer flavor

Add one row to the devcontainer flavors tables in CLAUDE.md and
CONTRIBUTING.md; update CONTRIBUTING.md's "four devcontainer flavors"
phrasing to "five".

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5 — End-to-end validation + open PR

**Files:** none modified

This task confirms the four committed changes work together, then opens the PR.

- [ ] **Step 1: Verify all expected files exist**

Run:
```bash
ls -l \
  .devcontainer/shared/install_lean.sh \
  .devcontainer/proofs/devcontainer.json \
  .devcontainer/proofs/docker-compose.yml \
  .devcontainer/proofs/setup.sh
```
Expected: all four files exist. `install_lean.sh` and `setup.sh` have the executable bit (`x` in the permission string).

- [ ] **Step 2: Verify all JSON / YAML / shell syntax**

Run:
```bash
python3 -c "import json; json.load(open('.devcontainer/proofs/devcontainer.json'))" && echo "JSON OK"
python3 -c "import yaml; yaml.safe_load(open('.devcontainer/proofs/docker-compose.yml'))" && echo "YAML OK"
bash -n .devcontainer/shared/install_lean.sh && echo "install_lean.sh syntax OK"
bash -n .devcontainer/proofs/setup.sh && echo "setup.sh syntax OK"
```
Expected: four success lines.

- [ ] **Step 3: Run install_lean.sh one more time for end-to-end idempotency**

Run:
```bash
bash .devcontainer/shared/install_lean.sh
```
Expected: completes in <10 s (everything idempotent). Final line is `✓ Lean toolchain ready` (if `proofs/` is on disk) or `✓ elan installed; Lean toolchain install deferred` (if not).

- [ ] **Step 4: Run full pre-commit suite**

Run:
```bash
uv run pre-commit run --all-files
```
Expected: every hook passes. (Detect-secrets may flag entries in the new JSON file as high-entropy; if it does, regenerate the baseline with `uv run detect-secrets scan --baseline .secrets.baseline` and commit that change in the same task.)

- [ ] **Step 5: Push the branch**

```bash
git push -u origin feat/lean-in-devcontainer
```

- [ ] **Step 6: Open the PR**

```bash
cat > /tmp/lean_devcontainer_pr.md << 'EOF'
## Summary

Adds a fifth devcontainer flavor `proofs` for interactive review of the Lean 4 / mathlib4 formalization under `proofs/`. The Lean install logic lives in a new shared script `.devcontainer/shared/install_lean.sh` that any flavor can opt into.

## Changes

- **`.devcontainer/shared/install_lean.sh`** — installs `elan` (Lean version manager), lets `elan` resolve the toolchain from `proofs/lean-toolchain` lazily, fetches the mathlib4 build cache, runs a smoke `lake build Mononet.Basic`. Idempotent; guarded for `proofs/`-not-on-disk so the script is usable on branches where the proofs PR has not yet landed.
- **`.devcontainer/proofs/{devcontainer.json,docker-compose.yml,setup.sh}`** — the new flavor. Uses the same `python:3.13` base image as `default`. ML extras (torch, jax, keras) are deliberately omitted. Pre-commit hooks run on commits from this container (`uv sync --only-group lint --only-group docs` in `setup.sh`).
- **VS Code picker entry**: `python-3.13 — proofs (Lean)` with `leanprover.lean4` + `anthropic.claude-code` extensions only.
- **`CLAUDE.md` and `CONTRIBUTING.md`** — one new row in each devcontainer-flavors table; CONTRIBUTING.md's "four flavors" phrasing updated to "five".

## Expected create-time

- First create: ~5–7 min (Python base image pull + mathlib cache fetch).
- Rebuild: ~10–20 s (every install step is idempotent).

## Test plan

- [x] Shell scripts pass `bash -n` syntax check.
- [x] `devcontainer.json` parses as JSON; `docker-compose.yml` parses as YAML.
- [x] Schema-critical JSON keys (`service`, `updateContentCommand`, extensions) verified.
- [x] `install_lean.sh` is idempotent (verified by re-running in the current container).
- [x] `uv run pre-commit run --all-files` passes.
- [ ] Manual: open VS Code → "Reopen in Container" picker shows the new `proofs` flavor.
- [ ] Manual: rebuild the devcontainer in `proofs` mode; the Lean Infoview opens on `proofs/Mononet/Lemma1Mono.lean` within ~30 s of opening the file.

## Sphinx integration

Out of scope for this PR. Bringing the `doc-gen4` HTML into the Sphinx docs site is a follow-up (its own spec/plan/PR).

## Spec / plan

- Spec: [docs/superpowers/specs/2026-05-25-lean-devcontainer-design.md](docs/superpowers/specs/2026-05-25-lean-devcontainer-design.md)
- Plan: [docs/superpowers/plans/2026-05-25-lean-devcontainer.md](docs/superpowers/plans/2026-05-25-lean-devcontainer.md)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF

gh pr create \
  --title "feat: add proofs devcontainer flavor (Lean 4 review environment)" \
  --body-file /tmp/lean_devcontainer_pr.md
```

Expected: prints the PR URL.

- [ ] **Step 7: Verify the PR body**

```bash
gh pr view --json title,body --jq '{title, body_preview: (.body[:300])}'
```

Expected: title and body match what was just submitted. If `/tmp/lean_devcontainer_pr.md` was clobbered by a stale prior file (this has happened in earlier PRs in this project), fix with `gh pr edit <number> --body-file /tmp/lean_devcontainer_pr.md`.

---

## Self-review summary

Spec coverage:

- **§3 devcontainer.json** → Task 3.
- **§4 docker-compose.yml** → Task 2 Step 2.
- **§5 setup.sh** → Task 2 Step 4.
- **§6 install_lean.sh** → Task 1.
- **§7 post-create.sh (unchanged)** → relies on the existing file; no task needed.
- **§8 VS Code picker entries** → indirectly tested in Task 5 Step 6 (manual checkbox).
- **§9 expected create-time profile** → mirrored in the PR description's "Expected create-time" section.
- **§10 documentation updates** → Task 4.
- **§11 open items** → carried through as PR description bullets (acknowledged but not actioned).
- **§12 intentionally NOT in spec** → mirrored by what tasks the plan does NOT include (no doc-gen4 build, no Sphinx integration, etc.).

Placeholder scan: none. Type / name consistency: `python-3.13-mononet-proofs` service name and `feat/lean-in-devcontainer` branch name are used identically in every task that references them.
