# gh CLI authentication inside devcontainers — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `gh` CLI inside any of the four mononet devcontainer flavors auto-authenticate using the developer's host `gh` OAuth token, while keeping the existing Codespaces `GITHUB_TOKEN` path intact.

**Architecture:** A new host-side shell script (`.devcontainer/shared/host-init.sh`), invoked from each `devcontainer.json` via `initializeCommand`, extracts the host's `gh` OAuth token (if any) into `${HOME}/.config/mononet-devcontainer/gh-token` with mode 0600. Each devcontainer bind-mounts that directory read-only at `/var/run/devcontainer-host-secrets/`. The shared `install_common_tools.sh` post-create logic gains a fallback that reads this file when `$GITHUB_TOKEN` is unset, then runs `gh auth login --with-token` exactly as today.

**Tech Stack:** Bash, devcontainer.json (VS Code Dev Containers spec), `gh` CLI.

**Spec:** [docs/superpowers/specs/2026-05-22-gh-auth-in-devcontainers-design.md](../specs/2026-05-22-gh-auth-in-devcontainers-design.md)

---

## File Structure

**Created:**
- `.devcontainer/shared/host-init.sh` — host-side token extractor invoked via `initializeCommand`.

**Modified:**
- `.devcontainer/shared/install_common_tools.sh` — replace the existing gh-auth block (lines 36-42) with a 2-source fallback.
- `.devcontainer/default/devcontainer.json` — add `initializeCommand` key, append a second entry to `mounts`.
- `.devcontainer/gpu-jax/devcontainer.json` — same edits as default.
- `.devcontainer/gpu-keras/devcontainer.json` — same edits as default.
- `.devcontainer/gpu-torch/devcontainer.json` — same edits as default.

**Untouched:**
- `.devcontainer/shared/post-create.sh` — only handles pre-commit hooks, unrelated.
- Per-flavor `setup.sh` / `install_dependencies.sh` — only orchestrate `install_common_tools.sh`, no change needed.
- `.gitignore` — host token lives outside the repo (`${HOME}/.config/mononet-devcontainer/`).

**Note on TDD:** The spec explicitly puts automated tests out of scope (host script is ~15 lines, container fallback is a small if/elif). This plan substitutes lightweight in-place verification (shell `--dry-run` checks, JSON-validity checks, manual rebuild) for `pytest`-style TDD. Each task still has a "verify" step before commit.

---

### Task 1: Create the host-side token extractor

**Files:**
- Create: `.devcontainer/shared/host-init.sh`

- [ ] **Step 1: Create the script**

Write `.devcontainer/shared/host-init.sh` with this exact content:

```bash
#!/usr/bin/env bash
# Runs on the host (not the container) via devcontainer.json
# `initializeCommand`. Extracts the host's gh CLI OAuth token (if any)
# into a file that each devcontainer bind-mounts read-only.
#
# Failure here MUST NOT block container start.
set -u

SECRETS_DIR="${HOME}/.config/mononet-devcontainer"
TOKEN_FILE="${SECRETS_DIR}/gh-token"

mkdir -p "${SECRETS_DIR}"
chmod 700 "${SECRETS_DIR}"

if command -v gh >/dev/null 2>&1 && token="$(gh auth token 2>/dev/null)" && [ -n "${token}" ]; then
    printf '%s' "${token}" > "${TOKEN_FILE}"
    chmod 600 "${TOKEN_FILE}"
else
    rm -f "${TOKEN_FILE}"
fi

exit 0
```

- [ ] **Step 2: Make it executable**

Run:
```bash
chmod +x .devcontainer/shared/host-init.sh
```

- [ ] **Step 3: Syntax-check the script**

Run:
```bash
bash -n .devcontainer/shared/host-init.sh && echo OK
```
Expected output: `OK`

- [ ] **Step 4: Smoke-test the negative path (no `gh` available)**

Run with `gh` masked off `PATH`:
```bash
PATH=/usr/bin:/bin bash .devcontainer/shared/host-init.sh
ls -la "${HOME}/.config/mononet-devcontainer/"
```
Expected: the script exits 0, the directory exists with mode `drwx------`, and `gh-token` is **absent** (because `gh` was not found and any previous file is removed).

- [ ] **Step 5: Smoke-test the positive path (stubbed `gh`)**

Create a temporary stub to simulate an authed host `gh`:
```bash
STUBDIR=$(mktemp -d)
cat > "${STUBDIR}/gh" <<'STUB'
#!/usr/bin/env bash
[ "$1 $2" = "auth token" ] && echo "gho_fake_token_for_smoke_test"
STUB
chmod +x "${STUBDIR}/gh"
PATH="${STUBDIR}:${PATH}" bash .devcontainer/shared/host-init.sh
ls -la "${HOME}/.config/mononet-devcontainer/gh-token"
cat "${HOME}/.config/mononet-devcontainer/gh-token"; echo
rm -rf "${STUBDIR}"
```
Expected: file `gh-token` exists with mode `-rw-------`, and its content is exactly `gho_fake_token_for_smoke_test` (no trailing newline).

- [ ] **Step 6: Clean up smoke-test artifacts**

Remove the stub token so a real token (from the actual host gh) can be written cleanly later:
```bash
rm -f "${HOME}/.config/mononet-devcontainer/gh-token"
```

- [ ] **Step 7: Commit**

```bash
git add .devcontainer/shared/host-init.sh
git commit -m "feat(devcontainer): add host-init.sh to forward host gh token"
```

---

### Task 2: Update `install_common_tools.sh` with host-token fallback

**Files:**
- Modify: `.devcontainer/shared/install_common_tools.sh:36-42`

- [ ] **Step 1: Replace the gh-auth block**

In `.devcontainer/shared/install_common_tools.sh`, replace the existing block:

```bash
# Authenticate GitHub CLI (installed via devcontainer feature)
echo -e "\033[32mAuthenticating GitHub CLI (if token provided)...\033[0m"
if [ -n "${GITHUB_TOKEN:-}" ]; then
  echo "$GITHUB_TOKEN" | gh auth login --with-token && echo -e "\033[32mGitHub CLI authenticated.\033[0m" || echo -e "\033[1;33mWARNING: GitHub CLI authentication failed.\033[0m"
else
  echo -e "\033[1;33mWARNING: GITHUB_TOKEN not set; gh CLI installed but not authenticated.\033[0m"
fi
```

with:

```bash
# Authenticate GitHub CLI. Two token sources, in order:
#   1. $GITHUB_TOKEN (Codespaces auto-injects this).
#   2. /var/run/devcontainer-host-secrets/gh-token (forwarded from the
#      host by .devcontainer/shared/host-init.sh via initializeCommand).
echo -e "\033[32mAuthenticating GitHub CLI (if a token is available)...\033[0m"

HOST_TOKEN_FILE="/var/run/devcontainer-host-secrets/gh-token"
gh_token=""

if [ -n "${GITHUB_TOKEN:-}" ]; then
  gh_token="${GITHUB_TOKEN}"
elif [ -s "${HOST_TOKEN_FILE}" ]; then
  gh_token="$(cat "${HOST_TOKEN_FILE}")"
fi

if [ -n "${gh_token}" ]; then
  if printf '%s' "${gh_token}" | gh auth login --with-token; then
    echo -e "\033[32mGitHub CLI authenticated.\033[0m"
  else
    echo -e "\033[1;33mWARNING: GitHub CLI authentication failed.\033[0m"
  fi
else
  echo -e "\033[1;33mWARNING: No GitHub token available (neither \$GITHUB_TOKEN nor host-forwarded token); gh CLI is unauthenticated.\033[0m"
fi
unset gh_token HOST_TOKEN_FILE
```

- [ ] **Step 2: Syntax-check the modified script**

Run:
```bash
bash -n .devcontainer/shared/install_common_tools.sh && echo OK
```
Expected output: `OK`

- [ ] **Step 3: Simulate the "no token" branch**

In a fresh shell, with `$GITHUB_TOKEN` unset and no host-token file present, source just the new block to confirm it warns and does not call `gh auth login`. Quickest way: run the script end-to-end, capturing the gh-auth section's output. From within the devcontainer (where gh is installed but not authed):

```bash
unset GITHUB_TOKEN
sudo rm -f /var/run/devcontainer-host-secrets/gh-token 2>/dev/null || true
# Re-run only the gh-auth fragment by sourcing a temp copy:
sed -n '/^# Authenticate GitHub CLI\./,/^unset gh_token HOST_TOKEN_FILE$/p' \
  .devcontainer/shared/install_common_tools.sh > /tmp/gh-auth-fragment.sh
bash /tmp/gh-auth-fragment.sh
rm /tmp/gh-auth-fragment.sh
```
Expected output: the yellow warning `WARNING: No GitHub token available...`. No `gh auth login` invocation.

- [ ] **Step 4: Simulate the "GITHUB_TOKEN set, bogus value" branch**

Confirm the failure-handling path emits the failure warning (the token is invalid, so `gh auth login --with-token` returns non-zero):

```bash
GITHUB_TOKEN="gho_not_a_real_token" bash /tmp/test-gh-auth.sh 2>&1 | tail -5
```

Reuse the fragment from Step 3 if `/tmp/gh-auth-fragment.sh` is still available; otherwise re-extract via the same `sed`. Expected: a red "Authenticating GitHub CLI..." line, then output from `gh` complaining about the bad token, then the yellow `WARNING: GitHub CLI authentication failed.` line. Critically: the **script returns 0** (because `if printf ... | gh ...` swallows the non-zero exit), so `set -euo pipefail` in the parent does not abort container creation.

Verify:
```bash
echo "exit was: $?"
```
Expected: `exit was: 0`.

- [ ] **Step 5: Commit**

```bash
git add .devcontainer/shared/install_common_tools.sh
git commit -m "feat(devcontainer): fall back to host-forwarded gh token"
```

---

### Task 3: Wire the four `devcontainer.json` files

All four flavors currently have an identical `mounts` array (`claude-code-config-${devcontainerId}` only) and no `initializeCommand`. The edits are byte-identical across all four files.

**Files:**
- Modify: `.devcontainer/default/devcontainer.json`
- Modify: `.devcontainer/gpu-jax/devcontainer.json`
- Modify: `.devcontainer/gpu-keras/devcontainer.json`
- Modify: `.devcontainer/gpu-torch/devcontainer.json`

- [ ] **Step 1: Edit `.devcontainer/default/devcontainer.json`**

Find this exact block:
```jsonc
    "updateContentCommand": "bash .devcontainer/default/setup.sh",
    "mounts": [
        "source=claude-code-config-${devcontainerId},target=/root/.claude,type=volume"
    ],
```

Replace with:
```jsonc
    "updateContentCommand": "bash .devcontainer/default/setup.sh",
    "initializeCommand": "bash .devcontainer/shared/host-init.sh",
    "mounts": [
        "source=claude-code-config-${devcontainerId},target=/root/.claude,type=volume",
        "source=${localEnv:HOME}/.config/mononet-devcontainer,target=/var/run/devcontainer-host-secrets,type=bind,readonly"
    ],
```

- [ ] **Step 2: Edit `.devcontainer/gpu-jax/devcontainer.json`**

Find this exact block:
```jsonc
    "updateContentCommand": "bash .devcontainer/gpu-jax/setup.sh",
    "mounts": [
        "source=claude-code-config-${devcontainerId},target=/root/.claude,type=volume"
    ],
```

Replace with:
```jsonc
    "updateContentCommand": "bash .devcontainer/gpu-jax/setup.sh",
    "initializeCommand": "bash .devcontainer/shared/host-init.sh",
    "mounts": [
        "source=claude-code-config-${devcontainerId},target=/root/.claude,type=volume",
        "source=${localEnv:HOME}/.config/mononet-devcontainer,target=/var/run/devcontainer-host-secrets,type=bind,readonly"
    ],
```

- [ ] **Step 3: Edit `.devcontainer/gpu-keras/devcontainer.json`**

Find this exact block:
```jsonc
    "updateContentCommand": "bash .devcontainer/gpu-keras/setup.sh",
    "mounts": [
        "source=claude-code-config-${devcontainerId},target=/root/.claude,type=volume"
    ],
```

Replace with:
```jsonc
    "updateContentCommand": "bash .devcontainer/gpu-keras/setup.sh",
    "initializeCommand": "bash .devcontainer/shared/host-init.sh",
    "mounts": [
        "source=claude-code-config-${devcontainerId},target=/root/.claude,type=volume",
        "source=${localEnv:HOME}/.config/mononet-devcontainer,target=/var/run/devcontainer-host-secrets,type=bind,readonly"
    ],
```

- [ ] **Step 4: Edit `.devcontainer/gpu-torch/devcontainer.json`**

Find this exact block:
```jsonc
    "updateContentCommand": "bash .devcontainer/gpu-torch/setup.sh",
    "mounts": [
        "source=claude-code-config-${devcontainerId},target=/root/.claude,type=volume"
    ],
```

Replace with:
```jsonc
    "updateContentCommand": "bash .devcontainer/gpu-torch/setup.sh",
    "initializeCommand": "bash .devcontainer/shared/host-init.sh",
    "mounts": [
        "source=claude-code-config-${devcontainerId},target=/root/.claude,type=volume",
        "source=${localEnv:HOME}/.config/mononet-devcontainer,target=/var/run/devcontainer-host-secrets,type=bind,readonly"
    ],
```

- [ ] **Step 5: Validate all four JSON files**

devcontainer.json files use JSON-with-comments (jsonc), so a strict parser will choke on `//` comments. There aren't any in these files today, so plain `json.tool` will work. Run:

```bash
for f in .devcontainer/{default,gpu-jax,gpu-keras,gpu-torch}/devcontainer.json; do
  echo "=== $f ==="
  python3 -m json.tool "$f" > /dev/null && echo "  valid JSON"
done
```
Expected: four `valid JSON` lines, no parse errors.

- [ ] **Step 6: Verify `initializeCommand` and the new mount are present in all four**

```bash
for f in .devcontainer/{default,gpu-jax,gpu-keras,gpu-torch}/devcontainer.json; do
  echo "=== $f ==="
  grep -c '"initializeCommand": "bash \.devcontainer/shared/host-init\.sh"' "$f"
  grep -c 'devcontainer-host-secrets,type=bind,readonly' "$f"
done
```
Expected: for each file, two `1` lines (one match per pattern).

- [ ] **Step 7: Commit**

```bash
git add .devcontainer/default/devcontainer.json \
        .devcontainer/gpu-jax/devcontainer.json \
        .devcontainer/gpu-keras/devcontainer.json \
        .devcontainer/gpu-torch/devcontainer.json
git commit -m "feat(devcontainer): wire host token forwarding into all flavors"
```

---

### Task 4: Manual end-to-end verification (devcontainer rebuild)

This task can only be run on the developer's local machine (Docker Desktop / VS Code Dev Containers), not from inside the current container session. It's the canonical proof that the design works.

**Files:** none (verification only).

- [ ] **Step 1: Ensure host gh is authenticated**

On the host shell (outside the container):
```bash
gh auth status
```
Expected: `✓ Logged in to github.com account <username>`.

- [ ] **Step 2: Rebuild the default devcontainer**

In VS Code: `Dev Containers: Rebuild Container`. Or via CLI:
```bash
devcontainer up --workspace-folder . --config .devcontainer/default/devcontainer.json --remove-existing-container
```

- [ ] **Step 3: Confirm the host token file was written**

On the host:
```bash
ls -la ~/.config/mononet-devcontainer/
```
Expected: directory mode `drwx------`, file `gh-token` mode `-rw-------`, non-empty.

- [ ] **Step 4: Confirm gh is authed inside the container**

Open a terminal in the rebuilt devcontainer:
```bash
gh auth status
gh api user --jq .login
```
Expected: `Logged in to github.com`, and `gh api user` prints the host's GitHub username.

- [ ] **Step 5: Confirm Codespaces path still works (optional)**

If you have a Codespace for this repo: open it, run `gh auth status` — expect it to be authed via the auto-injected `GITHUB_TOKEN`, with no host file present. (The bind-mount source `${localEnv:HOME}/.config/mononet-devcontainer` will not exist on a Codespaces host; depending on Docker version this may either fail soft or fail the mount. If the mount fails on Codespaces, follow up by gating the mount on host file existence — see "Follow-ups" below.)

- [ ] **Step 6: Open a PR**

Once steps 1-4 pass on at least the `default` flavor:

```bash
git push -u origin chore/devcontainer-gh-auth
gh pr create --title "Auto-authenticate gh CLI inside devcontainers" \
  --body "$(cat <<'EOF'
## Summary
- Forward the host's gh OAuth token into the devcontainer via a host-side `initializeCommand` and a read-only bind mount.
- Add a fallback in `install_common_tools.sh`: prefer `$GITHUB_TOKEN` (Codespaces), then the host-forwarded file.
- Apply to all four flavors (default, gpu-jax, gpu-keras, gpu-torch).

## Spec
docs/superpowers/specs/2026-05-22-gh-auth-in-devcontainers-design.md

## Test plan
- [x] `bash -n` and smoke tests on host-init.sh (stubbed gh, both branches)
- [x] `bash -n` on install_common_tools.sh + simulated no-token / bad-token branches
- [x] JSON validity for all four devcontainer.json files
- [x] Rebuild default devcontainer, `gh auth status` succeeds inside
- [ ] (Optional) rebuild one GPU flavor and confirm `gh auth status`
- [ ] (Optional) Codespaces check: GITHUB_TOKEN path still works

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Follow-ups (out of scope for this plan)

- If Step 5 of Task 4 reveals that Codespaces fails the bind-mount when the host-side `~/.config/mononet-devcontainer` doesn't exist: add a host-init wrapper that's safe to invoke on any host (it currently is — it always creates the directory). If the mount itself is the issue, gate it behind a feature flag or move to a Docker named volume seeded by the initializeCommand instead of a bind mount.
- Document the new behavior in CONTRIBUTING.md.
- Consider adding a small note to `host-init.sh` output (stderr) when gh isn't installed, to help newcomers understand why their container's gh isn't authed.

---

## Self-Review Checklist

**Spec coverage:**
- ✅ Host-side token extractor → Task 1
- ✅ Bind mount into container → Task 3 (mounts entry per flavor)
- ✅ `install_common_tools.sh` fallback logic, GITHUB_TOKEN preferred → Task 2
- ✅ All four flavors covered → Task 3
- ✅ Behavior matrix: local-authed, local-unauthed, Codespaces, rebuild, host re-auth → Task 1 + Task 2 smoke tests + Task 4 manual verification
- ✅ Edge cases (empty file `[ -s ]`, perm drift, mode 0600/0700) → Task 1 + Task 2 code
- ✅ Rollback steps → covered by the spec; no plan task required (git revert is sufficient)
- ✅ Testing approach (manual, automated out of scope) → reflected in plan

**Placeholder scan:** no TBD/TODO/"implement later" remain. All code blocks contain complete content. All exact commands shown.

**Type / name consistency:**
- `host-init.sh` path: `.devcontainer/shared/host-init.sh` — consistent across Tasks 1, 3, spec.
- Host secrets directory: `${HOME}/.config/mononet-devcontainer` — consistent.
- Container mount target: `/var/run/devcontainer-host-secrets` — consistent across Tasks 2, 3, spec.
- Token filename: `gh-token` — consistent.
- `initializeCommand` value: `bash .devcontainer/shared/host-init.sh` — consistent in all four edits.
