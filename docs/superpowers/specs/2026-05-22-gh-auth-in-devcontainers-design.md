# gh CLI authentication inside devcontainers

**Status:** Draft
**Date:** 2026-05-22
**Author:** Davor Runje (with Claude)

## Problem

`gh` CLI is installed inside the mononet devcontainers (via the
`ghcr.io/devcontainers/features/github-cli:1` feature) but is not authenticated
on container startup. The existing setup script
([.devcontainer/shared/install_common_tools.sh:36-42](../../../.devcontainer/shared/install_common_tools.sh))
tries `gh auth login --with-token` using `$GITHUB_TOKEN`, but that variable is
not forwarded from the host on local Docker, so the call is skipped and `gh`
remains unauthed.

Running `gh` commands (`gh pr create`, `gh issue list`, `gh api`, …) therefore
fails inside the container until the developer logs in manually, and the
manual login is lost on every container rebuild.

`git` HTTPS operations already work inside the container (VS Code Dev Containers
forwards git credentials), so the scope of this design is `gh` CLI only.

## Goals

1. After the next container creation, `gh auth status` succeeds inside the
   container with the same identity the developer has on their host.
2. No manual step on container start.
3. Token refreshes automatically every container creation — no stale tokens.
4. Works in **GitHub Codespaces** without regression (Codespaces auto-injects
   `GITHUB_TOKEN`; this design must not break that path).
5. Works uniformly across all four devcontainer flavors (`default`, `gpu-jax`,
   `gpu-keras`, `gpu-torch`).
6. Graceful no-op when the host does not have `gh` installed or authenticated
   — same behavior as today, plus a clear warning.

## Non-goals

- Fixing git HTTPS auth inside the container (already works).
- Supporting PAT-only flows or external secret managers (1Password, Doppler,
  etc.). A developer who prefers a PAT can still export `GITHUB_TOKEN` on the
  host and rely on the existing Codespaces path.
- Long-term persistence of the gh config inside the container (a named volume
  for `/root/.config/gh`). Re-extracting the host token on every container
  start is simpler and avoids stale-token drift.
- Windows host support. The `initializeCommand` shell snippet assumes a POSIX
  shell (macOS / Linux). WSL works because `gh` and bash are available there.

## Approach

Two halves, coordinated by `initializeCommand`:

### Host side

A new script `.devcontainer/shared/host-init.sh` runs on the host **before**
container creation, via the `initializeCommand` hook of each
`devcontainer.json`. Behavior:

- Create `${HOME}/.config/mononet-devcontainer/` (mode 0700).
- If `gh` is on `PATH` **and** `gh auth token` exits 0: write the token to
  `${HOME}/.config/mononet-devcontainer/gh-token` (mode 0600).
- Otherwise: remove any stale token file in that directory.
- Exit 0 in all cases — failure on the host must not block container start.

### Container side

Each `devcontainer.json` bind-mounts the host directory read-only into the
container at `/var/run/devcontainer-host-secrets/`.

`install_common_tools.sh` is updated so that gh authentication tries, in order:

1. `$GITHUB_TOKEN` (existing path; covers Codespaces and any developer who sets
   it explicitly).
2. `/var/run/devcontainer-host-secrets/gh-token` if the file exists and is
   non-empty.
3. Otherwise: log a warning, leave `gh` unauthed (same as today).

The token is piped into `gh auth login --with-token` exactly as today.

## File touchpoints

### New file: `.devcontainer/shared/host-init.sh`

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

### Modified: each `.devcontainer/<flavor>/devcontainer.json`

(`default`, `gpu-jax`, `gpu-keras`, `gpu-torch`)

Add an `initializeCommand` key and extend the existing `mounts` array:

```jsonc
"initializeCommand": "bash .devcontainer/shared/host-init.sh",
"mounts": [
  "source=claude-code-config-${devcontainerId},target=/root/.claude,type=volume",
  "source=${localEnv:HOME}/.config/mononet-devcontainer,target=/var/run/devcontainer-host-secrets,type=bind,readonly"
]
```

(GPU flavors must be inspected before editing — their existing `mounts` arrays
may differ from `default`. Only the second entry is added; existing mounts are
preserved.)

### Modified: `.devcontainer/shared/install_common_tools.sh`

Replace the existing block (lines 36-42) with:

```bash
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
```

## Behavior matrix

| Scenario | Host result | Container result |
|---|---|---|
| Local Docker, host `gh` authed | `host-init.sh` writes fresh token | `gh auth login --with-token` succeeds via the bind-mounted file |
| Local Docker, host `gh` missing or logged out | `host-init.sh` removes any stale file | No file, no `$GITHUB_TOKEN` → warning, gh stays unauthed |
| GitHub Codespaces | `initializeCommand` runs but host likely lacks `gh` → no file written | `$GITHUB_TOKEN` injected by Codespaces → existing path handles auth |
| Container rebuild | Fresh token extracted from host every time | Container is always in sync with current host auth |
| Host re-auths / OAuth token rotates | Detected on next rebuild | New token applied automatically |

## Edge cases

- **Bind-mount source missing first time:** `initializeCommand` runs before
  the mount is established, and `mkdir -p` ensures the directory exists before
  Docker tries to mount it.
- **`gh auth token` prints a non-token line on error:** Mitigated by checking
  exit code (`&&`) and discarding stderr. Output of `gh auth token` on success
  is the token alone (no trailing text).
- **Token file exists but is empty:** The container path uses `[ -s file ]`
  (non-empty) so an empty file is treated as "no token."
- **Permission drift:** `host-init.sh` re-applies `0700/0600` each run.
- **Multiple flavors running concurrently:** They write the same file with the
  same token. Last-writer-wins is safe.
- **Token revoked while container is running:** `gh` commands inside will
  start failing with 401. Resolved by rebuilding the container after the user
  re-auths on the host. Not detected automatically; out of scope.

## Security considerations

- Token is OAuth (`gho_*`), not a long-lived PAT — revocable from
  github.com/settings/applications.
- Token is stored only under the user's `$HOME` (mode 0600) on a personal
  machine, no different in sensitivity from the gh keyring entry itself or the
  forwarded git credential helper that already exists.
- The bind mount into the container is read-only (`readonly`), so a compromised
  container process cannot rewrite the host file.
- The token is never written into the repo or any tracked file. No `.gitignore`
  change is needed because the file lives outside the workspace.

## Testing / verification

Manual verification (documented in CONTRIBUTING or similar follow-up):

1. Confirm host: `gh auth status` shows `Logged in to github.com`.
2. Rebuild the default devcontainer ("Dev Containers: Rebuild Container").
3. Inside the new container: `gh auth status` shows `Logged in to github.com`
   with the same account name as on the host.
4. `gh api user` returns the expected username.

Negative case (optional): temporarily rename the host `gh` binary, rebuild,
confirm the container start succeeds and the warning is emitted in the
update-content log.

Automated tests are not in scope. The host script is ~15 lines and the
in-container fallback is a small if/elif. Failure modes are easy to inspect
manually.

## Rollback

Revert the spec's three edits:

1. Remove `initializeCommand` and the second `mounts` entry from each
   `devcontainer.json`.
2. Restore the original gh-auth block in `install_common_tools.sh`.
3. Delete `.devcontainer/shared/host-init.sh`.

Optionally clean up the host directory: `rm -rf ~/.config/mononet-devcontainer`.

## Open questions

None at design time. Implementation may surface flavor-specific quirks (e.g.
GPU flavors with custom mounts) — those will be resolved during the writing of
the implementation plan.
