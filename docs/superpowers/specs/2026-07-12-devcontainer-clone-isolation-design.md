# Devcontainer clone isolation

**Date:** 2026-07-12
**Author:** Davor Runje
**Status:** Partially implemented (`fix/devcontainer-clone-isolation`); session-bind hardening deferred.

## Problem

Running two devcontainers from two clones of this repo on the same machine
mixed up work: one container's `uv sync` changed the other's environment, and
git HEAD/branches bled across, stranding commits on the wrong branch. Root
cause: the devcontainer mounts are **machine-global** and thus shared by every
container regardless of clone.

The three shared mounts (identical across all five flavors):

| Mount | Kind | Problem |
|---|---|---|
| `mononet-venv` → `/workspaces/mononet/.venv` | named volume | one `.venv` shared machine-wide → `uv sync` in one clone mutates the other |
| `mononet-claude-config` → `/home/vscode/.claude` | named volume | shared Claude config/memory/todos across clones |
| `${HOME}/.config/mononet-devcontainer/claude-session` → `…/projects/-workspaces-mononet` | host bind | fixed path; `host-init.sh` re-points it every start (last-container-wins) |

## Done (this branch)

1. **Removed the `mononet-venv` volume.** It only ever existed as a bind-mount
   I/O speedup; uv does not require it. `.venv` now lives in the already-bind-
   mounted, gitignored workspace — clone-local by construction, and persistent
   across rebuilds. On this native-Linux host the I/O cost is negligible.
2. **`mononet-claude-config` → `mononet-claude-config-${devcontainerId}`.**
   `${devcontainerId}` is stable per rebuild and unique per workspace path, so
   each clone gets its own Claude-config volume. (`${devcontainerId}` is the
   devcontainer built-in intended for exactly this.)

Both take effect on the next container **rebuild** (cannot be validated from
inside a running container).

## Deferred: the `claude-session` host bind

`host-init.sh` (runs on the host via `initializeCommand`) symlinks a **fixed**
path `~/.config/mononet-devcontainer/claude-session` to the current clone's host
Claude session dir, doing `rm -rf` + re-link on every start. Two clones ⇒ the
last container to start wins, and both containers bind-mount that one path — so
this project's transcripts/memory can still cross concurrent clones.

**Proposed fix (needs rebuild testing; do not ship untested — the file warns
that a missing `SESSION_LINK` blocks container start):**
- Pass a per-clone key into `host-init.sh`, e.g. `initializeCommand`:
  `bash .devcontainer/shared/host-init.sh ${devcontainerId}`.
- In `host-init.sh`, use `SESSION_LINK=${SECRETS_DIR}/claude-session-${1}` (per
  clone) with the same safe symlink/fallback logic.
- Update each flavor's bind source to
  `…/claude-session-${devcontainerId}` so host and container agree on the key.
- **Open question:** confirm `${devcontainerId}` is substituted in
  `initializeCommand` (host-side, pre-create) on the target devcontainer CLI /
  editor; if not, derive the key from the host workspace path slug in
  `host-init.sh` and expose it to the mount another way.

**Acceptance:** rebuild two clones concurrently; verify each has its own
`.venv`, its own `~/.claude` volume, and its own project session dir, with no
cross-writes.

## Related

- Interacts with `2026-07-08-devcontainer-claude-plugin-session-sharing-design.md`
  (the host↔container session-sharing feature this must preserve for the
  single-clone case).
- The hardcoded `-workspaces-mononet` container-side slug is fine (container cwd
  is always `/workspaces/mononet`); only the host source needs per-clone keying.
