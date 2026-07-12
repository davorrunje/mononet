#!/usr/bin/env bash
# Runs on the host (not the container) via devcontainer.json
# `initializeCommand`. Extracts the host's gh CLI OAuth token (if any)
# into a file that each devcontainer bind-mounts read-only.
#
# Failure here MUST NOT block container start.
set -u

SECRETS_DIR="${HOME}/.config/mononet-devcontainer"
TOKEN_FILE="${SECRETS_DIR}/gh-token"

if ! mkdir -p "${SECRETS_DIR}"; then
    echo "WARNING: host-init.sh: could not create ${SECRETS_DIR}; skipping gh token forwarding." >&2
    exit 0
fi
chmod 700 "${SECRETS_DIR}" || true

if command -v gh >/dev/null 2>&1 && token="$(gh auth token 2>/dev/null)" && [ -n "${token}" ]; then
    (umask 077 && printf '%s' "${token}" > "${TOKEN_FILE}")
    chmod 600 "${TOKEN_FILE}"
else
    rm -f "${TOKEN_FILE}"
fi

# Expose this repo's HOST Claude session dir at a stable, host-user-agnostic
# path so the devcontainer can bind-mount it onto the container's session dir
# (container cwd /workspaces/mononet -> slug -workspaces-mononet). This shares
# *this project's* transcripts between host and container without sharing the
# rest of ~/.claude. Best-effort: must not block container start.
#
# The host session slug is the cwd with '/' -> '-'. host-init runs with cwd =
# the host workspace folder, so $PWD is that path. If Claude's slug algorithm
# ever diverges from slash->dash, sessions simply won't line up (no breakage).
# SESSION_LINK MUST exist (the container bind-mounts it); otherwise container
# start fails. Prefer a symlink to the real host session dir; on any failure
# fall back to a plain dir so the mount source always exists (sessions then
# persist there but aren't shared with a pre-existing host session dir).
CLAUDE_PROJECTS="${HOME}/.claude/projects"
# Per-clone key (devcontainer id, passed as $1 from initializeCommand). Keying
# the session dir by it means two clones on one machine get distinct session
# binds instead of overwriting a single shared "claude-session" link.
devcontainer_id="${1:-}"
SESSION_LINK="${SECRETS_DIR}/claude-session${devcontainer_id:+-${devcontainer_id}}"
host_slug="$(printf '%s' "${PWD}" | sed 's#/#-#g')"
rm -rf "${SESSION_LINK}" 2>/dev/null || true
if mkdir -p "${CLAUDE_PROJECTS}/${host_slug}" 2>/dev/null \
    && ln -sfn "${CLAUDE_PROJECTS}/${host_slug}" "${SESSION_LINK}" 2>/dev/null; then
    :  # symlinked to the real host session dir -> sessions shared
else
    echo "WARNING: host-init.sh: could not link host Claude session dir; using a standalone dir (sessions not shared)." >&2
    mkdir -p "${SESSION_LINK}" || true
fi

exit 0
