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

exit 0
