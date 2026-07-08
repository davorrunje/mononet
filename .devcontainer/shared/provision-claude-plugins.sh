#!/usr/bin/env bash
# Provision Claude Code plugins from the checked-in manifest
# (.devcontainer/claude-plugins.txt). Environment-agnostic and idempotent:
#
#   - devcontainer: called by shared/post-create.sh (writes into the
#     container-local ~/.claude volume; never touches the host).
#   - host:         run manually once — `bash .devcontainer/shared/provision-claude-plugins.sh`
#     (installs the same plugins at user scope in your host ~/.claude).
#
# Non-fatal by design: a missing `claude` CLI, no network, or an install
# failure must NEVER break the container build — we warn and continue.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${SCRIPT_DIR}/../claude-plugins.txt"

warn() { echo "provision-claude-plugins: $*" >&2; }

if ! command -v claude >/dev/null 2>&1; then
    warn "'claude' CLI not found on PATH; skipping plugin provisioning."
    exit 0
fi
if [ ! -f "${MANIFEST}" ]; then
    warn "manifest ${MANIFEST} not found; nothing to provision."
    exit 0
fi

installed="$(claude plugin list 2>/dev/null || true)"
markets="$(claude plugin marketplace list 2>/dev/null || true)"

while IFS= read -r line || [ -n "${line}" ]; do
    # strip comments / blanks
    line="${line%%#*}"
    [ -z "${line//[[:space:]]/}" ] && continue

    # two whitespace-separated fields: <source> <plugin@marketplace>
    src="$(printf '%s' "${line}" | awk '{print $1}')"
    plugin="$(printf '%s' "${line}" | awk '{print $2}')"
    [ -z "${src}" ] || [ -z "${plugin}" ] && { warn "malformed line: ${line}"; continue; }

    base="${plugin%@*}"      # plugin name without @marketplace
    market="${plugin#*@}"    # marketplace name

    # add marketplace if not already known (match by name or source)
    if ! printf '%s' "${markets}" | grep -qiF "${market}" \
        && ! printf '%s' "${markets}" | grep -qF "${src}"; then
        echo ">>> adding marketplace ${src}"
        claude plugin marketplace add "${src}" >/dev/null 2>&1 \
            || warn "failed to add marketplace ${src} (continuing)"
    fi

    # install plugin if not already installed
    if printf '%s' "${installed}" | grep -qiF "${base}"; then
        echo ">>> ${base} already installed; skipping"
    else
        echo ">>> installing ${plugin}"
        claude plugin install "${plugin}" --scope user >/dev/null 2>&1 \
            || warn "failed to install ${plugin} (continuing)"
    fi
done < "${MANIFEST}"

echo ">>> plugin provisioning done"
exit 0
