# ADR-0011: rclone private mirror; SHA-256 authoritative fixity

- Status: accepted · Date: 2026-07-17 · Deciders: Davor Runje

## Context

The private mirror must support Google Drive now and other backends later, keep
credentials out of the repo, and verify integrity across backends whose native
hashes differ (Drive/S3 expose MD5).

## Decision drivers

- Backend-agnostic, single-binary, scriptable; light-dep posture.
- Strong, citation-grade fixity; backend-independent verification.
- Secrets never committed.

## Considered options

1. **rclone (subprocess) as the mirror engine + SHA-256 authoritative in the
   manifest, with rclone's native hash used only as a transport check.**
2. Standardize on MD5 everywhere (rclone-native).
3. git-annex/DVC-managed remotes.

## Decision

Option 1. rclone (Google Drive + ~70 backends; logical remote name in the
registry, credentials via untracked config / env-vars). SHA-256 in the manifest
is the single source of truth; re-hash locally after every transfer.

## Consequences

- Add/swap backends by config; strong fixity; backend-independent verification.
- rclone is a runtime binary dependency (subprocess), not a Python dep.

## Rejected alternatives

- **MD5 everywhere** — weaker; conflates transport hash with authoritative fixity.
- **git-annex/DVC remotes** — drags in their whole model/deps (see ADR-0009).

## Links

sub-spec 4 §2.3; sub-spec 3 §4; digest `dataset-tooling-mirror.md`.
