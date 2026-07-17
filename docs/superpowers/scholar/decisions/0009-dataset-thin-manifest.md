# ADR-0009: `dataset` one skill + thin `datasets.yml` (reject DVC/DataLad/lakeFS)

- Status: accepted · Date: 2026-07-17 · Deciders: Davor Runje

## Context

Dataset management needs a registry + tiered retrieval + verification + a private
mirror, under a light-dependency posture (mononet avoids heavy deps / Rust-binary
conflicts). A tooling research pass compared the field.

## Decision drivers

- Light dependencies; scriptable; git-native; multi-source fallback (which
  off-the-shelf tools lack).
- One shared mental model (schema, tier, cache, resolution chain).

## Considered options

1. **Thin custom `datasets.yml` + `pooch` (Tier-B fetch) + `rclone` (mirror) +
   a small resolution core; one skill with verbs.**
2. DVC (`.dvc`/`dvc.yaml` + cache/remote).
3. DataLad (git-annex + provenance).
4. lakeFS / Quilt (server or S3-coupled).

## Decision

Option 1. Own the manifest; adopt only `pooch` (3 pure-Python deps) and `rclone`
(a Go binary invoked as a subprocess).

## Consequences

- Full control incl. the multi-source fallback; minimal deps; not in the
  published wheel.
- We maintain ~200 lines of resolution logic.

## Rejected alternatives

- **DVC** — ~40 direct deps; cache-bound manifest; no auto-fallback.
- **DataLad** — Python + git + git-annex triple dependency.
- **lakeFS/Quilt** — always-on server / S3-coupling; far too heavy.

## Links

sub-spec 3 §1–§2; digest `dataset-tooling-mirror.md`.
