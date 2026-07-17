# scholar sub-spec 3 — Dataset capability (`dataset`)

**Date:** 2026-07-17
**Author:** Davor Runje
**Status:** Brainstorming output; sub-spec of the `scholar` meta-spec, pending implementation plan.

> Sub-spec of [2026-07-17-scholar-plugin-design.md](2026-07-17-scholar-plugin-design.md).
> Builds on the asset substrate of
> [sub-spec 4](2026-07-17-scholar-4-substrate-and-experiment-contract-design.md) §2
> and feeds the experiment-backend contract (§3). Governed by the ⚑ agency (§2.1)
> and Understanding (§2.2) principles.
> Grounding: [dataset-management-standards](../scholar/references/dataset-management-standards.md),
> [dataset-tooling-mirror](../scholar/references/dataset-tooling-mirror.md).

## 1. The `dataset` skill

One skill, verbs `init | register | fetch | verify | mirror | audit`. It is
**upstream of the experiment backend**: runs declare the datasets they need; the
skill materializes + verifies them; the `id + version + sha256` fingerprint flows
into the backend's provenance stamp (sub-spec 4 §3), so every result resolves to
exact bytes.

| Verb | Action |
|---|---|
| `init` | scaffold `datasets.yml` + mirror config + gitignored cache |
| `register` | add an entry (ingest a published Croissant if present); classify tier; write datasheet |
| `fetch` | materialize via the substrate resolution chain (cache → mirror → source → gated) |
| `verify` | recompute SHA-256 vs. the manifest; report `verified-against-registry` |
| `mirror` | populate/refresh the private rclone mirror |
| `audit` | fixity + presence + license completeness across the whole manifest |

**Agency & Understanding:** the skill fetches, verifies, and reports; it never
decides which dataset is *appropriate* for a claim (a scientific judgement) — that
is the researcher's, and a `grill` `methodology`/`claim` probe may ask them to
justify the choice.

## 2. Registry — `datasets.yml`

A thin, repo-owned YAML manifest (**not** DVC's machinery — its dependency mass
fails the light posture). Field names map onto **schema.org / Croissant** +
**DataCite** so an entry can *ingest* and *emit* a Croissant file (venue-mandated
at NeurIPS D&B). Extends the substrate base record (sub-spec 4 §2.1) with:

- `tier` — A/B/C
- `retrieval` recipe — `kind: http | doi | openml | git-lfs | manual`, url/params, or acquisition `instructions`
- `datasheet` — path/URL to the Gebru-style datasheet (required; may be "N/A + reason")
- `sensitivity` — flag (required if PII/confidential)

Required core (per the standards digest): `id, version, tier, license,
redistributable, access, files[](path + sha256), datasheet`; `source`/`retrieval`
for B, `instructions` for C.

## 3. Storage tiers — (redistribution license × access automation)

| Tier | Condition | Behavior |
|---|---|---|
| **A — committed** | small **and** license permits redistribution | committed via git/LFS; reproducible with zero setup |
| **B — auto-retrievable** | public, fetchable via stable URL/API, but large or non-redistributable | registry holds fetch recipe + sha256; materialized into gitignored cache; mirror = link-rot insurance |
| **C — manual / gated** | login/EULA/registration, or no stable source | metadata + datasheet + acquisition instructions + sha256; skill **verifies presence/integrity only**, never fetches; mirror only if the DUA permits a private copy |

**Hard rule (from the standards digest):** a private mirror is **storage, not a
redistribution grant.** Tier and `redistributable` are set by the *license*, never
by mirror presence; the skill refuses to promote to Tier A on mirror existence and
never surfaces mirror contents publicly. Basis: FAIR A1.2/A2 (metadata stays open
even when bytes are gated) supports Tier C.

Tier assignment is proposed by the skill (from license + size + access) but
**confirmed by the human** on `register` (a license/tier judgement).

## 4. Retrieval, mirror & fixity

Uses the substrate mechanism (sub-spec 4 §2.3–2.4): the resolution chain (cache →
private mirror → public source → gated instructions), SHA-256 authoritative in the
manifest with rclone's native hash only as a transport check, content-addressed
mirror keys (`sha256/<hash>`). Tier-B public fetch uses **pooch** (3 pure-Python
deps; HTTP/FTP/SFTP + `doi:`); Tier-A uses git/LFS; Tier-C is instruct-drop-verify.
`huggingface_hub` is used opportunistically for HF-hosted/gated datasets.

## 5. Provenance → evidence & datasheets

- **Provenance:** on each run, the backend's stamp copies `id + version + sha256`
  per dataset (sub-spec 4 §3); `verify` gates a run as valid before its evidence
  counts.
- **Datasheet** (Gebru et al.) per entry closes the loop with the rigor kit's
  per-dataset datasheets (meta-spec §3.5) and is a `grill` `methodology` target
  ("what are this dataset's known biases / collection limits?").
- **Croissant** emit/ingest is the interop seam; DataCite `citation` + DOI make a
  dataset citable (and DOI-archivable at Zenodo/Dryad *iff* redistributable).

## 6. Plugin vs. consumer

- **Plugin:** the `dataset` skill; the `datasets.yml` schema + loader/validator;
  tier policy; the Tier-B (pooch) / Tier-A (git-LFS) / Tier-C (drop-verify)
  fetchers; Croissant import/export; datasheet template; the substrate resolution
  chain + rclone mirror. Deps: `pyyaml` + `pooch` (+ substrate rclone). Not in the
  published `mononet` wheel.
- **Consumer:** the `datasets.yml` entries + SHA-256 checksums; Tier-A blobs;
  rclone remote config (untracked/CI); the gitignored cache; any heavy loaders
  (`datasets`, pandas) — kept out of the plugin.

## 7. Open items

- **Mirror hash** — settled in sub-spec 4: SHA-256 authoritative, rclone native
  hash as transport check only.
- **Croissant version** — target the current MLCommons spec; treat as export
  format, registry is the superset source of truth.
- **`.datasets-cache/` vs `.scholar/`** — confirm cache directory placement with
  the meta-spec `.scholar/` decision.
