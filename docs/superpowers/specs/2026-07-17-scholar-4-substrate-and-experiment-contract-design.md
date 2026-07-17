# scholar sub-spec 4 — Shared substrate & experiment-backend contract

**Date:** 2026-07-17
**Author:** Davor Runje
**Status:** Brainstorming output; sub-spec of the `scholar` meta-spec, pending implementation plan.

> Sub-spec of [2026-07-17-scholar-plugin-design.md](2026-07-17-scholar-plugin-design.md).
> Defines the foundation that sub-specs 2 (literature) and 3 (dataset) build on:
> the **asset-provenance substrate** (registry pattern + private mirror + fixity +
> persistent-ID) and the **experiment-backend contract**. Governed by the meta-spec's
> ⚑ guiding principle (assistants, not researchers) and its light-dependency posture.
> Grounding: [dataset-management-standards](../scholar/references/dataset-management-standards.md),
> [dataset-tooling-mirror](../scholar/references/dataset-tooling-mirror.md).

## 1. Purpose

Two things are shared across the plugin and therefore specified once, here:

1. **The asset-provenance substrate** — the common machinery under `literature`
   and `dataset`: how the plugin keeps a git-tracked registry of externally
   sourced, persistently identified, license-bearing, mirror-able assets, and how
   it fetches and verifies durable copies. `literature` and `dataset` are distinct
   front-ends over this one substrate (they share the *mechanism*, not the *file*).
2. **The experiment-backend contract** — the stable interface through which the
   pipeline skills obtain and cite experimental evidence, so the runner is
   hot-swappable and no pipeline skill depends on a concrete tool.

## 2. The asset-provenance substrate

### 2.1 What is shared vs. extended

The substrate provides a **base asset record** and the **mechanism** to
materialize and verify it. Each front-end *extends* the base with its own schema
and adds its own acquisition/analysis logic (specified in sub-specs 2 and 3).

**Base asset record (the spine):**

| Field | Meaning |
|---|---|
| `id` | stable slug, unique within the registry |
| `pid` | persistent identifier — DOI / arXiv-id / DataCite DOI (optional but preferred) |
| `title` / `description` | human labels |
| `source` | canonical URL / API locator, or acquisition instructions |
| `files[]` | each: `path`, `checksum` (`sha256:…`), `size` |
| `license` | SPDX id or explicit terms + URL |
| `redistributable` | bool — may we republish the bytes? |
| `access` | `open` \| `gated` |
| `mirror` | logical rclone remote + content-addressed key (present iff a mirror copy exists) |
| `citation` | DataCite 6-tuple (Identifier, Creator, Title, Publisher, PublicationYear, ResourceType) when a citable record exists |

*Basis:* FAIR (Wilkinson 2016), FORCE11 data-citation, DataCite v4.6, fixity/BagIt — see the standards digest.

- **`literature` extends** it with bibliographic fields (authors/venue/year), a
  triage sidecar (role, disposition, rationale, `seeded` links, citation intent),
  and the PDF as the `files[]` payload. Its registry *format* is a standard
  bibliography (BibTeX / CSL-JSON) + a YAML triage sidecar, **not** a clone of the
  dataset manifest (sub-spec 2).
- **`dataset` extends** it with `tier` (A/B/C), a `retrieval` recipe, `datasheet`,
  `sensitivity`, and data blobs as the `files[]` payload; its registry is
  `datasets.yml` (sub-spec 3).

The plugin ships one **loader/validator** for the base record and one
**materialization engine** (below); the front-ends supply their extension schema.

### 2.2 Persistent-ID & citation layer

A single shared module resolves and records persistent identifiers (DOI, arXiv,
DataCite) and emits citations. Papers and datasets are both DOI-citable; the same
code mints/records `pid` and renders `citation`. This is the interop seam to
external archives (Zenodo/figshare/Dryad) and to Croissant/BibTeX export.

### 2.3 Private mirror + fixity (the mechanism)

The durable-copy mechanism is identical for a PDF and a `.parquet` and is
specified once:

- **Engine: `rclone`** — invoked as a subprocess (a single Go binary, *not* a
  Python dependency). Backend-agnostic (Google Drive first; S3/B2/R2/SFTP/WebDAV/…
  by config). The registry names a **logical remote + base path**; the
  credential-bearing `rclone.conf` is **never committed** (untracked file via
  `RCLONE_CONFIG`, or env-var remotes from CI secrets). A `rclone.conf.example`
  (remote name + type only) is committed. `rclone obscure` is not encryption and
  is never committed.
- **Fixity: two-layer.** The **authoritative checksum is SHA-256, stored in the
  registry** (integrity == identity == citation-verifiability). rclone's native
  per-backend hash (often MD5 on Google Drive/S3) is used only as a transfer
  check; after any transfer the local bytes are **re-hashed against the registry
  SHA-256** — so verification is backend-independent and the manifest is the
  single source of truth. *(Resolves the meta-spec §10 hash open item: SHA-256
  authoritative; MD5/native only as a transport check.)*
- **Store layout:** gitignored cache; **content-addressed on the wire**
  (`sha256/<hash>` mirror keys → dedup, integrity == identity), name-addressed for
  use (symlink/copy from the content-addressed blob).

### 2.4 Resolution chain

`materialize(asset)` — verify fixity at every hop; a file failing verification is
treated as absent and the chain continues:

```
1  LOCAL CACHE      exists and sha256==registry  → return; else discard-if-corrupt
2  PRIVATE MIRROR   (if configured) rclone copy mirror:base/<key> → verify → return; else fall through
3  PUBLIC SOURCE    front-end fetcher (literature: PDF/DOI; dataset Tier A git-LFS, Tier B pooch/http/doi)
                    → assert sha256==registry (hard fail) → populate mirror → return
4  GATED / MANUAL   print acquisition instructions → wait for operator drop → verify → populate mirror → return
```

Front-ends supply step-3 fetchers; the substrate owns steps 1, 2, 4 and all
fixity checks. The mirror is populated on first successful acquisition, giving
link-rot insurance (dataset Tier B) and a re-acquirable home for gated assets
(dataset Tier C, or paywalled PDFs).

### 2.5 License / redistribution rule (shared, non-negotiable)

Redistribution rights are set by the **license**, recorded on the base record,
and are **never** implied by the existence of a mirror. A private mirror is
storage, not a redistribution grant. The substrate refuses to treat mirror
presence as permission to commit bytes in-repo, and never surfaces mirror
contents publicly. (Applies identically to copyrighted PDFs and licensed data.)

### 2.6 Plugin vs. consumer

- **Plugin:** base-record schema + loader/validator; the persistent-ID/citation
  module; the rclone mirror integration + fixity + content-addressed cache; the
  resolution chain (steps 1/2/4); the `rclone.conf.example` generator and
  secret-hygiene docs. Deps: `rclone` (subprocess) + `pyyaml` (+ `pooch`, owned by
  the dataset front-end).
- **Consumer:** the registry entries + SHA-256 checksums; the real rclone remote
  config (untracked/CI); mirror blobs; the gitignored cache.

## 3. The experiment-backend contract

Pipeline skills never run experiments directly; they depend on this contract, and
each consumer binds a concrete implementation. The contract has **four
capabilities**:

| Capability | Purpose | Returns |
|---|---|---|
| **`run`** | execute (or resume) an experiment for a given config | a **run-ref** (opaque, stable handle) |
| **`evidence`** | fetch results for a run-ref | structured results + a **provenance stamp** (config hash, code/symbol provenance, dataset `id+version+sha256` per §2, timestamps, hardware) |
| **`tables`** | render results into the doc-facing artifacts (result tables/figures) a paper/hypothesis cites | rendered artifacts (managed blocks) |
| **`is-current`** | check whether a run-ref's evidence is still valid given current code/config/data | `current` \| `stale(reasons)` |

Semantics:

- A **run-ref** is the citable unit of evidence. `findings.md` (hypothesis) and
  the paper `ledger`/`decision` reference **run-refs**, never raw numbers copied
  by hand. The `tables` capability is the only writer of result numbers into docs
  (managed, regenerable blocks).
- The **provenance stamp** must carry the dataset fingerprint from §2
  (`id+version+sha256`), so every reported result resolves to exact bytes.
- **`is-current`** is what makes selective re-execution honest: a hypothesis or
  paper can ask "is my evidence stale?" without knowing how the backend computes
  staleness.
- The contract is **agnostic to how experiments run** (local, GPU fan-out,
  cluster, cached). Nothing above mentions a scheduler.

### 3.1 Binding

Each project binds a backend in `docs/research/papers.md` via a `backend:` field
(default `mononet-bench`). A backend is any implementation exposing the four
capabilities. Pipeline skills resolve the backend from the binding; they contain
no backend-specific logic.

### 3.2 Default implementation & migration

- The **default backend is `mononet-bench`** — the benchmark experiment
  orchestration designed in
  [2026-07-15-benchmark-experiment-orchestration-design.md](2026-07-15-benchmark-experiment-orchestration-design.md)
  (PR #127). It stays in `mononet` as *mononet's implementation* of this contract
  (`run` = doit/gpu-pool executor; `evidence` = committed-JSON results +
  `.provenance.json`; `tables` = the `render` managed-block write-back;
  `is-current` = the symbol-closure provenance hash).
- The **PINN application paper** (PR #116) adopts this default backend once the
  infrastructure lands (it currently has a divergent harness).
- Because the contract is abstract, a different project may bind a different
  backend (e.g. a lightweight local runner) without touching any pipeline skill.

### 3.3 Agency-principle interaction

The backend produces and stamps evidence; it never adjudicates it. Whether the
evidence *confirms or refutes* a hypothesis, or *supports publication*, is a
human decision recorded with a sign-off (meta-spec §2.1). `is-current` reports
staleness; it does not decide to re-run — the researcher does.

## 4. Open items

- **run-ref format** — opaque string vs structured (backend-hash + trial-id);
  settle when the first non-default backend is contemplated. Default: opaque
  string the backend can resolve.
- **Croissant/BibTeX export placement** — whether the persistent-ID module owns
  export, or each front-end does; lean toward the shared module for `pid`/citation
  and the front-end for its native format (BibTeX for literature, Croissant for
  dataset).
- **Cache directory name** — align with the meta-spec's `.scholar/` decision.

## 5. Downstream

Sub-spec 2 (literature) and sub-spec 3 (dataset) consume §2. Sub-spec 1
(lifecycle) consumes §3 (pipeline skills cite run-refs; `hypothesis-testing`,
`paper-synthesis`, and the `progress` roll-up read `evidence`/`is-current`).
