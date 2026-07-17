# Reproducible Dataset Management — standards digest

**Date:** 2026-07-17 · **For:** `scholar` `dataset` capability (sub-spec 3). · **Status:** verified-source digest; migrates to the plugin's `resources/references/`.

Grounding for a domain-neutral dataset registry + tiered retrieval + private mirror.

## 1. Named standards

- **FAIR** — Wilkinson et al. (2016), *Scientific Data* 3:160018, DOI:10.1038/sdata.2016.18. **A1.2** allows an auth step; **A2** requires metadata to persist even when data don't → the citable basis for gated **Tier C** (metadata/datasheet/citation stay open; blob gated). FAIR ≠ open. Map: **F**→persistent id/DOI/rich metadata/version; **A**→retrieval recipe + `access` (open/gated) + mirror; **I**→SPDX license ids, schema.org/Croissant names; **R**→license/provenance/datasheet.
- **Datasheets for Datasets** — Gebru et al. (2021), *CACM* 64(12):86–92, DOI:10.1145/3458723. Seven sections. Questions that become **mandatory registry fields**: Distribution→license & redistribution terms (drives tier), Distribution→access (open/gated), Composition→sensitivity/PII flag, Maintenance→version + maintainer, Collection→source. Rest stays as the linked datasheet doc.
- **MLCommons Croissant** — Akhtar et al. (2024), NeurIPS D&B, arXiv:2403.19546; workshop DOI:10.1145/3650203.3663326. JSON-LD on **schema.org/Dataset** + `distribution` (files + checksums), `recordSet`, RAI metadata. Emitted by HF/Kaggle/OpenML/Google Dataset Search. **Use as the interop target, not the internal source of truth**: model registry fields to map onto Croissant names (`name`, `license`, `citeAs`, `version`, `distribution[].contentUrl`, `.sha256`); ingest a published Croissant to bootstrap an entry; emit one per entry. Keep tier / mirror / resolution as registry-native. **NeurIPS D&B now requires a Croissant file** — emitting it is venue-mandated, not gold-plating.
- **Data citation** — FORCE11 Joint Declaration (2014): 8 principles; **Evidence** (a data-reliant claim cites the data) and **Specificity & Verifiability** (resolves to exact version/subset + fixity → version + checksum, not just a name) map directly. **DataCite Metadata Schema** (v4.6): **6 mandatory** — Identifier(DOI), Creator, Title, Publisher, PublicationYear, ResourceType → the citation core; also the minimum to mint a DOI. **DOIs** satisfy FAIR-F + JDDCP Unique-ID/Persistence.
- **DOI archival vs private mirror** — Zenodo / figshare / Dryad mint DataCite DOIs (durable public copies). **Decision rule:** redistributable → prefer a public DOI archive (record the DOI); **non-redistributable → private mirror only**. Not mutually exclusive for redistributable data.
- **Fixity** — checksums (recommend **SHA-256**; MD5 only if a source publishes only MD5); verify on every materialization + mirror hop. **BagIt (RFC 8493, 2018)**: `data/` payload + `manifest-<alg>.txt`; allow (not require) for multi-file snapshots to the mirror / a Zenodo deposit.
- **DMP** — funder/institutional requirement; the registry is the executable per-dataset instantiation of a DMP's storage/preservation/access/licensing sections. Treat DMP export as optional downstream, not a core field set. `[funder wording not individually verified]`

## 2. Registry schema (one entry) — names mapped to Croissant/DataCite

| Field | Req? | Basis |
|---|---|---|
| `id` | **req** | FAIR-F; JDDCP Unique-ID |
| `name`/`description` | **req** | schema.org; Datasheet |
| `version` | **req** | JDDCP Specificity |
| `tier` (A/B/C) | **req** | operational |
| `license` (SPDX or explicit + URL) | **req** | FAIR-R; Datasheet |
| `redistributable` (bool) | **req** | drives tier |
| `access` (open/gated) | **req** | FAIR-A1.2 |
| `files[]` `path`+`checksum`+`algorithm` (SHA-256) | **req** | fixity; BagIt; Croissant |
| `source` | **req B/C** | FAIR-A; JDDCP Access |
| `retrieval` recipe | **req B** | FAIR-A machine-actionable |
| `mirror` | req if a mirror copy exists | durability |
| `datasheet` | **req** (may be "N/A + reason") | Gebru 2021 |
| `citation`/`doi` | req if exists | JDDCP; DataCite |
| `sensitivity` | opt (req if PII) | Datasheet |
| `size_bytes`, `maintainer`, `retrieved_at`/`verified_at` | opt | heuristic/provenance |

Required core rationale: `id+version+checksum` = JDDCP Specificity + provenance; `license+redistributable+access` = compute tier + stay legal; `datasheet`+`citation` = FAIR-R/Credit.

## 3. Licensing → tier

Tier = f(redistribution right, access automation); redistribution dominates "can it go Tier A (committed)?"

| License / terms | Redistribute? | Tier |
|---|---|---|
| CC0, CC-BY | yes | **A if small** (+ attribution); else redistributable **B**, DOI-archive-eligible |
| CC-BY-SA / CC-BY-NC | yes, w/ condition | A/B; record constraint |
| OpenML | per-dataset license; platform API-fetchable | usually **B** (fetch by id + checksum); redistribute only if underlying license permits |
| UCI | per-dataset; often citation-request not explicit grant | **B** by default; A only with clear redistribution license |
| Gated / EULA / credentialed (e.g. PhysioNet/MIMIC) | no | **C**: metadata+datasheet+instructions+checksum; verify only; mirror only if DUA permits a private copy |

**Hard rule:** a private mirror is **storage, not a redistribution license.** Tier + `redistributable` set by the license, never by mirror presence; refuse to promote to Tier A on mirror existence; never surface mirror contents publicly.

## 4. Provenance → evidence

Each experiment record copies, at run time, `dataset_id` + `version` + `checksum` (+alg) of every dataset used (+ optionally `doi`). `verify` recomputes checksums against the registry before a run counts as valid (records verified-against-registry). Two-way traceability. Aligns with NeurIPS D&B Croissant requirement + reproducibility-checklist norms.

## 5. One skill, verbs

One `dataset` skill with `register / fetch(resolve) / verify / mirror / export(cite)` — verbs share one artifact (the entry) and one policy (tier + resolution chain); they're sequential lifecycle stages with heavy state overlap (fixity is not separate from fetch). Keep the generic engine in the plugin; entries + blobs in the consumer. If the SKILL grows, factor reference material (tier table, schema, datasheet template) into supporting files (progressive disclosure), not multiple skills.

## Sources
Wilkinson 2016 (sdata.2016.18) · Gebru 2021 (3458723) · Croissant (arXiv:2403.19546; spec docs.mlcommons.org/croissant) · FORCE11 JDDCP 2014 · DataCite v4.6 · BagIt RFC 8493 · Zenodo/Dryad/figshare · NeurIPS D&B Croissant requirement (blog.neurips.cc).

## Flags
DataCite 4.7 exists; anchored on v4.6 (6 mandatory props verified). DMP mandates generic. FAIR/Datasheets DOIs from known identifiers.
