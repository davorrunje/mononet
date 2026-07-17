# ADR-0008: Literature registry = standard bib + triage sidecar

- Status: accepted · Date: 2026-07-17 · Deciders: Davor Runje

## Context

Should the literature registry share the dataset manifest format? A field-level
comparison showed literature and dataset registries are more different than
similar — and literature has a mature standard format that datasets lack. The
author also wants to record *decisions about* papers (interesting? act on?).

## Decision drivers

- Bibliography has a mature ecosystem standard (BibTeX/CSL-JSON, Zotero, pandoc).
- The author needs a triage/decision layer (roles, dispositions, rationale) a bib
  file doesn't have.
- Provenance spine + mirror mechanism *are* shared with datasets (ADR-0012).

## Considered options

1. **Standard bib (facts) + a git-tracked triage sidecar (decisions), joined by
   citekey/DOI.**
2. A unified `datasets.yml`-style manifest for both papers and data.
3. Bib only (no triage).

## Decision

Option 1. Bib = immutable facts (BibTeX/CSL-JSON); triage YAML = our decisions
(role, disposition state-machine, rationale, `seeded` links, intent). `acted-on`
→ backlog; `screened` + rationale → PRISMA log.

## Consequences

- Reuses the citation ecosystem; captures the decision trail the author wanted.
- Two files per corpus (bib + sidecar), joined by key.

## Rejected alternatives

- **Unified manifest** — throws away the bib ecosystem; forces a hand-rolled clone.
- **Bib only** — no place for triage decisions / PRISMA reasons.

## Links

sub-spec 2 §4; meta-spec §3.4.
