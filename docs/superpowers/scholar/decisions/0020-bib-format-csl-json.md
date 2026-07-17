# ADR-0020: CSL-JSON as source of truth; BibTeX exported on demand

- Status: accepted · Date: 2026-07-17 · Deciders: Davor Runje

## Context

The literature registry's bibliographic-facts layer (ADR-0008) needs a concrete
format. BibTeX and CSL-JSON are losslessly inter-convertible (Zotero/pandoc), so
the real choice is which is the hand-edited/authoritative source and which is
generated. The `scholar` skills manipulate this layer programmatically (scout
appends rows, triage joins by key, position builds matrices).

## Decision drivers

- Robust, well-specified parsing/validation for programmatic manipulation.
- Git-native, scriptable posture.
- LaTeX manuscripts still need `.bib` to `\cite` from.

## Considered options

1. **CSL-JSON as source of truth; BibTeX exported at paper-build time.**
2. BibTeX as source of truth; CSL-JSON generated when needed.

## Decision

Option 1. CSL-JSON (clean schema, robust to parse/validate, good Unicode; the
format Zotero/pandoc use internally) is authoritative; `.bib` is produced on
demand via pandoc/Zotero for LaTeX.

## Consequences

- The machine-manipulated layer is robust; LaTeX is still served by export.
- Authors who prefer hand-editing `.bib` treat it as a generated view.

## Rejected alternatives

- **BibTeX-primary** — loosely specified (classic vs BibLaTeX dialects),
  inconsistent field semantics, awkward Unicode, harder to parse robustly — poor
  fit for the programmatic manipulation the skills do.

## Links

sub-spec 2 §4, §7; ADR-0008.
