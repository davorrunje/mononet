# scholar — Decision Log (ADRs)

MADR-style Architecture Decision Records for the `scholar` plugin. Each records
the context, the options considered, the rejected alternatives **and why**, and
links to the spec section + grounding digest. Captured 2026-07-17 during the
design session; the reasoning here is not reconstructable from the specs alone.
Migrates to the plugin's `decisions/` alongside `resources/references/`.

> Purpose beyond the record: this log is the raw material for a planned blog
> post / paper explaining the skills and the reasoning behind them — the intent
> is eventually to write that *using* `scholar` (`paper-synthesis` + `grill`).

| ADR | Decision | Status |
|---|---|---|
| [0001](0001-separate-plugin-repo.md) | Package the workflow as a standalone plugin, not in-mononet | accepted |
| [0002](0002-scientific-scope-only.md) | Scientific scope only; engineering delegated to `superpowers` | accepted |
| [0003](0003-agency-principle.md) | Agency principle — assistants, not researchers | accepted |
| [0004](0004-understanding-principle-and-grill.md) | Understanding principle + the `grill` skill | accepted |
| [0005](0005-three-level-mirror.md) | Three-level mirror; thesis as a partial mirror | accepted |
| [0006](0006-two-skills-per-level.md) | Two pipeline skills per level (generate/resolve) | accepted |
| [0007](0007-literature-one-skill-two-modes.md) | `literature` = one skill, scout/position modes | accepted |
| [0008](0008-literature-bib-plus-triage.md) | Literature registry = standard bib + triage sidecar | accepted |
| [0009](0009-dataset-thin-manifest.md) | `dataset` one skill + thin `datasets.yml` (reject DVC/DataLad/lakeFS) | accepted |
| [0010](0010-storage-tiers.md) | Tiers A/B/C by (license × access); mirror ≠ redistribution grant | accepted |
| [0011](0011-rclone-mirror-sha256.md) | rclone private mirror; SHA-256 authoritative fixity | accepted |
| [0012](0012-shared-substrate.md) | Shared asset substrate — share the mechanism, not the file | accepted |
| [0013](0013-experiment-backend-contract.md) | Experiment-backend *contract* + pluggable backend | accepted |
| [0014](0014-progress-cross-cutting.md) | `progress` cross-cutting; semantic roll-up; anti-Goodhart | accepted |
| [0015](0015-grill-cross-cutting.md) | `grill` cross-cutting; guardrail stop-and-confirm; teaching; non-grading | accepted |
| [0016](0016-mentor-personas.md) | Mentor personas author-selectable; no personality inference | accepted |
| [0017](0017-research-init-one-skill.md) | `research-init` one skill, init/adopt modes | accepted |
| [0018](0018-git-native-source-of-truth.md) | Git-native plain text as source of truth; cross-repo out of scope | accepted |
| [0019](0019-public-plugin-visibility.md) | Public plugin, named `scholar` | accepted |
| [0020](0020-bib-format-csl-json.md) | CSL-JSON source of truth; BibTeX exported on demand | accepted |
| [0021](0021-thesis-gate-per-gap-confirmation.md) | Thesis gate — per-gap acknowledged confirmation | accepted |
| [0022](0022-license-apache-2.0.md) | License the plugin under Apache-2.0 | accepted |

Format: MADR (Markdown Any Decision Records). Deciders: Davor Runje (with Claude).
