# scholar — Scientific Research-Workflow Plugin (Meta-Spec)

**Date:** 2026-07-17
**Author:** Davor Runje
**Status:** Brainstorming output; parent meta-spec, pending four sub-specs and implementation plans.

> **Scope of this document.** This is the *parent* spec for a new, standalone
> Claude Code plugin — **`scholar`** — that packages the scientific
> research-workflow skills currently being designed inside `mononet`
> (`docs/superpowers/specs/2026-07-15-*`). It establishes the global picture:
> identity, scope, plugin architecture, the plugin↔consumer boundary, the
> onboarding skill, distribution, and how the existing in-repo work migrates.
> The detailed designs live in four sub-specs (§8). **Read this first.**

## 1. Goals & non-goals

### Goals

- Package a **uniform scientific research-workflow** — idea → literature →
  hypothesis → test → publish-decision → paper — as a **domain-neutral,
  installable Claude Code plugin**, so that the author, company colleagues, and
  PhD peers all work the *same* way and re-derive neither the workflow nor its
  rigor per project. Reducing shared cognitive load is the primary motivation.
- Make the workflow **symmetric across two nested levels**: hypotheses within a
  paper, and papers within a portfolio (see §3). The same object×action shape
  applies at both levels.
- Provide **three shared capabilities** the workflow draws on — a literature
  engine, a dataset-management engine, and a pluggable experiment backend —
  behind stable contracts so they are hot-swappable.
- Ship an **onboarding skill** that both scaffolds a fresh consumer repo
  (`init`) and backfills an existing one (`adopt`). `mononet` is the first and
  highest-value `adopt` case.
- Keep the plugin **light** (the same dependency posture as `mononet`:
  single-binary tools over heavy Python deps; no Rust-binary surprises) and
  **git-native** (plain-text, diffable, PR-reviewable artifacts; no external
  trackers as the source of truth).

### Non-goals

- **No engineering workflow.** Design, planning, implementation, debugging, and
  test authoring are delegated to **`superpowers`** (brainstorming →
  writing-plans → implementation). `scholar` calls out to them; it does not
  reimplement them. This boundary is deliberate and load-bearing.
- **No domain assumptions.** Nothing monotonic-network- or even
  ML-specific ships in the plugin. Domain content is supplied by the consuming
  repo as config and data.
- **No hosted services as source of truth.** No MLflow/W&B/Zotero-DB
  dependency for provenance. Optional authoring front-ends (e.g. Zotero) may
  *export* into the git-tracked artifacts, but the repo is authoritative.
- **No experiment *runner*.** `scholar` defines the experiment-backend
  *contract*; each consuming repo supplies the implementation (for `mononet`,
  the benchmark orchestration of PR #127).

## 2. Identity & scope

`scholar` is the **scientific** counterpart to `superpowers` (engineering).
Its unit of work is a *scientific claim* and its lifecycle; its outputs are
hypotheses, evidence, decisions, and papers. Everything that is "how do I build
the thing that produces the evidence" is `superpowers`' job.

The firewall that governs the workflow:

- **Exploration proposes** (generates candidate hypotheses / papers).
- **Resolution disposes** (tests a hypothesis to a verdict; develops a paper).
- **Synthesis reports** (assembles the paper from confirmed evidence).

No skill both proposes and adjudicates the same claim.

## 3. Architecture overview

### 3.1 The two-level mirror

The workflow is one shape applied at two nested levels. This symmetry is the
core design principle — it is why there are not two different sets of skills.

| Level | **generate** skill | **resolve** skill | staged docs (the pipeline) | backlog |
|---|---|---|---|---|
| **hypothesis** (within a paper) | `hypothesis-exploration` | `hypothesis-testing` | hypothesis → **strategy** *(science)* → design/plan *(eng, delegated)* → **findings** *(verdict)* | `backlog.md` |
| **paper** (portfolio) | `paper-exploration` | `paper-synthesis` | pitch → **positioning** *(related works)* → outline/plan *(eng, delegated)* → **decision** *(publish verdict)* | `portfolio-backlog.md` |

The two "missing" lifecycle stages the user originally named resolve into this
mirror rather than into new skills:

- **"Research related works"** is the paper-level analog of a hypothesis's
  **strategy** (the scientific thinking) → it is `positioning`, produced via the
  `literature` capability's `position` mode.
- **"Decide whether to publish"** is the paper-level analog of a hypothesis's
  **findings verdict** → it is `decision`, a staged doc gated on accumulated
  hypothesis evidence + positioning.

Two nested flywheels result: a per-paper loop (hypotheses accumulate into a
paper) inside a portfolio loop (papers accumulate into a research program).

### 3.2 Pipeline skills (4)

`hypothesis-exploration`, `hypothesis-testing`, `paper-exploration`,
`paper-synthesis`. These are the current `2026-07-15-*` designs, refactored to
depend only on the capability contracts (not on `mononet` internals) and to
carry the mirrored staged-doc discipline at both levels.

### 3.3 Shared capabilities (3)

| Capability | Form | Ships where | Consumer supplies |
|---|---|---|---|
| `literature` | one skill, modes `scout` (generative → idea backlog) / `position` (defensive → related-works, PRISMA log) | **plugin** | anchors, API config |
| `dataset` | one skill, verbs `init/register/fetch/verify/mirror/audit` | **plugin** (engine) | `datasets.yml` entries, blobs, mirror creds |
| experiment backend | a **contract** (run / evidence / tables / is-current) | **plugin** (contract only) | the implementation, bound per project |

`scout`/`position` and the dataset verbs each take a `level` (or `mode`)
parameter that tunes ranking / depth / stopping — the level split is a
parameter, not a skill boundary (established by the literature and dataset
research passes; see sub-specs).

### 3.4 Shared substrate

`literature` and `dataset` are distinct front-ends over a common
**asset-provenance substrate**: a git-committed registry of externally-sourced,
persistently-identified, license-bearing, mirror-able assets. Shared pieces:

- **Registry pattern** — a git-tracked manifest with provenance.
- **Private mirror + fixity** — `rclone` (backend-agnostic; Google Drive first,
  S3/B2/… later with zero design change) + checksums + a content-addressed
  store. "Fetch and verify a durable copy of an external artifact" is identical
  for a PDF and a `.parquet`.
- **Persistent-ID / citation** — DOI / arXiv-id / DataCite; a shared vocabulary
  (papers *and* datasets are DOI-citable).
- **License / redistribution field** — drives what may be committed vs.
  mirror-only.

The registries themselves are **not** unified: literature uses a standard
bibliography format (BibTeX / CSL-JSON, Zotero-friendly) plus a git-tracked
**triage sidecar** (our decisions: role, disposition state-machine, rationale,
`seeded`→backlog links, citation intent); datasets use a custom `datasets.yml`
operational manifest. They share the *mechanism*, not the *file*.

### 3.5 Rigor kit

A cross-cutting set of checklists/templates baked into the staged docs, and the
part where a shared standard most reduces cognitive load across peers:
confirmatory-vs-exploratory tagging, rival hypotheses + discriminating tests,
severity / power / MDE + TOST for null claims, the disclosure checklist,
per-dataset datasheets (Gebru et al. — closing the loop with the `dataset`
capability), the red-team pass, and file-drawer discipline.

## 4. Plugin repo layout

Standalone repo, distributed as a Claude Code plugin. Working name `scholar`;
skills are namespaced `scholar:<skill>`.

```
scholar/                                  # plugin repo root
├── .claude-plugin/
│   └── plugin.json                       # manifest: name, version, description
├── skills/
│   ├── research-init/SKILL.md            # init | adopt (§6)
│   ├── hypothesis-exploration/SKILL.md
│   ├── hypothesis-testing/SKILL.md
│   ├── paper-exploration/SKILL.md
│   ├── paper-synthesis/SKILL.md
│   ├── literature/SKILL.md               # scout | position
│   └── dataset/SKILL.md                  # init/register/fetch/verify/mirror/audit
├── resources/                            # cross-skill shared material
│   ├── contracts/
│   │   └── experiment-backend.md         # the 4-capability contract
│   ├── substrate/
│   │   └── asset-registry.md             # spine schema; mirror/fixity/ID conventions
│   ├── templates/                        # staged-doc templates (both levels) + registries
│   ├── rigor/                            # rigor-kit checklists
│   └── references/                       # verified methodology digests (the research)
├── README.md
└── LICENSE
```

`resources/references/` carries the verified-source digests produced during
brainstorming: the four existing research-workflow reference docs (on PR #128)
plus the four generated this session — **citation scouting**, **related-works
synthesis**, **dataset-management standards**, and **dataset tooling / mirror
architecture**. These are the evidentiary base for the sub-specs and must be
persisted, not left in conversation.

## 5. Plugin↔consumer boundary

The plugin ships generic logic; the consuming repo owns content, config, and
the experiment-backend implementation. After `init`/`adopt`, a consumer repo
(illustrated for `mononet`) looks like:

```
<consumer-repo>/
├── docs/research/
│   ├── papers.md                         # registry: paper-id → root + backend binding
│   ├── <paper>/
│   │   ├── hypotheses/<YYYY-MM-DD-slug>/{hypothesis,strategy,design,plan,findings}.md
│   │   ├── backlog.md
│   │   └── paper/{positioning,outline,ledger,decision, sections/}
│   ├── portfolio-backlog.md
│   └── literature/
│       ├── references.bib                # or CSL-JSON — bibliographic facts
│       └── triage.yml                    # decision sidecar (keyed by citekey/DOI)
├── datasets.yml                          # dataset registry (entries + checksums + tiers)
├── .datasets-cache/                      # gitignored materialized data
├── .scholar/
│   ├── config.yml                        # rclone remote name, lit anchors, backend binding
│   ├── rclone.conf                       # gitignored (creds)
│   └── rclone.conf.example               # committed template (remote name/type only)
└── <experiment-backend implementation>   # e.g. mononet's benchmark orchestration (PR #127)
```

| Lives in the **plugin** | Lives in the **consumer** |
|---|---|
| the 7 skills; capability engines (literature, dataset); templates; rigor kit; methodology digests; the substrate + experiment-backend **contracts** | `docs/research/` content; `datasets.yml` entries + blobs; `.scholar/` config + mirror creds; the experiment-backend **implementation**; literature anchors |

## 6. The `research-init` skill (init / adopt)

One skill, two modes — both drive a repo to the layout of §5; `adopt` is `init`
plus an inventory-and-map phase.

- **`init` (greenfield)** — scaffold the `docs/research/` layout, the registries
  (`papers.md`, `datasets.yml`, `references.bib` + `triage.yml`), `.scholar/`
  config (rclone remote name, literature anchors, experiment-backend binding),
  and the staged-doc templates. Delegates per-item registration to the
  capability skills' own verbs rather than reimplementing them.
- **`adopt` (backfill)** — inventory an existing repo, propose mappings,
  materialize with the user confirming judgment calls (licenses, tiers,
  which result maps to which hypothesis). For `mononet` specifically:
  - `docs/references/` PDFs + digests + the CLAUDE.md paper table + the eight
    methodology digests → literature `references.bib` + `triage.yml`, with
    roles pre-tagged (Runje 2023 = anchor; Sartor 2025, DLN, Sill = rival /
    prior-art).
  - benchmark data + download scripts under `benchmarks/` → `datasets.yml`
    entries (compute checksums, infer source/license, assign tier).
  - existing results / specs / memories (depth-null, flavor-ablation,
    `monotone-depth-collapse-lean-brief.md`) → retroactive hypothesis docs with
    findings — the "detailed record per hypothesis" applied historically.
  - the benchmark orchestration (PR #127) → bound as `mononet`'s
    experiment-backend implementation in `.scholar/config.yml`.

`adopt` is the direct payoff for the "benchmarks folder out of control, no
systematic record" problem that motivated this whole effort.

## 7. Distribution

Distributed as a Claude Code plugin (git-repo marketplace install). Shared
privately with company colleagues and with PhD peers; a public release is
possible later but is not required for the initial audience. The plugin must
therefore be genuinely domain-neutral and self-documenting from day one — its
"users" include people who are not the author.

## 8. Sub-spec decomposition

This meta-spec defers detail to four sub-specs (each date-prefixed under
`docs/superpowers/specs/`, migrating to the plugin repo per §9):

1. **Lifecycle & pipeline skills + rigor kit** — the two-level mirror, the four
   pipeline skills, staged-doc templates, firewall, flywheels, and the rigor
   kit. *(Largely the current `2026-07-15-hypothesis-*` / `paper-*` specs,
   refactored to the contracts.)*
2. **Literature capability** — `scout`/`position`, the citation-graph toolchain
   (OpenAlex + Semantic Scholar; snowballing; SciCite intent; concept matrix;
   PRISMA log), the bib + triage registry, and backlog linkage. *(Grounded by
   the citation-scouting and related-works-synthesis digests.)*
3. **Dataset capability** — the `datasets.yml` schema (schema.org/Croissant +
   DataCite-aligned), the A/B/C tier policy, the resolution chain, the rclone
   private mirror, fixity, and datasheet integration. *(Grounded by the
   dataset-standards and dataset-tooling digests.)*
4. **Shared substrate + experiment-backend contract** — the asset-provenance
   spine, the rclone/fixity/persistent-ID mechanism common to literature and
   dataset, and the formal 4-capability experiment-backend contract
   (run / evidence / tables / is-current).

## 9. Migration of existing in-repo work

The scientific-workflow work currently lives inside `mononet`. It relocates:

- **PR #128** (four research-workflow specs + `docs/research/README.md` + four
  reference digests) → **retargets the `scholar` plugin repo** instead of
  `mononet/docs/superpowers/`. Content is refactored to depend on the
  capability contracts rather than `mononet` internals.
- **PR #127** (benchmark experiment orchestration) → **stays in `mononet`** as
  its implementation of the experiment-backend contract; it is re-described as
  "mononet's experiment backend" rather than a general facility.
- The methodology digests (this session's four + #128's four) → become
  `resources/references/` in the plugin.
- `mononet` becomes the reference **consumer**: it runs `research-init adopt`
  against itself once the plugin exists.

Sequencing: this meta-spec first (draft PR), then the four sub-specs, then
implementation plans, then the plugin repo is created and `mononet` adopts it.

## 10. Open items to confirm before implementation

- **Plugin repo name / marketplace** — `scholar` is the working name; confirm
  the GitHub repo name and whether it is initially private (peers/colleagues)
  or public.
- **Bibliography format** — BibTeX vs CSL-JSON for `references.bib` (both
  Zotero-exportable; CSL-JSON is richer/JSON-native, BibTeX is LaTeX-native).
- **`literature` one-skill-with-modes** — carried as decided (`scout`/`position`
  in one skill); reconfirm at sub-spec time.
- **Mirror hash algorithm** — MD5 (lowest common denominator across Google
  Drive + S3 via rclone) vs SHA-256 (stronger, but disjoint backend hash sets);
  resolve in sub-spec 3/4.
- **`.scholar/` vs existing conventions** — confirm the config directory name
  and that it does not collide with `superpowers`/repo conventions.

None of these block writing the sub-specs.
