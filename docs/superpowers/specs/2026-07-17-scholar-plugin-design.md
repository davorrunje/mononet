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
- Make the workflow **symmetric across nested levels**: hypotheses within a
  paper, papers within a portfolio, and (optionally) papers within a thesis
  (see §3). The same object×action shape applies at every level.
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

- **No autonomous research.** The skills assist; they do not conduct research or
  make material scientific decisions, and the workflow cannot be "run" to
  produce a paper or thesis unattended. The researcher drives (see §2.1).
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
- **No cross-repo aggregation.** Work-research and a PhD thesis live in
  separate repos with separate lives; linking a thesis across repos (e.g.
  rolling a company paper into the thesis) is explicitly out of scope for now.
  Each repo's top level is self-contained. (Recorded as a future item.)
- **No progress *scores*.** Progress tracking surfaces state and gaps, never a
  productivity number (see §3.6). This is a hard design principle, not a
  deferral.

## 2. Identity & scope

`scholar` is the **scientific** counterpart to `superpowers` (engineering).
Its unit of work is a *scientific claim* and its lifecycle; its outputs are
hypotheses, evidence, decisions, and papers. Everything that is "how do I build
the thing that produces the evidence" is `superpowers`' job.

### 2.1 Core principle — assistant, not researcher

`scholar` skills are **assistants, not autonomous researchers.** They keep the
accounts of a research program, advise as a mentor, and discuss as a colleague —
but they do **not** perform independent research, and they do **not** make
material decisions. The researcher is in the driving seat. This is the highest
principle in the design; every other rule sits under it.

- **Material decisions are the user's, and are recorded with a human sign-off +
  date** (accountability): whether a hypothesis is confirmed / refuted, whether
  a result is real, whether a paper is worth publishing, what the thesis claims,
  whether it is defensible. A skill marshals the evidence and *advises*; the user
  *decides*. Verdict/decision artifacts (`findings`, `decision`, thesis
  defensibility) must name their human decision-maker.
- **You author; the skill drafts.** A skill may draft prose, assemble a ledger,
  format a section, or cross-check citations — but the scientific claims and
  their wording are the user's. You cannot produce a paper or thesis by "running"
  the workflow; you drive it. It automates a great deal and helps you explore and
  understand, but you stay in the seat.
- **Automation is for the mechanical and the mnemonic**, never the judgemental:
  retrieval, bookkeeping, roll-ups, gap-surfacing, consistency checks,
  discussion, exploration. Anywhere a scientific judgement is required, the skill
  stops and asks rather than deciding.
- This **subsumes and strengthens** the firewall (§2.2) and the anti-Goodhart
  stance (§3.6): the system never adjudicates, scores, or decides on the
  researcher's behalf.

Aligns with research-integrity / authorship norms (ICMJE authorship criteria;
COPE, ICML, and NeurIPS positions that AI tools cannot be authors and a human
remains accountable for the work).

### 2.2 The firewall

The firewall that governs the workflow — each stage **human-driven**, the skill
assisting and the researcher deciding (§2.1):

- **Exploration proposes** (generates candidate hypotheses / papers).
- **Resolution disposes** (tests a hypothesis to a verdict; develops a paper).
- **Synthesis reports** (assembles the paper from confirmed evidence).

No skill both proposes and adjudicates the same claim — and no skill adjudicates
*at all* without the user's recorded decision.

## 3. Architecture overview

### 3.1 The three-level mirror

The workflow is one shape applied at three nested levels. This symmetry is the
core design principle — it is why there are not three different sets of skills.

| Level | **generate** skill | **resolve** skill | staged docs (the pipeline) | children |
|---|---|---|---|---|
| **hypothesis** (within a paper) | `hypothesis-exploration` | `hypothesis-testing` | hypothesis → **strategy** *(science)* → design/plan *(eng, delegated)* → **findings** *(verdict)* | `backlog.md` |
| **paper** (portfolio) | `paper-exploration` | `paper-synthesis` | pitch → **positioning** *(related works)* → outline/plan *(eng, delegated)* → **decision** *(publish verdict)* | `portfolio-backlog.md` |
| **thesis** (top, *optional*) | `thesis` — *framing (occasional)* | `thesis` — *synthesis* | prospectus → **aims/narrative** *(the through-line)* → chapter↔paper map → **kappa** + *defensibility* | the portfolio |

The two "missing" paper-level stages the user originally named resolve into this
mirror rather than into new skills:

- **"Research related works"** is the paper-level analog of a hypothesis's
  **strategy** (the scientific thinking) → it is `positioning`, produced via the
  `literature` capability's `position` mode.
- **"Decide whether to publish"** is the paper-level analog of a hypothesis's
  **findings verdict** → it is `decision`, a staged doc gated on accumulated
  hypothesis evidence + positioning.

The **thesis level is a partial mirror**, and honestly so:

- The cumulative / thesis-by-publication model *is* this nesting (papers bound
  by a synthesizing framing chapter — the Nordic **"kappa"**). A **monograph**
  thesis is the degenerate case: one "paper" spanning the whole thesis.
- Its `resolve` action is `synthesis` — assemble the kappa (aims/narrative,
  related work, per-paper contribution statement, unifying discussion, future
  work; *no new findings*) plus the appended papers, and clear the
  **defensibility** gate.
- Its `generate` action **degenerates to occasional `framing`** — define the
  aims and which papers compose the thesis. There is one thesis, framed once and
  refined, not a high-throughput flywheel. So `framing`+`synthesis` are one
  `thesis` skill, not a generate/resolve pair.
- The thesis→papers roll-up target is **narrative coverage of the aims** (does
  every aim have supporting papers; does the kappa state the through-line) — the
  exact thing examiners judge — **not** a paper count (there is no universal N;
  the binding norm is scope). Program **milestones** (proposal → candidacy →
  annual review → submission → defense) are a small configurable, time-based
  list at this level.

Three nested loops result: a per-paper loop (hypotheses accumulate into a paper)
inside a portfolio loop (papers accumulate) inside a thesis loop (papers cover
the aims). A repo that is not a thesis simply omits the top level; its top is the
portfolio.

### 3.2 Pipeline skills (5)

`hypothesis-exploration`, `hypothesis-testing`, `paper-exploration`,
`paper-synthesis` — the current `2026-07-15-*` designs, refactored to depend
only on the capability contracts (not on `mononet` internals) and to carry the
mirrored staged-doc discipline at every level — plus **`thesis`** (the
third-level skill: `framing` + `synthesis`, per §3.1). The `thesis` skill is
optional and only used by thesis repos.

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

### 3.6 Progress tracking (cross-cutting)

Progress is followed at **every** level, but it is not a fourth level and not a
claim-manipulating skill — it is a pure **reporting** function that respects the
firewall (it adjudicates nothing; it reads existing state).

- **Status lives in the artifact.** Every hypothesis / paper / thesis artifact
  carries a small **status block in its markdown frontmatter** (verdict /
  readiness / coverage + `last-updated`). Status is versioned with the thing it
  describes — the git-native "living document" pattern — so there is no separate
  progress file to drift out of sync.
- **One cross-cutting `progress` skill**, verbs `status` and `dashboard`.
  `status <level> [id]` reads and rolls up; `dashboard` regenerates a
  `dashboard.md` that is a **pure projection** of the frontmatter (never
  hand-edited).
- **Roll-up is semantic, not arithmetic.** Parent status is a function of
  children's states + the level's gate criteria, surfaced as **coverage and
  blockers**, never a percentage. Hypothesis→paper: "all hypotheses resolved AND
  the claim is written." Paper→thesis: "all aims covered by ≥1 paper AND the
  kappa states the through-line." A single refuted load-bearing hypothesis can
  block a paper; averaging would hide that.
- **Definition of done per level** (the Stage-Gate framing our resolve gates
  already embody): hypothesis = **resolved** (has a verdict backed by recorded
  evidence); paper = **done** (constituent hypotheses resolved *and*
  submission-ready); thesis = **defensible** (aims covered *and* kappa
  through-line stated).
- **Anti-Goodhart is a hard principle, not a nicety.** The tool surfaces state,
  gaps, and staleness — never a productivity score. **A refuted hypothesis reads
  as done/green, not failed/red** (verdict and readiness are distinct axes). It
  does *not* count words, papers, commits, %-complete on unresolved work, or a
  hypothesis "success rate." Rationale: Goodhart's / Campbell's law, and the
  DORA / Leiden Manifesto principle that metrics support — never replace —
  qualitative judgment. This is documented so it cannot be quietly "improved"
  into a score later.

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
│   ├── thesis/SKILL.md                   # framing | synthesis (optional, top level)
│   ├── literature/SKILL.md               # scout | position
│   ├── dataset/SKILL.md                  # init/register/fetch/verify/mirror/audit
│   └── progress/SKILL.md                 # status | dashboard (cross-cutting, read-only)
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
plus the five generated this session — **citation scouting**, **related-works
synthesis**, **dataset-management standards**, **dataset tooling / mirror
architecture**, and **thesis-by-publication & progress tracking**. These are the
evidentiary base for the sub-specs and must be persisted, not left in
conversation.

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
│   ├── thesis/                           # OPTIONAL — only in a thesis repo
│   │   ├── kappa/                         # framing chapter (aims, narrative, per-paper contribution)
│   │   ├── aims.md                        # the through-line + chapter↔paper map
│   │   └── milestones.yml                 # configurable program gates (candidacy, submission, defense)
│   ├── dashboard.md                      # GENERATED projection of status frontmatter (never hand-edited)
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

1. **Lifecycle & pipeline skills + rigor kit + progress** — the three-level
   mirror (incl. the `thesis` level and the kappa/defensibility gate), the five
   pipeline skills, staged-doc templates, firewall, flywheels, the cross-cutting
   `progress` skill (status frontmatter + generated dashboard + anti-Goodhart
   principle), and the rigor kit. *(Largely the current
   `2026-07-15-hypothesis-*` / `paper-*` specs, refactored to the contracts,
   plus thesis + progress.)*
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
- The methodology digests (this session's five + #128's four) → become
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
- **Thesis milestone schema** — the shape of `milestones.yml` (institution
  gates are time-boxed and vary); keep configurable, resolve in sub-spec 1.

None of these block writing the sub-specs.

### Deferred (future work, out of current scope)

- **Cross-repo thesis aggregation** — pulling papers that live in a separate
  repo (e.g. company work) into a thesis roll-up. Deliberately excluded now
  (§1). Capture as a self-contained GitHub issue when this spec is finalized.
