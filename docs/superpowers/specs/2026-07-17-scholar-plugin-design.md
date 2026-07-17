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

## ⚑ Guiding principle — assistants, not researchers

**This principle overrides everything else in this document.**

The `scholar` skills are **assistants, not researchers.** They keep the accounts
of a research program, advise as a mentor, and discuss as a colleague — but they
do **not** perform independent research and do **not** make material scientific
decisions. Accountability for research is *non-delegable and attaches only to
named humans* (ICMJE, COPE, Nature, Science; the Singapore Statement and ALLEA
2023; CRediT), which is also what EU AI Act Art. 14 and "meaningful human
control" require, and what the automation-bias literature (Parasuraman & Riley
1997; Skitka et al. 1999) demands. Therefore **every material decision is the
researcher's, recorded with a named human sign-off** — is a hypothesis confirmed
or refuted, is a result real, is a paper worth publishing, what the thesis
claims, is it defensible. The skills draft, keep the accounts, and advise; the
researcher authors and decides. You cannot "run" the workflow to produce a paper
or thesis — you drive it. Its dual is the **Understanding principle** (§2.2): the
plugin will not let you advance a decision you cannot understand or defend — it
grills and teaches until you can.

Elaborated in §2.1–§2.2; fully grounded, with verified sources and enforceable
guardrails, in
[`scholar/references/agency-principle.md`](../scholar/references/agency-principle.md)
and
[`scholar/references/understanding-and-grilling.md`](../scholar/references/understanding-and-grilling.md).

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

**Grounding.** This is not a preference; it is what the field's norms require.
Accountability for research is non-delegable and attaches only to named humans —
authorship/integrity norms (ICMJE; COPE 2023; Nature; Thorp, *Science* 2023;
the Singapore Statement 2010 and ALLEA 2023; the CRediT taxonomy) and ML-venue
policies (ICML 2023, NeurIPS, ACL 2023) all say a tool cannot be an author or
bear responsibility. The "recorded human sign-off" requirement is the *tracing*
condition of meaningful human control (Santoni de Sio & van den Hoven 2018) and
the human-oversight requirement of EU AI Act Art. 14; the "halt and ask at
judgement points" rule is demanded by the automation-bias literature
(Parasuraman & Riley 1997; Skitka et al. 1999), whose research-specific form is
LLM citation fabrication. The positive framing is the augmentation tradition
(Engelbart 1962; Bush 1945): amplify judgement, do not replace it. Full digest
with verified sources and the enforceable guardrails:
[`scholar/references/agency-principle.md`](../scholar/references/agency-principle.md).

### 2.2 Understanding principle

The dual of §2.1. **Every material claim, decision, and method must be understood
by the author to the standard a good mentor or reviewer would expect.** The plugin
verifies *and builds* that understanding; it will not let a paper, thesis, or
decision advance past something the author cannot explain or defend — silently.

- **Why it must be checked, not assumed.** Self-assessed understanding is
  unreliable until probed — the *illusion of explanatory depth* (Rozenblit & Keil
  2002): we rate our grasp high until asked to explain, then it collapses. Probing
  via retrieval + self-explanation (Roediger & Karpicke 2006; Chi et al. 1989)
  both *reveals* and *builds* understanding.
- **Two objects.** (1) *Your work* — claims, entailments, assumptions, cited
  works, limitations, rival explanations, falsifiers (the viva/reviewer standard).
  (2) *The methodology the skills embody* — the *why* behind the rigor kit and the
  principles (preregistration, severity, power/MDE, TOST, PRISMA, the firewall).
  Understanding must be *conceptual*, not merely procedural (Hiebert & Lefevre
  1986); following rigor rituals blindly is **cargo-cult science** (Feynman 1974;
  Gigerenzer 2004).
- **Mechanism — the `grill` skill (§3.7):** a Socratic tutor-examiner, self-invoked
  or fired automatically as a guardrail at material-decision checkpoints. On a
  gap it *teaches* (source-grounded explanations + references) and re-probes. The
  goal is growth, not producing work the author cannot understand.
- **Interlock with agency.** The guardrail *stops, surfaces the gap, offers to
  grill/teach, and records* — the human may override, but the override is logged
  (so a gap can never be passed silently). It never grades the substance of a
  novel claim or asserts an answer key; it teaches the established (methodology,
  cited works) and, for novel claims, teaches how to reason and defend.

Fully grounded, with sources and constraints, in
[`scholar/references/understanding-and-grilling.md`](../scholar/references/understanding-and-grilling.md).

### 2.3 The firewall

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

### 3.7 Grilling & understanding (cross-cutting)

The `grill` skill operationalizes §2.2 — a Socratic **tutor-examiner** whose goal
is the author's growth; companion to `progress` (both cross-cutting, both
respecting the firewall by producing evidence the human acts on).

- **Targets:** `claim | cited-work | methodology`, with stage presets —
  `hypothesis-testing` → strategy (assumptions/entailments/falsifiers/rivals);
  `paper-synthesis` → positioning (novelty, citation support); the thesis
  defensibility gate → a full **mock viva**.
- **Loop:** probe → detect gap → **teach** (explain + link references) → re-probe
  (the retrieval-practice loop that builds, not just audits, understanding).
- **Invocation:** (1) **self-invoked** on demand; (2) automatic **guardrail** at
  material-decision checkpoints (findings verdict, publish decision, thesis
  defensibility) and when a method's rationale is undemonstrated.
- **Guardrail semantics:** stop, surface, offer to grill/teach, **record**; the
  human may override with the override logged (agency, §2.1) — not a hard block.
  *(Open item: whether the thesis defensibility gate is made blocking.)*
- **Records:** an `understanding` status in the artifact frontmatter (fed to the
  `progress` roll-up) + optional grill transcript; unanswered probes and logged
  overrides are the accountability trail.
- **Constraints (§2.2):** ask, don't grade substance; teach the established
  (methodology, cited work) freely and source-grounded; never assert a novel
  claim's answer; settled-vs-contested calibration; depth by stakes; anti-Goodhart
  (reward articulable understanding, not ritual completion).
- **Mentor/reviewer personas.** The grill/advise voice offers a small set of
  **author-selectable** personas (derived from Lee × Gatfield, *not* personality
  theory): *sounding board*, *critical examiner* (the default), *directive
  editor*, and an opt-in *devil's advocate*. Chosen by three author-controllable
  levers — self-selected (default), task/stage-*suggested* (keyed to the artifact,
  overridable), and feedback-calibrated ("too harsh" / "push harder"). Autonomy
  support (SDT) is constant; only directiveness and challenge-intensity vary.
  **Matching to an *inferred personality* is forbidden** — it is unsupported (the
  learning-styles myth) and would violate agency; the rule is *match the voice to
  the task and the author's stated choice, never to an inferred personality.*
  Grounded in
  [`scholar/references/mentor-personas.md`](../scholar/references/mentor-personas.md).

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
│   ├── progress/SKILL.md                 # status | dashboard (cross-cutting, read-only)
│   └── grill/SKILL.md                    # claim|cited-work|methodology; self + guardrail (cross-cutting)
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
plus the eight generated this session — **citation scouting**, **related-works
synthesis**, **dataset-management standards**, **dataset tooling / mirror
architecture**, **thesis-by-publication & progress tracking**, the
**agency principle**, **the understanding principle & grilling**, and **mentor /
reviewer personas** (all already persisted under `scholar/references/`). These are
the evidentiary base for the sub-specs and must be persisted, not left in
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

Distributed as a **public** Claude Code plugin (git-repo marketplace install),
named **`scholar`** (ADR-0019) — for the author's own repos, company colleagues,
PhD peers, and the broader research community. The plugin must therefore be
genuinely domain-neutral and self-documenting from day one — most "users" are not
the author. A root `README.md` (staged at `docs/superpowers/scholar/README.md`)
explains purpose + reasoning and links to the decision log and reference digests.
Licensed **Apache-2.0** (ADR-0022), matching `mononet`.

## 8. Sub-spec decomposition

This meta-spec defers detail to four sub-specs (each date-prefixed under
`docs/superpowers/specs/`, migrating to the plugin repo per §9):

1. **Lifecycle & pipeline skills + rigor kit + progress + grill** — the
   three-level mirror (incl. the `thesis` level and the kappa/defensibility gate),
   the five pipeline skills, staged-doc templates, firewall, flywheels, the
   cross-cutting `progress` skill (status frontmatter + generated dashboard +
   anti-Goodhart), the cross-cutting `grill` skill (Understanding principle §2.2:
   self + guardrail invocation, teaching loop, non-grading/source-grounded stance),
   and the rigor kit (whose methods are also `grill` targets). *(Largely the
   current `2026-07-15-hypothesis-*` / `paper-*` specs, refactored to the
   contracts, plus thesis + progress + grill.)*
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
- The methodology digests (this session's eight + #128's four) → become
  `resources/references/` in the plugin. This session's eight are already staged
  under `docs/superpowers/scholar/references/`.
- `mononet` becomes the reference **consumer**: it runs `research-init adopt`
  against itself once the plugin exists.

Sequencing: this meta-spec first (draft PR), then the four sub-specs, then
implementation plans, then the plugin repo is created and `mononet` adopts it.

## 10. Open items to confirm before implementation

- **Plugin repo name / visibility** — *resolved:* public, named `scholar`
  (ADR-0019).
- **Bibliography format** — *resolved:* CSL-JSON as source of truth, BibTeX
  exported on demand (ADR-0020).
- **Plugin license** — *resolved:* Apache-2.0 (ADR-0022).
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
