# Thesis-by-Publication & Multi-Level Progress Tracking — digest

**Date:** 2026-07-17 · **For:** `scholar` sub-spec 1 (thesis level + `progress` skill). · **Status:** verified-source digest; migrates to the plugin's `resources/references/`.

## Part A — Thesis-by-publication & PhD milestones

- **The model.** Traditional **monograph** vs **thesis by publication** / cumulative / "stapler" (papers bound by a framing text). The cumulative model *is* the papers→thesis nesting → treat **monograph as the degenerate case** (one "paper" = the whole thesis), not a separate mode.
- **N-paper convention.** No universal N; most-cited concrete figure is Swedish medicine **3–5 articles** (Karolinska). The binding rule is **scope**, not count ("total scope corresponding to four years full-time"). → make paper-count a soft, configurable target, never a hard gate.
- **The framing chapter ("kappa").** Swedish for "overcoat" — the synthesizing chapter turning a stack of papers into one thesis; **more than concatenated introductions** (Stockholm Univ *Kappa Guidelines*; Karolinska; Lund). Components (reusable as the thesis-artifact schema): introduction/background; unifying narrative/aims; extensive independent related work; per-paper summary + **explicit statement of the candidate's contribution** to co-authored papers; comprehensive concluding discussion; future work. **Introduces no new findings** — it recontextualizes.
- **Coherence / "original contribution to knowledge."** The examinable through-line (Oxford MPLS integrated-thesis guidance; QUT; UNE): examiners judge (a) coherence of the whole, (b) how each paper contributes to the overarching project, (c) that it reads as a single coherent document. → thesis→papers roll-up target is **narrative coverage of the aims + kappa through-line**, not "N papers done." Surface **gaps in the argument**, not counts.
- **Milestones/gates** (UC Davis, Berkeley, Michigan, McGill handbooks): proposal/prospectus → committee formation → qualifying/comprehensive exam → **advancement to candidacy** (pivotal) → annual progress review → submission → defense/viva. Track per milestone: status (not-started/scheduled/passed), date, next binding deadline. Institution-specific + time-based → a small **configurable milestone list** at the thesis level, not hard-coded.

## Part B — Multi-level progress tracking (core design input)

**"Definition of done" per level (citable frames):**
- **Stage-Gate / phase-gate** (Cooper) — closest match; each stage ends at a **gate** with explicit go/kill/hold criteria + required deliverables. Maps to our resolve gates: hypothesis (evidence → verdict), paper (submission-ready), thesis (defensible). Borrow "gate = exit criteria + decision" — more honest than % burndown.
- **OKRs** — the hierarchical roll-up pattern (parent progress from children); adapt cautiously (research outcomes are binary-ish, not KPI-measurable).
- **Avoid** burndown/velocity — they assume known decomposable scope, which pre-verdict research lacks.

Suggested "done": **hypothesis = resolved** (verdict backed by recorded evidence; pending = no verdict); **paper = done** (constituent hypotheses resolved AND claim written/submission-ready; drafted→under-review→published as sub-states); **thesis = defensible** (papers cover all aims AND kappa states a coherent through-line + original contribution).

**Hierarchical roll-up — semantic, not arithmetic:**
- Hypothesis verdicts → **paper readiness**: not "80% at 4/5 resolved"; a single refuted **load-bearing** hypothesis can invalidate the claim. Roll up as **coverage + blocker** status, not an average.
- Paper states → **thesis narrative coverage**: map each paper to the aims; thesis status = "uncovered aims?" + "through-line stated?" (exactly what examiners check).
- **Keep verdict and readiness as distinct axes** — a **refuted hypothesis is done, successful science**; conflating refuted with failed/incomplete is a modeling error.

**Meaningful vs perverse metrics:**
- **Goodhart's law** / **Campbell's law** — a self-tracking PhD tool is *especially* Goodhart-prone (user sets and games the metric).
- **DORA** (2012) and the **Leiden Manifesto** (Hicks & Wouters et al. 2015) — metrics *support*, never *replace*, qualitative judgment.
- **Design rule — surface, don't score.** Show state + gaps (unresolved hypotheses, uncovered aims, overdue milestones, staleness/last-touched), **not** an aggregate productivity score.
- **Do NOT count:** paper/word counts, citation/impact proxies, commit counts, %-complete on unresolved research, hypothesis "success rate" (punishes refutation). **Refuted = green/done, not red.**

**Git-native markdown practice:** plain-text markdown lab notebooks under git (RSC *Digital Discovery* "GitHub as an open ELN," DOI:10.1039/D3DD00032J; arXiv:2408.09344; `mdlabbook`); **living documents** (re-generated/edited in place). A good markdown dashboard contains: current state per artifact (status in frontmatter), open questions/next actions, blockers, last-updated, links between artifacts (hypothesis→paper→thesis-aim). Kanban states travel well (`pending/resolved`).

**Design recommendation (crisp):**
1. **`status` = cross-cutting read verb, not a fourth level.** Each artifact owns a small status block in its markdown **frontmatter** (verdict/readiness/coverage + last-updated). `status <level> [id]` reads + rolls up — never invents a separate progress artifact to drift.
2. **One thin `dashboard` view** aggregating frontmatter into a **generated** overview markdown (never hand-edited) — a projection of the source of truth, nothing to game or sync.
3. **Roll-up = semantic function of children + explicit gate criteria**, surfaced as coverage + blockers, never %.
4. **Bake in the anti-Goodhart stance** (cite Goodhart/Leiden/DORA); refuted = done; document it so it isn't quietly turned into a score.

## Sources
Thesis-as-collection (Wikipedia overview) · Paltridge & Starfield, *Thesis and Dissertation Writing in a Second Language*, Routledge 2e 2020 (ISBN 9781138048706) · Stockholm Univ Kappa Guidelines · Karolinska "content of the compilation thesis" · Oxford MPLS integrated-thesis guidance · QUT / UNE thesis-by-publication guidelines · PhD milestone handbooks (UC Davis/Berkeley/Michigan/McGill) · Stage-Gate (stage-gate.com; Planview) · OKRs (Google re:Work; Atlassian) · Goodhart's law (FORRT) · Leiden Manifesto (Hicks et al. 2015) + DORA (2012) · RSC Digital Discovery (D3DD00032J); arXiv:2408.09344; mdlabbook; living-documents.

## Flags
"3–5 papers" is medical-Sweden-specific; binding norm is scope, not count. Kappa specifics from Nordic university guidance (authoritative), not the Paltridge book text. Milestone lists institution-specific — deadlines illustrative. Stage-Gate/OKR are industry PM material — adaptation flagged.
