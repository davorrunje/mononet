# Paper-exploration skill

**Status:** design
**Date:** 2026-07-15
**Author:** Davor Runje
**Scope:** a new Claude Code skill for choosing the next application/follow-up paper, the
portfolio backlog it maintains, and the papers registry the other research-workflow skills
resolve against. Repo tooling/process — no impact on the `mononet` wheel.

## Problem

`mononet` has a main paper (the CMNN extension) and a growing set of application papers
built on it under `applications/` (e.g. `applications/pinn`, PR #116). Deciding *which
application to pursue next* — where the main paper's core mechanism buys a real, by-
construction advantage in a new domain — is today ad-hoc, and there is no durable record of
the candidate papers considered, why one was chosen, or how the portfolio balances safe
extensions against exploratory bets. We want a **portfolio-level** skill that reads the
main paper and proposes, prioritizes, and scaffolds the next follow-up papers.

## Relationship to the other skills

One of the four composable skills in the **research-workflow family** — the full matrix,
the two nested flywheels, and the exploration/testing/synthesis firewall live in
[docs/research/README.md](../../research/README.md):

- `hypothesis-exploration` — generate candidate hypotheses · `hypothesis-testing` — test one to a verdict
- `paper-exploration` — propose application papers · `paper-synthesis` — assemble a paper spine

**This skill, `paper-exploration`,** is portfolio-level — `hypothesis-exploration` lifted one level: it proposes whole application papers and scaffolds a new project on acceptance. It depends on a pluggable **experiment backend** for executing experiments and producing
reproducible evidence — by default the benchmark orchestration spec
([2026-07-15-benchmark-experiment-orchestration-design.md](2026-07-15-benchmark-experiment-orchestration-design.md)),
bound per project in the registry. It is grounded in
[paper-exploration-references.md](../../research/paper-exploration-references.md).

## The papers registry

This skill owns `docs/research/papers.md` — the registry every research-workflow skill
resolves a `paper-id` against:

```
| paper-id | root                | kind        | backend       | status | title |
|----------|---------------------|-------------|---------------|--------|-------|
| main     | docs/research/main/ | main        | mononet-bench | active | Constrained Monotonic NNs (extension) |
| pinn     | applications/pinn/  | application | mononet-bench | draft  | Structure-Preserving PINNs (migrating to the default backend) |
```

`status` ∈ `planned | active | draft | submitted | published | shelved`. Exactly one paper
is `kind: main`. Application papers point at their `applications/<slug>/` root. `backend`
names the experiment backend that runs the paper's experiments (default `mononet-bench`,
the orchestration skill; the contract is in [the family overview](../../research/README.md)).

## Design principles

1. **Portfolio symmetry.** Reuse the `hypothesis-exploration` pipeline and firewall
   verbatim, one level up — a paper candidate is exploratory until it is scaffolded and its
   own strategy/experiments confirm it.
2. **Mechanism as hard core.** The main paper's central mechanism is the research
   programme's *hard core* (Lakatos 1970); application papers are protective-belt
   extensions. Prefer **progressive** proposals (predict a new admissible result in a new
   domain) over **degenerating** ones (restatements).
3. **Balance the slate.** Never return only safe near-core extensions; reserve room for at
   least one exploratory, farther-from-core bet (March 1991).
4. **No idea lost, no file drawer.** The portfolio backlog durably holds parked and
   dropped paper ideas with reasons.

## Layout & artifacts

```
.claude/skills/paper-exploration/
├── SKILL.md          # frontmatter + "When to use" + verbs (park/generate/rank/promote/drop)
└── BACKLOG.md        # portfolio-backlog entry format (as create-issue has STYLE.md)

docs/research/
├── papers.md              # the registry (above)
└── portfolio-backlog.md   # the paper-candidate queue
```

A **portfolio-backlog entry**:
```
- id: <slug>
  title: <one line>
  target_domain: <where the mechanism is brokered to>
  lens: mechanism-transfer | limitation-driven | result-driven
  feasibility: <low|med|high + note>     # main-paper mechanism maturity × domain readiness
  interest: <low|med|high + note>        # impact × venue fit
  posture: exploit | explore             # near-core vs farther-from-core (March 1991)
  status: parked | ranked | scaffolded | dropped
  link: <applications/<slug>/ if scaffolded>
  note: <optional; drop reason if dropped>
```

## The origin-agnostic pipeline

**origin (you propose, or generate) → dedup (vs registry + backlog) → prioritize (advisory)
→ human gate → scaffold as a new application project**

Verbs (mirroring `hypothesis-exploration`):
- **park** — stash a paper idea in `portfolio-backlog.md`.
- **generate** — read the main paper and propose candidates via the three lenses (below);
  mechanism chosen at invocation (lean checklist / fan-out+tournament / single-pass).
- **rank** — score candidates on the prioritization axes.
- **promote** — accept one or several; **scaffold** each (below).
- **drop** — reject, keep the reason.

## Generation lenses

Grounded in `paper-exploration-references.md`:

- **mechanism-transfer** (primary) — broker the main paper's core mechanism into a domain
  where it is novel but proven in the source (Hargadon & Sutton 1997), by structural
  analogy (Gentner 1983), preferring targets in the mechanism's **adjacent possible**
  (Kauffman 2000). The PINN paper is the exemplar: hard monotonicity → PDE admissibility.
- **limitation-driven** — turn a stated limitation/scope boundary of the main paper into an
  application a different domain's constraints make tractable (progressive problemshift,
  Lakatos 1970).
- **result-driven** — take a headline result/verdict and ask which domain most needs
  exactly that guarantee (use-inspired, Stokes 1997).

## Prioritization

Advisory; candidates are positioned, not flat-listed:

- **feasibility × interest** scatter (Alon 2009): feasibility ≈ mechanism maturity + domain
  data/tooling readiness; interest ≈ impact + venue fit. Pick from the high–high region.
- **importance × attackability** gate (Hamming 1986): promote only when the mechanism gives
  a concrete line of attack (a method, dataset, or domain hook), not just an important
  target.
- **fit** (Stokes 1997): prefer applications that advance *both* mechanism understanding and
  domain use (Pasteur's quadrant).
- **portfolio balance** (March 1991): the returned slate mixes exploit (near-core) and
  explore (farther-from-core); each candidate carries its posture.
- **reachability** discount (Butler 2008; Kauffman 2000): down-rank transfers with no
  realistic path across the "valley of death" or outside the adjacent possible.
- **progressive-over-degenerating** tie-breaker (Lakatos 1970).

## Accept → scaffold

On `promote`, the skill frames the chosen paper as answers to the **Heilmeier Catechism**
(DARPA) — Q1–Q4 fix the contribution, Q5–Q7 feasibility/cost/timeline, Q8 the success
"exams" — then scaffolds:

1. Create the project root `applications/<slug>/` with the standard research-workflow
   layout (`hypotheses/`, `backlog.md`, `paper/{outline.md, ledger.md, sections/}`) — the
   compendium layout PR #116 established, plus the record layer.
2. Register it in `docs/research/papers.md` (`kind: application`, `status: planned`,
   `backend:` the default `mononet-bench` unless the paper needs a different harness).
3. Seed the project's first `strategy.md` (or the paper's `outline.md`) from the Heilmeier
   answers, and mark the backlog entry `scaffolded` with the link.
4. Hand off to `superpowers:brainstorming` to design the application paper (its code lives
   under `applications/<slug>/`).

## Relationship & the portfolio flywheel

The skill closes the *outer* loop, above the per-paper flywheel:

**main paper matures → `paper-exploration` proposes applications → scaffold → the per-paper
family runs inside each application → application results feed back as evidence the main
paper can cite → new applications become reachable.**

Application papers read the main paper read-only (their claims may cite main-paper results
by `label`); the main paper does not depend on the applications.

## Skill process (resumable, verb-dispatched)

1. **park** — write a portfolio-backlog entry.
2. **generate** — read the main paper corpus; apply the three lenses; dedup vs registry +
   backlog; position candidates on the axes; write the ranked slate to the backlog.
3. **rank** — re-score after the main paper's verdicts change.
4. **promote** — Heilmeier-frame + scaffold + register + hand off (above).
5. **drop** — mark dropped with a reason.

## Grounding & dogfood

- Every lens and prioritization axis cites `paper-exploration-references.md`.
- **Dogfood**: retro-fit the registry with `main` + `pinn`, then run `generate` against the
  main paper. The PINN paper should be *reconstructible* as a mechanism-transfer candidate
  (hard monotonicity → admissible PDE solutions), validating that the lenses and rubric
  reproduce a paper the project already judged worth writing.

## Testing / validation

- **Dogfood** the verbs: `generate` writes a ranked portfolio slate; `promote` scaffolds a
  new `applications/<slug>/` project, registers it in `papers.md`, seeds its strategy, and
  links the backlog entry; `park`/`drop` behave.
- **Registry invariant**: exactly one `kind: main`; every `scaffolded` backlog entry has a
  registry row and an `applications/<slug>/` root, and vice versa.
- **Dedup**: a candidate matching an existing/registered/parked paper is flagged, not
  re-proposed.

## Out of scope / YAGNI

- Cross-repo papers (e.g. an application spun out to its own repo, as the Lean proofs were)
  — the registry could point at a submodule/URL later; not now.
- Automated venue/deadline tracking — the registry `status` is maintained by hand.
- Any change to the `mononet` wheel.

## Follow-ups (to become GitHub issues)

- Registry support for cross-repo application papers (root as a URL/submodule) when an
  application is spun out of the monorepo.
- A portfolio dashboard (render the registry + backlog into a single status view), once
  several application papers exist.
- Retire/`shelved` workflow for application papers that stall in the valley of death
  (Butler 2008), so the portfolio reflects reality.
