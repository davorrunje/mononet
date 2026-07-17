# Research workflow

This directory is the project's **research record** — hypotheses, their tests, and the
papers built from them — maintained by a family of four composable Claude Code skills over
a shared, multi-paper corpus. These are working records, not published documentation, so
`docs/research/` is excluded from the Sphinx site build.

## The four skills (object × action)

|  | **explore** — *generate* | **resolve** |
|---|---|---|
| **hypothesis** | [`hypothesis-exploration`](../superpowers/specs/2026-07-15-hypothesis-exploration-skill-design.md) — corpus → candidate hypotheses | [`hypothesis-testing`](../superpowers/specs/2026-07-15-hypothesis-testing-skill-design.md) — one hypothesis, claim → verdict |
| **paper** | [`paper-exploration`](../superpowers/specs/2026-07-15-paper-exploration-skill-design.md) — main paper → application-paper candidates | [`paper-synthesis`](../superpowers/specs/2026-07-15-paper-synthesis-skill-design.md) — verdicts → paper spine + gaps |

Each skill is grounded 1:1 in a `<skill>-references.md` in this directory
([hypothesis-testing](hypothesis-testing-references.md) ·
[hypothesis-exploration](hypothesis-exploration-references.md) ·
[paper-synthesis](paper-synthesis-references.md) ·
[paper-exploration](paper-exploration-references.md)).

## Multi-paper layout

- **Registry** [`papers.md`](papers.md) maps `paper-id → root · kind (main|application) · status`; owned by `paper-exploration`.
- **Main paper** root `docs/research/main/`; **application papers** at `applications/<slug>/` (co-located with code + results + `paper/`, per PR #116).
- Every project root has the uniform layout: `hypotheses/<date-slug>/` · `backlog.md` · `paper/{outline.md, ledger.md, sections/}`.
- Portfolio level: `portfolio-backlog.md` (application-paper candidates).
- Application papers read the **main paper read-only**; the main paper never depends on applications.

## Two nested flywheels

- **Per paper:** paper `gaps` → `hypothesis-exploration` → `hypothesis-testing` → verdicts → `paper-synthesis` → new gaps.
- **Portfolio:** main paper matures → `paper-exploration` proposes applications → scaffold → the per-paper family runs inside each → application results feed back as evidence.

## The firewall

Exploration **proposes**, testing **disposes**, synthesis **reports**. A generated
hypothesis or paper is *exploratory* until confirmed by a fresh, pre-registered test; no
skill confirms its own output. This is why generation and confirmation are separate skills.

## Experiment backend (pluggable)

The four skills never talk to a specific benchmark harness directly — they depend on an
abstract **experiment backend** with four capabilities:

| capability | what a skill asks of it |
|---|---|
| **run** | execute the experiments a hypothesis's design specifies |
| **evidence** | a durable result carrying a *run-ref* (stable id) + a *provenance stamp* (code + data version) sufficient to reproduce |
| **tables** | inject / regenerate result tables in a document |
| **is-current** | report whether a cited result is still valid for the current code + data |

The backend is bound **per project** by the `backend:` field in [`papers.md`](papers.md).
The **default** is the benchmark orchestration skill
([2026-07-15-benchmark-experiment-orchestration-design.md](../superpowers/specs/2026-07-15-benchmark-experiment-orchestration-design.md)),
which binds run→`mononet-bench run/reconcile`, evidence→run-hash + `.provenance.json`,
tables→`render` managed blocks, is-current→`render --check`. Depending on the *contract*
rather than the tool keeps the testing machinery **hot-swappable** — a future paper with
different execution needs sets a different `backend:` without touching any skill. (The PINN
application paper is currently a standalone draft; it adopts this default backend once the
infra lands.)

Every quantitative claim thus traces through the backend to committed results. The four
skills are grounded in 56 verified primary sources across the four reference docs.
