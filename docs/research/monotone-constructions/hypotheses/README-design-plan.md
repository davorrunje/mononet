<!-- Not a template — a pointer. design.md and plan.md are DELEGATED. -->

# design.md / plan.md — delegated to the engineering backend

`honest-scholar` ships **no template** for the design/plan stage of a hypothesis. This
stage is **engineering**, and engineering is delegated (ADR-0002, refined by
ADR-0023) to the **bound engineering backend** via the engineering-delegation
contract (`../../contracts/engineering.md`; capabilities `design` / `plan` /
`implement`, bound per project in `.honest-scholar/config.yml` as `engineering_backend:`).
The science is settled in `strategy.md`, then the how-to-build-it is handed to
the engineering backend.

## What to do

After `strategy.md` is written **and examined**:

1. Invoke the engineering backend's **`design`** capability to explore and settle
   the experiment design.
2. Invoke the engineering backend's **`plan`** capability to produce the stepwise
   plan.
3. Store the resulting `design.md` and `plan.md` **in this hypothesis folder**
   (`docs/research/<paper>/hypotheses/<YYYY-MM-DD-slug>/`), alongside the other
   staged docs.
4. Execute the plan (the backend's **`implement`** capability); execution produces
   the backend **run-refs** that `findings.md` cites.

## Guardrails

- **Science before engineering.** No design/plan until `strategy.md` exists and its
  examination gaps are resolved.
- `honest-scholar` does not design experiments, write plans, or implement — it composes
  with the bound engineering backend for all of that.

The paper-level analog (`outline.md` / `plan.md`) is delegated the same way from
`paper-synthesis`. See `../../../docs/design/01-lifecycle.md` §2 and the
`hypothesis-testing` skill (`../../../skills/hypothesis-testing/SKILL.md`).
