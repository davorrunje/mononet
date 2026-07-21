# Issue Style

The standard format for follow-up / deferred-work issues in this repo. Every
issue is **self-contained**: completable from the repository content plus the
issue text alone.

## Title

`<area>: <concise imperative>`

- `<area>` mirrors the commit-scope vocabulary: `bench`, `docs`, `torch`, `jax`,
  `keras`, `core`, `ci`, `build`, etc.
- Imperative and specific: `bench: parallelize + resume the Curve-B re-eval`,
  not `Curve B slow`.
- No trailing period.

## Body

Use these sections, in this order. Omit a section only when it genuinely does
not apply (keep **Context**, **Goal**, **Where**, and **Acceptance criteria**
always).

```markdown
## Context

Why this exists and where it came from. Written for someone with no prior
context: one short paragraph of background, then the trigger — link the spec,
docs page, merged PR, or commit that deferred it (by number/path, e.g. #117,
`docs/superpowers/specs/....md`). State what currently happens.

## Goal

What "done" looks like, in one or two sentences. The problem to fix or the
capability to add.

## Where

The exact repository anchors: file paths, function/class names, and line
references where the work lands or where the relevant code lives. Prefer
`path/to/file.py:function_name` — clickable and stable across line drift.

## Proposed approach   (optional)

A sketch of how, if there's a sensible one — enough to save the next person the
rediscovery, without over-prescribing. Note trade-offs or a cheaper/costlier
variant if relevant.

## Acceptance criteria

A checklist of verifiable conditions:

- [ ] concrete, testable outcomes
- [ ] tests / docs updated where the repo conventions require it
- [ ] any command to reproduce or verify

## References

Links to the committed artifacts: spec/plan paths, PR/issue numbers, docs pages,
relevant commits.
```

## Labels

Pick from the repo's labels (`gh label list`). Always add **`follow-up`**
(create it once if missing — see SKILL.md). Add the type label that fits:

- `enhancement` — new capability / speedup / feature.
- `bug` — something is wrong.
- `documentation` — docs-only work.

## Closing convention

A closed issue records **how** it was resolved (see SKILL.md § Closing):

- Resolved by a merged PR → the PR body's `Closes #NN` closes and links it
  (nothing else needed). This is the mirror of the [`create-pr`](../create-pr/STYLE.md)
  standard — keep the two in sync.
- Resolved otherwise, or superseded → `gh issue close <N> --comment "Resolved in
  <ref> — <one line>."` (add `--reason "not planned"` for won't-do / superseded).
- Never close silently; never close a checklist/umbrella issue with open,
  un-re-filed boxes.

## Self-containment check

Before creating, confirm every answer is "yes":

- Could someone who never saw this session start the work from the issue + a
  fresh clone?
- Does every reference point at **committed** content (paths, symbols, merged
  PRs) — nothing ephemeral ("the branch we were on", "the run earlier", "as
  discussed")?
- Are the file/function anchors named explicitly, not gestured at?
- Are the acceptance criteria verifiable without insider knowledge?

If any is "no", the issue is not ready — add the missing context.

## Scope

One deliverable per issue. Split multi-part follow-ups into separate issues that
can each be completed and closed independently.
