---
name: create-issue
description: Use when a follow-up, deferred task, or known problem is identified that will not be fixed now — turn it into a self-contained GitHub issue that a future session (with only the repo + the issue, not this conversation) can complete. Also defines how issues are CLOSED. Standard format lives in STYLE.md.
---

# Create Issue

When work is deferred, a follow-up is noted, or a problem is found that won't be
fixed in the current change, **record it as a GitHub issue** — not only in a
spec, a PR comment, a code `TODO`, or conversational memory.

**Core requirement: the issue must be self-contained.** Assume whoever picks it
up has *only* two things: the repository content and the issue text. They do not
have this conversation, the PR discussion, or any of your current context. If
the issue cannot be completed from those two inputs alone, it is not done.

## When to use

Invoke this whenever you would otherwise leave a loose end:

- "Follow-up:", "deferred", "out of scope", "later", "as a follow-up".
- A bug or limitation you're deliberately not fixing in this change.
- A spec's "Follow-ups" section — each entry becomes an issue.
- A review finding downgraded to "not now".
- A `# TODO`/`# FIXME` you're tempted to leave in the code.

If you catch yourself writing "we should eventually…" — stop and create the issue.

This skill also defines the **closing** standard (below); its `Closes #NN` half
lives in the paired [`create-pr`](../create-pr/SKILL.md) skill — keep the two in
sync.

## Process (opening)

1. **Gather repo-grounded context.** Before writing, look up the concrete
   anchors the issue needs: exact file paths, function/class names, line
   references, the spec or docs page it relates to, the PR/commit where it was
   deferred. The issue cites *committed* artifacts only (paths, symbols, merged
   PRs) — never "the branch we were on" or "the run from earlier".
2. **Write title + body per [STYLE.md](STYLE.md).** Follow that template
   exactly: title convention, the required body sections, and the labels.
3. **Run the self-containment check** (bottom of STYLE.md). If any answer is
   "you'd need this conversation to know that", fix the issue text.
4. **Create it** with the GitHub CLI:

   ```bash
   gh issue create --title "<title>" --body-file <path> --label "<labels>"
   ```

   Write the body to a temp file and pass `--body-file` (avoids shell-quoting
   problems with multi-line Markdown). Add every applicable label; create the
   `follow-up` label once if it does not exist:

   ```bash
   gh label list | grep -q '^follow-up' || \
     gh label create follow-up --description "Deferred work captured as a self-contained issue" --color ededed
   ```

5. **Link back.** Print the new issue URL. If the follow-up came from a spec's
   "Follow-ups" list, replace that prose entry with a link to the issue so the
   spec and the tracker don't drift.

## Closing

An issue is never closed silently — a closed issue must record **how** it was
resolved (the mirror of the `create-pr` `Closes #NN` rule). Two paths:

- **Resolved by a PR** (the common case): the PR body carries `Closes #NN` (per
  [`create-pr`](../create-pr/SKILL.md)), so merging the PR closes and links the
  issue automatically. Nothing else to do.
- **Resolved without a PR, or superseded / won't-do**: close it explicitly with a
  one-line resolution comment naming the cause — never a bare close:

  ```bash
  gh issue close <N> --comment "Resolved in <PR/commit/issue> — <one-line what changed>."
  # superseded / won't-do:
  gh issue close <N> --reason "not planned" --comment "Superseded by #<M> — <why>."
  ```

For an **umbrella/checklist issue**, close it only when every box is done (or the
remainder is re-filed as its own issue), and leave a comment mapping each item to
the PR that fixed it.

## Red flags

- "As we discussed" / "the figure we just made" / "the branch above" — the
  reader has none of that. Name the committed file/PR instead.
- A title like "fix it" or "improve X" with no body — every issue gets the full
  STYLE.md body.
- Deferring work with only a code `TODO` or a spec bullet and no issue.
- Batching many unrelated follow-ups into one issue — one issue per deliverable.
- Closing an issue with no resolution comment and no `Closes #NN` PR.
