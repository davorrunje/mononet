---
name: create-pr
description: Use when changes are ready to land — branch off main, go green, commit with this repo's attribution, and open a PR. Encodes the branch/checks/commit/body ritual and the Closes #NN convention. Never commit to main. Templates live in STYLE.md; the full review workflow lives in PULL_REQUEST_GUIDE.md.
---

# Create PR

Every change lands through a pull request. **`main` is protected — never commit
to it** (see CLAUDE.md § Commits). This skill encodes the repo's PR ritual so
branch, checks, attribution, and body are consistent without rediscovery. Its
`Closes #NN` half is the mirror of the paired
[`create-issue`](../create-issue/SKILL.md) close standard — keep the two in sync.

For the detailed mechanics beyond opening (description-file workflow, replying to
review comments via REST, resolving review threads via GraphQL), follow
[`PULL_REQUEST_GUIDE.md`](../../../PULL_REQUEST_GUIDE.md).

## Process

1. **Branch off `main`.** Name it `<area>/<slug>` using the same `<area>`
   vocabulary as `create-issue` (`bench`, `docs`, `torch`, `jax`, `keras`,
   `core`, `ci`, `build`, `feat`, `fix`, `chore`, …):

   ```bash
   git fetch origin && git switch -c <area>/<slug> origin/main
   ```

   Do the work on that branch.

2. **Go green before opening** — never open a red PR. From the repo root:

   ```bash
   uv run ruff check --exit-non-zero-on-fix
   uv run ruff format --check
   uv run mypy
   uv run pytest              # add MONONET_TEST_BACKEND=torch|jax|keras for equivalence tests
   ```

   `pytest` enforces the 100% coverage gate — it must pass. `pre-commit run
   --all-files` runs the same hooks in one shot.

3. **Commit with this repo's attribution.** Disable GPG signing (SSH signing is
   unavailable in these sessions) and end the message with the Claude trailer:

   ```bash
   git commit --no-gpg-sign -F <message-file>
   ```

   The commit-message shape (subject + body + trailer) is in [STYLE.md](STYLE.md).
   Commit proactively at sensible checkpoints (CLAUDE.md § Commits).

4. **Push and open the PR** against `main`:

   ```bash
   git push -u origin <area>/<slug>
   gh pr create --base main --title "<title>" --body-file <body-file>
   ```

   The PR body follows STYLE.md: concise what/why, **`Closes #NN`** for every
   issue it resolves (this is what closes them — see `create-issue` § Closing),
   and the Claude Code footer. Pass the body via `--body-file`.

5. **Report the PR URL.** Do **not** merge — merging is the maintainer's action.

## Hard rules

- **Never commit to `main`** (protected). Branch first, always.
- **Base on `main`, not on another feature branch.** Deleting a base branch on
  merge *closes* the dependent PR. Stacked work waits for its base to merge, or is
  restructured onto `main`.
- **Green before opening.** Don't outsource discovering a red build to CI.
- **Attribution is not optional** — the `Co-Authored-By` trailer +
  `--no-gpg-sign` on every commit.

## Red flags

- `git commit` on `main`, or a branch whose base is a soon-to-be-deleted branch.
- Opening a PR before the checks pass locally.
- A commit missing the `Co-Authored-By` trailer.
- A PR body with no `Closes #NN` when it resolves a tracked issue, or missing the
  Claude Code footer.
