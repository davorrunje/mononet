# PR Style

The concrete templates for landing changes in this repo. See
[SKILL.md](SKILL.md) for the process and hard rules, and
[`PULL_REQUEST_GUIDE.md`](../../../PULL_REQUEST_GUIDE.md) for the full workflow.

## Branch name

`<area>/<slug>`

- `<area>` mirrors the commit-scope / issue vocabulary: `bench`, `docs`, `torch`,
  `jax`, `keras`, `core`, `ci`, `build`, `feat`, `fix`, `chore`, `refactor`.
- `<slug>` is a short kebab-case description: `feat/alternate-construction`,
  `docs/claude-md`, `chore/honest-scholar-integration`.

## Commit message

```
<type>(<scope>): <concise imperative subject>

<body — what changed and why, wrapped ~72 cols. Reference issues/PRs/specs by
number/path. State the behavioral change, not just the mechanics.>

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```

- `<type>` is Conventional-Commits style: `feat`, `fix`, `docs`, `test`, `ci`,
  `build`, `chore`, `refactor`.
- `<scope>` is the `<area>` (e.g. `bench`, `torch`, `core`, `docs`).
- Commit with `--no-gpg-sign`: `git commit --no-gpg-sign -F <file>`.

## PR title

`<type>(<scope>): <concise summary>` — same shape as the commit subject
(e.g. `feat(bench): HP-search sensitivity curves`).

## PR body

```markdown
## What

One or two sentences: the change and why it matters (the behavioral / user-visible
effect, not a file list).

## Details   (optional)

Bullets for the notable pieces — the non-obvious decisions, trade-offs, or a
"deliberately not changed" note. Reference specs/issues by number/path.

Closes #NN            <!-- one per issue this PR resolves; this is what closes them -->

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

- **`Closes #NN`** for every issue the PR resolves — merging then closes and
  links them (the mirror of `create-issue` § Closing; keep in sync). One
  `Closes #NN` line per issue.
- Always end with the `🤖 Generated with [Claude Code](https://claude.com/claude-code)`
  footer.
- Pass the body via `--body-file` to avoid multi-line shell-quoting problems (see
  PULL_REQUEST_GUIDE.md § Description File First).

## Commands

```bash
git fetch origin && git switch -c <area>/<slug> origin/main
# … work + go green (see SKILL.md) …
git commit --no-gpg-sign -F msg.txt
git push -u origin <area>/<slug>
gh pr create --base main --title "<title>" --body-file body.md
```

Base is **always `main`** (never a soon-to-be-deleted feature branch — see
SKILL.md hard rules).
