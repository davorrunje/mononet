# About-section Tidy-up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `docs/about/contributing.md` mirror the canonical `CONTRIBUTING.md` via `{include}`, move the maintainer runbook `docs/releasing.md` under `about/`, and add a CHANGELOG-tag-link verification line to the release flow.

**Architecture:** Docs-only. One cohesive change validated by the strict Sphinx build (`sphinx-build -W`, with `nitpicky=True` already enforced). The `CONTRIBUTING.md`→`releasing.md` link depends on the move, so the whole thing is one task with the strict build as the forcing function.

**Tech Stack:** Sphinx + myst-nb, MyST `{include}`, `sphinx-autodoc2`.

**Spec:** [docs/superpowers/specs/2026-07-12-docs-about-tidy-design.md](../specs/2026-07-12-docs-about-tidy-design.md)

## Global Constraints

- **Branch:** `spec/docs-about-tidy` (already checked out). Never commit to `main`.
- **Commit signing is broken in this container** — always `git commit --no-gpg-sign`.
- **Docs-only** — no `mononet/**` code change, no CI-workflow change.
- The strict build already enforces `-W` **and** `nitpicky=True`; a broken `{include}`, unresolved intra-page anchor, dangling link, or orphan/dangling toctree entry is a **fatal** build error — the acceptance gate.
- `{include}` does not rewrite relative links, so `CONTRIBUTING.md`'s repo-relative **file** links must become absolute GitHub blob URLs (`https://github.com/davorrunje/mononet/blob/main/<path>`).
- Never `git add docs/_build/`. Run via `uv run`; never run any `uv sync` variant.

---

### Task 1: Move releasing under about/, include canonical CONTRIBUTING.md

**Files:**
- Move: `docs/releasing.md` → `docs/about/releasing.md`
- Modify: `docs/index.md` (drop `releasing` from the top-level toctree)
- Modify: `docs/about/index.md` (add `releasing` to the About toctree)
- Modify: `docs/about/releasing.md` (add the #7 verification line)
- Modify: `docs/about/contributing.md` (replace with an `{include}` wrapper)
- Modify: `CONTRIBUTING.md` (absolutize repo-relative file links)

**Interfaces:**
- Consumes: nothing. Produces: no code interface — docs content only.

- [ ] **Step 1: Move the runbook and re-point both toctrees**

```bash
git mv docs/releasing.md docs/about/releasing.md
```

In `docs/index.md`, the hidden toctree currently reads:

```
```{toctree}
:hidden:

installation
guides/index
concepts/index
benchmarks/index
reference
releasing
about/index
```
```

Remove the `releasing` line so it becomes:

```
```{toctree}
:hidden:

installation
guides/index
concepts/index
benchmarks/index
reference
about/index
```
```

In `docs/about/index.md`, the toctree currently reads:

```
```{toctree}
:maxdepth: 1

license
changelog
contributing
```
```

Add `releasing` as the last entry:

```
```{toctree}
:maxdepth: 1

license
changelog
contributing
releasing
```
```

- [ ] **Step 2: Add the #7 CHANGELOG-tag-link line to the runbook**

In `docs/about/releasing.md`, the `## Per-release flow` section ends at step 5
(the paragraph about the `v*.*.*` tag triggering the Docs workflow), immediately
before `## Notes`. Add a step 6 after step 5:

```markdown
6. After the release is published, confirm the CHANGELOG version/compare footer
   links resolve — the `v<version>` release-tag link only goes live once the
   GitHub Release (and its tag) exists.
```

- [ ] **Step 3: Absolutize `CONTRIBUTING.md`'s repo-relative file links**

In `CONTRIBUTING.md`, rewrite each repo-relative markdown **file** link to an
absolute GitHub blob URL. Apply every replacement (two of them occur twice — do
all occurrences):

| Current link target | New link target |
|---|---|
| `](NOTICE.md)` | `](https://github.com/davorrunje/mononet/blob/main/NOTICE.md)` |
| `](.devcontainer/claude-plugins.txt)` | `](https://github.com/davorrunje/mononet/blob/main/.devcontainer/claude-plugins.txt)` |
| `](docs/releasing.md)` (×2) | `](https://github.com/davorrunje/mononet/blob/main/docs/about/releasing.md)` |
| `](docs/superpowers/specs/2026-05-22-myst-docstrings-design.md)` (×2) | `](https://github.com/davorrunje/mononet/blob/main/docs/superpowers/specs/2026-05-22-myst-docstrings-design.md)` |
| `](PULL_REQUEST_GUIDE.md)` | `](https://github.com/davorrunje/mononet/blob/main/PULL_REQUEST_GUIDE.md)` |
| `](SECURITY.md)` | `](https://github.com/davorrunje/mononet/blob/main/SECURITY.md)` |

Note the `docs/releasing.md` link points to the **new** `docs/about/releasing.md`
path (from Step 1). Leave the two intra-document anchor links
(`](#claude-code-plugins--sessions)`, `](#lint-format-static-analysis)`) and the
already-absolute `https://` links unchanged.

- [ ] **Step 4: Replace `docs/about/contributing.md` with an include wrapper**

Overwrite `docs/about/contributing.md` entirely with (mirrors
`docs/about/changelog.md`'s include pattern):

````markdown
# Contributing

The repository's `CONTRIBUTING.md` is authoritative — this page mirrors it.

```{include} ../../CONTRIBUTING.md
:start-line: 2
```
````

(`:start-line: 2` skips `CONTRIBUTING.md`'s own `# Contributing to mononet` H1
and its blank line so the page's `# Contributing` heading is not duplicated —
same value `changelog.md` uses. Confirm against the rendered build in Step 5.)

- [ ] **Step 5: Build strictly and drive to zero warnings**

```bash
./tools/build-docs.sh
```
Expected: `build succeeded` with **zero** warnings. This is the acceptance gate.
If it warns, fix the specific cause and rebuild:
- *duplicate H1 / wrong content start* → adjust the `:start-line:` value in
  `contributing.md`.
- *unresolved anchor* `#claude-code-plugins--sessions` or
  `#lint-format-static-analysis` → the myst heading slug differs from GitHub's;
  change the **link** in `CONTRIBUTING.md` to the exact myst slug the warning
  names (do NOT change the heading text).
- *dangling/undefined link or orphan* → a file link was missed in Step 3, or a
  toctree entry is wrong in Step 1.

- [ ] **Step 6: Confirm nothing else still points at the old path**

```bash
grep -rn "releasing" docs/index.md docs/about/index.md | cat
grep -rn "docs/releasing.md" CONTRIBUTING.md README.md | cat
```
Expected: `docs/index.md` no longer lists `releasing`; `docs/about/index.md`
lists it; and no `docs/releasing.md` (old path) remains in `CONTRIBUTING.md` or
`README.md` (the CONTRIBUTING links now point at `docs/about/releasing.md`).

- [ ] **Step 7: Sanity-check the example suite is unaffected**

```bash
KERAS_BACKEND=jax uv run pytest tests/examples -q
```
Expected: PASS (docs content only — this just confirms nothing collateral broke).

- [ ] **Step 8: Lint and commit**

```bash
uv run ruff check --exit-non-zero-on-fix
git add docs/index.md docs/about/index.md docs/about/releasing.md docs/about/contributing.md CONTRIBUTING.md
git commit --no-gpg-sign -m "docs(about): include canonical CONTRIBUTING, move releasing under about/, add changelog-tag release check"
```

(The `git mv` from Step 1 is already staged as a rename; `git add` of the two
new-content paths plus the modified files completes the set. Verify with
`git status` that `docs/releasing.md` shows as renamed to
`docs/about/releasing.md`, not deleted+added.)

---

## After all tasks

Do NOT open the PR from within the task — the finishing step
(superpowers:finishing-a-development-branch, after the whole-branch review)
handles push/PR. The PR should note: this closes audit follow-ups #4, #8, and
#7; and that the newly-absolutized `blob/main/docs/about/releasing.md` link will
404 under `check-docs.sh` linkcheck until the PR merges (the path exists only on
the branch) — an expected, self-healing transient, and linkcheck is advisory,
not a CI gate.

## Notes for the implementer

- `docs/_build/` is a build artifact — never `git add` it.
- The strict build is the real gate here; there are no unit tests for docs
  content. Treat any `-W` warning as a failure to fix before committing.
- Absolutizing `CONTRIBUTING.md`'s links changes how they render on GitHub too
  (absolute instead of relative) — this is intended and keeps them working from
  both GitHub and the included docs page.
