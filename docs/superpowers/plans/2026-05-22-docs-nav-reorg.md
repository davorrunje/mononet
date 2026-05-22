# Docs Navigation Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse the docs top navigation from a flat list of framework/section pages into exactly five clean items: Guides · Concepts · Benchmarks · Reference · About.

**Architecture:** Replace the five captioned root toctrees on `docs/index.md` with one hidden root toctree of five entries, each pointing to a section landing page that owns its own subtoctree. Move the BibTeX citation onto the home page and delete the standalone citation page. Move `contributing.md` under `about/`. Add a thin `reference.md` wrapper so the API reference navbar item is labeled "Reference" rather than the autodoc2-generated title "mononet".

**Tech Stack:** Sphinx 8.x · MyST · PyData Sphinx Theme · sphinx-autodoc2 · `sphinx-build -W` (warnings → errors)

**Spec:** `docs/superpowers/specs/2026-05-22-docs-nav-reorg-design.md`

**Working branch:** `feat/docs-nav-reorg` (already created from `main`; spec already committed in `9505871`).

---

## File map

**Create:**
- `docs/guides/index.md` — Guides landing page with subtoctree (pytorch, jax, keras)
- `docs/concepts/index.md` — Concepts landing page with subtoctree (monotonicity, layers)
- `docs/about/index.md` — About landing page with subtoctree (license, changelog, contributing)
- `docs/reference.md` — Reference wrapper around `apidocs/mononet/mononet`

**Modify:**
- `docs/index.md` — replace five captioned toctrees with one hidden root toctree + Citation section
- `docs/benchmarks/index.md` — fix relative link to moved `contributing.md`
- `docs/contributing.md` (content unchanged, just moved — handled via `git mv`)

**Move (`git mv`):**
- `docs/contributing.md` → `docs/about/contributing.md`

**Delete:**
- `docs/about/citation.md`

---

## Conventions used in this plan

**Build verification command** (run after each task that touches docs):
```bash
uv run sphinx-build -W docs docs/_build/html
```
Expected: `build succeeded.` and exit code 0. With `-W`, any warning fails the build.

If a build needs a clean slate (e.g., after deleting a page that Sphinx still has cached as an orphan candidate):
```bash
rm -rf docs/_build && uv run sphinx-build -W docs docs/_build/html
```

**Commit style:** `docs(nav): <imperative one-line summary>`. Use HEREDOC to preserve formatting and append the Co-Authored-By trailer:

```bash
git commit -m "$(cat <<'EOF'
docs(nav): <message>

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 1: Create Guides landing page

**Files:**
- Create: `docs/guides/index.md`

- [ ] **Step 1: Write the new file**

Write `docs/guides/index.md` with this exact content:

````markdown
# Guides

Framework-specific quickstarts for `mononet`. Pick the backend you use:

```{toctree}
:maxdepth: 1

pytorch
jax
keras
```
````

- [ ] **Step 2: Stage and commit**

```bash
git add docs/guides/index.md
git commit -m "$(cat <<'EOF'
docs(nav): add Guides landing page

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

The full Sphinx build runs at the end of Task 6; do not run it here. The page is unreachable from any toctree until Task 6 wires it into `index.md`, and Sphinx would otherwise emit an "isn't included in any toctree" warning that `-W` upgrades to an error.

---

## Task 2: Create Concepts landing page

**Files:**
- Create: `docs/concepts/index.md`

- [ ] **Step 1: Write the new file**

Write `docs/concepts/index.md` with this exact content:

````markdown
# Concepts

Background reading on monotonic neural networks.

```{toctree}
:maxdepth: 1

monotonicity
layers
```
````

- [ ] **Step 2: Stage and commit**

```bash
git add docs/concepts/index.md
git commit -m "$(cat <<'EOF'
docs(nav): add Concepts landing page

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

(No build until Task 6 — same reasoning as Task 1.)

---

## Task 3: Create About landing page and move contributing.md

**Files:**
- Create: `docs/about/index.md`
- Move: `docs/contributing.md` → `docs/about/contributing.md`

- [ ] **Step 1: Move contributing.md with git so history is preserved**

```bash
git mv docs/contributing.md docs/about/contributing.md
```

- [ ] **Step 2: Write the About landing page**

Write `docs/about/index.md` with this exact content:

````markdown
# About

```{toctree}
:maxdepth: 1

license
changelog
contributing
```
````

- [ ] **Step 3: Stage and commit**

```bash
git add docs/about/index.md docs/about/contributing.md docs/contributing.md
git commit -m "$(cat <<'EOF'
docs(nav): add About landing page and move contributing into it

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

(No build until Task 6.)

---

## Task 4: Fix benchmarks link to moved contributing.md

`docs/benchmarks/index.md` references the contributing page via a relative
link `[\`CONTRIBUTING.md\`](../contributing.md)`. After Task 3, the file is
at `docs/about/contributing.md`, so the relative link must be updated.

**Files:**
- Modify: `docs/benchmarks/index.md` line 7

- [ ] **Step 1: Update the link**

In `docs/benchmarks/index.md`, replace:

```markdown
[`CONTRIBUTING.md`](../contributing.md).
```

with:

```markdown
[`CONTRIBUTING.md`](../about/contributing.md).
```

- [ ] **Step 2: Stage and commit**

```bash
git add docs/benchmarks/index.md
git commit -m "$(cat <<'EOF'
docs(nav): repoint benchmarks link to moved contributing page

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

(No build until Task 6.)

---

## Task 5: Create Reference wrapper page

The autodoc2-generated `docs/apidocs/mononet/mononet.md` has title `mononet`,
which would become the navbar label. The wrapper gives us the "Reference"
label.

**Files:**
- Create: `docs/reference.md`

- [ ] **Step 1: Write the wrapper file**

Write `docs/reference.md` with this exact content:

````markdown
# Reference

API reference for the `mononet` package.

```{toctree}
:maxdepth: 2

apidocs/mononet/mononet
```
````

- [ ] **Step 2: Stage and commit**

```bash
git add docs/reference.md
git commit -m "$(cat <<'EOF'
docs(nav): add Reference wrapper page

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

(No build until Task 6.)

---

## Task 6: Rewrite the home page

This is the load-bearing change: the home page's five captioned toctrees
collapse into one hidden five-entry toctree, and the BibTeX citation moves
inline. After this task the navbar should show exactly five items.

**Files:**
- Modify: `docs/index.md` (full rewrite of the toctree section + new Citation section)
- Delete: `docs/about/citation.md`

- [ ] **Step 1: Read the existing citation page to capture its BibTeX block**

```bash
cat docs/about/citation.md
```

Expected content (used in Step 2 below):

```markdown
# Citation

If you use `mononet` in academic work, please cite the reference paper:

```bibtex
@inproceedings{runje2023constrained,
  title     = {Constrained Monotonic Neural Networks},
  author    = {Runje, Davor and Shankaranarayana, Sharath M.},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {202},
  year      = {2023},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v202/runje23a.html},
  eprint    = {2205.11775},
  archivePrefix = {arXiv}
}
```

> Note: confirm the exact BibTeX entry against the PMLR proceedings page
> before the first PyPI release — venue, volume, and URL fields are
> sensitive to typos.
```

- [ ] **Step 2: Overwrite `docs/index.md` with the new structure**

Replace the entire contents of `docs/index.md` with:

````markdown
---
hide-toc: false
---

# mononet

**Unconstrained monotonic neural networks** with first-class support for
**PyTorch**, **JAX** (Flax NNX), and **Keras 3**.

Reference implementation of:

> Runje, D., Shankaranarayana, S. M. (2023). *Constrained Monotonic
> Neural Networks.* ICML 2023. [arXiv:2205.11775](https://arxiv.org/abs/2205.11775)

## Install

```
pip install "mononet[torch]"      # PyTorch
pip install "mononet[jax]"        # JAX + Flax NNX
pip install "mononet[keras]"      # Keras 3
pip install "mononet[all]"        # all three
```

## Citation

If you use `mononet` in academic work, please cite the reference paper:

```bibtex
@inproceedings{runje2023constrained,
  title     = {Constrained Monotonic Neural Networks},
  author    = {Runje, Davor and Shankaranarayana, Sharath M.},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {202},
  year      = {2023},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v202/runje23a.html},
  eprint    = {2205.11775},
  archivePrefix = {arXiv}
}
```

> Note: confirm the exact BibTeX entry against the PMLR proceedings page
> before the first PyPI release — venue, volume, and URL fields are
> sensitive to typos.

```{toctree}
:hidden:

guides/index
concepts/index
benchmarks/index
reference
about/index
```
````

- [ ] **Step 3: Delete the standalone citation page**

```bash
git rm docs/about/citation.md
```

- [ ] **Step 4: Clean-build the docs**

```bash
rm -rf docs/_build && uv run sphinx-build -W docs docs/_build/html
```

Expected: `build succeeded.` with zero warnings. Specifically watch for:

- ✅ No "document isn't included in any toctree" warnings.
- ✅ No "unknown document" warnings (would indicate broken `:doc:` refs or toctree entries).
- ✅ No "duplicate label" warnings (would indicate `# Citation` clashes with anything).

If a warning appears for `about/citation`, ensure Step 3 actually removed the file (`git status` should show it deleted) and that `_build/` was wiped in Step 4.

- [ ] **Step 5: Stage and commit**

```bash
git add docs/index.md
git commit -m "$(cat <<'EOF'
docs(nav): collapse home toctrees and inline the BibTeX citation

Replace the five captioned root toctrees with one hidden root toctree of
five entries, each pointing at a section landing page. The BibTeX block
moves onto the home page; about/citation.md is removed.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

`git rm` from Step 3 already staged the deletion; the commit captures both.

---

## Task 7: Visual smoke test

Build artifacts on disk verify Sphinx is happy. This step inspects the
generated HTML to confirm the navbar actually shows five items, in the
intended order, with the intended labels.

**Files:**
- Inspect: `docs/_build/html/index.html`

- [ ] **Step 1: Confirm the five navbar items appear in order**

```bash
python -c "
import re
from pathlib import Path
html = Path('docs/_build/html/index.html').read_text()
# PyData renders top-nav items as <li class='nav-item'>...<a>LABEL</a>...
labels = re.findall(r'<li class=\"nav-item[^\"]*\">\s*<a[^>]*>([^<]+)</a>', html)
print(labels)
"
```

Expected output (substring match — order matters, exact whitespace may vary):

```
['Guides', 'Concepts', 'Benchmarks', 'Reference', 'About']
```

If the order is wrong: re-check the toctree entries in `docs/index.md` Task 6 Step 2 — they must be in this exact order: `guides/index, concepts/index, benchmarks/index, reference, about/index`.

If "mononet" appears instead of "Reference": the Reference wrapper page from Task 5 is missing or `docs/index.md` is still pointing at `apidocs/mononet/mononet` directly.

- [ ] **Step 2: Confirm Citation section renders on the home page**

```bash
grep -c 'id="citation"' docs/_build/html/index.html
```

Expected output: `1`

- [ ] **Step 3: Confirm no stale citation page**

```bash
ls docs/_build/html/about/ 2>/dev/null
```

Expected output: `changelog.html  contributing.html  index.html  license.html` (no `citation.html`).

If `citation.html` is present, the clean build in Task 6 Step 4 was skipped; rerun it.

- [ ] **Step 4: Confirm subsection landing pages are reachable**

```bash
for f in guides concepts about; do
  test -f "docs/_build/html/$f/index.html" && echo "OK: $f/index.html" || echo "MISSING: $f/index.html"
done
test -f docs/_build/html/reference.html && echo "OK: reference.html" || echo "MISSING: reference.html"
```

Expected: all four `OK:` lines.

- [ ] **Step 5: Final commit (optional, only if any fix was needed in Steps 1-4)**

If smoke tests passed without any fixes, this step is a no-op — skip it.

If a fix *was* needed, commit it now:

```bash
git add -A
git commit -m "$(cat <<'EOF'
docs(nav): smoke-test fix

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Push branch and open PR

**Files:** none (git/GitHub operations only)

- [ ] **Step 1: Push the branch**

```bash
git push -u origin feat/docs-nav-reorg
```

Expected: branch created on origin, message like `* [new branch] feat/docs-nav-reorg -> feat/docs-nav-reorg`.

- [ ] **Step 2: Open the PR**

```bash
gh pr create --title "Reorganize docs top navigation into five buckets" --body "$(cat <<'EOF'
## Summary
- Collapse home page from five captioned toctrees to one hidden toctree of five entries (Guides · Concepts · Benchmarks · Reference · About).
- Move BibTeX citation to the home page; delete `about/citation.md`.
- Move `contributing.md` into `about/` so the About bucket is self-contained.
- Add a thin `reference.md` wrapper so the API reference navbar item is labeled "Reference" rather than "mononet".

## Test plan
- [x] `sphinx-build -W docs docs/_build/html` succeeds with zero warnings.
- [x] Generated `index.html` shows exactly five top-nav items in the expected order.
- [x] Home page renders a Citation section with the BibTeX block.
- [x] No stale `about/citation.html` in the build output.
- [ ] After merge: Docs workflow redeploys; eyeball https://davorrunje.github.io/mononet/ to confirm the navbar in both light and dark themes.

Spec: `docs/superpowers/specs/2026-05-22-docs-nav-reorg-design.md`
Plan: `docs/superpowers/plans/2026-05-22-docs-nav-reorg.md`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Expected: a PR URL printed to stdout.

- [ ] **Step 3: Report the PR URL back to the user**

The PR URL from Step 2 is what the user needs to review and merge. No further automated action.
