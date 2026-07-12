# Extract Lean Proofs to External Repo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove all in-repo Lean/proof scaffolding from `mononet` and replace the in-repo proofs docs with an outward link to the standalone `neural-network-proofs` repo and its hosted docs.

**Architecture:** Docs + scaffolding cleanup, no `mononet/**` code change. Two tasks: (1) point the docs outward, then (2) delete the scaffolding and update the guidance/tables. Doing docs first means nothing references `proofs/` when it's deleted. Gates: the strict Sphinx build (`sphinx-build -W`, `nitpicky=True`) and `pre-commit run --all-files`.

**Tech Stack:** Sphinx + myst-nb docs, detect-secrets, GitHub Actions, devcontainers.

**Spec:** [docs/superpowers/specs/2026-07-12-extract-proofs-design.md](../specs/2026-07-12-extract-proofs-design.md)

## Global Constraints

- **Branch:** `spec/extract-proofs` (already checked out). Never commit to `main`.
- **Commit signing is broken in this container** — always `git commit --no-gpg-sign`.
- **No `mononet/**` code change**; the external `neural-network-proofs` repo is untouched.
- **External repo URLs:** repo `https://github.com/davorrunje/neural-network-proofs`; hosted docs `https://davorrunje.github.io/neural-network-proofs/`.
- **Do NOT rebuild** the paper-claim↔Lean-theorem cross-reference table anywhere — the external blueprint is that map.
- Use `git rm`/`git rm -r` for deletions. Never `git add docs/_build/`. Run via `uv run`; never run any `uv sync` variant.
- **Verified adjustment to the spec:** the meta-spec (`2026-05-21-mononet-package-design.md`) has **no** Lean/sub-project-E reference (its "E" mentions are dependency *groups*), so it is NOT edited. Only the E sub-project spec is annotated; the live sub-project table lives in `CLAUDE.md`.
- Gates: `./tools/build-docs.sh` (strict, zero warnings); `uv run pre-commit run --all-files`.

---

### Task 1: Point the docs outward

**Files:**
- Modify: `docs/concepts/proofs.md` (full rewrite)
- Modify: `README.md` (§Formal proofs, ≈ lines 150–155)

**Interfaces:**
- Consumes: nothing. Produces: docs that no longer reference the in-repo `proofs/` (so Task 2 can delete it without leaving dangling links).

- [ ] **Step 1: Rewrite `docs/concepts/proofs.md` entirely**

Replace the whole file with:

```markdown
# Formal foundations

`mononet`'s constrained-monotonic construction rests on results with
machine-checked proofs. The formalization — Lean 4 + mathlib4, `sorry`-free —
lives in the companion repository
**[neural-network-proofs](https://github.com/davorrunje/neural-network-proofs)**.

It mechanizes the universal-approximation lineage behind the method: Cybenko
(1989), Leshno–Lin–Pinkus–Schocken (1993), Mikulincer–Reichman (2022),
Sartor et al. (2025), the deep constrained-monotonic result (Runje et al.), and
Amos et al. (2017) input-convex networks.

Browse the proofs, the interactive blueprint (proof sketches + dependency
graph), and the doc-gen4 API reference at the project's rendered docs:

- **Docs & blueprint:** <https://davorrunje.github.io/neural-network-proofs/>
- **Source:** <https://github.com/davorrunje/neural-network-proofs>
```

(The H1 changes to "Formal foundations"; the `concepts/index.md` toctree entry
`proofs` picks up the new title automatically — no toctree edit needed.)

- [ ] **Step 2: Rewrite the README §Formal proofs block**

In `README.md`, replace the current section (heading `## Formal proofs` and the
paragraph linking `proofs/` + the cross-reference page) with:

```markdown
## Formal proofs

The theory underpinning `mononet` is mechanized in Lean 4 + mathlib4
(`sorry`-free) in the companion repo
**[neural-network-proofs](https://github.com/davorrunje/neural-network-proofs)** —
browse the proofs, blueprint, and API docs at
<https://davorrunje.github.io/neural-network-proofs/>.
```

- [ ] **Step 3: Build strictly — confirm no dangling links from the rewrite**

Run: `./tools/build-docs.sh`
Expected: `build succeeded`, zero warnings. (`proofs/` still exists on disk at
this point — that's fine; the point is the rewritten page no longer links to
removed content, and the external links don't trip the `-W` build since it does
not check external URLs.)

- [ ] **Step 4: Commit**

```bash
git add docs/concepts/proofs.md README.md
git commit --no-gpg-sign -m "docs(proofs): point to external neural-network-proofs repo + hosted docs"
```

---

### Task 2: Remove the Lean scaffolding and update guidance

**Files:**
- Delete: `proofs/` (whole dir), `.devcontainer/proofs/` (whole dir), `.devcontainer/shared/install_lean.sh`, `.github/workflows/lean.yml`
- Modify: `CLAUDE.md` (sub-project table row E; devcontainer-flavor table)
- Modify: `CONTRIBUTING.md` (devcontainer-flavor table)
- Modify: `.devcontainer/gpu-torch/Dockerfile`, `.devcontainer/gpu-jax/Dockerfile`, `.devcontainer/gpu-keras/Dockerfile` (line-21 comment)
- Modify: `docs/superpowers/specs/2026-05-22-E-lean-proofs-design.md` (superseded note)
- Regenerate: `.secrets.baseline`

**Interfaces:**
- Consumes: Task 1's outward-pointing docs (so deleting `proofs/` leaves no dangling doc link).

- [ ] **Step 1: Delete the Lean project, its devcontainer flavor, its CI**

```bash
git rm -r proofs .devcontainer/proofs
git rm .devcontainer/shared/install_lean.sh .github/workflows/lean.yml
```

- [ ] **Step 2: Drop the `proofs` devcontainer-flavor row (5 → 4 flavors)**

In `CLAUDE.md`, the flavor table currently ends with:

```
| `gpu-keras` | GPU work with Keras 3 (JAX backend + CUDA 12 by default) |
| `proofs` | Reviewing the Lean 4 / mathlib4 formalization under `proofs/` (CPU, no ML extras) |
```

Remove the `proofs` row (leave the `gpu-keras` row as the last one).

In `CONTRIBUTING.md`, the flavor table currently ends with:

```
| `gpu-keras`     | GPU work with Keras 3 (backed by JAX with CUDA 12 by default).   |
| `proofs`        | Reviewing the Lean 4 / mathlib4 formalization under `proofs/` (CPU, no ML extras). |
```

Remove the `proofs` row there too.

- [ ] **Step 3: Repoint the sub-project-E table row in `CLAUDE.md`**

In `CLAUDE.md`, the sub-project decomposition table row currently reads:

```
| [E](docs/superpowers/specs/2026-05-22-E-lean-proofs-design.md) | Lean 4 + mathlib4 formalization of paper theorems |
```

Replace it with (keeps the historical spec link, marks it moved, points at the
external repo):

```
| [E](docs/superpowers/specs/2026-05-22-E-lean-proofs-design.md) *(moved out)* | Lean 4 + mathlib4 formalization — now the standalone [neural-network-proofs](https://github.com/davorrunje/neural-network-proofs) repo |
```

- [ ] **Step 4: Reword the gpu-\* Dockerfile comment**

In each of `.devcontainer/gpu-torch/Dockerfile`, `.devcontainer/gpu-jax/Dockerfile`,
`.devcontainer/gpu-keras/Dockerfile`, line 21 reads:

```
# mcr.microsoft.com/devcontainers base used by the CPU and proofs
```

Change it to:

```
# mcr.microsoft.com/devcontainers base used by the CPU devcontainer
```

(Verify the following line's continuation still reads correctly; if line 21 is a
wrapped sentence continuing on line 22, adjust so the reworded comment stays
grammatical.)

- [ ] **Step 5: Annotate the sub-project-E spec as superseded**

At the top of `docs/superpowers/specs/2026-05-22-E-lean-proofs-design.md`,
immediately after the `# Sub-project E — ...` H1 line, insert:

```markdown

> **Superseded (2026-07-12):** the Lean formalization now lives in the
> standalone [neural-network-proofs](https://github.com/davorrunje/neural-network-proofs)
> repo and is no longer part of mononet. This spec is kept for historical record.
```

- [ ] **Step 6: Regenerate `.secrets.baseline` (drop the `proofs/` entries)**

```bash
uv run detect-secrets scan --exclude-files 'uv.lock' > .secrets.baseline
grep -c "proofs/" .secrets.baseline    # expect 0
```
Expected: the second command prints `0` (no `proofs/` paths remain in the
baseline).

- [ ] **Step 7: Validate — strict build, pre-commit, and a grep sweep**

```bash
./tools/build-docs.sh
uv run pre-commit run --all-files
grep -rniE "proofs/|install_lean|lean\.yml" . 2>/dev/null \
  | grep -viE "docs/_build/|\.git/|\.superpowers/sdd/|docs/superpowers/(specs|plans|audits|2026-05-22-sphinx-migration-pr)"
```
Expected:
- `build succeeded`, zero warnings.
- `pre-commit` passes (notably `detect-secrets` against the regenerated baseline,
  plus codespell / file-hygiene / docs build).
- The grep sweep returns **nothing** — no live reference to `proofs/`,
  `install_lean`, or `lean.yml` remains outside intentionally-kept history
  (the `docs/superpowers/**` specs/plans/audits and the sphinx-migration-pr
  note). Content mentions of the word "proof(s)" in prose (e.g.
  `docs/references/*`, `docs/concepts/monotonicity.md`) are fine — the pattern
  above matches the `proofs/` *path*, not the word.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit --no-gpg-sign -m "chore: remove in-repo Lean scaffolding (proofs/, devcontainer flavor, lean CI); update flavor tables + E spec"
```

Verify with `git status` that the deletions are staged (the `proofs/` and
`.devcontainer/proofs/` trees show as deleted) and no `docs/_build/` slipped in.

---

## After all tasks

Do NOT open the PR from within a task — the finishing step
(superpowers:finishing-a-development-branch, after the whole-branch review)
handles push/PR. Notes for the PR body:
- This removes the in-repo Lean project now that proofs live in the standalone
  `neural-network-proofs` repo; docs point outward to it + its hosted site.
- `lean.yml` was a **separate** workflow and was never part of the required
  `check` status (branch protection requires only `check` from `build.yml`), so
  removing it orphans no required status check — confirm on the PR that no
  "expected — waiting for status" appears for a Lean check.

## Notes for the implementer

- `docs/_build/` is a build artifact — never `git add` it.
- Deleting `proofs/` also removes `proofs/tools/*.sh`; that's intended.
- If `pre-commit`'s `detect-secrets` still flags something after the baseline
  regen, it means a real secret-shaped string exists elsewhere — inspect before
  auditing it into the baseline; do not blanket-accept.
