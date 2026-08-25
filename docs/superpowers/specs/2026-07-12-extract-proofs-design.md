# Design: Extract Lean proofs to external repo — docs link-out + scaffolding removal

**Date:** 2026-07-12
**Status:** Approved (brainstorming) — pending implementation plan
**Scope:** Docs + repo scaffolding. No `mononet/**` code change.

## Problem

The Lean 4 / mathlib4 formalization has moved out of `mononet` into a standalone
repo, **[neural-network-proofs](https://github.com/davorrunje/neural-network-proofs)**,
which is broader than mononet's paper (Cybenko, Leshno–Lin–Pinkus–Schocken,
Mikulincer–Reichman, Sartor 2025, Runje et al. *forthcoming*, Amos ICNN), is
`sorry`-free, and publishes its own **hosted docs** (interactive blueprint +
doc-gen4 API) at <https://davorrunje.github.io/neural-network-proofs/>.

`mononet` still carries the original scaffolding, now stale:

- `proofs/` — the whole in-repo Lean project (moved out).
- `docs/concepts/proofs.md` — links to `proofs/` and to a cross-reference table
  whose "Empirical counterpart" column points at `tests/properties/` tests that
  **do not exist** in this repo (doubly stale).
- `.devcontainer/proofs/` flavor + `.devcontainer/shared/install_lean.sh`.
- `.github/workflows/lean.yml`.
- Sub-project-E design spec + the meta-spec's five-sub-project framing.
- Devcontainer-flavor tables in `CLAUDE.md` / `CONTRIBUTING.md` listing `proofs`.
- `README.md` §Formal proofs and `.secrets.baseline` entries for `proofs/`.

## Goal

Remove all Lean scaffolding from `mononet`, and replace the in-repo proofs docs
with an outward reference to the companion repo and its hosted docs.

## Decisions

- **Docs → slim page linking out.** Rewrite `concepts/proofs.md`, don't rebuild
  the cross-reference table (the external blueprint *is* that map; a mirror
  re-introduces the drift recent workstreams eliminated).
- **Remove all scaffolding.** Delete the Lean project, its devcontainer flavor,
  its CI, and update every table that referenced them.
- **Specs → annotate as superseded** (preserve history); **CLAUDE.md → update**
  (it is live guidance that would otherwise mislead future work).

## Non-goals

- No change to the external `neural-network-proofs` repo.
- No `mononet/**` source change.
- Not re-authoring the proof cross-reference table anywhere in `mononet`.

## Half A — Docs (link out)

- **Rewrite `docs/concepts/proofs.md`** as a slim "Formal foundations" page:
  1–2 paragraphs that mononet's constrained-monotonic construction rests on
  machine-checked (`sorry`-free) Lean 4 + mathlib4 results, formalized in the
  companion **neural-network-proofs** repo, with prominent links to the repo and
  its **hosted blueprint + API docs**
  (<https://davorrunje.github.io/neural-network-proofs/>). Note the broader
  approximation-theory lineage it covers. **Remove** the stale paper-claim ↔
  Lean-theorem ↔ Python-test table and the `cd proofs && lake build` local-build
  instructions. (The page H1 may be retitled; the concepts toctree picks up the
  new title automatically.)
- **`docs/concepts/index.md`** — keep the `proofs` toctree entry (points at the
  rewritten page); no structural change.
- **`README.md` §Formal proofs** (≈ lines 150–155) — re-point from `proofs/`
  (now gone) to the companion repo + its hosted docs; keep it short.

## Half B — Scaffolding removal

- **Delete:** `proofs/` (entire Lean project), `.devcontainer/proofs/` (flavor:
  `devcontainer.json`, `devcontainer-lock.json`, `docker-compose.yml`,
  `setup.sh`), `.devcontainer/shared/install_lean.sh`, `.github/workflows/lean.yml`.
- **Devcontainer-flavor tables → 4 flavors** (drop the `proofs` row) in
  `CLAUDE.md` (line ~98) and `CONTRIBUTING.md` (line ~25). The `default` flavor's
  "docs" description already covers documentation work.
- **gpu-\*/Dockerfile comments** — reword the "base used by the CPU and proofs"
  comment (in `gpu-torch`, `gpu-jax`, `gpu-keras` Dockerfiles) to drop "and
  proofs" (cosmetic; not a build dependency — verified).
- **Regenerate `.secrets.baseline`** so the 10 `proofs/` entries are dropped and
  the pre-commit `detect-secrets` hook stays clean.

## Half B — Specs & CLAUDE guidance

- **Sub-project-E spec** (`docs/superpowers/specs/2026-05-22-E-lean-proofs-design.md`)
  and the **meta-spec** (`docs/superpowers/specs/2026-05-21-mononet-package-design.md`):
  keep as history; add a top note:
  `> Superseded (2026-07-12): the Lean formalization now lives in the standalone`
  `neural-network-proofs repo and is no longer part of mononet.`
  (In the meta-spec, place it wherever the five-sub-project decomposition / E is
  framed.)
- **`CLAUDE.md`** (live guidance — update, do not merely annotate): change the
  sub-project decomposition table so **E** points at the external repo (or is
  dropped from the five-sub-project framing → four), and remove the `proofs`
  devcontainer-flavor row.

## Validation & acceptance

- `./tools/build-docs.sh` — strict `-W` + `nitpicky=True` build green, zero
  warnings (catches any dangling link left by the proofs-page rewrite / removed
  `proofs/` relative links).
- `./tools/check-docs.sh` — no new broken internal links; the external
  `neural-network-proofs` links resolve (200).
- `uv run pre-commit run --all-files` — passes, notably `detect-secrets` against
  the regenerated baseline, plus codespell/file-hygiene.
- `lean.yml` removed; `build.yml`'s `check` aggregation never referenced it, so
  no CI wiring breaks.
- Grep sweep: no remaining `proofs/`-path, `install_lean`, or `lean.yml`
  reference outside intentionally-annotated history (`docs/superpowers/**`, the
  sphinx-migration-pr note).

**Definition of done:** Lean scaffolding fully removed; docs point outward to the
companion repo + its hosted site; flavor tables read four flavors; specs/CLAUDE
reflect the move; the strict docs build and full pre-commit both pass.
