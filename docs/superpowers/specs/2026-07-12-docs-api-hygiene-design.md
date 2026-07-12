# Design: API-reference cross-ref hygiene + core-object docstrings

**Date:** 2026-07-12
**Status:** Approved (brainstorming) — pending implementation plan
**Scope:** Docs build config + core-package docstrings. Workstream B of the documentation-audit follow-ups.

## Problem

Three findings from the [documentation audit](../audits/2026-07-12-docs-audit.md):

- **#5 (intersphinx/autodoc2 cleanup):** a nitpicky (`sphinx-build -n`) build reports
  ~350 unresolved cross-references — 132 torch, 90 typing_extensions, 83 flax,
  21 numpy, 17 jax, ~4 keras. `nitpicky` is currently **off**, so none of this
  fails a build; the API reference silently ships dead cross-refs.
- **#10 (core-object docstrings):** `MonoConfig`, `MonoResidualConfig`,
  `MonotonicityMask`, `ActivationSpec`, `InitSpec` describe their fields in prose
  but do not carry MyST field-list `:param <field>:` documentation
  (`MonoResidualConfig` has a partial one).
- **#11 (`linkcheck_ignore`):** `patents.justia.com/patent/11551063` returns a
  bot-blocked 403 under linkcheck (loads fine in a browser) and is flagged as a
  broken link every run.

**Goal:** make the API-reference cross-refs resolve, document the core value
objects, and make the doc build hygienic — enforced so it stays that way.

## Decisions

- **Enforce cross-ref hygiene via the existing `-W` gate.** After cleanup,
  set `nitpicky = True` in `docs/conf.py` so the existing `./tools/build-docs.sh`
  (`sphinx-build -W`) — already the `docs-smoke` CI job — fails on any
  unresolved cross-reference. No workflow change.
- **Resolve stable public types; ignore irreducible internals.** Trying to
  resolve *every* external symbol is what makes an enforced nitpicky gate
  fragile, so the policy is deliberately split (see below).

## Non-goals

- No CHANGELOG tag-link work (#7) and no About-section IA (#4/#8) — Workstream C.
- No new CI job and no change to `.github/workflows/build.yml` (enforcement rides
  the existing `-W` docs-smoke gate purely via a `conf.py` setting).
- No behavior change to any `mononet/**` code — docstrings only.

## Files

- Modify: `docs/conf.py` — flax intersphinx inventory; `nitpick_ignore_regex`;
  `linkcheck_ignore`; `nitpicky = True`; resolve the commented-out keras entry.
- Modify: `mononet/core/config.py`, `mononet/core/types.py` — field-list
  docstrings (docstring-only).

## Cross-ref resolution & enforcement (#5 + #11)

Drive the ~350 nitpick warnings to **zero** with a two-pronged policy.

### Resolve (inventories)

- **flax** — `https://flax.readthedocs.io/en/latest` is Sphinx and publishes an
  `objects.inv`; add it (clears the ~83 flax refs, e.g. `flax.nnx.Module`,
  `flax.nnx.Rngs`). The implementer verifies the inventory URL by fetching its
  `objects.inv` before relying on it.
- **torch / jax / numpy** — inventories already configured; keep them so the
  high-value public types (`torch.Tensor`, `torch.nn.Module`, `jax.Array`,
  numpy arrays) stay clickable in the reference.

### Ignore (`nitpick_ignore_regex`, namespace-scoped)

For symbols with no resolvable inventory target, add **namespace-scoped regexes**
(not exact-target ignores), each with a one-line comment explaining why it can't
be resolved:

- `typing_extensions\..*` — no intersphinx inventory exists (the ~90).
- `keras\..*` — keras.io is not a Sphinx site (no `objects.inv`); the commented
  keras intersphinx entry is removed with a note (the ~4).
- residual **framework-internal** torch/jax/`numpy.typing` targets that are not
  in the public inventory (internal module paths, `numpy.typing.*` details) —
  scoped regexes.

**Why namespace regexes, not exact targets:** this is the mitigation for the one
real downside of enforcing nitpicky in CI — a Dependabot bump of torch/flax/etc.
could change an inventory and turn the `docs-smoke` gate red on an unrelated PR.
Ignoring *unstable external namespaces* by regex (while still resolving the
stable public types) means a dependency's internal drift cannot break the docs
gate. The set of regexes is kept as tight as possible: only namespaces that
genuinely have no stable public target.

### linkcheck_ignore (#11)

Add to a new `linkcheck_ignore` list, each with a one-line justification:

- `https://patents.justia.com/patent/11551063` — bot-blocked 403; loads in a
  browser.
- the pytorch-docs anchor URLs linkcheck reports as "Anchor not found"
  (stale-anchor quirk of the pytorch docs, not a real break).

This only cleans `./tools/check-docs.sh` output — linkcheck is not a CI gate.

### Enforcement

Once `sphinx-build -n` is clean, set `nitpicky = True` in `docs/conf.py`. The
existing `-W` build then treats any unresolved cross-ref as fatal.

## Core-object field-list docstrings (#10)

Bring the five core value objects to the
[MyST field-list spec](../specs/2026-05-22-myst-docstrings-design.md): every
public field documented with `:param <field>:`; types come from the annotations
(never `:type:`/`:rtype:`); body text is MyST markdown; `:raises ValueError:`
where `__post_init__` validates. **Docstring-only — zero behavior change**; the
existing `tests/core/test_config.py` and `tests/core/test_types.py` stay green.

Per object (existing conceptual body text is kept; the field-list is *added*, the
prose is not rewritten):

- **`MonoConfig`** (`config.py`) — `:param units:`, `mode`, `activation`,
  `convex_fraction`, `init`, `bias`.
- **`MonoResidualConfig`** (`config.py`) — has `:param activation:`; complete
  `units`, `mode`, `alpha_gate`, `beta_gate`, `init`.
- **`MonotonicityMask`** (`types.py`) — `:param values:` (fold the `{-1,+1}`
  semantics into the param/body); document the `shape` property and `__len__`.
- **`ActivationSpec`** (`types.py`) — `:param name:`; `:raises ValueError:` for
  an unknown activation.
- **`InitSpec`** (`types.py`) — `:param scheme:`, `:param seed:`.

Adding these field-lists may introduce new cross-refs (e.g. `InitSpec`,
`ActivationSpec`, `numpy.typing` in `MonotonicityMask.values`); the cross-ref
cleanup above must account for them (they are resolved internally or covered by
the ignore regexes), so the docstring work is sequenced **before** the
`nitpicky = True` flip.

## Validation & acceptance

- `./tools/build-docs.sh` (now `-W` **and** `nitpicky=True`) exits 0 with **zero**
  warnings — both the #5 acceptance criterion and the durable enforcement.
- `./tools/check-docs.sh` — the nitpicky section reports no warnings; linkcheck no
  longer lists the justia / pytorch-anchor URLs as breaks.
- `uv run pytest tests/core` — stays green (guards that the docstring edits did
  not change validation behavior).
- Spot-check: the rendered `MonoConfig` / `MonoResidual` reference pages show the
  documented fields, and a resolved public xref (e.g. `torch.Tensor`) is a
  working link.

**Definition of done:** `nitpicky=True` committed; strict build green at zero
warnings; the five core objects field-list-documented; every
`nitpick_ignore_regex` and `linkcheck_ignore` entry carrying a one-line
justification.
