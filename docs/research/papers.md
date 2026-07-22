# Paper registry

Maps `paper-id` → root + experiment-backend binding. All papers bind the same
backend (the `benchmarks/` harness; see `.honest-scholar/config.yml`). Status is
a projection of each paper's `paper/pitch.md` frontmatter — regenerated into
`dashboard.md` by `progress`, never hand-edited there.

| paper-id | root | backend | readiness | covers (thesis aims) |
|---|---|---|---|---|
| `cmnn-multibackend` | `docs/research/cmnn-multibackend/` | `benchmarks` | drafting | aim-1 |
| `monotone-constructions` | `docs/research/monotone-constructions/` | `benchmarks` | drafting | aim-1, aim-2 |
| `structure-preserving-pinns` | `docs/research/structure-preserving-pinns/` | `benchmarks` | drafting | aim-3 |
| `injective-monotonic-flows` | `docs/research/injective-monotonic-flows/` | `benchmarks` | drafting | aim-3 |

## Paper scope (from the integration spec — a starting split, expected to evolve)

- **`cmnn-multibackend`** — the multi-backend implementation (PyTorch / JAX / Keras)
  and reproduction of the CMNN paper Tables 1 & 2. Sub-projects A/B/C.
- **`monotone-constructions`** — methods paper on the construction flavors
  (`mixed` / `alternate` / `split`), initialization, residual depth, and the
  ablations. The current active research thread.
- **`structure-preserving-pinns`** — application paper: structure-preserving
  physics-informed networks (design on `spec/applications-structure-preserving-pinns`,
  PR #116). No results yet.
- **`injective-monotonic-flows`** — strictly-monotonic primitives and normalizing
  flows. Sub-project D. Future; no results yet.
