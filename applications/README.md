# Applications

Research papers built **on** `mononet` as a library. Distinct from
[`benchmarks/`](../benchmarks/) (which reproduces the `mononet` paper's own
Tables 1 & 2) — applications are *downstream* work that uses the published
layers to do new science. Nothing here ships in the PyPI wheel.

Each paper is a self-contained subpackage with its own `README.md` (abstract +
headline), `RUNBOOK.md` (exact reproduction commands), a Markdown manuscript
under `paper/`, and an executed notebook rendered into the Sphinx docs.

## Papers

| Dir | Paper | Status |
|---|---|---|
| [`pinn/`](pinn/) | Structure-Preserving PINNs — hard monotonicity as a PDE admissibility prior (inverse conservation-law / traffic flagship) | In progress (Paper 1) |

Planned follow-ups (own specs/plans): high-dim HJB (Paper 2), Fokker–Planck
(Paper 3), eikonal (Paper 4), arbitrage-free surfaces (Paper 5, regression —
not a PINN).

Design: [`docs/superpowers/specs/2026-07-12-applications-structure-preserving-pinns-design.md`](../docs/superpowers/specs/2026-07-12-applications-structure-preserving-pinns-design.md).
