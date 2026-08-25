# Follow-up: document that mononet expects normalized inputs (guides + reference)

Status: Idea / documentation backlog — not scheduled.
Origin: Structure-Preserving PINN application (`applications/pinn`); see
`applications/pinn/FINDINGS.md`.

## What we learned

mononet's `absolute`-mode weight initialisation (`mononet/core/init.py`,
`absolute_init_params`: gain scaled by `1/sqrt(in_features)`) implicitly assumes
**inputs of order ~unit scale**. When fed raw, unnormalised inputs (in the PINN
app: `x ∈ [-2, 3]`, `t ∈ [0, 1.5]`), fitting degrades badly and needs far more
training; mapping inputs to `[-1, 1]` fixes it. Concretely, in the PINN
comparison, normalising inputs improved the unconstrained baselines' L1 by ~2–3×
at the same budget.

This is a general usage requirement — the same standardisation the `benchmarks/`
harness already applies to tabular features — but it is **not currently called
out in the user-facing docs**.

## The follow-up

Add explicit guidance that **inputs should be standardised / scaled to roughly
unit range before a mononet layer**, and explain **why** (it is tied to the
`absolute`-mode weight-init scale, so poorly-scaled inputs start the network in a
bad regime and train slowly / to poor optima):

- **Guides** (`docs/guides/{pytorch,jax,keras}.md`): a short "Scale your inputs"
  note in each quickstart, ideally with a one-line `StandardScaler`-style example.
- **Reference** (`docs/reference.md` and/or the `MonoLinear` / init docstrings):
  state the input-scale assumption where `absolute` mode and
  `absolute_init_params` are described; cross-link to the guides.

## Scope
- Documentation only (guides + reference + possibly docstrings). No code change to
  mononet required. Optionally, consider whether a helper or a doc-recommended
  normalisation layer is worth offering — but that is a separate decision.
