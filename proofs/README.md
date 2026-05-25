# mononet Lean proofs

This Lake project is a Lean 4 + mathlib4 formalization of every theorem in:

> Runje, D., Shankaranarayana, S. M. (2023). *Constrained Monotonic Neural
> Networks.* ICML 2023. https://arxiv.org/abs/2205.11775

## Quickstart

````bash
# from this directory
lake exe cache get    # fetches the mathlib4 build cache (~30 s warm, ~5 min cold)
lake build            # builds every module, ~3 min after cache
````

## What is proved

| Module | Paper claim |
|---|---|
| `Mononet/Basic.lean` | Definitions: Sign, MonoMask, ActivationSplit, masked weights |
| `Mononet/Activations.lean` | BaseActivation class `Ă`; the derived ρ̂, ρ̃ |
| `Mononet/Layers.lean` | Definitions: constrainedLinear, combined, CMFCL |
| `Mononet/Lemma1Mono.lean` | Lemma 1 — sign of partial derivatives |
| `Mononet/Lemma2Combined.lean` | Lemma 2 + Corollary 3 |
| `Mononet/Lemma5Heaviside.lean` | Lemma 5 — Heaviside approximation by α·ρ̃ + β |
| `Mononet/Lemma6Equiv.lean` | Lemma 6 — affine rescale equivalence |
| `Mononet/DanielsVelikova.lean` | Theorem 4 from Daniels & Velikova 2010 — **axiomatized** |
| `Mononet/Theorem7Universal.lean` | Theorem 7 — universal approximation |

## Trust model

The formalization has **one** axiom — Theorem 4 of Daniels & Velikova
(2010, *Monotone and Partially Monotone Neural Networks*) — captured in
`Mononet/DanielsVelikova.lean`. A full Lean port of Daniels & Velikova's
proof is a deferred follow-up (no current ETA).

The CI job in `.github/workflows/lean.yml` enforces that no `sorry`
appears outside `DanielsVelikova.lean`.

## Versioning

This Lake project is versioned alongside the parent `mononet` Python
package. The toolchain pin and the mathlib4 revision live in
`lean-toolchain` and `lake-manifest.json` respectively. Bumping either is a
deliberate PR, not an automated dependabot update.

## License

Same as the parent project: PolyForm Noncommercial 1.0.0. See `../LICENSE`
and `../NOTICE.md`.
