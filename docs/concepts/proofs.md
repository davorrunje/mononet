# Formal proofs of paper theorems

The mononet repository contains a Lean 4 / mathlib4 formalization of every
theorem in the paper, living under [`proofs/`](https://github.com/davorrunje/mononet/tree/main/proofs).
This page is the cross-reference: paper claim ↔ Lean theorem ↔ Python
property test.

## Trust model

The formalization has **one** axiom: Theorem 4 of Daniels & Velikova
(2010, *Monotone and Partially Monotone Neural Networks*). Every other claim
is proved from first principles using mathlib4. See
[`proofs/Mononet/DanielsVelikova.lean`](https://github.com/davorrunje/mononet/blob/main/proofs/Mononet/DanielsVelikova.lean)
for the precise axiom statement.

A full port of Daniels & Velikova 2010's proof is a deferred follow-up
(tracked in the project's GitHub issues).

## Cross-reference table

| Paper claim | Lean theorem | Empirical counterpart |
|---|---|---|
| Lemma 1 (sign of partial derivatives) | `Mononet.constrainedLinear_monotone_pos` / `_antitone_neg` in `Lemma1Mono.lean` | `tests/properties/test_lemma1_constrained_linear_mono.py` |
| Lemma 2 (combined activation mono + convex/concave) | `Mononet.combined_monotone_componentwise`, `combined_convex_of_all_breve`, `combined_concave_of_all_hat` in `Lemma2Combined.lean` | `tests/properties/test_lemma2_combined_mono.py` |
| Corollary 3 (CMFCL properties) | `Mononet.CMFCL_props` in `Lemma2Combined.lean` | (covered by tests for Lemma 1 + Lemma 2) |
| Lemma 5 (Heaviside approximation) | `Mononet.saturated_approximates_heaviside_left` / `_right` in `Lemma5Heaviside.lean` | `tests/properties/test_lemma5_heaviside_limit.py` |
| Lemma 6 (affine rescale equivalence) | `Mononet.affine_rescale_equiv` in `Lemma6Equiv.lean` | `tests/properties/test_lemma6_affine_rescale.py` |
| Theorem 7 (universal approximation) | `Mononet.universal_approximation_for_monotone` in `Theorem7Universal.lean` | `tests/properties/test_theorem7_uat.py` |

## Building the proofs locally

```bash
cd proofs
lake exe cache get
lake build
```

Expected runtime: under 5 minutes if the mathlib4 cache is warm, ~15 minutes
on a cold cache.

## Doc-gen4 HTML

The CI job uploads a `doc-gen4`-rendered HTML view of every module as a
workflow artifact named `lean-docs`. Download from any successful Lean
workflow run on the project's Actions page. Hosting the rendered HTML at a
stable URL is a follow-up deliverable.
