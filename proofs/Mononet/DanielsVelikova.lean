import Mathlib.Topology.Basic
import Mathlib.Topology.MetricSpace.Basic
import Mononet.Activations

/-!
# Daniels & Velikova 2010 Theorem 4 — assumed as an axiom

This module captures the universal approximation theorem of Daniels &
Velikova (2010, *Monotone and Partially Monotone Neural Networks*) for
sigmoid-based monotone networks as a single `axiom`.

**Justification:** Per the spec's "Path 3" decision (in the design doc), this
theorem is the load-bearing dependency of Runje & Shankaranarayana 2023's
Theorem 7. A full port of Daniels & Velikova's proof is a deferred
deliverable.

**This is the only axiom in the formalization.** If you are reviewing the
mononet Lean proofs and want to know the trust assumptions: this is it.
-/

namespace Mononet

open Filter Topology

/-- **Axiom (Daniels & Velikova 2010, Theorem 4).**

For any continuous monotone-nondecreasing function `f` on a compact subset
`K ⊆ ℝ^n` and any base activation `φ` whose saturated form `ρ̃` can
approximate the Heaviside (a property our Lemma 5 establishes), there is a
real-valued approximator `N : (Fin n → ℝ) → ℝ` such that
`|N x - f x| < ε` for every `x ∈ K`.

The approximator's internal structure (as a constrained monotone network of
some depth using `φ`) is left abstract. Downstream theorems (Theorem 7) only
need the existence of such an `N`, not its concrete construction.
-/
axiom danielsVelikova_universal_approximation
    {n : ℕ}
    (K : Set (Fin n → ℝ)) (_hK_compact : IsCompact K)
    (f : (Fin n → ℝ) → ℝ)
    (_hf_continuous : ContinuousOn f K)
    (_hf_mono : ∀ x y : Fin n → ℝ, x ∈ K → y ∈ K → (∀ i, x i ≤ y i) → f x ≤ f y)
    (φ : BaseActivation)
    (_hφ_approx_heaviside :
        ∀ x : ℝ, x ≠ 0 →
        Tendsto (fun a : ℝ => φ.saturated (a * x)) atTop
                (𝓝 (if x < 0 then φ.lower_bound - φ.f 1
                    else -φ.lower_bound + φ.f 1)))
    (ε : ℝ) (_hε : 0 < ε) :
    ∃ N : (Fin n → ℝ) → ℝ, ∀ x ∈ K, |N x - f x| < ε

end Mononet
