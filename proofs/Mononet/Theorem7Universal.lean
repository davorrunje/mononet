import Mononet.DanielsVelikova
import Mononet.Lemma5Heaviside
import Mononet.Lemma6Equiv

/-!
# Theorem 7 — universal approximation for constrained monotone networks

**Paper:** Theorem 7, Runje & Shankaranarayana 2023.

**Empirical counterpart:** `tests/properties/test_theorem7_uat.py`
(a smoke test that fits a small mononet to a known monotone function).

This theorem inherits the Daniels & Velikova 2010 Theorem 4 dependency
captured as an axiom in `Mononet.DanielsVelikova`.
-/

namespace Mononet

open Filter Topology

/-- **Theorem 7.** Any continuous monotone-nondecreasing function on a
compact subset of `ℝ^k` can be approximated uniformly by an approximator
built using any `BaseActivation` `φ` (in particular, those built on ReLU,
ELU, SELU, GELU).

This is the load-bearing universal-approximation result of Runje &
Shankaranarayana 2023. Its proof composes:
- `Lemma5Heaviside`: `ρ̃[φ]` approximates the Heaviside (both branches).
- The Daniels & Velikova 2010 axiom (`Mononet.danielsVelikova_universal_approximation`),
  which gives the corresponding result for any activation that approximates
  the Heaviside.
-/
theorem universal_approximation_for_monotone
    {k : ℕ}
    (K : Set (Fin k → ℝ)) (hK_compact : IsCompact K)
    (f : (Fin k → ℝ) → ℝ)
    (hf_continuous : ContinuousOn f K)
    (hf_mono : ∀ x y : Fin k → ℝ, x ∈ K → y ∈ K → (∀ i, x i ≤ y i) → f x ≤ f y)
    (φ : BaseActivation)
    (ε : ℝ) (hε : 0 < ε) :
    ∃ N : (Fin k → ℝ) → ℝ, ∀ x ∈ K, |N x - f x| < ε := by
  -- Step 1: produce the Heaviside-approximation hypothesis from Lemma 5.
  have h_heaviside :
      ∀ x : ℝ, x ≠ 0 →
        Tendsto (fun a : ℝ => φ.saturated (a * x)) atTop
                (𝓝 (if x < 0 then φ.lower_bound - φ.f 1
                    else -φ.lower_bound + φ.f 1)) := by
    intro x hx
    by_cases h : x < 0
    · simp only [h, if_true]
      exact saturated_approximates_heaviside_left φ x h
    · simp only [h, if_false]
      have hpos : 0 < x := lt_of_le_of_ne (le_of_not_lt h) (Ne.symm hx)
      exact saturated_approximates_heaviside_right φ x hpos
  -- Step 2: apply the axiom.
  exact danielsVelikova_universal_approximation
    K hK_compact f hf_continuous hf_mono φ h_heaviside ε hε

end Mononet
