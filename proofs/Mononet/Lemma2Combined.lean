import Mathlib.Analysis.Convex.Function
import Mononet.Layers
import Mononet.Lemma1Mono

/-!
# Lemma 2 — combined activation preserves monotonicity (and tracks convexity)

**Paper:** Lemma 2 (and Corollary 3), Runje & Shankaranarayana 2023.

**Empirical counterpart:** `tests/properties/test_lemma2_combined_mono.py`.
-/

namespace Mononet

variable {m : ℕ}

/-- **Lemma 2 (monotone part).** Each output component of `combined φ s` is
monotone in its corresponding input component. -/
theorem combined_monotone_componentwise
    (φ : BaseActivation) (s : ActivationSplit m) (j : Fin m) :
    Monotone (fun (h : ℝ) =>
      combined φ s (fun k => if k = j then h else 0) j) := by
  intro a b hab
  unfold combined
  -- Three cases on the bucket of `j`.
  by_cases h1 : (j : ℕ) < s.s_breve
  · simp [h1]; exact φ.monotone hab
  · by_cases h2 : (j : ℕ) < s.s_breve + s.s_hat
    · simp [h1, h2]; exact φ.concaveReflection_monotone hab
    · simp [h1, h2]; exact φ.saturated_monotone hab

/-- **Lemma 2 (convex part).** If the activation split puts every neuron in
the convex bucket, every output component is convex in its corresponding
input. -/
theorem combined_convex_of_all_breve
    (φ : BaseActivation) (s : ActivationSplit m)
    (h_breve : s.s_breve = m) (j : Fin m) :
    ConvexOn ℝ Set.univ (fun (h : ℝ) =>
      combined φ s (fun k => if k = j then h else 0) j) := by
  -- Under `s.s_breve = m`, every `j` lies in the first bucket and gets `φ.f`.
  have h_bucket : (j : ℕ) < s.s_breve := by
    rw [h_breve]; exact j.isLt
  -- Reduce to the convexity of `φ.f`.
  have h_eq : (fun (h : ℝ) =>
                  combined φ s (fun k => if k = j then h else 0) j) = φ.f := by
    funext h
    unfold combined; simp [h_bucket]
  rw [h_eq]
  exact φ.convex

/-- **Lemma 2 (concave part).** If the activation split puts every neuron in
the concave bucket, every output component is concave in its corresponding
input. -/
theorem combined_concave_of_all_hat
    (φ : BaseActivation) (s : ActivationSplit m)
    (h_hat : s.s_hat = m) (h_breve_zero : s.s_breve = 0) (j : Fin m) :
    ConcaveOn ℝ Set.univ (fun (h : ℝ) =>
      combined φ s (fun k => if k = j then h else 0) j) := by
  -- Under these hypotheses, every `j` lies in the second bucket and gets `ρ̂`.
  have h_not_first : ¬ ((j : ℕ) < s.s_breve) := by
    rw [h_breve_zero]; omega
  have h_second : (j : ℕ) < s.s_breve + s.s_hat := by
    rw [h_breve_zero, h_hat, Nat.zero_add]; exact j.isLt
  have h_eq : (fun (h : ℝ) =>
                  combined φ s (fun k => if k = j then h else 0) j)
              = φ.concaveReflection := by
    funext h
    unfold combined; simp [h_not_first, h_second]
  rw [h_eq]
  -- ρ̂(x) = -φ.f(-x).  We prove ConcaveOn ℝ Set.univ φ.concaveReflection
  -- by writing it as (-φ.f) ∘ (-LinearMap.id) and using:
  --   φ.convex.neg : ConcaveOn ℝ Set.univ (-φ.f)
  --   ConcaveOn.comp_linearMap : compose with a linear map
  unfold BaseActivation.concaveReflection
  -- Rewrite as composition: (fun x => -(φ.f (-x))) = (- φ.f) ∘ (- LinearMap.id)
  have h_fun_eq : (fun x : ℝ => -(φ.f (-x))) = (fun x => -φ.f x) ∘ ((-LinearMap.id : ℝ →ₗ[ℝ] ℝ)) := by
    funext x
    simp [LinearMap.neg_apply, LinearMap.id_apply]
  rw [h_fun_eq]
  -- ConcaveOn of (-φ.f) on Set.univ
  have h_neg_convex : ConcaveOn ℝ Set.univ (fun x => -φ.f x) := by
    have : ConcaveOn ℝ Set.univ (-φ.f) := φ.convex.neg
    exact this
  -- Apply comp_linearMap: gives ConcaveOn on ((-id) ⁻¹' Set.univ) = Set.univ
  have h_comp := h_neg_convex.comp_linearMap (-LinearMap.id : ℝ →ₗ[ℝ] ℝ)
  simp [Set.preimage_univ] at h_comp
  exact h_comp

/-- **Corollary 3.** CMFCL inherits Lemma 1's monotonicity and Lemma 2's
componentwise monotonicity. Stated here as: CMFCL with `t i = .pos` is
non-decreasing in input feature `i`. -/
theorem CMFCL_props
    {n m : ℕ}
    (W : Matrix (Fin n) (Fin m) ℝ) (b : Fin m → ℝ)
    (t : MonoMask n) (φ : BaseActivation) (s : ActivationSplit m)
    (i : Fin n) (j : Fin m) (h_pos : t i = .pos)
    (x x' : Fin n → ℝ) (h_eq : ∀ k ≠ i, x k = x' k) (h_le : x i ≤ x' i) :
    CMFCL W b t φ s x j ≤ CMFCL W b t φ s x' j := by
  unfold CMFCL
  -- Step 1: the constrained linear part is monotone in feature i (Lemma 1).
  have h_lin :
      constrainedLinear W b t x j ≤ constrainedLinear W b t x' j :=
    constrainedLinear_monotone_pos W b t i j h_pos x x' h_eq h_le
  -- Step 2: combined is monotone componentwise (Lemma 2, monotone part).
  -- Each component of combined depends only on its own input.
  unfold combined
  by_cases h1 : (j : ℕ) < s.s_breve
  · simp [h1]; exact φ.monotone h_lin
  · by_cases h2 : (j : ℕ) < s.s_breve + s.s_hat
    · simp [h1, h2]; exact φ.concaveReflection_monotone h_lin
    · simp [h1, h2]; exact φ.saturated_monotone h_lin

end Mononet
