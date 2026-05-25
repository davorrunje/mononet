import Mononet.Layers

/-!
# Lemma 1 — sign of partial derivatives of the constrained linear layer

**Paper:** Lemma 1, Runje & Shankaranarayana 2023 (after Definition 1).

**Empirical counterpart:** `tests/properties/test_lemma1_constrained_linear_mono.py`
in the mononet Python repo (added in Sub-project A's implementation plan).
-/

namespace Mononet

variable {n m : ℕ}

/-- For a `.pos` input feature, the masked weight is `≥ 0`. -/
theorem masked_nonneg_of_pos
    (W : Matrix (Fin n) (Fin m) ℝ) (t : MonoMask n)
    (i : Fin n) (j : Fin m) (h : t i = .pos) :
    0 ≤ masked W t i j := by
  rw [masked_pos W t i j h]
  exact abs_nonneg _

/-- For a `.neg` input feature, the masked weight is `≤ 0`. -/
theorem masked_nonpos_of_neg
    (W : Matrix (Fin n) (Fin m) ℝ) (t : MonoMask n)
    (i : Fin n) (j : Fin m) (h : t i = .neg) :
    masked W t i j ≤ 0 := by
  rw [masked_neg W t i j h]
  linarith [abs_nonneg (W i j)]

/-- **Lemma 1, positive form.** The constrained linear layer is non-decreasing
in input feature `i` whenever `t i = .pos`, holding all other features fixed.

Formally: replacing `x i` with `x'` larger than the original `x i` only
increases each output `h j`. -/
theorem constrainedLinear_monotone_pos
    (W : Matrix (Fin n) (Fin m) ℝ) (b : Fin m → ℝ) (t : MonoMask n)
    (i : Fin n) (j : Fin m) (h_pos : t i = .pos)
    (x x' : Fin n → ℝ) (h_eq : ∀ k ≠ i, x k = x' k) (h_le : x i ≤ x' i) :
    constrainedLinear W b t x j ≤ constrainedLinear W b t x' j := by
  unfold constrainedLinear
  have h_mask : 0 ≤ masked W t i j := masked_nonneg_of_pos W t i j h_pos
  -- Identical sums except for the term at `i`.
  have h_sum_le : (∑ k, masked W t k j * x k) ≤ (∑ k, masked W t k j * x' k) := by
    apply Finset.sum_le_sum
    intro k _
    by_cases hk : k = i
    · subst hk
      exact mul_le_mul_of_nonneg_left h_le h_mask
    · rw [h_eq k hk]
  linarith

/-- **Lemma 1, negative form.** The constrained linear layer is
non-increasing in input feature `i` whenever `t i = .neg`. -/
theorem constrainedLinear_antitone_neg
    (W : Matrix (Fin n) (Fin m) ℝ) (b : Fin m → ℝ) (t : MonoMask n)
    (i : Fin n) (j : Fin m) (h_neg : t i = .neg)
    (x x' : Fin n → ℝ) (h_eq : ∀ k ≠ i, x k = x' k) (h_le : x i ≤ x' i) :
    constrainedLinear W b t x' j ≤ constrainedLinear W b t x j := by
  unfold constrainedLinear
  have h_mask : masked W t i j ≤ 0 := masked_nonpos_of_neg W t i j h_neg
  have h_sum_le : (∑ k, masked W t k j * x' k) ≤ (∑ k, masked W t k j * x k) := by
    apply Finset.sum_le_sum
    intro k _
    by_cases hk : k = i
    · subst hk
      -- masked ≤ 0, x i ≤ x' i ⇒ masked * x' i ≤ masked * x i
      exact mul_le_mul_of_nonpos_left h_le h_mask
    · rw [h_eq k hk]
  linarith

end Mononet
