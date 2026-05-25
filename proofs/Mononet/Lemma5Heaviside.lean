import Mathlib.Topology.Algebra.Order.Field
import Mathlib.Topology.Instances.Real
import Mathlib.Topology.Algebra.Group.Basic
import Mononet.Activations

/-!
# Lemma 5 — Heaviside approximation by an affine rescaling of `ρ̃`

**Paper:** Lemma 5, Runje & Shankaranarayana 2023.

**Empirical counterpart:** `tests/properties/test_lemma5_heaviside_limit.py`.

**Formal statement.** The Heaviside function

  H(x) = 1   if x ≥ 0
        = 0   if x < 0

is the pointwise limit of `φ.saturated (a · x)` as `a → ∞`, rescaled to the
lower/upper bounds of `φ.saturated`. Specifically:

- For `x < 0`: `φ.saturated (a · x) → φ.lower_bound - φ.f 1` as `a → ∞`.
- For `x > 0`: `φ.saturated (a · x) → -φ.lower_bound + φ.f 1` as `a → ∞`.

These two limits correspond (after affine normalization) to the 0 and 1 values
of the Heaviside. The `tendsto_lower` field of `BaseActivation` — which asserts
that `φ.f` attains its infimum `φ.lower_bound` at `-∞` — is the key hypothesis.
-/

namespace Mononet

open Filter Topology

variable (φ : BaseActivation)

/-- Heaviside function (right-continuous form). -/
noncomputable def heaviside : ℝ → ℝ := fun x => if x < 0 then 0 else 1

/-- The saturated activation `ρ̃` is bounded below by `φ.lower_bound - φ.f 1`. -/
theorem saturated_bounded_below :
    ∃ L : ℝ, ∀ x : ℝ, L ≤ φ.saturated x := by
  refine ⟨φ.lower_bound - φ.f 1, ?_⟩
  intro x
  unfold BaseActivation.saturated
  by_cases hx : x < 0
  · simp only [hx, if_true]
    have := φ.bounded_below (x + 1)
    linarith
  · simp only [hx, if_false]
    have hmono :
        φ.concaveReflection (-1) ≤ φ.concaveReflection (x - 1) :=
      φ.concaveReflection_monotone (by linarith)
    have hcr : φ.concaveReflection (-1) = -φ.f 1 := by
      unfold BaseActivation.concaveReflection; ring
    have h_bound_le_one : φ.lower_bound ≤ φ.f 1 := by
      have := φ.bounded_below 1; linarith
    linarith

/-- The saturated activation `ρ̃` is bounded above on any compact interval `[-M, M]`. -/
theorem saturated_bounded_on_compact (M : ℝ) :
    ∃ U : ℝ, ∀ x ∈ Set.Icc (-M) M, φ.saturated x ≤ U := by
  refine ⟨max (φ.f (M + 1)) (φ.concaveReflection (M - 1) + φ.f 1), ?_⟩
  intro x hx
  obtain ⟨hxl, hxr⟩ := hx
  unfold BaseActivation.saturated
  by_cases hx0 : x < 0
  · simp only [hx0, if_true]
    have hmono : φ.f (x + 1) ≤ φ.f (M + 1) := φ.monotone (by linarith)
    have hle : φ.f (x + 1) - φ.f 1 ≤ φ.f (M + 1) - φ.f 1 := by linarith
    have hone : φ.f 1 ≥ 0 := by
      have := φ.monotone (show (0 : ℝ) ≤ 1 by norm_num)
      simp [φ.zero_centred] at this
      linarith
    calc φ.f (x + 1) - φ.f 1
        ≤ φ.f (M + 1) - φ.f 1 := hle
      _ ≤ φ.f (M + 1) := by linarith
      _ ≤ max (φ.f (M + 1)) (φ.concaveReflection (M - 1) + φ.f 1) := le_max_left _ _
  · simp only [hx0, if_false]
    have h1 : φ.concaveReflection (x - 1) ≤ φ.concaveReflection (M - 1) :=
      φ.concaveReflection_monotone (by linarith)
    calc φ.concaveReflection (x - 1) + φ.f 1
        ≤ φ.concaveReflection (M - 1) + φ.f 1 := by linarith
      _ ≤ max (φ.f (M + 1)) (φ.concaveReflection (M - 1) + φ.f 1) := le_max_right _ _

/-- **Lemma 5 (left-branch limit).** For `x < 0`, the saturated activation
`φ.saturated (a · x)` converges to `φ.lower_bound - φ.f 1` as `a → ∞`.

The proof: as `a → ∞` with `x < 0`, `a · x → -∞`, so for large enough `a`,
`a · x < 0` and the left branch of `ρ̃` applies:
`ρ̃(a · x) = φ.f(a · x + 1) - φ.f 1`.
Since `a · x + 1 → -∞` and `φ.f` tends to `φ.lower_bound` at `-∞`
(the `tendsto_lower` hypothesis), the result follows. -/
theorem saturated_approximates_heaviside_left
    (x : ℝ) (hx : x < 0) :
    Tendsto (fun a : ℝ => φ.saturated (a * x)) atTop
            (𝓝 (φ.lower_bound - φ.f 1)) := by
  have h1 : Tendsto (fun a : ℝ => a * x) atTop atBot :=
    (tendsto_mul_const_atBot_of_neg hx).mpr tendsto_id
  have h2 : Tendsto (fun a : ℝ => a * x + 1) atTop atBot :=
    tendsto_atBot_add_const_right atTop (1 : ℝ) h1
  have hf : Tendsto (fun a : ℝ => φ.f (a * x + 1)) atTop (𝓝 φ.lower_bound) :=
    φ.tendsto_lower.comp h2
  have h_eventually : (fun a : ℝ => φ.saturated (a * x)) =ᶠ[atTop]
                     (fun a : ℝ => φ.f (a * x + 1) - φ.f 1) := by
    filter_upwards [eventually_ge_atTop (-2 / x)] with a ha
    have h_ax : a * x < 0 := by
      have h1' : -2 / x > 0 := div_pos_of_neg_of_neg (by norm_num) hx
      have h2' : a * x ≤ -2 / x * x := mul_le_mul_of_nonpos_right ha (le_of_lt hx)
      have h3' : -2 / x * x = -2 := by
        rw [div_mul_cancel₀ (-2 : ℝ) (ne_of_lt hx)]
      linarith
    unfold BaseActivation.saturated
    simp only [h_ax, if_true]
  exact Tendsto.congr' h_eventually.symm (hf.sub_const (φ.f 1))

/-- **Lemma 5 (right-branch limit).** For `x > 0`, the saturated activation
`φ.saturated (a · x)` converges to `-φ.lower_bound + φ.f 1` as `a → ∞`.

The proof: as `a → ∞` with `x > 0`, `a · x → +∞`, so for large enough `a`,
`a · x ≥ 0` and the right branch of `ρ̃` applies:
`ρ̃(a · x) = ρ̂(a · x - 1) + φ.f 1 = -(φ.f(-(a · x - 1))) + φ.f 1`.
Since `-(a · x - 1) → -∞` and `φ.f` tends to `φ.lower_bound` at `-∞`, we get
`φ.f(-(a · x - 1)) → φ.lower_bound`, so the negation goes to `-φ.lower_bound`
and the full expression to `-φ.lower_bound + φ.f 1`. -/
theorem saturated_approximates_heaviside_right
    (x : ℝ) (hx : 0 < x) :
    Tendsto (fun a : ℝ => φ.saturated (a * x)) atTop
            (𝓝 (-φ.lower_bound + φ.f 1)) := by
  have h1 : Tendsto (fun a : ℝ => a * x) atTop atTop :=
    (tendsto_mul_const_atTop_of_pos hx).mpr tendsto_id
  have h2 : Tendsto (fun a : ℝ => a * x - 1) atTop atTop := by
    have := tendsto_atTop_add_const_right atTop (-1 : ℝ) h1
    simpa using this
  have h3 : Tendsto (fun a : ℝ => -(a * x - 1)) atTop atBot :=
    tendsto_neg_atTop_atBot.comp h2
  have hf : Tendsto (fun a : ℝ => φ.f (-(a * x - 1))) atTop (𝓝 φ.lower_bound) :=
    φ.tendsto_lower.comp h3
  have hneg : Tendsto (fun a : ℝ => -(φ.f (-(a * x - 1)))) atTop (𝓝 (-φ.lower_bound)) :=
    hf.neg
  have h_eventually : (fun a : ℝ => φ.saturated (a * x)) =ᶠ[atTop]
                     (fun a : ℝ => -(φ.f (-(a * x - 1))) + φ.f 1) := by
    filter_upwards [eventually_ge_atTop (2 / x)] with a ha
    have h_ax : ¬ (a * x < 0) := by
      have h_mul : (2 / x) * x = 2 := div_mul_cancel₀ 2 (ne_of_gt hx)
      have h2' : a * x ≥ 2 := by
        calc a * x ≥ (2 / x) * x := by nlinarith
          _ = 2 := h_mul
      linarith
    unfold BaseActivation.saturated BaseActivation.concaveReflection
    simp only [h_ax, if_false]
  exact Tendsto.congr' h_eventually.symm (hneg.add_const (φ.f 1))

end Mononet
