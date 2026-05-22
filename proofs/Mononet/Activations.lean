import Mathlib.Analysis.Convex.Function
import Mathlib.Tactic.Linarith
import Mathlib.Topology.Algebra.Order.Field
import Mathlib.Topology.Instances.Real
import Mononet.Basic

/-!
# Base activation class `Ă` and the derived activations `ρ̂`, `ρ̃`

`BaseActivation` bundles the four properties of `Ă` from the paper:
zero-centred, monotone-increasing, convex, and lower-bounded. From a
`BaseActivation`, the concave reflection `ρ̂` and the saturated piecewise
`ρ̃` are defined (eqs. 4 and 5).
-/

open Filter Topology

namespace Mononet

/-- The class `Ă` of zero-centred, monotone-increasing, convex,
lower-bounded functions ℝ → ℝ.

A function `f` is in `Ă` iff it satisfies all four predicates:
- `f 0 = 0` (zero-centred)
- `Monotone f`
- `ConvexOn ℝ Set.univ f`
- `∃ L, ∀ x, L ≤ f x` (lower-bounded)
-/
structure BaseActivation where
  f : ℝ → ℝ
  zero_centred : f 0 = 0
  monotone     : Monotone f
  convex       : ConvexOn ℝ Set.univ f
  lower_bound  : ℝ
  bounded_below : ∀ x : ℝ, lower_bound ≤ f x
  -- New: f attains its infimum at -∞ (true for ReLU, ELU, SELU, GELU)
  tendsto_lower : Filter.Tendsto f Filter.atBot (𝓝 lower_bound)

namespace BaseActivation

variable (φ : BaseActivation)

/-- The concave reflection `ρ̂(x) = -ρ̆(-x)`. Eq. 4. -/
def concaveReflection : ℝ → ℝ := fun x => -(φ.f (-x))

notation:max "ρ̂[" φ "]" => BaseActivation.concaveReflection φ

/-- The saturated piecewise activation `ρ̃` from eq. 5. -/
noncomputable def saturated : ℝ → ℝ := fun x =>
  if x < 0 then φ.f (x + 1) - φ.f 1
  else      φ.concaveReflection (x - 1) + φ.f 1

notation:max "ρ̃[" φ "]" => BaseActivation.saturated φ

/-- The base activation passes through zero. (Trivially the structure field,
restated here as a theorem for symmetry with `concaveReflection_at_zero`.) -/
@[simp] theorem f_zero : φ.f 0 = 0 := φ.zero_centred

/-- The concave reflection also passes through zero. -/
@[simp] theorem concaveReflection_at_zero : φ.concaveReflection 0 = 0 := by
  unfold concaveReflection
  simp [φ.zero_centred]

/-- The concave reflection is monotone. -/
theorem concaveReflection_monotone : Monotone φ.concaveReflection := by
  intro a b hab
  unfold concaveReflection
  have h1 : -b ≤ -a := neg_le_neg hab
  have h2 : φ.f (-b) ≤ φ.f (-a) := φ.monotone h1
  exact neg_le_neg h2

/-- The saturated activation is monotone (combined L/R branches). -/
theorem saturated_monotone : Monotone φ.saturated := by
  intro a b hab
  unfold saturated
  split_ifs with ha hb
  · -- a < 0, b < 0: both left branch
    have hle : a + 1 ≤ b + 1 := by linarith
    have h2 : φ.f (a + 1) ≤ φ.f (b + 1) := φ.monotone hle
    linarith
  · -- a < 0, b ≥ 0: cross branch
    have left_le : φ.f (a + 1) ≤ φ.f 1 := φ.monotone (by linarith)
    have right_ge : φ.concaveReflection (-1) ≤ φ.concaveReflection (b - 1) :=
      φ.concaveReflection_monotone (by linarith)
    have hcr : φ.concaveReflection (-1) = -(φ.f 1) := by
      unfold concaveReflection; ring
    linarith
  · -- a ≥ 0, b < 0: impossible since a ≤ b
    push_neg at ha
    linarith
  · -- a ≥ 0, b ≥ 0: both right branch
    have hle : a - 1 ≤ b - 1 := by linarith
    have h2 : φ.concaveReflection (a - 1) ≤ φ.concaveReflection (b - 1) :=
      φ.concaveReflection_monotone hle
    linarith

end BaseActivation
end Mononet
