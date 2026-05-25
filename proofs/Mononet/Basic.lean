import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Fin.Basic

/-!
# Basic types for the constrained monotonic neural network construction

Definitions only — no theorems. The paper's notation is reproduced as Lean
notation where Unicode-friendly.

Paper: Runje & Shankaranarayana, *Constrained Monotonic Neural Networks*,
ICML 2023 — <https://arxiv.org/abs/2205.11775>.
-/

namespace Mononet

/-- The three possible monotonicity directions for a single input feature.
Mirrors `MonotonicityMask.values ∈ {-1, 0, +1}` in
[`mononet/core/types.py`](../../../mononet/core/types.py). -/
inductive Sign
  | pos
  | zero
  | neg
  deriving Repr, DecidableEq

namespace Sign

/-- Map a sign to its integer in `{+1, 0, -1}`. -/
def toInt : Sign → ℤ
  | .pos  =>  1
  | .zero =>  0
  | .neg  => -1

end Sign

/-- An `n`-feature monotonicity mask (Definition before eq. 1 in the paper).
`t i = .pos` (resp. `.neg`) means the output should be non-decreasing
(resp. non-increasing) in input `i`; `.zero` imposes no constraint. -/
abbrev MonoMask (n : ℕ) := Fin n → Sign

/-- An activation-split for a layer of width `m` neurons.
`s_breve + s_hat + s_tilde = m`. The three buckets receive `ρ̆`, `ρ̂`, and `ρ̃`
respectively (Definition 3 in the paper). -/
structure ActivationSplit (m : ℕ) where
  s_breve  : ℕ
  s_hat    : ℕ
  s_tilde  : ℕ
  sum_eq   : s_breve + s_hat + s_tilde = m
  deriving Repr

/-- The masked-weight operation `|W|_t` from eq. (2) of the paper, applied
element-wise to a real matrix. -/
def masked {n m : ℕ} (W : Matrix (Fin n) (Fin m) ℝ) (t : MonoMask n) :
    Matrix (Fin n) (Fin m) ℝ :=
  fun i j =>
    match t i with
    | .pos  =>  |W i j|
    | .neg  => -|W i j|
    | .zero =>  W i j

theorem masked_pos {n m : ℕ} (W : Matrix (Fin n) (Fin m) ℝ) (t : MonoMask n)
    (i : Fin n) (j : Fin m) (h : t i = .pos) :
    masked W t i j = |W i j| := by
  simp [masked, h]

theorem masked_neg {n m : ℕ} (W : Matrix (Fin n) (Fin m) ℝ) (t : MonoMask n)
    (i : Fin n) (j : Fin m) (h : t i = .neg) :
    masked W t i j = -|W i j| := by
  simp [masked, h]

theorem masked_zero {n m : ℕ} (W : Matrix (Fin n) (Fin m) ℝ) (t : MonoMask n)
    (i : Fin n) (j : Fin m) (h : t i = .zero) :
    masked W t i j = W i j := by
  simp [masked, h]

end Mononet
