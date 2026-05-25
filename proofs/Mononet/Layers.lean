import Mononet.Basic
import Mononet.Activations

/-!
# Constrained linear layer, combined activation, and the CMFCL composite

Eqs. 3 (constrained linear layer), 6 (combined activation `ρ_s`), and 7
(constrained monotone fully connected layer = CMFCL) from the paper.
-/

namespace Mononet

/-- The constrained linear layer of Definition 1:
`h = (|W|_t)ᵀ · x + b`. Returns a vector in `ℝ^m` (indexed by `Fin m`). -/
def constrainedLinear {n m : ℕ}
    (W : Matrix (Fin n) (Fin m) ℝ)
    (b : Fin m → ℝ)
    (t : MonoMask n)
    (x : Fin n → ℝ) : Fin m → ℝ :=
  fun j => (∑ i, masked W t i j * x i) + b j

/-- The combined activation `ρ_s` from Definition 3.
Splits the layer width `m` into three buckets `(s̆, ŝ, s̃)` and applies a
different activation to each. Indexing on the buckets is by an explicit
condition on `j : Fin m`. -/
noncomputable def combined {m : ℕ} (φ : BaseActivation) (s : ActivationSplit m)
    (h : Fin m → ℝ) : Fin m → ℝ :=
  fun j =>
    if (j : ℕ) < s.s_breve then
      φ.f (h j)
    else if (j : ℕ) < s.s_breve + s.s_hat then
      φ.concaveReflection (h j)
    else
      φ.saturated (h j)

/-- The Constrained Monotone Fully Connected Layer (CMFCL) of Definition 4.
This is the layer the paper builds the whole network out of: a constrained
linear layer composed with the combined activation. -/
noncomputable def CMFCL {n m : ℕ}
    (W : Matrix (Fin n) (Fin m) ℝ)
    (b : Fin m → ℝ)
    (t : MonoMask n)
    (φ : BaseActivation)
    (s : ActivationSplit m)
    (x : Fin n → ℝ) : Fin m → ℝ :=
  combined φ s (constrainedLinear W b t x)

end Mononet
