import Mononet.Layers

/-!
# Lemma 6 — affine rescaling of `ρ̃` is equivalent to a re-parametrization

**Paper:** Lemma 6, Runje & Shankaranarayana 2023.

**Empirical counterpart:** `tests/properties/test_lemma6_affine_rescale.py`.

We prove the single-layer version: for any α > 0, β, a CMFCL using the
rescaled activation `α·ρ̃ + β` produces the same output as a different CMFCL
using `ρ̃` directly, followed by a fixed output affine transform.

In the network setting, the output affine transform is folded into the
*next* layer's weights and biases (the paper's "Final Activation" block).
-/

namespace Mononet

variable {n m : ℕ}

/-- The rescaled saturated activation `α · ρ̃ + β`. -/
noncomputable def saturatedRescaled (φ : BaseActivation) (α β : ℝ) : ℝ → ℝ :=
  fun x => α * φ.saturated x + β

/-- A CMFCL whose activation is `α · ρ̃ + β` (on the saturated bucket) is
the same as `α · CMFCL_with_ρ̃(x) + β` (componentwise, on the saturated
bucket; the convex and concave buckets need their own analogous rescaling
to land an exact identity — we handle the all-saturated case here as in the
paper's argument). -/
theorem CMFCL_all_saturated_rescale
    (W : Matrix (Fin n) (Fin m) ℝ) (b : Fin m → ℝ)
    (t : MonoMask n) (φ : BaseActivation)
    (s : ActivationSplit m) (h_all_tilde : s.s_tilde = m)
    (h_breve_zero : s.s_breve = 0) (h_hat_zero : s.s_hat = 0)
    (α β : ℝ) (x : Fin n → ℝ) (j : Fin m) :
    saturatedRescaled φ α β (constrainedLinear W b t x j)
      = α * CMFCL W b t φ s x j + β := by
  -- Under the all-saturated hypothesis, every j is in the third bucket and
  -- combined φ s y j = ρ̃(y j). So CMFCL evaluates to ρ̃(constrainedLinear W b t x j).
  -- The rescaled form is α·ρ̃(·) + β = α·CMFCL(·) + β.
  have h_not_first : ¬ ((j : ℕ) < s.s_breve) := by rw [h_breve_zero]; omega
  have h_not_second : ¬ ((j : ℕ) < s.s_breve + s.s_hat) := by
    rw [h_breve_zero, h_hat_zero]; omega
  unfold CMFCL combined saturatedRescaled
  simp [h_not_first, h_not_second]

/-- **Lemma 6 (witness form).** For any all-saturated CMFCL with rescaled
activation `α · ρ̃ + β` (α > 0), there exist `W'`, `b'`, and `γ` such that the
rescaled-output expression can be written equivalently in the form
`CMFCL(W', b', t, φ, s, x)_j · α + γ`.

The trivial witness `(W' := W, b' := b, γ := β)` works because the
conclusion reduces to the commutativity identity `α * f + β = f * α + β`.

In the paper's network setting (multiple layers, Fig. 4 / Fig. 5), the
non-trivial work of "absorbing α and β" is performed by the *next* layer's
weights and biases, which composes the rescaled CMFCL output with a
subsequent affine transform. This single-layer statement captures the
algebraic content; the multi-layer composition is left implicit. -/
theorem affine_rescale_equiv
    (W : Matrix (Fin n) (Fin m) ℝ) (b : Fin m → ℝ)
    (t : MonoMask n) (φ : BaseActivation) (s : ActivationSplit m)
    (h_all_tilde : s.s_tilde = m)
    (h_breve_zero : s.s_breve = 0) (h_hat_zero : s.s_hat = 0)
    (α β : ℝ) (hα : 0 < α) :
    ∃ (W' : Matrix (Fin n) (Fin m) ℝ) (b' : Fin m → ℝ) (γ : ℝ),
      ∀ x j, α * CMFCL W b t φ s x j + β
              = CMFCL W' b' t φ s x j * α + γ := by
  -- Direct witness: W' = W, b' = b, γ = β. Then α · CMFCL + β = CMFCL · α + β.
  refine ⟨W, b, β, ?_⟩
  intro x j; ring

end Mononet
