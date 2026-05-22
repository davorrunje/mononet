# Sub-project E — Lean 4 formalization of the paper's theorems

**Date:** 2026-05-22
**Author:** Davor Runje
**Status:** Draft (brainstorming output); pending user review.
**Parent spec:** [`2026-05-21-mononet-package-design.md`](2026-05-21-mononet-package-design.md)
**Independence:** This sub-project has no code dependencies on Sub-projects A–D; it can start immediately and progress in parallel.
**Paper:** Runje & Shankaranarayana, *Constrained Monotonic Neural Networks*, ICML 2023 — <https://arxiv.org/abs/2205.11775>

## 1. Goals & non-goals

### Goals

- Formally prove the paper's three core lemmas (1, 2, 5), Corollary 3, Lemma 6, and the main universal-approximation theorem (Theorem 7) in **Lean 4** with **mathlib4**.
- Publish the proofs as part of the `mononet` project so they are versioned alongside the implementation and citable from the docs.
- Maintain a CI job that builds the Lean project on every PR — proofs that compile today should compile tomorrow.
- Cross-reference: each lemma in the formalization links to (a) the paper statement, (b) the relevant runtime test in the Python implementation that exercises the same property empirically.
- Build a `doc-gen4` browsable HTML rendering of the proofs hosted alongside the Sphinx docs.

### Non-goals

- No formalization of *implementation correctness*. We prove the mathematical claims of the paper, not that `mononet.torch.MonoLinear`'s code matches its mathematical specification. Bridging Lean to Python execution is a research-grade endeavor and out of scope.
- No formalization of Theorem 4 from Daniels & Velikova 2010 from first principles, unless mathlib4 happens to already contain it or a near-equivalent. The paper *uses* Theorem 4 as a black box; we may have to either port their proof or assume it as a `sorry`-free hypothesis. See §6.
- No mechanized verification of the patent claims, the experimental results, or the gradient calculations of the Python implementation.
- No formalization of the flow / injectivity claims from Sub-project D (those would be a separate follow-up if pursued at all).

## 2. What gets formalized

### Definitions

| Symbol | Lean type / structure |
|---|---|
| `Ă` — class of zero-centred, monotone-increasing, convex, lower-bounded functions | A structure `BaseActivation` bundling the four properties as `Prop` hypotheses |
| `ρ̆`, `ρ̂`, `ρ̃` — base, concave reflection, saturated piecewise | `def ρ̂ (φ : BaseActivation) : ℝ → ℝ := fun x ↦ -(φ.f (-x))` etc. |
| `t : monotonicity indicator vector` | `def MonoMask (n : ℕ) := Fin n → Sign` where `Sign := {-1, 0, +1}` |
| `|M|_t` — masked weight matrix | `def masked (W : Matrix (Fin n) (Fin m) ℝ) (t : MonoMask n) : Matrix ...` |
| Constrained linear layer (Def. 1) | `def constrainedLinear (W b t) : (Fin n → ℝ) → (Fin m → ℝ)` |
| Activation split `s = (s̆, ŝ, s̃)` | `structure ActivationSplit (m : ℕ) where s_breve s_hat s_tilde : ℕ; sum_eq : s_breve + s_hat + s_tilde = m` |
| Combined activation `ρ_s` (Def. 3) | `def combined (φ : BaseActivation) (s : ActivationSplit m) : (Fin m → ℝ) → (Fin m → ℝ)` |
| Constrained monotone FC layer (Def. 4) | `def CMFCL` — the composition of `constrainedLinear` with `combined` |

The paper uses real matrices; we use `Matrix (Fin n) (Fin m) ℝ` from mathlib4.

### Theorems

| Paper | Lean theorem name | Difficulty |
|---|---|---|
| Lemma 1 | `Mononet.constrainedLinear_mono` | Easy: a few rewriting steps using `abs_nonneg` / `neg_abs_nonpos`. |
| Lemma 2 | `Mononet.combined_mono`, `Mononet.combined_convex_of_split_breve`, `Mononet.combined_concave_of_split_hat` | Easy/Medium: induction on `Fin m` partitioned by the split bounds. |
| Corollary 3 | `Mononet.CMFCL_props` | Easy once L1 + L2 are in. Just a combinator. |
| Lemma 5 | `Mononet.heaviside_approx_by_saturated` | **Medium-Hard**: requires limit arguments on the piecewise definition of `ρ̃` at `±∞`. Needs `Filter.Tendsto` machinery from mathlib4. |
| Lemma 6 | `Mononet.affine_rescale_equiv` | Medium: an existential rewriting (every `ρ̃α,β`-network has an equivalent `ρ̃`-network). Algebra-heavy. |
| Theorem 7 | `Mononet.universal_approximation_for_monotone` | **Hard**: depends on Theorem 4 (Daniels & Velikova 2010, q.v. §6). Once Theorem 4 is available, the rest is Lemmas 5 + 6 plumbing. |

The paper's Theorem 4 (universal approximation for *sigmoid-based* constrained monotone networks) is invoked as a black box in the paper's own proof. Our strategy:

- Statement of Theorem 4 is captured as `theorem Mononet.universal_sigmoid_monotone` with a `sorry`-or-axiom marker initially.
- We attempt to either port Daniels & Velikova 2010's proof (a self-contained ~3-page argument using Korovkin-style approximation + sigmoid → Heaviside limits) or to express it via a mathlib4 existing universal approximation result if any analog already exists.
- If neither path works in finite time, we leave Theorem 7 with a clearly-marked axiom on Theorem 4 — but **without `sorry`** anywhere in our own code. This is documented prominently in the proof's `README.lean` and in the paper-side docs page.

## 3. Project layout

```
proofs/                              # standalone Lake project, in-tree
├── lakefile.lean
├── lean-toolchain                   # pinned Lean 4 version
├── proofs.lean                      # entry point, imports the rest
├── Mononet/
│   ├── Basic.lean                   # MonoMask, ActivationSplit, masked, ...
│   ├── Activations.lean             # BaseActivation, ρ̂, ρ̃ definitions
│   ├── Layers.lean                  # constrainedLinear, combined, CMFCL
│   ├── Lemma1Mono.lean              # Lemma 1
│   ├── Lemma2Combined.lean          # Lemma 2 + Cor. 3
│   ├── Lemma5Heaviside.lean         # Lemma 5
│   ├── Lemma6Equiv.lean             # Lemma 6
│   ├── DanielsVelikova.lean         # Theorem 4 (axiom or full port; see §6)
│   ├── Theorem7Universal.lean       # Theorem 7
│   └── README.lean                  # Top-level documentation comment
├── tools/
│   ├── doc-gen.sh                   # build doc-gen4 HTML
│   └── build.sh                     # mathlib4 cache + lake build
└── README.md
```

The `proofs/` tree is **a separate Lake project** colocated in the repository (not vendored as a git submodule). It uses its own `lean-toolchain` file, pins a mathlib4 revision via `lake-manifest.json`, and is independent from the Python package.

Including it in the same repo keeps versioning and changelog tight; mononet vX.Y.Z's proofs are the proofs at the same commit. Splitting into a sibling repo (`mononet-lean`) was considered and rejected because every paper update would force two PRs.

## 4. Toolchain choices

| Choice | Value | Why |
|---|---|---|
| Lean version | Latest stable (4.x.y) pinned in `lean-toolchain` | Get current mathlib4 features. |
| mathlib4 revision | Pinned in `lake-manifest.json` | Reproducibility. Bump deliberately. |
| Build cache | `lake exe cache get` | Use the Lean FRO's prebuilt mathlib4 cache — keeps CI under 5 min. |
| Documentation generator | `doc-gen4` | Standard mathlib doc tooling; produces searchable HTML. |
| Linter | `mathlib4`'s own lint suite via `Mathlib.Tactic.Linter.*` | Already-recognized style. |
| Style | Follow `mathlib4` style guide for naming and proof discipline | Cultural alignment. |

## 5. CI integration

New GitHub Actions workflow `.github/workflows/lean.yml`:

```yaml
name: Lean
on:
  push: { branches: [main], paths: ["proofs/**", ".github/workflows/lean.yml"] }
  pull_request: { paths: ["proofs/**"] }

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - uses: leanprover/lean-action@v1
        with:
          lake-package-directory: proofs
          use-mathlib-cache: true
      - run: |
          cd proofs
          lake exe cache get
          lake build
```

The job's expected runtime is 3–7 minutes (mostly mathlib4 cache fetch + incremental build). It runs **only** when `proofs/**` or the workflow itself changes — Sub-projects A–D will not slow down on this.

A second optional job builds the doc-gen4 HTML and uploads it as a workflow artifact for review.

## 6. The Daniels & Velikova 2010 dependency

This is the load-bearing risk for Theorem 7. Three execution paths, decided once we've scoped the proof:

### Path 1 — Port Daniels & Velikova 2010 in full (most rigorous)

Translate the original proof (Section 3 of Daniels & Velikova 2010) into Lean 4. Estimated effort: medium (the original is ~3 pages of analysis around sigmoid → Heaviside limits and Stone-Weierstrass adaptation). If pursued, this becomes its own deliverable in the formalization.

### Path 2 — Use mathlib4's universal approximation results, if any

Search mathlib4 for `UniversalApproximation`, `Korovkin`, `denseRange`, etc. As of mid-2026 mathlib4 has growing analysis content but no general UAT formalization as of writing. If a relevant result exists, leverage it.

### Path 3 — Axiomatize Theorem 4

Capture Daniels & Velikova 2010 as a single Lean `axiom`:

```lean
axiom universal_sigmoid_monotone
  {n : ℕ} (K : Set (Fin n → ℝ)) (hK : IsCompact K)
  (f : (Fin n → ℝ) → ℝ) (hf_cont : ContinuousOn f K) (hf_mono : ...) :
  ∀ ε > 0, ∃ N : ConstrainedNet sigmoid (depth := n), ∀ x ∈ K, |N x - f x| < ε
```

Mark this clearly in `DanielsVelikova.lean`. Theorem 7 then composes with our (proved) Lemmas 5 + 6 to reach the paper's claim. The user-facing documentation states the dependency explicitly.

**Recommended order**: start with Path 3 (so Theorem 7 is *available* end-to-end immediately), then attempt Path 1 as a follow-up effort. The Python library is happy to claim "Theorem 7 mechanized in Lean (depending on Daniels & Velikova 2010 as an axiom; see `proofs/Mononet/DanielsVelikova.lean`)" while the formal port is in progress.

## 7. Crosslinks from Lean to Python and back

### Lean → Python

Each `lemma`/`theorem` in Lean carries a doc-comment block:

```lean
/--
**Paper:** Lemma 1, Runje & Shankaranarayana 2023.
**Empirical counterpart:** `tests/properties/test_constrained_linear_mono.py`
  in the mononet Python repo.
-/
theorem Mononet.constrainedLinear_mono : ...
```

### Python → Lean

A new docs page `docs/concepts/proofs.md` lists each paper claim with two columns:

```
| Paper claim     | Lean theorem                            | Empirical test                    |
| Lemma 1         | Mononet.constrainedLinear_mono          | tests/properties/test_lemma1.py   |
| Lemma 2 (mono)  | Mononet.combined_mono                   | tests/properties/test_lemma2.py   |
| ...
```

This is the page paper readers land on when they want to see "how do you know your library actually does what the paper says".

The doc-gen4 HTML is served as a sibling path on GitHub Pages, e.g. `davorrunje.github.io/mononet/proofs/`. The Sphinx docs `proofs.md` page contains absolute links to that subtree.

## 8. Acceptance criteria

This sub-project is "done" when:

- `lake build` in `proofs/` succeeds on CI with zero `sorry` declarations *in our code*. The only `sorry`-free residual hypothesis is the Daniels & Velikova 2010 axiom (Path 3), which is documented.
- The doc-gen4 HTML is generated, deployed, and linked from the Sphinx docs.
- `docs/concepts/proofs.md` is a complete cross-reference table.
- Each formalized theorem carries the doc-comment with paper citation and counterpart test.
- The Lean CI workflow runs in under 10 minutes for incremental builds.

## 9. Open items

- **Lean version selection** — fixed via `lean-toolchain` file. We pick the version that has the freshest mathlib4 build cache available at the time of starting Sub-project E.
- **mathlib4 revision pin** — bumped via a dependabot-equivalent (a manual cadence at first; mathlib4 moves quickly enough that quarterly bumps are sensible).
- **`doc-gen4` deployment** — Sphinx docs use sphinx-polyversion via GitHub Pages already. Add a separate gh-pages prefix (`/proofs/`) populated by a second deploy step in the Lean workflow. Confirm gh-pages can serve from multiple workflows without races.
- **Daniels & Velikova 2010 axiomatization** — open the discussion above and pick a path. Default: Path 3 first, Path 1 attempted opportunistically.
- **Patent interaction** — formalizing the construction of the patented method in a public Lean proof has no patent-rights interaction (formalization is descriptive, not infringing). Confirm with counsel; document the absence-of-issue in `proofs/README.md` if applicable.
- **Citability** — once the proofs land, generate a Zenodo DOI for the proofs subtree separately from the Python library's DOI. Two artifacts, two DOIs, both citable.

## 10. What is intentionally NOT in this sub-project

- **Formal verification of the Python implementation** (i.e. proving `mononet.torch.MonoLinear` correctly implements its mathematical specification). That would require a Lean ↔ Python correspondence proof, which is a research project of its own.
- **Formalization of the patent's claims.** The patent is a legal artifact; the math is what we formalize.
- **Formalization of flow / injectivity from Sub-project D.** Could be a follow-up sub-project if Sub-project D ships and there is demand.
- **Theorem-prover-driven implementation generation** (e.g. extracting verified code from Lean to Python or to a CompCert-style target). Out of scope.
- **A second proof assistant.** No Coq / Isabelle parallels; Lean 4 only.
