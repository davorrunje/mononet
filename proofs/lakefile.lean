import Lake
open Lake DSL

package «mononet-proofs» where
  -- Build settings
  leanOptions := #[
    ⟨`pp.unicode.fun, true⟩,
    ⟨`autoImplicit, false⟩
  ]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.15.0"

@[default_target]
lean_lib «Mononet» where
  -- root namespace exported by the project

lean_lib «Proofs» where
  -- entry-point library that imports the rest
