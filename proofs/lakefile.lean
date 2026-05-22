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

lean_lib «Mononet» where
  -- library containing all Mononet namespace modules

@[default_target]
lean_lib «Proofs» where
  -- entry-point library that imports the rest as modules land
