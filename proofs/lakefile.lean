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

meta if get_config? env = some "dev" then
  require «doc-gen4» from git
    "https://github.com/leanprover/doc-gen4" @ "main"

lean_lib «Mononet» where
  -- library containing all Mononet namespace modules

@[default_target]
lean_lib «Proofs» where
  -- entry-point library that imports the rest as modules land
