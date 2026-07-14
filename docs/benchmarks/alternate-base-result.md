# Alternate base result (tuned, ≤4 layers)

This page reports the tuned shallow (`depth ∈ [1,3]`, i.e. ≤4 effective layers
counting the linear read-out head) `alternate` flavor against the best of the
existing `split`/`mixed` flavors, on the five paper datasets. Each flavor is
tuned per-dataset with its own Optuna HP search (including the activation),
`plain` only (no residual arm), following the
[standard benchmark protocol](protocol.md). The verdict per dataset is a
bootstrap CI on `alternate − best-of-{split, mixed}`: "alternate helps" if the
CI lies strictly on the better side of zero, "matches" if it straddles zero,
"loses" otherwise.

<!-- TABLE -->
