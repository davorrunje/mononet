# Non-monotone feature embedding

## The problem

A `MonoLinear` layer constrains its weights to `|W|` (non-negative effective
weights), so the network is monotone non-decreasing in *every* input it sees.
With a convex activation (e.g. ReLU) and non-negative weights it can represent
only **convex** monotone functions (Sartor 2025 Prop 3.2). But real tabular
datasets mix **monotone-constrained** features (where domain knowledge fixes a
direction) with **non-monotone** ("free") features that must be allowed an
arbitrary, unconstrained effect on the output. Feeding a free feature straight
into the monotone stack would wrongly force the output to be monotone in it.

## The construction

Free features are routed through an **unconstrained `Dense` embedding**; its
output is concatenated with the monotone-feature channels and the concatenation
is fed to the monotone stack. The network stays monotone in the *declared*
monotone inputs, while the free features reach the output only through the
unconstrained embedding — so their net effect can be arbitrary. This is the
standard embedding-composition trick the builder already implements.

## Why the embedding needs two `Dense` layers

For the free branch to represent an *arbitrary* function of the free features it
must be a universal approximator, which needs a hidden layer **plus** an output
projection — two weight layers (`Linear → act → Linear`). It is tempting to drop
the embedding's output layer and let the monotone stack's first `MonoLinear`
absorb it (two adjacent linear maps merge). That merge is invalid here: the
first `MonoLinear` is `|W|`-constrained, so it can only form **non-negative**
combinations of the embedding's hidden units, and by Prop 3.2 that is restricted
to *convex* functions of them — not a free output projection. A single `Dense`
layer therefore collapses the free branch's expressivity to convex, which is not
universal.

Giving the embedding its own unconstrained output layer (the second `Dense`)
computes the arbitrary free-feature function *before* the constrained stack,
which then reads it with a positive passthrough weight. Hence **two `Dense`
layers** (at the model width) for the non-monotone branch. (Fully-monotone
datasets have no free features, so the branch is empty and this is a no-op.)
