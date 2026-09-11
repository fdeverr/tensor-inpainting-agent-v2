# Tensor Ring Decomposition

Tensor Ring (TR) closes the Tensor Train boundary ranks into a cycle:

`X[i,j,k] = trace(G_h[:,i,:] G_w[:,j,:] G_c[:,k,:])`.

The cyclic contraction treats the mode boundaries more symmetrically than a
TT chain and uses one bounded ring rank in this baseline.

## When it can help

TR can provide more interaction capacity than CP or a small TT at a similar
parameter count. It is worth considering for repeated texture, strong
cross-mode interaction, or many dispersed missing components where cyclic
factor coupling may be useful.

## Rank guidance

Use small ring ranks such as 2, 4, 6, or 8. Computational cost grows cubically
with ring rank during contraction, so the built-in model caps it at 16 and the
bounded tuner keeps the candidate grid small.

## Limitations

Tensor Ring optimization is non-convex and has substantial gauge freedom. Its
extra coupling can be harder to optimize than CP or TT, and a larger rank does
not guarantee better generalization into an unobserved block.

## Failure modes

An oversized ring rank can overfit observed pixels and consume much more
compute. An undersized rank can produce repetitive textures or color coupling
artifacts. Large semantic holes remain difficult without a spatial or learned
image prior.
