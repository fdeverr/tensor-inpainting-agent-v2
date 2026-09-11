# Nonnegative CP Decomposition

## Mathematical form

Nonnegative CP represents the image as
`X[i,j,k] ≈ Σ_r A[i,r] B[j,r] C[k,r]`, with every effective factor constrained
to be nonnegative through a softplus parameterization. A positive learned
channel scale initializes the reconstruction near the visible RGB means.

## Strengths

- The reconstruction is nonnegative before clipping, matching image intensity
  support.
- Rank-one components are additive and can form interpretable parts.
- It has fewer parameters and a smaller tuning space than Nonnegative Tucker.

## Rank guidance

Start with rank 8 or 12 and compare ranks from `{4, 8, 12, 16, 24}` using only
held-out observed pixels. Larger rank adds more nonnegative components but also
increases redundancy and overfitting risk.

## Limitations and failure modes

Nonnegative factors cannot cancel one another, so this model is less flexible
than ordinary CP for signed latent interactions. Softplus saturation can slow
optimization. The global rank-one components contain no explicit locality,
edge, or semantic prior and may produce smooth bands inside a large block hole.
