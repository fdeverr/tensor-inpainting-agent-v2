# Nonnegative Tucker Decomposition

## Mathematical form

Nonnegative Tucker uses `X ≈ G ×₁ A ×₂ B ×₃ C` while constraining the
effective core and factors to be nonnegative through softplus parameters. A
positive learned channel scale initializes the output near visible RGB means.

## Strengths

- The reconstruction is nonnegative by construction, matching image intensity
  support before final clipping.
- Positive factors encourage additive, parts-based components.
- It retains Tucker's separate ranks for height, width, and channels.

## Rank guidance

Start with spatial ranks in `{4, 8, 16}` and channel rank in `{1, 2, 3}`.
Compare directly with ordinary Tucker: positivity is useful only when the held-out
observed error improves without excessive loss of detail.

## Limitations and failure modes

The constraint cannot express cancellation between components and may reduce
capacity or slow optimization. Softplus saturation can weaken gradients.
Nonnegativity alone does not provide edge, locality, or semantic priors, so block
interiors can remain smooth or structurally incorrect.
