# A Mode-3 E Factorization

## Mathematical form

For an RGB image tensor `X ∈ R^(H×W×C)`, learn a spatial coefficient tensor
`A ∈ R^(H×W×R)` and a channel factor matrix `E ∈ R^(C×R)`:

`X ≈ A ×₃ E`, or equivalently
`X[i,j,c] = sum_r A[i,j,r] E[c,r]`.

This is the rank-`R` matrix factorization of the mode-3 unfolding
`X_(3) ∈ R^(C×HW)`. Unlike the project's `matrix` baseline, it does not
flatten width and channels together.

## Parameter characteristics

The parameter count is `HWR + CR`, plus the RGB bias. For an RGB image the
rank is at most three, so the factorization compresses channel correlation but
retains an independent latent vector at every spatial location.

## Strengths

- Exactly matches the interpretable `A ×₃ E` channel-mode formulation.
- Models a low-dimensional colour subspace explicitly.
- Useful as a diagnostic baseline when RGB channels are strongly correlated.
- Supports rank 1, 2, or 3 on RGB inputs.

## Limitations

- The coefficient tensor `A` has no built-in coupling between neighbouring pixels.
- When all RGB channels at a pixel are missing, its coefficient vector receives no
  data-loss gradient; the plain baseline therefore cannot spatially extrapolate it.
- Rank 3 has almost the same spatial degrees of freedom as the original image.
- Large block holes generally require a spatial prior, structured parameterization,
  or a hybrid candidate on top of this baseline.

## Selection guidance

Treat this primarily as a channel-subspace ablation. High visible RGB correlation
supports trying rank 1 or 2, but it should not outrank spatially structured CP,
Tucker, TT, or Tensor Ring solely because channel correlation is high. Select the
rank using held-out observed pixels and interpret hidden-region quality separately.

## Common failure modes

- Untrained or noisy coefficients inside a fully missing spatial region.
- Good observed-pixel validation with poor block-hole reconstruction.
- Rank 1 causes colour shifts when the image needs more than one colour direction.
- Rank 3 over-parameterizes the channel mode without adding spatial regularity.
