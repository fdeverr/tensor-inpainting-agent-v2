# t-SVD / Low-Tubal-Rank Decomposition

## Mathematical form

The low-tubal-rank model applies an FFT along the third tensor mode and learns
`X_f[:,:,q] ≈ U_f[:,:,q] diag(S_f[:,q]) V_f[:,:,q]^H` at each frequency.
An inverse real FFT returns the RGB image. The tunable `rank` is the tubal rank.

## Strengths

- The t-product couples spatial matrices through channel-mode frequency tubes.
- A single tubal-rank parameter gives a compact, interpretable capacity control.
- It is useful when repeated spatial structure is correlated across channels.

## Rank guidance

Use ranks from `{2, 4, 8, 16}`, capped by `min(H, W)`. Select rank only with
held-out observed pixels. Larger rank increases each Fourier-slice matrix rank.

## Limitations and failure modes

RGB has only three entries in the third mode and therefore only two real-FFT
frequency bins. This makes the t-SVD prior much weaker than it is for videos or
hyperspectral tensors. It has no semantic knowledge and may blur or stripe a
large contiguous hole despite a good observed-pixel validation score.
