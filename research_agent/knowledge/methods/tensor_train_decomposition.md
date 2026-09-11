# Tensor Train Decomposition

Tensor Train (TT) represents an RGB tensor with a chain of three cores:

`X[i,j,k] = sum_{a,b} G_h[i,a] G_w[a,j,b] G_c[b,k]`.

The two internal ranks control different unfoldings. `rank_1` mainly controls
height-versus-rest capacity, while `rank_2` couples the spatial modes to RGB.
For an RGB image, `rank_2` cannot exceed three in an exact third-order TT.

## When it can help

TT is parameter-efficient for anisotropic images because it keeps height,
width, and channel modes explicit without learning Tucker's dense core. It is
a useful alternative when a matrix unfolding is too restrictive but Tucker's
independent spatial core is unnecessarily expensive.

## Rank guidance

Start with `rank_1` in 4--16 and `rank_2` in 1--3. A larger first rank increases
spatial capacity; a larger second rank preserves more channel interaction.
Both ranks must be selected only from held-out observed pixels.

## Limitations

TT depends on mode ordering: height-width-channel and width-height-channel are
not equivalent at a fixed rank. A third-order RGB tensor also gives the second
rank little room to vary. Like the other global factorizations, TT has no
semantic knowledge and can extrapolate poorly across a large contiguous hole.

## Failure modes

Ranks that are too small cause directional smoothing or banding. Large ranks
can overfit scattered observations without improving the hidden region. The
chain structure can favor one spatial direction, especially on highly
anisotropic images.
