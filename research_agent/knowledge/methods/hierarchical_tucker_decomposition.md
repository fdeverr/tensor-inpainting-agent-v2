# Hierarchical Tucker Decomposition

## Mathematical form

The RGB implementation uses a binary dimension tree: height and width leaf
factors first combine into a spatial transfer rank, then a root transfer couples
that spatial node to the channel factor. This factorizes Tucker's dense core.

## Strengths

- The dimension tree exposes a separate `rank_spatial` bottleneck for height-width
  interactions.
- It can be more parameter-efficient than a dense Tucker core.
- Leaf ranks still allow different capacity along height, width, and channels.

## Rank guidance

Use spatial leaf ranks in `{4, 8, 16}` and normally keep the RGB leaf rank at
three. `rank_spatial` must not exceed either `rank_h * rank_w` or `rank_c`; for
RGB images it is therefore usually 1--3.

## Limitations and failure modes

With only three tensor modes, the hierarchy is shallow and its advantage over
ordinary Tucker can be modest. A small root bottleneck can oversmooth channel or
spatial interactions. The chosen dimension tree imposes an inductive bias and
still does not recover semantic objects inside a large unobserved region.
