# Block-Term Decomposition (BTD)

## Mathematical form

BTD represents the image as a sum of Tucker blocks,
`X ≈ Σ_n G_n ×₁ A_n ×₂ B_n ×₃ C_n`. Each block has its own core and
height, width, and channel factors. The implementation bounds the number of
blocks to four so automatic tuning remains practical.

## Strengths

- Multiple blocks can model several spatial or texture components instead of
  forcing every interaction into one Tucker core.
- Separate mode ranks retain anisotropic height, width, and RGB capacity.
- It is more expressive than a single Tucker block at the same per-block rank.

## Rank guidance

Begin with one to three blocks, spatial ranks in `{4, 8, 16}`, and channel rank
in `{1, 2, 3}`. Increase the number of blocks only when validation improves;
blocks and core ranks multiply the parameter count together.

## Limitations and failure modes

BTD is not a local patch model. It can still produce global bands in a large
hole. Redundant blocks may converge to similar components, destabilize fitting,
or overfit observed pixels. High ranks across many blocks grow memory quickly.
