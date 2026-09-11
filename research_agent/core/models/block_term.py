"""Block-Term Decomposition baseline for spatial tensor data."""

from __future__ import annotations

import math
from typing import Any, Dict, Sequence, Tuple

import torch
from torch import nn

from .base import BaseTensorInpaintingModel


class BlockTermDecomposition(BaseTensorInpaintingModel):
    """Represent an image as a sum of compact Tucker blocks."""

    def __init__(
        self,
        image_shape: Tuple[int, ...],
        initial_channel_mean: Sequence[float],
        num_blocks: int = 2,
        rank_h: int = 8,
        rank_w: int = 8,
        rank_c: int = 2,
        init_scale: float = 0.1,
    ) -> None:
        super().__init__(image_shape, initial_channel_mean)
        height, width = self.image_shape[:2]
        features = self.feature_count
        ranks = (int(rank_h), int(rank_w), int(rank_c))
        limits = (height, width, features)
        if not 1 <= num_blocks <= 4:
            raise ValueError("BTD num_blocks must be in [1, 4]")
        if any(rank < 1 or rank > limit for rank, limit in zip(ranks, limits)):
            raise ValueError("BTD ranks must be positive and not exceed image dimensions")
        if init_scale <= 0.0:
            raise ValueError("init_scale must be positive")

        self.num_blocks = int(num_blocks)
        self.ranks = ranks
        factor_scale = init_scale / math.sqrt(self.num_blocks * max(ranks))
        self.cores = nn.Parameter(torch.empty(self.num_blocks, *ranks))
        self.height_factors = nn.Parameter(
            torch.empty(self.num_blocks, height, ranks[0])
        )
        self.width_factors = nn.Parameter(
            torch.empty(self.num_blocks, width, ranks[1])
        )
        self.channel_factors = nn.Parameter(
            torch.empty(self.num_blocks, features, ranks[2])
        )
        nn.init.normal_(self.cores, mean=0.0, std=factor_scale)
        nn.init.normal_(self.height_factors, mean=0.0, std=factor_scale)
        nn.init.normal_(self.width_factors, mean=0.0, std=factor_scale)
        nn.init.normal_(self.channel_factors, mean=0.0, std=factor_scale)

    def forward(self) -> torch.Tensor:
        decomposition = torch.einsum(
            "nabc,nia,njb,nkc->ijk",
            self.cores,
            self.height_factors,
            self.width_factors,
            self.channel_factors,
        )
        return self._restore_shape(decomposition + self.channel_bias)

    @classmethod
    def search_space(cls, image_shape: Tuple[int, ...]) -> Dict[str, Any]:
        height, width = image_shape[:2]
        features = math.prod(image_shape[2:])
        rank_h = sorted({max(1, min(height, value)) for value in (4, 8, 16)})
        rank_w = sorted({max(1, min(width, value)) for value in (4, 8, 16)})
        rank_c = sorted({max(1, min(features, value)) for value in (1, 2, 3, 4, 8, 16)})
        return {
            "num_blocks": [1, 2, 3],
            "rank_h": rank_h,
            "rank_w": rank_w,
            "rank_c": rank_c,
            "init_scale": [0.05, 0.1, 0.15],
        }
