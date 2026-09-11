"""Hierarchical Tucker baseline with a spatial binary dimension tree."""

from __future__ import annotations

import math
from typing import Any, Dict, Sequence, Tuple

import torch
from torch import nn

from .base import BaseTensorInpaintingModel


class HierarchicalTuckerDecomposition(BaseTensorInpaintingModel):
    """Factorize the Tucker core through a ``(height, width) -> channel`` tree."""

    def __init__(
        self,
        image_shape: Tuple[int, ...],
        initial_channel_mean: Sequence[float],
        rank_h: int = 8,
        rank_w: int = 8,
        rank_c: int = 3,
        rank_spatial: int = 2,
        init_scale: float = 0.1,
    ) -> None:
        super().__init__(image_shape, initial_channel_mean)
        height, width = self.image_shape[:2]
        features = self.feature_count
        ranks = (int(rank_h), int(rank_w), int(rank_c))
        limits = (height, width, features)
        if any(rank < 1 or rank > limit for rank, limit in zip(ranks, limits)):
            raise ValueError(
                "Hierarchical Tucker leaf ranks must be positive and not exceed image dimensions"
            )
        max_spatial_rank = min(ranks[0] * ranks[1], ranks[2])
        if not 1 <= rank_spatial <= max_spatial_rank:
            raise ValueError(
                "Hierarchical Tucker rank_spatial must be in [1, %d]"
                % max_spatial_rank
            )
        if init_scale <= 0.0:
            raise ValueError("init_scale must be positive")

        self.ranks = ranks
        self.rank_spatial = int(rank_spatial)
        factor_scale = init_scale / math.sqrt(max((*ranks, self.rank_spatial)))
        self.height_factor = nn.Parameter(torch.empty(height, ranks[0]))
        self.width_factor = nn.Parameter(torch.empty(width, ranks[1]))
        self.channel_factor = nn.Parameter(torch.empty(features, ranks[2]))
        self.spatial_transfer = nn.Parameter(
            torch.empty(ranks[0], ranks[1], self.rank_spatial)
        )
        self.root_transfer = nn.Parameter(
            torch.empty(self.rank_spatial, ranks[2])
        )
        for parameter in (
            self.height_factor,
            self.width_factor,
            self.channel_factor,
            self.spatial_transfer,
            self.root_transfer,
        ):
            nn.init.normal_(parameter, mean=0.0, std=factor_scale)

    def forward(self) -> torch.Tensor:
        decomposition = torch.einsum(
            "abq,qc,ia,jb,kc->ijk",
            self.spatial_transfer,
            self.root_transfer,
            self.height_factor,
            self.width_factor,
            self.channel_factor,
        )
        return self._restore_shape(decomposition + self.channel_bias)

    @classmethod
    def search_space(cls, image_shape: Tuple[int, ...]) -> Dict[str, Any]:
        height, width = image_shape[:2]
        features = math.prod(image_shape[2:])
        rank_h = sorted({max(1, min(height, value)) for value in (4, 8, 16)})
        rank_w = sorted({max(1, min(width, value)) for value in (4, 8, 16)})
        # Keeping the feature leaf at full rank makes every independently
        # sampled rank_spatial value valid in the generic search grid.
        rank_c = [features]
        max_spatial = min(height * width, features)
        rank_spatial = sorted({max(1, min(max_spatial, value)) for value in (1, 2, 4, 8)})
        return {
            "rank_h": rank_h,
            "rank_w": rank_w,
            "rank_c": rank_c,
            "rank_spatial": rank_spatial,
            "init_scale": [0.05, 0.1, 0.15],
        }
