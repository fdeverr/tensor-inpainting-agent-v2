"""Tensor Train decomposition baseline for spatial tensor data."""

from __future__ import annotations

import math
from typing import Any, Dict, Sequence, Tuple

import torch
from torch import nn

from .base import BaseTensorInpaintingModel


class TensorTrainDecomposition(BaseTensorInpaintingModel):
    """Represent an ``H x W x F`` tensor view with three Tensor Train cores.

    The boundary TT ranks are one and the two internal ranks are independently
    tunable. The joint trailing-feature mode bounds ``rank_2`` while
    ``rank_1`` controls most of the spatial capacity.
    """

    def __init__(
        self,
        image_shape: Tuple[int, ...],
        initial_channel_mean: Sequence[float],
        rank_1: int = 8,
        rank_2: int = 3,
        init_scale: float = 0.1,
    ) -> None:
        super().__init__(image_shape, initial_channel_mean)
        height, width = self.image_shape[:2]
        features = self.feature_count
        max_rank_1 = min(height, width * features)
        max_rank_2 = min(height * width, features)
        if not 1 <= rank_1 <= max_rank_1:
            raise ValueError("TT rank_1 must be in [1, %d]" % max_rank_1)
        if not 1 <= rank_2 <= max_rank_2:
            raise ValueError("TT rank_2 must be in [1, %d]" % max_rank_2)
        if init_scale <= 0.0:
            raise ValueError("init_scale must be positive")

        self.ranks = (int(rank_1), int(rank_2))
        factor_scale = init_scale / math.sqrt(max(self.ranks))
        self.height_core = nn.Parameter(torch.empty(height, self.ranks[0]))
        self.width_core = nn.Parameter(
            torch.empty(self.ranks[0], width, self.ranks[1])
        )
        self.channel_core = nn.Parameter(torch.empty(self.ranks[1], features))
        nn.init.normal_(self.height_core, mean=0.0, std=factor_scale)
        nn.init.normal_(self.width_core, mean=0.0, std=factor_scale)
        nn.init.normal_(self.channel_core, mean=0.0, std=init_scale)

    def forward(self) -> torch.Tensor:
        decomposition = torch.einsum(
            "ia,ajb,bk->ijk",
            self.height_core,
            self.width_core,
            self.channel_core,
        )
        return self._restore_shape(decomposition + self.channel_bias)

    @classmethod
    def search_space(cls, image_shape: Tuple[int, ...]) -> Dict[str, Any]:
        height, width = image_shape[:2]
        features = math.prod(image_shape[2:])
        max_rank_1 = min(height, width * features)
        max_rank_2 = min(height * width, features)
        rank_1 = sorted(
            {max(1, min(max_rank_1, value)) for value in (4, 8, 16, 32)}
        )
        rank_2 = sorted({max(1, min(max_rank_2, value)) for value in (1, 2, 3, 4, 8, 16)})
        return {
            "rank_1": rank_1,
            "rank_2": rank_2,
            "init_scale": [0.05, 0.1, 0.15],
        }
