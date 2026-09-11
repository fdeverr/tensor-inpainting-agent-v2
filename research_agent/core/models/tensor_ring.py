"""Tensor Ring decomposition baseline for spatial tensor data."""

from __future__ import annotations

import math
from typing import Any, Dict, Sequence, Tuple

import torch
from torch import nn

from .base import BaseTensorInpaintingModel


class TensorRingDecomposition(BaseTensorInpaintingModel):
    """Represent an image by cyclically contracting three third-order cores."""

    def __init__(
        self,
        image_shape: Tuple[int, ...],
        initial_channel_mean: Sequence[float],
        rank: int = 4,
        init_scale: float = 0.1,
    ) -> None:
        super().__init__(image_shape, initial_channel_mean)
        height, width = self.image_shape[:2]
        features = self.feature_count
        max_rank = min(16, height, width)
        if not 1 <= rank <= max_rank:
            raise ValueError("Tensor Ring rank must be in [1, %d]" % max_rank)
        if init_scale <= 0.0:
            raise ValueError("init_scale must be positive")

        self.rank = int(rank)
        factor_scale = init_scale / math.sqrt(self.rank)
        self.height_core = nn.Parameter(torch.empty(self.rank, height, self.rank))
        self.width_core = nn.Parameter(torch.empty(self.rank, width, self.rank))
        self.channel_core = nn.Parameter(
            torch.empty(self.rank, features, self.rank)
        )
        nn.init.normal_(self.height_core, mean=0.0, std=factor_scale)
        nn.init.normal_(self.width_core, mean=0.0, std=factor_scale)
        nn.init.normal_(self.channel_core, mean=0.0, std=factor_scale)

    def forward(self) -> torch.Tensor:
        decomposition = torch.einsum(
            "aib,bjc,cka->ijk",
            self.height_core,
            self.width_core,
            self.channel_core,
        )
        return self._restore_shape(decomposition + self.channel_bias)

    @classmethod
    def search_space(cls, image_shape: Tuple[int, ...]) -> Dict[str, Any]:
        height, width = image_shape[:2]
        max_rank = min(16, height, width)
        ranks = sorted({max(1, min(max_rank, value)) for value in (2, 4, 6, 8)})
        return {"rank": ranks, "init_scale": [0.05, 0.1, 0.15]}
