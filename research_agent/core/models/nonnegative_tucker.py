"""Nonnegative Tucker decomposition baseline for spatial tensor data."""

from __future__ import annotations

import math
from typing import Any, Dict, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .base import BaseTensorInpaintingModel


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(value.clamp_min(1e-6)))


class NonnegativeTuckerDecomposition(BaseTensorInpaintingModel):
    """Apply a softplus parameterization to every effective Tucker component."""

    def __init__(
        self,
        image_shape: Tuple[int, ...],
        initial_channel_mean: Sequence[float],
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
        if any(rank < 1 or rank > limit for rank, limit in zip(ranks, limits)):
            raise ValueError(
                "Nonnegative Tucker ranks must be positive and not exceed image dimensions"
            )
        if init_scale <= 0.0:
            raise ValueError("init_scale must be positive")

        self.ranks = ranks
        # Initialize the positive factors so their full contraction is near one;
        # channel_scale then places the reconstruction around the observed mean.
        component_scale = (1.0 / math.prod(ranks)) ** 0.25
        raw_mean = math.log(math.expm1(component_scale))
        self.raw_core = nn.Parameter(torch.empty(*ranks))
        self.raw_height_factor = nn.Parameter(torch.empty(height, ranks[0]))
        self.raw_width_factor = nn.Parameter(torch.empty(width, ranks[1]))
        self.raw_channel_factor = nn.Parameter(torch.empty(features, ranks[2]))
        for parameter in (
            self.raw_core,
            self.raw_height_factor,
            self.raw_width_factor,
            self.raw_channel_factor,
        ):
            nn.init.normal_(parameter, mean=raw_mean, std=float(init_scale))

        channel_mean = torch.as_tensor(initial_channel_mean, dtype=torch.float32).reshape(-1)
        with torch.no_grad():
            self.channel_bias.copy_(_inverse_softplus(channel_mean))

    def forward(self) -> torch.Tensor:
        channel_scale = F.softplus(self.channel_bias).reshape(-1, 1)
        channel_factor = F.softplus(self.raw_channel_factor) * channel_scale
        flattened = torch.einsum(
            "abc,ia,jb,kc->ijk",
            F.softplus(self.raw_core),
            F.softplus(self.raw_height_factor),
            F.softplus(self.raw_width_factor),
            channel_factor,
        )
        return self._restore_shape(flattened)

    @classmethod
    def search_space(cls, image_shape: Tuple[int, ...]) -> Dict[str, Any]:
        height, width = image_shape[:2]
        features = math.prod(image_shape[2:])
        rank_h = sorted({max(1, min(height, value)) for value in (4, 8, 16)})
        rank_w = sorted({max(1, min(width, value)) for value in (4, 8, 16)})
        rank_c = sorted({max(1, min(features, value)) for value in (1, 2, 3, 4, 8, 16)})
        return {
            "rank_h": rank_h,
            "rank_w": rank_w,
            "rank_c": rank_c,
            "init_scale": [0.05, 0.1, 0.15],
        }
