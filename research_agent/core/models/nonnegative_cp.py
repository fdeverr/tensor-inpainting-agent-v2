"""Nonnegative canonical polyadic decomposition for spatial tensor data."""

from __future__ import annotations

import math
from typing import Any, Dict, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .base import BaseTensorInpaintingModel


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(value.clamp_min(1e-6)))


class NonnegativeCPDecomposition(BaseTensorInpaintingModel):
    """Represent an image as a sum of nonnegative rank-one components."""

    def __init__(
        self,
        image_shape: Tuple[int, ...],
        initial_channel_mean: Sequence[float],
        rank: int = 12,
        init_scale: float = 0.1,
    ) -> None:
        super().__init__(image_shape, initial_channel_mean)
        height, width = self.image_shape[:2]
        features = self.feature_count
        if rank < 1:
            raise ValueError("Nonnegative CP rank must be positive")
        if init_scale <= 0.0:
            raise ValueError("init_scale must be positive")

        self.rank = int(rank)
        # With three effective factors, rank * component_scale**3 is one.
        # The positive channel scale then initializes each output channel near
        # its observed mean while retaining useful gradients.
        component_scale = self.rank ** (-1.0 / 3.0)
        raw_mean = math.log(math.expm1(component_scale))
        self.raw_height_factor = nn.Parameter(torch.empty(height, self.rank))
        self.raw_width_factor = nn.Parameter(torch.empty(width, self.rank))
        self.raw_channel_factor = nn.Parameter(torch.empty(features, self.rank))
        for parameter in (
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
            "ir,jr,kr->ijk",
            F.softplus(self.raw_height_factor),
            F.softplus(self.raw_width_factor),
            channel_factor,
        )
        return self._restore_shape(flattened)

    @classmethod
    def search_space(cls, image_shape: Tuple[int, ...]) -> Dict[str, Any]:
        del image_shape
        return {"rank": [4, 8, 12, 16, 24], "init_scale": [0.05, 0.1, 0.2]}
