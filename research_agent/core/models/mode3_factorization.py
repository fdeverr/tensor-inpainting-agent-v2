"""Joint-feature-mode tensor factorization ``X = A x_3 E``."""

from __future__ import annotations

import math
from typing import Any, Dict, Sequence, Tuple

import torch
from torch import nn

from .base import BaseTensorInpaintingModel


class Mode3Factorization(BaseTensorInpaintingModel):
    r"""Factorize the joint trailing-feature mode of an ``H x W x F`` view.

    ``coefficient_tensor`` is :math:`A \in R^{H x W x R}` and
    ``channel_factor`` is :math:`E \in R^{F x R}`. Their mode-3 product is
    ``X[i, j, f] = sum_r A[i, j, r] * E[f, r]``. For video, ``F=T*C``
    and the result is reshaped back to ``[H,W,T,C]``.
    """

    def __init__(
        self,
        image_shape: Tuple[int, ...],
        initial_channel_mean: Sequence[float],
        rank: int = 2,
        init_scale: float = 0.1,
    ) -> None:
        super().__init__(image_shape, initial_channel_mean)
        height, width = self.image_shape[:2]
        features = self.feature_count
        max_rank = min(height * width, features)
        if not 1 <= rank <= max_rank:
            raise ValueError("Mode-3 rank must be in [1, %d]" % max_rank)
        if init_scale <= 0.0:
            raise ValueError("init_scale must be positive")

        self.rank = int(rank)
        factor_scale = init_scale / math.sqrt(self.rank)
        self.coefficient_tensor = nn.Parameter(
            torch.empty(height, width, self.rank)
        )
        self.channel_factor = nn.Parameter(torch.empty(features, self.rank))
        nn.init.normal_(self.coefficient_tensor, mean=0.0, std=factor_scale)
        nn.init.normal_(self.channel_factor, mean=0.0, std=init_scale)

    def forward(self) -> torch.Tensor:
        decomposition = torch.einsum(
            "ijr,cr->ijc",
            self.coefficient_tensor,
            self.channel_factor,
        )
        return self._restore_shape(decomposition + self.channel_bias)

    @classmethod
    def search_space(cls, image_shape: Tuple[int, ...]) -> Dict[str, Any]:
        height, width = image_shape[:2]
        features = math.prod(image_shape[2:])
        max_rank = min(height * width, features)
        ranks = sorted({max(1, min(max_rank, value)) for value in (1, 2, 4, 8, 16, 32)})
        return {"rank": ranks, "init_scale": [0.05, 0.1, 0.2]}
