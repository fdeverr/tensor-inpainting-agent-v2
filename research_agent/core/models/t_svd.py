"""Low-tubal-rank t-SVD parameterization for third-order image tensors."""

from __future__ import annotations

import math
from typing import Any, Dict, Sequence, Tuple

import torch
from torch import nn

from .base import BaseTensorInpaintingModel


class TSVDDecomposition(BaseTensorInpaintingModel):
    """Synthesize an image with a rank-bounded t-product in the Fourier domain.

    The real-valued parameters correspond to ``U``, diagonal singular tubes,
    and ``V``. The contraction uses ``U_f S_f V_f^H`` independently at every
    third-mode Fourier frequency, followed by an inverse real FFT.
    """

    def __init__(
        self,
        image_shape: Tuple[int, ...],
        initial_channel_mean: Sequence[float],
        rank: int = 8,
        init_scale: float = 0.1,
    ) -> None:
        super().__init__(image_shape, initial_channel_mean)
        height, width = self.image_shape[:2]
        features = self.feature_count
        max_rank = min(height, width)
        if not 1 <= rank <= max_rank:
            raise ValueError("t-SVD tubal rank must be in [1, %d]" % max_rank)
        if init_scale <= 0.0:
            raise ValueError("init_scale must be positive")

        self.rank = int(rank)
        factor_scale = init_scale / math.sqrt(self.rank)
        self.left_factor = nn.Parameter(
            torch.empty(height, self.rank, features)
        )
        self.singular_tubes = nn.Parameter(torch.empty(self.rank, features))
        self.right_factor = nn.Parameter(
            torch.empty(width, self.rank, features)
        )
        nn.init.normal_(self.left_factor, mean=0.0, std=factor_scale)
        nn.init.normal_(self.singular_tubes, mean=0.0, std=init_scale)
        nn.init.normal_(self.right_factor, mean=0.0, std=factor_scale)

    def forward(self) -> torch.Tensor:
        features = self.feature_count
        left_frequency = torch.fft.rfft(self.left_factor, dim=-1)
        singular_frequency = torch.fft.rfft(self.singular_tubes, dim=-1)
        right_frequency = torch.fft.rfft(self.right_factor, dim=-1)
        reconstruction_frequency = torch.einsum(
            "hrf,rf,wrf->hwf",
            left_frequency,
            singular_frequency,
            right_frequency.conj(),
        )
        decomposition = torch.fft.irfft(
            reconstruction_frequency,
            n=features,
            dim=-1,
        )
        return self._restore_shape(decomposition + self.channel_bias)

    @classmethod
    def search_space(cls, image_shape: Tuple[int, ...]) -> Dict[str, Any]:
        height, width = image_shape[:2]
        max_rank = min(height, width)
        ranks = sorted({max(1, min(max_rank, value)) for value in (2, 4, 8, 16)})
        return {"rank": ranks, "init_scale": [0.05, 0.1, 0.2]}
