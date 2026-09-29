"""Coordinate-based SIREN baseline for image and tensor inpainting."""

from __future__ import annotations

import math
from typing import Any, Dict, Sequence, Tuple

import torch
from torch import nn

from .base import BaseTensorInpaintingModel


class _SineLayer(nn.Module):
    def __init__(
        self,
        input_features: int,
        output_features: int,
        omega_0: float,
        first: bool = False,
    ) -> None:
        super().__init__()
        self.omega_0 = float(omega_0)
        self.linear = nn.Linear(input_features, output_features)
        with torch.no_grad():
            if first:
                bound = 1.0 / input_features
            else:
                bound = math.sqrt(6.0 / input_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(coordinates))


class SirenImplicitNetwork(BaseTensorInpaintingModel):
    """Map normalized 2-D coordinates to all trailing tensor features.

    For RGB/MSI this is the standard coordinate-image formulation. For H-W-T-C
    tensors the temporal/channel axes are flattened into output features so the
    baseline remains compatible with the workflow's shared spatial mask.
    """

    def __init__(
        self,
        image_shape: Tuple[int, ...],
        initial_channel_mean: Sequence[float],
        hidden_features: int = 128,
        hidden_layers: int = 3,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
    ) -> None:
        super().__init__(image_shape, initial_channel_mean)
        if hidden_features < 8:
            raise ValueError("hidden_features must be at least 8")
        if not 1 <= hidden_layers <= 8:
            raise ValueError("hidden_layers must be in [1, 8]")
        if first_omega_0 <= 0.0 or hidden_omega_0 <= 0.0:
            raise ValueError("SIREN omega values must be positive")

        height, width = self.image_shape[:2]
        y = torch.linspace(-1.0, 1.0, height)
        x = torch.linspace(-1.0, 1.0, width)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        self.register_buffer(
            "coordinates",
            torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2),
            persistent=False,
        )

        layers = [
            _SineLayer(2, hidden_features, first_omega_0, first=True),
        ]
        layers.extend(
            _SineLayer(hidden_features, hidden_features, hidden_omega_0)
            for _ in range(hidden_layers - 1)
        )
        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_features, self.feature_count)
        output_bound = math.sqrt(6.0 / hidden_features) / hidden_omega_0
        with torch.no_grad():
            self.output_layer.weight.uniform_(-output_bound, output_bound)
            self.output_layer.bias.zero_()

    def forward(self) -> torch.Tensor:
        height, width = self.image_shape[:2]
        features = self.network(self.coordinates)
        flattened = self.output_layer(features).reshape(
            height,
            width,
            self.feature_count,
        )
        return self._restore_shape(flattened + self.channel_bias)

    @classmethod
    def search_space(cls, image_shape: Tuple[int, ...]) -> Dict[str, Any]:
        return {
            "hidden_features": [64, 128],
            "hidden_layers": [2, 3],
            "first_omega_0": [20.0, 30.0],
            "hidden_omega_0": [20.0, 30.0],
        }
