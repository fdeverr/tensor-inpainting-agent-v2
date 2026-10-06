"""Coordinate-based SIREN baseline for image and tensor inpainting."""

from __future__ import annotations

import math
from typing import Any, Dict, Sequence, Tuple

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

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
    """Image/MSI: (x,y) -> channels; video: (x,y,t) -> channels; audio: t -> channels.

    Audio frames are a storage layout only: time is continuous across their
    boundaries. Coordinate chunks bound activation memory while retaining the
    same full-observation loss and optimizer step as the other models.
    """

    def __init__(
        self,
        image_shape: Tuple[int, ...],
        initial_channel_mean: Sequence[float],
        hidden_features: int = 128,
        hidden_layers: int = 3,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
        coordinate_mode: str = "auto",
        coordinate_batch_size: int = 32768,
        sample_count: int | None = None,
    ) -> None:
        super().__init__(image_shape, initial_channel_mean)
        if hidden_features < 8:
            raise ValueError("hidden_features must be at least 8")
        if not 1 <= hidden_layers <= 8:
            raise ValueError("hidden_layers must be in [1, 8]")
        if first_omega_0 <= 0.0 or hidden_omega_0 <= 0.0:
            raise ValueError("SIREN omega values must be positive")
        if coordinate_batch_size < 1:
            raise ValueError("coordinate_batch_size must be positive")
        self.coordinate_mode = (
            ("video" if len(image_shape) == 4 else "image")
            if coordinate_mode == "auto" else coordinate_mode
        )
        if self.coordinate_mode not in {"image", "video", "audio"}:
            raise ValueError("coordinate_mode must be auto/image/video/audio")
        if (self.coordinate_mode == "video") != (len(image_shape) == 4):
            raise ValueError("video coordinates require an H-W-T-C tensor")
        self.coordinate_batch_size = int(coordinate_batch_size)

        height, width = self.image_shape[:2]
        if self.coordinate_mode == "audio":
            count = height * width
            valid_count = count if sample_count is None else int(sample_count)
            if not 1 <= valid_count <= count:
                raise ValueError("audio sample_count is outside the tensor length")
            # Padding follows the last valid time; its storage layout cannot
            # change coordinates of the original waveform samples.
            coordinates = (
                2.0 * torch.arange(count) / max(valid_count - 1, 1) - 1.0
            ).reshape(-1, 1)
        else:
            y = torch.linspace(-1.0, 1.0, height)
            x = torch.linspace(-1.0, 1.0, width)
            if self.coordinate_mode == "video":
                t = torch.linspace(-1.0, 1.0, self.image_shape[2])
                grid_y, grid_x, grid_t = torch.meshgrid(y, x, t, indexing="ij")
                coordinates = torch.stack((grid_x, grid_y, grid_t), dim=-1).reshape(-1, 3)
                channel_mean = self.channel_bias.detach().reshape(
                    self.image_shape[2], self.image_shape[3]
                ).mean(dim=0)
                self.channel_bias = nn.Parameter(channel_mean.reshape(1, 1, 1, -1))
            else:
                grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
                coordinates = torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2)
        self.register_buffer(
            "coordinates",
            coordinates,
            persistent=False,
        )

        layers = [
            _SineLayer(coordinates.shape[-1], hidden_features, first_omega_0, first=True),
        ]
        layers.extend(
            _SineLayer(hidden_features, hidden_features, hidden_omega_0)
            for _ in range(hidden_layers - 1)
        )
        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_features, self.image_shape[-1])
        output_bound = math.sqrt(6.0 / hidden_features) / hidden_omega_0
        with torch.no_grad():
            self.output_layer.weight.uniform_(-output_bound, output_bound)
            self.output_layer.bias.zero_()

    def forward(self) -> torch.Tensor:
        predictions = []
        for coordinates in self.coordinates.split(self.coordinate_batch_size):
            if self.training and torch.is_grad_enabled() and len(self.coordinates) > self.coordinate_batch_size:
                prediction = checkpoint(self._predict_coordinates, coordinates, use_reentrant=False)
            else:
                prediction = self._predict_coordinates(coordinates)
            predictions.append(prediction)
        return torch.cat(predictions).reshape(self.image_shape) + self.channel_bias

    def _predict_coordinates(self, coordinates: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.network(coordinates))

    @classmethod
    def search_space(cls, image_shape: Tuple[int, ...]) -> Dict[str, Any]:
        return {
            "hidden_features": [128, 256],
            "hidden_layers": [3, 4],
            "first_omega_0": [20.0, 30.0, 60.0],
            "hidden_omega_0": [20.0, 30.0],
        }
