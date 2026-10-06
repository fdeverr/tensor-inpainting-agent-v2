"""Deterministic interpolation baselines."""

from __future__ import annotations

from collections import deque

import numpy as np

from .audio_metrics import active_audio_metadata


def linear_waveform_fill(observed: np.ndarray, mask: np.ndarray, sample_count: int) -> np.ndarray:
    """Linear interpolation along valid sample time, independently per channel."""
    if observed.ndim != 3 or mask.shape != observed.shape or mask.dtype != np.bool_:
        raise ValueError("audio interpolation requires a framed waveform and matching bool mask")
    channels = observed.shape[-1]
    values = observed.reshape(-1, channels)
    known_mask = mask.reshape(-1, channels)
    if not 1 <= sample_count <= len(values) or not np.isfinite(observed).all():
        raise ValueError("invalid waveform length or nonfinite observed values")
    result = values.copy()
    positions = np.arange(sample_count)
    for channel in range(channels):
        known = positions[known_mask[:sample_count, channel]]
        if not len(known):
            raise ValueError("audio interpolation needs an observed sample per channel")
        result[:sample_count, channel] = np.where(
            known_mask[:sample_count, channel], values[:sample_count, channel],
            np.interp(positions, known, values[known, channel]))
    return result.reshape(observed.shape)


def _nearest_nd(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import distance_transform_cdt
    if not mask.any():
        raise ValueError("nearest-neighbor filling requires an observed value in each color channel")
    indices = distance_transform_cdt(
        ~mask, metric="taxicab", return_distances=False, return_indices=True)
    return values[tuple(indices)].copy()


def nearest_neighbor_fill(
    observed_image: np.ndarray,
    observed_mask: np.ndarray,
    data_type: str | None = None,
) -> np.ndarray:
    """Modality-aware interpolation with no access to hidden ground truth.

    RGB uses (y,x) independently per channel; video uses (y,x,t) independently
    per channel. MSI may use its ordered spectral axis to recover missing bands.
    An active audio protocol dispatches to continuous 1-D linear interpolation.
    A shared spatial mask retains the deterministic O(HW) breadth-first path.
    """

    if observed_image.ndim not in (3, 4):
        raise ValueError("observed_image must have shape [H,W,C] or [H,W,T,C]")
    if observed_mask.shape not in (observed_image.shape[:2], observed_image.shape) or observed_mask.dtype != np.bool_:
        raise ValueError("observed_mask must be bool and match image height and width")
    if not observed_mask.any():
        raise ValueError("nearest-neighbor filling requires at least one observed pixel")
    if not np.isfinite(observed_image).all():
        raise ValueError("observed_image contains NaN or Inf")

    audio = active_audio_metadata()
    if data_type is None:
        data_type = ("audio" if audio else "video" if observed_image.ndim == 4 else
                     "color_image" if observed_image.shape[-1] == 3 else "msi")
    aliases = {"Image": "color_image", "image": "color_image", "Video": "video", "MSI": "msi"}
    data_type = aliases.get(data_type, data_type)
    if data_type not in {"color_image", "msi", "video", "audio"}:
        raise ValueError("unsupported interpolation data_type")
    if (data_type == "video") != (observed_image.ndim == 4):
        raise ValueError("video interpolation requires H-W-T-C data; other modalities require H-W-C")
    if data_type == "audio":
        full_mask = (np.broadcast_to(observed_mask[..., None], observed_image.shape)
                     if observed_mask.ndim == 2 else observed_mask)
        sample_count = audio["sample_count"] if audio else int(np.prod(observed_image.shape[:2]))
        return linear_waveform_fill(observed_image, full_mask, sample_count)

    if observed_mask.ndim > 2:
        if data_type == "msi":
            return _nearest_nd(observed_image, observed_mask)
        reconstructed = observed_image.copy()
        for channel in range(observed_image.shape[-1]):
            reconstructed[..., channel] = _nearest_nd(
                observed_image[..., channel], observed_mask[..., channel])
        return reconstructed

    reconstructed = observed_image.copy()
    visited = observed_mask.copy()
    queue = deque(zip(*np.nonzero(observed_mask)))
    height, width = observed_mask.shape

    while queue:
        y, x = queue.popleft()
        for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if not 0 <= next_y < height or not 0 <= next_x < width:
                continue
            if visited[next_y, next_x]:
                continue
            reconstructed[next_y, next_x] = reconstructed[y, x]
            visited[next_y, next_x] = True
            queue.append((next_y, next_x))

    return reconstructed
