"""Reproducible synthetic observation masks."""

from __future__ import annotations

import math

import numpy as np


def _target_missing_pixels(height: int, width: int, missing_rate: float) -> int:
    if height < 1 or width < 1:
        raise ValueError("height and width must be positive")
    if not 0.0 < missing_rate < 1.0:
        raise ValueError("missing_rate must be strictly between 0 and 1")
    pixel_count = height * width
    return min(pixel_count - 1, max(1, int(round(pixel_count * missing_rate))))


def _random_mask(
    height: int,
    width: int,
    target_missing: int,
    rng: np.random.Generator,
) -> np.ndarray:
    observed = np.ones(height * width, dtype=np.bool_)
    missing_indices = rng.choice(height * width, size=target_missing, replace=False)
    observed[missing_indices] = False
    return observed.reshape(height, width)


def _block_mask(
    height: int,
    width: int,
    target_missing: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create one compact, rectangle-like hole with an exact pixel budget."""

    center_y = rng.uniform(0.3 * (height - 1), 0.7 * (height - 1))
    center_x = rng.uniform(0.3 * (width - 1), 0.7 * (width - 1))
    aspect = math.exp(rng.uniform(math.log(0.5), math.log(2.0)))

    y, x = np.indices((height, width), dtype=np.float64)
    y_distance = np.abs(y - center_y) / math.sqrt(aspect)
    x_distance = np.abs(x - center_x) * math.sqrt(aspect)
    compactness_score = np.maximum(y_distance, x_distance)

    flat_scores = compactness_score.ravel()
    missing_indices = np.argpartition(flat_scores, target_missing - 1)[:target_missing]
    observed = np.ones(height * width, dtype=np.bool_)
    observed[missing_indices] = False
    return observed.reshape(height, width)


def generate_observation_mask(
    height: int,
    width: int,
    missing_rate: float,
    mask_type: str,
    seed: int,
) -> np.ndarray:
    """Generate a mask where True=observed and False=missing."""

    target_missing = _target_missing_pixels(height, width, missing_rate)
    rng = np.random.default_rng(seed)

    if mask_type == "random":
        return _random_mask(height, width, target_missing, rng)
    if mask_type == "block":
        return _block_mask(height, width, target_missing, rng)
    raise ValueError("unsupported mask_type %r; expected 'random' or 'block'" % mask_type)


def split_observed_mask(
    observed_mask: np.ndarray,
    validation_ratio: float,
    seed: int,
    strategy: str = "mask_matched",
) -> tuple:
    """Split observed pixels into disjoint training and validation masks.

    Artificially missing pixels remain False in both returned masks.  The split
    is exact, seeded, and never exposes hidden ground truth to tuning.
    """

    if observed_mask.ndim != 2 or observed_mask.dtype != np.bool_:
        raise ValueError("observed_mask must be a bool array with shape [H, W]")
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be strictly between 0 and 1")
    if strategy not in {"random", "mask_matched"}:
        raise ValueError("strategy must be random or mask_matched")

    observed_indices = np.flatnonzero(observed_mask.ravel())
    if observed_indices.size < 2:
        raise ValueError("at least two observed pixels are required for a train/validation split")

    validation_count = min(
        observed_indices.size - 1,
        max(1, int(round(observed_indices.size * validation_ratio))),
    )
    rng = np.random.default_rng(seed)
    if strategy == "mask_matched" and _has_compact_missing_region(observed_mask):
        validation_indices = _compact_observed_holdout(
            observed_mask,
            validation_count,
            rng,
        )
    else:
        validation_indices = rng.choice(
            observed_indices,
            size=validation_count,
            replace=False,
        )

    train_mask = observed_mask.copy().ravel()
    validation_mask = np.zeros_like(train_mask)
    train_mask[validation_indices] = False
    validation_mask[validation_indices] = True
    return train_mask.reshape(observed_mask.shape), validation_mask.reshape(
        observed_mask.shape
    )


def _has_compact_missing_region(observed_mask: np.ndarray) -> bool:
    """Detect whether the real mask is dominated by one connected hole."""

    missing = ~observed_mask
    missing_count = int(missing.sum())
    if missing_count < 2:
        return False
    visited = np.zeros_like(missing, dtype=np.bool_)
    largest = 0
    height, width = missing.shape
    for start_y, start_x in np.argwhere(missing):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        size = 0
        while stack:
            y, x = stack.pop()
            size += 1
            for next_y, next_x in (
                (y - 1, x),
                (y + 1, x),
                (y, x - 1),
                (y, x + 1),
            ):
                if (
                    0 <= next_y < height
                    and 0 <= next_x < width
                    and missing[next_y, next_x]
                    and not visited[next_y, next_x]
                ):
                    visited[next_y, next_x] = True
                    stack.append((next_y, next_x))
        largest = max(largest, size)
    return largest / missing_count >= 0.5


def _compact_observed_holdout(
    observed_mask: np.ndarray,
    validation_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Choose an exact, compact pseudo-hole from genuinely observed pixels."""

    observed_coordinates = np.argwhere(observed_mask)
    missing_coordinates = np.argwhere(~observed_mask)
    if missing_coordinates.size:
        missing_height = int(np.ptp(missing_coordinates[:, 0])) + 1
        missing_width = int(np.ptp(missing_coordinates[:, 1])) + 1
        aspect = float(np.clip(missing_width / max(1, missing_height), 0.25, 4.0))
    else:
        aspect = 1.0

    candidate_count = min(32, len(observed_coordinates))
    center_indices = rng.choice(
        len(observed_coordinates),
        size=candidate_count,
        replace=False,
    )
    best_indices = None
    best_score = None
    for center_index in center_indices:
        center_y, center_x = observed_coordinates[center_index]
        y_distance = np.abs(observed_coordinates[:, 0] - center_y) * np.sqrt(aspect)
        x_distance = np.abs(observed_coordinates[:, 1] - center_x) / np.sqrt(aspect)
        compactness = np.maximum(y_distance, x_distance)
        order = np.lexsort(
            (
                observed_coordinates[:, 1],
                observed_coordinates[:, 0],
                compactness,
            )
        )
        selected = order[:validation_count]
        selected_coordinates = observed_coordinates[selected]
        box_height = int(np.ptp(selected_coordinates[:, 0])) + 1
        box_width = int(np.ptp(selected_coordinates[:, 1])) + 1
        box_area = box_height * box_width
        shape_mismatch = abs((box_width / max(1, box_height)) - aspect)
        score = (box_area, shape_mismatch, float(compactness[selected].max()))
        if best_score is None or score < best_score:
            best_score = score
            best_indices = selected

    if best_indices is None:
        raise RuntimeError("failed to construct a compact observed validation holdout")
    height, width = observed_mask.shape
    selected_coordinates = observed_coordinates[best_indices]
    return np.ravel_multi_index(
        (selected_coordinates[:, 0], selected_coordinates[:, 1]),
        dims=(height, width),
    )
