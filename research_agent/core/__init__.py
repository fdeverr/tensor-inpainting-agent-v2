"""Deterministic image-inpainting experiment core.

This package deliberately has no dependency on Tensor Inpainting Agent Framework.  Agent tools will
call into this package, never the other way around.
"""

from .data import (
    apply_observation_mask,
    infer_data_type,
    load_rgb_image,
    load_tensor_data,
    save_image,
    save_mat_companion,
    save_mask,
    save_tensor_data,
    to_rgb_preview,
    validate_tensor_array,
)
from .interpolation import nearest_neighbor_fill
from .masks import generate_observation_mask
from .metrics import (
    composite_ssim,
    evaluate_reconstruction_metrics,
    full_image_mse,
    full_image_psnr,
    learned_image_quality_metrics,
    missing_region_mse,
    missing_region_psnr,
)
from .pipeline import run_day1_baseline

__all__ = [
    "apply_observation_mask",
    "composite_ssim",
    "evaluate_reconstruction_metrics",
    "full_image_mse",
    "full_image_psnr",
    "generate_observation_mask",
    "load_rgb_image",
    "load_tensor_data",
    "save_tensor_data",
    "save_mat_companion",
    "to_rgb_preview",
    "validate_tensor_array",
    "infer_data_type",
    "learned_image_quality_metrics",
    "missing_region_mse",
    "missing_region_psnr",
    "nearest_neighbor_fill",
    "run_day1_baseline",
    "save_image",
    "save_mask",
]
