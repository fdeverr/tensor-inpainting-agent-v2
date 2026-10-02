"""Evaluation-only tensor reconstruction metrics.

The learned metrics are loaded lazily because PyIQA and its pretrained weights
are considerably heavier than the deterministic PSNR/SSIM implementation.
Metric failures are isolated per model so one unavailable checkpoint does not
discard the rest of an experiment.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np


LEARNED_IQA_SPECS = {
    "lpips": {"model_name": "lpips", "reference": True, "lower_better": True},
    "maniqa": {"model_name": "maniqa", "reference": False, "lower_better": False},
    "clip_iqa": {
        "model_name": "clipiqa",
        "reference": False,
        "lower_better": False,
    },
    "musiq": {"model_name": "musiq", "reference": False, "lower_better": False},
}

FULL_REFERENCE_LEARNED_METRICS = ("lpips",)
NO_REFERENCE_METRICS = ("maniqa", "clip_iqa", "musiq")

_LEARNED_MODEL_CACHE: Dict[Tuple[str, str], Any] = {}


def _validate_metric_inputs(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    observed_mask: np.ndarray,
) -> None:
    if prediction.shape != ground_truth.shape:
        raise ValueError("prediction and ground_truth must have identical shapes")
    if prediction.ndim not in (3, 4):
        raise ValueError("inputs must have shape [H,W,C] or [H,W,T,C]")
    if any(int(size) <= 0 for size in prediction.shape):
        raise ValueError("all tensor dimensions must be positive")
    if observed_mask.shape not in (prediction.shape[:2], prediction.shape) or observed_mask.dtype != np.bool_:
        raise ValueError("observed_mask must be bool and match image height and width")
    if observed_mask.all():
        raise ValueError("at least one missing pixel is required for missing-region metrics")
    if not np.isfinite(prediction).all() or not np.isfinite(ground_truth).all():
        raise ValueError("metric inputs contain NaN or Inf")


def missing_region_mse(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    observed_mask: np.ndarray,
) -> float:
    """Mean squared error over missing locations and every trailing feature."""

    _validate_metric_inputs(prediction, ground_truth, observed_mask)
    difference = prediction[~observed_mask] - ground_truth[~observed_mask]
    return float(np.mean(np.square(difference, dtype=np.float64)))


def missing_region_psnr(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    observed_mask: np.ndarray,
    data_range: float = 1.0,
) -> float:
    """Peak signal-to-noise ratio calculated only in the missing region."""

    if data_range <= 0:
        raise ValueError("data_range must be positive")
    mse = missing_region_mse(prediction, ground_truth, observed_mask)
    if mse == 0.0:
        return float("inf")
    return float(10.0 * math.log10((data_range * data_range) / mse))


def full_image_mse(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    observed_mask: np.ndarray,
) -> float:
    """MSE over the completed full tensor after restoring observed samples."""

    _validate_metric_inputs(prediction, ground_truth, observed_mask)
    composite = prediction.copy()
    composite[observed_mask] = ground_truth[observed_mask]
    difference = composite - ground_truth
    return float(np.mean(np.square(difference, dtype=np.float64)))


def full_image_psnr(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    observed_mask: np.ndarray,
    data_range: float = 1.0,
) -> float:
    """PSNR over the entire completed tensor, including restored observed samples."""

    if data_range <= 0:
        raise ValueError("data_range must be positive")
    mse = full_image_mse(prediction, ground_truth, observed_mask)
    if mse == 0.0:
        return float("inf")
    return float(10.0 * math.log10((data_range * data_range) / mse))


def _gaussian_kernel(window_size: int, sigma: float) -> np.ndarray:
    coordinates = np.arange(window_size, dtype=np.float64) - window_size // 2
    kernel_1d = np.exp(-(coordinates ** 2) / (2.0 * sigma * sigma))
    kernel_1d /= kernel_1d.sum()
    return np.outer(kernel_1d, kernel_1d)


def _filter2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    padding = kernel.shape[0] // 2
    padded = np.pad(image, padding, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel.shape)
    return np.einsum("ijkl,kl->ij", windows, kernel, optimize=True)


def _global_ssim(channel_x: np.ndarray, channel_y: np.ndarray, data_range: float) -> float:
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mean_x = float(channel_x.mean())
    mean_y = float(channel_y.mean())
    variance_x = float(channel_x.var())
    variance_y = float(channel_y.var())
    covariance = float(np.mean((channel_x - mean_x) * (channel_y - mean_y)))
    numerator = (2.0 * mean_x * mean_y + c1) * (2.0 * covariance + c2)
    denominator = (mean_x ** 2 + mean_y ** 2 + c1) * (
        variance_x + variance_y + c2
    )
    return numerator / denominator


def structural_similarity(
    image_x: np.ndarray,
    image_y: np.ndarray,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """Compute mean SSIM over all channel/frame feature planes."""

    if image_x.shape != image_y.shape or image_x.ndim not in (3, 4):
        raise ValueError("SSIM inputs must both be matching HWC or HWTC tensors")
    if data_range <= 0 or sigma <= 0:
        raise ValueError("data_range and sigma must be positive")
    if not np.isfinite(image_x).all() or not np.isfinite(image_y).all():
        raise ValueError("SSIM inputs contain NaN or Inf")

    height, width = image_x.shape[:2]
    flattened_x = image_x.reshape(height, width, -1)
    flattened_y = image_y.reshape(height, width, -1)
    features = flattened_x.shape[-1]
    effective_window = min(window_size, height, width)
    if effective_window % 2 == 0:
        effective_window -= 1

    channel_scores = []
    if effective_window < 3:
        for channel in range(features):
            channel_scores.append(
                _global_ssim(
                    flattened_x[..., channel], flattened_y[..., channel], data_range
                )
            )
        return float(np.mean(channel_scores))

    kernel = _gaussian_kernel(effective_window, sigma)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    for channel in range(features):
        x = flattened_x[..., channel].astype(np.float64, copy=False)
        y = flattened_y[..., channel].astype(np.float64, copy=False)
        mean_x = _filter2d(x, kernel)
        mean_y = _filter2d(y, kernel)
        variance_x = np.maximum(0.0, _filter2d(x * x, kernel) - mean_x * mean_x)
        variance_y = np.maximum(0.0, _filter2d(y * y, kernel) - mean_y * mean_y)
        covariance = _filter2d(x * y, kernel) - mean_x * mean_y
        numerator = (2.0 * mean_x * mean_y + c1) * (2.0 * covariance + c2)
        denominator = (mean_x * mean_x + mean_y * mean_y + c1) * (
            variance_x + variance_y + c2
        )
        channel_scores.append(float(np.mean(numerator / denominator)))

    return float(np.mean(channel_scores))


def composite_ssim(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    observed_mask: np.ndarray,
) -> float:
    """Compute SSIM after restoring known pixels from ground truth.

    This is intentionally named *composite* SSIM.  It is not a standardized
    masked SSIM: known pixels are copied from the ground truth so all remaining
    error comes from the inpainted region, but known areas can still dilute the
    global average.
    """

    _validate_metric_inputs(prediction, ground_truth, observed_mask)
    composite = prediction.copy()
    composite[observed_mask] = ground_truth[observed_mask]
    return structural_similarity(composite, ground_truth)


def _resolve_learned_metric_device(device: str) -> str:
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _image_tensor(image: np.ndarray, device: str):
    """Convert an HWC RGB image in [0, 1] to PyIQA's NCHW convention."""

    import torch

    clipped = np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)
    contiguous = np.ascontiguousarray(clipped.transpose(2, 0, 1)[None, ...])
    return torch.from_numpy(contiguous).to(device=device)


def _learned_metric_model(model_name: str, device: str):
    cache_key = (model_name, device)
    if cache_key not in _LEARNED_MODEL_CACHE:
        import pyiqa

        _LEARNED_MODEL_CACHE[cache_key] = pyiqa.create_metric(
            model_name,
            device=device,
        )
    return _LEARNED_MODEL_CACHE[cache_key]


def _error_text(error: Exception) -> str:
    message = "%s: %s" % (type(error).__name__, error)
    return message[:500]


def learned_image_quality_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    observed_mask: np.ndarray,
    device: str = "auto",
    include_full_reference_metrics: bool = False,
    include_no_reference_metrics: bool = False,
) -> Tuple[Dict[str, Optional[float]], Dict[str, Any]]:
    """Evaluate LPIPS, MANIQA, CLIP-IQA, and MUSIQ on the completed image.

    Known pixels are restored from ``ground_truth`` before inference. LPIPS is
    full-reference; the remaining models are no-reference quality estimators.
    This is deliberately an evaluation-only function and must never be used as
    a tuning or training signal.
    """

    _validate_metric_inputs(prediction, ground_truth, observed_mask)
    scores: Dict[str, Optional[float]] = {
        metric_name: None for metric_name in LEARNED_IQA_SPECS
    }
    requested_metrics = list(
        FULL_REFERENCE_LEARNED_METRICS
        if include_full_reference_metrics
        else ()
    ) + list(NO_REFERENCE_METRICS if include_no_reference_metrics else ())
    if not requested_metrics:
        return scores, {
            "enabled": False,
            "requested": False,
            "requested_metrics": [],
            "device": None,
            "scope": "composite_full_image",
            "errors": {},
            "skipped_reason": "optional perceptual/IQA metrics disabled",
        }
    if prediction.ndim != 3 or prediction.shape[-1] != 3:
        return scores, {
            "enabled": False,
            "requested": True,
            "requested_metrics": requested_metrics,
            "device": None,
            "scope": "unsupported_non_rgb_tensor",
            "errors": {},
            "skipped_reason": (
                "%s only support RGB [H,W,3]; "
                "PSNR and SSIM were computed over all tensor features"
                % "/".join(requested_metrics).upper()
            ),
        }
    status: Dict[str, Any] = {
        "enabled": True,
        "requested": True,
        "requested_metrics": requested_metrics,
        "device": None,
        "scope": "composite_full_image",
        "errors": {},
    }
    try:
        resolved_device = _resolve_learned_metric_device(device)
        status["device"] = resolved_device
        import pyiqa  # noqa: F401  # fail all learned metrics once if unavailable

        composite = prediction.copy()
        composite[observed_mask] = ground_truth[observed_mask]
        prediction_tensor = _image_tensor(composite, resolved_device)
        reference_tensor = _image_tensor(ground_truth, resolved_device)
        import torch
    except Exception as error:
        message = _error_text(error)
        status["errors"] = {
            metric_name: message for metric_name in requested_metrics
        }
        return scores, status

    for metric_name in requested_metrics:
        spec = LEARNED_IQA_SPECS[metric_name]
        try:
            model = _learned_metric_model(str(spec["model_name"]), resolved_device)
            with torch.inference_mode():
                if spec["reference"]:
                    output = model(prediction_tensor, reference_tensor)
                else:
                    output = model(prediction_tensor)
            value = float(torch.as_tensor(output).detach().cpu().reshape(-1)[0])
            if not math.isfinite(value):
                raise ValueError("metric returned NaN or Inf")
            scores[metric_name] = value
        except Exception as error:
            status["errors"][metric_name] = _error_text(error)
    return scores, status


def evaluate_reconstruction_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    observed_mask: np.ndarray,
    *,
    device: str = "auto",
    include_full_reference_metrics: bool = False,
    include_no_reference_metrics: bool = False,
) -> Dict[str, Any]:
    """Return the complete metric payload used by every experiment path."""

    from .audio_metrics import active_audio_metadata, audio_nmse
    audio = active_audio_metadata()
    if audio is not None:
        return audio_nmse(prediction, ground_truth, observed_mask, audio)

    missing_psnr = missing_region_psnr(prediction, ground_truth, observed_mask)
    full_psnr = full_image_psnr(prediction, ground_truth, observed_mask)
    result: Dict[str, Any] = {
        "missing_mse": missing_region_mse(prediction, ground_truth, observed_mask),
        "missing_psnr": missing_psnr if math.isfinite(missing_psnr) else None,
        "full_psnr": full_psnr if math.isfinite(full_psnr) else None,
        "perfect_reconstruction": not math.isfinite(full_psnr),
        "composite_ssim": composite_ssim(prediction, ground_truth, observed_mask),
    }
    if include_full_reference_metrics or include_no_reference_metrics:
        learned_scores, learned_status = learned_image_quality_metrics(
            prediction,
            ground_truth,
            observed_mask,
            device=device,
            include_full_reference_metrics=include_full_reference_metrics,
            include_no_reference_metrics=include_no_reference_metrics,
        )
    else:
        learned_scores = {
            metric_name: None for metric_name in LEARNED_IQA_SPECS
        }
        learned_status = {
            "enabled": False,
            "requested": False,
            "device": None,
            "scope": "composite_full_image",
            "errors": {},
            "requested_metrics": [],
            "skipped_reason": "optional perceptual/IQA metrics disabled by configuration",
        }
    result.update(learned_scores)
    result["learned_metric_status"] = learned_status
    result["metric_group_status"] = {
        "full_reference": {
            "enabled": True,
            "required_metrics": [
                "missing_mse",
                "missing_psnr",
                "full_psnr",
                "composite_ssim",
            ],
            "optional_metrics": (
                list(FULL_REFERENCE_LEARNED_METRICS)
                if include_full_reference_metrics
                else []
            ),
        },
        "no_reference": {
            "enabled": include_no_reference_metrics,
            "metrics": (
                list(NO_REFERENCE_METRICS)
                if include_no_reference_metrics
                else []
            ),
        },
    }
    return result
