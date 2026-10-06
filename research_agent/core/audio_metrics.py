"""Original-waveform NMSE protocol, scoped to the current execution context.

Shared synchronous research tools inherit one frozen protocol throughout an
audio run. Context tokens are always reset, including on failures, so a later
image run cannot accidentally inherit the audio metric.
"""

from contextlib import contextmanager
from contextvars import ContextVar
import json
import math
from pathlib import Path

import numpy as np


_AUDIO = ContextVar("recovery_audio_evaluation", default=None)


def active_audio_metadata():
    return _AUDIO.get()


@contextmanager
def audio_metric_context(metadata):
    token = _AUDIO.set(metadata if metadata and metadata.get("data_type") == "audio" else None)
    try:
        yield
    finally:
        _AUDIO.reset(token)


def audio_metadata_for_gt(path):
    case_path = Path(path).parent / "case.json"
    if case_path.is_file():
        metadata = json.loads(case_path.read_text(encoding="utf-8"))
        if metadata.get("data_type") == "audio":
            return metadata
    return None


def audio_valid_mask(tensor_shape, metadata=None):
    """Identify real waveform samples independently of the observation mask."""
    audio = metadata if metadata is not None else active_audio_metadata()
    if audio is None:
        raise ValueError("audio validity requires waveform metadata")
    if len(tensor_shape) != 3 or tensor_shape[-1] != audio["channels"]:
        raise ValueError("audio metadata does not match the framed waveform shape")
    total = int(np.prod(tensor_shape[:2]))
    count = audio["sample_count"]
    if not 1 <= count <= total:
        raise ValueError("invalid audio sample_count")
    return np.broadcast_to((np.arange(total) < count).reshape(*tensor_shape[:2], 1), tensor_shape)


def training_observation_mask(observed_mask, tensor_shape):
    """Exclude padding from supervision without changing the experiment mask."""
    if active_audio_metadata() is None:
        return observed_mask.copy()
    expanded = (np.broadcast_to(observed_mask[..., None], tensor_shape)
                if observed_mask.ndim == 2 else observed_mask)
    return expanded & audio_valid_mask(tensor_shape)


def restore_waveform(tensor, metadata):
    waveform = np.asarray(tensor, dtype=np.float64).reshape(-1, metadata["channels"])
    if len(waveform) < metadata["sample_count"]:
        raise ValueError("audio tensor is shorter than its valid waveform length")
    waveform = waveform[:metadata["sample_count"]]
    if metadata["normalization"] == "min_max":
        waveform = waveform * (metadata["original_max"] - metadata["original_min"]) + metadata["original_min"]
    elif metadata["normalization"] == "constant_to_zero":
        waveform = np.full_like(waveform, metadata["original_min"])
    if not np.isfinite(waveform).all():
        raise ValueError("audio waveform contains NaN or Inf")
    return waveform


def audio_nmse(prediction, ground_truth, observed_mask, metadata):
    if prediction.shape != ground_truth.shape or observed_mask.shape != ground_truth.shape:
        raise ValueError("audio NMSE requires matching full tensor masks and arrays")
    estimate = restore_waveform(prediction, metadata)
    target = restore_waveform(ground_truth, metadata)
    missing = ~observed_mask.reshape(-1, metadata["channels"])[:metadata["sample_count"]]
    if not missing.any():
        raise ValueError("audio NMSE requires at least one missing waveform sample")
    error_energy = float(np.square(estimate[missing] - target[missing]).sum())
    target_energy = float(np.square(target[missing]).sum())
    value = error_energy / target_energy if target_energy > 0 else 0.0 if error_energy == 0 else None
    return {
        "missing_nmse": value, "evaluation_metric": "missing_nmse",
        "perfect_reconstruction": error_energy == 0,
        "audio_metric_status": {
            "defined": value is not None,
            "scope": "missing_original_waveform_samples_all_channels_excluding_padding",
            "normalization": "squared_error_energy / reference_energy; no mean subtraction",
            "error_energy": error_energy, "reference_energy": target_energy,
            "zero_energy_convention": "0 for exact recovery; null for nonzero error",
        },
    }


def metric_score(metrics):
    """Higher-is-better sorting key, without computing audio PSNR."""
    if "missing_nmse" in metrics:
        value = metrics["missing_nmse"]
        return -float(value) if value is not None and math.isfinite(value) else -math.inf
    value = metrics.get("missing_psnr")
    return math.inf if value is None else float(value)


def trial_selection_loss(trial):
    if "best_missing_nmse" in trial:
        return float(trial["best_missing_nmse"]) if trial["best_missing_nmse"] is not None else math.inf
    return trial["best_validation_mse"]
