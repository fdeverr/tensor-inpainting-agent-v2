"""GT-only adapters for Image/MSI/Video and reversible waveform tensors."""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat, whosmat, wavfile

from .core.data import _normalize_array, _resize_tensor_spatial, load_tensor_data, save_mask, save_tensor_data
from .core.masks import generate_tensor_mask
from .core.audio_metrics import restore_waveform
from .workflow import _write_json


DATA_TYPES = {"Image": "color_image", "MSI": "msi", "Video": "video", "audio": "audio"}


def load_recovery_gt(path: str, image_size=None, audio_frame_size=256):
    """Never use Nhsi/mask. Dataset MAT videos use HWCT, converted to HWTC."""
    source = Path(path)
    metadata: dict[str, Any] = {"source": str(source.resolve())}
    if source.suffix.lower() == ".wav":
        with warnings.catch_warnings(record=True) as messages:
            warnings.simplefilter("always")
            sample_rate, raw = wavfile.read(source)
        if messages:
            raise ValueError("audio file integrity warning: " + "; ".join(str(x.message) for x in messages))
        if raw.ndim == 1:
            raw = raw[:, None]
        if raw.ndim != 2 or raw.shape[0] < 2 or audio_frame_size < 2:
            raise ValueError("expected mono/stereo waveform and frame_size >= 2")
        if np.issubdtype(raw.dtype, np.integer):
            info = np.iinfo(raw.dtype)
            scale = 128 if raw.dtype == np.uint8 else max(abs(info.min), info.max)
            waveform = (raw.astype(np.float32) - (128 if raw.dtype == np.uint8 else 0)) / scale
        else:
            waveform = raw.astype(np.float32)
        normalized, normalization = _normalize_array(waveform)
        length, channels = normalized.shape
        frame_size = min(audio_frame_size, length)
        frames = (length + frame_size - 1) // frame_size
        padded = np.zeros((frames * frame_size, channels), dtype=np.float32)
        padded[:length] = normalized
        data = padded.reshape(frames, frame_size, channels)
        metadata.update(normalization)
        metadata.update(data_type="audio", sample_rate=int(sample_rate), sample_count=length,
                        channels=channels, frame_size=frame_size, padding_samples=len(padded) - length,
                        tensor_axes=["time_frame", "sample_in_frame", "channel"])
    elif source.suffix.lower() == ".mat":
        contents = loadmat(source, variable_names=["Ohsi"])
        if "Ohsi" not in contents:
            raise ValueError("dataset MAT must contain GT variable Ohsi")
        raw = contents["Ohsi"]
        metadata["source_shape"] = list(raw.shape)
        if raw.ndim == 4:
            raw = raw.transpose(0, 1, 3, 2)
            category = "Video"
            metadata["tensor_axes"] = ["height", "width", "time", "channel"]
        elif raw.ndim == 3:
            category = "Image" if raw.shape[-1] == 3 else "MSI"
            metadata["tensor_axes"] = ["height", "width", "channel" if category == "Image" else "band"]
        else:
            raise ValueError("GT must have three or four axes")
        data, normalization = _normalize_array(raw)
        metadata.update(normalization)
        metadata["data_type"] = category
        data = _resize_tensor_spatial(data, image_size)
    else:
        data = load_tensor_data(str(source), max_size=image_size)
        metadata["data_type"] = "Image" if data.ndim == 3 and data.shape[-1] == 3 else "Video" if data.ndim == 4 else "MSI"
    metadata["tensor_shape"] = list(data.shape)
    metadata["gt_sha256"] = hashlib.sha256(data.tobytes()).hexdigest()
    return np.ascontiguousarray(data), metadata


def discover_recovery_data(root: str, image_size=None, audio_frame_size=256):
    directory = Path(root)
    if not directory.is_dir():
        raise ValueError("dataset directory does not exist: %s" % root)
    groups = {key: [] for key in DATA_TYPES}
    failures = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or any(part.startswith(".") for part in path.relative_to(directory).parts):
            continue
        if path.suffix.lower() not in {".mat", ".wav", ".npy", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            continue
        try:
            _, metadata = load_recovery_gt(str(path), image_size, audio_frame_size)
            groups[metadata["data_type"]].append(metadata)
        except Exception as error:
            category = "audio" if path.suffix.lower() == ".wav" else None
            if path.suffix.lower() == ".mat":
                try:
                    shape = next(shape for name, shape, _ in whosmat(path) if name == "Ohsi")
                    category = "Video" if len(shape) == 4 else "Image" if len(shape) == 3 and shape[-1] == 3 else "MSI"
                except Exception:
                    pass
            failures.append({"source": str(path.resolve()), "data_type": category,
                             "error": "%s: %s" % (type(error).__name__, error)})
    return {"groups": groups, "failures": failures}


def prepare_recovery_case(path, output_dir, missing_rate, mask_type, seed,
                          image_size=None, audio_frame_size=256):
    data, metadata = load_recovery_gt(path, image_size, audio_frame_size)
    kind = metadata["data_type"]
    pattern = "slices" if mask_type == "sildes" else mask_type
    axis = {"Image": 0, "MSI": 2, "Video": 2, "audio": 0}[kind]
    # Audio missingness is defined on time samples, jointly across channels.
    if kind == "audio":
        frames, width, channels = data.shape
        length = metadata["sample_count"]
        rng = np.random.default_rng(seed)
        count = min(length - 1, max(1, round(length * missing_rate)))
        mask = np.ones((frames * width, channels), dtype=np.bool_)
        if pattern == "random":
            hidden = rng.choice(length, count, replace=False)
        elif pattern in {"block", "slices"}:
            start = int(rng.integers(0, length - count + 1))
            hidden = np.arange(start, start + count)
        else:
            raise ValueError("unsupported mask_type")
        mask[hidden] = False
        mask = mask.reshape(data.shape)
        actual_rate = count / length
    else:
        mask = generate_tensor_mask(data.shape, missing_rate, pattern, seed, axis)
        actual_rate = float((~mask).mean())
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    gt_path, mask_path = destination / "gt.npy", destination / "mask.npy"
    save_tensor_data(str(gt_path), data)
    save_mask(str(mask_path), mask)
    metadata.update(mask_type=pattern, slice_axis=axis if pattern == "slices" else None,
                    requested_missing_rate=missing_rate, actual_missing_rate=actual_rate, seed=seed,
                    mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
                    gt_path=str(gt_path.resolve()), mask_path=str(mask_path.resolve()))
    _write_json(destination / "case.json", metadata)
    return metadata


def export_audio(prediction, metadata, path):
    waveform = restore_waveform(prediction, metadata)
    if metadata["channels"] == 1:
        waveform = waveform[:, 0]
    wavfile.write(path, metadata["sample_rate"], waveform.astype(np.float32))
