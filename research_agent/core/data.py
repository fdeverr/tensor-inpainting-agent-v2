"""Tensor data I/O, previews, and observation-mask application."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image


SUPPORTED_TENSOR_NDIMS = (3, 4)


def validate_tensor_array(data: np.ndarray) -> None:
    """Validate an in-memory HWC or HWTC tensor normalized to floating point."""

    if data.ndim not in SUPPORTED_TENSOR_NDIMS:
        raise ValueError("expected data with shape [H, W, C] or [H, W, T, C]")
    if any(int(size) <= 0 for size in data.shape):
        raise ValueError("all tensor dimensions must be positive")
    if not np.issubdtype(data.dtype, np.floating):
        raise ValueError("expected a floating-point tensor")
    if not np.isfinite(data).all():
        raise ValueError("tensor contains NaN or Inf")


def infer_data_type(data: np.ndarray, source_is_image: bool = False) -> str:
    """Return ``color_image``, ``msi``, or ``video`` for a supported tensor."""

    validate_tensor_array(data)
    if data.ndim == 4:
        return "video"
    if source_is_image or data.shape[-1] == 3:
        return "color_image"
    return "msi"


def _normalize_array(array: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError("MAT variable must be a real numeric array")
    if not np.isfinite(array).all():
        raise ValueError("input tensor contains NaN or Inf")

    original_dtype = str(array.dtype)
    original_min = float(np.min(array))
    original_max = float(np.max(array))
    data = np.asarray(array, dtype=np.float32)
    normalization = "none"
    if original_min < 0.0 or original_max > 1.0:
        if original_max > original_min:
            data = (data - original_min) / (original_max - original_min)
            normalization = "min_max"
        else:
            data = np.zeros_like(data, dtype=np.float32)
            normalization = "constant_to_zero"
    data = np.clip(data, 0.0, 1.0).astype(np.float32, copy=False)
    return data, {
        "original_dtype": original_dtype,
        "original_min": original_min,
        "original_max": original_max,
        "normalization": normalization,
    }


def _resize_tensor_spatial(data: np.ndarray, max_size: Optional[int]) -> np.ndarray:
    if max_size is None:
        return data
    if max_size <= 0:
        raise ValueError("max_size must be positive")
    height, width = data.shape[:2]
    scale = float(max_size) / float(max(height, width))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    if (new_height, new_width) == (height, width):
        return data

    feature_shape = data.shape[2:]
    flattened = data.reshape(height, width, -1)
    resized_planes = []
    for feature_index in range(flattened.shape[-1]):
        plane = Image.fromarray(flattened[..., feature_index], mode="F")
        plane = plane.resize((new_width, new_height), resample=Image.Resampling.BILINEAR)
        resized_planes.append(np.asarray(plane, dtype=np.float32))
    resized = np.stack(resized_planes, axis=-1)
    return np.ascontiguousarray(resized.reshape(new_height, new_width, *feature_shape))


def _mat_candidates(contents: Dict[str, Any]) -> Dict[str, np.ndarray]:
    return {
        str(key): value
        for key, value in contents.items()
        if not str(key).startswith("__")
        and isinstance(value, np.ndarray)
        and value.ndim in SUPPORTED_TENSOR_NDIMS
        and np.issubdtype(value.dtype, np.number)
        and not np.iscomplexobj(value)
    }


def _load_mat(path: Path, mat_key: Optional[str]) -> Tuple[np.ndarray, str, str]:
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise RuntimeError(
            "reading MAT input requires scipy; install project requirements"
        ) from exc

    scipy_error: Optional[Exception] = None
    try:
        contents = loadmat(path)
        backend = "scipy"
    except (NotImplementedError, ValueError) as error:
        scipy_error = error
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError(
                "MATLAB v7.3 input requires h5py; install project requirements"
            ) from exc
        try:
            with h5py.File(path, "r") as handle:
                contents = {
                    str(key): np.asarray(value).transpose(
                        tuple(range(value.ndim - 1, -1, -1))
                    )
                    for key, value in handle.items()
                    if hasattr(value, "shape")
                }
        except OSError:
            raise scipy_error
        backend = "h5py"

    candidates = _mat_candidates(contents)
    if mat_key is not None:
        if mat_key not in contents:
            raise ValueError("MAT variable %r was not found" % mat_key)
        selected = contents[mat_key]
        if mat_key not in candidates:
            raise ValueError(
                "MAT variable %r must be a real numeric [H,W,C] or [H,W,T,C] array"
                % mat_key
            )
        return np.asarray(selected), mat_key, backend

    conventional_names = ("data", "tensor", "image", "msi", "video", "X", "x")
    for name in conventional_names:
        if name in candidates:
            return np.asarray(candidates[name]), name, backend
    if not candidates:
        raise ValueError("MAT file contains no real numeric 3-D or 4-D variable")

    selected_name = sorted(
        candidates,
        key=lambda name: (-int(candidates[name].size), name),
    )[0]
    return np.asarray(candidates[selected_name]), selected_name, backend


def load_tensor_data(
    path: str,
    max_size: Optional[int] = None,
    mat_key: Optional[str] = None,
    return_metadata: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
    """Load RGB, MSI, or video data as float32 in ``[0, 1]``.

    Image files become ``[H,W,3]`` tensors. MAT files must contain a real
    numeric ``[H,W,C]`` or ``[H,W,T,C]`` variable. When ``mat_key`` is not
    supplied, a conventional variable name is preferred; otherwise the
    largest compatible variable is selected deterministically.
    """

    source = Path(path)
    if not source.is_file():
        raise ValueError("data file does not exist: %s" % source)

    suffix = source.suffix.lower()
    metadata: Dict[str, Any] = {"source_path": str(source), "source_format": suffix}
    if suffix == ".mat":
        raw, selected_key, backend = _load_mat(source, mat_key)
        data, normalization = _normalize_array(raw)
        metadata.update(normalization)
        metadata.update({"mat_key": selected_key, "mat_backend": backend})
        source_is_image = False
    elif suffix == ".npy":
        if mat_key is not None:
            raise ValueError("mat_key is only valid for .mat input")
        raw = np.load(source, allow_pickle=False)
        data, normalization = _normalize_array(raw)
        metadata.update(normalization)
        metadata.update({"mat_key": None, "mat_backend": None})
        source_is_image = False
    else:
        with Image.open(source) as pil_image:
            raw = np.asarray(pil_image.convert("RGB"), dtype=np.uint8)
        data = raw.astype(np.float32) / 255.0
        metadata.update(
            {
                "original_dtype": str(raw.dtype),
                "original_min": float(np.min(raw)),
                "original_max": float(np.max(raw)),
                "normalization": "uint8_to_unit",
                "mat_key": None,
                "mat_backend": None,
            }
        )
        source_is_image = True

    validate_tensor_array(data)
    original_shape = tuple(int(value) for value in data.shape)
    data = _resize_tensor_spatial(data, max_size)
    validate_tensor_array(data)
    metadata.update(
        {
            "original_shape": list(original_shape),
            "loaded_shape": [int(value) for value in data.shape],
            "data_type": infer_data_type(data, source_is_image=source_is_image),
            "feature_shape": [int(value) for value in data.shape[2:]],
            "feature_count": int(math.prod(data.shape[2:])),
        }
    )
    if return_metadata:
        return data, metadata
    return data


def load_rgb_image(path: str, max_size: Optional[int] = None) -> np.ndarray:
    """Backward-compatible strict RGB image loader."""

    data, metadata = load_tensor_data(path, max_size=max_size, return_metadata=True)
    if data.ndim != 3 or data.shape[-1] != 3 or metadata["source_format"] == ".mat":
        raise ValueError("expected a standard RGB image with shape [H, W, 3]")
    return data


def load_tensor_prediction(path: str) -> np.ndarray:
    """Read saved predictions without rescaling errors or clipping model outputs."""
    source = Path(path)
    if source.suffix.lower() == ".npy":
        data = np.load(source, allow_pickle=False)
    elif source.suffix.lower() == ".mat":
        data, _, _ = _load_mat(source, None)
    else:
        return load_tensor_data(path)
    data = np.asarray(data, dtype=np.float32)
    validate_tensor_array(data)
    return data


def to_rgb_preview(data: np.ndarray) -> np.ndarray:
    """Create an RGB preview without changing the underlying tensor data."""

    validate_tensor_array(data)
    preview_source = data[:, :, data.shape[2] // 2, :] if data.ndim == 4 else data
    channels = preview_source.shape[-1]
    if channels == 1:
        preview = np.repeat(preview_source, 3, axis=-1)
    elif channels == 2:
        preview = np.stack(
            (preview_source[..., 0], preview_source[..., 1], preview_source.mean(axis=-1)),
            axis=-1,
        )
    elif channels == 3:
        preview = preview_source
    else:
        preview = preview_source[..., [channels - 1, channels // 2, 0]]
    return np.asarray(np.clip(preview, 0.0, 1.0), dtype=np.float32)


def save_image(path: str, data: np.ndarray) -> None:
    """Save an RGB visualization of HWC or HWTC tensor data."""

    preview = to_rgb_preview(data)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    uint8_image = np.rint(preview * 255.0).astype(np.uint8)
    Image.fromarray(uint8_image).save(destination)


def save_tensor_data(path: str, data: np.ndarray, key: str = "data") -> None:
    """Save the complete tensor to MAT or NPY without preview loss."""

    validate_tensor_array(data)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower()
    if suffix == ".mat":
        try:
            from scipy.io import savemat
        except ImportError as exc:
            raise RuntimeError("saving MAT data requires scipy") from exc
        savemat(destination, {key: np.asarray(data, dtype=np.float32)})
    elif suffix == ".npy":
        np.save(destination, np.asarray(data, dtype=np.float32), allow_pickle=False)
    else:
        raise ValueError("tensor output path must end in .mat or .npy")


def save_mat_companion(path: str, data: np.ndarray, key: str = "data") -> Optional[str]:
    """Save a MAT companion when SciPy is available.

    SciPy is a declared project dependency. The optional return keeps basic
    RGB workflows usable in deliberately minimal environments that have not
    installed the full requirements yet.
    """

    try:
        save_tensor_data(path, data, key=key)
    except RuntimeError as error:
        if isinstance(error.__cause__, ImportError):
            return None
        raise
    return str(Path(path))


def save_mask(path: str, observed_mask: np.ndarray) -> None:
    """Save an observation mask with white=observed and black=missing."""

    destination = Path(path)
    if destination.suffix.lower() == ".npy":
        if observed_mask.dtype != np.bool_ or observed_mask.ndim not in (2, 3, 4):
            raise ValueError("mask must be a 2-D, 3-D or 4-D boolean array")
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, observed_mask, allow_pickle=False)
        return
    if observed_mask.ndim != 2 or observed_mask.dtype != np.bool_:
        raise ValueError("observed_mask must be a bool array with shape [H, W]")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    mask_image = observed_mask.astype(np.uint8) * 255
    Image.fromarray(mask_image).save(destination)


def load_observation_mask(path: str) -> np.ndarray:
    """Load a saved mask using the project convention white=observed."""

    source = Path(path)
    if not source.is_file():
        raise ValueError("mask file does not exist: %s" % source)
    if source.suffix.lower() == ".npy":
        mask = np.load(source, allow_pickle=False)
        if mask.dtype != np.bool_:
            raise ValueError("NPY mask must be boolean")
    else:
        with Image.open(source) as mask_image:
            mask = np.asarray(mask_image.convert("L"), dtype=np.uint8) >= 128
    if mask.ndim not in (2, 3, 4) or not mask.any() or mask.all():
        raise ValueError("mask must contain both observed and missing pixels")
    return mask.astype(np.bool_)


def apply_observation_mask(
    data: np.ndarray,
    observed_mask: np.ndarray,
    missing_fill_value: float = 0.0,
) -> np.ndarray:
    """Hide missing spatial samples across every channel/frame feature."""

    validate_tensor_array(data)
    if observed_mask.shape not in (data.shape[:2], data.shape) or observed_mask.dtype != np.bool_:
        raise ValueError("observed_mask must be bool and match tensor height and width")
    if not 0.0 <= missing_fill_value <= 1.0:
        raise ValueError("missing_fill_value must be in [0, 1]")

    corrupted = data.copy()
    corrupted[~observed_mask] = missing_fill_value
    return corrupted
