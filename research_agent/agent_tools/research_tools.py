"""Deterministic Tensor Inpainting Agent Framework tools around the Day 1/2 experiment core.

The tools exchange paths and small JSON payloads. They never place complete
image/MSI/video arrays or model tensors in an Agent context.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from ..core.data import (
    apply_observation_mask,
    load_observation_mask,
    load_tensor_data,
    load_tensor_prediction,
    save_image,
    save_mat_companion,
    save_mask,
    save_tensor_data,
    to_rgb_preview,
)
from ..core.interpolation import nearest_neighbor_fill
from ..core.masks import generate_observation_mask
from ..core.metrics import evaluate_reconstruction_metrics
from ..core.audio_metrics import (
    active_audio_metadata, audio_valid_mask, training_observation_mask,
    metric_score, trial_selection_loss,
)
from ..core.models import get_default_hyperparameters
from ..core.models.registry import MODEL_CLASSES
from ..core.trainer import fit_tensor_model_on_all_observations, train_tensor_model, observed_feature_mean
from ..schemas import SUPPORTED_MASK_TYPES, SUPPORTED_MODEL_NAMES, TrainingConfig
from .framework import Tool, ToolErrorCode, ToolParameter, ToolResponse


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2, allow_nan=False)
        output_file.write("\n")


def _required_string(parameters: Dict[str, Any], name: str) -> str:
    value = parameters.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("%s must be a non-empty string" % name)
    return value


def _persist_selected_training_output(
    output_dir: Path,
    run_id: str,
    model_name: str,
    hyperparameters: Dict[str, Any],
    learning_rate: float,
    observed: np.ndarray,
    observed_mask: np.ndarray,
    output: Any,
) -> Dict[str, Any]:
    """Persist the already-trained GT-best trial for direct downstream reuse."""

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "model_raw.npy"
    completed_path = output_dir / "model_completed.npy"
    raw_mat_path = output_dir / "model_raw.mat"
    completed_mat_path = output_dir / "model_completed.mat"
    raw_preview_path = output_dir / "model_raw_preview.png"
    completed_preview_path = output_dir / "model_completed_preview.png"
    history_path = output_dir / "selection_history.json"
    checkpoint_path = output_dir / "model.pt"
    result_path = output_dir / "training_result.json"
    completed = output.reconstruction.copy()
    completed[observed_mask] = observed[observed_mask]
    save_tensor_data(str(raw_path), output.reconstruction)
    save_tensor_data(str(completed_path), completed)
    raw_mat = save_mat_companion(str(raw_mat_path), output.reconstruction)
    completed_mat = save_mat_companion(str(completed_mat_path), completed)
    save_image(str(raw_preview_path), output.reconstruction)
    save_image(str(completed_preview_path), completed)
    _write_json(history_path, {"history": output.history})
    torch.save(
        {
            "model_name": model_name,
            "hyperparameters": dict(hyperparameters),
            "selected_steps": output.best_step,
            "learning_rate": learning_rate,
            "state_dict": output.state_dict,
            "training_phase": "all_observed_training_gt_checkpoint_selection",
        },
        checkpoint_path,
    )
    best_history = next(
        item for item in output.history if item["step"] == output.best_step
    )
    artifacts = {
        "raw_reconstruction": str(raw_path),
        "reconstruction": str(completed_path),
        "raw_preview": str(raw_preview_path),
        "preview": str(completed_preview_path),
        "history": str(history_path),
        "checkpoint": str(checkpoint_path),
        "training_result": str(result_path),
    }
    if raw_mat is not None:
        artifacts["raw_reconstruction_mat"] = raw_mat
    if completed_mat is not None:
        artifacts["reconstruction_mat"] = completed_mat
    result = {
        "run_id": run_id,
        "model_name": model_name,
        "hyperparameters": dict(hyperparameters),
        "learning_rate": learning_rate,
        "fitted_steps": output.best_step,
        "final_train_mse": best_history["data_train_loss"],
        "final_total_loss": best_history["total_train_loss"],
        "selection_missing_gt_mse": output.best_validation_mse,
        "selection_missing_psnr": output.best_missing_psnr,
        "observed_pixels_used": int(output.train_mask.sum()),
        "parameter_count": output.parameter_count,
        "runtime_seconds": output.runtime_seconds,
        "device": output.device,
        "ground_truth_used_for_selection": True,
        "reused_without_retraining": True,
        "artifacts": artifacts,
    }
    if active_audio_metadata() is not None:
        result.pop("selection_missing_psnr")
        result["selection_missing_nmse"] = output.best_missing_nmse
    _write_json(result_path, {key: value for key, value in result.items() if key != "artifacts"})
    return result


def _load_observed_pair(corrupted_path: str, mask_path: str) -> tuple:
    corrupted = load_tensor_data(corrupted_path, max_size=None)
    observed_mask = load_observation_mask(mask_path)
    if observed_mask.shape not in (corrupted.shape[:2], corrupted.shape):
        raise ValueError("corrupted tensor and mask shapes do not match")
    return corrupted, observed_mask


def _training_config(parameters: Dict[str, Any], learning_rate: float) -> TrainingConfig:
    return TrainingConfig(
        learning_rate=float(learning_rate),
        max_steps=int(parameters.get("max_steps", 200)),
        validation_observed_ratio=float(parameters.get("validation_ratio", 0.1)),
        validation_strategy=str(
            parameters.get("validation_strategy", "mask_matched")
        ),
        validation_interval=int(parameters.get("validation_interval", 10)),
        early_stopping_patience=int(parameters.get("patience", 20)),
        device=str(parameters.get("device", "auto")),
    )


def _metric_payload(
    reconstruction: np.ndarray,
    ground_truth: np.ndarray,
    observed_mask: np.ndarray,
    device: str = "auto",
    include_full_reference_metrics: bool = False,
    include_no_reference_metrics: bool = False,
) -> Dict[str, Any]:
    return evaluate_reconstruction_metrics(
        reconstruction,
        ground_truth,
        observed_mask,
        device=device,
        include_full_reference_metrics=include_full_reference_metrics,
        include_no_reference_metrics=include_no_reference_metrics,
    )


def _missing_component_statistics(observed_mask: np.ndarray, valid_mask=None) -> tuple:
    """Return component count and largest-hole share using 4-connectivity."""

    missing = ~observed_mask
    if valid_mask is not None:
        missing &= valid_mask
    visited = np.zeros_like(missing, dtype=np.bool_)
    component_sizes = []
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
            for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if (
                    0 <= next_y < height
                    and 0 <= next_x < width
                    and missing[next_y, next_x]
                    and not visited[next_y, next_x]
                ):
                    visited[next_y, next_x] = True
                    stack.append((next_y, next_x))
        component_sizes.append(size)
    largest = max(component_sizes) if component_sizes else 0
    return len(component_sizes), float(largest / (int(valid_mask.sum()) if valid_mask is not None else missing.size))


def _visible_structure_statistics(
    image: np.ndarray,
    observed_mask: np.ndarray,
) -> Dict[str, Any]:
    """Compute structure features without sampling artificially hidden values."""

    height, width = image.shape[:2]
    flattened = image.reshape(height, width, -1)
    visible_pixels = flattened[observed_mask]
    feature_count = flattened.shape[-1]
    sampled_indices = np.unique(
        np.linspace(0, feature_count - 1, min(feature_count, 16), dtype=int)
    )
    sampled_pixels = visible_pixels[:, sampled_indices].astype(np.float64)
    if sampled_pixels.shape[1] == 1:
        correlation_matrix = np.ones((1, 1), dtype=np.float64)
    elif np.all(sampled_pixels.std(axis=0) > 1e-8):
        correlation_matrix = np.corrcoef(sampled_pixels, rowvar=False)
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
    else:
        correlation_matrix = np.eye(sampled_pixels.shape[1], dtype=np.float64)
    off_diagonal = correlation_matrix[
        np.triu_indices(correlation_matrix.shape[0], k=1)
    ]

    horizontal_pairs = observed_mask[:, 1:] & observed_mask[:, :-1]
    vertical_pairs = observed_mask[1:, :] & observed_mask[:-1, :]
    absolute_differences = []
    squared_gray_differences = []
    preview = to_rgb_preview(image)
    gray = 0.299 * preview[..., 0] + 0.587 * preview[..., 1] + 0.114 * preview[..., 2]
    if horizontal_pairs.any():
        absolute_differences.append(
            np.abs(image[:, 1:] - image[:, :-1])[horizontal_pairs]
        )
        squared_gray_differences.append(
            np.square(gray[:, 1:] - gray[:, :-1])[horizontal_pairs]
        )
    if vertical_pairs.any():
        absolute_differences.append(
            np.abs(image[1:, :] - image[:-1, :])[vertical_pairs]
        )
        squared_gray_differences.append(
            np.square(gray[1:, :] - gray[:-1, :])[vertical_pairs]
        )
    mean_local_difference = (
        float(np.concatenate(absolute_differences).mean())
        if absolute_differences
        else 0.0
    )
    high_frequency_energy = (
        float(np.concatenate(squared_gray_differences).mean())
        if squared_gray_differences
        else 0.0
    )
    visible_gray_energy = float(np.square(gray[observed_mask]).mean())
    high_frequency_ratio = high_frequency_energy / (visible_gray_energy + 1e-12)
    return {
        "visible_channel_correlation_matrix": correlation_matrix.tolist(),
        "correlation_sampled_feature_indices": sampled_indices.tolist(),
        "visible_mean_absolute_channel_correlation": float(
            np.abs(off_diagonal).mean() if off_diagonal.size else 0.0
        ),
        "visible_mean_local_absolute_difference": mean_local_difference,
        "visible_local_smoothness_score": float(
            np.clip(1.0 - 4.0 * mean_local_difference, 0.0, 1.0)
        ),
        "visible_high_frequency_energy_ratio": float(high_frequency_ratio),
    }


def _default_candidates(
    model_name: str,
    image_shape: tuple[int, ...],
) -> List[Dict[str, Any]]:
    """Build two valid defaults from the selected model's bounded search space."""

    model_class = MODEL_CLASSES[model_name]
    search_space = model_class.search_space(image_shape)
    defaults = get_default_hyperparameters(model_name)
    primary = {
        name: value if value in options else options[-1]
        for name, options in search_space.items()
        for value in [defaults.get(name, options[0])]
    }
    alternatives = []
    for name, options in search_space.items():
        alternative = next((value for value in options if value != primary[name]), None)
        if alternative is not None:
            varied = dict(primary)
            varied[name] = alternative
            alternatives.append(varied)
            break
    configurations = [primary, *alternatives]
    learning_rate = 1e-4 if model_name == "siren" else 0.03
    return [
        {"hyperparameters": item, "learning_rate": learning_rate}
        for item in configurations
    ]


class ResearchTool(Tool):
    """Shared conversion of common input errors to structured responses."""

    @staticmethod
    def invalid(error: Exception) -> ToolResponse:
        return ToolResponse.error(
            code=ToolErrorCode.INVALID_PARAM,
            message=str(error),
        )

    @staticmethod
    def failed(error: Exception) -> ToolResponse:
        return ToolResponse.error(
            code=ToolErrorCode.EXECUTION_ERROR,
            message=str(error),
        )


class AnalyzeImageTool(ResearchTool):
    """Create a masked image/MSI/video case and summarize visible samples."""

    def __init__(self) -> None:
        super().__init__(
            name="analyze_image",
            description=(
                "读取彩图或 MAT 格式的 MSI/视频，创建可复现实验 mask，并根据"
                "可见位置返回数据尺寸、缺失率、特征统计和局部平滑度。"
            ),
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="run_id", type="string", description="工作流运行 ID"),
            ToolParameter(name="image_path", type="string", description="完整评测图片路径"),
            ToolParameter(name="run_dir", type="string", description="本次运行产物目录"),
            ToolParameter(name="mask_type", type="string", description="random 或 block"),
            ToolParameter(name="missing_rate", type="number", description="目标缺失率"),
            ToolParameter(name="seed", type="integer", description="随机种子"),
            ToolParameter(name="observation_mask_path", type="string", description="可选预生成 NPY/PNG mask", required=False),
            ToolParameter(name="data_type", type="string", description="显式张量语义：color_image/msi/video/audio", required=False),
            ToolParameter(name="valid_element_count", type="integer", description="有效元素数，排除音频 padding", required=False),
            ToolParameter(
                name="image_size",
                type="integer",
                description="最长边缩放尺寸",
                required=False,
                default=128,
            ),
            ToolParameter(
                name="mat_key",
                type="string",
                description="MAT 文件中的变量名；省略时自动选择",
                required=False,
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        try:
            run_id = _required_string(parameters, "run_id")
            image_path = _required_string(parameters, "image_path")
            run_dir = Path(_required_string(parameters, "run_dir"))
            mask_type = _required_string(parameters, "mask_type")
            if mask_type == "sildes":
                mask_type = "slices"
            if mask_type not in SUPPORTED_MASK_TYPES:
                raise ValueError("mask_type must be random or block")
            missing_rate = float(parameters["missing_rate"])
            if not 0.0 < missing_rate < 1.0:
                raise ValueError("missing_rate must be strictly between 0 and 1")
            seed = int(parameters["seed"])
            raw_size = parameters.get("image_size", 128)
            image_size = None if raw_size in (None, 0) else int(raw_size)
            if image_size is not None and image_size < 8:
                raise ValueError("image_size must be at least 8")

            mat_key = parameters.get("mat_key")
            if mat_key is not None and (not isinstance(mat_key, str) or not mat_key.strip()):
                raise ValueError("mat_key must be a non-empty string when provided")
            ground_truth, source_metadata = load_tensor_data(
                image_path,
                max_size=image_size,
                mat_key=mat_key,
                return_metadata=True,
            )
            height, width = ground_truth.shape[:2]
            if parameters.get("observation_mask_path"):
                observed_mask = load_observation_mask(parameters["observation_mask_path"])
                if observed_mask.shape not in (ground_truth.shape[:2], ground_truth.shape):
                    raise ValueError("supplied mask does not match loaded tensor; disable resizing")
            else:
                observed_mask = generate_observation_mask(
                    height=height, width=width, missing_rate=missing_rate,
                    mask_type=mask_type, seed=seed,
                )
            valid_count = parameters.get("valid_element_count")
            valid_count = observed_mask.size if valid_count is None else valid_count
            analysis_observed = training_observation_mask(observed_mask, ground_truth.shape)
            if active_audio_metadata():
                valid_count = int(audio_valid_mask(ground_truth.shape).sum())
                missing_count = valid_count - int(analysis_observed.sum())
            else:
                missing_count = int((~observed_mask).sum())
            if not 0 < valid_count <= analysis_observed.size or not 0 <= missing_count < valid_count:
                raise ValueError("invalid valid_element_count")
            if parameters.get("data_type"):
                if parameters["data_type"] not in {"color_image", "msi", "video", "audio"}:
                    raise ValueError("unsupported data_type")
                source_metadata["data_type"] = parameters["data_type"]
            corrupted = apply_observation_mask(ground_truth, observed_mask)
            run_dir.mkdir(parents=True, exist_ok=True)
            ground_truth_path = run_dir / "evaluation_ground_truth.npy"
            corrupted_path = run_dir / "corrupted.npy"
            corrupted_mat_path = run_dir / "corrupted.mat"
            corrupted_preview_path = run_dir / "corrupted_preview.png"
            mask_path = run_dir / ("mask.png" if observed_mask.ndim == 2 else "mask.npy")
            profile_path = run_dir / "image_profile.json"
            save_tensor_data(str(ground_truth_path), ground_truth)
            save_tensor_data(str(corrupted_path), corrupted)
            corrupted_mat = save_mat_companion(str(corrupted_mat_path), corrupted)
            save_image(str(corrupted_preview_path), corrupted)
            save_mask(str(mask_path), observed_mask)

            analysis_mask = analysis_observed
            analysis_tensor = ground_truth
            if analysis_observed.ndim > 2:
                means = observed_feature_mean(corrupted, analysis_observed)
                analysis_tensor = np.where(analysis_observed, corrupted, means)
                analysis_mask = analysis_observed.reshape(height, width, -1).any(axis=-1)
            visible_pixels = analysis_tensor.reshape(height, width, -1)[analysis_mask]
            component_count, largest_hole_ratio = _missing_component_statistics(
                analysis_mask, audio_valid_mask(ground_truth.shape).any(axis=-1) if active_audio_metadata() else None
            )
            structure_statistics = _visible_structure_statistics(
                analysis_tensor,
                analysis_mask,
            )
            profile = {
                "run_id": run_id,
                "image_shape": list(ground_truth.shape),
                "data_type": source_metadata["data_type"],
                "feature_shape": source_metadata["feature_shape"],
                "feature_count": source_metadata["feature_count"],
                "source_metadata": source_metadata,
                "mask_type": mask_type,
                "requested_missing_rate": missing_rate,
                "actual_missing_rate": float(missing_count / valid_count),
                "observed_pixels": int(valid_count - missing_count),
                "missing_pixels": missing_count,
                "missing_component_count": component_count,
                "largest_missing_component_image_ratio": largest_hole_ratio,
                "image_aspect_ratio": float(max(height, width) / min(height, width)),
                "visible_channel_mean": visible_pixels.mean(axis=0).tolist(),
                "visible_channel_std": visible_pixels.std(axis=0).tolist(),
                **structure_statistics,
                "analysis_scope": "visible_pixels_only",
                "mask_shape": list(observed_mask.shape),
                "mask_scope": "elementwise" if observed_mask.ndim > 2 else "shared_spatial",
                "tensor_axes": {
                    "color_image": ["height", "width", "channel"],
                    "msi": ["height", "width", "band"],
                    "video": ["height", "width", "time", "channel"],
                    "audio": ["time_frame", "sample_in_frame", "channel"],
                }[source_metadata["data_type"]],
                "valid_element_count": int(valid_count),
                "structure_statistics_imputation": observed_mask.ndim > 2,
            }
            _write_json(profile_path, profile)
            artifacts = {
                "corrupted": str(corrupted_path),
                "corrupted_preview": str(corrupted_preview_path),
                "mask": str(mask_path),
                "profile": str(profile_path),
            }
            if corrupted_mat is not None:
                artifacts["corrupted_mat"] = corrupted_mat
            return ToolResponse.success(
                text=(
                    "数据分析完成：type=%s，shape=%s，实际缺失率=%.4f。"
                    % (
                        profile["data_type"],
                        profile["image_shape"],
                        profile["actual_missing_rate"],
                    )
                ),
                data={
                    "run_id": run_id,
                    "profile": profile,
                    "artifacts": artifacts,
                },
            )
        except (KeyError, TypeError, ValueError) as error:
            return self.invalid(error)
        except Exception as error:
            return self.failed(error)


class RunInterpolationTool(ResearchTool):
    """Run modality-aware interpolation without accessing ground truth."""

    def __init__(self) -> None:
        super().__init__(
            name="run_interpolation",
            description="仅根据缺损张量、观测 mask 和数据类型运行空间/时空最近邻或音频线性插值。",
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="run_id", type="string", description="工作流运行 ID"),
            ToolParameter(name="corrupted_path", type="string", description="缺损张量路径"),
            ToolParameter(name="mask_path", type="string", description="观测 mask 路径"),
            ToolParameter(name="output_path", type="string", description="插值结果路径"),
            ToolParameter(name="data_type", type="string", description="显式语义：color_image/msi/video/audio", required=False),
            ToolParameter(
                name="preview_path",
                type="string",
                description="RGB 预览图路径",
                required=False,
            ),
            ToolParameter(
                name="mat_output_path",
                type="string",
                description="完整 MAT 插值结果路径",
                required=False,
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        try:
            run_id = _required_string(parameters, "run_id")
            corrupted, observed_mask = _load_observed_pair(
                _required_string(parameters, "corrupted_path"),
                _required_string(parameters, "mask_path"),
            )
            output_path = Path(_required_string(parameters, "output_path"))
            preview_path = Path(
                str(
                    parameters.get("preview_path")
                    or output_path.with_name(output_path.stem + "_preview.png")
                )
            )
            mat_output_path = Path(
                str(parameters.get("mat_output_path") or output_path.with_suffix(".mat"))
            )
            reconstruction = nearest_neighbor_fill(corrupted, observed_mask, parameters.get("data_type"))
            save_tensor_data(str(output_path), reconstruction)
            mat_output = save_mat_companion(str(mat_output_path), reconstruction)
            save_image(str(preview_path), reconstruction)
            artifacts = {
                "reconstruction": str(output_path),
                "preview": str(preview_path),
            }
            if mat_output is not None:
                artifacts["reconstruction_mat"] = mat_output
            return ToolResponse.success(
                text="波形线性插值完成。" if active_audio_metadata() else "最近邻插值完成。",
                data={
                    "run_id": run_id,
                    "algorithm": "linear_interpolation_waveform" if active_audio_metadata() else "nearest_neighbor_manhattan",
                    "artifacts": artifacts,
                },
            )
        except (KeyError, TypeError, ValueError) as error:
            return self.invalid(error)
        except Exception as error:
            return self.failed(error)


class TuneTensorModelTool(ResearchTool):
    """Select hyperparameters and checkpoints using missing-region GT."""

    def __init__(self) -> None:
        super().__init__(
            name="tune_tensor_model",
            description=(
                "使用全部观测像素训练，并用缺失区 Ground Truth 的 MSE "
                "选择候选结构、学习率和 checkpoint。"
            ),
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="run_id", type="string", description="工作流运行 ID"),
            ToolParameter(name="corrupted_path", type="string", description="缺损图片路径"),
            ToolParameter(name="mask_path", type="string", description="观测 mask 路径"),
            ToolParameter(name="ground_truth_path", type="string", description="用于每轮选择的完整真值"),
            ToolParameter(name="output_path", type="string", description="调参结果 JSON 路径"),
            ToolParameter(
                name="model_name",
                type="string",
                description="SUPPORTED_MODEL_NAMES 中的张量分解名称",
            ),
            ToolParameter(name="seed", type="integer", description="随机种子"),
            ToolParameter(name="candidates", type="array", description="候选配置列表", required=False),
            ToolParameter(name="max_steps", type="integer", description="每个 trial 最大步数", required=False, default=200),
            ToolParameter(name="refine_learning_rate", type="boolean", description="在粗搜胜出学习率附近进行二阶段精搜", required=False, default=False),
            ToolParameter(name="learning_rate_refinement_factor", type="number", description="学习率精搜的缩放倍数", required=False, default=3.0),
            ToolParameter(name="auto_expand_steps", type="boolean", description="最佳点靠近上限时自动扩大训练预算", required=False, default=False),
            ToolParameter(name="max_steps_ceiling", type="integer", description="自动扩展的硬上限", required=False),
            ToolParameter(name="near_limit_ratio", type="number", description="触发扩展的 best_step/max_steps 阈值", required=False, default=0.9),
            ToolParameter(name="expansion_factor", type="number", description="每次扩展的预算倍数", required=False, default=2.0),
            ToolParameter(name="device", type="string", description="auto、cpu 或 cuda", required=False, default="auto"),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        try:
            run_id = _required_string(parameters, "run_id")
            model_name = _required_string(parameters, "model_name")
            if model_name not in SUPPORTED_MODEL_NAMES:
                raise ValueError(
                    "model_name must be one of %s"
                    % ", ".join(sorted(SUPPORTED_MODEL_NAMES))
                )
            corrupted, observed_mask = _load_observed_pair(
                _required_string(parameters, "corrupted_path"),
                _required_string(parameters, "mask_path"),
            )
            ground_truth = load_tensor_data(
                _required_string(parameters, "ground_truth_path"), None
            )
            seed = int(parameters["seed"])
            candidates = parameters.get("candidates") or _default_candidates(
                model_name,
                tuple(int(value) for value in corrupted.shape),
            )
            if not isinstance(candidates, list) or not candidates:
                raise ValueError("candidates must be a non-empty list")

            initial_max_steps = int(parameters.get("max_steps", 200))
            refine_learning_rate = parameters.get("refine_learning_rate", False)
            if not isinstance(refine_learning_rate, bool):
                raise ValueError("refine_learning_rate must be a bool")
            learning_rate_refinement_factor = float(
                parameters.get("learning_rate_refinement_factor", 3.0)
            )
            auto_expand_steps = parameters.get("auto_expand_steps", False)
            if not isinstance(auto_expand_steps, bool):
                raise ValueError("auto_expand_steps must be a bool")
            max_steps_ceiling = int(
                parameters.get("max_steps_ceiling", initial_max_steps)
            )
            near_limit_ratio = float(parameters.get("near_limit_ratio", 0.9))
            expansion_factor = float(parameters.get("expansion_factor", 2.0))
            if initial_max_steps < 1:
                raise ValueError("max_steps must be positive")
            if learning_rate_refinement_factor <= 1.0:
                raise ValueError(
                    "learning_rate_refinement_factor must be greater than 1"
                )
            if max_steps_ceiling < initial_max_steps:
                raise ValueError("max_steps_ceiling must be at least max_steps")
            if not 0.0 < near_limit_ratio <= 1.0:
                raise ValueError("near_limit_ratio must be in (0, 1]")
            if expansion_factor <= 1.0:
                raise ValueError("expansion_factor must be greater than 1")

            print(
                "\n🔧 进入自动调参：%s，共 %d 个 trial，每个最多 %d 步"
                % (model_name, len(candidates), initial_max_steps),
                flush=True,
            )
            trials = []
            training_run_count = 0
            selected_output = None
            selected_score = float("inf")

            def execute_trial(
                hyperparameters: Dict[str, Any],
                learning_rate: float,
                stage: str,
                progress_label: str,
            ) -> Dict[str, Any]:
                nonlocal training_run_count, selected_output, selected_score
                trial_parameters = dict(parameters)
                trial_parameters["max_steps"] = initial_max_steps
                trial_config = _training_config(trial_parameters, learning_rate)
                output = train_tensor_model(
                    model_name=model_name,
                    model_hyperparameters=dict(hyperparameters),
                    observed_image=corrupted,
                    observed_mask=observed_mask,
                    config=trial_config,
                    seed=seed,
                    progress_label=progress_label,
                    ground_truth=ground_truth,
                )
                training_run_count += 1
                record = {
                    "trial_index": len(trials),
                    "search_stage": stage,
                    "model_name": model_name,
                    "hyperparameters": dict(hyperparameters),
                    "learning_rate": learning_rate,
                    "max_steps": initial_max_steps,
                    "best_step": output.best_step,
                    "best_validation_mse": output.best_validation_mse,
                    "best_missing_gt_mse": output.best_validation_mse,
                    "best_missing_psnr": output.best_missing_psnr,
                    "runtime_seconds": output.runtime_seconds,
                    "parameter_count": output.parameter_count,
                    "device": output.device,
                    "stopped_early": output.stopped_early,
                    "budget_history": [
                        {
                            "max_steps": initial_max_steps,
                            "best_step": output.best_step,
                            "best_validation_mse": output.best_validation_mse,
                            "runtime_seconds": output.runtime_seconds,
                            "stopped_early": output.stopped_early,
                        }
                    ],
                }
                if active_audio_metadata() is not None:
                    record.pop("best_missing_psnr")
                    record["best_missing_nmse"] = output.best_missing_nmse
                score = trial_selection_loss(record)
                if selected_output is None or score < selected_score:
                    selected_score = score
                    selected_output = output
                return record

            for trial_index, candidate in enumerate(candidates):
                if not isinstance(candidate, dict):
                    raise ValueError("each candidate must be an object")
                hyperparameters = candidate.get("hyperparameters")
                if hyperparameters is None:
                    hyperparameters = get_default_hyperparameters(model_name)
                if not isinstance(hyperparameters, dict):
                    raise ValueError("candidate hyperparameters must be an object")
                learning_rate = float(candidate.get("learning_rate", 0.03))
                trials.append(
                    execute_trial(
                        dict(hyperparameters),
                        learning_rate,
                        "coarse",
                        "方法选择自动调参 %s trial %d/%d"
                        % (model_name, trial_index + 1, len(candidates)),
                    )
                )

            best_trial = min(trials, key=trial_selection_loss)
            coarse_trial_count = len(trials)
            fine_trial_count = 0
            if refine_learning_rate:
                coarse_best = best_trial
                fine_rates = (
                    coarse_best["learning_rate"]
                    / learning_rate_refinement_factor,
                    coarse_best["learning_rate"]
                    * learning_rate_refinement_factor,
                )
                for learning_rate in fine_rates:
                    already_tested = any(
                        trial["hyperparameters"] == coarse_best["hyperparameters"]
                        and math.isclose(
                            trial["learning_rate"], learning_rate, rel_tol=1e-12
                        )
                        for trial in trials
                    )
                    if not 1e-5 <= learning_rate <= 1.0 or already_tested:
                        continue
                    trials.append(
                        execute_trial(
                            coarse_best["hyperparameters"],
                            learning_rate,
                            "fine",
                            "学习率精搜 %s trial %d"
                            % (model_name, len(trials) + 1),
                        )
                    )
                    fine_trial_count += 1
                best_trial = min(
                    trials, key=trial_selection_loss
                )
            expansion_history = []
            while (
                auto_expand_steps
                and not best_trial["stopped_early"]
                and best_trial["max_steps"] < max_steps_ceiling
                and best_trial["best_step"]
                >= near_limit_ratio * best_trial["max_steps"]
            ):
                previous_max_steps = int(best_trial["max_steps"])
                expanded_max_steps = min(
                    max_steps_ceiling,
                    max(
                        previous_max_steps + 1,
                        int(math.ceil(previous_max_steps * expansion_factor)),
                    ),
                )
                print(
                    "\n↗ 当前最佳 trial %d 的最佳点 %d/%d 靠近上限，扩展至 %d 步重跑…"
                    % (
                        best_trial["trial_index"],
                        best_trial["best_step"],
                        previous_max_steps,
                        expanded_max_steps,
                    ),
                    flush=True,
                )
                expanded_parameters = dict(parameters)
                expanded_parameters["max_steps"] = expanded_max_steps
                expanded_config = _training_config(
                    expanded_parameters, best_trial["learning_rate"]
                )
                output = train_tensor_model(
                    model_name=model_name,
                    model_hyperparameters=dict(best_trial["hyperparameters"]),
                    observed_image=corrupted,
                    observed_mask=observed_mask,
                    config=expanded_config,
                    seed=seed,
                    progress_label=(
                        "调参预算扩展 %s trial %d（%d 步）"
                        % (model_name, best_trial["trial_index"], expanded_max_steps)
                    ),
                    ground_truth=ground_truth,
                )
                training_run_count += 1
                expansion_history.append(
                    {
                        "trial_index": best_trial["trial_index"],
                        "from_max_steps": previous_max_steps,
                        "to_max_steps": expanded_max_steps,
                        "previous_best_step": best_trial["best_step"],
                        "new_best_step": output.best_step,
                        "new_best_validation_mse": output.best_validation_mse,
                        "accepted": (
                            output.best_validation_mse
                            <= best_trial["best_validation_mse"]
                        ),
                    }
                )
                if output.best_validation_mse > best_trial["best_validation_mse"]:
                    break
                best_trial["max_steps"] = expanded_max_steps
                best_trial["best_step"] = output.best_step
                best_trial["best_validation_mse"] = output.best_validation_mse
                best_trial["best_missing_gt_mse"] = output.best_validation_mse
                best_trial["best_missing_psnr"] = output.best_missing_psnr
                if active_audio_metadata() is not None:
                    best_trial.pop("best_missing_psnr")
                    best_trial["best_missing_nmse"] = output.best_missing_nmse
                best_trial["runtime_seconds"] += output.runtime_seconds
                best_trial["parameter_count"] = output.parameter_count
                best_trial["device"] = output.device
                best_trial["stopped_early"] = output.stopped_early
                best_trial["budget_history"].append(
                    {
                        "max_steps": expanded_max_steps,
                        "best_step": output.best_step,
                        "best_validation_mse": output.best_validation_mse,
                        "runtime_seconds": output.runtime_seconds,
                        "stopped_early": output.stopped_early,
                    }
                )
                selected_output = output
                selected_score = trial_selection_loss(best_trial)
                best_trial = min(
                    trials, key=trial_selection_loss
                )
            if selected_output is None:
                raise RuntimeError("tuning did not retain a selected training output")
            output_path = Path(_required_string(parameters, "output_path"))
            selected_result = _persist_selected_training_output(
                output_dir=output_path.parent / "selected_model",
                run_id=run_id,
                model_name=model_name,
                hyperparameters=best_trial["hyperparameters"],
                learning_rate=float(best_trial["learning_rate"]),
                observed=corrupted,
                observed_mask=observed_mask,
                output=selected_output,
            )
            best_trial["artifacts"] = selected_result["artifacts"]
            result = {
                "run_id": run_id,
                "selection_metric": "missing_original_waveform_nmse" if active_audio_metadata() is not None else "missing_region_ground_truth_mse",
                "selection_scope": "missing_region_ground_truth",
                "ground_truth_used": True,
                "training_pixels": "all_observed_pixels",
                "trial_count": len(trials),
                "coarse_trial_count": coarse_trial_count,
                "fine_trial_count": fine_trial_count,
                "training_run_count": training_run_count,
                "learning_rate_refinement_enabled": refine_learning_rate,
                "learning_rate_refinement_factor": (
                    learning_rate_refinement_factor
                ),
                "initial_max_steps": initial_max_steps,
                "max_steps_ceiling": max_steps_ceiling,
                "auto_expansion_enabled": auto_expand_steps,
                "near_limit_ratio": near_limit_ratio,
                "expansion_factor": expansion_factor,
                "expansion_history": expansion_history,
                "trials": trials,
                "best": best_trial,
                "selected_output": selected_result,
            }
            _write_json(output_path, result)
            return ToolResponse.success(
                text=(
                    "调参完成：选择 trial %d，缺失区 GT MSE=%.8f。"
                    % (best_trial["trial_index"], best_trial["best_validation_mse"])
                ),
                data={
                    "run_id": run_id,
                    "best": best_trial,
                    "selected_output": selected_result,
                    "trial_count": len(trials),
                    "artifacts": {
                        "tuning_result": str(output_path),
                        **selected_result["artifacts"],
                    },
                },
            )
        except (KeyError, TypeError, ValueError) as error:
            return self.invalid(error)
        except Exception as error:
            return self.failed(error)


class TrainTensorModelTool(ResearchTool):
    """Final-refit a selected model on every observed pixel."""

    def __init__(self) -> None:
        super().__init__(
            name="train_tensor_model",
            description=(
                "使用已选配置和步数，在全部观测像素上重新训练张量分解模型；"
                "不读取 Ground Truth。"
            ),
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="run_id", type="string", description="工作流运行 ID"),
            ToolParameter(name="corrupted_path", type="string", description="缺损图片路径"),
            ToolParameter(name="mask_path", type="string", description="观测 mask 路径"),
            ToolParameter(name="output_dir", type="string", description="模型产物目录"),
            ToolParameter(
                name="model_name",
                type="string",
                description="SUPPORTED_MODEL_NAMES 中的张量分解名称",
            ),
            ToolParameter(name="hyperparameters", type="object", description="模型超参数"),
            ToolParameter(name="learning_rate", type="number", description="学习率"),
            ToolParameter(name="selected_steps", type="integer", description="调参阶段选出的步数"),
            ToolParameter(name="seed", type="integer", description="随机种子"),
            ToolParameter(name="device", type="string", description="auto、cpu 或 cuda", required=False, default="auto"),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        try:
            run_id = _required_string(parameters, "run_id")
            model_name = _required_string(parameters, "model_name")
            if model_name not in SUPPORTED_MODEL_NAMES:
                raise ValueError(
                    "model_name must be one of %s"
                    % ", ".join(sorted(SUPPORTED_MODEL_NAMES))
                )
            hyperparameters = parameters.get("hyperparameters")
            if not isinstance(hyperparameters, dict):
                raise ValueError("hyperparameters must be an object")
            selected_steps = int(parameters["selected_steps"])
            learning_rate = float(parameters["learning_rate"])
            seed = int(parameters["seed"])
            corrupted, observed_mask = _load_observed_pair(
                _required_string(parameters, "corrupted_path"),
                _required_string(parameters, "mask_path"),
            )
            config_parameters = dict(parameters)
            config_parameters["max_steps"] = max(1, selected_steps)
            config = _training_config(config_parameters, learning_rate)
            print(
                "\n🎯 调参完成，使用最佳配置在全部观测像素上重训：%s，%d 步"
                % (model_name, selected_steps),
                flush=True,
            )
            output = fit_tensor_model_on_all_observations(
                model_name=model_name,
                model_hyperparameters=dict(hyperparameters),
                observed_image=corrupted,
                observed_mask=observed_mask,
                config=config,
                selected_steps=selected_steps,
                seed=seed,
                progress_label="方法选择最终重训 %s" % model_name,
            )

            output_dir = Path(_required_string(parameters, "output_dir"))
            output_dir.mkdir(parents=True, exist_ok=True)
            raw_path = output_dir / "model_raw.npy"
            completed_path = output_dir / "model_completed.npy"
            raw_mat_path = output_dir / "model_raw.mat"
            completed_mat_path = output_dir / "model_completed.mat"
            raw_preview_path = output_dir / "model_raw_preview.png"
            completed_preview_path = output_dir / "model_completed_preview.png"
            history_path = output_dir / "final_fit_history.json"
            checkpoint_path = output_dir / "model.pt"
            result_path = output_dir / "training_result.json"
            completed = output.reconstruction.copy()
            completed[observed_mask] = corrupted[observed_mask]
            save_tensor_data(str(raw_path), output.reconstruction)
            save_tensor_data(str(completed_path), completed)
            raw_mat = save_mat_companion(str(raw_mat_path), output.reconstruction)
            completed_mat = save_mat_companion(str(completed_mat_path), completed)
            save_image(str(raw_preview_path), output.reconstruction)
            save_image(str(completed_preview_path), completed)
            _write_json(history_path, {"history": output.history})
            torch.save(
                {
                    "model_name": model_name,
                    "hyperparameters": dict(hyperparameters),
                    "selected_steps": selected_steps,
                    "learning_rate": learning_rate,
                    "state_dict": output.state_dict,
                    "training_phase": "refit_on_all_observations",
                },
                checkpoint_path,
            )
            result = {
                "run_id": run_id,
                "model_name": model_name,
                "hyperparameters": dict(hyperparameters),
                "learning_rate": learning_rate,
                "fitted_steps": output.fitted_steps,
                "final_train_mse": output.final_train_mse,
                "observed_pixels_used": int(output.fit_mask.sum()),
                "parameter_count": output.parameter_count,
                "runtime_seconds": output.runtime_seconds,
                "device": output.device,
                "ground_truth_used": False,
            }
            _write_json(result_path, result)
            artifacts = {
                "raw_reconstruction": str(raw_path),
                "reconstruction": str(completed_path),
                "raw_preview": str(raw_preview_path),
                "preview": str(completed_preview_path),
                "history": str(history_path),
                "checkpoint": str(checkpoint_path),
                "training_result": str(result_path),
            }
            if raw_mat is not None:
                artifacts["raw_reconstruction_mat"] = raw_mat
            if completed_mat is not None:
                artifacts["reconstruction_mat"] = completed_mat
            return ToolResponse.success(
                text=(
                    "最终拟合完成：使用 %d 个观测像素训练 %d 步。"
                    % (result["observed_pixels_used"], output.fitted_steps)
                ),
                data={
                    **result,
                    "artifacts": artifacts,
                },
            )
        except (KeyError, TypeError, ValueError) as error:
            return self.invalid(error)
        except Exception as error:
            return self.failed(error)


class EvaluateReconstructionTool(ResearchTool):
    """Evaluate one finished reconstruction with the full-reference tensor."""

    def __init__(self) -> None:
        super().__init__(
            name="evaluate_reconstruction",
            description=(
                "训练和选择结束后，计算 PSNR、SSIM、LPIPS、MANIQA、"
                "CLIP-IQA 与 MUSIQ。"
            ),
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="run_id", type="string", description="工作流运行 ID"),
            ToolParameter(name="algorithm_name", type="string", description="算法名称"),
            ToolParameter(name="reconstruction_path", type="string", description="重建张量路径"),
            ToolParameter(name="ground_truth_path", type="string", description="仅用于最终评估的完整张量"),
            ToolParameter(name="mask_path", type="string", description="观测 mask 路径"),
            ToolParameter(name="output_path", type="string", description="指标 JSON 路径"),
            ToolParameter(
                name="device",
                type="string",
                description="学习式 IQA 指标使用的 auto、cpu 或 cuda",
                required=False,
                default="auto",
            ),
            ToolParameter(
                name="full_reference_metrics",
                type="boolean",
                description="是否计算全参考 LPIPS；音频只使用 NMSE，其他类型计算 PSNR/SSIM",
                required=False,
                default=False,
            ),
            ToolParameter(
                name="no_reference_metrics",
                type="boolean",
                description="是否计算 MANIQA/CLIP-IQA/MUSIQ",
                required=False,
                default=False,
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        try:
            run_id = _required_string(parameters, "run_id")
            algorithm_name = _required_string(parameters, "algorithm_name")
            reconstruction = load_tensor_prediction(
                _required_string(parameters, "reconstruction_path"),
            )
            ground_truth = load_tensor_data(
                _required_string(parameters, "ground_truth_path"),
                max_size=None,
            )
            observed_mask = load_observation_mask(
                _required_string(parameters, "mask_path")
            )
            if reconstruction.shape != ground_truth.shape:
                raise ValueError("reconstruction and ground truth shapes do not match")
            if observed_mask.shape not in (ground_truth.shape[:2], ground_truth.shape):
                raise ValueError("mask and tensor spatial shapes do not match")
            metrics = _metric_payload(
                reconstruction,
                ground_truth,
                observed_mask,
                device=str(parameters.get("device", "auto")),
                include_full_reference_metrics=bool(
                    parameters.get("full_reference_metrics", False)
                ),
                include_no_reference_metrics=bool(
                    parameters.get("no_reference_metrics", False)
                ),
            )
            result = {
                "run_id": run_id,
                "algorithm_name": algorithm_name,
                **metrics,
                "ground_truth_usage": "final_evaluation_only",
            }
            output_path = Path(_required_string(parameters, "output_path"))
            _write_json(output_path, result)
            evaluation_text = (
                "%s 评估完成：Missing NMSE=%s。" % (algorithm_name, metrics["missing_nmse"])
                if "missing_nmse" in metrics else
                "%s 评估完成：PSNR=%s dB。" % (algorithm_name, "infinite" if metrics["full_psnr"] is None else "%.4f" % metrics["full_psnr"])
            )
            return ToolResponse.success(
                text=evaluation_text,
                data={
                    **result,
                    "artifacts": {"metrics": str(output_path)},
                },
            )
        except (KeyError, TypeError, ValueError) as error:
            return self.invalid(error)
        except Exception as error:
            return self.failed(error)


class CompareExperimentsTool(ResearchTool):
    """Compare saved metrics without rerunning either algorithm."""

    def __init__(self) -> None:
        super().__init__(
            name="compare_experiments",
            description="读取指标 JSON；音频仅按缺失波形 NMSE 选优，其他类型按缺失 PSNR、SSIM 比较。",
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="run_id", type="string", description="工作流运行 ID"),
            ToolParameter(name="baseline_metrics_path", type="string", description="插值指标 JSON"),
            ToolParameter(name="candidate_metrics_path", type="string", description="张量模型指标 JSON"),
            ToolParameter(name="output_path", type="string", description="比较结果 JSON"),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        try:
            run_id = _required_string(parameters, "run_id")
            baseline_path = Path(_required_string(parameters, "baseline_metrics_path"))
            candidate_path = Path(_required_string(parameters, "candidate_metrics_path"))
            if not baseline_path.is_file() or not candidate_path.is_file():
                raise ValueError("both metric files must exist")
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))

            if "missing_nmse" in baseline or "missing_nmse" in candidate:
                if "missing_nmse" not in baseline or "missing_nmse" not in candidate:
                    raise ValueError("cannot mix audio NMSE and image metrics")
                baseline_score, candidate_score = metric_score(baseline), metric_score(candidate)
                winner = "candidate" if candidate_score > baseline_score else "baseline" if candidate_score < baseline_score else "tie"
                result = {"run_id": run_id, "winner": winner, "primary_metric": "missing_nmse",
                          "baseline_algorithm": baseline.get("algorithm_name"),
                          "candidate_algorithm": candidate.get("algorithm_name"),
                          "candidate_nmse_reduction": (candidate_score - baseline_score) if math.isfinite(candidate_score) and math.isfinite(baseline_score) else None,
                          "promotion_allowed": winner == "candidate"}
                output_path = Path(_required_string(parameters, "output_path"))
                _write_json(output_path, result)
                return ToolResponse.success(text="NMSE 比较完成：winner=%s。" % winner,
                                            data={**result, "artifacts": {"comparison": str(output_path)}})

            baseline_psnr = baseline.get("missing_psnr")
            candidate_psnr = candidate.get("missing_psnr")
            baseline_score = math.inf if baseline_psnr is None else float(baseline_psnr)
            candidate_score = math.inf if candidate_psnr is None else float(candidate_psnr)
            psnr_delta = candidate_score - baseline_score
            if math.isinf(candidate_score) and math.isinf(baseline_score):
                psnr_delta_json = None
                ssim_delta = float(candidate["composite_ssim"]) - float(
                    baseline["composite_ssim"]
                )
                winner = "candidate" if ssim_delta > 0 else "baseline" if ssim_delta < 0 else "tie"
            elif math.isinf(candidate_score):
                psnr_delta_json = None
                ssim_delta = float(candidate["composite_ssim"]) - float(
                    baseline["composite_ssim"]
                )
                winner = "candidate"
            elif math.isinf(baseline_score):
                psnr_delta_json = None
                ssim_delta = float(candidate["composite_ssim"]) - float(
                    baseline["composite_ssim"]
                )
                winner = "baseline"
            else:
                psnr_delta_json = psnr_delta
                ssim_delta = float(candidate["composite_ssim"]) - float(
                    baseline["composite_ssim"]
                )
                winner = "candidate" if psnr_delta > 0 else "baseline" if psnr_delta < 0 else "tie"
            result = {
                "run_id": run_id,
                "winner": winner,
                "primary_metric": "missing_psnr",
                "baseline_algorithm": baseline.get("algorithm_name"),
                "candidate_algorithm": candidate.get("algorithm_name"),
                "candidate_psnr_delta": psnr_delta_json,
                "candidate_ssim_delta": ssim_delta,
                "promotion_allowed": winner == "candidate",
            }
            output_path = Path(_required_string(parameters, "output_path"))
            _write_json(output_path, result)
            return ToolResponse.success(
                text="实验比较完成：winner=%s。" % winner,
                data={
                    **result,
                    "artifacts": {"comparison": str(output_path)},
                },
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            return self.invalid(error)
        except Exception as error:
            return self.failed(error)
