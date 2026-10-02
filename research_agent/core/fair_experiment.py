"""Paired per-image oracle tuning for baseline and candidate models."""

from __future__ import annotations

import itertools
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..schemas import TrainingConfig
from .data import load_tensor_data, load_tensor_prediction, save_image, save_mat_companion, save_tensor_data
from .metrics import evaluate_reconstruction_metrics
from .audio_metrics import active_audio_metadata, trial_selection_loss
from .trainer import ModelBuilder, train_tensor_model


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _persist_selected_trial(
    output_dir: str,
    model_name: str,
    trial: Dict[str, Any],
    output: Any,
) -> Dict[str, str]:
    """Save the winner produced during tuning so evaluation need not retrain it."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    raw_path = directory / "selected_raw.npy"
    checkpoint_path = directory / "selected_model.pt"
    history_path = directory / "selection_history.json"
    save_tensor_data(str(raw_path), output.reconstruction)
    torch.save(
        {
            "model_name": model_name,
            "hyperparameters": trial["hyperparameters"],
            "selected_steps": output.best_step,
            "learning_rate": trial["learning_rate"],
            "state_dict": output.state_dict,
            "training_phase": "all_observed_training_gt_checkpoint_selection",
        },
        checkpoint_path,
    )
    _write_json(history_path, {"history": output.history})
    return {
        "selected_raw_reconstruction": str(raw_path),
        "selected_checkpoint": str(checkpoint_path),
        "selected_history": str(history_path),
    }


def _validate_search_space(search_space: Dict[str, Any]) -> None:
    if not isinstance(search_space, dict) or not search_space:
        raise ValueError("search_space must be a non-empty dict")
    for name, values in search_space.items():
        if not isinstance(name, str) or not isinstance(values, list) or not values:
            raise ValueError("each search-space entry must be a non-empty list")


def _grid(search_space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    _validate_search_space(search_space)
    names = sorted(search_space)
    return [
        dict(zip(names, values))
        for values in itertools.product(*(search_space[name] for name in names))
    ]


def _select_unique_configurations(
    grid: List[Dict[str, Any]],
    count: int,
    seed: int,
    anchor: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Select distinct configurations deterministically, including a valid anchor."""

    selected: List[Dict[str, Any]] = []
    if anchor is not None and anchor in grid:
        selected.append(anchor)
    remaining = [config for config in grid if config not in selected]
    order = np.random.default_rng(seed).permutation(len(remaining)).tolist()
    selected.extend(remaining[index] for index in order)
    return selected[:count]


def independent_trial_configurations(
    base_search_space: Dict[str, List[Any]],
    candidate_search_space: Dict[str, List[Any]],
    trial_count: int,
    seed: int,
    anchor_base_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Tune each model independently up to the same per-model structure limit.

    A smaller search space is exhausted without reducing the other model's search.
    This compares each algorithm after reasonable model-specific tuning rather than
    forcing equal numbers of distinct configurations.
    """

    if trial_count < 1 or trial_count > 5:
        raise ValueError("trial_count must be in [1, 5]")
    base_grid = _grid(base_search_space)
    candidate_grid = _grid(candidate_search_space)
    baseline_trial_count = min(trial_count, len(base_grid))
    candidate_trial_count = min(trial_count, len(candidate_grid))
    anchor = None
    if anchor_base_config is not None:
        anchor = {
            key: anchor_base_config[key]
            for key in base_search_space
            if key in anchor_base_config
        }
        anchor_is_valid = len(anchor) == len(base_search_space) and all(
            anchor[key] in base_search_space[key] for key in base_search_space
        )
        if not anchor_is_valid:
            anchor = None
    selected_base = _select_unique_configurations(
        base_grid,
        baseline_trial_count,
        seed,
        anchor,
    )
    candidate_configs = _select_unique_configurations(
        candidate_grid,
        candidate_trial_count,
        seed,
    )

    shared_space = {
        key: values
        for key, values in candidate_search_space.items()
        if key in base_search_space and values == base_search_space[key]
    }
    candidate_specific_space = {
        key: values
        for key, values in candidate_search_space.items()
        if key not in shared_space
    }
    return {
        "tuning_policy": "independent_model_specific_search_up_to_shared_limit",
        "requested_trial_limit_per_model": trial_count,
        "baseline_effective_trial_count": baseline_trial_count,
        "candidate_effective_trial_count": candidate_trial_count,
        "baseline_trial_count_reduced": baseline_trial_count < trial_count,
        "candidate_trial_count_reduced": candidate_trial_count < trial_count,
        "base_available_config_count": len(base_grid),
        "candidate_available_config_count": len(candidate_grid),
        "baseline_search_coverage": baseline_trial_count / len(base_grid),
        "candidate_search_coverage": candidate_trial_count / len(candidate_grid),
        "baseline": selected_base,
        "candidate": candidate_configs,
        "shared_parameter_names": sorted(shared_space),
        "candidate_only_parameter_names": sorted(
            key for key in candidate_specific_space if key not in base_search_space
        ),
        "independently_tuned_parameter_names": sorted(candidate_specific_space),
        "same_name_different_range_parameters": sorted(
            key
            for key in candidate_specific_space
            if key in base_search_space
        ),
    }


def paired_trial_configurations(
    base_search_space: Dict[str, List[Any]],
    candidate_search_space: Dict[str, List[Any]],
    trial_count: int,
    seed: int,
    anchor_base_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Backward-compatible alias for independent model-specific tuning."""

    return independent_trial_configurations(
        base_search_space=base_search_space,
        candidate_search_space=candidate_search_space,
        trial_count=trial_count,
        seed=seed,
        anchor_base_config=anchor_base_config,
    )


def tune_model_on_observed_pixels(
    model_name: str,
    model_builder: Optional[ModelBuilder],
    configurations: List[Dict[str, Any]],
    observed_image: np.ndarray,
    observed_mask: np.ndarray,
    training_config: TrainingConfig,
    seed: int,
    ground_truth: np.ndarray,
    selected_output_dir: Optional[str] = None,
    learning_rate_candidates: Optional[List[float]] = None,
    refine_learning_rate: bool = False,
    learning_rate_refinement_factor: float = 3.0,
) -> Dict[str, Any]:
    """Tune architecture, learning rate, and checkpoint on missing-region GT."""

    if not configurations:
        raise ValueError("at least one configuration is required")
    rates = list(learning_rate_candidates or [training_config.learning_rate])
    if any(
        isinstance(rate, bool)
        or not isinstance(rate, (int, float))
        or not 1e-5 <= float(rate) <= 1.0
        for rate in rates
    ):
        raise ValueError("learning_rate_candidates must be numbers in [1e-5, 1.0]")
    rates = list(dict.fromkeys(float(rate) for rate in rates))
    if learning_rate_refinement_factor <= 1.0:
        raise ValueError("learning_rate_refinement_factor must be greater than 1")
    coarse_trial_count = len(configurations) * len(rates)
    print(
        "\n🔍 开始 GT 引导的联合调参：%s，%d 组结构 × %d 个学习率"
        % (model_name, len(configurations), len(rates)),
        flush=True,
    )
    trials = []
    selected_output = None
    selected_score = float("inf")
    coarse_pairs = [
        (hyperparameters, learning_rate)
        for hyperparameters in configurations
        for learning_rate in rates
    ]

    def execute_trial(
        hyperparameters: Dict[str, Any],
        learning_rate: float,
        search_stage: str,
    ) -> Dict[str, Any]:
        nonlocal selected_output, selected_score
        trial_index = len(trials)
        trial_config = replace(training_config, learning_rate=learning_rate)
        output = train_tensor_model(
            model_name=model_name,
            model_hyperparameters=hyperparameters,
            observed_image=observed_image,
            observed_mask=observed_mask,
            config=trial_config,
            seed=seed,
            model_builder=model_builder,
            ground_truth=ground_truth,
            progress_label=(
                "公平实验 %s %s trial %d%s"
                % (
                    model_name,
                    "粗搜" if search_stage == "coarse" else "学习率精搜",
                    trial_index + 1,
                    "/%d" % coarse_trial_count
                    if search_stage == "coarse"
                    else "",
                )
            ),
        )
        record = {
            "trial_index": trial_index,
            "search_stage": search_stage,
            "hyperparameters": hyperparameters,
            "learning_rate": learning_rate,
            "best_step": output.best_step,
            "best_validation_mse": output.best_validation_mse,
            "best_missing_gt_mse": output.best_validation_mse,
            "best_missing_psnr": output.best_missing_psnr,
            "selection_metric": output.selection_metric,
            "ground_truth_used_for_selection": output.ground_truth_used_for_selection,
            "runtime_seconds": output.runtime_seconds,
            "parameter_count": output.parameter_count,
            "stopped_early": output.stopped_early,
            "history": output.history,
        }
        if active_audio_metadata() is not None:
            record.pop("best_missing_psnr")
            record["best_missing_nmse"] = output.best_missing_nmse
        score = trial_selection_loss(record)
        if selected_output is None or score < selected_score:
            selected_score = score
            selected_output = output
        return record

    for hyperparameters, learning_rate in coarse_pairs:
        trials.append(
            execute_trial(hyperparameters, learning_rate, "coarse")
        )

    best = min(trials, key=trial_selection_loss)
    fine_trial_count = 0
    if refine_learning_rate:
        fine_rates = (
            best["learning_rate"] / learning_rate_refinement_factor,
            best["learning_rate"] * learning_rate_refinement_factor,
        )
        if any(not 1e-5 <= rate <= 1.0 for rate in fine_rates):
            raise ValueError(
                "learning-rate refinement around the coarse winner leaves "
                "the supported [1e-5, 1.0] range"
            )
        for learning_rate in fine_rates:
            trials.append(
                execute_trial(best["hyperparameters"], learning_rate, "fine")
            )
            fine_trial_count += 1
        best = min(trials, key=trial_selection_loss)
    if selected_output_dir is not None:
        if selected_output is None:
            raise RuntimeError("tuning did not retain a selected training output")
        best["artifacts"] = _persist_selected_trial(
            selected_output_dir,
            model_name,
            best,
            selected_output,
        )
    return {
        "model_name": model_name,
        "selection_metric": "missing_original_waveform_nmse" if active_audio_metadata() is not None else "missing_region_ground_truth_mse",
        "selection_scope": "missing_region_ground_truth",
        "ground_truth_used": True,
        "training_pixels": "all_observed_pixels",
        "trial_count": len(trials),
        "hyperparameter_configuration_count": len(configurations),
        "coarse_trial_count": coarse_trial_count,
        "fine_trial_count": fine_trial_count,
        "learning_rate_candidates": rates,
        "trials": trials,
        "best": best,
    }


def final_fit_and_evaluate(
    model_name: str,
    model_builder: Optional[ModelBuilder],
    selected_trial: Dict[str, Any],
    observed_image: np.ndarray,
    observed_mask: np.ndarray,
    ground_truth: np.ndarray,
    training_config: TrainingConfig,
    seed: int,
    output_dir: str,
    include_full_reference_metrics: bool = False,
    include_no_reference_metrics: bool = False,
) -> Dict[str, Any]:
    """Evaluate the trained winner directly, fitting only when no artifact exists."""

    selected_learning_rate = float(
        selected_trial.get("learning_rate", training_config.learning_rate)
    )
    selected_artifacts = selected_trial.get("artifacts") or {}
    selected_raw_path = selected_artifacts.get("selected_raw_reconstruction")
    selected_checkpoint_path = selected_artifacts.get("selected_checkpoint")
    reuse_selected = bool(
        selected_raw_path
        and selected_checkpoint_path
        and Path(selected_raw_path).is_file()
        and Path(selected_checkpoint_path).is_file()
    )
    if reuse_selected:
        print(
            "\n🎯 直接使用 GT 评分最优 checkpoint：%s，%d 步"
            % (model_name, int(selected_trial["best_step"])),
            flush=True,
        )
        raw_reconstruction = load_tensor_prediction(selected_raw_path)
        history = selected_trial["history"]
        best_step = int(selected_trial["best_step"])
        best_mse = float(selected_trial["best_validation_mse"])
        best_psnr = selected_trial.get("best_missing_psnr")
        runtime_seconds = 0.0
        parameter_count = int(selected_trial["parameter_count"])
    else:
        print(
            "\n🎯 未找到可复用 checkpoint，执行 GT 引导拟合：%s"
            % model_name,
            flush=True,
        )
        final_training_config = replace(
            training_config, learning_rate=selected_learning_rate
        )
        output = train_tensor_model(
            model_name=model_name,
            model_hyperparameters=selected_trial["hyperparameters"],
            observed_image=observed_image,
            observed_mask=observed_mask,
            config=final_training_config,
            seed=seed,
            model_builder=model_builder,
            progress_label="GT 引导拟合 %s" % model_name,
            ground_truth=ground_truth,
        )
        raw_reconstruction = output.reconstruction
        history = output.history
        best_step = output.best_step
        best_mse = output.best_validation_mse
        best_psnr = output.best_missing_psnr
        runtime_seconds = output.runtime_seconds
        parameter_count = output.parameter_count
    completed = raw_reconstruction.copy()
    completed[observed_mask] = observed_image[observed_mask]
    metrics = evaluate_reconstruction_metrics(
        completed,
        ground_truth,
        observed_mask,
        device=training_config.device,
        include_full_reference_metrics=include_full_reference_metrics,
        include_no_reference_metrics=include_no_reference_metrics,
    )
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    raw_path = directory / "model_raw.npy"
    completed_path = directory / "model_completed.npy"
    raw_mat_path = directory / "model_raw.mat"
    completed_mat_path = directory / "model_completed.mat"
    raw_preview_path = directory / "model_raw_preview.png"
    completed_preview_path = directory / "model_completed_preview.png"
    checkpoint_path = directory / "model.pt"
    history_path = directory / "final_fit_history.json"
    metrics_path = directory / "metrics.json"
    save_tensor_data(str(raw_path), raw_reconstruction)
    save_tensor_data(str(completed_path), completed)
    raw_mat = save_mat_companion(str(raw_mat_path), raw_reconstruction)
    completed_mat = save_mat_companion(str(completed_mat_path), completed)
    save_image(str(raw_preview_path), raw_reconstruction)
    save_image(str(completed_preview_path), completed)
    if reuse_selected:
        shutil.copy2(selected_checkpoint_path, checkpoint_path)
    else:
        torch.save(
            {
                "model_name": model_name,
                "hyperparameters": selected_trial["hyperparameters"],
                "selected_steps": best_step,
                "learning_rate": selected_learning_rate,
                "state_dict": output.state_dict,
                "training_phase": "all_observed_training_gt_checkpoint_selection",
            },
            checkpoint_path,
        )
    _write_json(
        history_path,
        {"history": history, "reused_without_retraining": reuse_selected},
    )
    artifacts = {
        "raw_reconstruction": str(raw_path),
        "reconstruction": str(completed_path),
        "raw_preview": str(raw_preview_path),
        "preview": str(completed_preview_path),
        "checkpoint": str(checkpoint_path),
        "history": str(history_path),
        "metrics": str(metrics_path),
    }
    if raw_mat is not None:
        artifacts["raw_reconstruction_mat"] = raw_mat
    if completed_mat is not None:
        artifacts["reconstruction_mat"] = completed_mat
    result = {
        "model_name": model_name,
        "hyperparameters": selected_trial["hyperparameters"],
        "selected_steps": best_step,
        "learning_rate": selected_learning_rate,
        "selection_validation_mse": best_mse,
        "selection_missing_gt_mse": best_mse,
        "selection_missing_psnr": best_psnr,
        "selection_metric": "missing_region_ground_truth_mse",
        "ground_truth_used_for_selection": True,
        "final_train_mse": next(
            item["data_train_loss"] for item in history
            if item["step"] == best_step
        ),
        "final_total_loss": next(
            item["total_train_loss"] for item in history
            if item["step"] == best_step
        ),
        "metrics": metrics,
        "runtime_seconds": runtime_seconds,
        "selection_runtime_seconds": selected_trial["runtime_seconds"],
        "total_selected_runtime_seconds": (
            runtime_seconds + selected_trial["runtime_seconds"]
        ),
        "parameter_count": parameter_count,
        "observed_pixels_used": int(observed_mask.sum()),
        "reused_without_retraining": reuse_selected,
        "artifacts": artifacts,
    }
    if "missing_nmse" in metrics:
        result.pop("selection_missing_psnr")
        result["selection_missing_nmse"] = selected_trial.get("best_missing_nmse", metrics["missing_nmse"])
        result["selection_metric"] = "missing_original_waveform_nmse"
    _write_json(metrics_path, result)
    return result
