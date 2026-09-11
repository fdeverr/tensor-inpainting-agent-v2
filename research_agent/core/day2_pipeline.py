"""Day 2 tensor-model experiment pipeline."""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch

from ..schemas import Day2ExperimentConfig
from .data import (
    apply_observation_mask,
    load_tensor_data,
    save_image,
    save_mat_companion,
    save_mask,
    save_tensor_data,
)
from .interpolation import nearest_neighbor_fill
from .masks import generate_observation_mask
from .metrics import evaluate_reconstruction_metrics
from .trainer import fit_tensor_model_on_all_observations, train_tensor_model


def _make_run_id(model_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return "day2-%s-%s-%s" % (model_name, timestamp, uuid.uuid4().hex[:6])


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2, allow_nan=False)
        output_file.write("\n")


def _evaluate(
    reconstruction,
    ground_truth,
    observed_mask,
    device: str,
    include_learned_metrics: bool,
) -> Dict[str, Any]:
    return evaluate_reconstruction_metrics(
        reconstruction,
        ground_truth,
        observed_mask,
        device=device,
        include_learned_metrics=include_learned_metrics,
    )


def run_day2_experiment(config: Day2ExperimentConfig) -> Dict[str, Any]:
    """Compare one trainable tensor model with the Day 1 interpolation baseline."""

    config.validate()
    started_at = time.perf_counter()
    run_id = _make_run_id(config.model_name)
    run_dir = Path(config.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    ground_truth = load_tensor_data(
        config.image_path,
        max_size=config.image_size,
        mat_key=config.mat_key,
    )
    height, width = ground_truth.shape[:2]
    observed_mask = generate_observation_mask(
        height=height,
        width=width,
        missing_rate=config.missing_rate,
        mask_type=config.mask_type,
        seed=config.seed,
    )
    corrupted = apply_observation_mask(
        ground_truth,
        observed_mask,
        missing_fill_value=config.missing_fill_value,
    )

    interpolation = nearest_neighbor_fill(corrupted, observed_mask)
    selection_output = train_tensor_model(
        model_name=config.model_name,
        model_hyperparameters=config.model_hyperparameters,
        observed_image=corrupted,
        observed_mask=observed_mask,
        config=config.training,
        seed=config.seed,
    )
    final_fit_output = fit_tensor_model_on_all_observations(
        model_name=config.model_name,
        model_hyperparameters=config.model_hyperparameters,
        observed_image=corrupted,
        observed_mask=observed_mask,
        config=config.training,
        selected_steps=selection_output.best_step,
        seed=config.seed,
    )

    selection_reconstruction = selection_output.reconstruction.copy()
    selection_reconstruction[observed_mask] = corrupted[observed_mask]
    raw_reconstruction = final_fit_output.reconstruction
    completed_reconstruction = raw_reconstruction.copy()
    completed_reconstruction[observed_mask] = corrupted[observed_mask]

    artifact_paths = {
        "config": str(run_dir / "config.json"),
        "original": str(run_dir / "original.png"),
        "corrupted": str(run_dir / "corrupted.png"),
        "mask": str(run_dir / "mask.png"),
        "selection_train_mask": str(run_dir / "selection_train_mask.png"),
        "selection_validation_mask": str(run_dir / "selection_validation_mask.png"),
        "interpolated": str(run_dir / "interpolated.png"),
        "selection_completed": str(run_dir / "selection_completed.png"),
        "model_raw": str(run_dir / "model_raw.png"),
        "model_completed": str(run_dir / "model_completed.png"),
        "original_data": str(run_dir / "original.npy"),
        "corrupted_data": str(run_dir / "corrupted.npy"),
        "interpolated_data": str(run_dir / "interpolated.npy"),
        "selection_completed_data": str(run_dir / "selection_completed.npy"),
        "model_raw_data": str(run_dir / "model_raw.npy"),
        "model_completed_data": str(run_dir / "model_completed.npy"),
        "original_mat": str(run_dir / "original.mat"),
        "corrupted_mat": str(run_dir / "corrupted.mat"),
        "interpolated_mat": str(run_dir / "interpolated.mat"),
        "selection_completed_mat": str(run_dir / "selection_completed.mat"),
        "model_raw_mat": str(run_dir / "model_raw.mat"),
        "model_completed_mat": str(run_dir / "model_completed.mat"),
        "selection_history": str(run_dir / "selection_history.json"),
        "final_fit_history": str(run_dir / "final_fit_history.json"),
        "checkpoint": str(run_dir / "best_model.pt"),
        "metrics": str(run_dir / "metrics.json"),
    }
    save_image(artifact_paths["original"], ground_truth)
    save_image(artifact_paths["corrupted"], corrupted)
    save_mask(artifact_paths["mask"], observed_mask)
    save_mask(artifact_paths["selection_train_mask"], selection_output.train_mask)
    save_mask(
        artifact_paths["selection_validation_mask"],
        selection_output.validation_mask,
    )
    save_image(artifact_paths["interpolated"], interpolation)
    save_image(artifact_paths["selection_completed"], selection_reconstruction)
    save_image(artifact_paths["model_raw"], raw_reconstruction)
    save_image(artifact_paths["model_completed"], completed_reconstruction)
    save_tensor_data(artifact_paths["original_data"], ground_truth)
    save_tensor_data(artifact_paths["corrupted_data"], corrupted)
    save_tensor_data(artifact_paths["interpolated_data"], interpolation)
    save_tensor_data(
        artifact_paths["selection_completed_data"], selection_reconstruction
    )
    save_tensor_data(artifact_paths["model_raw_data"], raw_reconstruction)
    save_tensor_data(
        artifact_paths["model_completed_data"], completed_reconstruction
    )
    for artifact_name, data in (
        ("original_mat", ground_truth),
        ("corrupted_mat", corrupted),
        ("interpolated_mat", interpolation),
        ("selection_completed_mat", selection_reconstruction),
        ("model_raw_mat", raw_reconstruction),
        ("model_completed_mat", completed_reconstruction),
    ):
        if save_mat_companion(artifact_paths[artifact_name], data) is None:
            artifact_paths.pop(artifact_name)
    torch.save(
        {
            "model_name": config.model_name,
            "model_hyperparameters": config.model_hyperparameters,
            "image_shape": tuple(ground_truth.shape),
            "state_dict": final_fit_output.state_dict,
            "training_phase": "refit_on_all_observations",
            "selected_steps": selection_output.best_step,
            "selection_best_validation_mse": selection_output.best_validation_mse,
        },
        artifact_paths["checkpoint"],
    )

    interpolation_metrics = _evaluate(
        interpolation,
        ground_truth,
        observed_mask,
        config.training.device,
        config.learned_metrics,
    )
    selection_metrics = _evaluate(
        selection_reconstruction,
        ground_truth,
        observed_mask,
        config.training.device,
        config.learned_metrics,
    )
    model_metrics = _evaluate(
        completed_reconstruction,
        ground_truth,
        observed_mask,
        config.training.device,
        config.learned_metrics,
    )
    psnr_delta = None
    if (
        interpolation_metrics["missing_psnr"] is not None
        and model_metrics["missing_psnr"] is not None
    ):
        psnr_delta = (
            model_metrics["missing_psnr"] - interpolation_metrics["missing_psnr"]
        )

    payload = {
        "run_id": run_id,
        "status": "success",
        "image_shape": list(ground_truth.shape),
        "requested_missing_rate": config.missing_rate,
        "actual_missing_rate": float((~observed_mask).mean()),
        "model_name": config.model_name,
        "model_hyperparameters": config.model_hyperparameters,
        "selection": {
            "best_step": selection_output.best_step,
            "best_validation_mse": selection_output.best_validation_mse,
            "runtime_seconds": selection_output.runtime_seconds,
            "parameter_count": selection_output.parameter_count,
            "device": selection_output.device,
            "stopped_early": selection_output.stopped_early,
            "train_observed_pixels": int(selection_output.train_mask.sum()),
            "validation_observed_pixels": int(selection_output.validation_mask.sum()),
        },
        "final_fit": {
            "fitted_steps": final_fit_output.fitted_steps,
            "final_train_mse": final_fit_output.final_train_mse,
            "runtime_seconds": final_fit_output.runtime_seconds,
            "parameter_count": final_fit_output.parameter_count,
            "device": final_fit_output.device,
            "observed_pixels_used": int(final_fit_output.fit_mask.sum()),
        },
        "interpolation": interpolation_metrics,
        "selection_model_diagnostic": selection_metrics,
        "tensor_model": model_metrics,
        "comparison": {
            "psnr_delta_vs_interpolation": psnr_delta,
            "ssim_delta_vs_interpolation": (
                model_metrics["composite_ssim"]
                - interpolation_metrics["composite_ssim"]
            ),
            "refit_psnr_delta_vs_selection": (
                None
                if model_metrics["missing_psnr"] is None
                or selection_metrics["missing_psnr"] is None
                else model_metrics["missing_psnr"]
                - selection_metrics["missing_psnr"]
            ),
        },
        "total_runtime_seconds": float(time.perf_counter() - started_at),
        "artifacts": artifact_paths,
        "notes": {
            "ground_truth_usage": "final_evaluation_only",
            "tuning_signal": "held_out_observed_pixels_only",
            "final_fit": "Selected steps are refit from scratch on 100% of observed pixels.",
            "model_output": "Known samples in model_completed.npy are copied from observations.",
            "learned_iqa": (
                "LPIPS/MANIQA/CLIP-IQA/MUSIQ are evaluation-only diagnostics "
                "and never participate in tuning; they are skipped for MSI/video."
            ),
        },
    }

    config_payload = config.to_dict()
    config_payload["run_id"] = run_id
    config_payload["resolved_image_shape"] = list(ground_truth.shape)
    _write_json(Path(artifact_paths["config"]), config_payload)
    _write_json(
        Path(artifact_paths["selection_history"]),
        {"history": selection_output.history},
    )
    _write_json(
        Path(artifact_paths["final_fit_history"]),
        {"history": final_fit_output.history},
    )
    _write_json(Path(artifact_paths["metrics"]), payload)
    return payload
