"""Day 1 deterministic interpolation experiment pipeline."""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path

from ..schemas import ExperimentConfig, ExperimentResult
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


def _make_run_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return "day1-%s-%s" % (timestamp, uuid.uuid4().hex[:6])


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2, allow_nan=False)
        output_file.write("\n")


def run_day1_baseline(config: ExperimentConfig) -> ExperimentResult:
    """Run the reproducible Day 1 nearest-neighbor experiment.

    The interpolation function receives only ``corrupted`` and ``mask``.  The
    complete image is passed to metric functions only after reconstruction.
    """

    config.validate()
    started_at = time.perf_counter()
    run_id = _make_run_id()
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

    # The baseline has no access to ground_truth after the mask is applied.
    reconstruction = nearest_neighbor_fill(corrupted, observed_mask)

    metrics = evaluate_reconstruction_metrics(
        reconstruction,
        ground_truth,
        observed_mask,
        include_full_reference_metrics=config.full_reference_metrics,
        include_no_reference_metrics=config.no_reference_metrics,
    )

    artifact_paths = {
        "config": str(run_dir / "config.json"),
        "original": str(run_dir / "original.png"),
        "corrupted": str(run_dir / "corrupted.png"),
        "mask": str(run_dir / "mask.png"),
        "interpolated": str(run_dir / "interpolated.png"),
        "original_data": str(run_dir / "original.npy"),
        "corrupted_data": str(run_dir / "corrupted.npy"),
        "interpolated_data": str(run_dir / "interpolated.npy"),
        "original_mat": str(run_dir / "original.mat"),
        "corrupted_mat": str(run_dir / "corrupted.mat"),
        "interpolated_mat": str(run_dir / "interpolated.mat"),
        "metrics": str(run_dir / "metrics.json"),
    }
    save_image(artifact_paths["original"], ground_truth)
    save_image(artifact_paths["corrupted"], corrupted)
    save_mask(artifact_paths["mask"], observed_mask)
    save_image(artifact_paths["interpolated"], reconstruction)
    save_tensor_data(artifact_paths["original_data"], ground_truth)
    save_tensor_data(artifact_paths["corrupted_data"], corrupted)
    save_tensor_data(artifact_paths["interpolated_data"], reconstruction)
    for artifact_name, data in (
        ("original_mat", ground_truth),
        ("corrupted_mat", corrupted),
        ("interpolated_mat", reconstruction),
    ):
        if save_mat_companion(artifact_paths[artifact_name], data) is None:
            artifact_paths.pop(artifact_name)

    result = ExperimentResult(
        run_id=run_id,
        algorithm_name="nearest_neighbor_manhattan",
        status="success",
        seed=config.seed,
        requested_missing_rate=config.missing_rate,
        actual_missing_rate=float((~observed_mask).mean()),
        missing_mse=metrics["missing_mse"],
        missing_psnr=metrics["missing_psnr"],
        full_psnr=metrics["full_psnr"],
        perfect_reconstruction=metrics["perfect_reconstruction"],
        composite_ssim=metrics["composite_ssim"],
        lpips=metrics["lpips"],
        maniqa=metrics["maniqa"],
        clip_iqa=metrics["clip_iqa"],
        musiq=metrics["musiq"],
        learned_metric_status=metrics["learned_metric_status"],
        metric_group_status=metrics["metric_group_status"],
        runtime_seconds=float(time.perf_counter() - started_at),
        image_shape=list(ground_truth.shape),
        artifacts=artifact_paths,
        notes={
            "mask_convention": "1/white=observed, 0/black=missing",
            "ground_truth_usage": "evaluation_only",
            "ssim_definition": (
                "Known pixels are replaced by ground truth before SSIM averaged over "
                "all feature planes; "
                "this is composite SSIM, not a standardized masked SSIM."
            ),
            "learned_iqa_definition": (
                "LPIPS is full-reference; MANIQA, CLIP-IQA, and MUSIQ are "
                "no-reference. All receive the composite full image with known "
                "pixels restored before evaluation. These learned metrics are skipped "
                "for MSI/video because their pretrained inputs are RGB-only."
            ),
        },
    )

    config_payload = config.to_dict()
    config_payload["resolved_image_shape"] = list(ground_truth.shape)
    config_payload["run_id"] = run_id
    _write_json(Path(artifact_paths["config"]), config_payload)
    _write_json(Path(artifact_paths["metrics"]), result.to_dict())
    return result
