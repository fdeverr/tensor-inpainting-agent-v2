from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import torch

from research_agent.core.data import (
    apply_observation_mask,
    load_tensor_data,
    save_tensor_data,
    to_rgb_preview,
)
from research_agent.core.interpolation import nearest_neighbor_fill
from research_agent.core.metrics import evaluate_reconstruction_metrics
from research_agent.core.models.registry import MODEL_CLASSES
from research_agent.core.trainer import train_tensor_model
from research_agent.schemas import TrainingConfig
from research_agent.workflow_day4 import Day4WorkflowConfig, run_day4_workflow


MODEL_PARAMETERS = {
    "matrix": {"rank": 3, "init_scale": 0.1},
    "mode3": {"rank": 2, "init_scale": 0.1},
    "cp": {"rank": 3, "init_scale": 0.1},
    "nonnegative_cp": {"rank": 3, "init_scale": 0.1},
    "tucker": {"rank_h": 3, "rank_w": 3, "rank_c": 3, "init_scale": 0.1},
    "btd": {
        "num_blocks": 2,
        "rank_h": 3,
        "rank_w": 3,
        "rank_c": 2,
        "init_scale": 0.1,
    },
    "tsvd": {"rank": 3, "init_scale": 0.1},
    "nonnegative_tucker": {
        "rank_h": 3,
        "rank_w": 3,
        "rank_c": 2,
        "init_scale": 0.1,
    },
    "hierarchical_tucker": {
        "rank_h": 3,
        "rank_w": 3,
        "rank_c": 3,
        "rank_spatial": 2,
        "init_scale": 0.1,
    },
    "tt": {"rank_1": 3, "rank_2": 3, "init_scale": 0.1},
    "tensor_ring": {"rank": 3, "init_scale": 0.1},
    "siren": {
        "hidden_features": 16,
        "hidden_layers": 2,
        "first_omega_0": 30.0,
        "hidden_omega_0": 30.0,
    },
}


def test_npy_round_trip_and_video_preview(tmp_path):
    video = np.random.default_rng(1).random((8, 9, 4, 3), dtype=np.float32)
    path = tmp_path / "video.npy"
    save_tensor_data(str(path), video)
    loaded, metadata = load_tensor_data(str(path), return_metadata=True)

    assert np.array_equal(loaded, video)
    assert metadata["data_type"] == "video"
    assert metadata["feature_shape"] == [4, 3]
    assert metadata["feature_count"] == 12
    assert to_rgb_preview(loaded).shape == (8, 9, 3)


@pytest.mark.skipif(
    importlib.util.find_spec("scipy") is None,
    reason="scipy is installed by the project requirements on deployment",
)
def test_mat_key_and_normalization(tmp_path):
    from scipy.io import loadmat, savemat

    source = np.arange(8 * 9 * 5, dtype=np.uint16).reshape(8, 9, 5)
    path = tmp_path / "msi.mat"
    savemat(path, {"ignored": np.ones((2, 2)), "cube": source})
    loaded, metadata = load_tensor_data(
        str(path), mat_key="cube", return_metadata=True
    )

    assert loaded.shape == source.shape
    assert loaded.dtype == np.float32
    assert loaded.min() == pytest.approx(0.0)
    assert loaded.max() == pytest.approx(1.0)
    assert metadata["data_type"] == "msi"
    assert metadata["mat_key"] == "cube"
    assert metadata["normalization"] == "min_max"

    completed_path = tmp_path / "completed.mat"
    save_tensor_data(str(completed_path), loaded)
    saved = loadmat(completed_path)["data"]
    assert saved.shape == source.shape
    assert np.allclose(saved, loaded)


def test_spatial_mask_and_interpolation_preserve_video_shape():
    video = np.random.default_rng(2).random((8, 9, 3, 2), dtype=np.float32)
    mask = np.ones((8, 9), dtype=np.bool_)
    mask[2:5, 3:7] = False
    corrupted = apply_observation_mask(video, mask)
    reconstructed = nearest_neighbor_fill(corrupted, mask)

    assert corrupted.shape == video.shape
    assert reconstructed.shape == video.shape
    assert np.array_equal(corrupted[mask], video[mask])
    assert np.all(corrupted[~mask] == 0.0)


@pytest.mark.parametrize("model_name", sorted(MODEL_CLASSES))
def test_every_builtin_model_supports_hwtc(model_name):
    shape = (8, 9, 2, 3)
    model = MODEL_CLASSES[model_name](
        image_shape=shape,
        initial_channel_mean=np.full((2, 3), 0.5, dtype=np.float32),
        **MODEL_PARAMETERS[model_name],
    )
    prediction = model()
    mask = torch.ones(shape[:2], dtype=torch.bool)
    mask[2:4, 3:5] = False
    loss = sum(model.loss_terms(prediction, torch.rand(shape), mask).values())
    loss.backward()

    assert tuple(prediction.shape) == shape
    assert torch.isfinite(prediction).all()


def test_video_training_and_metrics_use_all_features():
    shape = (8, 9, 2, 3)
    ground_truth = np.random.default_rng(3).random(shape, dtype=np.float32)
    mask = np.ones(shape[:2], dtype=np.bool_)
    mask[2:5, 3:6] = False
    corrupted = apply_observation_mask(ground_truth, mask)
    output = train_tensor_model(
        model_name="matrix",
        model_hyperparameters={"rank": 3, "init_scale": 0.1},
        observed_image=corrupted,
        observed_mask=mask,
        config=TrainingConfig(
            learning_rate=0.01,
            max_steps=2,
            validation_observed_ratio=0.1,
            validation_interval=1,
            early_stopping_patience=2,
            device="cpu",
        ),
        seed=5,
    )
    metrics = evaluate_reconstruction_metrics(
        ground_truth,
        ground_truth,
        mask,
        include_full_reference_metrics=True,
        include_no_reference_metrics=True,
    )

    assert output.reconstruction.shape == shape
    assert metrics["missing_mse"] == pytest.approx(0.0)
    assert metrics["composite_ssim"] == pytest.approx(1.0)
    assert metrics["lpips"] is None
    assert metrics["learned_metric_status"]["scope"] == "unsupported_non_rgb_tensor"


def test_day4_workflow_preserves_hwtc_outputs(tmp_path):
    shape = (8, 9, 2, 3)
    source = np.random.default_rng(4).random(shape, dtype=np.float32)
    source_path = tmp_path / "video.npy"
    np.save(source_path, source)

    state = run_day4_workflow(
        Day4WorkflowConfig(
            image_path=str(source_path),
            output_dir=str(tmp_path / "outputs"),
            image_size=None,
            mat_key=None,
            model_name="matrix",
            missing_rate=0.25,
            seed=7,
            max_steps=2,
            max_steps_ceiling=2,
            validation_interval=1,
            patience=2,
            device="cpu",
            full_reference_metrics=True,
            no_reference_metrics=True,
            llm_mode="off",
            retrieval_top_k=3,
        )
    )

    assert state["stage"] == "COMPLETED"
    assert state["results"]["image_profile"]["data_type"] == "video"
    assert state["results"]["image_profile"]["image_shape"] == list(shape)
    assert np.load(state["artifacts"]["interpolation"]).shape == shape
    assert np.load(state["artifacts"]["tensor_reconstruction"]).shape == shape
    assert state["results"]["tensor_metrics"]["learned_metric_status"][
        "scope"
    ] == "unsupported_non_rgb_tensor"
