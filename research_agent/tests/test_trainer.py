import numpy as np
import pytest
import torch

from research_agent.core.data import apply_observation_mask
from research_agent.core.masks import generate_observation_mask
from research_agent.core.trainer import (
    fit_tensor_model_on_all_observations,
    train_tensor_model,
)
from research_agent.schemas import TrainingConfig
from research_agent.core.models.base import BaseTensorInpaintingModel


class _ConstantModel(BaseTensorInpaintingModel):
    def __init__(self, shape, mean, parameters):
        super().__init__(shape, mean)
        self.value = torch.nn.Parameter(torch.tensor(parameters.get("value", -1.1)))

    def forward(self):
        return self.value.expand(self.image_shape)

    @classmethod
    def search_space(cls, shape):
        return {}


class _TrackedConstantModel(_ConstantModel):
    checkpoints = []

    def forward(self):
        if not self.training:
            self.checkpoints.append(float(self.value.detach()))
        return super().forward()

    def regularization_terms(self):
        return {"value_regularization": self.value.square() * 0.1}


def test_curve_losses_and_gt_scores_refer_to_same_post_update_checkpoint():
    _TrackedConstantModel.checkpoints = []
    gt = np.ones((2, 2, 1), dtype=np.float32)
    gt[-1, -1, 0] = 0.8
    mask = np.ones_like(gt, dtype=bool)
    mask[-1, -1, 0] = False
    output = train_tensor_model("diagnostic", {"value": 0.4}, np.where(mask, gt, 0), mask,
        TrainingConfig(learning_rate=0.1, max_steps=3, validation_interval=1,
                       early_stopping_patience=5, device="cpu"), 42,
        model_builder=_TrackedConstantModel, ground_truth=gt)
    for value, point in zip(_TrackedConstantModel.checkpoints, output.history):
        assert point["data_train_loss"] == pytest.approx((value - 1) ** 2)
        assert point["total_train_loss"] == pytest.approx((value - 1) ** 2 + 0.1 * value ** 2)
        assert point["regularization_losses"]["value_regularization"] == pytest.approx(0.1 * value ** 2)
        assert point["validation_mse"] == pytest.approx((value - 0.8) ** 2, abs=1e-7)
        assert point["training_loss_timing"] == "post_optimizer_step"
    best = next(point for point in output.history if point["step"] == output.best_step)
    value = float(output.state_dict["value"])
    assert best["data_train_loss"] == pytest.approx((value - 1) ** 2)


def test_persisted_and_reused_checkpoint_losses_match_raw_model_and_llm_feedback(tmp_path, monkeypatch):
    from research_agent.core import fair_experiment
    from research_agent.agent_tools.research_tools import _persist_selected_training_output
    from research_agent.dataset_evolution import training_feedback
    gt = np.ones((2, 2, 1), dtype=np.float32)
    gt[-1, -1, 0] = 0
    mask = np.ones_like(gt, dtype=bool)
    mask[-1, -1, 0] = False
    observed = np.where(mask, gt, 0)
    config = TrainingConfig(learning_rate=0.6, max_steps=2, validation_interval=1,
                            early_stopping_patience=5, device="cpu")
    output = train_tensor_model("diagnostic", {}, observed, mask, config, 42,
        model_builder=_ConstantModel, ground_truth=gt)
    expected = (float(output.state_dict["value"]) - 1) ** 2
    saved = _persist_selected_training_output(tmp_path / "tool", "test", "diagnostic", {}, 0.6, observed, mask, output)
    assert saved["final_train_mse"] == pytest.approx(expected)
    trial = {"hyperparameters": {}, "learning_rate": 0.6, "best_step": output.best_step,
             "best_validation_mse": output.best_validation_mse, "best_missing_psnr": output.best_missing_psnr,
             "parameter_count": output.parameter_count, "history": output.history, "runtime_seconds": 0.0}
    trial["artifacts"] = fair_experiment._persist_selected_trial(str(tmp_path / "selected"), "diagnostic", trial, output)
    def forbidden(*args, **kwargs):
        raise AssertionError("must reuse trained checkpoint")
    monkeypatch.setattr(fair_experiment, "train_tensor_model", forbidden)
    result = fair_experiment.final_fit_and_evaluate("diagnostic", _ConstantModel, trial, observed, mask, gt,
        config, 42, str(tmp_path / "final"))
    assert result["reused_without_retraining"]
    assert result["final_train_mse"] == result["final_total_loss"] == pytest.approx(expected)
    feedback = training_feedback(result)
    best_point = next(point for point in feedback["curve_points"] if point["step"] == output.best_step)
    assert best_point["data_train_loss"] == pytest.approx(expected)
    assert feedback["curve_summary"]["training_loss_protocol"]["timing"] == ["post_optimizer_step"]


@pytest.mark.parametrize("refit", [False, True])
def test_nonfinite_post_update_prediction_is_not_hidden_by_clipping(refit):
    class BadEvaluationModel(_ConstantModel):
        def forward(self):
            output = super().forward()
            return output if self.training else output * float("inf")
    gt = np.ones((2, 2, 1), dtype=np.float32)
    mask = np.ones_like(gt, dtype=bool)
    mask[-1, -1, 0] = False
    parameters = {"model_name": "diagnostic", "model_hyperparameters": {"value": 0.4},
                  "observed_image": np.where(mask, gt, 0), "observed_mask": mask,
                  "config": TrainingConfig(max_steps=1, validation_interval=1, device="cpu"),
                  "seed": 1, "model_builder": BadEvaluationModel}
    with pytest.raises(FloatingPointError, match="checkpoint prediction"):
        if refit:
            fit_tensor_model_on_all_observations(**parameters, selected_steps=1)
        else:
            train_tensor_model(**parameters, ground_truth=gt)


@pytest.mark.parametrize("audio", [False, True])
def test_checkpoint_selection_scores_the_same_clipped_output_as_final_metrics(audio):
    from research_agent.core.audio_metrics import audio_metric_context, audio_nmse
    from research_agent.core.metrics import missing_region_mse
    gt = np.ones((2, 2, 1), dtype=np.float32)
    gt[-1, -1, 0] = 0
    mask = np.ones_like(gt, dtype=bool)
    mask[-1, -1, 0] = False
    metadata = {"data_type": "audio", "channels": 1, "sample_count": 4,
                "normalization": "min_max", "original_min": -1, "original_max": 1} if audio else None
    with audio_metric_context(metadata):
        output = train_tensor_model("diagnostic", {}, np.where(mask, gt, 0), mask,
            TrainingConfig(learning_rate=0.6, max_steps=2, validation_interval=1,
                           early_stopping_patience=5, device="cpu"), 42,
            model_builder=_ConstantModel, ground_truth=gt)
    # Step 1 predicts -0.5, which becomes the exact missing value after clipping.
    # Step 2 predicts +0.086: its raw MSE is lower, but its actual output is worse.
    assert output.best_step == 1
    assert output.best_validation_mse == missing_region_mse(output.reconstruction, gt, mask) == 0
    if audio:
        assert output.best_missing_nmse == audio_nmse(output.reconstruction, gt, mask, metadata)["missing_nmse"] == 0
    else:
        assert output.best_missing_psnr is None  # Perfect PSNR uses the existing null convention.


def test_min_delta_only_controls_patience_not_the_best_checkpoint():
    gt = np.ones((2, 2, 1), dtype=np.float32)
    gt[-1, -1, 0] = 0.8
    mask = np.ones_like(gt, dtype=bool)
    mask[-1, -1, 0] = False
    output = train_tensor_model("diagnostic", {"value": 0.4}, np.where(mask, gt, 0), mask,
        TrainingConfig(learning_rate=1e-6, max_steps=3, validation_interval=1,
                       early_stopping_patience=5, early_stopping_min_delta=1e-3, device="cpu"), 42,
        model_builder=_ConstantModel, ground_truth=gt)
    assert output.best_step == 3
    assert output.best_validation_mse == min(item["validation_mse"] for item in output.history)


def _smooth_rgb_image(height=12, width=10):
    y, x = np.indices((height, width), dtype=np.float32)
    return np.stack(
        (
            x / (width - 1),
            y / (height - 1),
            (x + y) / (width + height - 2),
        ),
        axis=-1,
    ).astype(np.float32)


def test_trainer_uses_disjoint_observed_split_and_is_reproducible():
    ground_truth = _smooth_rgb_image()
    observed_mask = generate_observation_mask(12, 10, 0.25, "random", seed=5)
    corrupted = apply_observation_mask(ground_truth, observed_mask)
    config = TrainingConfig(
        learning_rate=0.05,
        max_steps=60,
        validation_observed_ratio=0.15,
        validation_interval=5,
        early_stopping_patience=20,
        device="cpu",
    )
    hyperparameters = {"rank": 3, "init_scale": 0.1}

    first = train_tensor_model(
        "matrix", hyperparameters, corrupted, observed_mask, config, seed=17
    )
    second = train_tensor_model(
        "matrix", hyperparameters, corrupted, observed_mask, config, seed=17
    )

    assert first.reconstruction.shape == ground_truth.shape
    assert np.isfinite(first.reconstruction).all()
    assert np.array_equal(first.train_mask | first.validation_mask, observed_mask)
    assert not np.logical_and(first.train_mask, first.validation_mask).any()
    assert not first.train_mask[~observed_mask].any()
    assert not first.validation_mask[~observed_mask].any()
    assert first.best_validation_mse <= first.history[0]["validation_mse"]
    assert first.parameter_count > 0
    assert first.device == "cpu"
    assert np.array_equal(first.reconstruction, second.reconstruction)
    assert first.best_validation_mse == second.best_validation_mse


def test_final_refit_uses_every_observed_pixel_and_no_hidden_pixel():
    ground_truth = _smooth_rgb_image()
    observed_mask = generate_observation_mask(12, 10, 0.25, "block", seed=9)
    corrupted = apply_observation_mask(ground_truth, observed_mask)
    config = TrainingConfig(
        learning_rate=0.05,
        max_steps=40,
        validation_observed_ratio=0.15,
        validation_interval=5,
        early_stopping_patience=20,
        device="cpu",
    )

    result = fit_tensor_model_on_all_observations(
        model_name="matrix",
        model_hyperparameters={"rank": 3, "init_scale": 0.1},
        observed_image=corrupted,
        observed_mask=observed_mask,
        config=config,
        selected_steps=25,
        seed=17,
    )

    assert result.fitted_steps == 25
    assert np.array_equal(result.fit_mask, observed_mask)
    assert not result.fit_mask[~observed_mask].any()
    assert result.final_train_mse >= 0.0
    assert np.isfinite(result.reconstruction).all()


def test_ground_truth_selection_trains_on_all_observed_and_scores_only_missing():
    ground_truth = _smooth_rgb_image()
    observed_mask = generate_observation_mask(12, 10, 0.25, "random", seed=8)
    corrupted = apply_observation_mask(ground_truth, observed_mask)
    result = train_tensor_model(
        model_name="matrix",
        model_hyperparameters={"rank": 3, "init_scale": 0.1},
        observed_image=corrupted,
        observed_mask=observed_mask,
        ground_truth=ground_truth,
        config=TrainingConfig(
            learning_rate=0.05,
            max_steps=20,
            validation_interval=5,
            early_stopping_patience=20,
            device="cpu",
        ),
        seed=17,
    )

    assert np.array_equal(result.train_mask, observed_mask)
    assert np.array_equal(result.validation_mask, ~observed_mask)
    assert result.selection_metric == "missing_region_ground_truth_mse"
    assert result.ground_truth_used_for_selection is True
    assert result.best_missing_psnr is not None
    assert all(item["missing_gt_mse"] is not None for item in result.history)


def test_training_progress_reports_current_step_and_eta(capsys):
    ground_truth = _smooth_rgb_image(8, 8)
    observed_mask = generate_observation_mask(8, 8, 0.25, "random", seed=3)
    corrupted = apply_observation_mask(ground_truth, observed_mask)
    config = TrainingConfig(
        learning_rate=0.03,
        max_steps=3,
        validation_observed_ratio=0.15,
        validation_interval=1,
        early_stopping_patience=5,
        device="cpu",
    )

    train_tensor_model(
        "matrix",
        {"rank": 3, "init_scale": 0.1},
        corrupted,
        observed_mask,
        config,
        seed=17,
        progress_label="测试进度",
    )
    output = capsys.readouterr().out

    assert "测试进度" in output
    assert "1/3" in output
    assert "3/3" in output
    assert "ETA" in output


def test_audio_padding_is_excluded_from_training_initialization_and_validation():
    from research_agent.core.audio_metrics import audio_metric_context
    from research_agent.core.trainer import observed_feature_mean
    metadata = {"data_type": "audio", "sample_count": 3, "channels": 1,
                "normalization": "none"}
    gt = np.array([1, 0.5, 1, 0], dtype=np.float32).reshape(1, 4, 1)
    mask = np.array([True, False, True, True]).reshape(gt.shape)
    config = TrainingConfig(learning_rate=0.001, max_steps=3, validation_interval=1,
                            early_stopping_patience=5, device="cpu")
    with audio_metric_context(metadata):
        assert observed_feature_mean(gt, mask)[0] == 1
        first = train_tensor_model("siren", {"hidden_features": 16}, np.where(mask, gt, 0),
                                   mask, config, 42, ground_truth=gt)
        poisoned = gt.copy()
        poisoned.reshape(-1)[3] = 999
        second = train_tensor_model("siren", {"hidden_features": 16}, np.where(mask, poisoned, 0),
                                    mask, config, 42, ground_truth=poisoned)
        refit = fit_tensor_model_on_all_observations("siren", {"hidden_features": 16},
                    np.where(mask, gt, 0), mask, config, 3, 42)
    assert not first.train_mask.reshape(-1)[3]
    assert not first.validation_mask.reshape(-1)[3]
    assert not refit.fit_mask.reshape(-1)[3]
    assert first.train_mask.sum() == 2
    np.testing.assert_array_equal(first.reconstruction, second.reconstruction)


def test_siren_audio_training_is_invariant_to_frame_size_and_padding_count():
    from research_agent.core.audio_metrics import audio_metric_context
    waveform = (0.5 + 0.3 * np.sin(np.arange(10))).astype(np.float32)
    outputs = []
    metadata = {"data_type": "audio", "sample_count": 10, "channels": 1, "normalization": "none"}
    config = TrainingConfig(learning_rate=1e-4, max_steps=5, validation_interval=1,
                            early_stopping_patience=10, device="cpu")
    for shape in ((3, 4, 1), (2, 8, 1)):
        gt = np.zeros(shape, dtype=np.float32)
        gt.reshape(-1)[:10] = waveform
        mask = np.ones_like(gt, dtype=bool)
        mask.reshape(-1)[[2, 5, 7]] = False
        with audio_metric_context(metadata):
            outputs.append(train_tensor_model("siren", {"hidden_features": 16, "hidden_layers": 2},
                np.where(mask, gt, 0), mask, config, 42, ground_truth=gt))
    np.testing.assert_allclose(outputs[0].reconstruction.reshape(-1)[:10],
                               outputs[1].reconstruction.reshape(-1)[:10], atol=1e-6)
    assert outputs[0].best_step == outputs[1].best_step
    assert outputs[0].best_missing_nmse == pytest.approx(outputs[1].best_missing_nmse, abs=1e-6)


def test_audio_padding_cannot_supply_the_only_observations_or_missing_targets():
    from research_agent.core.audio_metrics import audio_metric_context
    metadata = {"data_type": "audio", "sample_count": 2, "channels": 1, "normalization": "none"}
    gt = np.ones((1, 4, 1), dtype=np.float32)
    with audio_metric_context(metadata):
        for bits, message in (([0, 0, 1, 1], "genuinely observed"), ([1, 1, 0, 0], "missing valid")):
            mask = np.array(bits, dtype=bool).reshape(gt.shape)
            with pytest.raises(ValueError, match=message):
                train_tensor_model("siren", {"hidden_features": 16}, np.where(mask, gt, 0), mask,
                                   TrainingConfig(max_steps=1, device="cpu"), 42, ground_truth=gt)
