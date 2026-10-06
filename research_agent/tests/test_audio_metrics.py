import json

import numpy as np
import pytest

from research_agent.core.audio_metrics import (
    active_audio_metadata, audio_metric_context, audio_nmse, metric_score,
)
from research_agent.core.metrics import evaluate_reconstruction_metrics
from research_agent.core.experiment_judge import judge_candidate
from research_agent.candidate.generator import _algorithm_comparison_reference
from research_agent.evolution_knowledge import describe_metrics
from research_agent.recovery import summarize_evaluations


def audio_case():
    original = np.array([[-0.8, 0.4], [0.2, -0.6], [0.7, 0.1], [-0.3, 0.5], [0.4, -0.2]])
    tensor = np.zeros((3, 2, 2), dtype=np.float64)
    tensor.reshape(-1, 2)[:5] = (original + 1) / 2
    mask = np.ones_like(tensor, dtype=bool)
    mask.reshape(-1, 2)[1:3] = False
    metadata = {"data_type": "audio", "channels": 2, "sample_count": 5,
                "normalization": "min_max", "original_min": -1.0, "original_max": 1.0}
    return tensor, mask, metadata, original


def test_nmse_uses_original_amplitude_missing_samples_not_shifted_tensor():
    gt, mask, metadata, original = audio_case()
    prediction = gt.copy()
    prediction[~mask] += 0.1
    prediction[mask] = 100  # Known entries and padding must not affect error.
    result = audio_nmse(prediction, gt, mask, metadata)
    expected = 4 * 0.2 ** 2 / np.square(original[1:3]).sum()
    assert result["missing_nmse"] == pytest.approx(expected)
    assert set(result) == {"missing_nmse", "evaluation_metric", "perfect_reconstruction", "audio_metric_status"}
    json.dumps(result, allow_nan=False)


def test_audio_metric_context_never_calls_image_metrics_and_resets(monkeypatch):
    from research_agent.core import metrics
    gt, mask, metadata, _ = audio_case()
    def forbidden(*args, **kwargs):
        raise AssertionError("audio must not calculate image scores")
    for name in ("missing_region_psnr", "full_image_psnr", "composite_ssim", "learned_image_quality_metrics"):
        monkeypatch.setattr(metrics, name, forbidden)
    with pytest.raises(RuntimeError):
        with audio_metric_context(metadata):
            result = evaluate_reconstruction_metrics(gt, gt, mask,
                include_full_reference_metrics=True, include_no_reference_metrics=True)
            assert result["missing_nmse"] == 0
            assert active_audio_metadata() is metadata
            raise RuntimeError("simulate failed run")
    assert active_audio_metadata() is None


def test_zero_energy_is_not_fabricated_and_undefined_scores_do_not_rank():
    gt = np.zeros((2, 2, 1))
    mask = np.array([False, False, True, True]).reshape(gt.shape)
    metadata = {"data_type": "audio", "channels": 1, "sample_count": 4, "normalization": "none"}
    assert audio_nmse(gt, gt, mask, metadata)["missing_nmse"] == 0
    invalid = audio_nmse(np.ones_like(gt), gt, mask, metadata)
    assert invalid["missing_nmse"] is None and not invalid["audio_metric_status"]["defined"]
    assert metric_score(invalid) < metric_score({"missing_nmse": 1.0})
    summary = summarize_evaluations([{"status": "completed", "result": {"metrics": invalid}}], 1, "audio")
    assert not summary["complete"] and summary["mean_missing_nmse"] is None


@pytest.mark.parametrize("candidate,threshold,accepted", [(0.1, 0, True), (0.2, 0, False), (0.3, 0, False), (0.1, 0.15, False), (None, 0, False)])
def test_audio_judge_only_gates_nmse(candidate, threshold, accepted):
    tuning = {"trials": [{"runtime_seconds": 1}], "trial_count": 1, "best": {"best_step": 2}}
    baseline = {"metrics": {"missing_nmse": 0.2}, "runtime_seconds": 0}
    proposed = {"metrics": {"missing_nmse": candidate}, "runtime_seconds": 0}
    result = judge_candidate(tuning, baseline, tuning, proposed,
                            minimum_psnr_delta=1000, ssim_tolerance=0,
                            minimum_nmse_delta=threshold)
    assert result["accepted"] is accepted
    assert result["primary_metric"] == "missing_nmse"
    assert result["thresholds"] == {"minimum_nmse_delta": threshold}


def test_audio_reference_ranks_lower_nmse_and_describes_no_image_metrics():
    state = {"selected_model": "tucker", "results": {
        "interpolation_metrics": {"missing_nmse": 0.1},
        "tensor_metrics": {"missing_nmse": 0.3},
        "siren_comparison": {"status": "completed", "metrics": {"missing_nmse": 0.2}},
    }}
    reference = _algorithm_comparison_reference(state)
    assert reference["selection_metric"] == "missing_nmse"
    assert reference["winner"] == "linear_interpolation_waveform"
    assert reference["selected_tensor_nmse_above_best"] == pytest.approx(0.2)
    assert "selected_tensor_gap_to_best_missing_psnr_db" not in reference
    assert list(describe_metrics({"missing_nmse": 0.2})) == ["missing_nmse"]
    results = [{"status": "completed", "result": {"metrics": {"missing_nmse": value}}} for value in (0.1, 0.3)]
    summary = summarize_evaluations(results, 2, "audio")
    assert summary["mean_missing_nmse"] == pytest.approx(0.2)
    assert "mean_missing_psnr" not in summary and "mean_composite_ssim" not in summary


def test_audio_llm_prompt_has_nmse_objective_not_image_objective():
    from research_agent.candidate.generator import CandidateGenerator
    context = {"base_metrics": {"missing_nmse": 0.2}, "base_method": "tucker",
               "base_class_name": "TuckerDecomposition",
               "image_profile": {"data_type": "audio"}, "previous_failure_feedback": []}
    messages = CandidateGenerator(None)._messages(context)
    payload = json.loads(messages[-1]["content"])
    protocol = payload["evolution_protocol"]
    assert "psnr_feedback_rule" not in protocol
    assert "Audio uses only missing original-waveform NMSE" in protocol["optimization_objective_rule"]


def test_audio_profile_excludes_padding_even_when_padding_mask_is_missing(tmp_path):
    from research_agent.agent_tools.research_tools import AnalyzeImageTool
    gt, mask, metadata, _ = audio_case()
    source = tmp_path / "audio.npy"
    np.save(source, gt)
    profiles = []
    for padding_observed in (True, False):
        mask.reshape(-1, 2)[5] = padding_observed
        mask_path = tmp_path / "mask.npy"
        np.save(mask_path, mask)
        with audio_metric_context(metadata):
            response = AnalyzeImageTool().run({"run_id": "test", "image_path": str(source),
                "run_dir": str(tmp_path / str(padding_observed)), "mask_type": "random",
                "missing_rate": 0.4, "seed": 1, "image_size": None,
                "data_type": "audio", "observation_mask_path": str(mask_path), "valid_element_count": 10})
        profiles.append(response.data["profile"])
    assert profiles[0]["observed_pixels"] == profiles[1]["observed_pixels"] == 6
    assert profiles[0]["actual_missing_rate"] == profiles[1]["actual_missing_rate"] == 0.4
    assert profiles[0]["visible_channel_mean"] == profiles[1]["visible_channel_mean"]
