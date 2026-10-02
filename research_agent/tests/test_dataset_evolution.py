import json
import numpy as np

from research_agent.dataset_evolution import evaluate_across_cases, judge_across_cases, training_feedback, compact_case_feedback


def _local():
    return {"accepted": False, "decision": "reject", "psnr_delta": -1.0,
            "ssim_delta": 0.0, "lpips_delta": None,
            "training_behavior": {"candidate_selected_checkpoint_unstable": False},
            "gate_failures": ["development sample is worse"],
            "suspected_causes": [], "next_round_constraints": []}


def test_whole_dataset_can_accept_when_development_sample_is_worse():
    before = {"summary": {"complete": True, "mean_missing_psnr": 20.0,
                          "mean_composite_ssim": 0.80, "perfect_count": 0}}
    after = {"summary": {"complete": True, "mean_missing_psnr": 21.0,
                         "mean_composite_ssim": 0.805, "perfect_count": 0}}
    judgment = judge_across_cases(_local(), before, after, 0.2, 0.002, 0.0)
    assert judgment["accepted"]
    assert judgment["psnr_delta"] == 1.0
    assert not judgment["development_sample_judgment"]["accepted"]


def test_whole_dataset_rejects_missing_sample_even_with_strong_mean():
    before = {"summary": {"complete": True, "mean_missing_psnr": 20.0,
                          "mean_composite_ssim": 0.8, "perfect_count": 0}}
    after = {"summary": {"complete": False, "mean_missing_psnr": None,
                         "mean_composite_ssim": None, "perfect_count": 0}}
    judgment = judge_across_cases(_local(), before, after, 0.2, 0.002, 0.0)
    assert not judgment["accepted"]
    assert "every valid sample" in judgment["gate_failures"][0]


def test_complete_candidate_can_replace_incomplete_incumbent():
    before = {"summary": {"complete": False, "mean_missing_psnr": None,
                          "mean_composite_ssim": None, "perfect_count": 0}}
    after = {"summary": {"complete": True, "mean_missing_psnr": 19.0,
                         "mean_composite_ssim": 0.8, "perfect_count": 0}}
    judgment = judge_across_cases(_local(), before, after, 0.2, 0.002, 0.0)
    assert judgment["accepted"]
    assert judgment["rescued_incomplete_incumbent"]


def test_audio_gate_uses_mean_nmse_only():
    before = {"summary": {"complete": True, "mean_missing_nmse": 0.4}}
    after = {"summary": {"complete": True, "mean_missing_nmse": 0.3}}
    judgment = judge_across_cases(_local(), before, after, 0.2, 0.002, 0.0)
    assert judgment["accepted"]
    assert abs(judgment["nmse_delta"] - 0.1) < 1e-9
    assert judgment["psnr_delta"] is None


def test_each_case_is_fitted_with_frozen_configuration_and_budget(tmp_path, monkeypatch):
    from research_agent import dataset_evolution
    calls = []
    monkeypatch.setattr(dataset_evolution, "load_tensor_data", lambda *args, **kwargs: np.ones((8, 8, 3)))
    monkeypatch.setattr(dataset_evolution, "load_observation_mask", lambda *args: np.ones((8, 8, 3), dtype=bool))
    monkeypatch.setattr(dataset_evolution, "apply_observation_mask", lambda gt, mask: gt)

    def fake_fit(*args, **kwargs):
        calls.append((args, kwargs))
        return {"metrics": {"missing_psnr": 20.0 + len(calls), "missing_mse": 0.01,
                            "composite_ssim": 0.8}, "artifacts": {"reconstruction": "fake.npy"}}

    monkeypatch.setattr(dataset_evolution, "final_fit_and_evaluate", fake_fit)
    cases = [{"source": "first.mat", "gt_path": "a.npy", "mask_path": "a-mask.npy",
              "data_type": "Image", "seed": 5},
             {"source": "second.mat", "gt_path": "b.npy", "mask_path": "b-mask.npy",
              "data_type": "Image", "seed": 6}]
    trial = {"hyperparameters": {"rank": 2}, "learning_rate": 0.01,
             "artifacts": {"selected_checkpoint": "development-only.pt"}}
    result = evaluate_across_cases("candidate", None, trial, cases, tmp_path, 30, 3, 0, "cpu")
    assert result["summary"]["complete"]
    assert result["summary"]["mean_missing_psnr"] == 21.5
    assert len(calls) == 2
    for args, _ in calls:
        assert args[2] == {"hyperparameters": {"rank": 2}, "learning_rate": 0.01,
                           "runtime_seconds": 0.0}
        assert args[6].max_steps == 30
        assert args[6].validation_interval == 3
        assert args[6].early_stopping_patience == 31


def test_per_sample_curve_digest_reaches_feedback(tmp_path):
    path = tmp_path / "final_fit_history.json"
    path.write_text(json.dumps({"history": [
        {"step": step, "data_train_loss": 1 / step, "total_train_loss": 1 / step,
         "validation_mse": 1 / step, "missing_gt_psnr": float(step)}
        for step in range(1, 7)]}), encoding="utf-8")
    result = {"selected_steps": 6, "runtime_seconds": 1.2, "parameter_count": 10,
              "artifacts": {"history": str(path)}}
    digest = training_feedback(result)
    assert digest["curve_summary"]["record_count"] == 6
    assert digest["curve_summary"]["best_validation"]["step"] == 6
    assert digest["curve_points"][0]["step"] == 1
    feedback = compact_case_feedback({"results": [{"source": "a.mat", "status": "completed",
                                                   "metrics": {"missing_psnr": 25.0},
                                                   "training": digest}]})
    assert feedback[0]["training"]["selected_step"] == 6
