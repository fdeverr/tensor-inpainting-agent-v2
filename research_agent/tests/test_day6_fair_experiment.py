import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

from research_agent.core.experiment_judge import judge_candidate
from research_agent.core.fair_experiment import (
    final_fit_and_evaluate,
    independent_trial_configurations,
    tune_model_on_observed_pixels,
)
from research_agent.candidate.generator import (
    CandidateGenerationResult,
    deterministic_candidate,
)
from research_agent.workflow_day6 import Day6Workflow
from research_agent.schemas import TrainingConfig


def test_each_model_searches_independently_up_to_the_same_limit():
    plan = independent_trial_configurations(
        base_search_space={"rank": [4, 8, 12], "init_scale": [0.1]},
        candidate_search_space={
            "rank": [4, 8, 12],
            "init_scale": [0.1],
            "tv_weight": [0.0, 0.0001, 0.001],
        },
        trial_count=3,
        seed=42,
        anchor_base_config={"rank": 8, "init_scale": 0.1},
    )
    assert len(plan["baseline"]) == len(plan["candidate"]) == 3
    assert {"rank": 8, "init_scale": 0.1} in plan["baseline"]
    assert len({tuple(sorted(item.items())) for item in plan["candidate"]}) == 3
    assert plan["requested_trial_limit_per_model"] == 3
    assert plan["baseline_effective_trial_count"] == 3
    assert plan["candidate_effective_trial_count"] == 3
    assert plan["candidate_only_parameter_names"] == ["tv_weight"]


def test_small_incumbent_grid_does_not_reduce_candidate_search():
    plan = independent_trial_configurations(
        base_search_space={"rank": [4]},
        candidate_search_space={"rank": [4], "hidden": [16, 32, 64]},
        trial_count=4,
        seed=42,
        anchor_base_config={"rank": 4},
    )

    assert plan["requested_trial_limit_per_model"] == 4
    assert plan["baseline_effective_trial_count"] == 1
    assert plan["candidate_effective_trial_count"] == 3
    assert plan["baseline_trial_count_reduced"] is True
    assert plan["candidate_trial_count_reduced"] is True
    assert plan["base_available_config_count"] == 1
    assert plan["candidate_available_config_count"] == 3
    assert len(plan["baseline"]) == 1
    assert len(plan["candidate"]) == 3
    assert plan["baseline_search_coverage"] == 1.0
    assert plan["candidate_search_coverage"] == 1.0


def test_small_candidate_grid_does_not_reduce_incumbent_search():
    plan = independent_trial_configurations(
        base_search_space={"rank": [4, 8, 12, 16]},
        candidate_search_space={"hidden": [32]},
        trial_count=4,
        seed=42,
    )

    assert plan["baseline_effective_trial_count"] == 4
    assert plan["candidate_effective_trial_count"] == 1
    assert plan["base_available_config_count"] == 4
    assert plan["candidate_available_config_count"] == 1
    assert len(plan["baseline"]) == 4
    assert len(plan["candidate"]) == 1


def test_different_deep_architecture_receives_only_its_own_parameters():
    paired = independent_trial_configurations(
        base_search_space={"rank": [4, 8, 12], "init_scale": [0.1]},
        candidate_search_space={
            "hidden_dim": [32, 64],
            "num_frequencies": [4, 8],
        },
        trial_count=3,
        seed=42,
        anchor_base_config={"rank": 8, "init_scale": 0.1},
    )

    assert paired["shared_parameter_names"] == []
    assert paired["candidate_only_parameter_names"] == [
        "hidden_dim",
        "num_frequencies",
    ]
    assert all(
        set(config) == {"hidden_dim", "num_frequencies"}
        for config in paired["candidate"]
    )


def test_hybrid_can_use_its_own_range_for_same_named_rank_parameters():
    paired = independent_trial_configurations(
        base_search_space={
            "rank_h": [4, 8, 16],
            "rank_w": [4, 8, 16],
            "rank_c": [1, 2, 3],
            "init_scale": [0.1, 0.15, 0.2],
        },
        candidate_search_space={
            "rank_h": [4, 8],
            "rank_w": [8, 12],
            "rank_c": [2, 3],
            "factor_hidden": [16, 32],
        },
        trial_count=3,
        seed=42,
    )

    assert paired["shared_parameter_names"] == []
    assert paired["same_name_different_range_parameters"] == [
        "rank_c",
        "rank_h",
        "rank_w",
    ]
    assert all(config["rank_h"] in {4, 8} for config in paired["candidate"])
    assert all(config["rank_w"] in {8, 12} for config in paired["candidate"])
    assert all(config["rank_c"] in {2, 3} for config in paired["candidate"])


def test_manifest_contract_accepts_declared_changed_rank_ranges():
    base = {"rank_h": [4, 8, 16], "rank_c": [1, 2, 3]}
    candidate = {"rank_h": [4, 8], "rank_c": [2, 3], "hidden": [16, 32]}
    manifest = {"declared_search_space": dict(candidate)}

    Day6Workflow._validate_candidate_search_contract(base, candidate, manifest)

    with pytest.raises(ValueError):
        Day6Workflow._validate_candidate_search_contract(
            base,
            candidate,
            {"declared_search_space": {**candidate, "rank_h": [8]}},
        )

    Day6Workflow._validate_candidate_search_contract(
        base,
        candidate,
        {"allowed_search_space": dict(candidate)},
    )


def test_llm_training_budget_is_shared_and_clamped_to_user_ceiling():
    workflow = object.__new__(Day6Workflow)
    workflow.config = SimpleNamespace(
        max_steps=2000,
        learning_rate_candidates=(0.001, 0.01, 0.1),
        refine_learning_rate=True,
        learning_rate_refinement_factor=3.0,
        validation_interval=10,
        patience=40,
        device="cpu",
    )
    training, audit = workflow._resolve_training_budget(
        {
            "training_budget": {
                "max_steps": 5000,
                "validation_interval": 25,
                "early_stopping_patience": 60,
                "rationale": "A deeper coordinate network needs a longer horizon.",
            }
        },
        learning_rate=0.01,
    )

    assert training.max_steps == 2000
    assert training.validation_interval == 25
    assert training.early_stopping_patience == 60
    assert audit["max_steps_was_clamped"] is True
    assert audit["effective_shared_budget"]["max_steps"] == 2000


def test_fair_tuner_searches_and_refines_learning_rate(monkeypatch):
    calls = []

    def fake_train(**kwargs):
        learning_rate = kwargs["config"].learning_rate
        calls.append(learning_rate)
        return SimpleNamespace(
            best_step=2,
            best_validation_mse=abs(np.log10(learning_rate) - np.log10(0.03)),
            best_missing_psnr=20.0,
            selection_metric="missing_region_ground_truth_mse",
            ground_truth_used_for_selection=True,
            runtime_seconds=0.1,
            parameter_count=10,
            stopped_early=False,
            history=[{"step": 2}],
        )

    monkeypatch.setattr(
        "research_agent.core.fair_experiment.train_tensor_model", fake_train
    )
    result = tune_model_on_observed_pixels(
        model_name="tucker",
        model_builder=None,
        configurations=[{"rank_h": 2, "rank_w": 2, "rank_c": 2}],
        observed_image=np.zeros((4, 4, 3), dtype=np.float32),
        observed_mask=np.ones((4, 4), dtype=np.bool_),
        ground_truth=np.zeros((4, 4, 3), dtype=np.float32),
        training_config=TrainingConfig(max_steps=2, device="cpu"),
        seed=1,
        learning_rate_candidates=[0.001, 0.01, 0.1],
        refine_learning_rate=True,
        learning_rate_refinement_factor=3.0,
    )

    assert calls == pytest.approx([0.001, 0.01, 0.1, 0.01 / 3.0, 0.03])
    assert result["coarse_trial_count"] == 3
    assert result["fine_trial_count"] == 2
    assert result["best"]["learning_rate"] == pytest.approx(0.03)


def test_deterministic_candidate_uses_longer_fair_training_budget():
    budget = deterministic_candidate("tucker").training_budget

    assert budget.max_steps == 2000
    assert budget.validation_interval == 20
    assert budget.early_stopping_patience == 20


def test_tuner_api_requires_ground_truth_for_per_image_selection():
    parameter = inspect.signature(tune_model_on_observed_pixels).parameters[
        "ground_truth"
    ]
    assert parameter.default is inspect.Parameter.empty


def _tuning(validation_mse, trial_count=2):
    return {
        "trial_count": trial_count,
        "trials": [{"runtime_seconds": 1.0}] * trial_count,
        "best": {
            "best_validation_mse": validation_mse,
            "best_step": 10,
            "hyperparameters": {"tv_weight": 0.0001},
        },
    }


def _final(
    psnr,
    ssim,
    final_train_mse=None,
    final_total_loss=None,
    missing_psnr=None,
    lpips=None,
):
    result = {
        "metrics": {
            "full_psnr": psnr,
            "missing_psnr": psnr - 1.0 if missing_psnr is None else missing_psnr,
            "composite_ssim": ssim,
            "lpips": lpips,
        },
        "runtime_seconds": 1.0,
    }
    if final_train_mse is not None:
        result["final_train_mse"] = final_train_mse
    if final_total_loss is not None:
        result["final_total_loss"] = final_total_loss
    return result


def test_judge_uses_missing_region_psnr_and_ssim_gate():
    accepted = judge_candidate(
        _tuning(0.03, trial_count=1),
        _final(20.0, 0.80),
        _tuning(0.02, trial_count=3),
        _final(20.25, 0.799),
    )
    assert accepted["decision"] == "accept"
    assert accepted["psnr_metric"] == "missing_psnr"
    assert accepted["budget_audit"]["trial_count_equality_required"] is False

    lpips_feedback = judge_candidate(
        _tuning(0.03),
        _final(20.0, 0.80, lpips=0.30),
        _tuning(0.02),
        _final(20.25, 0.80, lpips=0.24),
    )
    assert lpips_feedback["lpips_delta"] == pytest.approx(-0.06)
    assert lpips_feedback["lpips_delta_interpretation"].endswith(
        "lower_is_better"
    )

    rejected = judge_candidate(
        _tuning(0.03),
        _final(20.0, 0.80),
        _tuning(0.02),
        _final(20.25, 0.797),
    )
    assert rejected["decision"] == "reject"
    assert rejected["next_round_constraints"]

    full_image_improves_but_missing_region_regresses = judge_candidate(
        _tuning(0.03),
        _final(20.0, 0.80, missing_psnr=15.0),
        _tuning(0.02),
        _final(25.0, 0.80, missing_psnr=14.0),
    )
    assert full_image_improves_but_missing_region_regresses["decision"] == "reject"


class _RepairGenerator:
    def __init__(self):
        self.llm = object()
        self.contexts = []

    def generate(self, context):
        self.contexts.append(context)
        proposal = deterministic_candidate("tucker")
        proposal.generation_mode = "llm"
        return CandidateGenerationResult(
            proposal=proposal,
            attempts=1,
            raw_outputs=["test output"],
            validation_errors=[],
            fallback_reason=None,
            prompt_version="test-day6-repair",
        )


class _RejectThenAcceptValidator:
    def __init__(self, accept_call=2):
        self.calls = 0
        self.accept_call = accept_call

    def validate(self, model_path, output_path=None):
        self.calls += 1
        passed = self.calls == self.accept_call
        result = {
            "passed": passed,
            "status": "validated" if passed else "rejected",
            "static_validation": {"passed": passed, "checks": [], "errors": []},
            "smoke_test": {"passed": passed, "skipped": not passed},
            "feedback": [] if passed else ["Tucker ranks exceed image dimensions"],
            "runtime_seconds": 0.001,
        }
        if output_path is not None:
            Path(output_path).write_text(json.dumps(result), encoding="utf-8")
        return result


class _AlwaysAcceptValidator(_RejectThenAcceptValidator):
    def validate(self, model_path, output_path=None):
        self.calls += 1
        result = {
            "passed": True,
            "status": "validated",
            "static_validation": {"passed": True, "checks": [], "errors": []},
            "smoke_test": {"passed": True, "skipped": False},
            "feedback": [],
            "runtime_seconds": 0.001,
        }
        if output_path is not None:
            Path(output_path).write_text(json.dumps(result), encoding="utf-8")
        return result


class _ManifestMismatchThenRepairGenerator(_RepairGenerator):
    def generate(self, context):
        result = super().generate(context)
        if len(self.contexts) == 1:
            result.proposal.search_space = {"tv_weight": [0.9]}
        return result


class _Trace:
    def log_event(self, *args, **kwargs):
        return None


def _day6_generation_workflow(tmp_path, validator):
    workflow = object.__new__(Day6Workflow)
    workflow.config = SimpleNamespace(candidate_root=str(tmp_path / "candidates"))
    workflow.generator = _RepairGenerator()
    workflow.validator = validator
    workflow.trace = _Trace()
    workflow.state_path = tmp_path / "state.json"
    workflow.state = {
        "candidate_generation_attempts": [],
        "updated_at": None,
    }
    return workflow


def _generation_context():
    return {
        "base_run_id": "base-run-test",
        "base_method": "tucker",
        "base_class_name": "TuckerDecomposition",
        "image_profile": {"image_shape": [16, 16, 3]},
        "previous_failure_feedback": [{"decision": "reject"}],
    }


def test_day6_repairs_a_rejected_next_candidate_instead_of_crashing(tmp_path):
    validator = _RejectThenAcceptValidator(accept_call=2)
    workflow = _day6_generation_workflow(tmp_path, validator)

    candidate_dir = workflow._make_next_candidate(_generation_context(), 2)

    assert candidate_dir is not None
    assert validator.calls == 2
    assert len(workflow.generator.contexts) == 2
    assert workflow.generator.contexts[1]["previous_failure_feedback"] == [
        "Tucker ranks exceed image dimensions"
    ]
    assert "previous_candidate" in workflow.generator.contexts[1]
    manifest = json.loads(
        (Path(candidate_dir) / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["validation_round"] == 2
    assert manifest["eligible_for_training"] is True


def test_day6_feeds_manifest_mismatch_back_to_llm_for_regeneration(tmp_path):
    validator = _AlwaysAcceptValidator()
    workflow = _day6_generation_workflow(tmp_path, validator)
    workflow.generator = _ManifestMismatchThenRepairGenerator()

    candidate_dir = workflow._make_next_candidate(_generation_context(), 2)

    assert candidate_dir is not None
    assert validator.calls == 2
    assert len(workflow.generator.contexts) == 2
    assert "SEARCH_SPACE_CONTRACT_MISMATCH" in workflow.generator.contexts[1][
        "previous_failure_feedback"
    ][0]
    assert "previous_candidate" in workflow.generator.contexts[1]
    first_attempt, second_attempt = workflow.state["candidate_generation_attempts"]
    assert first_attempt["passed"] is False
    assert first_attempt["search_space_contract_passed"] is False
    assert second_attempt["passed"] is True
    assert second_attempt["search_space_contract_passed"] is True
    assert second_attempt["generation_mode"] == "llm_repaired"
    manifest = json.loads(
        (Path(candidate_dir) / "manifest.json").read_text(encoding="utf-8")
    )
    idea = json.loads(
        (Path(candidate_dir) / "idea.json").read_text(encoding="utf-8")
    )
    assert manifest["search_space_contract"]["passed"] is True
    assert manifest["declared_search_space"]["tv_weight"] == [
        0.0,
        0.0001,
        0.0005,
        0.001,
    ]
    assert manifest["executable_search_space"]["tv_weight"] == [
        0.0,
        0.0001,
        0.0005,
        0.001,
    ]
    assert manifest["effective_search_space"] == manifest[
        "executable_search_space"
    ]
    assert idea["declared_search_space"] == manifest["declared_search_space"]
    assert idea["effective_search_space"] == manifest[
        "effective_search_space"
    ]
    assert "allowed_search_space" not in manifest


def test_day6_returns_none_after_llm_repairs_and_safe_fallback_all_fail(tmp_path):
    validator = _RejectThenAcceptValidator(accept_call=99)
    workflow = _day6_generation_workflow(tmp_path, validator)

    candidate_dir = workflow._make_next_candidate(_generation_context(), 2)

    assert candidate_dir is None
    assert validator.calls == 3
    assert len(workflow.generator.contexts) == 2
    assert len(workflow.state["candidate_generation_attempts"]) == 3
    assert workflow.state["candidate_generation_attempts"][-1][
        "generation_mode"
    ] == "deterministic_template"


def test_judge_reports_selected_checkpoint_instability_to_the_next_llm_round():
    rejected = judge_candidate(
        _tuning(0.004),
        _final(20.0, 0.80, final_train_mse=0.003, final_total_loss=0.0035),
        _tuning(0.004),
        _final(20.3, 0.81, final_train_mse=0.02, final_total_loss=791.0),
    )

    assert rejected["decision"] == "reject"
    assert "numerically unstable" in rejected["gate_failures"][-1]
    assert rejected["training_behavior"]["candidate_selected_checkpoint_unstable"] is True
    assert any("unstable" in cause for cause in rejected["suspected_causes"])
    assert any("bounded residual gates" in item for item in rejected["next_round_constraints"])


def test_final_evaluation_reuses_tuning_checkpoint_without_training(tmp_path, monkeypatch):
    shape = (12, 12, 3)
    ground_truth = np.full(shape, 0.5, dtype=np.float32)
    observed_mask = np.ones(shape[:2], dtype=np.bool_)
    observed_mask[3:7, 4:8] = False
    observed = ground_truth.copy()
    observed[~observed_mask] = 0.0
    selected_raw = np.full(shape, 0.45, dtype=np.float32)
    raw_path = tmp_path / "selected_raw.npy"
    checkpoint_path = tmp_path / "selected.pt"
    np.save(raw_path, selected_raw)
    checkpoint_path.write_bytes(b"already-trained-checkpoint")

    def fail_if_retrained(**kwargs):
        raise AssertionError("selected checkpoint should be reused")

    monkeypatch.setattr(
        "research_agent.core.fair_experiment.train_tensor_model",
        fail_if_retrained,
    )
    selected_trial = {
        "hyperparameters": {"rank": 2},
        "learning_rate": 0.01,
        "best_step": 5,
        "best_validation_mse": 0.0025,
        "best_missing_psnr": 26.0206,
        "runtime_seconds": 0.2,
        "parameter_count": 10,
        "history": [{
            "step": 5,
            "data_train_loss": 0.01,
            "total_train_loss": 0.01,
        }],
        "artifacts": {
            "selected_raw_reconstruction": str(raw_path),
            "selected_checkpoint": str(checkpoint_path),
        },
    }

    result = final_fit_and_evaluate(
        model_name="matrix",
        model_builder=None,
        selected_trial=selected_trial,
        observed_image=observed,
        observed_mask=observed_mask,
        ground_truth=ground_truth,
        training_config=TrainingConfig(max_steps=5, device="cpu"),
        seed=1,
        output_dir=str(tmp_path / "final"),
        include_full_reference_metrics=False,
    )

    assert result["reused_without_retraining"] is True
    assert result["runtime_seconds"] == 0.0
    assert Path(result["artifacts"]["checkpoint"]).read_bytes() == (
        b"already-trained-checkpoint"
    )
