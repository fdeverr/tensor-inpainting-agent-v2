import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from research_agent.core.experiment_judge import judge_candidate
from research_agent.core.fair_experiment import (
    paired_trial_configurations,
    tune_model_on_observed_pixels,
)
from research_agent.candidate.generator import (
    CandidateGenerationResult,
    deterministic_candidate,
)
from research_agent.workflow_day6 import Day6Workflow


def test_paired_trials_share_base_parameters_and_budget():
    paired = paired_trial_configurations(
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
    assert len(paired["baseline"]) == len(paired["candidate"]) == 3
    for baseline, candidate in zip(paired["baseline"], paired["candidate"]):
        assert candidate["rank"] == baseline["rank"]
        assert candidate["init_scale"] == baseline["init_scale"]
    assert paired["candidate_only_parameter_names"] == ["tv_weight"]


def test_different_deep_architecture_receives_only_its_own_parameters():
    paired = paired_trial_configurations(
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
    paired = paired_trial_configurations(
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
    manifest = {"allowed_search_space": dict(candidate)}

    Day6Workflow._validate_candidate_search_contract(base, candidate, manifest)

    with pytest.raises(ValueError):
        Day6Workflow._validate_candidate_search_contract(
            base,
            candidate,
            {"allowed_search_space": {**candidate, "rank_h": [8]}},
        )


def test_llm_training_budget_is_shared_and_clamped_to_user_ceiling():
    workflow = object.__new__(Day6Workflow)
    workflow.config = SimpleNamespace(
        max_steps=2000,
        validation_interval=10,
        patience=40,
        validation_ratio=0.1,
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


def test_tuner_api_cannot_receive_hidden_ground_truth():
    assert "ground_truth" not in inspect.signature(tune_model_on_observed_pixels).parameters


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


def _final(psnr, ssim, final_train_mse=None, final_total_loss=None):
    result = {
        "metrics": {"missing_psnr": psnr, "composite_ssim": ssim},
        "runtime_seconds": 1.0,
    }
    if final_train_mse is not None:
        result["final_train_mse"] = final_train_mse
    if final_total_loss is not None:
        result["final_total_loss"] = final_total_loss
    return result


def test_judge_uses_fixed_psnr_and_ssim_gate():
    accepted = judge_candidate(
        _tuning(0.03),
        _final(20.0, 0.80),
        _tuning(0.02),
        _final(20.25, 0.799),
    )
    assert accepted["decision"] == "accept"

    rejected = judge_candidate(
        _tuning(0.03),
        _final(20.0, 0.80),
        _tuning(0.02),
        _final(20.25, 0.797),
    )
    assert rejected["decision"] == "reject"
    assert rejected["next_round_constraints"]


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


def test_judge_reports_final_refit_divergence_to_the_next_llm_round():
    rejected = judge_candidate(
        _tuning(0.004),
        _final(20.0, 0.80, final_train_mse=0.003, final_total_loss=0.0035),
        _tuning(0.004),
        _final(20.3, 0.81, final_train_mse=0.02, final_total_loss=791.0),
    )

    assert rejected["decision"] == "reject"
    assert "numerically unstable" in rejected["gate_failures"][-1]
    assert rejected["training_behavior"]["candidate_final_fit_diverged"] is True
    assert any("unstable" in cause for cause in rejected["suspected_causes"])
    assert any("bounded residual gates" in item for item in rejected["next_round_constraints"])
